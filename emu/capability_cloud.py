#!/usr/bin/env python3
"""EMU capability cloud helpers.

Low-dimensional on purpose:
  command = (vx, vy, yaw) in real units
  context = categorical disturbance family
  severity = one scalar inside that family
  outcome = survival, dynamic recovery probability and recovery time

The nuisance details inside each family are stratified across vectorized
environments, so direction remains measurable without becoming another map axis.
Every disturbed command is paired with an undisturbed control at identical
(vx, vy, yaw). This module provides the held-out sampler, evaluator and summary helpers
used by training.py for continuous capability telemetry.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from .contracts import (CONTROL_DT, COMBO_P, RECOVERY_TILT_MAX_DEG, RECOVERY_LINVEL_MAX_MPS,
                        RECOVERY_ANGVEL_MAX_RAD_S, RECOVERY_JOINT_MAX_DEG, task_param_defaults)
from .tracking_contract import recovery_outcome_numpy


FAMILIES = ('none', 'pose', 'momentum', 'impact', 'chaos')
# Deliberately a little beyond the current 120M reference envelope.  These are
# sampling scales, NOT claims of hardware capability and NOT training limits.
CMD_SCALE = {
    'vx_pos': 1.05,
    'vx_neg': 0.55,
    'vy': 0.55,
    'yaw': 4.8,
}


def halton(index: int, base: int) -> float:
    """Small deterministic low-discrepancy scalar in [0,1)."""
    i = int(index) + 1
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def pnorm(v, p: float = COMBO_P) -> float:
    a = np.abs(np.asarray(v, dtype=float))
    return float(np.power(np.sum(np.power(a, p)), 1.0 / p))


def _direction(index: int) -> np.ndarray:
    # Three irrational-looking low-discrepancy channels.  If a rare point lands
    # too close to the origin, rotate to another deterministic point rather than
    # creating an ill-defined direction.
    v = np.array([2*halton(index, 2)-1, 2*halton(index, 3)-1, 2*halton(index, 5)-1], dtype=float)
    n = pnorm(v)
    if n < 0.12:
        v = np.array([2*halton(index+17, 2)-1, 2*halton(index+17, 3)-1, 2*halton(index+17, 5)-1], dtype=float)
        n = max(pnorm(v), 1e-9)
    return v / n


def command_from_normalized(c: Iterable[float]) -> tuple[float,float,float]:
    cx, cy, cz = [float(x) for x in c]
    vx = cx * (CMD_SCALE['vx_pos'] if cx >= 0 else CMD_SCALE['vx_neg'])
    vy = cy * CMD_SCALE['vy']
    yaw = cz * CMD_SCALE['yaw']
    return float(vx), float(vy), float(yaw)


def command_sector(c: Iterable[float], active_frac: float = 0.34) -> str:
    """Human-readable coarse direction cell for a genuine 3-D command mixture.

    Raw cx/cy/cyaw remain authoritative.  The sector is only a compact way to
    keep direction visible in summaries so opposite/mixed corners are never
    averaged together merely because they share the same scalar command load.
    Components smaller than ``active_frac`` of the largest component are treated
    as nuisance mixture for the label, not discarded from the actual probe.
    """
    a=np.asarray(list(c),dtype=float)
    mx=max(float(np.max(np.abs(a))),1e-9)
    names=(('F','R'),('L','Rt'),('CCW','CW'))
    bits=[]
    for x,(pos,neg) in zip(a,names):
        if abs(float(x)) >= active_frac*mx:
            bits.append(pos if x>=0 else neg)
    return '+'.join(bits) if bits else 'mixed'



def make_probe(index: int, *, family: str | None = None, load: float | None = None,
               severity: float | None = None, source: str = 'explore',
               anchor_index: int | None = None, bundle_id: int | None = None,
               panel_id: int = 0) -> dict:
    """Generate one deterministic held-out point with matched-command bundling.

    Consecutive groups of ``len(FAMILIES)`` share the *exact same* vx/vy/yaw
    command and differ only in disturbance family.  This removes the stage-4
    family/yaw alias and gives every disturbed probe an undisturbed matched
    control without adding another map dimension.
    """
    fam_count = len(FAMILIES)
    if family is None:
        family = FAMILIES[int(index) % fam_count]
    if family not in FAMILIES:
        raise ValueError(f'unknown disturbance family {family!r}')
    if anchor_index is None:
        anchor_index = int(index) // fam_count
    anchor_index = int(anchor_index)
    if bundle_id is None:
        bundle_id = int(panel_id) * 100000 + anchor_index

    # Command generation is keyed only by the anchor, never by family.
    d = _direction(anchor_index)
    if load is None:
        load = 0.18 + 1.06 * halton(anchor_index, 7)
    c = d * float(load)
    vx, vy, yaw = command_from_normalized(c)

    if severity is None:
        if family == 'none':
            severity = 0.0
        else:
            fi = FAMILIES.index(family)
            # Family-specific severity sequence is offset independently from
            # command direction.  Because every family shares each command
            # anchor, any residual sequence structure cannot masquerade as
            # a direction-family effect.
            severity = 0.10 + 1.20 * halton(anchor_index + 97 * (fi + 1), 11)
    severity = max(0.0, float(severity))
    return {
        'probe_index': int(index), 'source': source, 'panel_id': int(panel_id),
        'bundle_id': int(bundle_id), 'anchor_index': anchor_index,
        'cmd_cx': float(c[0]), 'cmd_cy': float(c[1]), 'cmd_cyaw': float(c[2]),
        'cmd_load': float(pnorm(c)), 'cmd_sector': command_sector(c),
        'cmd_vx': vx, 'cmd_vy': vy, 'cmd_yaw': yaw,
        'family': family, 'severity': severity, 'cloud_role': 'heldout_probe',
    }


def make_panel(point_budget: int, *, panel_id: int = 0, source: str = 'panel',
               core_bundles: int = 2) -> list[dict]:
    """Return complete matched-command bundles with fixed core + rotating scouts.

    A routine 20-point panel contains four command anchors × five contexts.
    The first two command anchors are permanent longitudinal controls.  The
    remaining anchors rotate deterministically with ``panel_id`` so the held-out
    point cloud expands over policy age without sacrificing direct comparison.
    A larger final panel with panel_id=0 begins at anchor zero and therefore
    densely rechecks the region accumulated by the early routine panels.
    """
    fam_count = len(FAMILIES)
    bundles = max(1, int(math.ceil(max(1, int(point_budget)) / fam_count)))
    core = min(max(0, int(core_bundles)), bundles)
    rotating = max(0, bundles - core)
    out = []
    for b in range(bundles):
        if b < core:
            anchor = b
        else:
            anchor = core + int(panel_id) * max(1, rotating) + (b - core)
        bundle_id = int(panel_id) * 100000 + anchor
        for fi, fam in enumerate(FAMILIES):
            idx = anchor * fam_count + fi
            out.append(make_probe(idx, family=fam, anchor_index=anchor, bundle_id=bundle_id,
                                  panel_id=panel_id, source=source))
    return out


def make_calibration_panel(commands, *, panel_id: int = 0, source: str = 'field_calibration') -> list[dict]:
    """Build matched held-out bundles from commands chosen by the live field.

    ``commands`` items are (vx, vy, yaw, predicted_competence, evidence).  Only
    the coordinates/prediction are copied; evaluator outcomes stay quarantined
    and are never returned to the training-derived field.
    """
    out=[]
    for b,item in enumerate(commands):
        vx,vy,yaw=[float(x) for x in item[:3]]
        pred=float(item[3]) if len(item)>3 else float('nan')
        evidence=float(item[4]) if len(item)>4 else float('nan')
        cx=vx/(CMD_SCALE['vx_pos'] if vx>=0 else CMD_SCALE['vx_neg'])
        cy=vy/CMD_SCALE['vy']; cz=yaw/CMD_SCALE['yaw']
        c=np.asarray([cx,cy,cz],dtype=float)
        bundle_id=int(panel_id)*100000+b
        for fi,fam in enumerate(FAMILIES):
            sev=0.0 if fam=='none' else 0.10+1.20*halton(b+97*(fi+1)+panel_id*13,11)
            out.append({
                'probe_index':bundle_id*len(FAMILIES)+fi,'source':source,'panel_id':int(panel_id),
                'bundle_id':bundle_id,'anchor_index':b,
                'cmd_cx':float(c[0]),'cmd_cy':float(c[1]),'cmd_cyaw':float(c[2]),
                'cmd_load':float(pnorm(c)),'cmd_sector':command_sector(c),
                'cmd_vx':vx,'cmd_vy':vy,'cmd_yaw':yaw,'family':fam,'severity':float(sev),
                'cloud_role':'heldout_calibration','field_predicted_competence':pred,
                'field_effective_evidence':evidence,
            })
    return out


def make_common_recovery_panel(point_budget: int = 50, *, source: str = 'final_common_recovery') -> list[dict]:
    """Deterministic mature-policy recovery exam with useful matched controls.

    Sample a conservative but
    genuinely 3-D operating region; physical limits still belong to the dense
    atlas.  Every command is tested as none/pose/momentum/impact/chaos.
    """
    anchors = [
        (0.10,  0.00,  0.00),
        (0.30,  0.00,  0.00),
        (0.50,  0.00,  0.00),
        (-0.12, 0.00,  0.00),
        (-0.28, 0.00,  0.00),
        (0.22,  0.10,  0.00),
        (0.22, -0.10,  0.00),
        (0.22,  0.00,  0.75),
        (0.22,  0.00, -0.75),
        (0.30,  0.09,  0.70),
        (0.30, -0.09, -0.70),
        (-0.18, 0.07,  0.55),
    ]
    fam_count = len(FAMILIES)
    bundles = max(1, int(math.ceil(max(1, int(point_budget)) / fam_count)))
    out=[]
    for b in range(bundles):
        vx,vy,yaw=anchors[b % len(anchors)]
        # If callers ask for more bundles than the canonical set, repeat at a
        # slightly reduced amplitude rather than inventing unbounded extremes.
        cycle=b//len(anchors)
        scale=max(0.70,1.0-0.08*cycle)
        vx,vy,yaw=vx*scale,vy*scale,yaw*scale
        cx=vx/(CMD_SCALE['vx_pos'] if vx>=0 else CMD_SCALE['vx_neg'])
        cy=vy/CMD_SCALE['vy']; cz=yaw/CMD_SCALE['yaw']
        c=np.asarray([cx,cy,cz],dtype=float)
        bundle_id=900000+b
        for fi,fam in enumerate(FAMILIES):
            # Moderate fixed severities make the exam comparable and avoid
            # conflating capability with a random severity draw.
            sev=0.0 if fam=='none' else (0.55 + 0.05*((b+fi)%3))
            out.append({
                'probe_index':bundle_id*fam_count+fi,'source':source,'panel_id':-1,
                'bundle_id':bundle_id,'anchor_index':b,
                'cmd_cx':float(c[0]),'cmd_cy':float(c[1]),'cmd_cyaw':float(c[2]),
                'cmd_load':float(pnorm(c)),'cmd_sector':command_sector(c),
                'cmd_vx':float(vx),'cmd_vy':float(vy),'cmd_yaw':float(yaw),
                'family':fam,'severity':float(sev),'cloud_role':'heldout_common_recovery',
            })
    return out

def disturbance_overrides(spec: dict) -> dict:
    """Map one family+severity into explicit physical probe quantities.

    Directions/signs remain randomized per environment inside environment.py,
    which deliberately integrates nuisance dimensions rather than mapping them.
    """
    fam = spec['family']; s = float(spec['severity'])
    out = {
        'probe_tilt_rad': -1.0,
        'probe_linvel_mps': -1.0,
        'probe_angvel_rad_s': -1.0,
        'probe_joint_rad': -1.0,
        'probe_push_n': -1.0,
        'probe_push_steps': 5.0,
    }
    if fam == 'pose':
        out['probe_tilt_rad'] = math.radians(38.0) * s
        out['probe_joint_rad'] = math.radians(20.0) * s
    elif fam == 'momentum':
        out['probe_linvel_mps'] = 0.95 * s
        out['probe_angvel_rad_s'] = 5.0 * s
    elif fam == 'impact':
        # Five control frames = 0.10 s.  At severity 1 this is 0.28 N*s.
        out['probe_push_steps'] = 5.0
        out['probe_push_n'] = 2.8 * s
    elif fam == 'chaos':
        out['probe_tilt_rad'] = math.radians(RECOVERY_TILT_MAX_DEG) * s
        out['probe_joint_rad'] = math.radians(RECOVERY_JOINT_MAX_DEG) * s
        out['probe_linvel_mps'] = RECOVERY_LINVEL_MAX_MPS * s
        out['probe_angvel_rad_s'] = RECOVERY_ANGVEL_MAX_RAD_S * s
        out['probe_push_steps'] = 5.0
        out['probe_push_n'] = 2.0 * s
    return out


def physical_descriptor(spec: dict) -> dict:
    ov = disturbance_overrides(spec)
    return {
        'tilt_deg_max': max(0.0, math.degrees(ov['probe_tilt_rad'])) if ov['probe_tilt_rad'] >= 0 else 0.0,
        'joint_offset_deg_max': max(0.0, math.degrees(ov['probe_joint_rad'])) if ov['probe_joint_rad'] >= 0 else 0.0,
        'linvel_kick_mps_max': max(0.0, ov['probe_linvel_mps']),
        'angvel_kick_rad_s_max': max(0.0, ov['probe_angvel_rad_s']),
        'push_force_n': max(0.0, ov['probe_push_n']),
        'push_duration_s': max(0.0, ov['probe_push_steps']) * CONTROL_DT if ov['probe_push_n'] >= 0 else 0.0,
        'push_impulse_ns': max(0.0, ov['probe_push_n']) * max(0.0, ov['probe_push_steps']) * CONTROL_DT if ov['probe_push_n'] >= 0 else 0.0,
    }


def base_episode(spec: dict) -> dict:
    ep = task_param_defaults()
    ep.update(regime_command_frac=1.0,
              probe_cmd_vx=float(spec['cmd_vx']), probe_cmd_vy=float(spec['cmd_vy']),
              probe_cmd_yaw=float(spec['cmd_yaw']))
    ep.update(disturbance_overrides(spec))
    return ep



def _not_before_for_spec(spec: dict) -> int:
    if spec.get('family') in ('impact', 'chaos'):
        return 6 + int(round(float(disturbance_overrides(spec)['probe_push_steps'])))
    return 0


def _row_from_times(spec: dict, ff, fr, not_before: int) -> dict:
    row = dict(spec)
    row.update(physical_descriptor(spec))
    row.update(recovery_outcome_numpy(ff, fr, not_before_steps=not_before, direction_bins=8))
    return row


def evaluate_specs_batch(trainer, env, params, obs_mean, obs_std, keys, horizon: int,
                         specs: list[dict], stable_steps: int = 12, chunk_size: int = 4) -> list[dict]:
    """Evaluate several cloud points concurrently with an outer vmap.

    Each point still contains ``env.num_envs`` independently stratified robots.
    Batching four 256-env points gives MJX roughly 1024 environments of work per
    dispatch instead of repeatedly under-filling the GPU with one small cloud
    point at a time. The final partial chunk is padded and then discarded so JAX
    sees one stable batch shape and compiles it once per horizon/chunk size.
    """
    if not specs:
        return []
    import jax
    import jax.numpy as jnp
    chunk_size=max(1,int(chunk_size))
    horizon=int(horizon); stable_steps=int(stable_steps)
    cache=trainer.setdefault('_cloud_batch_eval_cache', {})
    cache_key=(horizon, stable_steps, chunk_size)
    batched=cache.get(cache_key)
    if batched is None:
        def fn(policy_params, om, osd, cps, ks, nbs):
            return jax.vmap(
                lambda cp, k, nb: trainer['evaluate_recovery'](
                    policy_params, cp, om, osd, k, horizon, stable_steps, nb),
                in_axes=(0,0,0))(cps, ks, nbs)
        batched=jax.jit(fn)
        cache[cache_key]=batched

    rows=[]
    keys=list(keys)
    if len(keys) != len(specs):
        raise ValueError('keys/specs length mismatch')
    for start in range(0,len(specs),chunk_size):
        real_specs=list(specs[start:start+chunk_size])
        real_keys=list(keys[start:start+chunk_size])
        nreal=len(real_specs)
        pad_spec=real_specs[-1]
        pad_key=real_keys[-1]
        padded_specs=real_specs + [pad_spec]*(chunk_size-nreal)
        padded_keys=real_keys + [pad_key]*(chunk_size-nreal)
        cps=jnp.stack([env.params_array(base_episode(x)) for x in padded_specs],axis=0)
        ks=jnp.stack(padded_keys,axis=0)
        nbs=jnp.asarray([_not_before_for_spec(x) for x in padded_specs],dtype=jnp.int32)
        ff,fr=batched(params,obs_mean,obs_std,cps,ks,nbs)
        ff,fr=jax.device_get((ff,fr))
        for i in range(nreal):
            rows.append(_row_from_times(real_specs[i],ff[i],fr[i],int(nbs[i])))

    # Node competence is calibrated against a
    # held-out NODE event, not against the recovery event above. Only controls
    # need this extra pass; disturbed rows remain purely recovery evidence.
    control_idx=[i for i,spec in enumerate(specs) if spec.get('family') == 'none']
    if control_idx and 'evaluate_node_window' in trainer:
        ncache=trainer.setdefault('_cloud_node_eval_cache',{})
        nkey=(horizon,chunk_size)
        nbatched=ncache.get(nkey)
        if nbatched is None:
            def nfn(policy_params,om,osd,cps,ks):
                return jax.vmap(lambda cp,k: trainer['evaluate_node_window'](
                    policy_params,cp,om,osd,k,horizon,50,25),in_axes=(0,0))(cps,ks)
            nbatched=jax.jit(nfn); ncache[nkey]=nbatched
        for start in range(0,len(control_idx),chunk_size):
            ids=control_idx[start:start+chunk_size]; nreal=len(ids)
            pad=ids[-1]; pids=ids+[pad]*(chunk_size-nreal)
            cps=jnp.stack([env.params_array(base_episode(specs[i])) for i in pids],axis=0)
            ks=jnp.stack([keys[i] for i in pids],axis=0)
            verdict=np.asarray(jax.device_get(nbatched(params,obs_mean,obs_std,cps,ks)))
            for j,ri in enumerate(ids):
                rows[ri]['node_window_rate']=float(verdict[j].mean())
    return rows


def annotate_matched_controls(rows: list[dict]) -> list[dict]:
    """Attach matched undisturbed competence to every disturbed row.

    A disturbed result is treated as valid resilience evidence only when the
    exact same command is demonstrably competent without disturbance.
    """
    groups={}
    for r in rows:
        step=int(float(r.get('policy_step', -1) or -1))
        bid=int(float(r.get('bundle_id', -1) or -1))
        groups.setdefault((step,bid),[]).append(r)

    for vals in groups.values():
        controls=[r for r in vals if r.get('family') == 'none']
        if not controls:
            for r in vals:
                r['baseline_competent']=0
                r['resilience_valid']=0
                r['recovery_drop_from_control']=float('nan')
            continue
        # Panels intentionally have one control per bundle. If a final panel
        # duplicates a routine point at the same step, average the duplicates.
        base_rec=float(np.mean([float(r.get('recovery_rate',0.0)) for r in controls]))
        base_surv=float(np.mean([float(r.get('survival',0.0)) for r in controls]))
        competent = int(base_rec >= 0.80 and base_surv >= 0.95)
        for r in vals:
            r['baseline_competent']=competent
            r['resilience_valid']=int(competent and r.get('family') != 'none')
            r['recovery_drop_from_control']=(base_rec-float(r.get('recovery_rate',0.0))
                                             if r.get('family') != 'none' else 0.0)
    return rows


def _bin(x, edges):
    x=float(x)
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]: return i
    return len(edges)-2



def summarize_rows(rows: list[dict]) -> dict:
    rows=annotate_matched_controls(rows)
    load_edges=[0.0,0.35,0.60,0.85,1.10,1.40]
    sev_edges=[0.0,0.25,0.50,0.75,1.00,1.35,9.0]
    out={'load_edges':load_edges,'severity_edges':sev_edges,'families':{}}

    for fam in FAMILIES:
        rr=[r for r in rows if r.get('family')==fam]
        grid=[]
        for li in range(len(load_edges)-1):
            line=[]
            for si in range(len(sev_edges)-1):
                vals=[r for r in rr if _bin(float(r['cmd_load']),load_edges)==li and _bin(float(r['severity']),sev_edges)==si]
                p90=[float(v['recovery_time_p90_s']) for v in vals if float(v.get('recovery_time_p90_s',-1))>=0]
                line.append({
                    'n':len(vals),
                    'recovery':float(np.mean([float(v['recovery_rate']) for v in vals])) if vals else None,
                    'survival':float(np.mean([float(v['survival']) for v in vals])) if vals else None,
                    'worst_direction':float(np.mean([float(v.get('recovery_worst_direction',0.0)) for v in vals])) if vals else None,
                    'p90_s':float(np.mean(p90)) if p90 else None,
                })
            grid.append(line)
        out['families'][fam]={'n':len(rr),'grid':grid}

    controls=[r for r in rows if r.get('family')=='none']
    disturbed=[r for r in rows if r.get('family')!='none']
    valid=[r for r in disturbed if int(float(r.get('resilience_valid',0) or 0))==1]
    out['paired']={
        'control_points':len(controls),
        'baseline_competent_fraction':float(np.mean([float(r.get('baseline_competent',0)) for r in controls])) if controls else None,
        'baseline_recovery_mean':float(np.mean([float(r.get('recovery_rate',0)) for r in controls])) if controls else None,
        'disturbed_points':len(disturbed),
        'valid_disturbance_points':len(valid),
        'valid_recovery_mean':float(np.mean([float(r.get('recovery_rate',0)) for r in valid])) if valid else None,
        'valid_worst_direction_mean':float(np.mean([float(r.get('recovery_worst_direction',0)) for r in valid])) if valid else None,
        'valid_survival_mean':float(np.mean([float(r.get('survival',0)) for r in valid])) if valid else None,
        'valid_recovery_drop_mean':float(np.mean([float(r.get('recovery_drop_from_control',0)) for r in valid])) if valid else None,
    }

    # Direction-aware sector summaries remain descriptive; raw vectors remain authoritative.
    sector_rows=[]
    keys=sorted(set((str(r.get('family')),str(r.get('cmd_sector'))) for r in rows))
    for fam,sector in keys:
        vals=[r for r in rows if str(r.get('family'))==fam and str(r.get('cmd_sector'))==sector]
        sector_rows.append({
            'family':fam,'sector':sector,'n':len(vals),
            'mean_load':float(np.mean([float(v['cmd_load']) for v in vals])),
            'mean_recovery':float(np.mean([float(v['recovery_rate']) for v in vals])),
            'mean_survival':float(np.mean([float(v['survival']) for v in vals])),
            'mean_worst_direction':float(np.mean([float(v.get('recovery_worst_direction',0)) for v in vals])),
        })
    out['direction_sectors']=sector_rows
    out['worst_points']=sorted(rows,key=lambda r:(float(r.get('recovery_rate',0)),float(r.get('survival',0))))[:12]
    return out
