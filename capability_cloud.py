#!/usr/bin/env python3
"""EMU capability cloud (current VERSION read from VERSION file).

Low-dimensional on purpose:
  command = (vx, vy, yaw) in real units
  context = categorical disturbance family
  severity = one scalar inside that family
  outcome = survival, dynamic recovery probability and recovery time

The nuisance details inside each family are stratified across vectorized
environments, so direction remains measurable without becoming another map axis.
Every disturbed command is paired with an undisturbed control at identical
(vx, vy, yaw). This module is both a standalone cloud survey and the held-out
sampler/summary helper used by training.py for continuous capability telemetry.
"""
from __future__ import annotations

import argparse, csv, json, math, pickle, sys, time
from pathlib import Path
from typing import Iterable

import numpy as np

VERSION = (Path(__file__).with_name('VERSION').read_text(encoding='utf-8').strip())

FAMILIES = ('none', 'pose', 'momentum', 'impact', 'chaos')
COMMAND_P = 1.35
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


def pnorm(v, p: float = COMMAND_P) -> float:
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

    v5010's final Halton panel sometimes spent almost all of its budget on
    commands the policy could not execute cleanly, leaving only four valid
    disturbed points.  This exam deliberately samples a conservative but
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
        out['probe_tilt_rad'] = math.radians(28.0) * s
        out['probe_joint_rad'] = math.radians(14.0) * s
        out['probe_linvel_mps'] = 0.65 * s
        out['probe_angvel_rad_s'] = 3.6 * s
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
        'push_duration_s': max(0.0, ov['probe_push_steps']) * 0.02 if ov['probe_push_n'] >= 0 else 0.0,
        'push_impulse_ns': max(0.0, ov['probe_push_n']) * max(0.0, ov['probe_push_steps']) * 0.02 if ov['probe_push_n'] >= 0 else 0.0,
    }


def base_episode(spec: dict) -> dict:
    ep = dict(steady_difficulty=0.0, transition_difficulty=0.0, resilience_difficulty=0.0,
              posture_difficulty=0.0, regime_frac_steady=1.0, regime_frac_transition=0.0,
              regime_frac_resilience=0.0, regime_command_frac=1.0,
              core_rehearsal_frac=0.0, athletic_rehearsal_frac=0.0,
              probe_cmd_vx=float(spec['cmd_vx']), probe_cmd_vy=float(spec['cmd_vy']),
              probe_cmd_yaw=float(spec['cmd_yaw']))
    ep.update(disturbance_overrides(spec))
    return ep



def _not_before_for_spec(spec: dict) -> int:
    ep = base_episode(spec)
    if spec.get('family') in ('impact', 'chaos'):
        return 6 + int(round(float(ep.get('probe_push_steps', 5.0))))
    return 0


def _row_from_times(spec: dict, ff, fr, not_before: int) -> dict:
    ff=np.asarray(ff); fr=np.asarray(fr)
    survived = ff < 0
    recovered = fr >= 0
    sustained = recovered & survived
    rt = (fr[sustained] - int(not_before)) * 0.02

    bins = np.arange(len(ff), dtype=np.int32) % 8
    rec_bins=[]; surv_bins=[]
    for b in range(8):
        mask = bins == b
        rec_bins.append(float(sustained[mask].mean()) if np.any(mask) else float('nan'))
        surv_bins.append(float(survived[mask].mean()) if np.any(mask) else float('nan'))

    row = dict(spec)
    row.update(physical_descriptor(spec))
    row.update({
        'survival': float(survived.mean()),
        'recovery_event_rate': float(recovered.mean()),
        'recovery_rate': float(sustained.mean()),
        'recovery_given_survival': float(sustained.sum() / max(1, survived.sum())),
        'recovery_time_mean_s': float(rt.mean()) if len(rt) else -1.0,
        'recovery_time_p50_s': float(np.percentile(rt, 50)) if len(rt) else -1.0,
        'recovery_time_p90_s': float(np.percentile(rt, 90)) if len(rt) else -1.0,
        'fall_time_mean_s': float(ff[~survived].mean() * 0.02) if np.any(~survived) else -1.0,
        'recovery_not_before_step': int(not_before),
        'recovery_worst_direction': float(np.nanmin(rec_bins)),
        'recovery_best_direction': float(np.nanmax(rec_bins)),
        'survival_worst_direction': float(np.nanmin(surv_bins)),
    })
    for b,(rr,ss) in enumerate(zip(rec_bins,surv_bins)):
        row[f'recovery_dir{b}'] = rr
        row[f'survival_dir{b}'] = ss
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

    # v5012 calibration doctrine: node competence is calibrated against a
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


def evaluate_spec(trainer, env, params, obs_mean, obs_std, key, horizon: int, spec: dict,
                  stable_steps: int = 12) -> dict:
    return evaluate_specs_batch(trainer, env, params, obs_mean, obs_std, [key], horizon,
                                [spec], stable_steps=stable_steps, chunk_size=1)[0]


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
                r['matched_control_present']=0
                r['baseline_recovery']=float('nan')
                r['baseline_survival']=float('nan')
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
            r['matched_control_present']=1
            r['baseline_recovery']=base_rec
            r['baseline_survival']=base_surv
            r['baseline_competent']=competent
            r['resilience_valid']=int(competent and r.get('family') != 'none')
            r['recovery_drop_from_control']=(base_rec-float(r.get('recovery_rate',0.0))
                                             if r.get('family') != 'none' else 0.0)
    return rows


def refinement_specs(rows: list[dict], count: int, start_index: int) -> list[dict]:
    """Spend refinement budget as matched control/disturbed pairs near cliffs."""
    if count <= 1 or not rows:
        return []
    annotate_matched_controls(rows)
    candidates=[r for r in rows if r.get('family') != 'none' and int(float(r.get('matched_control_present',0) or 0)) == 1]
    if not candidates:
        return []
    candidates.sort(key=lambda r: abs(float(r.get('recovery_rate',0.0)) - 0.60))
    pairs=max(1, int(count)//2)
    next_bundle=max([int(float(r.get('bundle_id',-1) or -1)) for r in rows]+[-1])+1
    out=[]
    for j in range(pairs):
        parent=candidates[j % len(candidates)]
        sign=-1.0 if j % 2 == 0 else 1.0
        sev=max(0.03,float(parent['severity'])*(1.0+sign*0.12))
        load=max(0.08,float(parent['cmd_load'])*(1.0-sign*0.08))
        c0=np.array([float(parent['cmd_cx']),float(parent['cmd_cy']),float(parent['cmd_cyaw'])])
        c=c0/max(pnorm(c0),1e-9)*load
        vx,vy,yaw=command_from_normalized(c)
        bid=next_bundle+j
        common={
            'panel_id':int(float(parent.get('panel_id',0) or 0)),
            'bundle_id':bid,'anchor_index':int(float(parent.get('anchor_index',0) or 0)),
            'cmd_cx':float(c[0]),'cmd_cy':float(c[1]),'cmd_cyaw':float(c[2]),
            'cmd_load':float(load),'cmd_sector':command_sector(c),
            'cmd_vx':vx,'cmd_vy':vy,'cmd_yaw':yaw,'cloud_role':'heldout_probe',
        }
        out.append({'probe_index':start_index+2*j,'source':'refine_control',
                    **common,'family':'none','severity':0.0})
        out.append({'probe_index':start_index+2*j+1,'source':'refine',
                    **common,'family':parent['family'],'severity':sev})
    return out

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


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


def render_markdown(summary: dict, rows: list[dict], *, checkpoint: str, wall_s: float) -> str:
    L=[f'# EMU {VERSION} paired sparse capability cloud','',
       f'- checkpoint: `{checkpoint}`',f'- cloud points: `{len(rows)}`',f'- wall time: `{wall_s:.1f}s`','',
       'The active map stays deliberately compact: **combined command vector (vx, vy, yaw) + disturbance family + one severity scalar**. Each disturbed command is paired with the exact same undisturbed command, and nuisance direction is stratified inside the vector batch rather than becoming another map axis.','',
       'This is a **held-out probe cloud**. It is evaluation evidence; it does not train the policy.','',
       '## Paired command / recovery summary','']
    q=summary.get('paired',{})
    L += ['| metric | value |','|---|---:|',
          f"| undisturbed command points | {q.get('control_points',0)} |",
          f"| baseline command-competent fraction | {q.get('baseline_competent_fraction') if q.get('baseline_competent_fraction') is not None else '-'} |",
          f"| valid disturbed points (matched baseline competent) | {q.get('valid_disturbance_points',0)} |",
          f"| mean recovery on valid disturbed points | {q.get('valid_recovery_mean') if q.get('valid_recovery_mean') is not None else '-'} |",
          f"| mean worst-direction recovery on valid disturbed points | {q.get('valid_worst_direction_mean') if q.get('valid_worst_direction_mean') is not None else '-'} |",'',
          'A disturbed point is only resilience evidence when its exact undisturbed control recovered >=80% and survived >=95%. This prevents command incompetence from being mislabelled as poor disturbance recovery.','',
          '## Direction-aware command sectors','',
          '| family | command sector | n | mean load | mean recovery | mean survival | mean worst-direction recovery |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.get('direction_sectors',[]):
        L.append(f"| {r['family']} | {r['sector']} | {r['n']} | {r['mean_load']:.2f} | {r['mean_recovery']:.3f} | {r['mean_survival']:.3f} | {r['mean_worst_direction']:.3f} |")
    L += ['', '## Worst sampled combined states','',
          '| family | severity | load | vx | vy | yaw | baseline recovery | recovery | worst dir | survival |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.get('worst_points',[]):
        br=r.get('baseline_recovery',float('nan'))
        L.append(f"| {r['family']} | {float(r['severity']):.2f} | {float(r['cmd_load']):.2f} | {float(r['cmd_vx']):+.2f} | {float(r['cmd_vy']):+.2f} | {float(r['cmd_yaw']):+.2f} | {br if isinstance(br,(int,float)) and br==br else '-'} | {float(r['recovery_rate']):.3f} | {float(r.get('recovery_worst_direction',0)):.3f} | {float(r['survival']):.3f} |")
    L += ['', 'The full `capability_cloud.csv` is authoritative. No single physical capability score is produced; the compact JSON is a navigation aid over the point cloud.','']
    return '\n'.join(L)

def main(argv=None):
    ap=argparse.ArgumentParser(description=f'EMU {VERSION} sparse combined-command capability cloud')
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); ap.add_argument('--allow-cpu-mjx',action='store_true')
    ap.add_argument('--envs',type=int,default=256); ap.add_argument('--horizon',type=int,default=220)
    ap.add_argument('--explore-points',type=int,default=64); ap.add_argument('--refine-points',type=int,default=24)
    ap.add_argument('--batch-points',type=int,default=4)
    ap.add_argument('--seed',type=int,default=733)
    args=ap.parse_args(argv)

    from environment import EmuMJXEnv
    from training import configure_jax, require_backend, make_trainer, payload_params_are_finite, PPOState
    configure_jax(args.device,args.allow_cpu_mjx)
    import jax
    require_backend(args.allow_cpu_mjx)
    ck=Path(args.checkpoint).expanduser().resolve(); out=Path(args.out_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    sys.modules['__main__'].PPOState=PPOState
    with ck.open('rb') as f: payload=pickle.load(f)
    if not payload_params_are_finite(payload): raise SystemExit(f'checkpoint has non-finite params: {ck}')
    state=payload['state']; params=jax.device_put(state.params); norm=payload.get('obs_norm') or {}
    env=EmuMJXEnv(str(Path(__file__).with_name('robot.xml')),num_envs=args.envs,seed=args.seed,
                  max_episode_steps=max(700,args.horizon+50),actuator_realism=True,comm_latency_steps=1,
                  push_max_force_n=4.0,vx_hi_max=1.2,vx_lo_min=-0.65,vy_max_mps=0.65,yaw_max_rad_s=5.0,
                  disable_target_slew=True,phase_speed_gain=1.25,phase_rate_max_hz=2.6,sensor_noise_scale=1.5,
                  open_frontier=False,combo_mix_frac=0.0)
    tr=make_trainer(env,seed=args.seed,lr=2e-4,rollout=32,epochs=1,minibatches=1,obs_norm_mode='continuous')
    jnp=jax.numpy
    obs_mean=jnp.asarray(norm.get('mean',np.zeros((tr['obs_dim'],),np.float32)),dtype=jnp.float32)
    obs_std=jnp.asarray(np.sqrt(norm.get('var',np.ones((tr['obs_dim'],),np.float32))),dtype=jnp.float32)
    basekey=jax.random.PRNGKey(args.seed+44000); rows=[]; t0=time.time()
    requested=make_panel(args.explore_points,panel_id=0,source='explore_panel')
    controls=[x for x in requested if x.get('family')=='none']
    ckeys=[jax.random.fold_in(basekey,int(x.get('probe_index',i))) for i,x in enumerate(controls)]
    crows=evaluate_specs_batch(tr,env,params,obs_mean,obs_std,ckeys,args.horizon,controls,chunk_size=args.batch_points)
    rows.extend(crows); annotate_matched_controls(rows)
    good={int(r.get('bundle_id',-1)) for r in crows if float(r.get('recovery_rate',0))>=0.80 and float(r.get('survival',0))>=0.95}
    disturbed=[x for x in requested if x.get('family')!='none' and int(x.get('bundle_id',-1)) in good]
    dkeys=[jax.random.fold_in(basekey,int(x.get('probe_index',i))+1000003) for i,x in enumerate(disturbed)]
    drows=evaluate_specs_batch(tr,env,params,obs_mean,obs_std,dkeys,args.horizon,disturbed,chunk_size=args.batch_points) if disturbed else []
    rows.extend(drows); annotate_matched_controls(rows)
    print(f'[cloud] screened={len(controls)} competent={len(good)} disturbed={len(drows)}/{len(requested)-len(controls)}',flush=True)
    refs=refinement_specs(rows,args.refine_points,len(requested))
    if refs:
        rkeys=[jax.random.fold_in(basekey,args.explore_points+i) for i in range(len(refs))]
        rrows=evaluate_specs_batch(tr,env,params,obs_mean,obs_std,rkeys,args.horizon,refs,chunk_size=args.batch_points)
        rows.extend(rrows)
    annotate_matched_controls(rows)
    wall=time.time()-t0; summary=summarize_rows(rows)
    summary.update({'checkpoint':str(ck),'envs_per_point':args.envs,'horizon_steps':args.horizon,'wall_seconds':round(wall,2),'points':len(rows)})
    write_csv(out/'capability_cloud.csv',rows)
    (out/'capability_cloud.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'capability_cloud.md').write_text(render_markdown(summary,rows,checkpoint=str(ck),wall_s=wall),encoding='utf-8')
    print(f'[cloud] complete: {len(rows)} points in {wall:.1f}s -> {out}',flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
