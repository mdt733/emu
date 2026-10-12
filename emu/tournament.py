"""Successive-halving v5013 capability-teacher tournament.

The tournament deliberately spends GPU in a funnel rather than running twelve
120M-step branches.  All seed-733 arms share one foundation checkpoint.  Twelve
hypotheses are screened cheaply, four representatives reach 80M, two challengers
plus shadow reach 120M, and only the best challenger is confirmed on seed 1742.

Held-out exams choose what earns more compute; they never update training state.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .artifacts import read_csv_rows
from .checkpoint_io import load_checkpoint
from .experiments import (
    checkpoint_step, git_provenance, resolve_foundation_handoff,
    resolve_target_checkpoint, run_training, summarize,
)
from .paths import VERSION, run_base, results_base, source_repo_root
from .training_config import FOUNDATION_STEPS, TrainingConfig, foundation_config, mature_config

LOG = f"[emu_{VERSION}:tournament]"
DOSE_TARGET = 50_000_000
SCHEDULE_TARGET = 60_000_000
SEMIFINAL_TARGET = 80_000_000
FINAL_TARGET = 120_000_000


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    mode: str = 'graph'
    dose: float = 0.25
    teach_nodes: bool = True
    teach_edges: bool = True
    teach_recovery: bool = True
    evidence_from_taught: bool = True
    influence_start: int = FOUNDATION_STEPS
    influence_full: int = 30_000_000
    fade_start: int = 0
    zero_at: int = 0
    family: str = 'graph'

    def config(self, *, run_dir: Path, steps: int, seed: int, device: str,
               envs: int, resume: Path) -> TrainingConfig:
        cfg = mature_config(self.mode, run_dir=run_dir, steps=steps, seed=seed,
                            device=device, envs=envs, resume=resume)
        return replace(
            cfg,
            field_max_influence=float(self.dose),
            field_teach_nodes=bool(self.teach_nodes),
            field_teach_edges=bool(self.teach_edges),
            field_teach_recovery=bool(self.teach_recovery),
            field_evidence_from_taught=bool(self.evidence_from_taught),
            field_influence_start_steps=int(self.influence_start),
            field_influence_full_steps=int(self.influence_full),
            field_influence_fade_start_steps=int(self.fade_start),
            field_influence_zero_steps=int(self.zero_at),
        )


def base_specs() -> tuple[ExperimentSpec, list[ExperimentSpec]]:
    shadow = ExperimentSpec(
        'A0_shadow', 'Shadow control: field observes but never teaches',
        mode='shadow', dose=0.0, family='dose')
    doses = [
        ExperimentSpec('A1_graph10', 'Graph teaching at 10%', dose=.10, family='dose'),
        ExperimentSpec('A2_graph25', 'Graph teaching at 25%', dose=.25, family='dose'),
        ExperimentSpec('A3_graph40', 'Graph teaching at 40%', dose=.40, family='dose'),
        ExperimentSpec('A4_graph65', 'v5012-strength graph teaching at 65%', dose=.65, family='dose'),
    ]
    return shadow, doses


def mechanism_specs(dose: float) -> list[ExperimentSpec]:
    return [
        ExperimentSpec('B1_nodes_only', 'Node births only; node recovery severity remains active',
                       dose=dose, teach_nodes=True, teach_edges=False, teach_recovery=True,
                       family='mechanism'),
        ExperimentSpec('B2_edges_only', 'Maneuver-edge teaching only',
                       dose=dose, teach_nodes=False, teach_edges=True, teach_recovery=False,
                       family='mechanism'),
        ExperimentSpec('B3_no_recovery', 'Node + edge teaching, but no field-selected recovery severity',
                       dose=dose, teach_nodes=True, teach_edges=True, teach_recovery=False,
                       family='mechanism'),
        ExperimentSpec('B4_clean_evidence', 'Full teacher; field-selected practice cannot reinforce the field',
                       dose=dose, teach_nodes=True, teach_edges=True, teach_recovery=True,
                       evidence_from_taught=False, family='mechanism'),
    ]


def schedule_specs(base: ExperimentSpec) -> list[ExperimentSpec]:
    common = dict(
        mode=base.mode, dose=base.dose,
        teach_nodes=base.teach_nodes, teach_edges=base.teach_edges,
        teach_recovery=base.teach_recovery,
        evidence_from_taught=base.evidence_from_taught,
        family='schedule',
    )
    return [
        ExperimentSpec('C1_slow_ramp', 'Slow teacher ramp: 20M -> 60M',
                       influence_start=FOUNDATION_STEPS, influence_full=60_000_000, **common),
        ExperimentSpec('C2_delayed', 'Delayed teacher: governor-only to 40M, full by 60M',
                       influence_start=40_000_000, influence_full=60_000_000, **common),
        ExperimentSpec('C3_tutor_release', 'Tutor then release: full by 30M, fade 45M -> 60M',
                       influence_start=FOUNDATION_STEPS, influence_full=30_000_000,
                       fade_start=45_000_000, zero_at=60_000_000, **common),
    ]


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _number(v: Any, default: float = -1.0) -> float:
    try:
        x = float(v)
        return x if x == x else default
    except (TypeError, ValueError):
        return default


def summary_with_screen(run_dir: Path) -> dict[str, Any]:
    out = summarize(run_dir)
    screen = _json(run_dir / 'screen_eval.json', {}) or {}
    out.update(screen)
    # Existing fixed panels remain useful as secondary context.
    evals = read_csv_rows(run_dir / 'fixed_eval.csv')
    ev = evals[-1] if evals else {}
    for name in ('resilience75', 'robust40', 'growth110', 'growthcombo110'):
        out[f'{name}_survival'] = _number(ev.get(f'eval_{name}_survival'))
    return out


def hard_recovery(s: dict[str, Any]) -> float:
    return _number(s.get('screen_hard_recovery'))


def growth_headroom(s: dict[str, Any]) -> float:
    return _number(s.get('screen_growth_headroom'))


def utility(s: dict[str, Any]) -> float:
    """Transparent promotion utility; gates below remain more important."""
    broad = max(-1.0, _number(s.get('broad')))
    hard = max(0.0, hard_recovery(s))
    growth = max(0.0, growth_headroom(s))
    rec = _number(s.get('recovery'))
    worst = _number(s.get('worst_direction'))
    rec = hard if rec < 0 else rec
    worst = hard if worst < 0 else worst
    cal = _number(s.get('calibration_mae'))
    calibration_honesty = .5 if cal < 0 else max(0.0, 1.0 - min(1.0, cal))
    return float(.30*broad + .30*hard + .15*growth + .10*rec + .10*worst + .05*calibration_honesty)


def gate(candidate: dict[str, Any], shadow: dict[str, Any], *, strict_cloud: bool = False) -> tuple[bool, list[str]]:
    """Regression guard relative to paired shadow at the same training stage."""
    reasons: list[str] = []
    if _number(candidate.get('broad')) < _number(shadow.get('broad')) - .04:
        reasons.append('broad >0.04 below shadow')
    if hard_recovery(candidate) < hard_recovery(shadow) - .08:
        reasons.append('hard recovery >0.08 below shadow')
    c_rec, s_rec = _number(candidate.get('recovery')), _number(shadow.get('recovery'))
    if s_rec >= 0 and c_rec >= 0 and c_rec < s_rec - .10:
        reasons.append('held-out recovery >0.10 below shadow')
    if strict_cloud and s_rec >= 0 and c_rec < 0:
        reasons.append('held-out recovery invalid while shadow is valid')
    c_cal, s_cal = _number(candidate.get('calibration_mae')), _number(shadow.get('calibration_mae'))
    if c_cal >= 0 and s_cal >= 0 and c_cal > s_cal + .20:
        reasons.append('field calibration MAE >0.20 worse than shadow')
    return not reasons, reasons


def ranked(specs: Iterable[ExperimentSpec], summaries: dict[str, dict[str, Any]],
           shadow: dict[str, Any], *, strict_cloud: bool = False) -> list[dict[str, Any]]:
    rows=[]
    for spec in specs:
        s=summaries[spec.name]
        ok,reasons=gate(s,shadow,strict_cloud=strict_cloud)
        rows.append({'name':spec.name,'label':spec.label,'dose':spec.dose,'gate_ok':ok,
                     'gate_reasons':reasons,'utility':utility(s),
                     'broad':_number(s.get('broad')),'hard_recovery':hard_recovery(s),
                     'growth_headroom':growth_headroom(s),'recovery':_number(s.get('recovery')),
                     'worst_direction':_number(s.get('worst_direction')),
                     'calibration_mae':_number(s.get('calibration_mae'))})
    return sorted(rows,key=lambda r:(r['gate_ok'],r['utility']),reverse=True)


def pick_best(specs: list[ExperimentSpec], summaries: dict[str, dict[str, Any]], shadow: dict[str, Any],
              *, prefer_lower_dose_within: float = 0.0, strict_cloud: bool = False) -> tuple[ExperimentSpec, list[dict[str, Any]]]:
    order=ranked(specs,summaries,shadow,strict_cloud=strict_cloud)
    eligible=[r for r in order if r['gate_ok']] or order
    top=eligible[0]
    if prefer_lower_dose_within > 0:
        near=[r for r in eligible if r['utility'] >= top['utility']-prefer_lower_dose_within]
        top=min(near,key=lambda r:r['dose'])
    by_name={s.name:s for s in specs}
    return by_name[top['name']], order


def ensure_spec(run_dir: Path, spec: ExperimentSpec) -> None:
    path=run_dir/'experiment_spec.json'
    wanted=asdict(spec)
    if path.exists():
        existing=_json(path,{})
        if existing != wanted:
            raise RuntimeError(f'experiment spec drift in {run_dir}: existing={existing} wanted={wanted}')
    else:
        run_dir.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(wanted,indent=2,sort_keys=True),encoding='utf-8')


def run_screen_batch(run_dirs: list[Path], *, device: str, envs: int, horizon: int) -> None:
    """Screen all stale arms in one JAX process so evaluator worlds compile once."""
    stale=[]
    for run_dir in run_dirs:
        final=run_dir/'checkpoints'/'final.pkl'
        step=checkpoint_step(final)
        old=_json(run_dir/'screen_eval.json',{}) or {}
        if not (int(old.get('policy_step',-1)) == step and
                int(old.get('envs_per_panel',-1)) == int(envs) and
                int(old.get('horizon_steps',-1)) == int(horizon)):
            stale.append(run_dir)
    if not stale:
        return
    cmd=[sys.executable,'-m','emu.screening','--run-dir',*[str(x) for x in stale],
         '--device',device,'--envs',str(envs),'--horizon',str(horizon)]
    print(f"{LOG} held-out screen batch: {', '.join(x.name for x in stale)}",flush=True)
    subprocess.run(cmd,check=True)


def refresh_summaries(summaries: dict[str,dict[str,Any]], seed_root: Path,
                      specs: Iterable[ExperimentSpec]) -> None:
    for spec in specs:
        summaries[spec.name]=summary_with_screen(seed_root/'arms'/spec.name)

def ensure_foundation(seed_root: Path, *, seed: int, device: str, envs: int,
                      provenance: dict[str, Any]) -> Path:
    foundation=seed_root/'foundation20'
    complete,partial,expected=resolve_foundation_handoff(foundation,envs=envs)
    if complete is None:
        source=str(partial) if partial else 'random weights'
        print(f'{LOG} seed={seed} foundation: {source} -> {expected:,}',flush=True)
        cfg=foundation_config(run_dir=foundation,seed=seed,device=device,envs=envs,resume=partial)
        run_training(cfg,provenance=provenance)
        complete,partial,expected=resolve_foundation_handoff(foundation,envs=envs)
        if complete is None:
            raise RuntimeError(f'foundation failed to reach handoff boundary {expected:,}; partial={partial}')
    return complete


def ensure_arm(seed_root: Path, spec: ExperimentSpec, *, seed: int, target: int,
               foundation: Path, device: str, envs: int, provenance: dict[str, Any]) -> dict[str, Any]:
    run=seed_root/'arms'/spec.name
    ensure_spec(run,spec)
    complete,partial,expected=resolve_target_checkpoint(run,target_steps=target,envs=envs,label=spec.name)
    if complete is None:
        resume=partial or foundation
        print(f'{LOG} {spec.name}: -> {target:,} (boundary {expected:,}) from {resume}',flush=True)
        cfg=spec.config(run_dir=run,steps=target,seed=seed,device=device,envs=envs,resume=resume)
        t0=time.time(); run_training(cfg,provenance=provenance)
        print(f'{LOG} {spec.name}: train wall={time.time()-t0:.1f}s',flush=True)
        complete,partial,expected=resolve_target_checkpoint(run,target_steps=target,envs=envs,label=spec.name)
        if complete is None:
            raise RuntimeError(f'{spec.name} failed to reach {expected:,}; partial={partial}')
    else:
        print(f'{LOG} {spec.name}: already at {checkpoint_step(complete):,}',flush=True)
    return summarize(run)


def write_stage(root: Path, decisions: dict[str, Any], summaries: dict[str, dict[str, Any]], *, mirror: Path | None = None) -> None:
    root.mkdir(parents=True,exist_ok=True)
    (root/'tournament_decisions.json').write_text(json.dumps(decisions,indent=2,sort_keys=True),encoding='utf-8')
    names=sorted(summaries)
    if names:
        keys=['name','final_step','broad','screen_hard_recovery','screen_growth_headroom','recovery','worst_direction','calibration_mae','competent_nodes','known_nodes','competent_edges','known_edges','frontier_vx_hi','frontier_vx_lo']
        with (root/'tournament_summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader()
            for name in names: w.writerow({'name':name,**summaries[name]})
    if mirror is not None:
        mirror.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'tournament_decisions.json',mirror/'tournament_decisions.json')
        if (root/'tournament_summary.csv').exists():
            shutil.copy2(root/'tournament_summary.csv',mirror/'tournament_summary.csv')


def report_markdown(root: Path, decisions: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> None:
    lines=[f'# EMU {VERSION} capability-teacher tournament','',
           'Successive-halving experiment: twelve hypotheses share one seed-733 foundation; only promoted arms receive late-stage compute.','',
           '| arm | step | broad | hard recovery | growth headroom | cloud recovery | worst dir | cal MAE | nodes | edges |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in sorted(summaries):
        s=summaries[name]
        lines.append(f"| {name} | {int(_number(s.get('final_step'),-1)):,} | {_number(s.get('broad')):.3f} | {hard_recovery(s):.3f} | {growth_headroom(s):.3f} | {_number(s.get('recovery')):.3f} | {_number(s.get('worst_direction')):.3f} | {_number(s.get('calibration_mae')):.3f} | {int(_number(s.get('competent_nodes'),-1))}/{int(_number(s.get('known_nodes'),-1))} | {int(_number(s.get('competent_edges'),-1))}/{int(_number(s.get('known_edges'),-1))} |")
    lines += ['', '## Decisions', '', '```json', json.dumps(decisions,indent=2,sort_keys=True), '```', '']
    (root/'tournament_results.md').write_text('\n'.join(lines),encoding='utf-8')


def plan_dict(seed: int = 733, confirm_seed: int = 1742) -> dict[str, Any]:
    shadow,doses=base_specs()
    # B inherits A's selected dose. C inherits B's selected mechanism *and* the
    # selected dose. Do not print a fake concrete mechanism for C before data
    # exists; only its schedule transformation is knowable in advance.
    mechs=mechanism_specs(.25)
    schedules = [
        {'name':'C1_slow_ramp','inherits':'B selected mechanism/dose',
         'schedule':'influence ramps from 20M to full at 60M'},
        {'name':'C2_delayed','inherits':'B selected mechanism/dose',
         'schedule':'governor-only to 40M; influence ramps to full at 60M'},
        {'name':'C3_tutor_release','inherits':'B selected mechanism/dose',
         'schedule':'full by 30M; fade from 45M to zero at 60M'},
    ]
    return {
        'seed':seed,'confirm_seed':confirm_seed,
        'phase_A_50M':[asdict(x) for x in [shadow,*doses]],
        'phase_B_50M_reference_dose_25pct':[asdict(x) for x in mechs],
        'phase_B_note':'actual B dose is selected by phase A',
        'phase_C_60M':schedules,
        'promotion':'shadow + best dose + best mechanism variant + best schedule variant -> 80M; top two challengers + shadow -> 120M; best challenger + shadow -> confirmation seed',
        'automatic_third_seed':False,
    }


def print_plan(seed: int = 733, confirm_seed: int = 1742) -> None:
    p=plan_dict(seed,confirm_seed)
    print(json.dumps(p,indent=2,sort_keys=True))
    print('\nCompute funnel (nominal targets):')
    print('  A: 5 arms to 50M; B: 4 new arms to 50M; C: 3 new arms to 60M')
    print('  only 4 representatives -> 80M; shadow + 2 challengers -> 120M')
    print(f'  seed {confirm_seed}: only shadow + best challenger -> 120M; no automatic third seed')


def run_tournament(*, seed: int, confirm_seed: int, device: str, envs: int,
                   root: Path, results_dir: Path, screen_envs: int, screen_horizon: int,
                   stop_after: str) -> None:
    root=root.expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    repo=source_repo_root()
    if repo is not None and (repo==root or repo in root.parents):
        raise SystemExit(f'raw tournament root must be outside source repository: {root}')
    manifest={'seed':int(seed),'confirm_seed':int(confirm_seed),'device':str(device),'envs':int(envs),
              'screen_envs':int(screen_envs),'screen_horizon':int(screen_horizon),
              'dose_target':DOSE_TARGET,'schedule_target':SCHEDULE_TARGET,
              'semifinal_target':SEMIFINAL_TARGET,'final_target':FINAL_TARGET}
    manifest_path=root/'tournament_manifest.json'
    if manifest_path.exists():
        existing=_json(manifest_path,{})
        if existing != manifest:
            raise RuntimeError(f'tournament manifest drift: existing={existing} wanted={manifest}')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    provenance=git_provenance()
    seed_root=root/f'seed{seed}'
    seed_root.mkdir(parents=True,exist_ok=True)
    foundation=ensure_foundation(seed_root,seed=seed,device=device,envs=envs,provenance=provenance)

    decisions: dict[str,Any]={'version':VERSION,'git':provenance,'seed':seed,'confirm_seed':confirm_seed,
                              'foundation_checkpoint':str(foundation),'stages':{}}
    summaries: dict[str,dict[str,Any]]={}
    shadow,doses=base_specs()

    # A: dose response. Five arms, all only to 50M.
    for spec in [shadow,*doses]:
        summaries[spec.name]=ensure_arm(seed_root,spec,seed=seed,target=DOSE_TARGET,foundation=foundation,
            device=device,envs=envs,provenance=provenance)
    run_screen_batch([seed_root/'arms'/s.name for s in [shadow,*doses]],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(summaries,seed_root,[shadow,*doses])
    best_dose,rank_a=pick_best(doses,summaries,summaries[shadow.name],prefer_lower_dose_within=.015)
    decisions['stages']['A_dose']={'target':DOSE_TARGET,'selected':best_dose.name,'ranking':rank_a}
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)
    if stop_after=='A': return

    # B: mechanism. Four new arms at the dose learned in A.
    mechs=mechanism_specs(best_dose.dose)
    for spec in mechs:
        summaries[spec.name]=ensure_arm(seed_root,spec,seed=seed,target=DOSE_TARGET,foundation=foundation,
            device=device,envs=envs,provenance=provenance)
    run_screen_batch([seed_root/'arms'/s.name for s in mechs],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(summaries,seed_root,mechs)
    best_mech,rank_b=pick_best([best_dose,*mechs],summaries,summaries[shadow.name])
    best_mech_variant,rank_b_variants=pick_best(mechs,summaries,summaries[shadow.name])
    decisions['stages']['B_mechanism']={'target':DOSE_TARGET,'selected_overall':best_mech.name,
        'selected_variant':best_mech_variant.name,'ranking':rank_b,'variant_ranking':rank_b_variants}
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)
    if stop_after=='B': return

    # C: schedule. Compare at a common 60M step. Shadow and the baseline mechanism
    # get only the extra 10M they need; the three schedule arms start from foundation.
    summaries[shadow.name]=ensure_arm(seed_root,shadow,seed=seed,target=SCHEDULE_TARGET,foundation=foundation,
        device=device,envs=envs,provenance=provenance)
    summaries[best_mech.name]=ensure_arm(seed_root,best_mech,seed=seed,target=SCHEDULE_TARGET,foundation=foundation,
        device=device,envs=envs,provenance=provenance)
    schedules=schedule_specs(best_mech)
    for spec in schedules:
        summaries[spec.name]=ensure_arm(seed_root,spec,seed=seed,target=SCHEDULE_TARGET,foundation=foundation,
            device=device,envs=envs,provenance=provenance)
    cset=[shadow,best_mech,*schedules]
    run_screen_batch([seed_root/'arms'/s.name for s in cset],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(summaries,seed_root,cset)
    best_schedule_overall,rank_c=pick_best([best_mech,*schedules],summaries,summaries[shadow.name])
    best_schedule_variant,rank_c_variants=pick_best(schedules,summaries,summaries[shadow.name])
    decisions['stages']['C_schedule']={'target':SCHEDULE_TARGET,'selected_overall':best_schedule_overall.name,
        'selected_variant':best_schedule_variant.name,'ranking':rank_c,'variant_ranking':rank_c_variants}
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)
    if stop_after=='C': return

    # Four scientifically distinct representatives earn 80M: shadow, best dose,
    # best mechanism variant, best schedule variant. No generic top-4 lottery.
    semifinal_specs=[shadow,best_dose,best_mech_variant,best_schedule_variant]
    # Preserve order while guarding against an accidental duplicate name.
    semifinal_specs=list({s.name:s for s in semifinal_specs}.values())
    for spec in semifinal_specs:
        summaries[spec.name]=ensure_arm(seed_root,spec,seed=seed,target=SEMIFINAL_TARGET,foundation=foundation,
            device=device,envs=envs,provenance=provenance)
    run_screen_batch([seed_root/'arms'/s.name for s in semifinal_specs],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(summaries,seed_root,semifinal_specs)
    challengers=[s for s in semifinal_specs if s.name!=shadow.name]
    ranked80=ranked(challengers,summaries,summaries[shadow.name],strict_cloud=True)
    promoted_names=[r['name'] for r in ranked80[:2]]
    by_name={s.name:s for s in semifinal_specs}
    promoted=[by_name[n] for n in promoted_names]
    decisions['stages']['semifinal80']={'target':SEMIFINAL_TARGET,'promoted':['A0_shadow',*promoted_names],
                                       'ranking':ranked80}
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)
    if stop_after=='80': return

    # Only shadow + two challengers get the expensive 80->120M continuation.
    final_specs=[shadow,*promoted]
    for spec in final_specs:
        summaries[spec.name]=ensure_arm(seed_root,spec,seed=seed,target=FINAL_TARGET,foundation=foundation,
            device=device,envs=envs,provenance=provenance)
    run_screen_batch([seed_root/'arms'/s.name for s in final_specs],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(summaries,seed_root,final_specs)
    final_rank=ranked(promoted,summaries,summaries[shadow.name],strict_cloud=True)
    challenger=next(s for s in promoted if s.name==final_rank[0]['name'])
    decisions['stages']['final120']={'target':FINAL_TARGET,'challenger':challenger.name,'ranking':final_rank,
                                    'shadow_utility':utility(summaries[shadow.name])}
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)
    if stop_after=='120': return

    # Independent confirmation seed: only shadow + best challenger. No automatic
    # third seed -- mixed evidence should be inspected before buying more compute.
    confirm_root=root/f'seed{confirm_seed}'
    confirm_foundation=ensure_foundation(confirm_root,seed=confirm_seed,device=device,envs=envs,provenance=provenance)
    cshadow=replace(shadow,name='Z0_confirm_shadow',label=f'Confirmation shadow, seed {confirm_seed}')
    cchall=replace(challenger,name=f'Z1_confirm_{challenger.name}',label=f'Confirmation of {challenger.name}, seed {confirm_seed}')
    confirm_summaries={}
    confirm_specs=[cshadow,cchall]
    for spec in confirm_specs:
        confirm_summaries[spec.name]=ensure_arm(confirm_root,spec,seed=confirm_seed,target=FINAL_TARGET,
            foundation=confirm_foundation,device=device,envs=envs,provenance=provenance)
    run_screen_batch([confirm_root/'arms'/s.name for s in confirm_specs],device=device,envs=screen_envs,horizon=screen_horizon)
    refresh_summaries(confirm_summaries,confirm_root,confirm_specs)
    ok,reasons=gate(confirm_summaries[cchall.name],confirm_summaries[cshadow.name],strict_cloud=True)
    delta=utility(confirm_summaries[cchall.name])-utility(confirm_summaries[cshadow.name])
    seed733_delta=utility(summaries[challenger.name])-utility(summaries[shadow.name])
    if ok and delta>0.01 and seed733_delta>0.01:
        status='confirmed_challenger_advantage'
    elif (delta>0) != (seed733_delta>0) or abs(delta)<=0.01:
        status='mixed_or_tied_review_before_third_seed'
    else:
        status='shadow_favoured'
    decisions['stages']['confirmation']={'seed':confirm_seed,'challenger':challenger.name,
        'status':status,'seed733_utility_delta':seed733_delta,'confirm_utility_delta':delta,
        'confirm_gate_ok':ok,'confirm_gate_reasons':reasons}
    for k,v in confirm_summaries.items(): summaries[f'seed{confirm_seed}:{k}']=v
    write_stage(root,decisions,summaries,mirror=results_dir); report_markdown(results_dir,decisions,summaries)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=f'EMU {VERSION} efficient 12-hypothesis tournament')
    p.add_argument('--seed',type=int,default=733)
    p.add_argument('--confirm-seed',type=int,default=1742)
    p.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    p.add_argument('--envs',type=int,default=2048)
    p.add_argument('--screen-envs',type=int,default=512)
    p.add_argument('--screen-horizon',type=int,default=350)
    p.add_argument('--run-root',default='')
    p.add_argument('--results-dir',default='')
    p.add_argument('--stop-after',choices=['A','B','C','80','120','confirm'],default='confirm',
                   help='safe inspection breakpoint; rerunning resumes exactly where it stopped')
    p.add_argument('--plan',action='store_true',help='print the 12-arm funnel without starting JAX/training')
    a=p.parse_args(argv)
    if a.plan:
        print_plan(a.seed,a.confirm_seed)
        return 0
    suffix=f'{VERSION}_tournament'
    root=Path(a.run_root).expanduser() if a.run_root else run_base()/suffix
    results=Path(a.results_dir).expanduser() if a.results_dir else results_base()/suffix
    run_tournament(seed=a.seed,confirm_seed=a.confirm_seed,device=a.device,envs=a.envs,
                   root=root,results_dir=results,screen_envs=a.screen_envs,
                   screen_horizon=a.screen_horizon,stop_after=a.stop_after)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
