"""Four maintained run contracts: held-out physics, source-locked forks and checkpoint safety.

The 140M / 20M / eight-arm tournament launch scripts were disposable.
Test the *one training CLI and checkpoint contract* they exercised rather
than retaining obsolete campaign code just to keep its tests green.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

import numpy as np
import pytest

from scripts.smoke_training_lifecycle import configs
from emu.training_config import training_sensor_contract
from emu.contracts import OBS_SCHEMA_TERRAIN
from emu.run import one_run_config

EXAMPLE_STEPS=50_000_000
ENVS=2048

def training_flags(run_dir: Path, seed: int = 735) -> list[str]:
    """Representative modern training CLI, NOT a second experiment launcher."""
    return [
        "train", "--run-dir", str(run_dir),
        "--steps", str(EXAMPLE_STEPS), "--envs", str(ENVS),
        "--seed", str(seed), "--device", "cuda", "--profile", "sim2real",
        "--pitch-deg", "8", "--terrain-actor",
        "--terrain-step-max-m", ".08", "--unified-capabilities",
        "--terrain-carpet", "--carpet-prisms",
        "--carpet-transition-cap-deg", "3",
        "--foundation-steps", "10000000", "--foundation-mix", "main",
        "--growth-recipe", "main-mature", "--teaching-schedule", "terrain-early",
        "--terrain-lesson", "coherent-careful", "--force-program", "gentle",
        "--athletic-rehearsal-share", ".10", "--retention-share", ".15",
        "--academy-stream", "--academy-intro-steps", "600000000",
        "--academy-start-steps", "800000000",
        "--academy-intro-share", ".10", "--academy-ramp-steps", "10000000",
        "--academy-max-share", ".40", "--academy-height-cap-m", ".15",
        "--academy-course-episode-frames", "1200",
        "--fixed-eval-every-steps", "50000000",
        "--solver-iterations", "2", "--solver-ls-iterations", "4",
    ]


def test_current_cli_forks_preserve_sensor_and_reward_economics(tmp_path):
    from emu.cli import main as emu_cli_main
    cli_args=training_flags(tmp_path/"run")
    seen=[]
    def capture(ns):
        seen.append(one_run_config(ns))
        return 0
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args)==0
    cfg=seen[0]
    assert cfg.terrain_lesson=="coherent-careful"
    assert cfg.teaching_schedule=="terrain-early"
    assert cfg.fixed_eval_every_steps==50_000_000
    assert cfg.difficulty_base_step==pytest.approx(.0024)
    assert cfg.fall_target==pytest.approx(.34)
    assert (cfg.vx_hi_max,cfg.vx_lo_min,cfg.vy_max_mps,cfg.yaw_max_rad_s)==pytest.approx(
        (.60,-.20,.22,1.50))
    assert (cfg.combo_probe_frac,cfg.phase_speed_gain,cfg.regime_command_frac,
            cfg.combo_mix_frac)==pytest.approx((.12,1.25,.85,.88))
    assert cfg.frontier_push_safety_n==pytest.approx(.70)
    assert cfg.push_max_force_n==pytest.approx(4.)
    assert cfg.capability_field_mode=="shadow"
    assert cfg.freeze_obs_norm_after_steps==0
    assert cfg.stillness_birth_frac==0.
    assert cfg.science_step_finish_on_transfer is False
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args+["--science-step-finish-on-transfer"])==0
    assert seen[-1].science_step_finish_on_transfer is True
    # Real research forks may change teacher and solver, but not the
    # 360-input actor geometry, physical robot or sensor history.
    from emu.run import verify_experiment_fork
    fork=replace(cfg,steps=100_000_000,academy_intro_steps=50_000_000,
                 academy_start_steps=50_000_000,stillness_birth_frac=.10)
    verify_experiment_fork(cfg,fork)
    with pytest.raises(RuntimeError,match="observation_schema"):
        verify_experiment_fork(cfg,replace(fork,observation_schema="sim2real_232"))
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args+["--stillness-birth-frac",".12"])==0
    assert seen[-1].stillness_birth_frac==pytest.approx(.12)
    with patch("emu.run.train_one",side_effect=lambda ns:one_run_config(ns)):
        with pytest.raises(ValueError,match="stillness-birth-frac"):
            emu_cli_main(cli_args+["--stillness-birth-frac",".4"])
    from emu.rewards import _gait_maturity
    m=dict(age_s=np.asarray([2.],np.float32),
           stepping_activity_so_far=np.asarray([0.],np.float32),
           cmd_vx=np.asarray([0.],np.float32),
           cmd_vy=np.asarray([0.],np.float32),
           cmd_yaw=np.asarray([0.],np.float32))
    assert float(_gait_maturity(m,np)[1][0])==pytest.approx(.30)
    assert float(_gait_maturity(m,np,allow_zero_command_balance=True)[1][0])==pytest.approx(1.)
    m["cmd_vx"]=np.asarray([.12],np.float32)
    assert float(_gait_maturity(m,np,allow_zero_command_balance=True)[1][0])==pytest.approx(.30)
    assert cfg.physical_profile=="h40"
    assert cfg.gait_guidance_fade_steps == 0
    # One new research profile has an explicit outcome transition, not a
    # rewired frozen champion or an undocumented change of teaching weights.
    from emu.capability_focus import gait_guidance_strength
    with patch.dict("os.environ", {"EMU_PHYSICAL_PROFILE": "h40-spring-v1"}):
        v2 = one_run_config(SimpleNamespace(
                profile="sim2real", run_dir=str(tmp_path/"spring-outcomes"),
                steps=120_000_000, envs=2048, seed=735, device="cuda",
                terrain_actor=True, unified_capabilities=True,
                terrain_step_max_m=.08, terrain_carpet=True, carpet_prisms=True,
                academy_stream=True, terrain_lesson="coherent-careful",
                growth_recipe="main-mature", force_program="gentle"))
    assert v2.physical_profile == "h40-spring-v1"
    assert v2.gait_guidance_fade_steps == 20_000_000
    # Use actual source-pinned step ages, no fixed batch-age reset effects.
    assert [gait_guidance_strength(step=n, foundation_steps=20_000_000,
                                  fade_steps=v2.gait_guidance_fade_steps)
            for n in (0, 20_000_000, 25_000_000, 30_000_000,
                      35_000_000, 40_000_000, 120_000_000)
            ] == pytest.approx([1.,1.,.75,.5,.25,0.,0.])
    assert gait_guidance_strength(step=120_000_000,
                                  foundation_steps=20_000_000,
                                  fade_steps=0) == 1.
    from emu.rewards import weighted_training_terms
    money=dict(alive=0., locomotion=0., posture=1., contact=1.,
               anti_slip=0., clearance=1., smoothness=0.,
               joint_safety=0.)
    early, early_terms = weighted_training_terms(
        **money, stepping_gate=.3, gait_guidance_strength=1.)
    late, late_terms = weighted_training_terms(
        **money, stepping_gate=1., gait_guidance_strength=0.)
    assert early_terms["clearance"] == pytest.approx(.16)
    assert late_terms["clearance"] == pytest.approx(0.)
    assert late_terms["posture"] == pytest.approx(.68)
    assert late_terms["contact"] == pytest.approx(.5)
    assert early_terms["posture"] < late_terms["posture"]
    assert late > early
    # At 40M the step cadence *and* foot-under-hip pose cease to earn
    # rewards. Real attitude/height/support remain as outcome constraints.
    from emu.rewards import _physical_qualities
    qualities=dict(attitude_quality=np.array([.9]),
                   height_error=np.array([0.]),
                   cmd_pitch=np.array([0.]),
                   foot_mid_cone=np.array([0.]),
                   bad_contact=np.array([0.]),
                   support=np.array([1.]),
                   stance_slip=np.array([0.]),
                   swing_clearance=np.array([.04]),
                   target_step_height=np.array([.03]),
                   organic_step_shape=np.array([.8]),
                   action_delta_norm=np.array([0.]),
                   joint_limit_pressure=np.array([0.]),
                   scissor_pressure=np.array([0.]))
    first = _physical_qualities(qualities,np,gait_guidance_strength=1.)
    free = _physical_qualities(qualities,np,gait_guidance_strength=0.)
    same = _physical_qualities({**qualities,
              "foot_mid_cone":np.array([1.])},np,
              gait_guidance_strength=0.)
    assert float(first[0][0]) < float(free[0][0])
    assert float(free[0][0]) == pytest.approx(float(same[0][0]))
    assert not {"cloud_probes_per_eval","cloud_final_probes",
                "science_reset_step_evidence_on_fork"} & asdict(cfg).keys()
    assert training_sensor_contract(cfg) is not None
    # The final one-paste WSL command contains embedded Python report and
    # 20M safety-gate blocks. Compile all heredocs and the surrounding Bash
    # syntax in existing core acceptance (no extra test family).
    import re
    import shutil
    import subprocess
    runner = Path(__file__).resolve().parents[1] / "scripts/run_h40_spring_v1_wsl.sh"
    source = runner.read_text(encoding="utf-8")
    for block in re.findall(r"python - <<'PY'\n(.*?)\nPY", source, re.S):
        compile(block, str(runner), "exec")
    assert len(re.findall(r"python - <<'PY'",source)) >= 2
    if shutil.which("bash"):
        subprocess.run(["bash","-n",str(runner)],check=True)
    # The one follow-on Spring V1 lesson is a SINGLE source-pinned fork from
    # the fully trained 120M final PPO/optimiser. It preserves the two-million
    # proof gate and actual 4–80mm frozen physical A/B exam.
    import ast
    tutor_runner = runner.parent / "run_h40_stair_outcome_tutor_wsl.sh"
    tutor_source = tutor_runner.read_text(encoding="utf-8")
    for block in re.findall(r"<<'PY'\n(.*?)\nPY",tutor_source,re.S):
        compile(block,str(tutor_runner),"exec")
    assert len(re.findall(r"<<'PY'",tutor_source)) >= 3
    if shutil.which("bash"):
        subprocess.run(["bash","-n",str(tutor_runner)],check=True)
    assert 'ORIGIN_CKPT_SHA="a542722eaf0d' in tutor_source
    assert '--steps 122000000' in tutor_source
    assert '--steps 140000000 --resume auto' in tutor_source
    assert '--science-step-goal-weight .22' in tutor_source
    assert '--science-step-velocity-relief .55' in tutor_source
    assert '--science-step-approach-m .22' in tutor_source
    assert '--science-step-finish-on-transfer' not in tutor_source
    assert '--academy-height-cap-m .15' in tutor_source
    assert '--repair-partial-parent' not in tutor_source
    assert 'tutor_2m_gate.json' in tutor_source
    # Next single experiment: resume authenticated 140M explorer,
    # reward physical two-foot support and forward followthrough rather
    # than paying additional lead-foot contacts or reinstating gait phases.
    transfer_runner=runner.parent/"run_h40_stair_transfer_160m_wsl.sh"
    transfer_src=transfer_runner.read_text(encoding="utf-8")
    if shutil.which("bash"):
        subprocess.run(["bash","-n",str(transfer_runner)],check=True)
    snippets=re.findall(r"<<'PY'\n(.*?)\nPY",transfer_src,re.S)
    assert len(snippets)>=3
    for snippet in snippets:
        compile(snippet,str(transfer_runner),"exec")
    assert '17f6fb17cc699342f2ff394e1c6eb8d237675023' in transfer_src
    assert '0324fa18d5e624c7a38a7085050cc2abc964f30e8eb453d00d5b280e6097e039' in transfer_src
    assert '--steps 142000000' in transfer_src and '--steps 160000000' in transfer_src
    assert '--science-step-goal-weight .12' in transfer_src
    assert '--science-bilateral-support-bonus 6' in transfer_src
    assert '--science-followthrough-bonus 7' in transfer_src
    assert '--science-step-velocity-relief .55' in transfer_src
    assert '--science-step-approach-m .22' in transfer_src
    assert '--retention-share .20' in transfer_src
    assert '--science-step-finish-on-transfer' not in transfer_src
    assert '--science-foothold-bonus' not in transfer_src
    assert '--science-supported-cross-bonus' not in transfer_src
    assert 'transfer_2m_gate.json' in transfer_src
    # Regression for the real 160M completed-run failure: the comparison
    # report was written as transfer_comparison.*, but the final ZIP still
    # required obsolete tutor_comparison.* filenames. ZIP assembly MUST
    # use the files that this same comparator actually writes.
    transfer_comparator=(transfer_runner.parent/
                         "compare_h40_stair_transfer_tutor.py")
    transfer_cmp_source=transfer_comparator.read_text(encoding="utf-8")
    assert '(args.run/"transfer_comparison.json").write_text(' in transfer_cmp_source
    assert '(args.run/"transfer_comparison.md").write_text(' in transfer_cmp_source
    assert 'args.run/"transfer_comparison.json"' in transfer_cmp_source.split(
        "    paths=[",1)[1]
    assert 'args.run/"transfer_comparison.md"' in transfer_cmp_source.split(
        "    paths=[",1)[1]
    assert 'tutor_comparison.md' not in transfer_cmp_source
    assert 'tutor_comparison.json' not in transfer_cmp_source
    # Exercise the complete ZIP writer using tiny synthetic files. The real
    # 160M PPO/exams succeeded, but this final packaging step was previously
    # untested and failed only AFTER the 4000 physical trials finished.
    import zipfile
    from unittest.mock import patch
    import scripts.compare_h40_stair_transfer_tutor as transfer_mod
    fake_run=tmp_path/"transfer-zip-smoke"
    fake_parent=tmp_path/"frozen-140m-exam"
    fake_run.mkdir()
    for name in (
        "transfer_source_manifest.json", "transfer_2m_gate.json",
        "training_config.json","run_meta.json","metrics.csv",
        "h40-stair-transfer-tutor-160m.emupolicy",
    ):
        (fake_run/name).write_text("test fixture",encoding="utf-8")
    for root in (fake_parent,fake_run/"heldout"):
        for kind in ("stairs","flat"):
            d=root/kind
            d.mkdir(parents=True)
            for name in ("matrix_summary.json","matrix_curve.csv","matrix_worlds.csv"):
                (d/name).write_text("test fixture",encoding="utf-8")
    with (patch.object(transfer_mod,"PARENT_EXAM",fake_parent),
          patch.object(transfer_mod,"compare",return_value={"test":True}),
          patch.object(transfer_mod,"report",return_value=(
              "# Synthetic comparison\n",{"champion_candidate":False})),
          patch("sys.argv",["transfer-comparison","--run",str(fake_run)])):
        transfer_mod.main()
    with zipfile.ZipFile(fake_run/"transfer_results_bundle.zip") as bundle:
        files=set(bundle.namelist())
    assert {"transfer_comparison.md","transfer_comparison.json",
            "heldout/stairs/matrix_worlds.csv",
            "heldout/flat/matrix_worlds.csv",
            "reference/stairs/matrix_worlds.csv",
            "reference/flat/matrix_worlds.csv"}.issubset(files)
    assert not any("tutor_comparison" in name for name in files)
    from scripts.compare_h40_stair_transfer_tutor import report as transfer_report
    tasks=("flat","up_4_40","up_44_60","up_64_80","down_44_80")
    old_vals=dict(attempts=250,falls=20,lead_top=.55,both_top=.20,
                  upright_cross=.16,aligned_cross=.016,survival=.56,
                  fell_before_lead=3,fell_during_transfer=10,
                  fell_after_both=2,
                  mean_forward_progress_m=.65,mean_max_lateral_offset_m=.70)
    new_vals=dict(old_vals)
    old_groups={k:dict(old_vals) for k in tasks}
    new_groups={k:dict(new_vals) for k in tasks}
    report_text,gate=transfer_report(
        dict(parent=old_groups,tutor=new_groups,
             delta={k:dict(both_top=0.,aligned_cross=0.) for k in tasks}))
    assert "two-foot" in report_text.lower() or "both feet" in report_text.lower()
    assert not gate["champion_candidate"]
    new_groups["up_44_60"].update(
        both_top=.35,aligned_cross=.10,survival=.70,lead_top=.55)
    new_groups["up_4_40"].update(aligned_cross=.45,survival=.95)
    new_groups["down_44_80"].update(aligned_cross=.48,survival=.90)
    new_groups["flat"].update(aligned_cross=.54,survival=1.)
    report_text,gate=transfer_report(dict(
        parent=old_groups,tutor=new_groups,
        delta={k:dict(
            both_top=new_groups[k]["both_top"]-old_groups[k]["both_top"],
            aligned_cross=new_groups[k]["aligned_cross"]-old_groups[k]["aligned_cross"])
            for k in tasks}))
    assert gate["all_gates_pass"]
    assert gate["champion_candidate"]
    comparator=tutor_runner.parent / "compare_h40_stair_outcome_tutor.py"
    ast.parse(comparator.read_text(encoding="utf-8"))
    from scripts.compare_h40_stair_outcome_tutor import summarize, report
    sample=dict(
        attempts=50,falls=5,lead_top=.4,both_top=.3,upright_cross=.2,
        aligned_cross=.1,survival=.9,mean_forward_progress_m=.2,
        mean_max_lateral_offset_m=.3)
    d=dict(parent={k:sample.copy() for k in (
                "flat","up_4_40","up_44_60","up_64_80","down_44_80")},
           tutor={k:sample.copy() for k in (
                "flat","up_4_40","up_44_60","up_64_80","down_44_80")},
           delta={k:dict(lead_top=0.,upright_cross=0.) for k in (
                "flat","up_4_40","up_44_60","up_64_80","down_44_80")})
    report_md,gate=report(d)
    assert "44–60" in report_md
    assert gate["champion_candidate"] is False
    assert gate["foothold_gain"] is False
    # The launcher must obey smoke_gpu_carpet's --full-update restriction:
    # 128 worlds = small PPO check; 2048 worlds = real full PPO update.
    smoke_command = source.split('if [[ "$MODE" == smoke ]]; then',1)[1]
    smoke_command, full_command = smoke_command.split('  else',1)
    full_command = full_command.split('  fi',1)[0]
    assert "--worlds 128" in smoke_command
    assert "--full-update" not in smoke_command
    assert "--worlds 2048" in full_command
    assert "--full-update" in full_command
    # The terminal metrics bug was caused by two different report schemas.
    # Periodic AND final report sites must both call one common function.
    from emu.training import attach_unified_teaching_report
    import inspect
    src = inspect.getsource(__import__("emu.training",fromlist=["train"]).train)
    assert src.count("attach_unified_teaching_report(") == 2
    p = dict(regime_frac_steady=.8,regime_frac_transition=.1,
             regime_frac_resilience=.1,foundation_control_mix=0.,
             gait_guidance_strength=.75,
             athletic_rehearsal_frac=.1,specialist_mix=.5,
             focus_weight_step=.25,academy_episode_probability=.2)
    for name in ("command","resilience","slope","step","transition","control"):
        p["focus_weight_"+name]=.25 if name=="step" else .15
    report={}
    attach_unified_teaching_report(
        report,p,step=25_000_000,
        args=SimpleNamespace(foundation_steps=20_000_000,foundation_mix="main"),
        governor=SimpleNamespace(params=SimpleNamespace(steady_difficulty=.2)))
    assert report["gait_guidance_strength"] == .75
    assert report["foundation_control_mix"] == 0.
    assert report["academy_expected_birth_fraction"] == pytest.approx(.0225)
    assert report["teaching_profile"] == "growth"

    # Normal cross-source experiments STILL insist on a published final.pkl.
    # Only an explicit byte+source-pinned repair flag permits an incomplete
    # latest/snapshot ancestor; nobody can quietly bypass source provenance.
    from emu.run import _validate_cross_source_research_parent
    import hashlib
    ck = tmp_path/"repair-origin"/"checkpoints"/"latest.pkl"
    ck.parent.mkdir(parents=True,exist_ok=True)
    ck.write_bytes(b"validated ancestor bytes for permission test")
    parent_sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    original="a"*40
    meta={"meta":{"source_provenance":{"git_commit":original}}}
    provenance={"commit":"b"*40}
    with pytest.raises(RuntimeError,match="completed final.pkl"):
        _validate_cross_source_research_parent(
            ck,payload=meta,expected_source=original,
            expected_sha256=parent_sha,provenance=provenance)
    _validate_cross_source_research_parent(
        ck,payload=meta,expected_source=original,
        expected_sha256=parent_sha,provenance=provenance,
        repair_partial_parent=True)
    with pytest.raises(RuntimeError,match="SHA256 changed"):
        _validate_cross_source_research_parent(
            ck,payload=meta,expected_source=original,
            expected_sha256="f"*64,provenance=provenance,
            repair_partial_parent=True)
    with pytest.raises(RuntimeError,match="best/latest/snapshot"):
        _validate_cross_source_research_parent(
            ck.with_name("recovery.pkl"),payload=meta,expected_source=original,
            expected_sha256=parent_sha,provenance=provenance,
            repair_partial_parent=True)
    # The final frozen exam ran BEFORE terminal CSV failure and saved
    # fully trained best.pkl weights + optimiser at the exact 20M boundary.
    # Only a genuine scored/evaluated actor is eligible for this rescue.
    best=ck.with_name("best.pkl")
    best.write_bytes(ck.read_bytes())
    meta_best={"meta":{"source_provenance":{"git_commit":original},
                       "evaluated_policy":True,"best_eval_score":.157,
                       "step":20_054_016}}
    _validate_cross_source_research_parent(
        best,payload=meta_best,expected_source=original,
        expected_sha256=parent_sha,provenance=provenance,
        repair_partial_parent=True)
    with pytest.raises(RuntimeError,match="best/latest/snapshot"):
        _validate_cross_source_research_parent(
            best,payload=meta,expected_source=original,
            expected_sha256=parent_sha,provenance=provenance,
            repair_partial_parent=True)
    import ast
    ast.parse((runner.parent/"recover_h40_spring_v1.py").read_text(encoding="utf-8"))
    if shutil.which("bash"):
        subprocess.run(
            ["bash","-n",str(runner.parent/"recover_h40_spring_v1_wsl.sh")],
            check=True)
    # Keep 24 core tests: this UI is another assertion of the *existing*
    # public CLI contract, not a new training experiment/test family.
    _check_readonly_dashboard(tmp_path)


def _check_readonly_dashboard(tmp_path):
    import csv
    import json
    from emu.console_dashboard import (
        read_history, render, sampled_window, delta, target_from_config)
    from emu.cli import main as cli_main
    directory = tmp_path / "dashboard-run"
    directory.mkdir()
    (directory/"training_config.json").write_text(
        json.dumps({"steps":120000000}),encoding="utf-8")
    assert target_from_config(directory)==120000000
    rows=[
        dict(step=95000000,reward_mean=2.0,fall_rate=.2,
             academy_near_attempts_up=10,academy_near_successes_up=1),
        dict(step=100000000,reward_mean=2.5,fall_rate=.1,
             academy_near_attempts_up=8,academy_near_successes_up=1),
    ]
    source=directory/"metrics.csv"
    with source.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before=source.read_bytes()
    history=read_history(source)
    assert delta(history,"reward_mean",precision=1)=="+0.5"
    assert sampled_window(history,"academy_near_attempts_up",
                          "academy_near_successes_up")=="1/8 (12.5%)"
    screen=render(history,run_name=directory.name,
                  target=120000000,width=100,height=27)
    assert "100.0M / 120.0M" in screen
    assert "STAIRS" in screen and "NOT all-episode counts" in screen
    assert "not yet evaluated" in screen
    assert cli_main(["dashboard","--run-dir",str(directory),
                     "--once","--plain"]) == 0
    assert source.read_bytes()==before


def test_actual_evaluator_uses_prisms_not_boxes(tmp_path, monkeypatch):
    import emu.evaluation as evaluate
    cli_args = training_flags(tmp_path / "run")
    config = []

    with patch("emu.run.train_one", side_effect=lambda ns: (config.append(one_run_config(ns)), 0)[1]):
        from emu.cli import main as emu_cli_main
        emu_cli_main(cli_args)
    cfg = config[0]
    call = {}

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            call.update(kwargs)
            self.sensor_contract = training_sensor_contract(cfg)

    monkeypatch.setattr(evaluate, "EmuMJXEnv", FakeEnv)
    monkeypatch.setattr(evaluate, "make_trainer", lambda env, **kwargs: None)
    env, trainer = evaluate._world(cfg, seed_offset=0,
                                   limits=(.60, -.20, .22, 1.50))
    assert env is not None and trainer is None
    assert call["terrain_carpet"] is True
    assert call["terrain_carpet_prisms"] is True
    assert call["observation_schema"] == OBS_SCHEMA_TERRAIN
    assert call["solver_iterations"] == 2
    assert call["solver_ls_iterations"] == 4
    # Frozen panels must use the actor's actual trained physics,
    # including a changed 4/4 treatment rather than XML defaults.
    from dataclasses import replace
    evaluate._world(replace(cfg, solver_iterations=4, solver_ls_iterations=4),
                    seed_offset=0, limits=(.60,-.20,.22,1.50))
    assert call["solver_iterations"] == 4
    assert call["solver_ls_iterations"] == 4
    # A frozen exam may never fall back to old nine-box or slab contact
    # because it happens to request a flat lesson.
    from dataclasses import replace
    with pytest.raises(ValueError, match="real unified prism world"):
        evaluate._world(replace(cfg, terrain_carpet_prisms=False),
                        seed_offset=0, limits=(.60, -.20, .22, 1.50))


def test_prism_two_update_checkpoint_lifecycle_recipe(tmp_path):
    first, second = configs(tmp_path, unified=True, carpet=True,
                            prisms=True, envs=8, seed=735)
    assert first.terrain_carpet_prisms and second.terrain_carpet_prisms
    assert first.terrain_carpet and first.academy_stream
    assert first.observation_schema == OBS_SCHEMA_TERRAIN
    assert first.steps == 256 and second.steps == 512
    # A source-locked training config cannot silently rebuild missing 5045
    # physical fields using a pre-prism checkpoint's defaults.
    import json
    from emu.training_config import TrainingConfig
    cfg_path = tmp_path / "training_config.json"
    first.save(cfg_path)
    assert TrainingConfig.load(cfg_path) == first
    partial = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The new physical lineage has no pre-v5054 migration exceptions.
    # A missing teaching field is a different recipe, not a compatible run.
    partial.pop("science_step_finish_on_transfer")
    cfg_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError,match="training config keys mismatch"):
        TrainingConfig.load(cfg_path)
    partial = asdict(first)
    partial.pop("terrain_carpet_prisms")
    cfg_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError,match="training config keys mismatch"):
        TrainingConfig.load(cfg_path)
    with pytest.raises(ValueError,match="prisms"):
        configs(tmp_path, unified=True, carpet=True, envs=8, seed=735)
    # Old synthetic terrain randomization is not an alternative collider.
    from emu.domain_randomization import DomainRandomizationConfig
    manufacturing = DomainRandomizationConfig(
        enabled=True, robot_radius=.6, hardware_radius=.6).normalized()
    assert manufacturing.robot_radius == .6 and manufacturing.hardware_radius == .6
    assert not hasattr(manufacturing, 'terrain_radius')
    assert not hasattr(first, 'domain_terrain_radius')
    with pytest.raises(TypeError):
        DomainRandomizationConfig(terrain_radius=.3)
    with pytest.raises(ValueError):
        configs(tmp_path, prisms=True, envs=8, seed=735)

    # Exact-current-source curriculum state is part of the same checkpoint
    # contract. Older pooled evidence cannot quietly become v5045 progress.
    from copy import deepcopy
    from emu.curriculum import RegimeGovernor
    gov = RegimeGovernor(unified_capability_focus=True)
    gov.terrain_evidence['up_near_attempts'] = 5.
    gov.terrain_evidence['up_near_successes'] = 3.
    gov.academy_evidence['up_near_attempts'] = 2.
    original = gov.state_dict()
    restored = RegimeGovernor(unified_capability_focus=True)
    restored.load_state_dict(original)
    assert restored.state_dict() == original
    # A successful instant stair finish is NOT a 40-frame fall. The
    # 350-frame equivalent is used solely for governor competence, while
    # truthful episode lengths remain available in the report.
    succeeded = RegimeGovernor(unified_capability_focus=True)
    corrected=succeeded.update(dict(
        fall_rate=0.,episodes=1.,ep_len_mean=40.,
        core_fall_rate=0.,core_ep_len_mean=40.,
        core_ep_len_competence_mean=350.))
    assert corrected["core_competence"]==pytest.approx(1.)
    uncorrected=RegimeGovernor(unified_capability_focus=True)
    old=uncorrected.update(dict(
        fall_rate=0.,episodes=1.,ep_len_mean=40.,
        core_fall_rate=0.,core_ep_len_mean=40.))
    assert old["core_competence"]<.12
    for remove in ('academy_evidence', 'terrain_evidence', 'command_evidence'):
        corrupt = deepcopy(original)
        del corrupt[remove]
        with pytest.raises(ValueError, match='curriculum state keys mismatch'):
            restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    del corrupt['params']['academy_height_up_m']
    with pytest.raises(ValueError, match='curriculum parameter keys mismatch'):
        restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    del corrupt['command_evidence']['vx_hi']['easy_attempts']
    with pytest.raises(ValueError, match='bad command frontier evidence'):
        restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    corrupt['academy_evidence']['updates'] = 2.
    with pytest.raises(ValueError, match='Academy evidence checkpoint mismatch'):
        restored.load_state_dict(corrupt)
    # A STEP teacher advances only when both signs have independently
    # demonstrated supported near-frontier crossing.
    step_gov = RegimeGovernor(unified_capability_focus=True)
    assert step_gov.advance_terrain({}, max_height_m=.08) == 'terrain_unified_seed_4mm'
    before = step_gov.params.terrain_height_m
    near = {f'terrain_near_{kind}_{direction}': val
            for direction in ('up','down')
            for kind,val in (('attempts',4.),('successes',3.))}
    assert step_gov.advance_terrain(near,max_height_m=.08) == 'terrain_height_expand_real_cross'
    assert step_gov.params.terrain_height_m > before
    # An easy failed encounter has priority over near successes; this is
    # an actual retreat, not an academic metric-only diagnostic.
    easy = {f'terrain_easy_{kind}_up': val
            for kind,val in (('attempts',4.),('successes',0.))}
    assert step_gov.advance_terrain(easy,max_height_m=.08) == 'terrain_retreat_easy_failure'
    assert step_gov.params.terrain_height_m < before + .0011
    # The unified controller's generic frontier helper must NOT independently
    # promote signed command axes: six specialist proof cohorts own those.
    g = RegimeGovernor(unified_capability_focus=True, open_frontier=True)
    g.fall_ema['resilience'] = .20
    g.competence['resilience'] = .60
    old_vx = g.params.frontier_vx_hi
    old_push = g.params.frontier_push_n
    ev = g._advance_physical_frontiers(
        {},global_fall=.20,global_retreat=False,specialist_growth_enabled=True)
    assert ev['push_n'] == 'expand' and g.params.frontier_push_n > old_push
    assert ev['combo'] == 'no_edge_data'
    assert g.params.frontier_vx_hi == old_vx
    assert g._advance_physical_frontiers(
        {},global_fall=.20,global_retreat=False,specialist_growth_enabled=False) == {}


class _SavedLearner(NamedTuple):
    params: Any
    opt_state: Any


def test_current_360_checkpoint_roundtrip_rejects_sensor_drift(tmp_path):
    """A real 360-input contract must survive a saved optimizer+normalizer."""
    from emu.contracts import TERRAIN_OBS_DIM
    from emu.training_config import TrainingConfig
    from emu.checkpoint_io import (
        save_checkpoint, load_checkpoint, TERRAIN_CHECKPOINT_SCHEMA)
    from emu.ppo import expected_param_shapes

    cfg=TrainingConfig(observation_schema=OBS_SCHEMA_TERRAIN,
                       unified_capability_focus=True)
    contract=training_sensor_contract(cfg)
    def build(spec):
        if isinstance(spec, dict):
            return {k:build(v) for k,v in spec.items()}
        if isinstance(spec, list):
            return [build(v) for v in spec]
        return np.zeros(spec,dtype=np.float32)
    params=build(expected_param_shapes())
    for family in ('pi','vf'):
        old=params[family][0]['w']
        params[family][0]['w']=np.zeros((TERRAIN_OBS_DIM,old.shape[1]),np.float32)
    state=_SavedLearner(params,{'count':np.asarray(1,np.int32)})
    norm={'mean':np.zeros(TERRAIN_OBS_DIM,np.float32),
          'var':np.ones(TERRAIN_OBS_DIM,np.float32),'count':8.0}
    meta=dict(step=256,lr_scale=.25,
              checkpoint_schema=TERRAIN_CHECKPOINT_SCHEMA,
              observation_schema=OBS_SCHEMA_TERRAIN,
              obs_dim=TERRAIN_OBS_DIM,sensor_contract=contract)
    file=tmp_path/'current_360.pkl'
    save_checkpoint(file,state,meta,norm,
                    curriculum_state={'params':{}},health_state={},field_state={})
    restored=load_checkpoint(file,validate=True,
                expected_observation_schema=OBS_SCHEMA_TERRAIN,
                expected_sensor_contract=contract)
    assert restored['meta']['step']==256
    np.testing.assert_array_equal(restored['state']['params']['pi'][0]['w'],
                                  params['pi'][0]['w'])
    corrupted={**contract,'sensor_model_revision':'tripod_tof_flow_async_v1'}
    with pytest.raises(ValueError):
        load_checkpoint(file,expected_observation_schema=OBS_SCHEMA_TERRAIN,
                        expected_sensor_contract=corrupted)

