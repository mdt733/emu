"""v5020 branch-lab acceptance: no CUDA or heavyweight training required."""
import json
from pathlib import Path
from dataclasses import asdict

import pytest

from emu import campaign5020 as lab
from emu.contracts import OBS_SCHEMA_PREVIEW
from emu.training_config import training_sensor_contract


def test_recipe_has_shared_root_multiple_challengers_and_winner():
    p=lab.plan(733,512,32,300)
    assert p["total_environment_steps_all_arms"]==380_000_000
    assert p["foundation_steps"]==20_000_000
    assert p["round_targets_absolute_steps"]=={
        "round1":80_000_000,"round2":120_000_000,"round3":160_000_000}
    assert list(lab.ARMS)==["balanced","slope_recovery","step_focus","efficient"]
    assert p["observation_contract"]==OBS_SCHEMA_PREVIEW


def test_all_arms_inherit_exact_same_physical_actor_contract(tmp_path):
    root=tmp_path/"v5020"
    shared=lab.config(root/"foundation20",stage="foundation",arm=None,
                      seed=733,envs=512,device="cuda")
    assert shared.observation_schema==OBS_SCHEMA_PREVIEW
    assert shared.domain_terrain_mode=="contact"
    assert shared.domain_terrain_radius>0
    assert shared.terrain_height_safety_m==0.
    contract=training_sensor_contract(shared)
    for stage,limit in (("round1",.030),("round2",.050),("round3",.060)):
        for arm in lab.ARMS:
            run=root/stage/arm
            cfg=lab.config(run,stage=stage,arm=arm,seed=733,envs=512,device="cuda")
            assert cfg.steps==lab.ROUNDS[stage]
            assert cfg.observation_schema==OBS_SCHEMA_PREVIEW
            assert training_sensor_contract(cfg)==contract
            assert cfg.domain_randomization and cfg.domain_terrain_mode=="contact"
            assert cfg.domain_terrain_radius>0.
            assert cfg.terrain_height_safety_m==limit
            assert cfg.terrain_random_placement
            assert cfg.terrain_height_initial_m==.003
            assert cfg.forward_preview_masked is False
            assert cfg.attitude_profile=="v5015"
            assert cfg.push_max_force_n is not None


def test_plan_and_status_do_not_create_campaign(tmp_path,capsys):
    root=tmp_path/"planned"
    assert lab.main(["--root",str(root)])==0
    out=json.loads(capsys.readouterr().out)
    assert out["total_environment_steps_all_arms"]==380_000_000
    assert not root.exists()
    assert lab.main(["--root",str(root),"--status"])==0
    out=json.loads(capsys.readouterr().out)
    assert out["context"] is None
    assert not root.exists()


def test_result_rank_requires_measured_physical_course_and_slope():
    p={}
    for panel in ("core","resilience100","robust70","growth140",
                  "growthcombo140","contact_slope","contact_harder"):
        p[f"screen_{panel}_survival"]=.9
    for panel in ("robust70","growth140","contact_slope"):
        for axis in ("vx","vy","yaw"):
            p[f"screen_{panel}_{axis}_mae"]=.08
    for panel in ("core","robust70","contact_slope"):
        p[f"screen_{panel}_height_shortfall_m"]=.005
    for panel in ("robust70","contact_slope"):
        p[f"screen_{panel}_step_evidence"]=.5
    p.update(course_20mm_crossing_rate=.25,course_20mm_contact_rate=.50,
             course_20mm_attempt_count=20.,
             power_robust_shaft_abs_power_w=3.0,
             power_contact_slope_shaft_abs_power_w=3.2,
             power_robust_over_rated_frac=.20,
             power_contact_slope_over_rated_frac=.25)
    score=lab._score_arm(p,"balanced")
    assert score["eligible"] and score["course_20mm_crossing"]==.25
    assert score["servo_over_rated_frac"]==.225
    high_power=lab._score_arm({**p,
        "power_robust_shaft_abs_power_w":15.,
        "power_contact_slope_shaft_abs_power_w":15.},"efficient")
    assert high_power["score"]<score["score"]
    assert lab._score_arm({**p,"course_20mm_attempt_count":0},"balanced")["eligible"] is False
    assert lab._score_arm({k:v for k,v in p.items() if k!="screen_contact_slope_survival"},
                          "balanced")["eligible"] is False


def test_auto_promotion_is_locked_to_frozen_scoreboard(tmp_path):
    root=tmp_path/"lab";root.mkdir()
    lab._save(root/"round1_scoreboard.json",{"round":"round1","results":[]})
    board={"results":[
        {"arm":"step_focus","eligible":True,"score":.7},
        {"arm":"balanced","eligible":True,"score":.8},
        {"arm":"efficient","eligible":False,"score":.99},
        {"arm":"slope_recovery","eligible":True,"score":.4}]}
    select=lab._choose(root,"round1","round2",board=board,
                       previous=list(lab.ARMS))
    assert select==["balanced","step_focus"]
    assert lab._choose(root,"round1","round2",board=board,
                       previous=list(lab.ARMS))==select
    with pytest.raises(RuntimeError,match="locked"):
        lab._choose(root,"round1","round2",board={
            "results":[{"arm":"efficient","eligible":True,"score":.9},
                       {"arm":"balanced","eligible":True,"score":.8}]},
            previous=list(lab.ARMS))
    with pytest.raises(RuntimeError,match="eligible"):
        lab._choose(root,"round1","round2",board={
            "results":[{"arm":"balanced","eligible":True,"score":.8}]},
            previous=list(lab.ARMS))


def test_public_cli_routes_to_v5020_plan_without_writing(tmp_path,capsys):
    from emu.cli import main
    root=tmp_path/"cli"
    assert main(["campaign5020","--root",str(root)])==0
    p=json.loads(capsys.readouterr().out)
    assert p["total_environment_steps_all_arms"]==380_000_000
    assert not root.exists()


def test_report_with_null_foundation_parent_is_safe(tmp_path):
    """The original 20M foundation crashed in report writing after a valid save."""
    from emu import results
    root=tmp_path/"lab"
    foundation=root/"foundation20"
    foundation.mkdir(parents=True)
    (root/"lab_context.json").write_text(json.dumps({
        "seed":733,"envs":4096,"git_commit":"old-source"}))
    (foundation/"lab_spec.json").write_text(json.dumps({
        "stage":"foundation","parent":None,"target":20_000_000}))
    (foundation/"training_config.json").write_text(json.dumps({
        "steps":20_000_000,"domain_terrain_radius":.16}))
    (foundation/"run_meta.json").write_text(json.dumps({
        "source_provenance":{"git_commit":"old-source"}}))
    assert results.collect(root)["runs"]["foundation20"]["parent"]=={}
    written=results.write_report(root)
    assert {x.name for x in written}=={
        "results.md","results_data.json","results_bundle.zip"}
    assert "Parent SHA-256: `unavailable`" in (root/"results.md").read_text()


def test_312_input_parent_handoff_never_falls_back_to_legacy_schema(tmp_path,monkeypatch):
    """The actual v5020 fork formerly crashed just printing '20M -> 80M'."""
    from emu import campaign5020 as lab
    root=tmp_path/"lab"
    parent=root/"foundation20"/"checkpoints"/"final.pkl"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"validated 312 actor fixture")
    calls=[]
    def strict_step(path,*,expected_observation_schema,expected_sensor_contract):
        assert Path(path)==parent
        assert expected_observation_schema==OBS_SCHEMA_PREVIEW
        assert expected_sensor_contract is not None
        calls.append("schema9-parent")
        return 20_054_016
    def done(run,target,envs,cfg):
        calls.append("find-completed")
        if "trained" in calls:
            return run/"checkpoints"/"final.pkl",None
        return None,None
    def trained(cfg,*,provenance):
        assert cfg.resume==str(parent)
        assert cfg.observation_schema==OBS_SCHEMA_PREVIEW
        calls.append("trained")
    monkeypatch.setattr(lab,"checkpoint_step",strict_step)
    monkeypatch.setattr(lab,"_completed",done)
    monkeypatch.setattr(lab,"run_training",trained)
    monkeypatch.setattr(lab,"verify_existing_phase_config",lambda *a:None)
    cfg=lab.config(root/"round1"/"balanced",stage="round1",
                   arm="balanced",seed=733,envs=4096,device="cuda")
    ck=lab._stage_run(root,stage="round1",arm="balanced",parent=parent,
       seed=733,envs=4096,device="cuda",
       provenance={"commit":"fixed-source","available":True,"dirty":False})
    assert ck==root/"round1"/"balanced"/"checkpoints"/"final.pkl"
    assert "trained" in calls
    assert calls.count("schema9-parent")>=2
    spec=json.loads((root/"round1"/"balanced"/"lab_spec.json").read_text())
    assert spec["parent"]["step"]==20_054_016
    assert spec["config"]["observation_schema"]==OBS_SCHEMA_PREVIEW


def test_full_lab_dag_promotes_two_then_one_without_trainer_side_effects(tmp_path,monkeypatch):
    """Exercise orchestrator after foundation, through two scoreboards and winner."""
    from emu import campaign5020 as lab
    root=tmp_path/"lab"
    visited=[]
    def phase(root,*,stage,arm,parent,seed,envs,device,provenance):
        if stage=="foundation":
            assert parent is None
        else:
            assert parent is not None
            assert parent.is_file()
        checkpoint=(root/"foundation20" if stage=="foundation"
                    else root/stage/arm)/"checkpoints"/"final.pkl"
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.write_text(f"{stage}/{arm or 'shared'}")
        visited.append((stage,arm))
        return checkpoint
    def score(root,stage,names,**kwargs):
        rows=[{"arm":arm,"eligible":True,"score":1.0-idx*.1}
              for idx,arm in enumerate(names)]
        board={"round":stage,"results":rows}
        lab._save(root/f"{stage}_scoreboard.json",board)
        return board
    monkeypatch.setattr(lab,"git_provenance",lambda:{
       "available":True,"commit":"fixed-source","dirty":False})
    monkeypatch.setattr(lab,"_stage_run",phase)
    monkeypatch.setattr(lab,"_screen",score)
    monkeypatch.setattr(lab,"_report",lambda root:None)
    lab.run(root,seed=733,envs=4096,screen_envs=32,horizon=300,
            device="cuda",stop_after="round3")
    assert [stage for stage,arm in visited].count("foundation")==1
    assert [stage for stage,arm in visited].count("round1")==4
    assert [stage for stage,arm in visited].count("round2")==2
    assert [stage for stage,arm in visited].count("round3")==1
    assert json.loads((root/"round3_selection.json").read_text())["selected"]==["balanced"]
