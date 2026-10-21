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
             course_20mm_attempt_count=20.)
    score=lab._score_arm(p,"balanced")
    assert score["eligible"] and score["course_20mm_crossing"]==.25
    assert lab._score_arm({**p,"course_20mm_done_count":0},"balanced")["eligible"] is False
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
