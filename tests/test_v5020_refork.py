"""CPU-only acceptance of v5020's winner-refork campaign.

No JAX compile and no GPU training. Real screening remains a separate MJX gate.
"""
import csv
import json
from pathlib import Path

import pytest

from emu import campaign_refork as c
from emu.contracts import OBS_SCHEMA_PREVIEW
from emu.training_config import training_sensor_contract


def _evidence(core=.9, hard=.4, slope=.7, crossing=.2):
    d = {}
    for panel, survival in (
        ("core", core), ("resilience100", hard), ("robust70", hard),
        ("growth140", .4), ("growthcombo140", .4),
        ("contact_slope", slope), ("contact_harder", slope)
    ):
        d[f"screen_{panel}_survival"] = survival
    for panel in ("robust70", "growth140", "contact_slope"):
        for axis in ("vx", "vy", "yaw"):
            d[f"screen_{panel}_{axis}_mae"] = .08
    for panel in ("core", "robust70", "contact_slope"):
        d[f"screen_{panel}_height_shortfall_m"] = .005
    for panel in ("robust70", "contact_slope"):
        d[f"screen_{panel}_step_evidence"] = .50
    for name in ("robust", "contact_slope"):
        d[f"power_{name}_shaft_abs_power_w"] = 3.
        d[f"power_{name}_over_rated_frac"] = .12
    d.update(course_5mm_crossing_rate=crossing,course_5mm_attempt_count=32.)
    return d


def test_plan_is_read_only_and_three_walking_generations_plus_gated_terrain(tmp_path,capsys):
    root=tmp_path/"empty"
    assert c.main(["--root",str(root)]) == 0
    result=json.loads(capsys.readouterr().out)
    assert [x["stage"] for x in result["generations"]]==list(c.STAGES)
    assert result["generations"][0]["target_steps_per_actor"]==40_000_000
    assert result["generations"][1]["target_steps_per_actor"]==60_000_000
    assert result["generations"][2]["target_steps_per_actor"]==80_000_000
    assert all(len(x["arms"])==4 for x in result["generations"])
    assert [x["physical_step_training"] for x in result["generations"]]==[False,False,False,True]
    assert not root.exists()
    assert c.main(["--root",str(root),"--status"])==0
    status=json.loads(capsys.readouterr().out)
    assert all(status[s]["selection"] is None for s in c.STAGES)
    assert not root.exists()


def test_contract_and_mild_main_style_stage_progression(tmp_path):
    root=tmp_path/"refork"
    configurations=[]
    for stage in c.STAGES:
        for arm in c.arms(stage):
            cfg=c.config(root/stage/arm/"finish",stage=stage,arm=arm,
                         target=c.TARGET[stage],seed=733,envs=4096,device="cuda")
            configurations.append(cfg)
            assert cfg.observation_schema==OBS_SCHEMA_PREVIEW
            assert not cfg.forward_preview_masked
            assert cfg.attitude_profile=="v5015"
            assert cfg.domain_terrain_mode=="contact"
            assert cfg.domain_randomization
            assert cfg.domain_terrain_radius<=.20
            assert cfg.open_frontier is False
            assert cfg.steps==c.TARGET[stage]
            assert cfg.allow_rollback
            if stage!="terrain100":
                assert cfg.terrain_height_safety_m==0.
                assert cfg.terrain_adaptive_flat_control is False
                assert cfg.terrain_random_placement is False
                assert cfg.pitch_command_max_deg<=12.
            else:
                assert cfg.terrain_adaptive_flat_control
                assert cfg.terrain_random_placement
                assert 0 <= cfg.terrain_height_safety_m <= .005
    first=training_sensor_contract(configurations[0])
    assert all(training_sensor_contract(x)==first for x in configurations)
    foundation_profile=c.config(root/"walk40"/"general"/"finish",stage="walk40",
                                arm="general",target=40_000_000,
                                seed=733,envs=4096,device="cuda")
    assert foundation_profile.push_max_force_n==.7
    assert foundation_profile.combo_mix_frac==0.
    assert foundation_profile.capability_field_mode=="shadow"
    assert c.config(root/"walk80"/"turns"/"finish",stage="walk80",
                    arm="turns",target=80_000_000,seed=733,envs=4096,
                    device="cuda").combo_mix_frac==.55


def test_probe_detects_repeated_100_percent_falls_and_spring_explosion(tmp_path):
    p=tmp_path/"probe"
    p.mkdir()
    with (p/"metrics.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=[
            "fall_rate","ep_len_mean","spring_potential_energy_j"])
        w.writeheader()
        w.writerow(dict(fall_rate=".15",ep_len_mean="580",
                        spring_potential_energy_j="0.7"))
        w.writerow(dict(fall_rate="1.",ep_len_mean="14",
                        spring_potential_energy_j="0.8"))
        w.writerow(dict(fall_rate="1.",ep_len_mean="15",
                        spring_potential_energy_j="31000000"))
        w.writerow(dict(fall_rate="1.",ep_len_mean="13",
                        spring_potential_energy_j="1.2"))
    x=c.probe_health(p)
    assert not x["eligible"]
    assert any("spring" in reason for reason in x["reasons"])
    assert any("fall rate" in reason for reason in x["reasons"])
    with (p/"metrics.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=[
            "fall_rate","ep_len_mean","spring_potential_energy_j"])
        w.writeheader()
        for _ in range(3):
            w.writerow(dict(fall_rate=".25",ep_len_mean="430",
                            spring_potential_energy_j=".8"))
    assert c.probe_health(p)["eligible"]


def test_heldout_selection_requires_real_survival_and_step_for_terrain():
    assert c._score(_evidence(core=.03,hard=.0,slope=0.),
                    stage="walk40",name="dead")["eligible"] is False
    assert c._score(_evidence(),stage="walk80",name="walker")["eligible"]
    assert c._score(_evidence(crossing=.0),
                    stage="terrain100",name="non_crosser")["eligible"] is False
    assert c._score(_evidence(crossing=.25),
                    stage="terrain100",name="crosser")["eligible"]
    assert not c._terrain_entry_ok({"winner":"a","rows":[{"arm":"a",
        "eligible":True,"core":.81,"hard":.01,"slope":.2,"gait":.4}]})


def test_full_three_generation_refork_uses_one_winner_each_time(tmp_path,monkeypatch):
    root=tmp_path/"root"
    foundation=tmp_path/"original"/"final.pkl"
    foundation.parent.mkdir()
    foundation.write_bytes(b"schema9-parent-mock")
    trace=[]
    def fake_phase(run,cfg,*,parent,provenance):
        assert parent.exists()
        ck=run/"checkpoints"/"final.pkl"
        ck.parent.mkdir(parents=True,exist_ok=True)
        ck.write_text(f"{run} <- {parent}")
        trace.append((run,parent))
        return ck
    def fake_step(path,cfg):
        return (20_054_016 if Path(path)==foundation else cfg.steps)
    def fake_exam(ck,*,stage,root,screen_envs,horizon,device):
        # Give the general policy a small advantage over specialists and
        # the old incumbent; do not learn from score during PPO.
        extra=.04 if "/general/" in str(ck) else 0.
        if "/turns/" in str(ck):
            extra=.02
        if "/step5/" in str(ck):
            extra=.08
        return _evidence(core=.84+extra,hard=.25+extra,
                         slope=.30+extra,crossing=.10+extra)
    monkeypatch.setattr(c,"git_provenance",lambda:{
        "available":True,"dirty":False,
        "commit":"test-pinned-source","branch":"feature/v5020-coldstart-terrain"})
    monkeypatch.setattr(c,"_foundation",lambda path,*,envs:{
        "path":str(path),"sha256":c._sha(path),"step":20_054_016})
    monkeypatch.setattr(c,"_validated_step",fake_step)
    monkeypatch.setattr(c,"_phase",fake_phase)
    monkeypatch.setattr(c,"probe_health",lambda run:{
        "eligible":True,"reasons":[],"fall_rate":.25,
        "mean_episode_frames":500,"max_logged_spring_j":.9})
    monkeypatch.setattr(c,"_exam",fake_exam)
    c.run(root,foundation=foundation,seed=733,envs=4096,
          screen_envs=32,horizon=300,device="cuda",stop_after="walk80")
    for stage in c.STAGES[:3]:
        selection=json.loads((root/stage/"selection.json").read_text())
        assert selection["winner"]=="general"
        assert len([x for x in trace if x[0].parts[-3]==stage])==8
        assert (root/stage/"scoreboard.json").exists()
    assert not (root/"terrain100").exists()
    parent60=json.loads((root/"walk60"/"general"/"probe"/"stage_spec.json").read_text())
    assert "/walk40/general/finish/checkpoints/final.pkl" in parent60["parent"]["path"]
    parent80=json.loads((root/"walk80"/"general"/"probe"/"stage_spec.json").read_text())
    assert "/walk60/general/finish/checkpoints/final.pkl" in parent80["parent"]["path"]
    # A second launch with the same pinned source/evidence reuses sealed scores.
    original_count=len(trace)
    c.run(root,foundation=foundation,seed=733,envs=4096,
          screen_envs=32,horizon=300,device="cuda",stop_after="walk80")
    assert len(trace)==original_count+24  # fake_phase is deliberately not idempotent


def test_collapse_rejects_all_and_keeps_the_parent(tmp_path,monkeypatch):
    root=tmp_path/"root"
    foundation=tmp_path/"original"/"final.pkl"
    foundation.parent.mkdir()
    foundation.write_bytes(b"parent")
    monkeypatch.setattr(c,"git_provenance",lambda:{
        "available":True,"dirty":False,"commit":"same",
        "branch":"feature/v5020-coldstart-terrain"})
    monkeypatch.setattr(c,"_foundation",lambda path,*,envs:{
        "path":str(path),"sha256":c._sha(path),"step":20_054_016})
    monkeypatch.setattr(c,"_validated_step",lambda path,cfg:20_054_016
                        if Path(path)==foundation else cfg.steps)
    def fake_phase(run,cfg,*,parent,provenance):
        ck=run/"checkpoints"/"final.pkl"
        ck.parent.mkdir(parents=True,exist_ok=True)
        ck.write_bytes(b"checkpoint")
        return ck
    monkeypatch.setattr(c,"_phase",fake_phase)
    monkeypatch.setattr(c,"probe_health",lambda run:{
        "eligible":False,"reasons":["100% falls"],"fall_rate":1.,
        "mean_episode_frames":15.,"max_logged_spring_j":20000.})
    monkeypatch.setattr(c,"_exam",lambda ck,**kwargs:_evidence())
    with pytest.raises(RuntimeError,match="no safely promotable winner"):
        c.run(root,foundation=foundation,seed=733,envs=4096,
              screen_envs=32,horizon=300,device="cuda",stop_after="walk80")
    assert not (root/"walk60").exists()
    board=json.loads((root/"walk40"/"scoreboard.json").read_text())
    assert board["winner"] is None
    assert not any((root/"walk40"/a/"finish").exists() for a in c.WALK_ARMS)
