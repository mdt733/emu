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
    d = {"physical_step_exact_height": True}
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
    for mm in c.EXAM_HEIGHTS_MM:
        d[f"course_{mm}mm_crossing_rate"]=crossing
        d[f"course_{mm}mm_attempt_count"]=32.
        d[f"course_{mm}mm_contact_rate"]=min(1., crossing+.25)
    return d


def test_plan_is_read_only_and_three_walking_generations_plus_gated_terrain(tmp_path,capsys):
    root=tmp_path/"empty"
    assert c.main(["--root",str(root)]) == 0
    result=json.loads(capsys.readouterr().out)
    assert [x["stage"] for x in result["generations"]]==list(c.STAGES)
    assert result["generations"][0]["target_steps_per_actor"]==40_000_000
    assert result["generations"][1]["target_steps_per_actor"]==60_000_000
    assert result["generations"][2]["target_steps_per_actor"]==80_000_000
    assert all(len(x["arms"])==5 for x in result["generations"])
    assert all("steps" in x["arms"] for x in result["generations"])
    assert [x["physical_step_training"] for x in result["generations"]]==[True]*4
    assert result["earned_step_height_ceiling_mm"]==150
    assert result["independent_step_exam_mm"]==[5,10,20,40,80,150]
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
            if arm == "steps":
                assert cfg.terrain_height_safety_m==.150
                assert cfg.terrain_height_initial_m==.003
                assert cfg.terrain_height_growth_m==.001
                assert cfg.terrain_random_placement is True
                assert cfg.domain_terrain_radius==0.  # real boxes on level floor
                assert cfg.terrain_height_exposure==.60
                assert cfg.terrain_progress_weight>0.
            else:
                assert cfg.terrain_height_safety_m==0.
                assert cfg.terrain_adaptive_flat_control is False
                assert cfg.terrain_random_placement is False
                assert cfg.terrain_progress_weight==0.
            if stage!="terrain100":
                assert cfg.pitch_command_max_deg<=12.
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



def _write_probe_metrics(path, samples):
    path.mkdir(exist_ok=True)
    columns=["fall_rate","ep_len_mean","spring_potential_energy_j",
             "servo_shaft_abs_power_w","episodes"]
    with (path/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for fall, frames, spring, watts, count in samples:
            writer.writerow(dict(zip(columns,(fall,frames,spring,watts,count))))


def test_probe_recovers_from_early_collapse_without_suppressing_warnings(tmp_path):
    sample=[(.99,20.,.8,5.,30.) for _ in range(5)]
    sample += [(.20,400.,.8,5.,80.) for _ in range(15)]
    _write_probe_metrics(tmp_path,sample)
    health=c.probe_health(tmp_path)
    assert health["eligible"],health["reasons"]
    assert health["rows_checked"]==20
    assert health["recent_window_rows"]==4
    assert health["severe_collapse_windows"]>=3
    assert health["warnings"]
    assert health["total_completed_episodes"]==5*30+15*80


def test_probe_fails_persistent_late_collapse_not_just_final_row(tmp_path):
    sample=[(.15,480.,.9,6.,100.) for _ in range(15)]
    sample += [(1.,15.,.9,6.,300.) for _ in range(5)]
    _write_probe_metrics(tmp_path,sample)
    health=c.probe_health(tmp_path)
    assert not health["eligible"]
    assert any("fall rate" in r for r in health["reasons"])
    assert any("episode length" in r for r in health["reasons"])


def test_probe_rejects_absurd_finite_power_or_energy_anywhere(tmp_path):
    samples=[(.18,450.,.8,8.,30.) for _ in range(14)]
    samples[1]=(.18,450.,.8,120000.,30.)
    _write_probe_metrics(tmp_path,samples)
    health=c.probe_health(tmp_path)
    assert not health["eligible"]
    assert any("shaft" in r for r in health["reasons"])
    samples[1]=(.18,450.,2500000.,8.,30.)
    _write_probe_metrics(tmp_path,samples)
    health=c.probe_health(tmp_path)
    assert not health["eligible"]
    assert any("spring" in r for r in health["reasons"])


def test_probe_invalid_episodes_and_nonfinite_data_are_not_good_news(tmp_path):
    samples=[(.2,450.,.8,4.,0.),(.2,450.,.8,4.,2.)]
    _write_probe_metrics(tmp_path,samples)
    with pytest.raises(ValueError,match="insufficient completed"):
        c.probe_health(tmp_path)
    samples[1]=(.2,450.,float("nan"),4.,80.)
    _write_probe_metrics(tmp_path,samples)
    with pytest.raises(ValueError,match="nonfinite spring"):
        c.probe_health(tmp_path)



def test_probe_episode_weighting_prevents_tiny_cohorts_hiding_major_falls(tmp_path):
    # One rollout ending 1000 episodes is stronger evidence than two
    # rollouts with one successful episode apiece.
    _write_probe_metrics(tmp_path, [
        (1.0,15.,.8,4.,1000.),
        (.0,500.,.8,4.,1.),
        (.0,500.,.8,4.,1.),
    ])
    health=c.probe_health(tmp_path)
    assert not health["eligible"]
    assert health["fall_rate"]> .99
    assert health["recent_completed_episodes"]==1002


def test_sparse_training_log_cannot_hide_a_single_world_physics_peak(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    columns=("fall_rate","ep_len_mean","spring_potential_energy_j",
             "servo_shaft_abs_power_w","spring_potential_energy_peak_j",
             "servo_shaft_abs_power_peak_w","episodes")
    healthy=[.20,450.,.8,4.,1.2,8.,64.]
    rows=[healthy.copy() for _ in range(16)]
    # Single world's extreme physical state is nearly invisible in the
    # mean, but appears in the per-world peak accumulated by training.py.
    rows[3][4]=2000000.
    rows[6][5]=100000.
    with (tmp_path/"metrics.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(columns)
        w.writerows(rows)
    result=c.probe_health(tmp_path)
    assert not result["eligible"]
    assert result["max_logged_spring_j"] == 2000000.
    assert result["max_logged_shaft_abs_power_w"] == 100000.
    assert any("spring" in reason for reason in result["reasons"])
    assert any("shaft" in reason for reason in result["reasons"])


def test_heldout_score_rewards_real_steps_without_bypassing_survival_gates():
    assert c._score(_evidence(core=.03,hard=.0,slope=0.),
                    stage="walk40",name="dead")["eligible"] is False
    assert c._score(_evidence(),stage="walk80",name="walker")["eligible"]
    flat=c._score(_evidence(crossing=.0),
                  stage="walk40",name="flat")
    step=c._score(_evidence(crossing=.25),
                  stage="walk40",name="step")
    assert flat["eligible"] and step["eligible"]
    assert step["score"]>flat["score"]
    # A brilliant obstacle score cannot rescue a policy that cannot walk.
    assert not c._score(_evidence(core=.03,hard=0.,slope=0.,crossing=1.),
                        stage="walk40",name="fragile")["eligible"]
    missing=_evidence()
    del missing["course_80mm_crossing_rate"]
    assert not c._score(missing,stage="walk40",name="incomplete")["eligible"]
    too_few=_evidence()
    too_few["course_20mm_attempt_count"]=1.
    assert not c._score(too_few,stage="walk40",name="no_trials")["eligible"]
    impossible=_evidence()
    impossible["course_20mm_contact_rate"]=0.
    assert not c._score(impossible,stage="walk40",name="impossible")["eligible"]
    mixed=_evidence()
    mixed["physical_step_exact_height"]=False
    assert not c._score(mixed,stage="walk40",name="mixed_heights")["eligible"]
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
        if "/steps/" in str(ck):
            extra=-.04
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
        assert len([x for x in trace if x[0].parts[-3]==stage])==10
        assert (root/stage/"scoreboard.json").exists()
    assert not (root/"terrain100").exists()
    # Mock phase deliberately doesn't write stage_spec: inspect actual calls.
    parent60=next(parent for run,parent in trace
                  if run==root/"walk60"/"general"/"probe")
    assert "/walk40/general/finish/checkpoints/final.pkl" in str(parent60)
    parent80=next(parent for run,parent in trace
                  if run==root/"walk80"/"general"/"probe")
    assert "/walk60/general/finish/checkpoints/final.pkl" in str(parent80)
    # A second launch with the same pinned source/evidence reuses sealed scores.
    original_count=len(trace)
    c.run(root,foundation=foundation,seed=733,envs=4096,
          screen_envs=32,horizon=300,device="cuda",stop_after="walk80")
    assert len(trace)==original_count+30  # fake_phase is deliberately not idempotent


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



def test_step_frontier_is_earned_continuously_not_stopped_at_exam_heights():
    from emu.terrain import next_height_frontier
    good=dict(terrain_edge_exposure=.25,terrain_edge_done_count=16.,
              terrain_edge_crossing=.60,terrain_edge_fall_rate=.12,
              terrain_edge_contact=.30)
    height=.003
    for _ in range(40):
        new,event=next_height_frontier(height,good,safety_m=.150,growth_m=.001)
        assert event=="expand_measured_crossing"
        assert new>height
        height=new
    assert height>.040  # 3mm -> over 40mm, no hardcoded 5mm stop
    at_max,event=next_height_frontier(.150,good,safety_m=.150,growth_m=.001)
    assert at_max==.150
    bad={**good,"terrain_edge_crossing":0.,"terrain_edge_fall_rate":.80}
    retreat,event=next_height_frontier(.050,bad,safety_m=.150,growth_m=.001)
    assert retreat<.050
    assert "retreat" in event


def test_first_verified_crossing_reward_is_additive_and_opt_in():
    import numpy as np
    from emu.rewards import terrain_progress_payment
    m={
        "terminal":np.array([0.,0.]),
        "terrain_forward_delta_m":np.array([.006,.006]),
        "terrain_height_birth_m":np.array([.020,.020]),
        "terrain_course_x":np.array([.25,.25]),
        "cmd_vx":np.array([.20,.20]),
        "cmd_vy":np.array([0.,0.]),
        "cmd_yaw":np.array([0.,0.]),
        "support":np.array([1.,1.]),
        "gravity_z":np.array([1.,1.]),
        "terrain_first_verified_crossing":np.array([1.,0.]),
    }
    w=.035
    observed=terrain_progress_payment(m,xp=np,weight=w)
    assert np.isclose(observed[1], w*.5)
    assert np.isclose(observed[0],w*.5+4*w)
    assert np.all(terrain_progress_payment(m,xp=np,weight=0.)==0.)
