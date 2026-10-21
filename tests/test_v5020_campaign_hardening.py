"""Preflight and full DAG tests with no expensive real-world campaign required."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from emu import campaign5020 as lab
from emu.contracts import OBS_SCHEMA_PREVIEW


def test_round1_handoff_never_uses_legacy_220_checkpoint_defaults(tmp_path,monkeypatch):
    root=tmp_path/"study"
    parent=root/"foundation20"/"checkpoints"/"final.pkl"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"validated placeholder for mocked checker")
    provenance={"available":True,"dirty":False,"commit":"fixed","branch":"feature/test"}
    calls=[]
    finished=False

    def validated_step(path,*,expected_observation_schema=None,expected_sensor_contract=None):
        assert expected_observation_schema==OBS_SCHEMA_PREVIEW
        assert expected_sensor_contract==lab.training_sensor_contract(
            lab.config(root/"round1"/"balanced",stage="round1",arm="balanced",
                       seed=733,envs=4096,device="cuda"))
        calls.append(Path(path))
        return 20_054_016

    def completed(run,target,envs,cfg):
        assert target==80_000_000 and envs==4096
        if finished:
            ck=run/"checkpoints"/"final.pkl"
            ck.parent.mkdir(parents=True,exist_ok=True)
            ck.write_bytes(b"mock final")
            return ck,None
        return None,None

    def run_training(cfg,*,provenance):
        nonlocal finished
        assert cfg.resume==str(parent)
        assert cfg.steps==80_000_000
        assert provenance["commit"]=="fixed"
        finished=True

    monkeypatch.setattr(lab,"checkpoint_step",validated_step)
    monkeypatch.setattr(lab,"_completed",completed)
    monkeypatch.setattr(lab,"run_training",run_training)
    monkeypatch.setattr(lab,"verify_existing_phase_config",lambda *_:None)
    ck=lab._stage_run(root,stage="round1",arm="balanced",parent=parent,
                      seed=733,envs=4096,device="cuda",provenance=provenance)
    assert ck.name=="final.pkl"
    assert calls==[parent,parent]
    spec=json.loads((root/"round1"/"balanced"/"lab_spec.json").read_text())
    assert spec["parent"]["step"]==20_054_016
    assert spec["parent"]["sha256"]==lab._sha(parent)


def test_all_seven_campaign_transitions_and_two_locked_selections(tmp_path,monkeypatch):
    root=tmp_path/"study"
    stage_calls=[]
    reports=[]
    provenance={"available":True,"dirty":False,"commit":"frozen-commit","branch":"feature/test"}
    monkeypatch.setattr(lab,"git_provenance",lambda:provenance)
    monkeypatch.setattr(lab,"source_repo_root",lambda:None)
    def simulated_stage(root,*,stage,arm,parent,seed,envs,device,provenance):
        assert seed==733 and envs==4096 and device=="cuda"
        if stage=="foundation":
            assert arm is None and parent is None
        else:
            assert parent is not None and parent.name=="final.pkl"
        stage_calls.append((stage,arm,parent))
        ck=root/("foundation20" if stage=="foundation" else f"{stage}/{arm}")/"checkpoints"/"final.pkl"
        ck.parent.mkdir(parents=True,exist_ok=True)
        ck.write_bytes(f"{stage}/{arm}".encode())
        return ck
    def simulated_screen(root,stage,names,**kwargs):
        assert stage in ("round1","round2","round3")
        # stable measured sorted order and eligibility; same four arms
        # remain the actual round1 candidates, 2 finalists then 1 winner.
        rows=[{"arm":name,"eligible":True,"score":dict(
            balanced=.8,step_focus=.7,slope_recovery=.3,efficient=.2)[name]}
            for name in names]
        board={"round":stage,"target":lab.ROUNDS[stage],
               "score_version":lab.VERSION,
               "screen_envs":32,"screen_horizon":300,
               "results":rows}
        lab._save(root/f"{stage}_scoreboard.json",board)
        (root/f"{stage}_scorecard.md").write_text("Mock exam report")
        return board
    monkeypatch.setattr(lab,"_stage_run",simulated_stage)
    monkeypatch.setattr(lab,"_screen",simulated_screen)
    monkeypatch.setattr(lab,"_report",lambda root:reports.append(root))
    lab.run(root,seed=733,envs=4096,screen_envs=32,horizon=300,
            device="cuda",stop_after="round3")
    assert [stage for stage,_,_ in stage_calls]==[
        "foundation",*(["round1"]*4),*(["round2"]*2),"round3"]
    assert [arm for stage,arm,_ in stage_calls if stage=="round2"]==["balanced","step_focus"]
    assert [arm for stage,arm,_ in stage_calls if stage=="round3"]==["balanced"]
    assert stage_calls[1][2]==root/"foundation20"/"checkpoints"/"final.pkl"
    assert stage_calls[5][2]==root/"round1"/"balanced"/"checkpoints"/"final.pkl"
    assert stage_calls[-1][2]==root/"round2"/"balanced"/"checkpoints"/"final.pkl"
    assert len(reports)==11
    assert (root/"round2_selection.json").exists()
    assert (root/"round3_selection.json").exists()

    # Fully completed campaign must be safely rerunnable with same source.
    stage_calls.clear()
    lab.run(root,seed=733,envs=4096,screen_envs=32,horizon=300,
            device="cuda",stop_after="round3")
    assert len(stage_calls)==8
    assert len(list(root.glob("*_selection.json")))==2
    with pytest.raises(RuntimeError,match="source/seed/world count changed"):
        lab.run(root,seed=733,envs=1024,screen_envs=32,horizon=300,
                device="cuda",stop_after="round3")


def test_cli_preflight_isolated_from_campaign_directory(tmp_path,monkeypatch,capsys):
    from emu.cli import main
    from emu import campaign5020_preflight
    root=tmp_path/"must_not_exist"
    seen=[]
    def dry(**kwargs):
        seen.append(kwargs)
        return {"status":"PASS","obs_dim":312}
    monkeypatch.setattr(campaign5020_preflight,"run_preflight",dry)
    assert main(["campaign5020","--root",str(root),
                 "--preflight","--device","cpu","--envs","4096"])==0
    assert json.loads(capsys.readouterr().out)=={"status":"PASS","obs_dim":312}
    assert seen==[{"seed":733,"device":"cpu","worlds":8,"horizon":6}]
    assert not root.exists()


def test_reporting_errors_do_not_abort_campaign(tmp_path,monkeypatch,capsys):
    from emu import results
    def broken(_):
        raise AttributeError("bad formatter")
    monkeypatch.setattr(results,"write_report",broken)
    lab._report(tmp_path)
    assert "WARNING: offline report generation failed" in capsys.readouterr().out
