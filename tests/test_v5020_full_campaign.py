"""Fresh 20M -> 5-way refork lifecycle acceptance, without costly GPU jobs."""
import json
import zipfile

import pytest

from emu import campaign_full5020 as full
from emu import results_refork as reports


def test_new_full_command_plans_without_touching_fs(tmp_path,capsys):
    from emu.cli import main
    root=tmp_path/"not_created"
    assert main(["campaign-full5020","--root",str(root)])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary["foundation_steps"]==20_000_000
    assert summary["stop_after"]=="walk80"
    assert not root.exists()


def test_full_study_hands_new_validated_foundation_to_five_way_refork(tmp_path,monkeypatch):
    root=tmp_path/"newcampaign"
    seen=[]
    pinned={"available":True,"dirty":False,
            "commit":"full-test-locked","branch":"feature/v5020-full-campaign"}
    monkeypatch.setattr(full,"git_provenance",lambda:pinned)
    monkeypatch.setattr(full,"preflight",lambda:seen.append("mjx_pass"))
    monkeypatch.setattr(full,"_report",lambda *_:seen.append("foundation_results"))
    def stage(run_root,*,stage,arm,parent,seed,envs,device,provenance):
        assert stage=="foundation" and arm is None and parent is None
        assert run_root==root
        assert provenance==pinned and envs==4096 and seed==733
        ck=run_root/"foundation20"/"checkpoints"/"final.pkl"
        ck.parent.mkdir(parents=True,exist_ok=True)
        ck.write_bytes(b"valid schema9 mock")
        seen.append("foundation_complete")
        return ck
    def refork(dst,*,foundation,seed,envs,screen_envs,horizon,device,stop_after):
        assert dst==root/"refork"
        assert foundation==root/"foundation20"/"checkpoints"/"final.pkl"
        assert (envs,screen_envs,horizon,stop_after)==(4096,32,300,"walk80")
        dst.mkdir()
        (dst/"refork_context.json").write_text(json.dumps({
            "source_commit":"full-test-locked","foundation":{"path":str(foundation),
            "sha256":"mock","step":20_054_016},"seed":733,
            "envs":4096,"screen_envs":32,"horizon":300}))
        seen.append("five_arm_generation")
    monkeypatch.setattr(full,"_stage_run",stage)
    monkeypatch.setattr(full,"refork_run",refork)
    full.run(root,envs=4096,seed=733,screen_envs=32,horizon=300,
             device="cuda",stop_after="walk80")
    assert seen==["mjx_pass","foundation_complete","foundation_results","five_arm_generation"]
    assert (root/"refork"/"results.md").exists()
    assert (root/"refork"/"results_bundle.zip").exists()
    assert (root/"lab_context.json").exists()
    # Source pin remains protected before another call to preflight/training.
    with pytest.raises(RuntimeError,match="source/seed/world count changed"):
        full.run(root,envs=512,seed=733,screen_envs=32,horizon=300,
                 device="cuda",stop_after="walk80")


def test_refork_report_reads_completed_and_interrupted_phases(tmp_path):
    root=tmp_path/"refork"
    root.mkdir()
    (root/"refork_context.json").write_text(json.dumps({
        "source_commit":"abc","foundation":{"path":"/safe/foundation.pkl",
        "sha256":"aaa","step":20_054_016},
        "seed":733,"envs":4096,"screen_envs":32,"horizon":300}))
    stage=root/"walk40"
    stage.mkdir()
    (stage/"scoreboard.json").write_text(json.dumps({
        "winner":"steps","reason":"measured superiority",
        "rows":[{"arm":"steps","eligible":True,"score":.82,
                 "core":.9,"hard":.4,"slope":.4,"graded_step_score":.5}]}))
    (stage/"selection.json").write_text(json.dumps({"winner":"steps"}))
    arm=stage/"steps"
    arm.mkdir()
    (arm/"probe_health.json").write_text(json.dumps({
        "eligible":True,"fall_rate":.1}))
    probe=arm/"probe"
    probe.mkdir()
    (probe/"metrics.csv").write_text(
        "step,reward_mean,fall_rate,ep_len_mean,servo_over_rated_frac\n"
        "2000000,1.0,.25,120,.13\n2200000,1.2,.1,500,.10\n")
    (probe/"stage_spec.json").write_text(json.dumps({"parent":{"sha256":"aaa"}}))
    # Deliberately no finish checkpoint. It must report an honest partial.
    report=reports.collect(root)
    assert report["stages"]["walk40"]["arms"]["steps"]["phases"]["probe"]["metrics_end"]["fall_rate"]==.1
    outputs=reports.write_refork_report(root)
    assert {p.name for p in outputs}=={
        "results.md","results_data.json","results_bundle.zip"}
    md=(root/"results.md").read_text()
    assert "Winner: **steps**" in md and "fall 0.100" in md
    with zipfile.ZipFile(root/"results_bundle.zip") as z:
        names=z.namelist()
        assert "walk40/steps/probe/metrics.csv" in names
        assert "walk40/selection.json" in names
        assert not any(x.endswith(".pkl") for x in names)


def test_examination_failure_is_propagated_without_starting_training(tmp_path,monkeypatch):
    monkeypatch.setattr(full,"git_provenance",lambda:{
        "available":True,"dirty":False,"commit":"fail-exam","branch":"feature/v5020-full-campaign"})
    def abort():
        raise RuntimeError("physical preflight failed")
    monkeypatch.setattr(full,"preflight",abort)
    monkeypatch.setattr(full,"_stage_run",
                        lambda *a,**kw:pytest.fail("training launched before preflight"))
    with pytest.raises(RuntimeError,match="physical preflight failed"):
        full.run(tmp_path/"fresh",envs=4096,seed=733,screen_envs=32,
                 horizon=300,device="cuda",stop_after="walk80")
    assert not (tmp_path/"fresh"/"foundation20").exists()
