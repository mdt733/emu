"""Offline watchdog for live-school pilot orchestration and raw window coverage."""
from pathlib import Path
import json
from contextlib import nullcontext

import pytest

from scripts import live_skill_pilot as p


def test_pilot_unattended_resume(tmp_path,monkeypatch):
    from emu import checkpoint_io
    from emu.training_config import TrainingConfig
    root=tmp_path/"pilot"
    parent_root=tmp_path/"parents"
    complete=set()
    events=[]
    interrupt=[True]
    monkeypatch.setattr(p,"source",lambda:"commit")
    monkeypatch.setattr(p,"verify_model",lambda:None)
    monkeypatch.setattr(p,"root_checked",lambda root:root.resolve())
    monkeypatch.setattr(p,"locked",lambda root:nullcontext())
    monkeypatch.setattr(p,"stage_parent",lambda root,old,seed:root/"parents"/str(seed))
    monkeypatch.setattr(checkpoint_io,"find_resume",lambda run:
        run/"checkpoints"/"latest.pkl" if run.name in complete else None)
    monkeypatch.setattr(TrainingConfig,"load_research_ancestor",lambda path:None)
    def call(*args):
        if args[1]=="scripts.smoke_gpu_carpet":
            assert "--live-recovery-smoke" in args and "--live-uphill-smoke" in args
            events.append("smoke")
            return
        n=Path(args[args.index("--run-dir")+1]).name
        mode=args[args.index("--resume")+1]
        if "--check-resume-only" in args:
            events.append(("check",n,mode))
            return
        events.append(("train",n,mode))
        if n=="R-s9501" and interrupt[0]:
            interrupt[0]=False
            complete.add(n)
            raise RuntimeError("interrupted")
        complete.add(n)
    monkeypatch.setattr(p,"call",call)
    monkeypatch.setattr(p,"endpoint",lambda folder,policy,cfg,expected,plan:
        events.append(("endpoint",expected["run_name"])))
    monkeypatch.setattr(p,"verify_final",lambda root,arm,seed,commit,plan:
        (root/"evidence"/f"{arm}-s{seed}",None,None))
    monkeypatch.setattr(p,"report",lambda root,plan:events.append("archive") or root/"results.zip")
    with pytest.raises(RuntimeError,match="interrupted"):
        p.go(root,parent_root)
    assert events.count("smoke")==1 and "archive" not in events
    p.go(root,parent_root)
    assert events.count("smoke")==1 and events[-1]=="archive"
    train=[x for x in events if isinstance(x,tuple) and x[0]=="train"]
    assert [x[1] for x in train[-6:]]==[
        "U-s9501","R-s9501","H-s9501","H-s9502","R-s9502","U-s9502"]
    assert train[-6][2]==train[-5][2]=="auto"
    assert train[-4][2].endswith("/checkpoints/parent.pkl")
    with pytest.raises(RuntimeError,match="registration differs"):
        p.go(root,tmp_path/"wrong-parent")


def test_live_coverage_preserves_torn_tail_and_separates_epochs(tmp_path):
    d=tmp_path/"live_skills"
    d.mkdir()
    def window(start,end,epoch):
        return dict(source_commit="commit",worlds=p.WORLDS,
            start_step=start,end_step=end,world_frames=end-start,policy_epoch=epoch,
            cells={"reference_recovery/mission/0":dict(attempts=41,passes=31),
                   "reference_uphill/mission/5":dict(attempts=42,passes=8)},
            practice_frames_by_case=[100]*6,uphill_practice_frames_by_case=[200]*6)
    start=p.PARENT_STEP
    mid=start+131072
    first=d/"segment-a.jsonl"
    first.write_text(json.dumps(window(start,mid,0))+'\n{"torn":')
    second=d/"segment-b.jsonl"
    second.write_text(json.dumps(window(start,mid,0))+"\n"+
                      json.dumps(window(mid,p.TARGET,1))+"\n")
    r=p.live_coverage(tmp_path,"commit")
    assert len(r["segments"])==3 and r["unparsed_interrupted_tails"]==[first.name]
    assert r["segments"][-1]["reference_uphill_attempts"][5]==42
    assert first.read_text().endswith('{"torn":')
    second.write_text(second.read_text()+"not json\n")
    with pytest.raises(json.JSONDecodeError):
        p.live_coverage(tmp_path,"commit")


def test_cliff_plan_equal_budget_and_distinct_result(tmp_path):
    q=p.cliff_plan()
    assert q.profile=="cliff" and q.updates==100
    assert q.target==133169152
    assert tuple(dict(q.arms))==("U","H")
    assert q.spec(tmp_path)["total_additional_steps"]==52428800
    assert q.exam_seed!=p.DEFAULT.exam_seed
    assert q.result!=p.DEFAULT.result
    for flag in ("--skill-uphill-reference-share","--skill-uphill-practice-share"):
        assert q.common[q.common.index(flag)+1]==".25"
    assert q.common[q.common.index("--skill-uphill-profile")+1]=="cliff"
    assert p.DEFAULT.uphill_share==.15 and p.DEFAULT.target==p.TARGET


def test_engagement_plan_preserves_matched_control_and_old_profile():
    q=p.engagement_plan()
    assert q.profile=="engagement" and q.updates==160
    assert q.target==141033472
    assert q.spec(Path("/tmp/parent"))["total_additional_steps"]==83886080
    assert dict(q.arms)=={"U":(),"H":("--skill-adaptive-uphill",)}
    assert q.exam_seed==9863 and q.result=="results-live-skill-engagement.zip"
    assert p.cliff_plan().profile=="cliff" and p.cliff_plan().updates==100


def test_freedom_plan_matches_engagement_and_isolates_treatment():
    q=p.freedom_plan()
    assert q.profile=="engagement" and q.updates==160
    assert q.common==p.engagement_plan().common
    assert q.target==141033472
    assert q.spec(Path("/tmp/parent"))["total_additional_steps"]==125829120
    assert dict(q.arms)=={"U":(),"H":("--skill-adaptive-uphill",),
        "F":("--skill-adaptive-uphill","--skill-novice-fall-relief")}
    assert q.exam_seed==9864 and q.result=="results-live-skill-freedom.zip"
