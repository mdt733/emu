"""Read-only progress and evidence snapshots for EMU Studio.

All numbers are based on completed updates or persisted receipts. Older
pinned runners do not announce every acceptance update: show a lower bound,
never guess a completion percentage from elapsed time.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
import zipfile
from pathlib import Path

MAX_FILE=16*1024*1024
MAX_TOTAL=64*1024*1024
MAX_LOG=8*1024*1024


def progress(focus,students,log,active,log_age=None):
    m=focus["manifest"]
    root=Path(focus["root"])
    parent,target=m["parent_step"],m["target_step"]
    per_student=target-parent
    total=per_student*len(m["students"])
    completed=0
    steps={}
    for s in students:
        step=max(parent,s.get("last",{}).get("step") or parent,
                 (s.get("lesson") or {}).get("end_step") or parent)
        if s.get("final_checkpoint") or s.get("endpoint"):
            step=target
        step=max(parent,min(target,int(step)))
        steps[s["name"]]=step
        completed+=step-parent
    sealed=sum(bool(s.get("endpoint")) for s in students)
    # Reset preflight counts across retries; historical log records are
    # not evidence of completed updates in the current process.
    log=log.rsplit("\n--- ",1)[-1]
    smoke_done=(root/"acceptance"/"combined.json").is_file()
    current=None
    lower=False
    done=0
    count=0
    unit=""
    if (root/m["result"]).is_file():
        phase,detail="Complete","Registered results archive exists"
        done,count,unit=1,1,"archive"
    elif not smoke_done:
        match=re.findall(r"PRECHECK: both mature lesson families, (\d+) updates",log)
        if match or "real batched PPO training update PASS" in log:
            phase="GPU acceptance"
            count=int(match[-1]) if match else 12
            notices=re.findall(r"STUDIO PREFLIGHT (\d+)/(\d+)",log)
            if notices:
                done=min(count,int(notices[-1][0]))
            elif "steady state PPO PASS:" in log:
                steady=re.findall(r"steady state PPO PASS: (\d+)",log)
                done=min(count,1+int(steady[-1]))
            elif "real batched PPO training update PASS" in log:
                done=1
                lower=True
            unit="updates"
            detail=("Older runner reports only the first and final acceptance updates."
                    if lower else "Real PPO acceptance updates; research students have not started.")
        else:
            phase,detail="Preflight setup","Authenticating parents and checking fork configurations"
            count=len(m["students"])
            done=min(count,log.count("RESUME VERIFICATION PASS:"))
            unit="fork checks"
            lower=bool("[earlier output omitted]" in log)
    else:
        seeds=sorted({s.rsplit("-s",1)[-1] for s in m["students"]})
        baselines=sum((root/"evidence"/("parent-s"+s)/"endpoint.json").is_file() for s in seeds)
        if baselines<len(seeds):
            phase,detail="Parent examinations","Frozen parent pulse and staircase exams"
            done,count,unit=baselines,len(seeds),"parent endpoints"
        elif sealed<len(m["students"]):
            matches=re.findall(r"^STUDENT ([a-zA-Z0-9_-]+):",log,re.M)
            pending=[s for s in students if not s.get("endpoint")]
            chosen=next((s for s in pending if matches and s["name"]==matches[-1]),None)
            chosen=chosen or (pending[0] if pending else None)
            current=chosen["name"] if chosen else None
            if chosen and chosen.get("final_checkpoint"):
                phase,detail="Student examinations","Training finished; frozen endpoint exams"
                evidence=root/"evidence"/current
                done=sum((evidence/n).is_file() for n in ("pulse.json","matrix.json"))
                count,unit=2,"exam panels"
            else:
                phase,detail="Student training","Completed world steps from live metrics and evidence"
                done=max(0,steps.get(current,parent)-parent)
                count,unit=per_student,"world steps"
        else:
            phase,detail="Packaging results","All endpoints sealed; assembling the evidence ZIP"
    if not active and phase!="Complete":
        detail="Last observed phase — "+detail
    return dict(phase=phase,detail=detail,student=current,done=done,total=count,unit=unit,
                lower_bound=lower,training_done=completed,training_total=total,
                sealed=sealed,students_total=len(m["students"]),log_age_seconds=log_age,
                active=active)


def partial_archive(focus,log_path,status):
    """Temporary diagnostic ZIP. Never includes checkpoints or actor binaries.

    Reads a bounded snapshot of each allowed file, reports omissions, and
    keeps unfinished JSONL tails intact. It is not a final sealed archive.
    """
    root=Path(focus["root"]).resolve()
    m=focus["manifest"]
    out=tempfile.TemporaryFile(mode="w+b")
    omitted=[]
    used=0
    try:
        with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            def copy(path,name,*,limit=MAX_FILE,last=False,scope=root):
                nonlocal used
                try:
                    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(scope):
                        return
                    with path.open("rb") as f:
                        size=path.stat().st_size
                        budget=min(limit,MAX_TOTAL-used)
                        if budget<=0 or (size>budget and not last):
                            omitted.append(dict(file=name,bytes=size,reason="size limit"))
                            return
                        if size>budget:
                            f.seek(size-budget)
                            omitted.append(dict(file=name,bytes=size,reason="earlier log truncated"))
                        data=f.read(min(size,budget))
                    z.writestr(name,data)
                    used+=len(data)
                except OSError as e:
                    omitted.append(dict(file=name,reason=str(e)))
            if log_path is not None:
                copy(log_path,"studio.log",limit=MAX_LOG,last=True,scope=log_path.parent.resolve())
            copy(root/"study.json","study.json")
            for p in sorted((root/"acceptance").glob("*.json")):
                copy(p,str(p.relative_to(root)))
            for name in m["students"]:
                folder=root/"runs"/name
                for n in ("training_config.json","run_meta.json","metrics.csv","fixed_eval.csv",
                          "performance.json","results_data.json"):
                    p=folder/n;copy(p,str(p.relative_to(root)))
                for p in sorted((folder/"live_skills").glob("*.jsonl")):
                    copy(p,str(p.relative_to(root)))
            # Only receipt and raw exam/evidence files; never binary policies.
            for p in sorted((root/"evidence").rglob("*")):
                if p.is_file() and p.suffix in (".json",".jsonl",".csv",".zip"):
                    copy(p,str(p.relative_to(root)))
            z.writestr("snapshot.json",json.dumps(dict(schema="emu.studio.partial.v1",
                created_unix=time.time(),complete=False,source_commit=focus["commit"],
                manifest=m,progress=status,omitted=omitted,
                note="Partial diagnostic snapshot; active files can change. Not a sealed final result."),
                indent=2,allow_nan=False))
        size=out.tell()
        out.seek(0)
        return out,size
    except BaseException:
        out.close()
        raise
