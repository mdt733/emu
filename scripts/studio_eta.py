"""Read-only wall-clock forecasts, independent of the measured progress bar.

Durations are estimates, not evidence of finished work. Legacy pinned jobs
remain readable. New runner timing notices record successful wall-clock
stages, including process initialization, compilation and receipt writing.
"""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from statistics import median
import math
import re
import time

# Warm recent KS150 archives, widened for cold compilation / host overhead.
# These are priors, not measured timings for the current job.
PRIORS={
    "setup":(30.,10.,120.), "acceptance":(150.,65.,360.),
    "startup":(75.,10.,180.), "pulse":(50.,35.,180.),
    "matrix":(30.,18.,120.), "seal":(8.,2.,30.),
    "package":(6.,2.,30.), "rate":(39700.,18000.,43000.),
}
NOTICE=re.compile(r"^STUDIO TIME (\d+(?:\.\d+)?) (setup|acceptance|train|pulse|matrix|seal|package) ([\w-]+) (start|end)$",re.M)

@contextmanager
def timed_stage(kind,name):
    print(f"STUDIO TIME {time.time():.3f} {kind} {name} start",flush=True)
    yield
    print(f"STUDIO TIME {time.time():.3f} {kind} {name} end",flush=True)

def valid(x):
    return isinstance(x,(int,float)) and math.isfinite(x) and x>0

def stat_time(path):
    try: return path.stat().st_mtime
    except OSError: return None

def read_timings(log):
    completed={}
    starts={}
    for line in log.splitlines():
        if line.startswith("--- "):
            starts.clear()
        match=NOTICE.fullmatch(line)
        if not match: continue
        stamp,kind,name,event=match.groups()
        key=(kind,name); stamp=float(stamp)
        if event=="start": starts[key]=stamp
        elif key in starts:
            seconds=stamp-starts.pop(key)
            if 0<seconds<86400:
                completed.setdefault(kind,[]).append(seconds)
    return completed,starts

@lru_cache(maxsize=32)
def log_samples(path,mtime,size):
    try:
        with open(path,"rb") as f:
            f.seek(max(0,size-2_000_000))
            log=f.read().decode(errors="replace")
    except OSError: return {}
    measured,_=read_timings(log)
    # Legacy timings cover compute only; allow explicit host overhead.
    if not measured.get("pulse"):
        measured["pulse"]=[float(x)+8 for x in re.findall(r"\(([\d.]+)s MJX call\)",log)]
    if not measured.get("matrix"):
        measured["matrix"]=[512*350/float(x.replace(",",""))+8
            for x in re.findall(r"v5053 complete: 400 independent trials, ([\d,]+) world-steps/s",log)
            if float(x.replace(",",""))>0]
    measured["startup"]=[float(x)+5 for x in re.findall(r"compile\+first_update=([\d.]+)s",log)]
    return measured

def history_samples(state,focus):
    result={}
    # Only local KS150 Study Studio logs; not unrelated jobs or network data.
    if focus["manifest"].get("profile")!="h40-spring-only" or focus["manifest"].get("worlds")!=4096:
        return result
    try:
        paths=sorted(Path(state).glob("ks150-live-skill-*.log"),
                     key=lambda p:p.stat().st_mtime,reverse=True)[:8]
        for p in reversed(paths):
            s=p.stat()
            for k,v in log_samples(str(p),s.st_mtime_ns,s.st_size).items():
                result.setdefault(k,[]).extend(x for x in v if valid(x))
    except OSError: pass
    return result

def recent_rates(history):
    rates=[]
    for a,b in zip(history,history[1:]):
        sa,sb=a.get("step"),b.get("step")
        ta,tb=a.get("elapsed_s"),b.get("elapsed_s")
        if all(valid(x) for x in (sa,sb,ta,tb)) and sb>sa and tb>ta:
            rate=(sb-sa)/(tb-ta)
            if 100<=rate<=1_000_000: rates.append(rate)
        else: rates=[] # do not bridge a resumed/rolled-back timeline
    return rates[-6:]

def estimate(kind,samples):
    v=[x for x in samples if valid(x)][-12:]
    mid,lo,hi=PRIORS[kind]
    if not v: return (mid,lo,hi)
    mid=median(v)
    # Empirical planning envelope, not a confidence interval. Preserve
    # observed slow/cold samples; sparse samples retain wide bounds.
    return (mid,min(v)*(.8 if len(v)>=3 else .55),
            max(v)*(1.25 if len(v)>=3 else 1.8))

def subtract(triple,elapsed):
    if elapsed is None: return triple
    mid,lo,hi=triple
    # Never claim an unfinished stage has zero time left. An overrun
    # expands the range instead of indefinitely counting down to zero.
    if elapsed>=mid:
        return (max(5.,mid*.3),2.,max(hi-elapsed,mid,elapsed*.35))
    return (max(2.,mid-elapsed),max(1.,lo-elapsed),max(5.,hi-elapsed))

def forecast(focus,students,status,log,state,now=None):
    now=time.time() if now is None else now
    m=focus["manifest"]; root=Path(focus["root"])
    phase=status["phase"]; active=status["active"]
    if phase=="Complete":
        return dict(remaining_seconds=0,low_seconds=0,high_seconds=0,
                    finish_unix=None,current_seconds=0,rate=None,components=[],timeline=[],
                    note="Results archive complete",sample_counts={},stale=False)
    if not active: return None
    # The phase plan below is specific to this registered runner family.
    if m.get("module")!="scripts.live_skill_pilot": return None
    samples=history_samples(state,focus)
    _,starts=read_timings(log)
    parent,target=m["parent_step"],m["target_step"]
    seeds=sorted({s.rsplit("-s",1)[-1] for s in m["students"]})
    by_name={s["name"]:s for s in students}
    all_rates=[]
    for s in students:
        r=recent_rates(s.get("history",[]))
        if r: all_rates.append(median(r))
    current=status.get("student")
    current_row=by_name.get(current,{})
    live_rates=recent_rates(current_row.get("history",[]))
    # Wall-clock interval throughput includes host/observer/checkpoint
    # overhead. Prefer current student, then other students on this run.
    rate_samples=live_rates or all_rates
    if not rate_samples:
        r=current_row.get("last",{}).get("train_steps_per_sec")
        if valid(r): rate_samples=[r*.94]
    rate,rate_low,rate_high=estimate("rate",rate_samples)
    estimates={k:estimate(k,samples.get(k,[])) for k in PRIORS if k!="rate"}
    # Use the measured stage events where available; for legacy jobs
    # receipt mtimes establish a rough stage start, not an exact timestamp.
    components=[]; timeline=[]; elapsed_unknown=False; overrun=False
    def add(kind,name,triple=None,started=None,is_current=False):
        nonlocal elapsed_unknown,overrun
        t=estimates[kind] if triple is None else triple
        start=starts.get((kind,name),started) if is_current else None
        if is_current and valid(start) and start<=now:
            age=now-start
            overrun=age>t[2]
            t=subtract(t,age)
        elif is_current and kind!="training":
            elapsed_unknown=True
        components.append(dict(kind=kind,name=name,seconds=t[0],low=t[1],high=t[2],current=is_current))
    def elapsed_stage(kind,name,default):
        return starts.get((kind,name),default)
    def exams(name,previous,first=False):
        folder=root/"evidence"/name
        if (folder/"endpoint.json").is_file(): return
        pulse=(folder/"pulse.json").is_file()
        matrix=(folder/"matrix.json").is_file()
        if not pulse:
            add("pulse",name,started=previous,is_current=first)
        if not matrix:
            add("matrix",name,started=stat_time(folder/"pulse.json"),
                is_current=first and pulse)
        add("seal",name,started=stat_time(folder/"matrix.json"),
            is_current=first and pulse and matrix)
    if phase=="Preflight setup":
        add("setup","study",is_current=True)
    if phase in ("Preflight setup","GPU acceptance"):
        t=estimates["acceptance"]
        if phase=="GPU acceptance" and status["done"]>0:
            # The first update includes compilation. Once it has completed,
            # only remaining steady updates and final checks are predicted.
            n=max(0,status["total"]-status["done"])
            steps=n*m["worlds"]*32
            t=(steps/rate+15,steps/rate_high+5,steps/rate_low+60)
            add("acceptance","study",t)
            components[-1]["current"]=True
        else: add("acceptance","study",is_current=phase=="GPU acceptance")
    previous=stat_time(root/"acceptance"/"combined.json")
    for seed in seeds:
        name="parent-s"+seed
        seal=root/"evidence"/name/"endpoint.json"
        first=phase=="Parent examinations" and not components and not seal.is_file()
        exams(name,previous,first)
        previous=stat_time(seal) or previous
    for name in m["students"]:
        row=by_name.get(name,{})
        folder=root/"runs"/name
        seal=root/"evidence"/name/"endpoint.json"
        if row.get("endpoint") or seal.is_file():
            previous=stat_time(seal) or previous
            timeline.append(dict(name=name,done=True,seconds=0))
            continue
        step=status["student_steps"].get(name,parent)
        is_current=phase=="Student training" and current==name
        if not row.get("final_checkpoint"):
            remaining=max(0,target-step)
            if step==parent or (is_current and not row.get("history") and not live_rates):
                # First compile or a resumed process without fresh evidence.
                t=estimates["startup"]
                start=elapsed_stage("train",name,previous)
                add("startup",name,t,started=start,is_current=is_current)
            if remaining:
                t=(remaining/rate,remaining/rate_high,remaining/rate_low)
                add("training",name,t)
                components[-1]["current"]=is_current and not any(c["current"] for c in components[:-1])
        exams(name,stat_time(folder/"checkpoints"/"final.pkl"),
              phase=="Student examinations" and name==current)
        timeline.append(dict(name=name,done=False,seconds=sum(c["seconds"] for c in components)))
        previous=stat_time(seal) or previous
    add("package","study",is_current=phase=="Packaging results")
    if phase=="Packaging results" and len(components)>1:
        components=[components[-1]]
    total=[sum(c[k] for c in components) for k in ("seconds","low","high")]
    first=next((c for c in components if c["current"]),None)
    stale=bool(status.get("log_age_seconds") is not None and
               status["log_age_seconds"]>max(180,estimates["startup"][2]) and phase=="Student training")
    note=("Estimate overdue; stage completion not confirmed" if overrun else
          "No recent console progress; finish estimate provisional" if stale else
          "Legacy stage start unknown; full remaining-stage budget used" if elapsed_unknown else
          "Wall-clock forecast; range reflects observed variability and cold starts")
    return dict(remaining_seconds=total[0],low_seconds=total[1],high_seconds=total[2],
                finish_unix=now+total[0],current_seconds=first["seconds"] if first else None,
                rate=rate,rate_source="recent wall-clock intervals" if live_rates else
                     "other current-study intervals" if all_rates else "PPO rate adjusted for host overhead" if rate_samples else "historical prior",
                components=components,timeline=timeline,note=note,
                sample_counts={**{k:len(v) for k,v in samples.items()},"rate":len(rate_samples)},
                stale=stale or overrun)
