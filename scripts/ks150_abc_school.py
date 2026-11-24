"""Source-pinned six-student A/B/C cold-school campaign.

plan is read-only; lock creates a new registration; smoke and advance
are explicit GPU operations. advance runs ONE common milestone and
never skips the registered C intervention decision. No champion writes.
Run from the project's venv: python -m scripts.ks150_abc_school ...
"""
from __future__ import annotations

import argparse
import csv
import io
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
MODEL = "21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
SCHEMA = "ks150_abc_cold_school_v1"
TRAIN_SEEDS = (9501, 9502)
DISCOVERY = (6901, 6902)
HOLDOUT = (7901, 7902)
STAGES = (90_000_000,100_000_000,110_000_000,120_000_000,140_000_000)
WORLDS = 2048
ARMS = {"A":(), "B":("--cross-specialty-stairs",),
        "C":("--cross-specialty-stairs","--mission-refreshers")}
RECIPE = (
    "--profile","sim2real","--envs","2048","--device","cuda",
    "--pitch-deg","8","--terrain-actor","--terrain-step-max-m",".055",
    "--unified-capabilities","--terrain-carpet","--carpet-prisms",
    "--carpet-transition-cap-deg","3","--academy-stream",
    "--stair-survival-curriculum","--terrain-lesson","coherent-careful",
    "--foundation-steps","20000000","--foundation-mix","main",
    "--growth-recipe","main-mature","--teaching-schedule","terrain-early",
    "--force-program","gentle","--athletic-rehearsal-share",".15",
    "--retention-share",".30","--academy-intro-steps","60000000",
    "--academy-start-steps","80000000","--academy-ramp-steps","20000000",
    "--academy-intro-share",".10","--academy-max-share",".28",
    "--academy-height-cap-m",".055","--academy-course-episode-frames","1200",
    "--fixed-eval-every-steps","20000000","--science-step-goal-weight",".12",
    "--science-step-velocity-relief",".55","--science-step-approach-m",".22",
    "--science-bilateral-support-bonus","6","--science-followthrough-bonus","7",
)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    """Exclusive atomic publication of a complete receipt."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False,
                                         encoding="utf-8") as f:
            temp=Path(f.name)
            json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def source():
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()
    if git("status","--porcelain"):
        raise RuntimeError("Need clean immutable checkout")
    return git("rev-parse","HEAD")


def boundary(target):
    from emu.checkpoint_io import update_boundary_at_or_after
    return update_boundary_at_or_after(target,WORLDS)


def env_for(digest="",bounded=False):
    env=os.environ.copy()
    for k in ("EMU_UNIFIED_ACCEPTANCE","EMU_MISSION_REFRESHER_SHA256"):
        env.pop(k,None)
    env.update(EMU_PHYSICAL_PROFILE="h40-spring-only",
               EMU_UNIFIED_CONTINUOUS="cold-growth-approved",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",PYTHONPATH=str(REPO))
    if bounded:
        env["EMU_UNIFIED_ACCEPTANCE"]="bounded-ppo-lifecycle"
    if digest:
        env["EMU_MISSION_REFRESHER_SHA256"]=digest
    return env


def call(*args,digest="",bounded=False):
    subprocess.run([sys.executable,*map(str,args)],cwd=REPO,
                   env=env_for(digest,bounded),check=True)


def student(root,arm,seed):
    if arm not in ARMS or seed not in TRAIN_SEEDS:
        raise ValueError("Unregistered student")
    return root/"runs"/f"{arm}-s{seed}"


def recipe(root,arm,seed,target):
    if target not in STAGES:
        raise ValueError("Unregistered milestone")
    return [*RECIPE,*ARMS[arm],"--run-dir",str(student(root,arm,seed)),
            "--seed",str(seed),"--steps",str(target)]


def evidence(root,arm,seed,target):
    return root/"evidence"/student(root,arm,seed).name/f"{target//1_000_000}m"


def spec(root,commit):
    return dict(schema=SCHEMA,root=str(root),source_commit=commit,
                model_sha256=MODEL,train_seeds=list(TRAIN_SEEDS),
                discovery_seeds=list(DISCOVERY),holdout_seeds=list(HOLDOUT),
                stages=list(STAGES),boundaries=[boundary(t) for t in STAGES],
                arms={a:list(v) for a,v in ARMS.items()},recipe=list(RECIPE),
                worlds=WORLDS,activation_stage=110_000_000,
                policy="explicit review; no automatic promotion")


def root_checked(path):
    root=Path(path).expanduser().resolve()
    if root==Path.home() or root==REPO or REPO in root.parents:
        raise ValueError("Campaign root must be outside Git, not HOME")
    return root


@contextmanager
def locked(root):
    with (root/"campaign.lock").open("a") as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Campaign already running") from exc
        yield


def verify_model():
    # The parent process must also use KS150 for actor loading and validation.
    from emu.contracts import ROBOT_XML
    from emu.physical_variants import selected_profile
    if selected_profile()!="h40-spring-only" or sha(ROBOT_XML)!=MODEL:
        raise RuntimeError("Incorrect KS150 physical model; export EMU_PHYSICAL_PROFILE=h40-spring-only")


def load_lock(root):
    lock=read(root/"study.json")
    if lock!=spec(root,source()):
        raise RuntimeError("Campaign registration/source changed; use original checkout")
    verify_model()
    return lock


def receipt_path(root,arm,seed,target):
    return evidence(root,arm,seed,target)/"stage.json"


def validated_stage(root,arm,seed,target):
    path=receipt_path(root,arm,seed,target)
    if not path.exists():
        return None
    r=read(path)
    if (r.get("schema")!="ks150_abc_stage_v1" or
            r.get("arm")!=arm or r.get("training_seed")!=seed or
            r.get("target")!=target or r.get("actor_source",{}).get("step")!=boundary(target)):
        raise RuntimeError("Wrong stage receipt identity")
    wanted=set(DISCOVERY+HOLDOUT if target==STAGES[-1] else DISCOVERY)
    if {x["seed"] for x in r["exams"]}!=wanted or len(r["exams"])!=len(wanted):
        raise RuntimeError("Incomplete stage exam receipt")
    if (r["actor_source"].get("run_name")!=student(root,arm,seed).name
            or r["actor_source"].get("seed")!=seed
            or r["actor_source"].get("git_commit")!=read(root/"study.json")["source_commit"]
            or r["actor_source"].get("robot_model_sha256")!=MODEL):
        raise RuntimeError("Stage actor lineage mismatch")
    for rel,digest in r["files"].items():
        p=path.parent/rel
        if not p.resolve().is_relative_to(path.parent.resolve()):
            raise RuntimeError("Stage artifact escapes evidence folder")
        if sha(p)!=digest:
            raise RuntimeError("Stage artifact changed: "+str(p))
    required={"actor.emupolicy","checkpoint.pkl","training_config.json"}
    for s in wanted:
        required.update({f"pulse-{s}.zip",f"stairs-{s}/matrix_summary.json",
                         f"stairs-{s}/matrix_worlds.csv",f"stairs-{s}/matrix_curve.csv"})
    if target==STAGES[-1]:
        required.update(f"deep-{s}.zip" for s in HOLDOUT)
    if not required.issubset(r["files"]):
        raise RuntimeError("Missing registered stage artifacts")
    if (r["checkpoint_sha256"]!=r["files"]["checkpoint.pkl"] or
            r["artifact_sha256"]!=r["files"]["actor.emupolicy"]):
        raise RuntimeError("Stage identity differs from frozen artifacts")
    for x in r["exams"]:
        expected=path.parent/f"pulse-{x['seed']}.zip"
        if (Path(x["path"])!=expected or
                x["sha256"]!=r["files"][expected.name]):
            raise RuntimeError("Pulse receipt mismatch")
    return r


def stages_done(root,target):
    return all(validated_stage(root,a,s,target) is not None
               for s in TRAIN_SEEDS for a in ARMS)


def decision_path(root,seed):
    return root/"decisions"/f"C-s{seed}.json"


def approval_digest(root,arm,seed,target):
    if arm!="C" or target<=110_000_000:
        return ""
    p=decision_path(root,seed)
    if not p.exists():
        raise RuntimeError("C review decision required before post-110M training")
    d=read(p)
    if d.get("seed")!=seed or d.get("schema")!="ks150_abc_decision_v1":
        raise RuntimeError("Bad C decision")
    if sha(p.parent/f"C-s{seed}-registration.json")!=d["registration_sha256"]:
        raise RuntimeError("C registration changed")
    if sha(p.parent/f"C-s{seed}-advisory.json")!=d["advisory_sha256"]:
        raise RuntimeError("C advisory changed")
    path=student(root,arm,seed)/"mission_refresher_approval.json"
    if d["action"]=="no_intervention":
        if path.exists() or d["approval_sha256"]:
            raise RuntimeError("Unexpected approval for no-intervention student")
        return ""
    if d["action"]!="approve" or sha(path)!=d["approval_sha256"]:
        raise RuntimeError("Approved receipt missing/changed")
    return d["approval_sha256"]



def reset_recovery_epoch(root,arm,seed):
    """Match the 110M rollback boundary in ALL arms, including no-dose C.

    Preserve old recovery candidates rather than silently deleting them.
    The PPO checkpoint/optimizer/governor are NOT changed by this reset.
    """
    run=student(root,arm,seed)
    folder=root/"decisions"/"recovery-epoch"
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/f"{run.name}.json"
    archive=folder/f"{run.name}.pkl"
    live=run/"checkpoints"/"recovery.pkl"
    if marker.exists():
        m=read(marker)
        if m!=dict(student=run.name,archived_sha256=sha(archive) if archive.exists() else None):
            raise RuntimeError("Recovery epoch receipt changed")
        return
    if read(run/"training_config.json")["steps"]!=110_000_000:
        raise RuntimeError("Missing matched recovery-epoch reset before stage extension")
    if archive.exists() and live.exists():
        raise RuntimeError("Ambiguous interrupted recovery reset")
    if live.exists():
        os.replace(live,archive)
    write_new(marker,dict(student=run.name,
                         archived_sha256=sha(archive) if archive.exists() else None))


def run_stage(root,lock,arm,seed,target):
    if validated_stage(root,arm,seed,target) is not None:
        return
    run=student(root,arm,seed)
    out=evidence(root,arm,seed,target)
    digest=approval_digest(root,arm,seed,target)
    args=recipe(root,arm,seed,target)
    ckpt=run/"checkpoints"
    resume=["--resume","auto"] if ckpt.exists() and any(ckpt.glob("*.pkl")) else []
    if target!=STAGES[0] and not resume:
        raise RuntimeError("Missing prior same-run checkpoint")
    if target!=STAGES[0]:
        previous=validated_stage(root,arm,seed,STAGES[STAGES.index(target)-1])
        if previous is None:
            raise RuntimeError("Missing frozen predecessor stage")
        # Before changing target, require the exact completed predecessor.
        # A silent fallback to an older rotating snapshot would replay
        # training under a different milestone/restart history.
        cfg_file=run/"training_config.json"
        if cfg_file.exists() and read(cfg_file)["steps"]==previous["target"]:
            final=ckpt/"final.pkl"
            latest=ckpt/"latest.pkl"
            if (not final.exists() or sha(final)!=previous["checkpoint_sha256"]
                    or (latest.exists() and sha(latest)!=previous["checkpoint_sha256"])):
                raise RuntimeError("Live predecessor differs from frozen stage checkpoint")
    if (run/"training_config.json").exists():
        old_target=read(run/"training_config.json")["steps"]
        if old_target>target or (old_target!=target and old_target!=STAGES[STAGES.index(target)-1]):
            raise RuntimeError("Existing target does not match current or preceding stage")
    call("-m","emu.cli","train","--check-resume-only",*args,*resume,digest=digest)
    call("-m","emu.cli","train",*args,*resume,digest=digest)
    # Validate complete exact boundary before pinning immutable evidence.
    from emu.training_config import TrainingConfig,training_sensor_contract
    from emu.checkpoint_io import load_checkpoint
    from emu.developmental_school import refresher_recipe_sha256
    from emu.policy_export import build_bundle,write_bundle,load_exported_actor_for_exam
    from scripts.mission_refresher_review import source_hashes,read_pulse
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    cfgpath=run/"training_config.json"
    cfg=TrainingConfig.load(cfgpath)
    final=ckpt/"final.pkl"
    payload=load_checkpoint(final,expected_sensor_contract=training_sensor_contract(cfg))
    m=payload["meta"]
    if (cfg.steps!=target or cfg.seed!=seed or cfg.envs!=WORLDS or
            cfg.physical_profile!="h40-spring-only" or
            cfg.cross_specialty_stairs!=(arm!="A") or cfg.mission_refreshers!=(arm=="C") or
            m["step"]!=boundary(target) or m["run_name"]!=run.name or
            m["source_provenance"]["git_commit"]!=lock["source_commit"] or
            m["robot_model_sha256"]!=MODEL):
        raise RuntimeError("Stage checkpoint does not match registration")
    if target!=STAGES[0] and refresher_recipe_sha256(cfg)!=previous["recipe_sha256"]:
        raise RuntimeError("Substantive training recipe changed between stages")
    if digest:
        pin_path=run/"mission_refresher_pin.json"
        if not pin_path.exists() or m.get("mission_refresher_pin")!=read(pin_path):
            raise RuntimeError("Approved programme not persisted in final checkpoint")
        if m["mission_refresher_pin"].get("receipt_sha256")!=digest:
            raise RuntimeError("Final checkpoint has wrong programme digest")
    elif m.get("mission_refresher_pin") is not None:
        raise RuntimeError("Unexpected programme in unapproved student")
    checkpoint_sha=sha(final)
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/"checkpoint.pkl"
    if frozen.exists():
        if sha(frozen)!=checkpoint_sha:
            raise RuntimeError("Frozen stage checkpoint changed")
    else:
        os.link(final,frozen)
    saved_cfg=out/"training_config.json"
    if not saved_cfg.exists():
        write_new(saved_cfg,read(cfgpath))
    # Pretty formatting differs; verify semantic identity, not whitespace.
    if read(saved_cfg)!=read(cfgpath):
        raise RuntimeError("Frozen config differs")
    policy=out/"actor.emupolicy"
    bundle=build_bundle(final)
    if not policy.exists():
        temporary=out/"actor.partial.emupolicy"
        if temporary.exists():
            raise RuntimeError("Preserve incomplete actor export before retry")
        write_bundle(bundle,temporary)
        os.link(temporary,policy)
        temporary.unlink()
    if read(policy)!=json.loads(json.dumps(bundle)):
        raise RuntimeError("Frozen actor differs from exact final checkpoint")
    _,_,actor=load_exported_actor_for_exam(policy)
    actor_source=actor["source"]
    artifact_sha=actor["artifact_sha256"]
    sensor=training_sensor_contract(cfg)["sha256"]
    sources=source_hashes()
    exams=[]
    files={"checkpoint.pkl":checkpoint_sha,"training_config.json":sha(saved_cfg),
           "actor.emupolicy":artifact_sha}
    seeds=DISCOVERY+HOLDOUT if target==STAGES[-1] else DISCOVERY
    for s in seeds:
        pulse=out/f"pulse-{s}.zip"
        # Existing completed evidence may be reused only with its immutable
        # local receipt. An interrupted/unsealed output must be preserved
        # under a different name before an explicit retry.
        seal=out/f"pulse-{s}.json"
        if not seal.exists():
            if pulse.exists():
                raise RuntimeError("Unsealed pulse; preserve it under another name before retry")
            call("-m","scripts.examine_h40_specialties","--policy",policy,
                 "--output",pulse,"--worlds",2048,"--horizon",600,"--seed",s,"--device","cuda")
            entry=dict(seed=s,path=str(pulse),sha256=sha(pulse))
            read_pulse(pulse,archive_sha256=entry["sha256"],actor_source=actor_source,
                       artifact_sha256=artifact_sha,seed=s,model_sha256=MODEL,
                       sensor_sha256=sensor,expected_sources=sources)
            write_new(seal,entry)
        entry=read(seal)
        if entry!=dict(seed=s,path=str(pulse),sha256=sha(pulse)):
            raise RuntimeError("Pulse changed after seal")
        read_pulse(pulse,archive_sha256=entry["sha256"],actor_source=actor_source,
                   artifact_sha256=artifact_sha,seed=s,model_sha256=MODEL,
                   sensor_sha256=sensor,expected_sources=sources)
        exams.append(entry)
        files[pulse.name]=entry["sha256"]
        files[seal.name]=sha(seal)
        matrix=out/f"stairs-{s}"
        matrix_seal=out/f"stairs-{s}.json"
        if not matrix_seal.exists():
            if matrix.exists():
                raise RuntimeError("Unsealed matrix; preserve directory before retry")
            call("-m","scripts.v5053_matrix_exam","--policy",policy,
                 "--output",matrix,"--mode","stairs","--worlds",512,
                 "--levels",20,"--trials-per-level",10,"--horizon",350,
                 "--seed",s,"--device","cuda")
            manifest=read(matrix/"matrix_summary.json")["manifest"]
            expected=dict(actor_training_source=actor_source,policy_sha256=artifact_sha,
                          robot_model_sha256=MODEL,exam_source_sha256=exam_source_fingerprint(),
                          seed=s,worlds=512,assigned_worlds=400,horizon=350,
                          mode="stairs",levels=20,trials_per_level=10,
                          device="gpu",zero_optimizer_updates=True,first_life_only=True)
            if any(manifest.get(k)!=v for k,v in expected.items()):
                raise RuntimeError("Signed matrix provenance mismatch")
            write_new(matrix_seal,{str((matrix/n).relative_to(out)):sha(matrix/n)
                for n in ("matrix_summary.json","matrix_worlds.csv","matrix_curve.csv")})
        for rel,h in read(matrix_seal).items():
            if sha(out/rel)!=h:
                raise RuntimeError("Signed matrix changed")
            files[rel]=h
        files[matrix_seal.name]=sha(matrix_seal)
        if target==STAGES[-1] and s in HOLDOUT:
            deep=out/f"deep-{s}.zip"
            deep_seal=out/f"deep-{s}.json"
            if not deep_seal.exists():
                if deep.exists():
                    raise RuntimeError("Unsealed deep exam; preserve before retry")
                call("-m","emu.physical_capability_exam","--policy",policy,
                     "--output",deep,"--worlds",8,"--seed",s,"--device","cuda")
                with zipfile.ZipFile(deep) as archive:
                    m=json.loads(archive.read("manifest.json"))
                expected=dict(actor_training_source=actor_source,
                    actor_artifact_sha256=artifact_sha,robot_model_sha256=MODEL,
                    physical_profile="h40-spring-only",sensor_contract_sha256=sensor,
                    seed=s,worlds=8,device="gpu",no_optimizer_updates=True,first_life_only=True)
                if any(m.get(k)!=v for k,v in expected.items()):
                    raise RuntimeError("Deep exam provenance mismatch")
                write_new(deep_seal,dict(sha256=sha(deep)))
            if sha(deep)!=read(deep_seal)["sha256"]:
                raise RuntimeError("Deep exam changed")
            files[deep.name]=sha(deep)
            files[deep_seal.name]=sha(deep_seal)
    stage=dict(schema="ks150_abc_stage_v1",arm=arm,training_seed=seed,target=target,
               checkpoint_sha256=checkpoint_sha,actor_source=actor_source,
               artifact_sha256=artifact_sha,sensor_sha256=sensor,
               source_sha256=sources,recipe_sha256=refresher_recipe_sha256(cfg),
               exams=exams,files=files)
    # Freeze sparse training telemetry at this age, not a later live CSV.
    for name in ("metrics.csv","fixed_eval.csv","performance.json",
                 "results_data.json","run_meta.json"):
        original=run/name
        snapshot=out/name
        if original.is_file():
            if not snapshot.exists():
                # Atomic complete copy, never link a mutable training log.
                tmp=snapshot.with_suffix(snapshot.suffix+".partial")
                shutil.copyfile(original,tmp)
                os.replace(tmp,snapshot)
            if sha(snapshot)!=sha(original):
                raise RuntimeError("Frozen training telemetry differs: "+name)
            files[name]=sha(snapshot)
    write_new(receipt_path(root,arm,seed,target),stage)
    validated_stage(root,arm,seed,target)


def prepare_decision(root,seed):
    from emu.training_config import TrainingConfig
    from emu.capability_focus import birth_lesson_weights
    from scripts.mission_refresher_review import review
    if not stages_done(root,110_000_000):
        raise RuntimeError("Complete all six 110M stages before review")
    before=validated_stage(root,"C",seed,90_000_000)
    after=validated_stage(root,"C",seed,110_000_000)
    if before["recipe_sha256"]!=after["recipe_sha256"]:
        raise RuntimeError("C recipe changed between examinations")
    cfg=TrainingConfig.load(evidence(root,"C",seed,110_000_000)/"training_config.json")
    weights,academy=birth_lesson_weights(schedule=cfg.teaching_schedule,
        step=after["actor_source"]["step"],foundation_steps=cfg.foundation_steps,
        retention_share=cfg.retention_share,academy_enabled=cfg.academy_stream,
        athletic_share=cfg.athletic_rehearsal_frac,academy_intro_steps=cfg.academy_intro_steps,
        academy_start_steps=cfg.academy_start_steps,academy_ramp_steps=cfg.academy_ramp_steps,
        academy_intro_share=cfg.academy_intro_share,academy_max_share=cfg.academy_max_share)
    r=dict(schema="emu_mission_refresher_review_v1",seeds=list(DISCOVERY),
           base_weights=list(weights),model_sha256=MODEL,sensor_sha256=after["sensor_sha256"],
           source_sha256=after["source_sha256"],training_config_sha256=after["recipe_sha256"],
           reference=before,current=after)
    advisory=review(r)
    folder=root/"decisions"
    for path,obj in ((folder/f"C-s{seed}-registration.json",r),
                     (folder/f"C-s{seed}-advisory.json",advisory)):
        # JSON normalizes tuples and integer dictionary keys.
        canonical=json.loads(json.dumps(obj))
        if path.exists():
            if read(path)!=canonical:
                raise RuntimeError("Existing decision evidence differs")
        else:
            write_new(path,canonical)
    return r,advisory,academy


def decide(root,seed,action):
    from scripts.mission_refresher_review import approve_schedule
    r,a,academy=prepare_decision(root,seed)
    p=decision_path(root,seed)
    if p.exists():
        raise RuntimeError("Decision already frozen")
    positive=sum(a["plan"]["extra_clean"].values())>0.
    if (action=="approve")!=positive:
        raise RuntimeError("Approve only a qualifying nonzero dose; otherwise record no_intervention")
    digest=""
    if positive:
        start=r["current"]["actor_source"]["step"]
        receipt=approve_schedule(r,until_step=start+10_000_000,academy_probability=academy)
        path=student(root,"C",seed)/"mission_refresher_approval.json"
        if path.exists():
            if read(path)!=json.loads(json.dumps(receipt)):
                raise RuntimeError("Existing approval differs")
        else:
            write_new(path,receipt)
        digest=sha(path)
    write_new(p,dict(schema="ks150_abc_decision_v1",seed=seed,action=action,
        approval_sha256=digest,
        registration_sha256=sha(p.parent/f"C-s{seed}-registration.json"),
        advisory_sha256=sha(p.parent/f"C-s{seed}-advisory.json")))



def _exam_cells(stage,seed):
    """Per-task fractions; never combine unlike missions into a score."""
    e=next(x for x in stage["exams"] if x["seed"]==seed)
    p=Path(e["path"])
    with zipfile.ZipFile(p) as z:
        pulse=json.loads(z.read("summary.json"))
        runtime=json.loads(z.read("manifest.json"))["runtime"]
        worlds=list(csv.DictReader(io.StringIO(z.read("worlds.csv").decode("utf-8"))))
    cells={}
    for r in worlds:
        key=f"{r['specialty']}/{r['kind']}/{r['level']}"
        for gate,field in (("mission","physical_mission_pass"),("survival","survived")):
            if r[field] not in ("True","False"):
                raise RuntimeError("Malformed raw pulse outcome")
            idx=("pulse",key,gate)
            k,n=cells.get(idx,(0,0))
            cells[idx]=(k+int(r[field]=="True"),n+1)
    if len(worlds)!=2048 or len(cells)!=92:
        raise RuntimeError("Incomplete pulse cells")
    for r in pulse["curves"]:
        key=f"{r['specialty']}/{r['kind']}/{r['level']}"
        for gate,count in (("mission",r["passed"]),("survival",r["survived"])):
            if cells[("pulse",key,gate)]!=(count,r["attempted"]):
                raise RuntimeError("Pulse summary disagrees with raw mission evidence")
    matrix=read(p.parent/f"stairs-{seed}"/"matrix_summary.json")
    for r in matrix["curves"]:
        key=f"{r['kind']}/{r['requested_step_mm']}mm"
        n=r["worlds"]
        for gate in ("leading_top","each_foot_top_five","traversed_5frame",
                     "completed_destinations"):
            cells[("signed_stairs",key,gate)]=(r[gate],n)
        cells[("signed_stairs",key,"survival")]=(round(r["survival_rate"]*n),n)
    deep=p.parent/f"deep-{seed}.zip"
    if deep.exists():
        with zipfile.ZipFile(deep) as z:
            summary=json.loads(z.read("summary.json"))["scenarios"]
        for r in summary:
            for gate in ("full_survivals","zero_command_balance_passes",
                         "bounded_standing_passes","locomotion_mission_passes"):
                if r[gate] is not None:
                    cells[("deep",r["scenario"],gate)]=(r[gate],r["trials"])
    if any(not 0<=k<=n or n<=0 for k,n in cells.values()):
        raise RuntimeError("Invalid examination denominator")
    return cells,runtime


def comparison_rows(root,until):
    rows=[]
    for target in STAGES[:STAGES.index(until)+1]:
        if not stages_done(root,target):
            raise RuntimeError("Comparison requires all six students at every included age")
        for seed in TRAIN_SEEDS:
            stages={a:validated_stage(root,a,seed,target) for a in ARMS}
            for exam_seed in DISCOVERY+HOLDOUT if target==STAGES[-1] else DISCOVERY:
                data={a:_exam_cells(st,exam_seed) for a,st in stages.items()}
                if any(v[1]!=data["A"][1] or set(v[0])!=set(data["A"][0])
                       for v in data.values()):
                    raise RuntimeError("Nonmatched exam task or runtime")
                for key in sorted(data["A"][0]):
                    counts=[data[a][0][key] for a in ARMS]
                    n=counts[0][1]
                    if any(m!=n for _,m in counts):
                        raise RuntimeError("Nonmatched exam cohort")
                    a,b,c=(k/n for k,_ in counts)
                    rows.append(dict(target=target,training_seed=seed,
                        exam_seed=exam_seed,cohort="holdout" if exam_seed in HOLDOUT else "discovery",
                        exam=key[0],task=key[1],gate=key[2],attempts_each=n,
                        A_pass=counts[0][0],B_pass=counts[1][0],C_pass=counts[2][0],
                        B_minus_A_pp=round(100*(b-a),6),C_minus_B_pp=round(100*(c-b),6),
                        C_minus_A_pp=round(100*(c-a),6)))
    return rows


def report(root,until):
    rows=comparison_rows(root,until)
    out=root/f"results-{until//1_000_000}m.zip"
    if out.exists():
        raise FileExistsError(out)
    csvbuf=io.StringIO()
    writer=csv.DictWriter(csvbuf,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary=out.with_suffix(".zip.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary,"x",zipfile.ZIP_DEFLATED) as z:
        z.write(root/"study.json","study.json")
        z.write(root/"smoke.json","smoke.json")
        z.writestr("comparison.csv",csvbuf.getvalue())
        z.writestr("README.md",
            "# A/B/C school comparison\n\n"
            "B−A isolates shared-stair teaching; C−B screens the refresher opportunity. "
            "Separate training seeds and examination seeds are retained. "
            "A no-intervention C is an intention-to-treat outcome, not proof of effective revision. "
            "Raw pulse, signed stairs and final deep exam evidence are included. "
            "No composite ranking, no automatic promotion. No full checkpoints or actor binaries.\n")
        for target in STAGES[:STAGES.index(until)+1]:
            for seed in TRAIN_SEEDS:
                for arm in ARMS:
                    st=validated_stage(root,arm,seed,target)
                    p=receipt_path(root,arm,seed,target)
                    z.write(p,str(p.relative_to(root)))
                    for name in st["files"]:
                        if name not in ("checkpoint.pkl","actor.emupolicy"):
                            artifact=p.parent/name
                            z.write(artifact,str(artifact.relative_to(root)))
        if until>=120_000_000:
            for seed in TRAIN_SEEDS:
                for p in (root/"decisions").glob(f"C-s{seed}*.json"):
                    z.write(p,str(p.relative_to(root)))
                p=student(root,"C",seed)/"mission_refresher_approval.json"
                if p.exists():
                    z.write(p,str(p.relative_to(root)))
                p=student(root,"C",seed)/"mission_refresher_pin.json"
                if p.exists():
                    z.write(p,str(p.relative_to(root)))
            for p in (root/"decisions"/"recovery-epoch").glob("*.json"):
                z.write(p,str(p.relative_to(root)))
    os.link(temporary,out)
    temporary.unlink()
    print("Upload-ready comparison:",out)



def smoke_lifecycle(root,lock):
    """Real two-update checkpoint/target-extension acceptance, not an exam."""
    from emu.training_config import ROLLOUT_STEPS,TrainingConfig,training_sensor_contract
    from emu.checkpoint_io import load_checkpoint
    from emu.policy_export import build_bundle,write_bundle,load_exported_actor_for_exam
    run=root/"acceptance"/"lifecycle"
    folder=root/"acceptance"
    folder.mkdir(parents=True,exist_ok=True)
    for target in (8*ROLLOUT_STEPS,16*ROLLOUT_STEPS):
        receipt=folder/f"lifecycle-{target}.json"
        if receipt.exists():
            r=read(receipt)
            if r["source_commit"]!=lock["source_commit"] or r["step"]!=target:
                raise RuntimeError("Lifecycle receipt differs")
            if sha(folder/f"lifecycle-{target}.emupolicy")!=r["artifact_sha256"]:
                raise RuntimeError("Lifecycle actor changed")
            continue
        args=[*RECIPE,*ARMS["C"],"--envs","8","--run-dir",str(run),
              "--seed","9581","--steps",str(target)]
        ckpt=run/"checkpoints"
        resume=["--resume","auto"] if ckpt.exists() and any(ckpt.glob("*.pkl")) else []
        call("-m","emu.cli","train","--check-resume-only",*args,*resume,bounded=True)
        call("-m","emu.cli","train",*args,*resume,bounded=True)
        cfg=TrainingConfig.load(run/"training_config.json")
        final=ckpt/"final.pkl"
        payload=load_checkpoint(final,expected_sensor_contract=training_sensor_contract(cfg))
        m=payload["meta"]
        if (m["step"]!=target or m["source_provenance"]["git_commit"]!=lock["source_commit"]
                or m["run_name"]!=run.name or m["robot_model_sha256"]!=MODEL
                or m.get("mission_refresher_pin") is not None
                or cfg.envs!=8 or not cfg.mission_refreshers):
            raise RuntimeError("Lifecycle acceptance checkpoint mismatch")
        policy=folder/f"lifecycle-{target}.emupolicy"
        bundle=build_bundle(final)
        if not policy.exists():
            tmp=policy.with_suffix(".partial.emupolicy")
            write_bundle(bundle,tmp)
            os.link(tmp,policy)
            tmp.unlink()
        if read(policy)!=json.loads(json.dumps(bundle)):
            raise RuntimeError("Lifecycle export differs")
        load_exported_actor_for_exam(policy)
        write_new(receipt,dict(source_commit=lock["source_commit"],step=target,
                               checkpoint_sha256=sha(final),artifact_sha256=sha(policy)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("plan","lock","smoke","advance","go","review","decide","status","report"))
    p.add_argument("--root",type=Path,default=Path.home()/"emu-data/studies/ks150-abc-v1")
    p.add_argument("--ci-verified-sha")
    p.add_argument("--stage",type=int,choices=STAGES)
    p.add_argument("--seed",type=int,choices=TRAIN_SEEDS)
    p.add_argument("--action",choices=("approve","no_intervention"))
    a=p.parse_args(argv)
    root=root_checked(a.root)
    # Must precede imports of emu.contracts (physical variant is import-time).
    os.environ["EMU_PHYSICAL_PROFILE"]="h40-spring-only"
    if a.command=="plan":
        print(json.dumps(spec(root,source()),indent=2))
        for arm in ARMS:
            print(arm," ".join(recipe(root,arm,TRAIN_SEEDS[0],STAGES[0])))
        return 0
    if a.command=="lock":
        verify_model()
        registration=spec(root,source())
        root.mkdir(parents=True,exist_ok=False)
        write_new(root/"study.json",registration)
        return 0
    with locked(root):
        lock=load_lock(root)
        if a.command=="report":
            if a.stage is None:
                p.error("report requires --stage")
            report(root,a.stage)
            return 0
        if a.command=="status":
            for target in STAGES:
                print(target,{f"{arm}-s{seed}":validated_stage(root,arm,seed,target) is not None
                              for seed in TRAIN_SEEDS for arm in ARMS})
            return 0
        if a.command=="smoke":
            if a.ci_verified_sha!=lock["source_commit"]:
                raise RuntimeError("Explicit confirmation of green CI for exact source required")
            if (root/"smoke.json").exists():
                raise RuntimeError("Smoke already recorded")
            call("-m","emu.cli","train","--check-resume-only",
                 *recipe(root,"A",TRAIN_SEEDS[0],STAGES[0]))
            call("scripts/smoke_gpu_carpet.py","--worlds",2048,"--frames",2,
                 "--device","cuda","--prisms","--ppo","--full-update",
                 "--stair-survival-curriculum","--abc-refresher-smoke")
            smoke_lifecycle(root,lock)
            write_new(root/"smoke.json",dict(source_commit=lock["source_commit"],model_sha256=MODEL,
                                            operator_attested_ci=True,gpu_smoke=True,lifecycle=True))
            return 0
        if a.command=="review":
            for seed in TRAIN_SEEDS:
                _,advisory,_=prepare_decision(root,seed)
                print(seed,json.dumps(advisory["plan"],indent=2))
            return 0
        if a.command=="decide":
            if a.seed is None or a.action is None:
                p.error("decide requires --seed and --action")
            decide(root,a.seed,a.action)
            return 0
        if a.command=="advance" and a.stage is None:
            p.error("advance requires --stage")
        if a.command=="go" and a.stage is not None:
            p.error("go uses registered milestones, not --stage")
        smoke=read(root/"smoke.json")
        if smoke!=dict(source_commit=lock["source_commit"],model_sha256=MODEL,
                       operator_attested_ci=True,gpu_smoke=True,lifecycle=True):
            raise RuntimeError("Need exact-source CI attestation, GPU smoke and lifecycle acceptance")
        targets=STAGES if a.command=="go" else (a.stage,)
        for stage in targets:
            for target in STAGES[:STAGES.index(stage)]:
                if not stages_done(root,target):
                    raise RuntimeError("Cannot skip matched stage")
            if stages_done(root,stage):
                continue
            if stage>110_000_000:
                if a.command=="go" and not all(decision_path(root,s).exists() for s in TRAIN_SEEDS):
                    # The review gate is an evidence handoff, not just a
                    # console pause. Preserve a complete uploadable report
                    # before any treatment decision can be made.
                    output=root/"results-110m.zip"
                    if not output.exists():
                        report(root,110_000_000)
                    print("STOP at matched 110M review. Review results-110m.zip, then review and decide for both C students; go again.")
                    return 0
                for seed in TRAIN_SEEDS:
                    approval_digest(root,"C",seed,stage)
            if stage==120_000_000:
                for seed in TRAIN_SEEDS:
                    for arm in ARMS:
                        reset_recovery_epoch(root,arm,seed)
            for seed in TRAIN_SEEDS:
                for arm in ARMS:
                    run_stage(root,lock,arm,seed,stage)
            print("Matched stage complete:",stage,flush=True)
        if a.command=="go":
            output=root/"results-140m.zip"
            if not output.exists():
                report(root,STAGES[-1])
        else:
            print("No later stage started.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
