"""Unattended matched 4096-world Live Skill School pilot.

One command: authenticate parents, register source, GPU acceptance,
matched endpoint exams and six sequential 40-update forks, sealed ZIP.
No interactive teacher approvals; no routine external exam during learning.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from scripts.ks150_abc_school import (
    call, locked, read, root_checked, sha, source, verify_model, write_new,
    _exam_cells,
)
from scripts.live_skill_pilot_plan import (
    ARMS, COMMON, MODEL, PARENTS, PARENT_SOURCE, PARENT_STEP, TARGET,
    WORLDS, ROLLOUT, UPDATES, registration, verify_parent,
)

EXAM_SEED=9861
SCHEMA="ks150_live_skill_pilot_run_v1"

@dataclass(frozen=True)
class StudyPlan:
    name: str
    profile: str
    arms: tuple
    common: tuple
    updates: int
    exam_seed: int
    uphill_share: float
    result: str

    @property
    def target(self):
        return PARENT_STEP+self.updates*WORLDS*ROLLOUT

    def spec(self,parent_root):
        base=registration(parent_root)
        base.update(schema=SCHEMA,study_kind=self.name,uphill_profile=self.profile,
                    arms={k:list(v) for k,v in self.arms},common=list(self.common),
                    updates=self.updates,target=self.target,exam_seed=self.exam_seed,
                    total_additional_steps=len(PARENTS)*len(self.arms)*self.updates*WORLDS*ROLLOUT,
                    uphill_share=self.uphill_share,result=self.result)
        return base

DEFAULT=StudyPlan("original","ladder",tuple(ARMS.items()),tuple(COMMON),
                  UPDATES,EXAM_SEED,.15,"results-live-skill-pilot.zip")
def cliff_plan():
    common=list(COMMON)
    for flag in ("--skill-uphill-reference-share","--skill-uphill-practice-share"):
        common[common.index(flag)+1]=".25"
    common+=["--skill-uphill-profile","cliff"]
    return StudyPlan("cliff","cliff",(("U",()),("H",("--skill-adaptive-uphill",))),
                     tuple(common),100,9862,.25,"results-live-skill-cliff.zip")


def engagement_plan():
    base=cliff_plan()
    common=list(base.common)
    common[common.index("--skill-uphill-profile")+1]="engagement"
    return StudyPlan("engagement","engagement",base.arms,tuple(common),
                     160,9863,.25,"results-live-skill-engagement.zip")


def freedom_plan():
    base=engagement_plan()
    return StudyPlan("freedom","engagement",(*base.arms,
        ("F",("--skill-adaptive-uphill","--skill-novice-fall-relief"))),
        base.common,160,9864,.25,"results-live-skill-freedom.zip")


def publish_copy(src,dst,digest):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(dst)!=digest:
            raise RuntimeError(f"Registered file changed: {dst}")
        return
    tmp=dst.with_name(dst.name+".partial")
    shutil.copyfile(src,tmp)
    if sha(tmp)!=digest:
        raise RuntimeError(f"Copy failed digest check: {src}")
    os.replace(tmp,dst)


def require_files(folder,files):
    for name,digest in files.items():
        p=folder/name
        if not p.is_file() or sha(p)!=digest:
            raise RuntimeError(f"Sealed file missing or changed: {p}")


def stage_parent(root,parent_root,seed):
    old=verify_parent(parent_root,seed)
    dest=root/"parents"/f"s{seed}"
    p=PARENTS[seed]
    publish_copy(old/"checkpoint.pkl",dest/"checkpoints"/"parent.pkl",p["checkpoint"])
    publish_copy(old/"training_config.json",dest/"training_config.json",p["config"])
    publish_copy(old/"actor.emupolicy",dest/"actor.emupolicy",p["actor"])
    publish_copy(old/"stage.json",dest/"stage.json",sha(old/"stage.json"))
    return dest


def student(root,arm,seed):
    return root/"runs"/f"{arm}-s{seed}"


def train_args(root,arm,seed,parent,plan=DEFAULT):
    from emu.checkpoint_io import find_resume
    run=student(root,arm,seed)
    args=[*plan.common,*dict(plan.arms)[arm],"--seed",str(seed),"--steps",str(plan.target),
          "--run-dir",str(run)]
    if find_resume(run) is not None:
        args+=["--resume","auto"]
    else:
        args+=["--resume",str(parent/"checkpoints"/"parent.pkl"),
               "--research-parent-source",PARENT_SOURCE,
               "--research-parent-sha256",PARENTS[seed]["checkpoint"]]
    return args


def smoke(root,commit,plan=DEFAULT):
    """Bounded mature physics checks; no random-actor mastery requirement."""
    out=root/"acceptance"
    out.mkdir(exist_ok=True)
    receipt=out/"combined.json"
    args=["-m","scripts.smoke_gpu_carpet","--worlds",str(WORLDS),
          "--device","cuda","--prisms","--ppo","--full-update",
          "--stair-survival-curriculum","--live-recovery-smoke","--live-uphill-smoke",
          "--benchmark-updates","12"]
    if plan.profile!="ladder": args+=["--uphill-profile",plan.profile]
    if plan.name=="freedom": args+=["--novice-fall-relief"]
    expected=dict(source_commit=commit,args=args)
    if receipt.exists():
        if read(receipt)!=expected:
            raise RuntimeError("Acceptance receipt identity mismatch")
        return
    print("PRECHECK: both mature lesson families, 12 updates",flush=True)
    call(*args)
    write_new(receipt,expected)


def preserve_unsealed(path):
    """Never trust half-published exam output; keep it for diagnosis."""
    if path.exists():
        os.replace(path,path.with_name(path.name+f".unsealed-{time.time_ns()}"))


def endpoint(folder,policy,cfg,expected_actor,plan=DEFAULT):
    """One matched independent seed at start/end; raw exams sealed by hash."""
    from emu.policy_export import load_exported_actor_for_exam
    from emu.training_config import training_sensor_contract
    from scripts.mission_refresher_review import read_pulse,source_hashes
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    folder.mkdir(parents=True,exist_ok=True)
    _,_,actor=load_exported_actor_for_exam(policy)
    src=actor["source"]
    if any(src.get(k)!=v for k,v in expected_actor.items()):
        raise RuntimeError("Endpoint actor lineage mismatch")
    artifact=actor["artifact_sha256"]
    sensor=training_sensor_contract(cfg)["sha256"]
    seed=plan.exam_seed
    pulse=folder/f"pulse-{seed}.zip"
    matrix=folder/f"stairs-{seed}"
    seal=folder/"endpoint.json"
    if seal.exists():
        receipt=read(seal)
        if (receipt["actor_source"]!=src or receipt["artifact_sha256"]!=artifact
                or receipt["exam_seed"]!=seed):
            raise RuntimeError("Endpoint receipt actor mismatch")
        require_files(folder,receipt["files"])
        return receipt
    pulse_seal=folder/"pulse.json"
    if not pulse_seal.exists():
        preserve_unsealed(pulse)
        call("-m","scripts.examine_h40_specialties","--policy",policy,
             "--output",pulse,"--worlds","2048","--horizon","600",
             "--seed",seed,"--device","cuda")
        digest=sha(pulse)
        read_pulse(pulse,archive_sha256=digest,actor_source=src,
                   artifact_sha256=artifact,seed=seed,model_sha256=MODEL,
                   sensor_sha256=sensor,expected_sources=source_hashes())
        write_new(pulse_seal,dict(sha256=digest,artifact_sha256=artifact))
    ps=read(pulse_seal)
    if ps["artifact_sha256"]!=artifact:
        raise RuntimeError("Pulse actor changed")
    counts,manifest=read_pulse(pulse,archive_sha256=ps["sha256"],actor_source=src,
                   artifact_sha256=artifact,seed=seed,model_sha256=MODEL,
                   sensor_sha256=sensor,expected_sources=source_hashes())
    matrix_seal=folder/"matrix.json"
    if not matrix_seal.exists():
        preserve_unsealed(matrix)
        call("-m","scripts.v5053_matrix_exam","--policy",policy,
             "--output",matrix,"--mode","stairs","--worlds","512",
             "--levels","20","--trials-per-level","10","--horizon","350",
             "--seed",seed,"--device","cuda")
        m=read(matrix/"matrix_summary.json")["manifest"]
        expected=dict(actor_training_source=src,policy_sha256=artifact,
                      robot_model_sha256=MODEL,exam_source_sha256=exam_source_fingerprint(),
                      seed=seed,worlds=512,assigned_worlds=400,horizon=350,
                      mode="stairs",levels=20,trials_per_level=10,device="gpu",
                      zero_optimizer_updates=True,first_life_only=True)
        if any(m.get(k)!=v for k,v in expected.items()):
            raise RuntimeError("Signed matrix provenance mismatch")
        write_new(matrix_seal,dict(artifact_sha256=artifact,files={
            str((matrix/n).relative_to(folder)):sha(matrix/n)
            for n in ("matrix_summary.json","matrix_worlds.csv","matrix_curve.csv")}))
    ms=read(matrix_seal)
    if ms["artifact_sha256"]!=artifact:
        raise RuntimeError("Matrix actor changed")
    require_files(folder,ms["files"])
    files={pulse.name:ps["sha256"],pulse_seal.name:sha(pulse_seal),
           matrix_seal.name:sha(matrix_seal),**ms["files"]}
    receipt=dict(actor_source=src,artifact_sha256=artifact,exam_seed=seed,
        files=files,exams=[dict(seed=seed,path=str(pulse),sha256=ps["sha256"])])
    # Reuse established raw pulse and matrix evidence checks before sealing.
    cells,_=_exam_cells(receipt,seed)
    if len(cells)!=292:
        raise RuntimeError("Incomplete endpoint cells")
    write_new(seal,receipt)
    return receipt


def live_coverage(run,commit,plan=DEFAULT):
    """Validate raw windows; preserve abandoned epochs and interrupted tails.

    Counts are per segment/epoch, never pooled across a rollback or retry.
    A torn final JSONL record can exist after power loss; ignore only an
    unterminated, unparsable final line, preserving its exact original bytes.
    """
    from emu.live_skill_uphill import uphill_heights
    groups={}
    last=[]
    torn=[]
    for p in sorted((run/"live_skills").glob("segment-*.jsonl")):
        data=p.read_bytes()
        lines=data.splitlines(keepends=True)
        previous=None
        for i,line in enumerate(lines):
            try:
                row=json.loads(line)
            except (UnicodeError,json.JSONDecodeError):
                if i==len(lines)-1 and not line.endswith(b"\n"):
                    torn.append(p.name)
                    break
                raise
            if (row["source_commit"]!=commit or row["worlds"]!=WORLDS
                    or not PARENT_STEP<=row["start_step"]<row["end_step"]<=plan.target
                    or (plan.profile!="ladder" and row.get("uphill_heights_m")!=list(uphill_heights(plan.profile)))
                    or (plan.profile=="engagement" and (row.get("uphill_profile")!="engagement"
                         or not isinstance(row.get("uphill_motion_reward_sums"),dict)))
                    or (plan.name=="freedom" and not isinstance(row.get("uphill_fall_cost_sums"),dict))
                    or row["world_frames"]!=row["end_step"]-row["start_step"]
                    or (previous is not None and row["start_step"]!=previous)):
                raise RuntimeError("Live observer lineage/window mismatch")
            previous=row["end_step"]
            for cell in row["cells"].values():
                n,k=cell["attempts"],cell["passes"]
                if type(n) is not int or type(k) is not int or not 0<=k<=n:
                    raise RuntimeError("Invalid live observer counts")
            key=(p.name,row["policy_epoch"])
            g=groups.setdefault(key,dict(segment=p.name,policy_epoch=row["policy_epoch"],
                start_step=row["start_step"],end_step=row["end_step"],world_frames=0,
                reference_recovery_attempts=[0]*6,reference_uphill_attempts=[0]*6,
                recovery_reference_frames=[0]*6,uphill_reference_frames=[0]*6,
                recovery_practice_frames=[0]*6,uphill_practice_frames=[0]*6))
            g["end_step"]=row["end_step"]
            g["world_frames"]+=row["world_frames"]
            for j in range(6):
                for family in ("recovery","uphill"):
                    cell=row["cells"].get(f"reference_{family}/mission/{j}",{"attempts":0})
                    g[f"reference_{family}_attempts"][j]+=cell["attempts"]
                g["recovery_practice_frames"][j]+=row["practice_frames_by_case"][j]
                g["uphill_practice_frames"][j]+=row["uphill_practice_frames_by_case"][j]
                g["recovery_reference_frames"][j]+=row.get("reference_frames_by_case",[0]*6)[j]
                g["uphill_reference_frames"][j]+=row.get("uphill_reference_frames_by_case",[0]*6)[j]
            last.append(row["end_step"])
    if plan.target not in last:
        raise RuntimeError("Missing final live observer window")
    return dict(segments=list(groups.values()),unparsed_interrupted_tails=torn,
                note="Counts from different segments/epochs may overlap and must not be pooled blindly.")


def verify_final(root,arm,seed,commit,plan=DEFAULT):
    from emu.training_config import TrainingConfig,training_sensor_contract
    from emu.checkpoint_io import load_checkpoint
    from emu.live_skill_teacher import RecoveryTeacher
    from emu.live_skill_uphill_teacher import UphillTeacher
    from emu.policy_export import build_bundle,write_bundle,load_exported_actor_for_exam
    run=student(root,arm,seed)
    cfg=TrainingConfig.load(run/"training_config.json")
    ck=run/"checkpoints"/"final.pkl"
    payload=load_checkpoint(ck,expected_sensor_contract=training_sensor_contract(cfg))
    meta=payload["meta"]
    if (cfg.envs!=WORLDS or cfg.steps!=plan.target or cfg.seed!=seed
            or not cfg.live_skill_observer or cfg.fixed_eval_every_steps!=0 or cfg.physical_profile!="h40-spring-only"
            or cfg.skill_adaptive_recovery!=(arm=="R") or cfg.skill_adaptive_uphill!=(arm in ("H","F"))
            or cfg.skill_novice_fall_relief!=(arm=="F")
            or cfg.skill_uphill_stage!="none" or cfg.mission_refreshers or cfg.cross_specialty_stairs
            or cfg.skill_uphill_profile!=plan.profile
            or any(getattr(cfg,k)!=.15 for k in ("skill_reference_recovery_share","skill_practice_recovery_share"))
            or any(getattr(cfg,k)!=plan.uphill_share for k in ("skill_uphill_reference_share","skill_uphill_practice_share"))
            or meta["step"]!=plan.target or meta["run_name"]!=run.name
            or meta["source_provenance"]["git_commit"]!=commit or meta["robot_model_sha256"]!=MODEL):
        raise RuntimeError("Final learner differs from registered treatment")
    teacher={}
    for key,active,cls in (("live_recovery_teacher",arm=="R",RecoveryTeacher),
                           ("live_uphill_teacher",arm in ("H","F"),UphillTeacher)):
        if active:
            restored=(cls.from_state_dict(meta.get(key),plan.target,plan.profile) if arm in ("H","F") else
                      cls.from_state_dict(meta.get(key),plan.target))
            teacher[key]=restored.state_dict(plan.target)
            if teacher[key]["clock"]>plan.updates*WORLDS*ROLLOUT:
                raise RuntimeError("Teacher observed more than registered budget")
        elif key in meta:
            raise RuntimeError("Unexpected active teacher")
    rm=read(run/"run_meta.json")
    if rm["effective_ppo_minibatches"]!=16:
        raise RuntimeError("Wrong effective minibatch count")
    out=root/"evidence"/run.name
    out.mkdir(parents=True,exist_ok=True)
    export=out/"actor.emupolicy"
    bundle=build_bundle(ck)
    if not export.exists():
        tmp=export.with_suffix(".partial")
        write_bundle(bundle,tmp)
        os.replace(tmp,export)
    if read(export)!=json.loads(json.dumps(bundle)):
        raise RuntimeError("Final export differs from checkpoint")
    _,_,actor=load_exported_actor_for_exam(export)
    coverage=live_coverage(run,commit,plan)
    # Preserve raw restart/rollback segments rather than silently pooling.
    files={}
    for p in sorted((run/"live_skills").glob("*.jsonl")):
        name="live_skills/"+p.name
        publish_copy(p,out/name,sha(p))
        files[name]=sha(out/name)
    for name in ("training_config.json","run_meta.json","metrics.csv",
                 "fixed_eval.csv","performance.json","results_data.json"):
        if (run/name).is_file():
            publish_copy(run/name,out/name,sha(run/name))
            files[name]=sha(out/name)
    receipt=dict(arm=arm,seed=seed,step=plan.target,source_commit=commit,
                 parent_sha256=PARENTS[seed]["checkpoint"],checkpoint_sha256=sha(ck),
                 actor_source=actor["source"],artifact_sha256=actor["artifact_sha256"],
                 teacher=teacher,live_coverage=coverage,files=files)
    seal=out/"training.json"
    if seal.exists():
        if read(seal)!=receipt:
            raise RuntimeError("Final training receipt changed")
    else:
        write_new(seal,receipt)
    return out,cfg,receipt


def compare(root,plan=DEFAULT):
    arms=dict(plan.arms)
    rows=[]
    for seed in PARENTS:
        paths={"parent":root/"evidence"/f"parent-s{seed}",
               **{a:root/"evidence"/f"{a}-s{seed}" for a in arms}}
        data={a:_exam_cells(read(p/"endpoint.json"),plan.exam_seed) for a,p in paths.items()}
        base,runtime=data["parent"]
        if any(rt!=runtime or set(c)!=set(base) for c,rt in data.values()):
            raise RuntimeError("Endpoint runtime/tasks differ")
        for key,(p,n) in sorted(base.items()):
            k={a:data[a][0][key][0] for a in arms}
            if any(data[a][0][key][1]!=n for a in arms):
                raise RuntimeError("Unmatched endpoint denominator")
            row=dict(training_seed=seed,exam_seed=plan.exam_seed,exam=key[0],
                task=key[1],gate=key[2],attempts_each=n,parent_pass=p)
            row.update({f"{a}_pass":v for a,v in k.items()})
            row.update({f"{a}_minus_U_pp":round(100*(v-k["U"])/n,6) for a,v in k.items() if a!="U"})
            if "F" in k: row["F_minus_H_pp"]=round(100*(k["F"]-k["H"])/n,6)
            rows.append(row)
    return rows


def cliff_windows(root,plan):
    """One row per physical height, cohort and raw window. Do not merge
    overlapping restart/rollback epochs into fictitious extra evidence."""
    from emu.live_skill_uphill import uphill_heights
    gates=("approach","first_contact","first_support","each_foot_five","traversed","mission","survived")
    frame_gates=("forward_gt_005","reverse_lt_minus005","slow_abs_le_005",
                 "heading_aligned","command_016")
    rows=[]
    for seed in PARENTS:
        for arm,_ in plan.arms:
            name=f"{arm}-s{seed}"
            folder=root/"evidence"/name
            for path in sorted((folder/"live_skills").glob("segment-*.jsonl")):
                lines=path.read_bytes().splitlines(keepends=True)
                for j,line in enumerate(lines):
                    try: raw=json.loads(line)
                    except (UnicodeError,json.JSONDecodeError):
                        if j==len(lines)-1 and not line.endswith(b"\n"): break
                        raise
                    for i,h in enumerate(uphill_heights(plan.profile)):
                        for cohort in ("reference","practice"):
                            prefix=f"{cohort}_uphill"
                            cells=raw["cells"]
                            row=dict(student=name,segment=path.name,epoch=raw["policy_epoch"],
                                start_step=raw["start_step"],end_step=raw["end_step"],
                                world_frames=raw["world_frames"],height_mm=round(h*1000),cohort=cohort,
                                case_frames=raw[f"uphill_{cohort}_frames_by_case"][i])
                            row["world_frame_share"]=row["case_frames"]/row["world_frames"]
                            for gate in gates:
                                cell=cells.get(f"{prefix}/{gate}/{i}",{"attempts":0,"passes":0})
                                row[gate+"_attempts"]=cell["attempts"]
                                row[gate+"_passes"]=cell["passes"]
                            for gate in frame_gates:
                                cell=cells.get(f"reference_uphill_before_approach_frame/{gate}/{i}",{}) if cohort=="reference" else {}
                                row[gate+"_frames"]=cell.get("attempts","")
                                row[gate+"_pass_frames"]=cell.get("passes","")
                            if plan.profile=="engagement":
                                for s in range(6):
                                    key=f"practice_uphill_stage_{s}"
                                    credit=cells.get(f"{key}_frames/{i}",{})
                                    birth=cells.get(f"{key}_births/{i}",{})
                                    row[f"stage_{s}_frames"]=credit.get("attempts",0) if cohort=="practice" else ""
                                    row[f"stage_{s}_payments"]=credit.get("passes",0) if cohort=="practice" else ""
                                    row[f"stage_{s}_births"]=birth.get("attempts",0) if cohort=="practice" else ""
                                    row[f"stage_{s}_reward_sum"]=sum(
                                        a*cells.get(f"{key}_bonus_{a}/{i}",{}).get("passes",0) for a in range(1,5)
                                    ) if cohort=="practice" else ""
                                for motion in ("forward","reverse","slow"):
                                    n,total,loc=raw.get("uphill_motion_reward_sums",{}).get(f"{motion}/{i}",[0,0.,0.])
                                    row[motion+"_reward_frames"]=n if cohort=="reference" else ""
                                    row[motion+"_mean_total_reward"]=total/n if cohort=="reference" and n else ""
                                    row[motion+"_mean_locomotion_reward"]=loc/n if cohort=="reference" and n else ""
                            if plan.name=="freedom":
                                for s in range(6):
                                    n,relieved,full,actual,age,diverged=raw["uphill_fall_cost_sums"].get(
                                        f"{cohort}/{s}/{i}",[0,0,0.,0.,0.,0])
                                    if (type(n) is not int or type(relieved) is not int or
                                        type(diverged) is not int or not 0<=relieved<=n or
                                        not 0<=diverged<=n or not full<=actual<=0. or
                                        not 0<=age or (not relieved and abs(full-actual)>1e-5)):
                                        raise RuntimeError("Invalid fall-cost evidence")
                                    if relieved and (arm!="F" or cohort!="practice" or s==0):
                                        raise RuntimeError("Relief escaped registered treatment")
                                    row[f"stage_{s}_falls"]=n
                                    row[f"stage_{s}_relieved_falls"]=relieved
                                    row[f"stage_{s}_full_fall_cost"]=-full
                                    row[f"stage_{s}_actual_fall_cost"]=-actual
                                    row[f"stage_{s}_fall_age_sum"]=age
                                    row[f"stage_{s}_diverged_falls"]=diverged
                            rows.append(row)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def report(root,plan=DEFAULT):
    rows=compare(root,plan)
    buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    cliff=cliff_windows(root,plan) if plan.profile in ("cliff","engagement") else None
    out=root/plan.result
    tmp=out.with_suffix(".zip.partial")
    if out.exists():
        with zipfile.ZipFile(out) as z:
            if (json.loads(z.read("study.json"))!=read(root/"study.json")
                    or z.read("comparison.csv").decode()!=buf.getvalue()
                    or (cliff is not None and z.read("cliff_windows.csv").decode()!=cliff)
                    or z.testzip() is not None):
                raise RuntimeError("Existing results archive differs")
        return out
    preserve_unsealed(tmp)
    with zipfile.ZipFile(tmp,"x",zipfile.ZIP_DEFLATED) as z:
        z.write(root/"study.json","study.json")
        z.writestr("comparison.csv",buf.getvalue())
        if cliff is not None: z.writestr("cliff_windows.csv",cliff)
        z.writestr("README.md","Matched "+plan.name+" pilot. Raw live windows and teacher decisions "
          "are training evidence, not independent trials. Endpoint exams use the same registered "
          "seed on each parent and child. Do not pool restart/rollback epochs blindly. "
          "Two independent parent learners, no automatic promotion. No actor/checkpoint binaries.\n")
        for p in sorted((root/"acceptance").glob("*.json")):
            z.write(p,str(p.relative_to(root)))
        for seed in PARENTS:
            parent=root/"parents"/f"s{seed}"
            for name in ("stage.json","training_config.json"):
                z.write(parent/name,str((parent/name).relative_to(root)))
            for name in (f"parent-s{seed}",*(f"{a}-s{seed}" for a,_ in plan.arms)):
                folder=root/"evidence"/name
                for kind in ("endpoint","training"):
                    seal=folder/(kind+".json")
                    if kind=="training" and name.startswith("parent-"):
                        continue
                    r=read(seal)
                    require_files(folder,r["files"])
                    z.write(seal,str(seal.relative_to(root)))
                    for rel in r["files"]:
                        p=folder/rel
                        z.write(p,str(p.relative_to(root)))
    os.replace(tmp,out)
    return out


def go(root,parent_root,plan=DEFAULT):
    os.environ["EMU_PHYSICAL_PROFILE"]="h40-spring-only"
    commit=source()
    verify_model()
    root=root_checked(root)
    parent_root=Path(parent_root).expanduser().resolve()
    if root==parent_root or parent_root in root.parents or root in parent_root.parents:
        raise ValueError("Pilot must use a separate study root")
    root.mkdir(parents=True,exist_ok=True)
    with locked(root):
        spec=plan.spec(parent_root)
        spec.update(status="registered_unattended",source_commit=commit,
                    exam_policy="parent and final only, matched single seed")
        spec=json.loads(json.dumps(spec))
        lock=root/"study.json"
        if lock.exists():
            if read(lock)!=spec:
                raise RuntimeError("Study source/recipe registration differs")
        else:
            write_new(lock,spec)
        parents={s:stage_parent(root,parent_root,s) for s in PARENTS}
        # Verify each actual fork using production CLI before expensive GPU work.
        for seed,p in parents.items():
            for arm,_ in plan.arms:
                call("-m","emu.cli","train","--check-resume-only",*train_args(root,arm,seed,p,plan))
        smoke(root,commit,plan)
        from emu.training_config import TrainingConfig
        for seed,p in parents.items():
            cfg=TrainingConfig.load_research_ancestor(p/"training_config.json")
            endpoint(root/"evidence"/f"parent-s{seed}",p/"actor.emupolicy",cfg,
                     dict(git_commit=PARENT_SOURCE,step=PARENT_STEP,seed=seed,
                          robot_model_sha256=MODEL,run_name=f"A-s{seed}"),plan)
        # Rotate treatment order across parents to reduce systematic time bias.
        order=tuple(a for a,_ in plan.arms)
        for seed,order in ((9501,order),(9502,order[::-1])):
            for arm in order:
                print(f"STUDENT {arm}-s{seed}: {PARENT_STEP:,} -> {plan.target:,}",flush=True)
                call("-m","emu.cli","train",*train_args(root,arm,seed,parents[seed],plan))
                out,cfg,r=verify_final(root,arm,seed,commit,plan)
                endpoint(out,out/"actor.emupolicy",cfg,dict(git_commit=commit,
                         step=plan.target,seed=seed,robot_model_sha256=MODEL,run_name=f"{arm}-s{seed}"),plan)
        print("UPLOAD READY:",report(root,plan),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--parent-root",type=Path,
        default=Path.home()/"emu-data/studies/ks150-abc-unattended-v2")
    p.add_argument("--classroom",choices=("original","cliff","engagement","freedom"),default="original")
    a=p.parse_args(argv)
    go(a.root,a.parent_root,{"original":DEFAULT,"cliff":cliff_plan(),"engagement":engagement_plan(),"freedom":freedom_plan()}[a.classroom])


if __name__=="__main__":
    main()
