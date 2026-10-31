#!/usr/bin/env python3
"""v5033: source-guarded 24-arm matched research from the immutable v5032 final.pkl.

Plan-only by default. --run performs foreground sequential experiments,
one GPU training process at a time; never changes source, public main, or
the parent checkpoint. Stage A = 24 independent 8M continuations;
Stage B = selected arms +16M *from their own Stage A result*.

Every treatment receives a separate run, source/config/provenance manifest,
CPU+CUDA PPO gate, validated full learner checkpoint and independent exams.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from emu.science import ARM_COUNT, ARM_NAMES, arm_name, expected_fields
BRANCH="research/v5033-24-arm-science"
PARENT_SOURCE="9b11bbe596aea7201d29a6beb2d192922f8e2e9a"
PARENT_SHA256="238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b"
PARENT_STEP=260_046_848
PARENT=Path.home()/"robot_lab/emu_runs/v5032_pinnacle_s761_v2/pinnacle_cold260/checkpoints/final.pkl"
ROOT=Path.home()/"robot_lab/emu_runs/v5033_science_s761"
ENVIRONMENTS=2048
SEED=761
TARGET_A=268_000_000
TARGET_B=284_000_000
# Same base as the ACTUALLY TRAINED v5032 resolved config, not its
# erroneous launcher manifest: verified ascent was ZERO, not 3.
BASE_ARGS=(
    "--profile","sim2real","--device","cuda",
    "--seed",str(SEED),"--envs",str(ENVIRONMENTS),
    "--terrain-actor","--terrain-step-max-m",".08",
    "--unified-capabilities","--foundation-steps","20000000",
    "--foundation-mix","main","--teaching-schedule","balanced",
    "--growth-recipe","main-mature","--terrain-lesson","coherent-careful",
    "--force-program","gentle","--athletic-rehearsal-share",".25",
    "--temporal-vx-reward-share",".5","--verified-ascent-reallocation","0",
    "--fixed-eval-every-steps","20000000",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git",*args],cwd=REPO,text=True,stderr=subprocess.STDOUT).strip()


def source_guard() -> str:
    sha=git("rev-parse","HEAD")
    if git("branch","--show-current")!=BRANCH or git("status","--porcelain"):
        raise RuntimeError("Run only from clean "+BRANCH+" source checkout")
    # A shared editable venv may still point at an older EMU checkout.
    import emu.run
    actual=Path(emu.run.__file__).resolve()
    wanted=REPO/"emu/run.py"
    if actual!=wanted:
        raise RuntimeError(f"Wrong EMU import: {actual}, expected {wanted}")
    return sha


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda:fh.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_parent() -> None:
    """Never trust a result-folder label as proof of policy ancestry."""
    if not PARENT.is_file():
        raise FileNotFoundError("Required full optimizer checkpoint: "+str(PARENT))
    if sha256(PARENT)!=PARENT_SHA256:
        raise RuntimeError("v5032 parent final.pkl bytes do not match recorded SHA256")
    cfg=PARENT.parent.parent/"training_config.json"
    meta=PARENT.parent.parent/"run_meta.json"
    if not cfg.is_file() or not meta.is_file():
        raise RuntimeError("Parent needs original run_meta and training_config")
    c=json.loads(cfg.read_text())
    m=json.loads(meta.read_text())
    if (c.get("verified_ascent_reallocation")!=0.0
            or c.get("temporal_vx_reward_share")!=.5
            or c.get("athletic_rehearsal_frac")!=.25
            or c.get("observation_schema")!="terrain_360"
            or c.get("envs")!=ENVIRONMENTS
            or (m.get("source_provenance") or {}).get("git_commit")!=PARENT_SOURCE):
        raise RuntimeError("Parent source or effective reward settings differ")


def _python_env() -> dict[str,str]:
    py=os.environ.get("PYTHONPATH","")
    return {**os.environ,
            "PYTHONPATH":str(REPO)+(os.pathsep+py if py else ""),
            "EMU_UNIFIED_CONTINUOUS":"cold-growth-approved",
            "XLA_PYTHON_CLIENT_PREALLOCATE":"false"}


def command(arm: int, stage: str, root: Path, *, has_local_checkpoint: bool) -> list[str]:
    from emu.science import arm_name
    destination=root/"arms"/arm_name(arm)
    target=TARGET_A if stage=="a" else TARGET_B
    resume=(("auto",) if has_local_checkpoint else (str(PARENT),))
    cmd=[sys.executable,"-m","emu.cli","train",
         "--run-dir",str(destination),"--steps",str(target),
         *BASE_ARGS,"--science-arm",str(arm),
         "--resume",*resume]
    if not has_local_checkpoint:
        cmd.extend(("--research-parent-source",PARENT_SOURCE,
                    "--research-parent-sha256",PARENT_SHA256))
    return cmd


def resolved_config(cmd: list[str], env: dict[str,str]) -> dict:
    """Inspect the ACTUAL emu.run/TrainingConfig resolution, no JAX required."""
    out=subprocess.check_output(cmd+["--plan"],cwd=REPO,env=env,text=True)
    start=out.find("{")
    if start<0:
        raise RuntimeError("Train --plan produced no resolved JSON config")
    cfg=json.JSONDecoder().raw_decode(out[start:])[0]
    return cfg


def verify_arm_configuration(arm: int, cfg: dict, root: Path, stage: str) -> None:
    # Check all registered fields AND ensure the remaining recipe does not
    # drift between arms. Source, run_dir, steps and science_arm legitimately
    # differ from the previous finished v5032 case.
    exp=expected_fields(arm)
    for field,value in exp.items():
        if cfg.get(field)!=value:
            raise RuntimeError(f"arm {arm_name(arm)} {field} resolved "
                               f"{cfg.get(field)!r}, expected {value!r}")
    base={
        "observation_schema":"terrain_360",
        "foundation_steps":20_000_000,"foundation_mix":"main",
        "growth_recipe":"main-mature","terrain_lesson":"coherent-careful",
        "envs":ENVIRONMENTS,"seed":SEED,"unified_capability_focus":True,
        "open_frontier":True,"terrain_step_max_m":.08,
    }
    for key,value in base.items():
        if cfg.get(key)!=value:
            raise RuntimeError(f"{arm_name(arm)} unexpected {key}={cfg.get(key)!r}")
    if cfg.get("steps")!=(TARGET_A if stage=="a" else TARGET_B):
        raise RuntimeError("Target changed during plan")
    if Path(cfg["run_dir"]).resolve()!=root/"arms"/arm_name(arm):
        raise RuntimeError("Run output path does not match arm identity")
    if arm not in (4,5,6,7) and cfg["verified_ascent_reallocation"]!=0.:
        raise RuntimeError("Ascent reward leaked into unrelated science arm")


def write_manifest(root: Path, sha: str) -> None:
    dest=root/"science_manifest.json"
    content={
        "schema":"v5033_24_arm_science_v1","science_source_sha":sha,
        "branch":BRANCH,"parent_source_sha":PARENT_SOURCE,
        "parent_checkpoint_sha256":PARENT_SHA256,
        "parent_step":PARENT_STEP,"seed":SEED,"envs":ENVIRONMENTS,
        "target_stage_a":TARGET_A,"target_stage_b":TARGET_B,
        "arm_names":list(ARM_NAMES),
        "baseline":"v5032 actual resolved (verified_ascent_reallocation=0.0)",
        "publication_approved":False,
    }
    if dest.is_file():
        if json.loads(dest.read_text())!=content:
            raise RuntimeError("Existing science results use another source/recipe")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Nonempty science root lacks its original manifest")
        root.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(content,sort_keys=True,indent=2)+"\n")


def run_with_log(cmd: list[str], path: Path, env: dict[str,str]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    print("RUN:",subprocess.list2cmdline(cmd),flush=True)
    with path.open("a",encoding="utf-8",buffering=1) as log:
        log.write("\nRUN: "+subprocess.list2cmdline(cmd)+"\n")
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1)
        try:
            for line in proc.stdout:
                print(line,end="",flush=True)
                log.write(line)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Command exited {code}; see {path}")


def preflight(arm: int, root: Path, sha: str, env: dict[str,str]) -> None:
    """Every arm gets real 24-world CPU and full 2048-world GPU PPO smoke.

    Separate from source and checkpoint guard; do not skip merely because
    a different arm on the same source happened to compile successfully.
    """
    name=arm_name(arm)
    stamp=root/"preflights"/(name+".json")
    expected={"source_sha":sha,"arm":arm,"cpu_24":True,"cuda_2048":True}
    if stamp.is_file() and json.loads(stamp.read_text())==expected:
        print("PRE-FLIGHT RETAINED:",name,flush=True)
        return
    base=[sys.executable,"scripts/smoke_unified_focus.py",
          "--science-arm",str(arm),"--growth-recipe","main-mature",
          "--terrain-lesson","coherent-careful","--terrain-max-m",".08",
          "--force-program","gentle","--fast-rehearsal-share",".25",
          "--temporal-vx-reward-share",".5"]
    for device,envs,flag in [("cpu","24",[]),("cuda","2048",["--full-update"])]:
        run_with_log(base+["--device",device,"--envs",envs,*flag],
                     root/"logs"/f"preflight_{name}_{device}.log",env)
    stamp.parent.mkdir(parents=True,exist_ok=True)
    stamp.write_text(json.dumps(expected,indent=2,sort_keys=True)+"\n")


def exam_commands(policy: Path, location: Path, *, final: bool) -> list[list[str]]:
    # Stage A inexpensive but discriminative. Stage B/confirm uses full
    # physical matrix and speed ladder with the 16-world matched protocol.
    common=[sys.executable,"-m","emu.cli","capability-exam",
            "--policy",str(policy),"--device","cpu","--seed","1701"]
    grades=["5","8","12","14.9"] if final else ["8","12"]
    heights=["1","2","2.5","3","3.5"] if final else ["2.5","3"]
    speeds=[".1",".2",".3",".4",".5",".6",".8","1.0"] if final else [".3",".6",".8","1.0"]
    size="16" if final else "8"
    exams=[
        [*common,"--output",str(location/"terrain_matrix"),"--worlds",size,
         "--horizon","350","--speed",".15","--terrain-matrix",
         "--matrix-grades-deg",*grades,"--matrix-heights-cm",*heights],
        [*common,"--output",str(location/"speed_ladder"),"--worlds",size,
         "--horizon","700","--speed",".15","--speed-ladder",
         "--ladder-speeds",*speeds],
    ]
    if final:
        exams.append([*common,"--output",str(location/"downhill_sweep"),
                      "--worlds",size,"--horizon","350","--speed",".15",
                      "--downhill-sweep"])
    return exams


def preserve_stage_a_full_checkpoint(arm: int, root: Path) -> None:
    """Copy immutable stage A full learner state BEFORE extending to stage B.

    B legitimately replaces its own final.pkl on extension, but it must
    not destroy the stage A optimiser/curriculum evidence. Preserve by a
    full independent copy, never a mutable hardlink.
    """
    name=arm_name(arm)
    original=root/"arms"/name/"checkpoints"/"final.pkl"
    dest=root/"stage_a_finals"/(name+".pkl")
    stamp=dest.with_suffix(".sha256")
    if not original.is_file():
        raise RuntimeError("Cannot preserve missing Stage A final: "+name)
    digest=sha256(original)
    if dest.exists() or stamp.exists():
        if not dest.is_file() or not stamp.is_file() or stamp.read_text().strip()!=digest:
            raise RuntimeError("Prior Stage A backup/source hash mismatch: "+name)
        if sha256(dest)!=digest:
            raise RuntimeError("Preserved Stage A checkpoint bytes changed: "+name)
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(".tmp")
    if temp.exists():
        raise RuntimeError("Incomplete Stage A backup: "+str(temp))
    shutil.copy2(original,temp)
    if sha256(temp)!=digest:
        raise RuntimeError("Stage A copy checksum mismatch: "+name)
    temp.replace(dest)
    stamp.write_text(digest+"\n")


def export_and_examine(arm: int, stage: str, root: Path, env: dict[str,str],
                       *, no_exams: bool = False) -> None:
    name=arm_name(arm)
    final=root/"arms"/name/"checkpoints"/"final.pkl"
    if not final.is_file():
        raise RuntimeError("Full, completed final checkpoint required: "+str(final))
    policy=root/"exports"/stage/(name+".emupolicy")
    policy.parent.mkdir(parents=True,exist_ok=True)
    if not policy.exists():
        run_with_log([sys.executable,"-m","emu.cli","export-policy",
                      str(final),str(policy)],
                     root/"logs"/("export_"+stage+"_"+name+".log"),env)
    if no_exams:
        print("EXAMS SKIPPED by explicit operator option:",name)
        return
    location=root/"exams"/stage/name
    for cmd in exam_commands(policy,location,final=(stage=="b")):
        output=Path(cmd[cmd.index("--output")+1])
        result=output/"unified_exam.json"
        if result.is_file() and (output/"unified_exam.md").is_file():
            j=json.loads(result.read_text())
            if j.get("actor_artifact_sha256")!=sha256(policy):
                raise RuntimeError("Existing exam is for another exported policy: "+str(result))
            print("EXAM RETAINED:",result)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("Incomplete exam path, preserve then investigate: "+str(output))
        run_with_log(cmd,root/"logs"/(stage+"_"+name+"_"+output.name+".log"),env)


def summarize(root: Path, stage: str) -> Path:
    """Read-only transparent scorecard, no 'highest reward wins' magic."""
    rows=[]
    for arm in range(ARM_COUNT):
        name=arm_name(arm)
        cases={}
        for exam in ("speed_ladder","terrain_matrix","downhill_sweep"):
            f=root/"exams"/stage/name/exam/"unified_exam.json"
            if f.is_file():
                j=json.loads(f.read_text())
                for c in j["results"]:
                    cases[c["case"]["name"]]=c
        if not cases:
            continue
        def case(key):
            c=cases.get(key)
            if not c:return {}
            ws=c["worlds"]
            return dict(mission=c["mission_passed"],survived=c["survived"],
                        count=len(ws),
                        crossing=sum(w["crossing_pass"] is True for w in ws),
                        up=sum(w["crossing_pass"] is True for w in ws if w["direction"]=="step-up"),
                        down=sum(w["crossing_pass"] is True for w in ws if w["direction"]=="step-down"),
                        yaw_bias=sum(abs(w["tracking_signed_bias_yaw_rad_s"])
                                     for w in ws if w["tracking_signed_bias_yaw_rad_s"] is not None)/len(ws),
                        mean_vx=sum(w["achieved_mean_vx_mps"] or 0 for w in ws)/len(ws))
        row={"arm":f"{arm:02d}","name":name,"comparator":("05" if arm==7 else "00"),
             "downhill12_mission":case("downhill_12deg").get("mission",""),
             "downhill12_bias":round(case("downhill_12deg").get("yaw_bias",0),4),
             "up2p5":case("paired_steps_2p5cm").get("up",""),
             "up3":case("paired_steps_3p0cm").get("up",""),
             "down3":case("paired_steps_3p0cm").get("down",""),
             "fast08_survival":case("flat_vx_0p80").get("survived",""),
             "fast10_survival":case("flat_vx_1p00").get("survived",""),
             "fast08_actual":round(case("flat_vx_0p80").get("mean_vx",0),3),
             "exams_present":len(cases)}
        rows.append(row)
    if not rows:
        raise RuntimeError("No completed independent exams yet; run --run first")
    target=root/("science_"+stage+"_scorecard.csv")
    with target.open("w",newline="") as fp:
        out=csv.DictWriter(fp,fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)
    md=["# v5033 "+stage.upper()+" — physical results (do not rank by PPO reward)",
        "","All cases use identical frozen seeds, physics, world count and age.",
        "06/07 etc are not independent seed replications; 07 compares to 05.",
        "",
        "| Arm | 12° downhill mission | 12° mean abs yaw bias | 2.5cm UP | 3cm UP | 0.8m/s survived |",
        "|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        md.append(f"| {x['name']} | {x['downhill12_mission']} | "
                  f"{x['downhill12_bias']:.4f} | {x['up2p5']} | {x['up3']} | "
                  f"{x['fast08_survival']} |")
    md.extend(["","All cells are raw 8- or 16-world counts, NOT percentages.",
               "No automatic winner: inspect terrain safety, yaw jitter and "
               "mechanical J/m before selecting finalists."])
    (root/("science_"+stage+"_scorecard.md")).write_text("\n".join(md)+"\n")
    print("SCORECARD:",target,flush=True)
    return target


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",action="store_true",help="execute, not just print validated plans")
    ap.add_argument("--stage",choices=("a","b"),default="a")
    ap.add_argument("--arms",nargs="+",type=int,default=None,
                    help="IDs 0..23; default all arms in A; MUST specify up to six for B")
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--no-exams",action="store_true",
                    help="do not run independent exams (only if diagnosing broken exam setup)")
    ap.add_argument("--summary-only",action="store_true")
    args=ap.parse_args()
    arms=tuple(range(ARM_COUNT)) if args.arms is None and args.stage=="a" else tuple(args.arms or ())
    if args.summary_only:
        summarize(args.root.expanduser().resolve(),args.stage)
        return 0
    if not arms or len(arms)!=len(set(arms)) or any(not 0<=v<ARM_COUNT for v in arms):
        ap.error("Choose distinct arm IDs 0..23")
    if args.stage=="b" and len(arms)>6:
        ap.error("Stage B must name at most six confirmed finalists")
    sha=source_guard()
    root=args.root.expanduser().resolve()
    if root==REPO or REPO in root.parents:
        ap.error("Results belong OUTSIDE Git")
    immutable_parent_root=PARENT.parent.parent.resolve()
    if (root==immutable_parent_root or
            immutable_parent_root in root.parents or root in immutable_parent_root.parents):
        ap.error("Science root must not include or overwrite v5032 parent run")
    env=_python_env()
    print("V5033",args.stage.upper(),"source SHA",sha,"parent",PARENT_SOURCE,flush=True)
    print("Parent checkpoint:",PARENT,"(sha256",PARENT_SHA256+")",flush=True)
    print("Arms:",", ".join(arm_name(x) for x in arms),flush=True)
    if not args.run:
        for arm in arms:
            dst=root/"arms"/arm_name(arm)
            local=(dst/"checkpoints"/"latest.pkl").exists() or (
                dst/"checkpoints"/"final.pkl").exists()
            cfg=resolved_config(command(arm,args.stage,root,has_local_checkpoint=local),env)
            verify_arm_configuration(arm,cfg,root,args.stage)
            print("PLAN:",arm_name(arm), "effective verified",
                  cfg["verified_ascent_reallocation"],"temporal",
                  cfg["temporal_vx_reward_share"],"target",cfg["steps"])
        print("PLAN ONLY: no files, GPU training or exams started.",flush=True)
        return 0
    ensure_parent()
    write_manifest(root,sha)
    for arm in arms:
        source_guard()
        name=arm_name(arm)
        dest=root/"arms"/name
        local=(dest/"checkpoints"/"latest.pkl").is_file() or (
            dest/"checkpoints"/"final.pkl").is_file()
        if args.stage=="b":
            if not (dest/"checkpoints"/"final.pkl").is_file():
                raise RuntimeError("Stage B requires COMPLETED own Stage A final.pkl: "+name)
            preserve_stage_a_full_checkpoint(arm,root)
        cmd=command(arm,args.stage,root,has_local_checkpoint=local)
        cfg=resolved_config(cmd,env)
        verify_arm_configuration(arm,cfg,root,args.stage)
        # Inspect the actual cross-source optimizer and checkpoint before
        # spending GPU time; explicit parent source and bytes must match.
        run_with_log(cmd+["--check-resume-only"],
                     root/"logs"/("source_gate_"+args.stage+"_"+name+".log"),env)
        preflight(arm,root,sha,env)
        run_with_log(cmd,root/"logs"/("train_"+args.stage+"_"+name+".log"),env)
        export_and_examine(arm,args.stage,root,env,no_exams=args.no_exams)
        print("COMPLETE:",name,args.stage.upper(),flush=True)
        if not args.no_exams:
            summarize(root,args.stage)
    print("V5033 STAGE COMPLETE. Public main remains unchanged.",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
