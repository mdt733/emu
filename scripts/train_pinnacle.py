#!/usr/bin/env python3
"""Build the next EMU Pinnacle CANDIDATE, never publish it implicitly.

One source-pinned, 360-input, 260M-step COLD start. Plan-only by default.
The experiment is single-run; no cross-source checkpoint transplant and
no legacy A/B launcher. The source SHA, manifests, CPU + real 2048 CUDA
preflight, policy export, frozen physical exams and provenance are required.

The 20M main-style foundation, 25% genuine fast-flat rehearsal and two
confirmed reward discoveries are explicitly pinned, not latest defaults.
Published main/Pinnacle remains untouched pending graduation review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import campaign_support as support
from verify_showcase_release import validate_browser_bundle

REPO = Path(__file__).resolve().parents[1]
BRANCH = "release/v5032-pinnacle-candidate"
SEED = 761
ENVIRONMENTS = 2048
TARGET_STEPS = 260_000_000
FOUNDATION = 20_000_000
REHEARSAL = .25
TEMPORAL_REWARD = .5
ASCENT_REALLOCATION = 3.
DEFAULT_ROOT = Path.home()/"robot_lab/emu_runs/v5032_pinnacle_s761_v2"
CASE = {"name":"pinnacle_cold260","recipe":"main",
        "schedule":"balanced","steps":TARGET_STEPS,"from_steps":0}
EXAMS = {
    "speed_ladder": ["--speed-ladder","--worlds","16","--horizon","700",
                     "--speed",".15","--ladder-speeds",
                     ".1",".2",".3",".4",".5",".6",".8","1.0"],
    "terrain_matrix": ["--terrain-matrix","--worlds","16","--horizon","350",
                       "--speed",".15","--matrix-grades-deg","5","8","12","14.9",
                       "--matrix-heights-cm","1","2","2.5","3","3.5"],
}
MILESTONES = [20,60,100,160,200,260]


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def source_guard(sha: str) -> None:
    if (support.git("branch","--show-current") != BRANCH or
            support.git("rev-parse","HEAD") != sha or
            support.git("status","--porcelain")):
        raise RuntimeError("Source/branch changed or worktree dirty: "
                           "do not touch source-pinned Pinnacle research")


def command(root: Path, *, resume: bool=False) -> list[str]:
    args=[
        sys.executable,"-m","emu.cli","train",
        "--run-dir",str(root/CASE["name"]),
        "--profile","sim2real","--device","cuda",
        "--seed",str(SEED),"--envs",str(ENVIRONMENTS),
        "--steps",str(TARGET_STEPS),
        "--terrain-actor","--terrain-step-max-m",".08",
        "--unified-capabilities","--foundation-steps",str(FOUNDATION),
        "--foundation-mix","main","--teaching-schedule","balanced",
        "--growth-recipe","main-mature",
        "--terrain-lesson","coherent-careful","--force-program","gentle",
        "--athletic-rehearsal-share",str(REHEARSAL),
        "--temporal-vx-reward-share",str(TEMPORAL_REWARD),
        "--verified-ascent-reallocation",str(ASCENT_REALLOCATION),
        "--fixed-eval-every-steps","20000000"]
    if resume:
        args.extend(("--resume","auto"))
    return args


def smoke(device: str, *, full: bool=False) -> list[str]:
    args=[sys.executable,"scripts/smoke_unified_focus.py",
          "--device",device,"--envs",str(ENVIRONMENTS if full else 24),
          "--terrain-max-m",".08","--growth-recipe","main-mature",
          "--terrain-lesson","coherent-careful","--force-program","gentle",
          "--fast-rehearsal-share",str(REHEARSAL),
          "--temporal-vx-reward-share",str(TEMPORAL_REWARD),
          "--verified-ascent-reallocation",str(ASCENT_REALLOCATION)]
    if full:
        args.append("--full-update")
    return args


def manifest(sha: str) -> dict:
    return {
      "schema":1,"project":"EMU new Pinnacle candidate (no public promotion)",
      "source_sha":sha,"branch":BRANCH,"cold_start":True,
      "source_parent_checkpoint":None,"seed":SEED,"envs":ENVIRONMENTS,
      "target_steps":TARGET_STEPS,"foundation_steps":FOUNDATION,
      "fixed_eval_interval":20_000_000,"checkpoints_to_review_M":MILESTONES,
      "obs_schema":"terrain_360","output_policy":"pinnacle_candidate_260M.emupolicy",
      "training":{
        "growth_recipe":"main-mature","foundation_mix":"main",
        "teaching_schedule":"balanced","terrain_lesson":"coherent-careful",
        "force_program":"gentle","terrain_step_max_m":.08,
        "athletic_rehearsal_share":REHEARSAL,
        "temporal_vx_reward_share":TEMPORAL_REWARD,
        "verified_ascent_reallocation":ASCENT_REALLOCATION,
        "retention_share":0.0},
      "release_authorisation":False,
    }


def assert_import_from_checkout(env: dict) -> None:
    code=("import importlib.util,pathlib,sys\n"
          "root=pathlib.Path(sys.argv[1]).resolve()\n"
          "for name in ('emu.cli','emu.run','emu.environment','emu.training'):\n"
          "    spec=importlib.util.find_spec(name)\n"
          "    actual=pathlib.Path(spec.origin).resolve() if spec else None\n"
          "    expected=root.joinpath(*name.split('.')).with_suffix('.py')\n"
          "    assert actual==expected,(name,str(actual),str(expected))\n"
          "print('PINNACLE SOURCE IMPORT PASS:',root,flush=True)\n")
    subprocess.run([sys.executable,"-c",code,str(REPO)],cwd=REPO,env=env,check=True)


def training_environment() -> dict[str, str]:
    """Authorize only the CLI's documented continuous cold-start gate.

    This is distinct from EMU_UNIFIED_ACCEPTANCE=bounded-ppo-lifecycle,
    which is set by the standalone bounded CUDA smoke. The v5032 launch
    must never pass preflight and then fail because it set another key.
    """
    existing=os.environ.get("PYTHONPATH","")
    return {**os.environ,
            "PYTHONPATH":str(REPO)+(os.pathsep+existing if existing else ""),
            "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
            "EMU_UNIFIED_CONTINUOUS":"cold-growth-approved"}


def generate_candidate(root: Path, sha: str, env: dict) -> None:
    run=root/CASE["name"]
    final=run/"checkpoints"/"final.pkl"
    if not final.is_file():
        raise RuntimeError("No complete final checkpoint; cannot export candidate")
    policy=root/"pinnacle_candidate_260M.emupolicy"
    source_guard(sha)
    if not policy.exists():
        support.execute([sys.executable,"-m","emu.cli","export-policy",
                         str(final),str(policy)],root/"logs"/"export.log",env)
    bundle=json.loads(policy.read_text(encoding="utf-8"))
    identity=validate_browser_bundle(bundle)
    if (identity["schema"]!=3 or identity["obs_dim"]!=360 or
            identity["checkpoint_schema"]!=9 or
            identity["step"] < TARGET_STEPS or
            identity["training_source_commit"]!=sha):
        raise RuntimeError("Export provenance/step/360 sensor contract differs")
    evidence={
      "status":"RESEARCH CANDIDATE ONLY, NOT APPROVED",
      "source_sha":sha,"seed":SEED,"envs":ENVIRONMENTS,
      "checkpoint_path":str(final),"checkpoint_sha256":file_hash(final),
      "policy_path":str(policy),"policy_sha256":file_hash(policy),
      "validated_browser_bundle":identity,
      "release_approved":False,
    }
    proof=root/"candidate_provenance.json"
    if proof.exists() and json.loads(proof.read_text())!=evidence:
        raise RuntimeError("Prior candidate provenance changed: do not overwrite")
    proof.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")

    for name,flags in EXAMS.items():
        output=root/"physical_exams"/name
        if (output/"unified_exam.json").is_file() and (output/"unified_exam.md").is_file():
            print("RETAINED finished independent exam:",output,flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("Incomplete exam directory: "+str(output))
        source_guard(sha)
        support.execute([sys.executable,"-m","emu.cli","capability-exam",
                         "--policy",str(policy),"--output",str(output),
                         "--device","cpu",*flags],
                        root/"logs"/("exam_"+name+".log"),env)
    print("PHYSICAL EXAMS READY: review BOTH exams before promotion",flush=True)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",action="store_true",
                   help="preflight and train in FOREGROUND; no tmux or daemon")
    p.add_argument("--skip-preflight",action="store_true",
                   help="resume only if SAME source/recipe completed CPU + 2048 CUDA checks")
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT,
                   help="new results directory outside Git; immutable after first run")
    args=p.parse_args()
    sha=support.git("rev-parse","HEAD")
    source_guard(sha)
    root=args.root.expanduser().resolve()
    if root==REPO or REPO in root.parents:
        p.error("Results cannot be placed inside the source checkout")
    manifest_value=manifest(sha)
    print("EMU NEXT PINNACLE — COLD 360 actor, same-source single lineage",flush=True)
    print("SOURCE:",sha,"BRANCH:",BRANCH,flush=True)
    print("20M foundation, continuing to 260M; fixed exam every 20M",flush=True)
    print("Core recipe: temporal 0.5, verified support 3, fast-flat 25%, "
          "coherent-careful step lessons",flush=True)
    print("RUN ROOT:",root,flush=True)
    print("LAUNCH:", " ".join(command(root)),flush=True)
    if not args.run:
        print("PLAN ONLY: no GPU, files or checkpoints touched. Use --run.",flush=True)
        return 0
    root.mkdir(parents=True,exist_ok=True)
    mf=root/"campaign_manifest.json"
    if mf.exists():
        if json.loads(mf.read_text(encoding="utf-8"))!=manifest_value:
            raise RuntimeError("Manifest/source/recipe mismatch; choose a NEW --root")
    elif any(root.iterdir()):
        raise RuntimeError("Non-empty result root has no Pinnacle manifest")
    else:
        mf.write_text(json.dumps(manifest_value,indent=2,sort_keys=True)+"\n")
    env=training_environment()
    assert_import_from_checkout(env)
    run=root/CASE["name"]
    resume=support.has_checkpoint(run)
    if run.exists() and any(run.iterdir()) and not resume:
        raise RuntimeError("Existing run has state but no validated checkpoint; stop for recovery")
    support.execute(command(root,resume=resume)+["--plan"],
                    root/"logs"/"resolved_train_plan.log",env)
    # The central source guard runs on actual checkpoint material BEFORE
    # expensive CUDA preflights, not only once the trainer starts.
    support.execute(command(root,resume=resume)+["--check-resume-only"],
                    root/"logs"/"checkpoint_precheck.log",env)
    stamp=root/"preflight_pass.json"
    expected={"source_sha":sha,"recipe_sha256":hashlib.sha256(
        json.dumps(manifest_value,sort_keys=True).encode()).hexdigest(),
        "envs":ENVIRONMENTS,"cpu_24":True,"cuda_2048_full_update":True}
    try:
        if args.skip_preflight:
            if not stamp.is_file() or json.loads(stamp.read_text())!=expected:
                raise RuntimeError("No matching source+2048 CUDA preflight: omit --skip-preflight")
        else:
            source_guard(sha)
            support.execute([sys.executable,"-m","pytest","-q",
                             "tests/test_v5031_verified_ascent.py"],
                            root/"logs"/"verified_support_tests.log",env)
            for device,full in [("cpu",False),("cuda",True)]:
                source_guard(sha)
                support.execute(smoke(device,full=full),
                                root/"logs"/("preflight_"+device+".log"),env)
            stamp.write_text(json.dumps(expected,indent=2,sort_keys=True)+"\n")
        support.collect_runtime(root)
        source_guard(sha)
        support.execute(command(root,resume=resume),
                        root/"logs"/"train.log",env,campaign=(root,[CASE]))
        generate_candidate(root,sha,env)
    except KeyboardInterrupt:
        print("Interrupted: use the SAME exact SHA with --run --skip-preflight",
              file=sys.stderr)
        return 130
    finally:
        support.summary(root,[CASE],sha)
        report=root/"overnight_results.zip"
        with zipfile.ZipFile(report,"a",compression=zipfile.ZIP_DEFLATED) as archive:
            for fname in ("candidate_provenance.json",):
                path=root/fname
                if path.is_file():archive.write(path,arcname=fname)
            for name in EXAMS:
                for fname in ("unified_exam.json","unified_exam.md"):
                    path=root/"physical_exams"/name/fname
                    if path.is_file():
                        archive.write(path,arcname="physical_exams/"+name+"/"+fname)
        print("RESEARCH BUNDLE:",report,flush=True)
        print("Published main/showcase: UNCHANGED",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
