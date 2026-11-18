"""Freeze the proven 120M Spring V1 actor, then run paired *frozen* MJX exams.

No PPO updates, physics changes, reward shaping or source mutation. Results
come from the same original saved actor on true prisms and flat seam controls.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT=Path(__file__).resolve().parent.parent
ORIGIN=Path.home()/"emu-data/runs/h40-spring-v1-outcomes-s735-repaired"
REFERENCE=Path.home()/"emu-data/actors/h40-spring-v1-120m-reference"
OUTPUT=Path.home()/"emu-data/exams/h40-spring-v1-mechanics-s735"
EXPECTED_ACTOR_SHA="26095f7f10084c815ee1f7ad24363834cdd017d4341bca9ec1d9f10e4372b16c"
EXPECTED_MODEL_SHA="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
EXPECTED_SOURCE="a7c6467c59f81c124cd76e3348a4f12baa6f438e"
POLICY_NAME="h40-spring-v1-outcomes-120m.emupolicy"


def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=ROOT,text=True,stderr=subprocess.PIPE).strip()


def sha256(path:Path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for blob in iter(lambda:stream.read(1024*1024),b""):
            digest.update(blob)
    return digest.hexdigest()


def immutable_copy(source:Path,target:Path):
    if target.is_file():
        if sha256(source)!=sha256(target):
            raise RuntimeError(f"Frozen reference differs; refusing overwrite: {target}")
        return
    if target.exists():
        raise RuntimeError(f"Reference target is not an ordinary file: {target}")
    temporary=target.with_name("."+target.name+".copying")
    if temporary.exists():
        raise RuntimeError(f"Partial prior copy exists: {temporary}")
    try:
        shutil.copy2(source,temporary)
        if sha256(source)!=sha256(temporary):
            raise RuntimeError("Checkpoint or actor changed while copying")
        os.replace(temporary,target)
        target.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def preserve():
    from emu.contracts import ROBOT_XML
    from emu.training_config import TrainingConfig,training_sensor_contract
    from emu.checkpoint_io import load_checkpoint
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    if os.environ.get("EMU_PHYSICAL_PROFILE")!="h40-spring-v1":
        raise RuntimeError("Must examine exact h40-spring-v1 physics")
    if git("branch","--show-current")!="work/emu-stair-mechanics-diagnostics":
        raise RuntimeError("Use one diagnostic experiment branch, not main or original training branch")
    if git("status","--porcelain"):
        raise RuntimeError("Dirty source: commit or stash before examining")
    if sha256(ROBOT_XML)!=EXPECTED_MODEL_SHA:
        raise RuntimeError("The Spring V1 physical robot unexpectedly changed")
    original=ORIGIN/POLICY_NAME
    checkpoint=ORIGIN/"checkpoints/final.pkl"
    config_path=ORIGIN/"training_config.json"
    meta_path=ORIGIN/"run_meta.json"
    for p in (original,checkpoint,config_path,meta_path):
        if not p.is_file():
            raise RuntimeError(f"Missing original saved 120M evidence: {p}")
    if sha256(original)!=EXPECTED_ACTOR_SHA:
        raise RuntimeError("Wrong actor: the known 120M Spring V1 frozen policy hash differs")
    cfg=TrainingConfig.load(config_path)
    if (cfg.physical_profile!="h40-spring-v1" or cfg.envs!=2048
            or cfg.seed!=735 or cfg.steps!=120_000_000
            or cfg.gait_guidance_fade_steps!=20_000_000):
        raise RuntimeError("Wrong 120M source training config")
    run_meta=json.loads(meta_path.read_text(encoding="utf-8"))
    origin_sha=(run_meta.get("source_provenance") or {}).get("git_commit")
    if origin_sha!=EXPECTED_SOURCE:
        raise RuntimeError("Original actor was not trained under the expected source")
    payload=load_checkpoint(checkpoint,validate=True,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    ckmeta=payload["meta"]
    if (int(ckmeta["step"])!=120_061_952 or
            ckmeta.get("robot_model_sha256")!=EXPECTED_MODEL_SHA or
            (ckmeta.get("source_provenance") or {}).get("git_commit")!=EXPECTED_SOURCE):
        raise RuntimeError("Final PPO checkpoint is not the validated 120M actor lineage")
    REFERENCE.mkdir(parents=True,exist_ok=True)
    immutable_copy(original,REFERENCE/POLICY_NAME)
    immutable_copy(checkpoint,REFERENCE/"final.pkl")
    manifest=dict(
        actor_sha256=sha256(REFERENCE/POLICY_NAME),
        checkpoint_sha256=sha256(REFERENCE/"final.pkl"),
        robot_model_sha256=EXPECTED_MODEL_SHA,
        actor_training_source=EXPECTED_SOURCE,
        diagnostic_source=git("rev-parse","HEAD"),
        exam_source_sha256=exam_source_fingerprint(),
        original_run=str(ORIGIN),
        checkpoint_step=120_061_952,
        checkpoint_optimizer_and_normalizer_saved=True,
        no_ppo_updates=True)
    manifest_path=REFERENCE/"reference_manifest.json"
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        # Source commit is allowed to evolve BEFORE first exam, but once a
        # reference manifest exists all fields are immutable.
        if existing!=manifest:
            raise RuntimeError("Frozen reference manifest conflicts with current source; refusing")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("PRESERVED FULL 120M ACTOR+OPTIMISER:",REFERENCE,flush=True)
    return manifest


def validate_existing(output,manifest,mode):
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    summary=output/"matrix_summary.json"
    if not summary.is_file():
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Partial exam exists: {output}; refusing stale/ambiguous results")
        return False
    got=json.loads(summary.read_text(encoding="utf-8"))["manifest"]
    wanted=dict(mode=mode,device="gpu",worlds=2048,
        assigned_worlds=2000,horizon=350,seed=1701,
        policy_sha256=manifest["actor_sha256"],
        robot_model_sha256=EXPECTED_MODEL_SHA,
        exam_source_sha256=exam_source_fingerprint())
    for k,v in wanted.items():
        if got.get(k)!=v:
            raise RuntimeError(f"{output}: incompatible existing exam {k}: {got.get(k)} vs {v}")
    if not (output/"matrix_worlds.csv").is_file():
        raise RuntimeError("Incomplete matrix: worlds CSV missing")
    return True


def exam(mode,manifest):
    directory=OUTPUT/("stairs" if mode=="stairs" else "flat")
    if validate_existing(directory,manifest,mode):
        print("REUSING authenticated frozen exam:",directory,flush=True)
        return
    command=[
        sys.executable,str(ROOT/"scripts/v5053_matrix_exam.py"),
        "--policy",str(REFERENCE/POLICY_NAME),
        "--output",str(directory),
        "--mode",mode,"--worlds","2048","--horizon","350",
        "--seed","1701","--device","cuda",
    ]
    if mode=="stairs":
        command.extend(("--levels","20","--trials-per-level","50",
                        "--max-step-m",".08"))
    else:
        command.extend(("--trials-per-level","2000",))
    print(f"READ-ONLY PHYSICAL EXAM: {mode}, 2000 first-life worlds",flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    manifest=preserve()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    exam("stairs",manifest)
    exam("flat",manifest)
    subprocess.run([
        sys.executable,str(ROOT/"scripts/summarize_stair_mechanics.py"),
        "--stairs",str(OUTPUT/"stairs"),
        "--flat",str(OUTPUT/"flat"),
        "--out",str(OUTPUT),
    ],cwd=ROOT,check=True)
    out=OUTPUT/"mechanics_results_bundle.zip"
    files=[REFERENCE/"reference_manifest.json",REFERENCE/POLICY_NAME]
    files.extend(p for p in OUTPUT.rglob("*") if p.is_file()
                 and p!=out and p.suffix in (".csv",".json",".md"))
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            name=("reference/"+p.name if p.parent==REFERENCE else
                  p.relative_to(OUTPUT).as_posix())
            archive.write(p,arcname=name)
    print("UPLOAD THIS ZIP:",out,flush=True)
    print("ORIGINAL RUN UNTOUCHED:",ORIGIN,flush=True)


if __name__=="__main__":
    main()
