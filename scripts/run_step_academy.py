"""v5037 Step Academy STAGE A: honest eight-arm SINGLE-LIP science screen.

All arms rewind the SAME immutable v5033 280M full PPO state.  The six-lip
course and staged verified-milestone rewards are NOT implemented in Stage A:
this script explicitly refuses to describe this screen as a multi-lip run.

The default run starts from v5033 arm00 control; no policy is promoted.
Run --matrix, --verify-only --all, then --run --all on a clean dedicated branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Invoking "python scripts/run_step_academy.py" sets sys.path[0] to scripts/,
# not the checkout root. A shared venv can have an editable "emu" package
# pointing at another Git worktree. Use this checkout's code explicitly.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
BRANCH = "feature/v5037-step-academy-stage-a"
PARENT = (Path.home() / "robot_lab/emu_runs/v5033_plateau_260_to_280/"
          "arm_00_control_seed761/checkpoints/final.pkl")
ROOT = Path.home() / "robot_lab/emu_runs/v5037_step_academy_stage_a"
PARENT_SOURCE = "52736f4d75b053f2bab57cfaa2b5f4061b59f426"
PARENT_STEP = 280_035_328
TARGET_STEP = 290_000_000

@dataclass(frozen=True)
class Arm:
    index: int
    name: str
    changes: dict
    hypothesis: str

    @property
    def directory(self):
        return f"arm_{self.index:02d}_{self.name}"


# A(01) vs A(02) isolates shorter exposure loops.
# A(02) vs A(03) isolates easy 2 cm nursery.
# A(02) vs A(05) isolates verified upper-support shaping.
# A(02) vs A(06) isolates stride/approach speed variation.
# A(02) vs A(07) isolates mild obliqueness.
BASE = dict(academy_step_focus_share=.85, academy_step_exposure=1.,
            academy_edge_yaw_max_deg=3.0, athletic_rehearsal_frac=0.)

def arms() -> tuple[Arm, ...]:
    return (
        Arm(0,"control",{}, "same source and original single-lip training"),
        Arm(1,"dense_step",dict(BASE),
            "85% STEP focus, real lips every STEP birth, ±3° edge"),
        Arm(2,"rapid_repeats",{**BASE,"academy_episode_frames":300},
            "shorter repeat single-lip episodes, same reward"),
        Arm(3,"easy_nursery",{**BASE,"academy_episode_frames":300,
                            "academy_step_height_cap_m":.020},
            "small steps teach a successful upward foothold first"),
        Arm(4,"pure_step",{**BASE,"academy_step_focus_share":1.,
                          "academy_episode_frames":300},
            "extreme 100% specialist, monitor catastrophic forgetting"),
        Arm(5,"verified_upper",{**BASE,"academy_episode_frames":300,
                                "verified_ascent_reallocation":3.},
            "move existing ascent shaping to genuine elevated-foot support"),
        Arm(6,"approach_rhythm",{**BASE,"academy_episode_frames":300,
                                 "academy_step_speed_span":.11},
            "vary straight approach vx 0.05..0.27 rather than 0.10..0.22"),
        Arm(7,"square_control",{**BASE,"academy_episode_frames":300,
                                "academy_edge_yaw_max_deg":0.},
            "matched square-on control for ±3° edge obliqueness"),
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def source_guard() -> dict:
    # run_training starts "python -c ..." in the caller's working directory.
    # Require that directory to be this checkout so the PPO subprocess uses
    # the exact same physical simulator as source-pinned preflight.
    if Path.cwd().resolve() != REPO:
        raise RuntimeError(
            f"Run Academy from its source checkout: cd {REPO}; "
            f"current directory is {Path.cwd().resolve()}")
    from emu import contracts
    from emu.checkpoint_io import git_provenance
    loaded = Path(contracts.__file__).resolve().parents[1]
    if loaded != REPO or contracts.source_repo_root() != REPO:
        raise RuntimeError(
            f"Wrong emu package: loaded {loaded}, expected Academy checkout {REPO}; "
            "refusing research training with a mismatched Python import")
    branch = git("branch", "--show-current")
    sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if branch != BRANCH or dirty or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"Expected clean {BRANCH}; branch={branch} dirty={bool(dirty)}")
    if sha == PARENT_SOURCE:
        raise RuntimeError("Study must not overwrite its immutable training ancestor")
    evidence = git_provenance()
    if (not evidence.get("available") or evidence.get("dirty") or
        evidence.get("commit") != sha or evidence.get("branch") != branch):
        raise RuntimeError(
            f"Git provenance mismatch: checkout {branch}@{sha[:12]}, "
            f"loaded emu {evidence.get('branch')}@"
            f"{str(evidence.get('commit',''))[:12]} "
            f"available={evidence.get('available')} dirty={evidence.get('dirty')}")
    return evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def parent_contract(path: Path):
    from emu.checkpoint_io import load_checkpoint
    from emu.training_config import TrainingConfig, training_sensor_contract
    if path.name!="final.pkl" or path.parent.name!="checkpoints":
        raise RuntimeError("Research parent must be checkpoints/final.pkl")
    cfg = TrainingConfig.load(path.parent.parent / "training_config.json")
    payload = load_checkpoint(path, expected_observation_schema=cfg.observation_schema,
                              expected_sensor_contract=training_sensor_contract(cfg))
    meta=payload["meta"]
    actual_source=(meta.get("source_provenance") or {}).get("git_commit")
    if actual_source!=PARENT_SOURCE or int(meta["step"])!=PARENT_STEP:
        raise RuntimeError("Wrong full-state parent (source or step); expected v5033 280M arm 00")
    if (cfg.observation_schema!="terrain_360" or not cfg.unified_capability_focus
        or cfg.terrain_lesson!="coherent-careful" or cfg.envs!=2048):
        raise RuntimeError("Parent lacks the required 360-input unified physical training")
    return cfg,sha256(path)


def config_for(cfg,arm:Arm,root:Path,target:int):
    return replace(cfg,run_dir=str(root/arm.directory),steps=target,
                   resume="",device="cuda",seed=761,envs=2048,
                   fixed_eval_every_steps=5_000_000,
                   science_reset_step_evidence_on_fork=True,**arm.changes)


def manifest(source:dict,parent:Path,digest:str,cfg,root:Path,target:int):
    desired={
        "schema":1,"study":"v5037-stage-a-single-lip",
        "source_commit":source["commit"],"source_branch":source["branch"],
        "parent_source":PARENT_SOURCE,"parent_step":PARENT_STEP,
        "parent_path":str(parent),"parent_sha256":digest,
        "steps_absolute":target,
        "limitations":"No consecutive multi-lip geometry or new milestone reward in stage A",
        "arms":[dict(index=a.index,name=a.directory,hypothesis=a.hypothesis,
                    changes=a.changes,
                    resolved_config=asdict(config_for(cfg,a,root,target)))
                for a in arms()]
    }
    return json.loads(json.dumps(desired,sort_keys=True))


def ensure_manifest(root:Path,wanted:dict):
    name=root/"campaign_manifest.json"
    if name.is_file():
        saved=json.loads(name.read_text(encoding="utf-8"))
        if saved!=wanted:
            raise RuntimeError("Campaign identity/lineage changed; choose a NEW --root")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Nonempty root without valid manifest; refusing takeover")
    root.mkdir(parents=True,exist_ok=True)
    name.write_text(json.dumps(wanted,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def exists_run(root:Path):
    if not root.exists():
        return False
    ck=root/"checkpoints"
    if ck.is_dir() and any(ck.glob("*.pkl")):
        return True
    unexpected={p.name for p in root.iterdir()}-{"training_config.json","console.log"}
    if unexpected:
        raise RuntimeError(f"Ambiguous run without checkpoint: {root}: {sorted(unexpected)}")
    return False


def preflight(conf,parent:Path,digest:str,provenance:dict):
    from emu.run import inspect_run
    current=exists_run(Path(conf.run_dir))
    ck,done=inspect_run(conf,provenance=provenance,
        resume="auto" if current else str(parent),
        research_parent_source="" if current else PARENT_SOURCE,
        research_parent_sha256="" if current else digest)
    if ck is None:
        raise RuntimeError("No validated resume parent; refusing cold restart")
    return ck,done,current


def run_arm(conf,arm:Arm,parent:Path,digest:str,provenance:dict):
    from emu.checkpoint_io import run_training, resolve_target_checkpoint
    from emu.training_config import training_sensor_contract
    ck,done,current=preflight(conf,parent,digest,provenance)
    if not done:
        actual=replace(conf,resume=str(ck))
        import os
        os.environ["EMU_UNIFIED_CONTINUOUS"]="cold-growth-approved"
        print(f"\nACADEMY {arm.index:02d}/07 {arm.directory}: "
              f"{PARENT_STEP:,}->{conf.steps:,} from {'own checkpoint' if current else 'same 280M parent'}",
              flush=True)
        run_training(actual,provenance=provenance)
    final,_,_=resolve_target_checkpoint(
        Path(conf.run_dir),target_steps=conf.steps,envs=conf.envs,
        label=arm.directory,expected_observation_schema=conf.observation_schema,
        expected_sensor_contract=training_sensor_contract(conf))
    if final is None:
        raise RuntimeError(f"No validated final checkpoint for {arm.directory}")
    print(f"VALIDATED FINAL {arm.index:02d}: {final}",flush=True)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=PARENT)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--target-steps",type=int,default=TARGET_STEP)
    parser.add_argument("--arm",type=int,choices=range(8))
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--matrix",action="store_true")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.arm is not None and args.all:
        parser.error("Choose --arm N or --all")
    chosen=arms() if args.all else (arms()[args.arm or 0],)
    if args.matrix:
        print("v5037 Stage A: eight SINGLE-LIP experiments; multi-lip Stage B not built")
        for arm in arms():
            print(f" {arm.index:02d} {arm.directory}: {arm.hypothesis}")
        return 0
    if not PARENT_STEP<args.target_steps<=300_000_000:
        raise RuntimeError("Choose absolute target >280,035,328 and <=300M")
    source=source_guard()
    parent=args.parent.expanduser().resolve()
    root=args.root.expanduser().resolve()
    cfg,digest=parent_contract(parent)
    wanted=manifest(source,parent,digest,cfg,root,args.target_steps)
    print(f"ACADEMY SOURCE {source['commit']} clean; parent {parent} "
          f"sha256={digest}; {len(chosen)} arms, target {args.target_steps:,}",
          flush=True)
    # Validate all selected arms before writing a single file or using GPU.
    for arm in chosen:
        conf=config_for(cfg,arm,root,args.target_steps)
        ck,done,_=preflight(conf,parent,digest,source)
        print(f"PREFLIGHT {arm.index:02d}: checkpoint={ck} complete={done}",flush=True)
    print("ACADEMY PREFLIGHT PASS",flush=True)
    if args.verify_only or not args.run:
        if not args.verify_only:
            print("DRY RUN: add --run to train")
        return 0
    ensure_manifest(root,wanted)
    for arm in chosen:
        run_arm(config_for(cfg,arm,root,args.target_steps),
                arm,parent,digest,source)
    print("STAGE A COMPLETE: run independent physical exams before Stage B",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
