#!/usr/bin/env python3
"""EMU v5033: source-pinned 260M -> 280M plateau continuation science.

Plan-only by default. One COMMON checkpoint, one committed research source,
24 independently recorded run directories; never touches the 260M release.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BRANCH = "feature/v5033-260m-plateau-science"
PARENT_SOURCE = "9b11bbe596aea7201d29a6beb2d192922f8e2e9a"
PARENT_STEP = 260_046_848
EXPECTED_PARENT_SHA256 = "238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b"
TARGET_STEPS = 280_000_000
DEFAULT_ROOT = Path.home() / "robot_lab/emu_runs/v5033_plateau_260_to_280"
FACTOR_NAMES = (
    "step_promotion_1mm", "step_retreat_1mm", "step_attempts_12",
    "verified_ascent_3", "step_crouch_4cm", "fast_rehearsal_40",
    "temporal_vx_80", "steering_attention",
)
# Resolution-IV 2^(8-4) fractional factorial. Columns are pairwise balanced;
# no main effect aliases another main effect or a two-factor interaction.
FACTOR_MASKS = (1, 2, 4, 8, 7, 11, 13, 14)
FACTOR_VALUES = (
    ("science_step_promotion_mm", 1.0),
    ("science_step_retreat_mm", 1.0),
    ("science_step_min_attempts", 12),
    ("verified_ascent_reallocation", 3.0),
    ("science_step_crouch_tolerance_m", .04),
    ("athletic_rehearsal_frac", .40),
    ("temporal_vx_reward_share", .80),
    ("teaching_schedule", "steering"),
)
# Additional, individually identifiable tests. Not part of the 16-run
# factorial main-effect estimator. Seed 762 measures control seed sensitivity.
EXTRA = (
    ("control_seed762", {"seed": 762}),
    ("ascent_1p5", {"verified_ascent_reallocation": 1.5}),
    ("ascent_4", {"verified_ascent_reallocation": 4.0}),
    ("rehearsal_10", {"athletic_rehearsal_frac": .10}),
    ("terrain_early", {"teaching_schedule": "terrain-early"}),
    ("velocity_i_1p5", {"velocity_i_multiplier": 1.5}),
    ("push_aggressive", {"force_program": "aggressive",
                         "frontier_push_safety_n": 4.0}),
    ("motor_guidance_025", {"motor_guidance_weight": .025}),
)


@dataclass(frozen=True)
class Arm:
    index: int
    name: str
    changes: dict
    factors: tuple[str, ...]

    @property
    def directory(self) -> str:
        return f"arm_{self.index:02d}_{self.name}"


def arms() -> tuple[Arm, ...]:
    out = []
    for i in range(16):
        enabled = tuple(name for name, mask in zip(FACTOR_NAMES, FACTOR_MASKS)
                        if (i & mask).bit_count() % 2)
        changes = {key: value for (name, (key, value)) in
                   zip(FACTOR_NAMES, FACTOR_VALUES) if name in enabled}
        out.append(Arm(i, "factorial" if i else "control_seed761",
                       changes, enabled))
    for j, (name, changes) in enumerate(EXTRA, start=16):
        out.append(Arm(j, name, changes, ()))
    assert len(out) == 24
    # Main effects are balanced and pairwise independent on screening arms.
    for a in FACTOR_NAMES:
        assert sum(a in arm.factors for arm in out[:16]) == 8
        for b in FACTOR_NAMES:
            if a != b:
                assert sum(a in x.factors and b in x.factors for x in out[:16]) == 4
    return tuple(out)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=REPO).strip()


def source_guard() -> str:
    branch, sha, dirty = git("branch", "--show-current"), git("rev-parse", "HEAD"), git("status", "--porcelain")
    if branch != BRANCH or dirty or not re.fullmatch("[0-9a-f]{40}", sha):
        raise RuntimeError(
            f"Use clean branch {BRANCH}; actual={branch}, dirty={bool(dirty)}")
    if sha == PARENT_SOURCE:
        raise RuntimeError("Research source must differ from immutable 260M parent")
    return sha


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover_parent() -> Path:
    from emu.checkpoint_io import load_checkpoint
    root = Path.home() / "robot_lab/emu_runs"
    if not root.is_dir():
        raise RuntimeError(f"No {root}; specify --parent /absolute/path/checkpoints/final.pkl")
    matches = []
    for file in root.rglob("checkpoints/final.pkl"):
        try:
            payload = load_checkpoint(file, validate=True)
            meta = payload["meta"]
            source = (meta.get("source_provenance") or {}).get("git_commit")
            if source == PARENT_SOURCE and int(meta["step"]) == PARENT_STEP:
                matches.append(file.resolve())
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one released 260M final.pkl; found {len(matches)}: "
                           f"{matches}. Supply --parent explicitly.")
    return matches[0]


def read_parent(path: Path):
    from emu.checkpoint_io import load_checkpoint
    from emu.training_config import TrainingConfig
    from emu.contracts import OBS_SCHEMA_TERRAIN
    if path.name != "final.pkl" or path.parent.name != "checkpoints":
        raise RuntimeError("Parent MUST be an immutable checkpoints/final.pkl, not .emupolicy or best.pkl")
    config_path = path.parent.parent / "training_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing original run config: {config_path}")
    parent_cfg = TrainingConfig.load(config_path)
    payload = load_checkpoint(path, validate=True)
    meta = payload["meta"]
    original_sha = (meta.get("source_provenance") or {}).get("git_commit")
    if original_sha != PARENT_SOURCE or int(meta["step"]) != PARENT_STEP:
        raise RuntimeError(
            f"Wrong parent: source={original_sha}, step={meta.get('step')}; "
            f"need {PARENT_SOURCE} at {PARENT_STEP:,}")
    requirements = {
        "observation_schema": OBS_SCHEMA_TERRAIN,
        "unified_capability_focus": True,
        "foundation_steps": 20_000_000,
        "growth_recipe": "main-mature",
        "terrain_lesson": "coherent-careful",
        "force_program": "gentle",
        "terrain_step_max_m": .08,
        "athletic_rehearsal_frac": .25,
        "temporal_vx_reward_share": .50,
        "verified_ascent_reallocation": 0.0,
        "teaching_schedule": "balanced",
        "motor_guidance_weight": 0.0,
        "envs": 2048,
        "seed": 761,
        "open_frontier": True,
    }
    wrong = {k: (getattr(parent_cfg, k), v) for k, v in requirements.items()
             if getattr(parent_cfg, k) != v}
    if wrong:
        raise RuntimeError(f"Parent training_config.json does not match actual Pinnacle launch: {wrong}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"Parent final.pkl bytes are NOT the approved Pinnacle checkpoint: "
            f"expected {EXPECTED_PARENT_SHA256}, got {actual_sha256}")
    return parent_cfg, actual_sha256


def arm_config(parent_cfg, arm: Arm, root: Path, target: int):
    # Never reinitialise weights/optimizer, normaliser, teacher, health, or
    # capability field. The trainer restores all six from final.pkl.
    conf = replace(parent_cfg, run_dir=str(root / arm.directory),
                   resume="", device="cuda", steps=target,
                   fixed_eval_every_steps=10_000_000,
                   science_reset_step_evidence_on_fork=True)
    return replace(conf, **arm.changes)


def manifest(sha: str, parent: Path, parent_hash: str, cfg, root: Path, target: int) -> dict:
    return {
        "schema": 1,
        "study": "v5033-260m-plateau",
        "source_sha": sha, "source_branch": BRANCH,
        "parent_source_sha": PARENT_SOURCE, "parent_step": PARENT_STEP,
        "parent_path": str(parent), "parent_sha256": parent_hash,
        "target_steps": target, "envs": cfg.envs,
        "common_ancestor": True, "independent_training_seeds": [761, 762],
        "control": "arm_00_control_seed761",
        "study_note": "16-run resolution-IV main-effect screen + 8 targeted arms; held-out physics needed",
        "arms": [
            {"id": a.index, "name": a.directory, "factors": a.factors,
             "overrides": a.changes,
             "resolved_config": asdict(arm_config(cfg, a, root, target))}
            for a in arms()
        ],
    }


def ensure_manifest(root: Path, wanted: dict) -> None:
    filename = root / "campaign_manifest.json"
    if filename.exists():
        if json.loads(filename.read_text(encoding="utf-8")) != wanted:
            raise RuntimeError("Existing manifest or lineage differs; choose a NEW --root")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Nonempty root without manifest; cannot safely claim existing runs")
    root.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(wanted, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def preflight(config, path: Path, parent_hash: str, existing: bool):
    from emu.checkpoint_io import git_provenance
    from emu.run import inspect_run
    lineage = git_provenance()
    resume = "auto" if existing else str(path)
    ckpt, complete = inspect_run(
        config, provenance=lineage, resume=resume,
        research_parent_source="" if existing else PARENT_SOURCE,
        research_parent_sha256="" if existing else parent_hash)
    return ckpt, complete, lineage


def existing_state(directory: Path) -> bool:
    checkpoint_dir = directory / "checkpoints"
    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.pkl")):
        return True
    if directory.exists():
        extra = {x.name for x in directory.iterdir()}
        if extra - {"training_config.json", "console.log"}:
            raise RuntimeError(f"Ambiguous interrupted {directory} with no checkpoint: {extra}")
    return False


def run_one(arm: Arm, conf, parent: Path, hash_: str, root: Path) -> None:
    from emu.checkpoint_io import run_training, resolve_target_checkpoint
    from emu.training_config import training_sensor_contract
    current = existing_state(Path(conf.run_dir))
    source_guard()
    ckpt, complete, provenance = preflight(conf, parent, hash_, current)
    if not complete:
        if ckpt is None:
            raise RuntimeError("No valid explicit external ancestor or recovery checkpoint")
        actual = replace(conf, resume=str(ckpt))
        os.environ["EMU_UNIFIED_CONTINUOUS"] = "cold-growth-approved"
        print(f"\nSCIENCE {arm.index:02d}/23: {arm.directory} "
              f"{PARENT_STEP:,}->{conf.steps:,} source={provenance['commit'][:12]}", flush=True)
        run_training(actual, provenance=provenance)
    final, _, _ = resolve_target_checkpoint(
        Path(conf.run_dir), target_steps=conf.steps, envs=conf.envs,
        label=arm.directory, expected_observation_schema=conf.observation_schema,
        expected_sensor_contract=training_sensor_contract(conf))
    if final is None:
        raise RuntimeError(f"Run incomplete or invalid: {arm.directory}")
    print(f"VALIDATED FINAL {arm.index:02d}: {final}", flush=True)
    # A policy export is evaluation-only; full optimizer/teacher rewind stays
    # in the actual run checkpoints. Do not overwrite any earlier export.
    policy = root / "exports" / f"{arm.directory}.emupolicy"
    if not policy.exists():
        policy.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "emu.cli",
                        "export-policy", str(final), str(policy)],
                       cwd=REPO, check=True)
    from emu.viewer_server import validate_browser_bundle
    export_identity = validate_browser_bundle(json.loads(policy.read_text(encoding="utf-8")))
    if (export_identity["schema"] != 3 or
            export_identity["obs_dim"] != 360 or
            export_identity["checkpoint_schema"] != 9 or
            export_identity["step"] < conf.steps or
            export_identity["training_source_commit"] != provenance["commit"]):
        raise RuntimeError(f"Invalid exported research actor for {arm.directory}: {export_identity}")
    print("VALIDATED RESEARCH EXPORT (not approved):", policy, flush=True)


def write_summary(root: Path, sha: str, target: int) -> None:
    import campaign_support as support
    cases = [
        {"name": a.directory, "recipe": "260m-full-state-fork",
         "schedule": "+".join(a.factors) if a.factors else a.name,
         "steps": target, "from_steps": PARENT_STEP}
        for a in arms()
    ]
    support.summary(root, cases, sha)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, help="original v5032 checkpoints/final.pkl; auto-discover if omitted")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--target-steps", type=int, default=TARGET_STEPS,
                   help="absolute world-step target, not additional steps")
    p.add_argument("--arm", type=int, choices=range(24),
                   help="one numbered arm 0..23 (defaults to the control arm 0)")
    p.add_argument("--all", action="store_true", help="select all 24 arms (sequential GPU)")
    p.add_argument("--matrix", action="store_true", help="show 24-arm design; no parent/GPU needed")
    p.add_argument("--verify-only", action="store_true", help="validate exact existing/external checkpoints without writing")
    p.add_argument("--run", action="store_true", help="explicitly allow writing and training")
    p.add_argument("--report", action="store_true", help="rebuild read-only results ZIP from existing runs")
    args = p.parse_args()
    if args.all and args.arm is not None:
        p.error("use either --all or --arm, not both")
    chosen = list(arms() if args.all else [arms()[args.arm or 0]])
    if args.matrix:
        for a in arms():
            print(f"{a.index:02d} {a.directory:28} " + (", ".join(a.factors) or str(a.changes) or "untouched"))
        print("16 balanced factorial runs + 8 targeted arms, including an independent-seed control.")
        return 0
    sha = source_guard()
    if args.target_steps <= PARENT_STEP or args.target_steps > 300_000_000:
        raise RuntimeError("Choose a cumulative target after 260,046,848 and <=300M")
    root = args.root.expanduser().resolve()
    if root == REPO or REPO in root.parents:
        raise RuntimeError("Results must be outside the Git checkout")
    if args.report:
        write_summary(root, sha, args.target_steps)
        return 0
    parent = (args.parent.expanduser().resolve()
              if args.parent else discover_parent())
    cfg, parent_hash = read_parent(parent)
    want = manifest(sha, parent, parent_hash, cfg, root, args.target_steps)
    print(f"RESEARCH SOURCE {sha} ({BRANCH}); clean: PASS")
    print(f"IMMUTABLE PARENT {parent} step={PARENT_STEP:,} sha256={parent_hash}")
    print(f"TARGET {args.target_steps:,} / {cfg.envs} worlds; {len(chosen)} arms selected")
    for a in chosen:
        conf = arm_config(cfg, a, root, args.target_steps)
        print(f"  {a.index:02d} {a.directory}: {a.changes or 'exact baseline'} "
              f"seed={conf.seed} ascent={conf.verified_ascent_reallocation}")
    if not (args.verify_only or args.run):
        print("PLAN ONLY: no GPU and no outputs touched. Start with --verify-only or --run --arm 0.")
        return 0
    # Check all selected arms BEFORE creating any result files.
    for a in chosen:
        conf = arm_config(cfg, a, root, args.target_steps)
        preflight(conf, parent, parent_hash, existing_state(Path(conf.run_dir)))
    print("PRE-FLIGHT PASS: checkpoint bytes, source, actor, and full-rewind contracts checked.")
    if args.verify_only:
        return 0
    ensure_manifest(root, want)
    for a in chosen:
        run_one(a, arm_config(cfg, a, root, args.target_steps), parent, parent_hash, root)
    write_summary(root, sha, args.target_steps)
    print("SCIENCE CAMPAIGN FINISHED; frozen physical examinations still required before claiming improvement.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by operator. Resume with the SAME clean source and --run selection.", file=sys.stderr)
        raise SystemExit(130)
