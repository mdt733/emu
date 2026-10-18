"""EMU v5016: one bounded, resumable 232-input campaign.

Four distinct arms -> two finalists -> successive guarded 100M refinements.
It never trains inside Git, never touches the public champion, and never
mistakes PPO reward for held-out physical competence.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from .checkpoint_io import (
    git_provenance, run_training, load_checkpoint, resolve_target_checkpoint,
    update_boundary_at_or_after,
)
from .contracts import OBS_SCHEMA_SENSOR, source_repo_root
from .training_config import (
    mature_config, TrainingConfig, training_sensor_contract,
    verify_existing_phase_config,
)

VERSION = "v5016-pinnacle-1"
FIRST_TARGET = 80_000_000
SECOND_TARGET = 200_000_000
POLISH_TARGETS = (300_000_000, 400_000_000, 500_000_000,
                  600_000_000, 700_000_000, 800_000_000)
DEFAULT_ROOT = Path.home() / "robot_lab" / "emu_runs" / "v5016_pinnacle"
DEFAULT_FOUNDATION = (Path.home() / "robot_lab" / "emu_runs" /
                      "v5016_sensor_foundation_20m")

# One question per branch, same 20M ancestor, same PPO seed and actor.
ARMS = {
    "baseline": {},
    "grace": {"motor_guidance_weight": .035},
    "contact": {"domain_randomization": True,
                "domain_terrain_mode": "contact",
                "domain_terrain_radius": .40},
    "combo": {"combo_mix_frac": 1.0, "combo_probe_frac": .20},
}
EFFORT_KEYS = ("power_robust_shaft_abs_power_w",
               "power_contact_slope_shaft_abs_power_w")
SLOPE_KEYS = ("contact_slope", "contact_harder")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validated_checkpoint(path: Path, contract: dict):
    if not path.is_file():
        raise FileNotFoundError(f"required schema-8 checkpoint missing: {path}")
    return load_checkpoint(
        path, validate=True, expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)


def base_policy(foundation: Path) -> dict:
    foundation = foundation.expanduser().resolve()
    config = TrainingConfig.load(foundation / "training_config.json")
    if config.observation_schema != OBS_SCHEMA_SENSOR or config.envs <= 0:
        raise ValueError("foundation is not a valid v5016 232-input run")
    contract = training_sensor_contract(config)
    final = foundation / "checkpoints" / "final.pkl"
    payload = validated_checkpoint(final, contract)
    step = int(payload["meta"]["step"])
    if not 19_000_000 <= step <= 22_000_000:
        raise ValueError(f"expected the ~20M v5016 foundation, got {step:,}")
    return {"path": str(final), "run_dir": str(foundation),
            "sha256": sha256(final), "step": step,
            "source": (payload["meta"].get("source_provenance") or {}).get("git_commit", ""),
            "sensor_sha256": contract["sha256"]}


def score(exam: dict, *, ancestor: dict | None = None) -> dict:
    """Rank solely on independently matched worlds, with strict safety floors."""
    def val(name):
        x = float(exam.get(name, float("nan")))
        if not math.isfinite(x):
            raise ValueError(f"missing/nonfinite exam value {name}")
        return x

    hard = (val("screen_resilience100_survival") +
            val("screen_robust70_survival")) / 2
    growth = (val("screen_growth140_survival") +
              val("screen_growthcombo140_survival")) / 2
    contact = sum(val(f"screen_{p}_survival") for p in SLOPE_KEYS) / 2
    core = val("screen_core_survival")
    panel_names = ("robust70", "growth140", "growthcombo140",
                   "contact_slope", "contact_harder")
    moving = sum(val(f"screen_{p}_{axis}_mae") / tolerance
                 for p in panel_names
                 for axis, tolerance in (("vx", .20), ("vy", .20),
                                         ("yaw", .90))) / 15
    height = sum(val(f"screen_{p}_height_shortfall_m")
                 for p in panel_names) / len(panel_names)
    pitch = sum(val(f"screen_{p}_pitch_mae")
                for p in panel_names) / len(panel_names)
    step = sum(val(f"screen_{p}_step_evidence")
               for p in panel_names) / len(panel_names)
    support = sum(val(f"screen_{p}_support")
                  for p in panel_names) / len(panel_names)
    watts = sum(val(k) for k in EFFORT_KEYS) / len(EFFORT_KEYS)
    overload = (val("power_robust_over_rated_frac") +
                val("power_contact_slope_over_rated_frac")) / 2
    # Low servo power alone must not beat a moving, recovering robot.
    rank = (.27 * hard + .19 * growth + .23 * contact +
            .13 * core + .07 * step + .05 * support -
            .11 * min(2., moving) -
            .06 * min(1., height / .035) -
            .035 * min(1., pitch / .45) -
            .018 * min(1.5, watts / 15.) -
            .015 * min(1., overload))
    reasons = []
    if core < .80:
        reasons.append("core survival below 80%")
    if hard < .045:
        reasons.append("hard survival below 4.5%")
    if contact < .04:
        reasons.append("real contact-slope survival below 4%")
    if height > .055:
        reasons.append("height shortfall above 55 mm")
    if pitch > .55:
        reasons.append("pitch error above .55 rad")
    if step < .12 or support < .70:
        reasons.append("insufficient steps/foot support")
    if ancestor is not None:
        if core < ancestor["core"] - .10:
            reasons.append("core survival dropped more than 10 points")
        if hard < ancestor["hard"] - .10:
            reasons.append("hard survival dropped more than 10 points")
    return {"score": round(rank, 6), "eligible": not reasons,
            "reasons": reasons, "core": core, "hard": hard, "growth": growth,
            "contact": contact, "step": step, "support": support,
            "motion_norm_error": moving, "height_shortfall_m": height,
            "pitch_mae_rad": pitch, "shaft_abs_power_w": watts,
            "over_rated_frac": overload}


def check_root(root: Path) -> None:
    root = root.expanduser().resolve()
    git = source_repo_root()
    if root == Path.home().resolve() or (
        git and (root == git.resolve() or git.resolve() in root.parents)):
        raise ValueError("campaign output must be outside home root and Git checkout")


def phase_config(path: Path, *, target: int, seed: int, envs: int,
                 device: str, arm: str, polish_index: int = -1) -> TrainingConfig:
    cfg = mature_config(
        "graph", run_dir=path, steps=target, seed=seed,
        envs=envs, device=device, resume=None)
    cfg = replace(
        cfg, observation_schema=OBS_SCHEMA_SENSOR,
        attitude_profile="v5015", pitch_command_max_deg=15.0,
        # Keep the new sensor observation normaliser adaptable during forks.
        freeze_obs_norm_after_steps=0,
        field_max_influence=.10,
        field_influence_start_steps=20_000_000,
        field_influence_full_steps=90_000_000,
        # A moderate difficulty bridge, with widening combos as training grows.
        phase_speed_gain=1.25, combo_mix_frac=.88,
        combo_probe_frac=.12)
    if arm not in ARMS:
        raise ValueError(f"unknown study arm {arm!r}")
    cfg = replace(cfg, **ARMS[arm])
    if polish_index >= 0:
        radius = (.12, .25, .40, .58, .75, .90)[polish_index]
        cfg = replace(
            cfg, domain_randomization=True, domain_terrain_mode="contact",
            domain_terrain_radius=radius,
            domain_hardware_radius=(0.0 if polish_index < 2 else
                                    .10 if polish_index < 4 else .25),
            combo_mix_frac=.90 if polish_index < 3 else .98,
            combo_probe_frac=.12 if polish_index < 3 else .20,
            motor_guidance_weight=max(cfg.motor_guidance_weight, .02),
            # Later phases should not move normaliser coordinates forever.
            freeze_obs_norm_after_steps=450_000_000)
    return cfg


def stage_run(root: Path, *, label: str, parent: dict, target: int,
              cfg: TrainingConfig, provenance: dict, contract: dict,
              ledger: dict) -> dict:
    run = Path(cfg.run_dir)
    final = run / "checkpoints" / "final.pkl"
    parent_file = Path(parent["path"])
    parent_payload = validated_checkpoint(parent_file, contract)
    if sha256(parent_file) != parent["sha256"]:
        raise RuntimeError(f"parent changed since selection: {parent_file}")
    if target <= int(parent_payload["meta"]["step"]):
        raise ValueError(f"stage {label} target must exceed parent checkpoint")
    spec = {"version": VERSION, "label": label,
            "parent": parent, "target": target,
            "configuration": {k: v for k, v in asdict(cfg).items()
                              if k not in ("device", "resume")},
            "source_commit": provenance["commit"]}
    spec_path = run / "campaign_stage.json"
    if spec_path.exists():
        if read_json(spec_path) != spec:
            raise RuntimeError(f"stage config/ancestor/source changed: {run}")
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"unrecognised nonempty stage directory: {run}")
        save_json(spec_path, spec)

    saved_config = run / "training_config.json"
    if saved_config.exists():
        verify_existing_phase_config(run, cfg)
        old_meta = read_json(run / "run_meta.json", {})
        if old_meta and (old_meta.get("source_provenance") or {}).get(
                "git_commit") != provenance["commit"]:
            raise RuntimeError(f"stage source commit changed: {run}")

    complete, partial, exact = resolve_target_checkpoint(
        run, target_steps=target, envs=cfg.envs, label=label,
        expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)
    if complete is None:
        resume = "auto" if partial is not None else str(parent_file)
        launched = replace(cfg, resume=resume)
        started = time.monotonic()
        try:
            run_training(launched, provenance=provenance)
        finally:
            ledger["gpu_seconds"] += time.monotonic() - started
            save_json(root / "campaign.json", ledger)
        complete, partial, exact = resolve_target_checkpoint(
            run, target_steps=target, envs=cfg.envs, label=label,
            expected_observation_schema=OBS_SCHEMA_SENSOR,
            expected_sensor_contract=contract)
        if complete is None:
            raise RuntimeError(f"no validated final stage checkpoint: {run}")
    data = validated_checkpoint(complete, contract)
    if int(data["meta"]["step"]) != exact:
        raise RuntimeError("final checkpoint is not on expected PPO update boundary")
    # A standalone run has raw results but not its Markdown until requested.
    from .results import write_report
    write_report(run, bundle=False)
    return {"path": str(complete), "run_dir": str(run),
            "sha256": sha256(complete),
            "step": int(data["meta"]["step"]),
            "source": provenance["commit"], "sensor_sha256": contract["sha256"]}


def earlier_best(final: dict, contract: dict) -> dict | None:
    """Represent a validated best.pkl as its own non-mutating candidate."""
    path = Path(final["run_dir"]) / "checkpoints" / "best.pkl"
    if not path.is_file():
        return None
    payload = validated_checkpoint(path, contract)
    if int(payload["meta"]["step"]) > final["step"]:
        raise RuntimeError(f"future best policy in {path}")
    digest = sha256(path)
    if digest == final["sha256"]:
        return None
    return {**final, "path": str(path), "sha256": digest,
            "step": int(payload["meta"]["step"])}


def screen(root: Path, ledger: dict, *, name: str, contenders: dict,
           baseline: str, envs: int, horizon: int, device: str) -> dict:
    """One read-only batch with identical worlds, seeds and commands."""
    output = root / "scoreboards" / (name + ".json")
    identities = {k: v["sha256"] for k, v in contenders.items()}
    expected = {"name": name, "identities": identities,
                "baseline": baseline, "envs": envs, "horizon": horizon,
                "device": device, "exam_revision": 2}
    cached = read_json(output)
    if cached is not None:
        if cached.get("request") != expected:
            raise RuntimeError(f"stale scorecard for {name}; don't reuse it")
        return cached

    from .evaluation import evaluate_runs
    dirs = [Path(contenders[k]["path"]) for k in contenders]
    started = time.monotonic()
    try:
        raw = evaluate_runs(
            dirs, envs=envs, horizon=horizon, device=device,
            eval_pitch_deg=15., eval_phase_speed_gain=1.,
            sensor_contact_exam=True, key_by_path=True,
            write_screen_files=False)
    finally:
        ledger["gpu_seconds"] += time.monotonic() - started
        save_json(root / "campaign.json", ledger)
    safety = score(raw[contenders[baseline]["path"]])
    rows = []
    for label, desc in contenders.items():
        exam = raw[desc["path"]]
        rating = score(exam, ancestor=None if label == baseline else safety)
        rows.append({"name": label, "policy": desc, "rating": rating,
                     "exam": exam, "incumbent": label == baseline})
    board = {"request": expected, "rows": rows, "baseline_score": safety["score"]}
    save_json(output, board)
    return board


def choose(board: dict, *, incumbent: str, slots: int = 1) -> list[str]:
    """New candidates may not displace a safer incumbent without evidence."""
    rows = [x for x in board["rows"]
            if x["rating"]["eligible"] and x["name"] != incumbent]
    rows.sort(key=lambda x: (-x["rating"]["score"], x["name"]))
    return [row["name"] for row in rows[:slots]]


def report(root: Path, ledger: dict) -> None:
    boards = sorted((root / "scoreboards").glob("*.json")) if (
        root / "scoreboards").is_dir() else []
    lines = ["# EMU v5016 pinnacle campaign", "",
             f"Source branch commit: `{ledger['source_commit']}`",
             f"Observation contract: `{ledger['sensor_sha256']}`",
             f"GPU-wall budget: {ledger['budget_hours']} hours (soft, stage boundaries)",
             f"Accounted active training + screening: {ledger['gpu_seconds'] / 3600:.2f} h",
             "",
             "## Champion", ""]
    best = ledger.get("champion") or ledger["foundation"]
    lines += [f"Checkpoint: `{best['path']}`",
              f"SHA-256: `{best['sha256']}`",
              f"Policy training steps: {best['step']:,}", "",
              "Scores come from fixed seeds, equal exam conditions, and a",
              "simulated servo shaft-power proxy—not measured electrical watts.",
              "", "## Held-out tournament", ""]
    for path in boards:
        b = read_json(path)
        lines.append("### " + b["request"]["name"])
        lines += ["", "| Contender | Eligible | Score | Core | Hard | Contact | Shaft W | Reasons |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for row in sorted(b["rows"], key=lambda r: r["rating"]["score"], reverse=True):
            s = row["rating"]
            lines.append(
                f"| {row['name']} | {s['eligible']} | {s['score']:.3f} | "
                f"{s['core']:.2f} | {s['hard']:.2f} | {s['contact']:.2f} | "
                f"{s['shaft_abs_power_w']:.1f} | {', '.join(s['reasons'])} |")
        lines.append("")
    lines += ["## Notes", "",
              "- All branches share the same immutable ~20M sensor foundation.",
              "- Every stage has its own config, console log, checkpoints and results.md.",
              "- The incumbent is retained whenever a continuation loses held-out competence.",
              "- Dynamic/real-contact exams are simulations, not physical hardware validation.",
              "- Schema-8 weights are not compatible with the legacy browser viewer.",
              ""]
    (root / "results.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(root / "results_data.json", ledger)
    save_json(root / "champion.json", best)
    archive = root / "results_bundle.zip"
    tmp = archive.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in (root / "results.md", root / "results_data.json",
                  root / "campaign.json", root / "champion.json"):
            if p.is_file(): z.write(p, arcname=p.relative_to(root))
        for p in boards:
            z.write(p, arcname=p.relative_to(root))
        for sub in ("round1", "round2", "polish"):
            folder = root / sub
            if folder.is_dir():
                for p in sorted(folder.rglob("*")):
                    if p.is_file() and p.name in (
                        "results.md", "results_data.json", "metrics.csv",
                        "fixed_eval.csv", "training_config.json",
                        "campaign_stage.json", "performance.json",
                        "envelope_summary.json", "capability_field_summary.json"):
                        z.write(p, arcname=p.relative_to(root))
    os.replace(tmp, archive)


def budget_allows(ledger: dict, *, steps: int) -> bool:
    # Conservative rate learned from actual campaign wall time including
    # per-run final fixed exams. No mid-run GPU interruption is ever attempted.
    rates = ledger.get("observed_rates", [])
    rate = max(12_000., min(45_000., sum(rates[-4:]) / len(rates[-4:]))) if rates else 30_000.
    projection = steps / rate * 1.12 + 150.
    remaining = ledger["budget_hours"] * 3600. - ledger["gpu_seconds"]
    if remaining < .75 * projection:
        print(f"Budget guard: next stage projects {projection / 3600:.2f}h "
              f"with {remaining / 3600:.2f}h remaining; retaining best brain.")
        return False
    return True


def campaign(root: Path, foundation: Path, *, hours: float, envs: int,
             seed: int, device: str, screen_envs: int, screen_horizon: int):
    root = root.expanduser().resolve()
    check_root(root)
    if not 0.5 <= hours <= 72:
        raise ValueError("hours budget must be between .5 and 72")
    if envs < 2 or screen_envs < 2 or screen_horizon < 20:
        raise ValueError("need >=2 worlds and >=20 screening frames")
    foundation_policy = base_policy(foundation)
    found_cfg = TrainingConfig.load(Path(foundation_policy["run_dir"]) /
                                    "training_config.json")
    contract = training_sensor_contract(found_cfg)
    provenance = git_provenance()
    if not provenance["available"] or provenance["dirty"] or not provenance["commit"]:
        raise RuntimeError("campaign requires a clean, identifiable Git checkout")
    if provenance["branch"] != "feature/v5016-sim2real":
        raise RuntimeError("campaign is strictly feature/v5016-sim2real only")
    path = root / "campaign.json"
    current = read_json(path)
    if current is None:
        if root.exists() and any(p.name != ".campaign.lock" for p in root.iterdir()):
            raise RuntimeError(f"campaign root contains unrecognised data: {root}")
        ledger = {"schema": 1, "version": VERSION,
                  "source_commit": provenance["commit"],
                  "foundation": foundation_policy, "champion": foundation_policy,
                  "sensor_sha256": contract["sha256"],
                  "seed": seed, "envs": envs, "screen_envs": screen_envs,
                  "screen_horizon": screen_horizon, "budget_hours": hours,
                  "gpu_seconds": 0., "observed_rates": [],
                  "round1": {}, "round2": {}, "polish": {},
                  "selected": [], "finished": False}
        save_json(path, ledger)
    else:
        ledger = current
        fixed = ("source_commit", "foundation", "sensor_sha256", "seed",
                 "envs", "screen_envs", "screen_horizon", "budget_hours")
        expected = (provenance["commit"], foundation_policy,
                    contract["sha256"], seed, envs, screen_envs,
                    screen_horizon, hours)
        if any(ledger[k] != v for k, v in zip(fixed, expected)):
            raise RuntimeError("campaign source/parameters/foundation changed; use a new root")
    if ledger["finished"]:
        report(root, ledger)
        print("Campaign already complete:", ledger["champion"]["path"])
        return

    def one(label, parent, target, arm, *, group, polish_index=-1):
        run = root / group / label
        cfg = phase_config(run, target=target, seed=seed, envs=envs,
                           device=device, arm=arm, polish_index=polish_index)
        started = time.monotonic()
        already_final = (run / "checkpoints" / "final.pkl").is_file()
        result = stage_run(
            root, label=group + "/" + label, parent=parent,
            target=target, cfg=cfg, provenance=provenance,
            contract=contract, ledger=ledger)
        duration = time.monotonic() - started
        added = max(0, result["step"] - parent["step"])
        if duration > 5 and added > 0 and not already_final:
            ledger["observed_rates"].append(added / duration)
        ledger[group][label] = result
        save_json(path, ledger)
        report(root, ledger)
        return result

    # Four independent questions from the SAME frozen sensor foundation.
    for arm in ARMS:
        if arm in ledger["round1"]:
            continue
        inc = FIRST_TARGET - ledger["foundation"]["step"]
        if not budget_allows(ledger, steps=inc):
            report(root, ledger)
            return
        print(f"\n=== Round 1: {arm} (20M -> 80M) ===", flush=True)
        one(arm, ledger["foundation"], FIRST_TARGET, arm, group="round1")

    if not ledger["selected"]:
        pool = {"foundation": ledger["foundation"], **ledger["round1"]}
        board = screen(root, ledger, name="round1", contenders=pool,
                       baseline="foundation", envs=screen_envs,
                       horizon=screen_horizon, device=device)
        ledger["selected"] = choose(board, incumbent="foundation", slots=2)
        if not ledger["selected"]:
            print("No round1 arm met the safety gates. Keeping foundation; see scorecard.")
            ledger["finished"] = True
            save_json(path, ledger); report(root, ledger)
            return
        save_json(path, ledger)
        report(root, ledger)

    for arm in ledger["selected"]:
        if arm in ledger["round2"]:
            continue
        inc = SECOND_TARGET - ledger["round1"][arm]["step"]
        if not budget_allows(ledger, steps=inc):
            report(root, ledger)
            return
        print(f"\n=== Round 2 finalist: {arm} (80M -> 200M) ===", flush=True)
        one(arm, ledger["round1"][arm], SECOND_TARGET, arm, group="round2")

    if "round2" not in ledger.get("winners", {}):
        contenders = {"foundation": ledger["foundation"],
                      **{"r1_" + k: v for k, v in ledger["round1"].items()},
                      **{"r2_" + k: v for k, v in ledger["round2"].items()}}
        for arm, final in ledger["round2"].items():
            candidate = earlier_best(final, contract)
            if candidate is not None:
                contenders["r2_" + arm + "_best"] = candidate
        board = screen(root, ledger, name="round2", contenders=contenders,
                       baseline="foundation", envs=screen_envs,
                       horizon=screen_horizon, device=device)
        winners = [x for x in board["rows"] if x["rating"]["eligible"]]
        winners.sort(key=lambda x: (-x["rating"]["score"], x["name"]))
        winner = winners[0] if winners else next(x for x in board["rows"]
                                               if x["name"] == "foundation")
        ledger.setdefault("winners", {})["round2"] = winner["name"]
        ledger["champion"] = winner["policy"]
        save_json(path, ledger)
        report(root, ledger)

    parent = ledger["champion"]
    winning_name = ledger["winners"]["round2"]
    lineage_arm = (winning_name[3:].removesuffix("_best")
                   if winning_name.startswith(("r1_", "r2_")) else "baseline")
    for idx, target in enumerate(POLISH_TARGETS):
        name = str(target // 1_000_000) + "m"
        if name in ledger["polish"]:
            # The championship may have stayed at an earlier stronger model.
            parent = ledger["champion"]
            continue
        additional = target - parent["step"]
        if not budget_allows(ledger, steps=additional):
            report(root, ledger)
            return
        print(f"\n=== Pinnacle refinement {name}: {parent['step']:,} -> {target:,} ===",
              flush=True)
        child = one(name, parent, target, lineage_arm,
                    group="polish", polish_index=idx)
        contenders = {"incumbent": parent, "challenger": child}
        best_before_end = earlier_best(child, contract)
        if best_before_end is not None and best_before_end["sha256"] != parent["sha256"]:
            contenders["challenger_best"] = best_before_end
        board = screen(root, ledger, name="polish_" + name,
                       contenders=contenders, baseline="incumbent",
                       envs=screen_envs, horizon=screen_horizon, device=device)
        incumbent = next(x for x in board["rows"] if x["name"] == "incumbent")
        challengers = [x for x in board["rows"] if x["name"].startswith("challenger")]
        challenger = max(challengers, key=lambda x: x["rating"]["score"]
                         if x["rating"]["eligible"] else -1e9)
        if (challenger["rating"]["eligible"] and
                challenger["rating"]["score"] >= incumbent["rating"]["score"] + .003):
            ledger["champion"] = challenger["policy"]
            print(f"Promoted {name} {challenger['name']}: held-out improvement.")
        else:
            print(f"Retained previous champion over {name}: no verified improvement.")
        parent = ledger["champion"]
        save_json(path, ledger)
        report(root, ledger)
    ledger["finished"] = True
    save_json(path, ledger)
    report(root, ledger)
    print("\nPinnacle finished! Champion:", ledger["champion"]["path"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="emu campaign5016",
        description="Staged, protected v5016 sensor actor tournament and refinement")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--foundation", type=Path, default=DEFAULT_FOUNDATION)
    p.add_argument("--hours", type=float, default=12.)
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--screen-envs", type=int, default=128)
    p.add_argument("--screen-horizon", type=int, default=300)
    p.add_argument("--seed", type=int, default=733)
    p.add_argument("--device", choices=("cuda", "cpu", "auto", "gpu"), default="cuda")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true",
                   help="execute full campaign; otherwise show read-only plan")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.status:
        data = read_json(root / "campaign.json")
        if data is None:
            raise SystemExit(f"No campaign at {root}")
        print(json.dumps({"gpu_hours": data["gpu_seconds"] / 3600,
            "budget_hours": data["budget_hours"], "champion": data["champion"],
            "round1": list(data["round1"]), "round2": list(data["round2"]),
            "polish": list(data["polish"]), "finished": data["finished"]},
            indent=2))
        return 0
    if args.plan or not args.run:
        f = base_policy(args.foundation)
        print(f"Foundation {f['step']:,}: {f['path']} ({f['sha256'][:12]})")
        print("4 × 80M branches -> 2 × 200M finalists -> 300/400/500/600/700/800M guarded refinement")
        print("Common real contact incline + lateral/yaw + servo shaft-effort exams")
        print("Approx total incremental policy steps: 1.08B (if no fallback)")
        print(f"Soft GPU-wall budget: {args.hours:g}h, {args.envs} training worlds, "
              f"{args.screen_envs} screen worlds; no writes, no training")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another pinnacle campaign is already running in this directory") from exc
        campaign(root, args.foundation, hours=args.hours, envs=args.envs,
                 seed=args.seed, device=args.device,
                 screen_envs=args.screen_envs, screen_horizon=args.screen_horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
