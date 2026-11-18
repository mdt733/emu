"""Read-only terminal dashboard for the existing EMU metrics.csv.

This module deliberately does not import MuJoCo, JAX, training modules or
checkpoint loaders. Attach from a SECOND terminal; training remains unchanged.
No writes, extra GPU work, reward changes or checkpoint compatibility changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path


def read_history(path: Path) -> list[dict[str, str]]:
    """Accept an in-progress CSV append; de-duplicate process-boundary rows."""
    try:
        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                return []
            by_step: dict[int, dict[str, str]] = {}
            for row in reader:
                try:
                    step = int(float(row["step"]))
                except (TypeError, ValueError, KeyError):
                    continue
                if None in row:  # An incomplete or malformed append.
                    continue
                by_step[step] = row
        return [by_step[k] for k in sorted(by_step)]
    except (OSError, UnicodeError, csv.Error):
        return []


def value(row: dict, key: str) -> float | None:
    try:
        result = float(row.get(key, ""))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(row: dict, key: str, precision: int = 2,
        scale: float = 1., suffix: str = "") -> str:
    n = value(row, key)
    return f"{n * scale:.{precision}f}{suffix}" if n is not None else "--"


def delta(history: list[dict], key: str, *, scale: float = 1.,
          precision: int = 2) -> str:
    """Approximately 5M-step trend, never an arbitrary old comparison."""
    if len(history) < 2:
        return "--"
    now, current = value(history[-1], "step"), value(history[-1], key)
    if now is None or current is None:
        return "--"
    target = now - 5_000_000.
    candidates = [(abs(value(r, "step") - target), r)
                  for r in history[:-1]
                  if value(r, "step") is not None and value(r, key) is not None]
    if not candidates:
        return "--"
    distance, old = min(candidates, key=lambda pair: pair[0])
    if distance > 1_250_000:
        return "--"
    return f"{(current - value(old, key)) * scale:+.{precision}f}"


def sampled_window(history: list[dict], attempts: str, successes: str) -> str:
    """Explicitly sampled PPO rollouts: NOT a census of all training attempts."""
    if not history:
        return "--"
    now = value(history[-1], "step")
    if now is None:
        return "--"
    rows = [r for r in history if now - 5_000_000 < (
        value(r, "step") if value(r, "step") is not None else -1) <= now]
    n = sum(max(0., value(r, attempts) or 0.) for r in rows)
    good = sum(max(0., value(r, successes) or 0.) for r in rows)
    if n < 1:
        return "no sampled attempts"
    if good > n:
        return "invalid sampled counts"
    return f"{int(good)}/{int(n)} ({100 * good / n:.1f}%)"


def latest_frozen(history: list[dict]) -> str:
    scores: dict[int, float] = {}
    for row in history:
        step, score = value(row, "step"), value(row, "eval_broad_score")
        source_step = value(row, "eval_policy_step")
        if source_step is None:
            source_step = step
        if source_step is not None and score is not None and score > -100:
            scores[int(source_step)] = score
    if not scores:
        return "not yet evaluated"
    ranked = sorted(scores.items())
    at, score = ranked[-1]
    if len(ranked) == 1:
        return f"{score:.3f} at {at/1e6:.0f}M"
    earlier = ranked[-2][1]
    return f"{score:.3f} at {at/1e6:.0f}M (Δ {score-earlier:+.3f})"


def target_from_config(root: Path) -> int | None:
    try:
        obj = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
        target = int(obj["steps"])
        return target if target > 0 else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def render(history: list[dict], *, run_name: str, target: int | None,
           width: int = 100, height: int = 27) -> str:
    """Screen-only projection of existing training metrics; no invented exams."""
    row = history[-1] if history else {}
    step = value(row, "step") or 0.
    if target is not None and target > 0:
        fraction = min(1., max(0., step / target))
        width_bar = max(8, min(30, width - 48))
        bars = round(width_bar * fraction)
        progress = (f"{'█' * bars}{'░' * (width_bar - bars)} "
                    f"{fraction * 100:5.1f}%   {step/1e6:.1f}M / {target/1e6:.1f}M")
        fps = value(row, "steps_per_sec")
        eta = f"{max(0., target-step)/fps/60:.0f}m approx." if fps and fps > 0 else "--"
    else:
        progress = f"{step/1e6:.1f}M   (awaiting training target)"
        eta = "--"
    lines = [
        f" EMU   |   {run_name}",
        " " + progress,
        f" Read-only training dashboard  |  ETA {eta}",
        "",
        " WALKING     current rollout (Δ = change over ~5M steps)",
        (f"   Reward {fmt(row,'reward_mean',3)} (Δ{delta(history,'reward_mean',precision=3)})"
         f"    Falls {fmt(row,'fall_rate',1,100)}% (Δ{delta(history,'fall_rate',scale=100,precision=1)}pp)"),
        (f"   Episode {fmt(row,'ep_len_mean',0)} frames    "
         f"Core {fmt(row,'core_competence')} (Δ{delta(history,'core_competence')})"),
        (f"   Actual/requested vx {fmt(row,'vx_mean')}/{fmt(row,'cmd_vx_mean')} m/s"
         f"    Shaft≈{fmt(row,'control_shaft_j_per_m',1)} J/m"),
        "",
        " COMMANDS    teacher targets, not certified skill",
        (f"   vx [{fmt(row,'frontier_vx_lo')},+{fmt(row,'frontier_vx_hi')}] m/s"
         f"    vy ±{fmt(row,'frontier_vy')} m/s    yaw ±{fmt(row,'frontier_yaw')} rad/s"),
        (f" SLOPES      target ≤{fmt(row,'slope_max_deg',1)}°"
         f"    measured maximum {fmt(row,'physical_slope_deg_max',1)}°"),
        "",
        " STAIRS      5M sampled training episodes, NOT all-episode counts",
        (f"   Near UP {sampled_window(history,'academy_near_attempts_up','academy_near_successes_up')}"
         f"    DOWN {sampled_window(history,'academy_near_attempts_down','academy_near_successes_down')}"),
        (f"   Footholds {fmt(row,'academy_footholds',0)}  "
         f"Transfers {fmt(row,'academy_transfers',0)}  "
         f"UP crosses {fmt(row,'academy_up_crossings',0)}  "
         f"DOWN crosses {fmt(row,'academy_down_crossings',0)}"),
        (f"   Lesson ceilings UP {fmt(row,'academy_height_up_m',1,100)}cm"
         f" DOWN {fmt(row,'academy_height_down_m',1,100)}cm"
         f"    All-update Academy episodes (process) "
         f"{fmt(row,'ppo_stair_audit_academy_episodes',0)}"),
        "",
        " RESILIENCE  difficulty " + fmt(row, "resilience_difficulty") +
        "    max push " + fmt(row, "frontier_push_n") + " N",
        (f" TRAINING    PPO {fmt(row,'train_steps_per_sec',0)}/s"
         f"    wall {fmt(row,'steps_per_sec',0)}/s"
         f"    KL {fmt(row,'ppo_approx_kl_final',4)}"
         f"    NaN skips {fmt(row,'nan_grad_skips_total',0)}"),
        " FROZEN      " + latest_frozen(history),
        "",
        " Source: metrics.csv  |  sampled events ≠ total attempts  |  Ctrl-C to exit",
    ]
    # Don't let a narrow terminal wrap into an endlessly scrolling output.
    max_lines = max(5, height - 1)
    if len(lines) > max_lines:
        lines = [line for line in lines if line]  # Drop spacing first.
    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + [lines[-1]]
    return "\n".join(line[:max(10, width-1)] for line in lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only live EMU dashboard (attach from a second terminal)")
    p.add_argument("--run-dir", type=Path, required=True, help="existing or starting EMU run directory")
    p.add_argument("--target-steps", type=int, default=None, help="optional until training_config.json exists")
    p.add_argument("--interval", type=float, default=1., help="refresh interval in seconds")
    p.add_argument("--once", action="store_true", help="print one snapshot and exit")
    p.add_argument("--plain", action="store_true", help="never clear or redraw terminal")
    args = p.parse_args(argv)
    if args.target_steps is not None and args.target_steps <= 0:
        p.error("--target-steps must be positive")
    if not math.isfinite(args.interval) or args.interval < .2 or args.interval > 60:
        p.error("--interval must be between 0.2 and 60 seconds")
    root = args.run_dir.expanduser().resolve()
    interactive = sys.stdout.isatty() and not args.plain
    signature = None
    try:
        while True:
            history = read_history(root / "metrics.csv")
            target = args.target_steps or target_from_config(root)
            latest = (value(history[-1], "step") if history else None)
            state = (latest, len(history), target)
            if state != signature or args.once:
                cols, rows = shutil.get_terminal_size(fallback=(100, 27))
                screen = render(history, run_name=root.name, target=target,
                                width=cols if interactive else 100,
                                height=rows if interactive else 27)
                if interactive:
                    sys.stdout.write("\x1b[2J\x1b[H" + screen + "\x1b[J")
                else:
                    print(screen)
                sys.stdout.flush()
                signature = state
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
