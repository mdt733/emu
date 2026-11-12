"""Read-only TTY dashboard for the EMU trainer subprocess.

No JAX/MJX imports, actor state, or curriculum decisions. The relay still
writes every original trainer line verbatim to console.log; this module only
chooses how much of that output to show on a human terminal.
"""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import sys
import threading
from collections import deque
from pathlib import Path


def console_mode(requested: str | None = None, stream=None) -> str:
    """Select a display, never a training configuration or checkpoint field."""
    stream = sys.stdout if stream is None else stream
    choice = (requested or os.environ.get("EMU_CONSOLE_MODE", "auto")).lower()
    if choice not in ("auto", "dashboard", "compact", "verbose"):
        raise ValueError("EMU_CONSOLE_MODE must be auto, dashboard, compact or verbose")
    if choice in ("compact", "verbose"):
        return choice
    tty = bool(getattr(stream, "isatty", lambda: False)())
    capable = (tty and os.environ.get("TERM", "") not in ("", "dumb")
               and "NO_COLOR" not in os.environ)
    return "dashboard" if capable else "compact"


def read_history(path: Path) -> list[dict[str, str]]:
    """Read the append-only metrics table after each infrequent log event.

    At 300M and one sample per 655k, this is only hundreds of rows.
    The full history survives checkpoint resume; no transient in-memory
    trend state is required and no trainer-side GPU transfer is added.
    """
    try:
        with path.open(newline="", encoding="utf-8") as file:
            # Same step may be logged on process boundary; latest wins.
            by_step: dict[int, dict[str, str]] = {}
            for row in csv.DictReader(file):
                try:
                    step = int(float(row["step"]))
                except (KeyError, TypeError, ValueError):
                    continue
                by_step[step] = row
        return [by_step[k] for k in sorted(by_step)]
    except (OSError, UnicodeError, csv.Error):
        return []


def metric_delta(history: list[dict], key: str, *, millions: float = 5.,
                 scale: float = 1., precision: int = 2,
                 suffix: str = "") -> str:
    """Signed change from a genuine roughly 5M-old sample, never a guess.

    Do not compare against a wildly old first row when there is a data gap.
    Delta direction alone does not establish quality (commands may change).
    """
    if not history:
        return "--"
    now = number(history[-1], "step")
    current = number(history[-1], key)
    if now is None or current is None:
        return "--"
    target = now - millions * 1e6
    eligible = [(abs(number(x, "step") - target), x)
                for x in history[:-1]
                if number(x, "step") is not None and number(x, key) is not None]
    if not eligible:
        return "--"
    diff, reference = min(eligible, key=lambda p: p[0])
    if diff > max(1e6, millions * 1e6 * .25):
        return "--"
    delta = (current - number(reference, key)) * scale
    return f"{delta:+.{precision}f}{suffix}"


def stair_window_trend(history: list[dict], *, millions: float = 5.) -> str:
    """Compare two disjoint sampled-rollout intervals, not an invented census.

    The nursery's sparse physical episode outcomes are sampled by metrics.csv
    every 10 updates. Explicitly label the statistic as sampled training
    evidence. No ratio is shown from fewer than 20 completed attempts.
    """
    if not history:
        return "awaiting data"
    now = number(history[-1], "step")
    if now is None:
        return "awaiting data"
    window = millions * 1e6
    def window_rate(start: float, end: float):
        rows = [r for r in history if start < (number(r, "step", -1)) <= end]
        events = sum(max(0., number(r, "academy_episodes", 0.)) for r in rows)
        successes = sum(max(0., number(r, "academy_short_climb_successes", 0.))
                        for r in rows)
        if events < 20 or successes > events or len(rows) < 2:
            return None
        return successes, events, 100. * successes / events
    current = window_rate(now-window, now)
    previous = window_rate(now-2*window, now-window)
    if current is None:
        return "5M sampled: insufficient completed episodes"
    s, n, pct = current
    if previous is None:
        return f"5M sampled: {int(s)}/{int(n)} ({pct:.1f}%) | prior window: --"
    old_rate = previous[2]
    return (f"5M sampled: {int(s)}/{int(n)} ({pct:.1f}%)  "
            f"Δ vs prior 5M {pct-old_rate:+.1f}pp")


def signed_stair_window(history: list[dict], direction: str,
                        *, millions:float=5.) -> str:
    """Physically supported near-frontier first-lip episodes, NOT a frozen exam."""
    if direction not in ('up','down') or not history:
        return '--'
    now=number(history[-1],'step')
    if now is None:
        return '--'
    rows=[r for r in history if now-millions*1e6 <
          number(r,'step',-1) <= now]
    attempts=sum(max(0.,number(row,f'academy_near_attempts_{direction}',0.))
                 for row in rows)
    successes=sum(max(0.,number(row,f'academy_near_successes_{direction}',0.))
                  for row in rows)
    if attempts<20 or successes>attempts:
        return 'insufficient encounters'
    return f'{successes:.0f}/{attempts:.0f} ({100*successes/attempts:.1f}%)'


def latest_heldout(history: list[dict]) -> str:
    """Latest actual frozen exam with delta versus the preceding exam."""
    seen = {}
    for row in history:
        step = number(row, "eval_policy_step", -1.)
        val = number(row, "eval_broad_score")
        if step >= 0 and val is not None and val > -100:
            seen[int(step)] = val
    if not seen:
        return "not yet evaluated"
    ordered = sorted(seen.items())
    at, val = ordered[-1]
    if len(ordered) == 1:
        return f"{val:.3f} at {at/1e6:.0f}M"
    old_at, old_val = ordered[-2]
    return (f"{val:.3f} at {at/1e6:.0f}M   "
            f"Δ since {old_at/1e6:.0f}M {val-old_val:+.3f}")


def number(row: dict, key: str, default: float | None = None) -> float | None:
    try:
        x = float(row.get(key, ""))
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def fmt(row: dict, key: str, spec: str = ".2f",
        *, scale: float = 1.0, missing: str = "--") -> str:
    val = number(row, key)
    return format(val * scale, spec) if val is not None else missing


def _rate(row: dict, success: str, attempts: str) -> str:
    yes = number(row, success, 0.)
    total = number(row, attempts, 0.)
    if total <= 0:
        return "no completed attempts"
    return f"{int(yes)}/{int(total)} ({100 * yes / total:.1f}%)"


def _duration(seconds: float) -> str:
    minutes = max(0, int(seconds) // 60)
    if minutes >= 1440:
        return f"{minutes // 1440}d {minutes % 1440 // 60}h"
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60}m"
    return f"{minutes}m"


def _pooled_outcomes(history: list[dict], successes: str, attempts: str,
                     *, millions: float, minimum: int) -> tuple[float, float] | None:
    """Pooled sampled episodes; absence and -1 sentinels are never successes."""
    if not history:
        return None
    now = number(history[-1], "step")
    if now is None:
        return None
    rows = [r for r in history
            if now - millions * 1e6 < number(r, "step", -1.) <= now
            and successes in r and attempts in r]
    if not rows:
        return None
    tried = sum(max(0., number(r, attempts, 0.)) for r in rows)
    passed = sum(max(0., number(r, successes, 0.)) for r in rows)
    if tried < minimum or passed > tried:
        return None
    return passed, tried


def _window_pass(history: list[dict], successes: str, attempts: str,
                 *, millions: float = 5., minimum: int = 20) -> str:
    pair = _pooled_outcomes(history, successes, attempts,
                            millions=millions, minimum=minimum)
    return f"{100. * pair[0] / pair[1]:.0f}%({pair[1]:.0f})" if pair else "--"


def _window_count(history: list[dict], successes: str, attempts: str,
                  *, millions: float = 5., minimum: int = 20) -> str:
    pair = _pooled_outcomes(history, successes, attempts,
                            millions=millions, minimum=minimum)
    return (f"{pair[0]:.0f}/{pair[1]:.0f} "
            f"({100. * pair[0] / pair[1]:.1f}%)") if pair else "--"


def _measured(row: dict, key: str, spec: str = ".1f",
              *, scale: float = 1., signed: bool = False) -> str:
    """Finite physical measurement; -1 is the trainer's missing sentinel."""
    val = number(row, key)
    if val is None or (val < 0. and not signed):
        return "--"
    return format(val * scale, spec)


def render_screen(row: dict[str, str], *, phase: str,
                  target_steps: int, run_name: str,
                  history: list[dict] | None = None,
                  events: tuple[str, ...] = (),
                  width: int = 88, height: int = 24) -> str:
    """Read-only specialist dashboard, designed to fit a 24-line WSL TTY.

    All cohort rates are sampled PPO evidence in disjoint physical episodes,
    NOT certification against an independent frozen exam. Missing/sparse
    cohorts display '--' rather than invented success or failure.
    """
    width = max(20, int(width))
    history = history or ([row] if row else [])
    step = number(row, "step", 0.)
    target = max(1, target_steps)
    progress = max(0., min(1., step / target))
    bar_n = max(8, min(32, width - 37))
    bar = "█" * round(progress * bar_n) + "░" * (bar_n - round(progress * bar_n))
    fps = number(row, "steps_per_sec", 0.)
    eta = _duration(max(0., target - step) / fps) if fps > 1. else "calculating"
    stage = str(row.get("school_stage") or row.get("teaching_profile") or "starting")
    walk_alpha = number(row, "school_walk_alpha")
    up_alpha = number(row, "school_stair_alpha")
    down_alpha = number(row, "school_down_stair_alpha")
    coach = lambda alpha: (f"{100. * (1. - alpha):.0f}%" if alpha is not None
                           and 0. <= alpha <= 1. else "--")
    pct = lambda key: _measured(row, key, ".0f", scale=100.)
    vx_lo = _measured(row, "frontier_vx_lo", ".2f", signed=True)
    up_height = _measured(row, "academy_height_up_m", ".1f", scale=100.)
    down_height = _measured(row, "academy_height_down_m", ".1f", scale=100.)
    yes = lambda s, a: _window_pass(history, s, a)
    cnt = lambda s, a: _window_count(history, s, a)
    entries: list[tuple[str, int]] = []

    def add(line: str, priority: int = 0) -> None:
        entries.append((line, priority))

    add(" EMU  |  " + run_name)
    add(f" {bar} {progress*100:5.1f}%   {step/1e6:,.1f}M / {target/1e6:,.1f}M")
    add(f" {phase} | {stage} | ETA ~{eta} (approx.)")
    add("", 3)

    add(f" MAIN       rew {fmt(row,'reward_mean','+.3f')} "
        f"({metric_delta(history,'reward_mean',precision=3)}) "
        f"falls {_measured(row,'fall_rate','.1f',scale=100.)}% "
        f"({metric_delta(history,'fall_rate',scale=100,precision=1,suffix='pp')})")
    add(f"            len {_measured(row,'ep_len_mean','.0f')}fr  "
        f"core {fmt(row,'core_competence','.2f')}  "
        f"vx actual/cmd {fmt(row,'vx_mean','+.2f')}/"
        f"{fmt(row,'cmd_vx_mean','+.2f')}  walk coach {coach(walk_alpha)}")

    add(f" COMMANDS   frontier vx +{_measured(row,'frontier_vx_hi','.2f')}/{vx_lo}m/s "
        f"vy ±{_measured(row,'frontier_vy','.2f')}m/s "
        f"yaw ±{_measured(row,'frontier_yaw','.2f')}rad/s")
    add("            5M near pass vx+ " +
        yes("command_probe_successes_vx_hi", "command_probe_attempts_vx_hi") +
        " vx- " + yes("command_probe_successes_vx_lo", "command_probe_attempts_vx_lo") +
        " vy+ " + yes("command_probe_successes_vy_pos", "command_probe_attempts_vy_pos") +
        " vy- " + yes("command_probe_successes_vy_neg", "command_probe_attempts_vy_neg"), 1)
    add("            5M near pass yaw+ " +
        yes("command_probe_successes_yaw_pos", "command_probe_attempts_yaw_pos") +
        " yaw- " + yes("command_probe_successes_yaw_neg", "command_probe_attempts_yaw_neg") +
        " mixed " + yes("command_probe_successes_mixed", "command_probe_attempts_mixed"), 1)
    if height >= 31:
        add(f"            settled command tracking vx {pct('precision_vx')}% "
            f"vy {pct('precision_vy')}% yaw {pct('precision_yaw')}%", 2)

    add(f" SLOPES     frontier {_measured(row,'slope_max_deg','.1f')}°  "
        f"sampled max {_measured(row,'physical_slope_deg_max','.1f')}° "
        f"exposure {pct('slope_exposure')}%")
    add("            5M near STABLE up " +
        yes("slope_near_stability_successes_uphill", "slope_near_attempts_uphill") +
        " down " + yes("slope_near_stability_successes_downhill", "slope_near_attempts_downhill") +
        " L " + yes("slope_near_stability_successes_cross_left", "slope_near_attempts_cross_left") +
        " R " + yes("slope_near_stability_successes_cross_right", "slope_near_attempts_cross_right"), 1)
    if height >= 31:
        add("            5M easier stable U " +
            yes("slope_easy_stability_successes_uphill", "slope_easy_attempts_uphill") +
            " D " + yes("slope_easy_stability_successes_downhill", "slope_easy_attempts_downhill"), 2)

    add(f" STEPS      frontier +{up_height}/-{down_height}cm  "
        "angle tolerance -- (no frozen oblique exam)")
    add("            5M near UP " +
        cnt("academy_near_successes_up", "academy_near_attempts_up") +
        "  DOWN " + cnt("academy_near_successes_down", "academy_near_attempts_down"))
    add(f"            coaching U/D {coach(up_alpha)}/{coach(down_alpha)}  "
        f"assisted windows U {_measured(row,'school_up_good_windows','.0f')}/2 "
        f"D {_measured(row,'school_down_good_windows','.0f')}/2", 1)
    add(f"            physical footholds {_measured(row,'academy_footholds','.0f')} "
        f"transfers {_measured(row,'academy_transfers','.0f')} "
        f"crossings U/D {_measured(row,'academy_up_crossings','.0f')}/"
        f"{_measured(row,'academy_down_crossings','.0f')}", 1)
    if height >= 31:
        add(f"            stages UP {stage} DOWN {row.get('school_down_stage','--')} "
            f"uncoached U {_measured(row,'school_up_nohelp_windows','.0f')}/2 "
            f"D {_measured(row,'school_down_nohelp_windows','.0f')}/2", 2)
        add("            5M anchor UP " +
            yes("academy_anchor_successes_up", "academy_anchor_attempts_up") +
            " DOWN " + yes("academy_anchor_successes_down", "academy_anchor_attempts_down"), 2)

    add(f" RESILIENCE push {_measured(row,'frontier_push_n','.2f')}N  "
        f"difficulty {_measured(row,'resilience_difficulty','.2f')}  "
        "5M focus fall " +
        _window_pass(history, "focus_falls_resilience", "focus_episodes_resilience",
                     minimum=10))
    add(f" TRANSITIONS precision {pct('transition_precision')}%  "
        f"ramp {_measured(row,'frontier_ramp_steps','.0f')}fr  "
        f"hold {_measured(row,'frontier_transition_hold_s','.2f')}s  "
        f"5M focus fall " +
        _window_pass(history, "focus_falls_transition", "focus_episodes_transition",
                     minimum=10))
    add(f" CONTROL    shaft {_measured(row,'control_shaft_j_per_m','.1f')}J/m "
        f"episode quality {_measured(row,'control_episode_quality','.2f')} "
        f"motor over-rated {pct('servo_over_rated_frac')}%")
    if height >= 31:
        add(f"            focus share {pct('focus_step_share_control')}% "
            f"motor near stall L/R (non-stair) "
            f"{pct('hip_roll_motor_near_stall_left_other')}%/"
            f"{pct('hip_roll_motor_near_stall_right_other')}%", 2)

    add(f" TRAINING   PPO {fmt(row,'train_steps_per_sec',',.0f')}/s  "
        f"wall {fmt(row,'steps_per_sec',',.0f')}/s  "
        f"KL {fmt(row,'ppo_approx_kl_final','.4f')}  "
        f"NaN skips {fmt(row,'nan_grad_skips_total','.0f')}")
    add(" FROZEN     walking retention score: " + latest_heldout(history))

    # In a short terminal discard optional details, never silently erase
    # specialist headings or silently fall back to printing raw trainer lines.
    recent = tuple(dict.fromkeys(events))
    add(" EVENTS     " + (recent[-1] if recent else "See console.log"))
    if height >= 31 and len(recent) > 1:
        add("            " + recent[-2], 2)
    add(" Full log: console.log   Raw evidence: metrics.csv, fixed_eval.csv")
    max_lines = max(4, int(height) - 2)
    for priority in (3, 2, 1):
        while len(entries) > max_lines:
            index = next((i for i, (_, p) in enumerate(entries) if p == priority),
                         None)
            if index is None:
                break
            entries.pop(index)
    if len(entries) > max_lines:
        entries = entries[:max_lines - 1] + entries[-1:]
    return "\n".join(line[:width - 1] for line, _ in entries)


class ConsoleRelay:
    """Only the parent process writes ANSI; the trainer/log remain unmodified."""

    STEP = re.compile(r"^\[emu_[^\]]+\] step=[\d,]+ ")
    IMPORTANT = (
        " WARNING:", " AUTO-RECOVERY", "rollback_count=",
        " SCHOOL ", " PPO TARGET REACHED", "training complete",
        "Traceback", "Error:", "fixed-eval starting",
    )
    VERBOSE_PREFIXES = (
        " step=", "   lessons ", "   speed ", " training start ",
        " one cold-start ", " obs_dim=", " compiling MJX/JAX update",
        " compile+first_update=", " fixed-eval ", " PPO TARGET REACHED",
        " final ", " WARNING:",
    )

    def __init__(self, run_dir: Path, *, target_steps: int,
                 stream=None, mode: str | None = None):
        self.stream = sys.stdout if stream is None else stream
        self.mode = console_mode(mode, self.stream)
        self.run_dir = Path(run_dir)
        self.target_steps = int(target_steps)
        self.events: deque[str] = deque(maxlen=4)
        self.phase = "Initialising H40 physics / first PPO compilation"
        self.row: dict[str, str] = {}
        self.history: list[dict[str, str]] = []
        self.lock = threading.Lock()
        self.drawn = False

    def _write(self, message: str) -> None:
        self.stream.write(message)
        self.stream.flush()

    def _redraw(self) -> None:
        if self.mode != "dashboard":
            return
        terminal = shutil.get_terminal_size(fallback=(88, 24))
        picture = render_screen(self.row, phase=self.phase,
                                target_steps=self.target_steps,
                                run_name=self.run_dir.name, history=self.history,
                                events=tuple(self.events),
                                width=terminal.columns, height=terminal.lines)
        if not self.drawn:
            self._write("\x1b[?25l\x1b[2J\x1b[H" + picture + "\x1b[J")
            self.drawn = True
        else:
            # Clear previous entire page. Cursor-home+erase-tail left
            # visible old-line fragments on WSL terminal resizes.
            self._write("\x1b[2J\x1b[H" + picture + "\x1b[J")

    def forward(self, line: str) -> None:
        t = line.strip()
        if not t:
            return
        is_progress = bool(self.STEP.match(t))
        event = any(flag in t for flag in self.IMPORTANT)
        with self.lock:
            if is_progress:
                self.history = read_history(self.run_dir / "metrics.csv")
                self.row = self.history[-1] if self.history else {}
                self.phase = "PPO training"
            elif "compiling MJX/JAX update" in t:
                self.phase = "Compiling first PPO update"
            elif "fixed-eval starting" in t:
                self.phase = "Frozen walking evaluation: PPO updates paused"
            elif "PPO TARGET REACHED" in t:
                self.phase = "Target reached: final exam / checkpoint"
            elif "training complete" in t:
                self.phase = "Training complete"
            if event:
                self.events.append(re.sub(r"^\[emu_[^\]]+\]\s*", "", t)[:120])
            if self.mode == "dashboard":
                if is_progress or event or not self.drawn:
                    self._redraw()
            elif self.mode == "compact":
                if is_progress:
                    brief = self.row
                    print(
                        f"    step={fmt(brief,'step',',.0f')} "
                        f"stage={brief.get('school_stage','-')} "
                        f"rew={fmt(brief,'reward_mean','+.3f')} "
                        f"Δ5M={metric_delta(self.history,'reward_mean',precision=3)} "
                        f"fall={fmt(brief,'fall_rate','.1f',scale=100)}% "
                        f"Δ5M={metric_delta(self.history,'fall_rate',scale=100,precision=1,suffix='pp')} "
                        f"stairs={stair_window_trend(self.history)}",
                        file=self.stream,flush=True)
                elif event or "compiling MJX/JAX update" in t or "compile+first_update=" in t:
                    print("    " + t[:400], file=self.stream, flush=True)
            else:
                match = re.match(r"^\[emu_[^\]]+\]", t)
                prefix = match.group() if match else ""
                visible = (is_progress or event or
                           any(t.startswith(prefix + p)
                               for p in self.VERBOSE_PREFIXES)
                           if prefix else event)
                if visible:
                    print("    " + t[:400], file=self.stream, flush=True)

    def pulse(self, message: str) -> None:
        with self.lock:
            if self.mode == "dashboard":
                self.phase = message
                self._redraw()
            else:
                print(message, file=self.stream, flush=True)

    def close(self) -> None:
        with self.lock:
            if self.drawn:
                # Leave the final dashboard visible, restore the normal cursor.
                self._write("\x1b[?25h\n")
                self.drawn = False


def watch_run(run_dir: Path, *, target_steps: int, interval: float = 2.) -> None:
    """External read-only dashboard for an already running PPO experiment.

    Running trainers import their own version of the console at startup.
    This allows the new display immediately, without restarting that trainer.
    """
    import time
    previous = None
    try:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        while True:
            history = read_history(run_dir / "metrics.csv")
            row = history[-1] if history else {}
            size = shutil.get_terminal_size(fallback=(88, 24))
            picture = render_screen(
                row, phase="READ-ONLY live metrics",
                target_steps=target_steps, run_name=run_dir.name,
                history=history, width=size.columns, height=size.lines)
            if picture != previous:
                sys.stdout.write("\x1b[2J\x1b[H" + picture + "\x1b[J")
                sys.stdout.flush()
                previous = picture
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()


def main() -> int:
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Read-only live specialist dashboard; never touches PPO or checkpoint.")
    parser.add_argument("--watch", type=Path, required=True,
                        help="Existing training run directory containing metrics.csv")
    parser.add_argument("--target-steps", type=int,
                        help="Override training_config.json total steps")
    parser.add_argument("--interval", type=float, default=2.)
    args = parser.parse_args()
    run_dir = args.watch.expanduser()
    target = args.target_steps
    if target is None:
        try:
            data = json.loads((run_dir / "training_config.json").read_text())
            target = int(data["steps"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"Cannot find target steps in training_config.json: {exc}; "
                         "supply --target-steps")
    if target <= 0 or not math.isfinite(args.interval) or args.interval < .25:
        parser.error("Require positive --target-steps and --interval >= 0.25")
    if not sys.stdout.isatty():
        parser.error("--watch requires an interactive terminal")
    watch_run(run_dir, target_steps=target, interval=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
