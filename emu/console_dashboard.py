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


def render_screen(row: dict[str, str], *, phase: str,
                  target_steps: int, run_name: str,
                  history: list[dict] | None = None,
                  events: tuple[str, ...] = (),
                  width: int = 88, height: int = 24) -> str:
    """Pure 24-row dashboard, with real 5M trends and no terminal wrapping."""
    width = max(20, int(width))
    history = history or ([row] if row else [])
    step = number(row, "step", 0.)
    target = max(1, target_steps)
    progress = max(0., min(1., step / target))
    bar_n = max(8, min(32, width - 37))
    bar = "█" * round(progress * bar_n) + "░" * (bar_n - round(progress * bar_n))
    fps = number(row, "steps_per_sec", 0.)
    remaining = max(0., target - step)
    eta = _duration(remaining / fps) if fps > 1. else "calculating"
    walk_alpha = number(row, "school_walk_alpha")
    stair_alpha = number(row, "school_stair_alpha")
    walking = (f"{100 * (1 - walk_alpha):.0f}% coaching left"
               if walk_alpha is not None else "--")
    stair = (f"{100 * (1 - stair_alpha):.0f}% coaching left"
             if stair_alpha is not None else "--")
    h_up = fmt(row, "academy_height_up_m", ".1f", scale=100)
    h_down = fmt(row, "academy_height_down_m", ".1f", scale=100)
    stage = str(row.get("school_stage") or row.get("teaching_profile") or "starting")
    lines = [
        " EMU  |  " + run_name,
        f" {bar} {progress*100:5.1f}%   {step/1e6:,.1f}M / {target/1e6:,.1f}M",
        f" {phase} | {stage} | ETA ~{eta} (approx.)",
        "",
        " WALKING  [rollout data; Δ~5M in parentheses]",
        (f"  Reward {fmt(row,'reward_mean','+.3f')} "
         f"({metric_delta(history,'reward_mean',precision=3)})   "
         f"Falls {fmt(row,'fall_rate','.1f',scale=100)}% "
         f"({metric_delta(history,'fall_rate',scale=100,precision=1,suffix='pp')})"),
        (f"  Length {fmt(row,'ep_len_mean','.0f')}fr "
         f"({metric_delta(history,'ep_len_mean',precision=0)})   "
         f"Core {fmt(row,'core_competence','.2f')} "
         f"({metric_delta(history,'core_competence')})"),
        (f"  vx actual/cmd {fmt(row,'vx_mean','+.2f')}/"
         f"{fmt(row,'cmd_vx_mean','+.2f')} m/s   "
         f"Walk teacher {walking}"),
        "",
        " STAIRS  [sampled training outcomes, NOT frozen stair exams]",
        "  " + stair_window_trend(history),
        (f"  Latest rollout climbs: "
         f"{_rate(row,'academy_short_climb_successes','academy_episodes')}"),
        (f"  Footholds {fmt(row,'academy_footholds','.0f')}  "
         f"Transfers {fmt(row,'academy_transfers','.0f')}  "
         f"Up crossings {fmt(row,'academy_up_crossings','.0f')}"),
        (f"  Frontier +{h_up}/-{h_down}cm   Stair teacher {stair}   "
         f"Proof {fmt(row,'school_up_good_windows','.0f')}/2"),
        "",
        " TRAINING / FROZEN WALKING RETENTION",
        (f"  PPO {fmt(row,'train_steps_per_sec',',.0f')}/s   "
         f"Wall {fmt(row,'steps_per_sec',',.0f')}/s   "
         f"KL {fmt(row,'ppo_approx_kl_final','.4f')}   "
         f"NaN skips {fmt(row,'nan_grad_skips_total','.0f')}"),
        f"  Frozen broad score: {latest_heldout(history)}",
        "",
        " RECENT EVENTS",
    ]
    lines.extend("  " + item for item in (tuple(events)[-2:] or
                                            ("No unusual events",)))
    lines.append("  Full log: console.log   Raw evidence: metrics.csv, fixed_eval.csv")
    # Don't scroll: all lines fit inside available terminal rows; remove
    # optional blank lines and least essential decorations first.
    max_lines = max(4, height - 2)
    while len(lines) > max_lines and "" in lines:
        lines.remove("")
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [lines[-1]]
    return "\n".join(line[:width - 1] for line in lines)


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
            self._write("\x1b[H" + picture + "\x1b[J")

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
                self.phase = "PPO training  |  " + str(self.row.get("school_stage", "learning"))
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
