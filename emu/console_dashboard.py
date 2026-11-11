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


def last_metrics(path: Path) -> dict[str, str]:
    """CSV is authoritative; never parse a printed teaching reward as proof."""
    try:
        with path.open(newline="", encoding="utf-8") as file:
            row = {}
            for candidate in csv.DictReader(file):
                row = candidate
            return row
    except (OSError, UnicodeError, csv.Error):
        return {}


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
                  events: tuple[str, ...] = (),
                  width: int = 88, height: int = 24) -> str:
    """Pure, bounded text screen, suitable for regression tests and WSL terminals."""
    width = max(40, int(width))
    step = number(row, "step", 0.)
    target = max(1, target_steps)
    progress = max(0., min(1., step / target))
    bar_n = max(8, min(36, width - 37))
    filled = round(progress * bar_n)
    bar = "█" * filled + "░" * (bar_n - filled)
    fps = number(row, "steps_per_sec", 0.)
    remaining = max(0., target - step)
    eta = _duration(remaining / fps) if fps > 1. else "calculating"
    school = str(row.get("school_stage") or row.get("teaching_profile") or "starting")
    walk_alpha = number(row, "school_walk_alpha")
    stair_alpha = number(row, "school_stair_alpha")
    walking = (f"{100 * (1 - walk_alpha):.0f}% coaching left"
               if walk_alpha is not None else "not reported")
    stair = (f"{100 * (1 - stair_alpha):.0f}% coaching left"
             if stair_alpha is not None else "not reported")
    eval_score = number(row, "eval_broad_score")
    eval_step = number(row, "eval_policy_step", -1.)
    exam = (f"{eval_score:.3f} at {eval_step/1e6:.0f}M (held-out WALK)"
            if eval_score is not None and eval_score > -100. and eval_step >= 0
            else "not yet evaluated")
    h_up = fmt(row, "academy_height_up_m", ".1f", scale=100)
    h_down = fmt(row, "academy_height_down_m", ".1f", scale=100)
    lines = [
        " EMU  |  LIVE TRAINING  |  " + run_name,
        "─" * min(width - 1, 88),
        f" {bar}  {progress*100:5.1f}%    {step/1e6:,.1f}M / {target/1e6:,.1f}M",
        f" {phase}    ETA ~{eta} (wall rate, approximate)",
        "",
        " WALKING  [current training rollout; not a held-out score]",
        (f"  Reward {fmt(row,'reward_mean','+.3f')}    Falls "
         f"{fmt(row,'fall_rate','.1f',scale=100)}%    Length "
         f"{fmt(row,'ep_len_mean','.0f')}fr    Core "
         f"{fmt(row,'core_competence','.2f')}"),
        (f"  vX actual / requested {fmt(row,'vx_mean','+.2f')} / "
         f"{fmt(row,'cmd_vx_mean','+.2f')} m/s   "
         f"Coach: {walking}"),
        (f"  Command frontier {fmt(row,'frontier_vx_lo','+.2f')} .. "
         f"{fmt(row,'frontier_vx_hi','+.2f')} m/s    "
         f"Resilience {fmt(row,'resilience_difficulty','.2f')}"),
        "",
        " STAIRS  [last logged training rollout; success = supported climb]",
        (f"  +5cm short climbs: {_rate(row,'academy_short_climb_successes','academy_episodes')}"),
        (f"  Upper footholds {fmt(row,'academy_footholds','.0f')}    "
         f"Supported transfers {fmt(row,'academy_transfers','.0f')}    "
         f"Up crossings {fmt(row,'academy_up_crossings','.0f')}"),
        (f"  Frontiers: +{h_up}cm / -{h_down}cm    "
         f"Stair coach: {stair}"),
        (f"  Proof windows {fmt(row,'school_up_good_windows','.0f')}/2    "
         f"Academy birth share {fmt(row,'academy_expected_birth_fraction','.0f',scale=100)}%"),
        "",
        " TRAINER / INDEPENDENT EVALUATION",
        (f"  GPU PPO {fmt(row,'train_steps_per_sec',',.0f')}/s    "
         f"Wall {fmt(row,'steps_per_sec',',.0f')}/s    "
         f"KL {fmt(row,'ppo_approx_kl_final','.4f')}    "
         f"NaN skips {fmt(row,'nan_grad_skips_total','.0f')}"),
        f"  Frozen broad score: {exam}",
        "",
        " RECENT EVENTS",
    ]
    if events:
        lines.extend("  " + item for item in tuple(events)[-2:])
    else:
        lines.append("  Waiting for noteworthy transitions...")
    lines.append("  Raw logs and metrics: run directory (console.log, metrics.csv)")
    # Never wrap at the terminal edge: doing so would cause scroll and destroy
    # the fixed-screen experience. Leave one spare row for terminal cursors.
    available = max(8, int(height) - 2)
    if len(lines) > available:
        # On 18- to 21-row terminals, prioritise progress, walking, stair
        # proof and latest event over less immediately useful details.
        for label in ("", "  Command frontier", "  Frontiers:"):
            if len(lines) <= available:
                break
            for i, value in enumerate(lines):
                if value.startswith(label) and (label or value == ""):
                    lines.pop(i)
                    break
        if len(lines) > available:
            lines = lines[:available - 1] + [lines[-1]]
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
                                run_name=self.run_dir.name, events=tuple(self.events),
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
                self.row = last_metrics(self.run_dir / "metrics.csv")
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
                    print("    " + t[:180], file=self.stream, flush=True)
                elif event or "compiling MJX/JAX update" in t or "compile+first_update=" in t:
                    print("    " + t[:400], file=self.stream, flush=True)
            else:
                if (is_progress or "AUTO-RECOVERY" in t or
                        "training complete" in t or "Traceback" in t or
                        any(t.startswith(re.match(r"^\[emu_[^\]]+\]", t).group() + p)
                            for p in self.VERBOSE_PREFIXES)
                        if re.match(r"^\[emu_[^\]]+\]", t) else
                        "Traceback" in t):
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
