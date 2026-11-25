"""Read-only learning evidence from existing PPO experience.

No actor inputs, reward changes, teacher feedback or extra MJX worlds.
Windows contain raw integer counts, not pooled mastery declarations.
Process restarts/rollbacks are explicit boundaries; incomplete episodes
are never manufactured into attempts.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np

SCHEMA = "emu_live_skill_window_v1"
WINDOW_STEPS = 1_000_000
FOCUS = ("command", "resilience", "slope", "step", "transition", "control")
LIP_EDGES_MM = (2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 55)
GRADE_EDGES_DEG = (0.25, 1, 2, 4, 6, 8, 10, 15)


class LiveSkillObserver:
    """Aggregate sparse evidence, publishing one append-only stream per process.

    Existing field events are reused, not re-simulated. Relative recovery
    severity and terminal slope geometry are NOT fixed physical reference
    tasks; comparisons against them must respect changing lesson exposure.
    """
    def __init__(self, folder, *, start_step, worlds, field_shape, field_bounds,
                 source_commit, window_steps=WINDOW_STEPS):
        if worlds <= 0 or window_steps <= 0:
            raise ValueError("Invalid observer dimensions")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"segment-{int(start_step)}-{uuid.uuid4().hex}.jsonl"
        self.header = dict(schema=SCHEMA, worlds=int(worlds),
                           source_commit=str(source_commit),
                           field_shape=list(map(int, field_shape)),
                           field_bounds=np.asarray(field_bounds).tolist(),
                           lip_edges_mm=list(LIP_EDGES_MM),
                           grade_edges_deg=list(GRADE_EDGES_DEG),
                           process_start_step=int(start_step))
        self.start = self.last = int(start_step)
        self.window_steps = int(window_steps)
        self.counts = {}
        self.updates = 0
        self.epoch = 0
        self.host_seconds = 0.

    def _add(self, key, attempts, passes):
        n, k = int(attempts), int(passes)
        if not 0 <= k <= n:
            raise ValueError(f"Invalid skill count {key}: {k}/{n}")
        if not n:
            return
        old = self.counts.get(key, (0, 0))
        self.counts[key] = (old[0] + n, old[1] + k)

    def _binned(self, prefix, bins, passed):
        bins = np.asarray(bins, dtype=np.int64).ravel()
        passed = np.asarray(passed, dtype=bool).ravel()
        if bins.shape != passed.shape:
            raise ValueError("Skill bin/outcome mismatch")
        if not bins.size:
            return
        ids, inv = np.unique(bins, return_inverse=True)
        n = np.bincount(inv)
        k = np.bincount(inv, weights=passed).astype(np.int64)
        for cell, a, p in zip(ids, n, k):
            self._add(f"{prefix}/{int(cell)}", a, p)

    def field_events(self, family, cell_ids, successes):
        """Raw verdicts from the existing capability event machines.

        COMMAND counts windows, RESILIENCE counts birth recovery windows,
        TRANSITION counts command maneuvers; none are mislabeled episodes.
        Cell encodings are the existing field's flattened grid indices.
        """
        if family not in ("command_window", "recovery_relative", "transition_edge"):
            raise ValueError("Unknown field family")
        t = time.perf_counter()
        self._binned(family, cell_ids, successes)
        self.host_seconds += time.perf_counter() - t

    def observe(self, metrics, *, step):
        t = time.perf_counter()
        if int(step) <= self.last:
            raise ValueError("Repeated or unordered rollout")
        done = np.asarray(metrics["episode_done"]) > .5
        if done.ndim != 2 or done.shape[1] != self.header["worlds"]:
            raise ValueError("Wrong live rollout shape")
        # Slice only terminal rows; avoid repeated reductions over 131k frames.
        def terminal(name):
            a = np.asarray(metrics[name])
            if a.shape != done.shape:
                raise ValueError("Wrong metric shape: " + name)
            return a[done]
        role = terminal("specialist_focus").astype(int)
        survived = terminal("episode_fell") < .5
        for i, name in enumerate(FOCUS):
            sel = role == i
            self._add(f"role_survival/{name}", sel.sum(), (sel & survived).sum())
        academy = terminal("academy_active") > .5
        lip = terminal("skill_first_birth_lip_m") * 1000.
        for sign, direction in ((1, "up"), (-1, "down")):
            take = academy & (sign * lip > 1.5) & np.isfinite(lip)
            bins = np.searchsorted(LIP_EDGES_MM, np.abs(lip[take]), side="right")
            # All gates share the denominator of completed first-lip births,
            # including failures to reach the lip. No later-lip substitution.
            for gate, value in (
                ("survived", survived),
                ("first_lip_armed", terminal("academy_episode_first_lip_armed") > .5),
                ("bilateral_ready", terminal("academy_episode_two_foot_ready") > .5),
                ("goal_completed", terminal("skill_first_goal_completed") > .5),
                ("safe_bilateral", (terminal("academy_episode_two_foot_ready") > .5) & survived)):
                self._binned(f"academy_first_birth/{direction}/{gate}", bins, value[take])
        # Existing ordinary STEP success is first-encounter + safe timeout,
        # not a birth-denominator Academy gate.
        height = terminal("physical_step_height_m") * 1000.
        for direction in ("up", "down"):
            take = terminal(f"terrain_episode_attempt_{direction}") > .5
            bins = np.searchsorted(LIP_EDGES_MM, height[take], side="right")
            passed = terminal(f"terrain_episode_success_{direction}") > .5
            self._binned(f"ordinary_first_encounter/{direction}/safe_cross", bins, passed[take])
        grade = terminal("physical_slope_deg")
        for direction in ("uphill", "downhill", "cross_left", "cross_right"):
            take = terminal(f"slope_episode_attempt_{direction}") > .5
            bins = np.searchsorted(GRADE_EDGES_DEG, grade[take], side="right")
            for gate in ("stable", "success"):
                passed = terminal(f"slope_episode_{gate}_{direction}") > .5
                self._binned(f"slope_terminal_grade/{direction}/{gate}", bins, passed[take])
        self.last = int(step)
        self.updates += 1
        self.host_seconds += time.perf_counter() - t

    def flush(self, *, reason="window", force=False):
        if not self.updates or (not force and self.last - self.start < self.window_steps):
            return
        row = dict(self.header, start_step=self.start, end_step=self.last,
                   policy_epoch=self.epoch, updates=self.updates, reason=reason,
                   host_aggregation_seconds=self.host_seconds,
                   cells={key: {"attempts": n, "passes": k}
                          for key, (n, k) in sorted(self.counts.items())})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
        self.start = self.last
        self.counts.clear()
        self.updates = 0
        self.host_seconds = 0.

    def rollback(self):
        self.flush(reason="before_rollback", force=True)
        self.epoch += 1
