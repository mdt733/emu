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
                 source_commit, window_steps=WINDOW_STEPS,
                 recovery_bins=6, edge_directions=(), field_evidence_from_taught=False):
        if worlds <= 0 or window_steps <= 0:
            raise ValueError("Invalid observer dimensions")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"segment-{int(start_step)}-{uuid.uuid4().hex}.jsonl"
        from .live_skill_reference import (REFERENCE_KICKS_MPS, REFERENCE_VX_MPS,
            REFERENCE_SENSOR_SCALE, REFERENCE_STEP_COMMAND_M, SETTLE_DEADLINE_S, SETTLE_FRAMES)
        self.header = dict(schema=SCHEMA, worlds=int(worlds),
                           reference_kicks_mps=list(REFERENCE_KICKS_MPS),
                           reference_vx_mps=REFERENCE_VX_MPS,
                           reference_sensor_scale=REFERENCE_SENSOR_SCALE,
                           reference_step_command_m=REFERENCE_STEP_COMMAND_M,
                           reference_settle_deadline_s=SETTLE_DEADLINE_S,
                           reference_settle_frames=SETTLE_FRAMES,
                           reference_horizon_s=SETTLE_DEADLINE_S,
                           reference_mastery_gate='settled_and_survived',
                           source_commit=str(source_commit),
                           field_shape=list(map(int, field_shape)),
                           field_bounds=np.asarray(field_bounds).tolist(),
                           recovery_relative_bins=int(recovery_bins),
                           transition_edge_directions=np.asarray(edge_directions).tolist(),
                           field_evidence_from_taught=bool(field_evidence_from_taught),
                           lip_edges_mm=list(LIP_EDGES_MM),
                           grade_edges_deg=list(GRADE_EDGES_DEG),
                           process_start_step=int(start_step))
        self.start = self.last = int(start_step)
        self.window_steps = int(window_steps)
        self.counts = {}
        self.updates = 0
        self.epoch = 0
        self.host_seconds = 0.
        self.reference_frames=np.zeros(12,dtype=np.int64)
        self.world_frames=0

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
        if np.any((np.asarray(successes) != 0) & (np.asarray(successes) != 1)):
            raise ValueError("Nonbinary capability verdict")
        self._binned(family, cell_ids, successes)
        self.host_seconds += time.perf_counter() - t

    def observe(self, metrics, *, step):
        t = time.perf_counter()
        if int(step) <= self.last:
            raise ValueError("Repeated or unordered rollout")
        done = np.asarray(metrics["episode_done"]) > .5
        if done.ndim != 2 or done.shape[1] != self.header["worlds"]:
            raise ValueError("Wrong live rollout shape")
        self.world_frames+=done.size
        # Gather only terminal rows, once per metric. Reuse one flat index.
        ids = np.flatnonzero(done.ravel())
        cache = {}
        def terminal(name):
            if name in cache:
                return cache[name]
            a = np.asarray(metrics[name])
            if a.shape != done.shape:
                raise ValueError("Wrong metric shape: " + name)
            cache[name] = a.ravel()[ids]
            return cache[name]
        role = terminal("specialist_focus").astype(int)
        survived = terminal("episode_fell") < .5
        for i, name in enumerate(FOCUS):
            sel = role == i
            self._add(f"role_survival/{name}", sel.sum(), (sel & survived).sum())
        # Separate fixed, signed, flat reference tasks from changing ordinary
        # recovery severity. No survival-only verdict can count as settling.
        teacher_counts=[[0,0] for _ in range(6)]
        if 'skill_reference_id' in metrics:
            assigned=np.asarray(metrics['skill_reference_id'])
            if assigned.shape!=done.shape or np.any((assigned<-1)|(assigned>=12)):
                raise ValueError('Unknown reference task or rollout shape')
            # Exposure is measured in actual PPO frames, NOT attempts or
            # nominal birth probability. Short trials change occupancy.
            self.reference_frames+=np.bincount(assigned[assigned>=0].astype(np.int64),minlength=12)
            reference=terminal('skill_reference_id').astype(int)
            take=reference>=0
            lengths=terminal('episode_length')[take]
            if np.any((lengths<1)|(lengths>150)) or np.any(survived[take] & (lengths!=150)):
                raise ValueError('Wrong reference first-life horizon')
            vx=terminal('skill_birth_vx_error')
            vy=terminal('skill_birth_vy_error')
            from .live_skill_reference import REFERENCE_KICKS_MPS
            expected=np.asarray(REFERENCE_KICKS_MPS)[reference[take]%6]
            if not np.allclose(vx[take],expected,atol=1e-5) or not np.allclose(vy[take],0.,atol=1e-5):
                raise ValueError('Reference physical birth velocity mismatch')
            for prefix, cohort in (('reference_recovery',take & (reference<6)),
                                   ('practice_recovery',reference>=6)):
                settled=terminal('skill_reference_settled')[cohort]>.5
                self._binned(prefix+'/settled',reference[cohort]%6,settled)
                self._binned(prefix+'/survived',reference[cohort]%6,survived[cohort])
                mission=settled & survived[cohort]
                self._binned(prefix+'/mission',reference[cohort]%6,mission)
                if prefix=='reference_recovery':
                    cases=reference[cohort]
                    n=np.bincount(cases,minlength=6)
                    k=np.bincount(cases,weights=mission,minlength=6).astype(np.int64)
                    teacher_counts=[[int(a),int(b)] for a,b in zip(n,k)]
            # Ordinary physical birth residuals include carryover and are
            # observational cohorts, not isolated kicks or matched references.
            ordinary=(role==1)&(~take)&np.isfinite(vx)&np.isfinite(vy)
            speed=np.hypot(vx[ordinary],vy[ordinary])
            bins=np.searchsorted((.10,.25,.40,.60,.80,1.),speed,side='right')
            self._binned('ordinary_recovery_birth_residual/survival',bins,survived[ordinary])
        academy = terminal("academy_active") > .5
        lip = terminal("skill_first_birth_lip_m") * 1000.
        if 'skill_uphill_id' in metrics:
            from .live_skill_uphill import UP_HEIGHTS_M,UP_FRAMES
            ids=terminal('skill_uphill_id').astype(int)
            fixed=ids>=0
            if np.any(ids[fixed]>=12) or np.any(fixed & ~academy):
                raise ValueError('Invalid uphill cohort identity')
            if fixed.any():
                expected=np.asarray(UP_HEIGHTS_M)[ids[fixed]%6]*1000.
                if not np.allclose(lip[fixed],expected,atol=.05):
                    raise ValueError('Physical uphill lip mismatch')
                length=terminal('episode_length')
                if np.any(length[fixed]>UP_FRAMES) or np.any(fixed & survived & (length!=UP_FRAMES)):
                    raise ValueError('Wrong uphill first-life horizon')
                for prefix,cohort in (('reference_uphill',fixed & (ids<6)),('practice_uphill',ids>=6)):
                    for gate in ('approach','first_contact','each_foot_five','traversed'):
                        self._binned(prefix+'/'+gate,ids[cohort]%6,(terminal('skill_stair_'+gate)>.5)[cohort])
                    self._binned(prefix+'/mission',ids[cohort]%6,(survived & (terminal('skill_stair_traversed')>.5))[cohort])
                    self._binned(prefix+'/survived',ids[cohort]%6,survived[cohort])
            academy=academy & ~fixed
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
            if 'skill_stair_first_contact' in metrics:
                for gate in ('approach','first_contact','each_foot_five','traversed'):
                    passed=terminal('skill_stair_'+gate)>.5
                    self._binned(f'academy_physical_first_lip/{direction}/{gate}',bins,passed[take])
                self._binned(f'academy_physical_first_lip/{direction}/traversed_and_survived',bins,
                             (survived & (terminal('skill_stair_traversed')>.5))[take])
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
        return teacher_counts

    def flush(self, *, reason="window", force=False):
        if not self.updates or (not force and self.last - self.start < self.window_steps):
            return
        row = dict(self.header, start_step=self.start, end_step=self.last,
                   policy_epoch=self.epoch, updates=self.updates, reason=reason,
                   host_aggregation_seconds=self.host_seconds,
                   world_frames=int(self.world_frames),
                   reference_frames_by_case=self.reference_frames[:6].tolist(),
                   practice_frames_by_case=self.reference_frames[6:].tolist(),
                   cells={key: {"attempts": n, "passes": k}
                          for key, (n, k) in sorted(self.counts.items())})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
        self.start = self.last
        self.counts.clear()
        self.updates = 0
        self.host_seconds = 0.
        self.reference_frames.fill(0)
        self.world_frames=0
        return row

    def rollback(self):
        self.flush(reason="before_rollback", force=True)
        self.epoch += 1
