"""Physical evidence contracts shared by training reports and frozen exams.

This module reports measured outcomes; it neither pays PPO rewards nor
chooses a curriculum or a tournament winner. All rates require an explicit
denominator. Missing completed trials are UNKNOWN, never zero success.
The independent examiner and on-policy training use separate observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def terrain_edge_trials(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Count outcomes from completed, frontier-edge training episodes only.

    Each edge event is published once at episode termination by the
    environment. A long-lived episode can span many 32-frame PPO rollouts;
    its intermediate frames must never count as unsuccessful attempts.
    The historical training metrics field names remain unchanged.
    """
    if "terrain_edge_mask" not in metrics:
        return {}

    def mean(name: str) -> float:
        arr=np.asarray(metrics[name],dtype=np.float64)
        return float(arr.mean())

    done=np.asarray(metrics["terrain_edge_done"],dtype=np.float64)
    if done.size == 0 or not np.isfinite(done).all():
        raise ValueError("nonfinite or empty terrain completion records")
    for key in ("terrain_edge_mask","terrain_edge_crossing",
                "terrain_edge_contact","terrain_edge_fell"):
        arr=np.asarray(metrics[key],dtype=np.float64)
        if arr.shape != done.shape or not np.isfinite(arr).all():
            raise ValueError(f"invalid terrain event shape/values: {key}")
    if np.any((done < 0.) | (done > 1.)):
        raise ValueError("completed episode flag outside [0,1]")
    ended=mean("terrain_edge_done")
    row={
        "terrain_edge_exposure":mean("terrain_edge_mask"),
        "terrain_edge_completed_fraction":ended,
        "terrain_edge_done_count":float(done.sum()),
        "terrain_edge_crossing":-1.,
        "terrain_edge_contact":-1.,
        "terrain_edge_fall_rate":-1.,
        "terrain_height_birth_m":mean("terrain_height_birth_m"),
    }
    if ended > 0.:
        row.update(
            terrain_edge_crossing=mean("terrain_edge_crossing")/ended,
            terrain_edge_contact=mean("terrain_edge_contact")/ended,
            terrain_edge_fall_rate=mean("terrain_edge_fell")/ended,
        )
    return row


@dataclass(frozen=True)
class ExactStepResult:
    """An independently measured fixed-height physical obstacle trial."""

    height_mm: int
    attempts: int
    contact_rate: float
    crossing_rate: float

    @classmethod
    def from_exam(cls, evidence: Mapping[str, Any], height_mm: int,
                  *, min_attempts: int = 8) -> "ExactStepResult":
        if evidence.get("physical_step_exact_height") is not True:
            raise ValueError("step exam lacks exact physical-height proof")
        key=f"course_{height_mm}mm_"
        values=(float(evidence[key+"attempt_count"]),
                float(evidence[key+"contact_rate"]),
                float(evidence[key+"crossing_rate"]))
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"{height_mm}mm obstacle evidence is nonfinite")
        attempts,contact,cross=values
        if attempts != int(attempts) or attempts < min_attempts:
            raise ValueError(f"insufficient exact {height_mm}mm obstacle attempts ({attempts:g})")
        if not (0. <= cross <= contact <= 1.):
            raise ValueError(f"invalid {height_mm}mm obstacle contact/crossing rates")
        return cls(height_mm,int(attempts),contact,cross)


def graded_step_score(evidence: Mapping[str, Any], heights: tuple[int, ...],
                      weights: tuple[float, ...],
                      *, min_attempts: int = 8) -> tuple[float, tuple[ExactStepResult, ...]]:
    """Grade exact-height success, not teaching rewards or a height ceiling.

    All heights use the same common frozen-policy exam. A policy cannot
    earn a grade from empty, mixed-height, or contradictory evidence.
    """
    if len(heights) != len(weights) or not heights or not all(
            math.isfinite(w) and w >= 0. for w in weights):
        raise ValueError("invalid graded step weights")
    if not math.isclose(sum(weights),1.,abs_tol=1e-6):
        raise ValueError("graded step weights must sum to one")
    trials=tuple(ExactStepResult.from_exam(evidence,mm,min_attempts=min_attempts)
                 for mm in heights)
    return sum(w*t.crossing_rate for w,t in zip(weights,trials)),trials
