"""Training-health state, separate from curriculum.

Curriculum answers "what tasks should we train next?".  This module answers two
orthogonal questions: whether PPO stress should temporarily veto curriculum
expansion, and whether a sustained global collapse warrants a policy rollback.
It deliberately owns no task-distribution parameters and cannot mutate the
curriculum.
"""
from __future__ import annotations

from typing import Any
import math


class TrainingHealth:
    KL_STRESS_EXPAND_FLOOR = 0.10
    KL_STRESS_CHRONIC_WINDOW = 50
    KL_STRESS_CHRONIC_FRACTION = 0.90

    COLLAPSE_FALL_THRESHOLD = 0.65
    COLLAPSE_STREAK_LIMIT = 100

    RECOVERY_GOOD_FLOOR = 0.60
    RECOVERY_EFFORT_FLOOR = 0.20
    RECOVERY_SEEN_FLOOR = 0.10
    RECOVERY_WINDOW = 8

    def __init__(self) -> None:
        self.kl_stress_history: list[float] = []
        self.kl_stress_chronic = False
        self.stepping_collapse_streak = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            'kl_stress_history': list(self.kl_stress_history),
            'kl_stress_chronic': bool(self.kl_stress_chronic),
            'stepping_collapse_streak': int(self.stepping_collapse_streak),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {'kl_stress_history', 'kl_stress_chronic', 'stepping_collapse_streak'}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError(f"training-health state keys mismatch: expected={sorted(expected)} got={sorted(state) if isinstance(state, dict) else type(state).__name__}")
        hist = state['kl_stress_history']
        if not isinstance(hist, list):
            raise ValueError('training-health kl_stress_history must be a list')
        self.kl_stress_history = [float(v) for v in hist][-self.KL_STRESS_CHRONIC_WINDOW:]
        self.kl_stress_chronic = bool(state['kl_stress_chronic'])
        self.stepping_collapse_streak = int(state['stepping_collapse_streak'])

    def observe_ppo(self, row: dict[str, float]) -> dict[str, float | bool]:
        """Update the KL-stress gate for this rollout before curriculum moves."""
        kl_stopped_frac = float(row.get('ppo_kl_stopped_frac', 0.0))
        was_chronic = self.kl_stress_chronic
        self.kl_stress_history.append(kl_stopped_frac)
        self.kl_stress_history = self.kl_stress_history[-self.KL_STRESS_CHRONIC_WINDOW:]
        engaged = sum(v > self.KL_STRESS_EXPAND_FLOOR for v in self.kl_stress_history)
        self.kl_stress_chronic = (
            len(self.kl_stress_history) >= self.KL_STRESS_CHRONIC_WINDOW
            and engaged >= self.KL_STRESS_CHRONIC_FRACTION * self.KL_STRESS_CHRONIC_WINDOW
        )
        stress_now = kl_stopped_frac > self.KL_STRESS_EXPAND_FLOOR
        return {
            'kl_stopped_frac_seen': kl_stopped_frac,
            'kl_stress_chronic': float(self.kl_stress_chronic),
            'kl_ok_to_expand': (not stress_now) or self.kl_stress_chronic,
            'kl_became_chronic': self.kl_stress_chronic and not was_chronic,
        }

    def observe_collapse(self, smoothed_global_fall_rate: float) -> dict[str, float | bool]:
        """Update global collapse evidence after the curriculum's pooled fall EMA."""
        collapsed = float(smoothed_global_fall_rate) > self.COLLAPSE_FALL_THRESHOLD
        self.stepping_collapse_streak = self.stepping_collapse_streak + 1 if collapsed else 0
        return {
            'stepping_collapse_streak': float(self.stepping_collapse_streak),
            'needs_recovery': self.stepping_collapse_streak >= self.COLLAPSE_STREAK_LIMIT,
        }


    @classmethod
    def recovery_candidate(cls, competence: float, core_competence: float) -> tuple[tuple[int, float], str]:
        """Rank one sustained-competence window for the canonical rollback snapshot."""
        if competence >= cls.RECOVERY_GOOD_FLOOR and core_competence >= cls.RECOVERY_GOOD_FLOOR:
            return (2, competence), 'confirmed-good'
        if competence >= cls.RECOVERY_EFFORT_FLOOR and core_competence >= cls.RECOVERY_EFFORT_FLOOR:
            return (1, competence), 'best-effort'
        if competence >= cls.RECOVERY_SEEN_FLOOR:
            return (0, competence), 'high-water-mark'
        return (-1, -1.0), 'none'

    @staticmethod
    def adjust_learning_rate(
        scale: float, *, approx_kl: float, target_kl: float,
        factor: float, min_scale: float, max_scale: float,
    ) -> float:
        """Soft PPO KL controller, distinct from the hard minibatch KL stop.

        Use only the KL of the collected TRAINING rollout, never held-out
        exam scores. Non-finite KL preserves the prior scale; an in-band
        update also leaves it alone. The clamp applies only to finite KL,
        matching the established controller's timing and arithmetic.
        """
        if math.isfinite(approx_kl):
            if approx_kl > 2.0 * target_kl:
                scale /= factor
            elif approx_kl < 0.5 * target_kl:
                scale *= factor
            scale = float(min(max_scale, max(min_scale, scale)))
        return scale

    def note_recovery_performed(self) -> None:
        self.stepping_collapse_streak = 0
