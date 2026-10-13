"""CURRENT EXPERIMENT RECIPE — the human-readable v5015 training ladder.

Edit this *between* training campaigns to define the next hypothesis sweep.
Never alter a recipe while a checkpoint lineage is running: its source SHA,
training config and parent checkpoint belong together.

A recipe describes WHAT to compare. branch_lab.py controls HOW to resume,
score and promote safely. training.py remains the reusable PPO learning engine.
"""
from __future__ import annotations

from dataclasses import dataclass

# All step targets are ABSOLUTE environment steps in a single policy lineage.
# E.g. round1 adds only 6M to the 20M foundation; it does not train 26M anew.
FOUNDATION = 20_000_000
ROUNDS = {
    'round1': 26_000_000,  # try all four directions (+6M each)
    'round2': 34_000_000,  # develop the two strongest eligible (+8M each)
    'round3': 46_000_000,  # continue the one strongest eligible (+12M)
}
SCREEN_PITCH_DEG = 15.0  # common evaluation task, not a training command


@dataclass(frozen=True)
class Arm:
    name: str
    hypothesis: str
    teacher_mode: str = 'graph'
    dose: float = 0.10
    terrain_radius: float = 0.0
    pitch_deg: float = 8.0


# A hypothesis should differ from baseline in ONE scientifically interpretable way.
ARMS = (
    Arm('baseline', '10% graph teaching, level-floor control'),
    Arm('shadow', 'No capability-field teaching', teacher_mode='shadow', dose=0.0),
    Arm('slope', 'Gentle gravity-equivalent slope practice', terrain_radius=0.12),
    Arm('pitch15', '15 degree chassis command ceiling', pitch_deg=15.0),
)
BY_NAME = {arm.name: arm for arm in ARMS}
