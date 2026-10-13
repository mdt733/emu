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
    """One hypothesis: changes to named, validated TrainingConfig knobs only.

    Each (field, value) pair becomes an explicit field override to the common
    mature baseline. No PPO-engine edits or new CLI switches are needed.
    """
    name: str
    hypothesis: str
    knobs: tuple[tuple[str, bool | int | float | str | None], ...] = ()


# Each challenger differs from the baseline in one understandable dimension.
# All knobs are existing fields of emu/training_config.py::TrainingConfig.
ARMS = (
    Arm('baseline', '10% graph teaching, level-floor control'),
    Arm('shadow', 'No capability-field teaching', (
        ('capability_field_mode', 'shadow'),
        ('field_max_influence', 0.0),
    )),
    Arm('slope', 'Gentle gravity-equivalent slope practice', (
        ('domain_randomization', True),
        ('domain_terrain_radius', 0.12),
    )),
    Arm('pitch15', '15 degree chassis command ceiling', (
        ('pitch_command_max_deg', 15.0),
    )),
)
BY_NAME = {arm.name: arm for arm in ARMS}
