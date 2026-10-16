"""CURRENT EXPERIMENT RECIPE — the human-readable v5015 training ladder.

Edit this *between* training campaigns to define the next hypothesis sweep.
Never alter a recipe while a checkpoint lineage is running: its source SHA,
training config and parent checkpoint belong together.

A recipe describes WHAT to compare. branch_lab.py controls HOW to resume,
score and promote safely. training.py remains the reusable PPO learning engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# All step targets are ABSOLUTE environment steps in a single policy lineage.
# Round1 adds 60M to the common 20M foundation: each fork reaches 80M total.
FOUNDATION = 20_000_000
ROUNDS = {
    'round1': 80_000_000,  # four comparable +60M continuations from 20M
    'round2': 100_000_000, # optional: +20M to two eligible finalists
    'round3': 120_000_000, # optional: +20M to best eligible finalist
}
# A succeeding round continues this many eligible policies from the last
# common frozen-policy exam. Baseline/forks/finallists share ONE recipe here.
PROMOTION_SLOTS = {'round2': 2, 'round3': 1}
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


def plan_dict(seed=733, envs=2048):
    """Human-readable hypothesis/budget plan, derived from this recipe only.

    The executor imports this function but does not define study budgets.
    A complete round costs (target - previous target) per continuing arm.
    """
    round_names = list(ROUNDS)
    prior_target = FOUNDATION
    nominal_steps = FOUNDATION
    for stage in round_names:
        contenders = len(ARMS) if stage == 'round1' else PROMOTION_SLOTS[stage]
        nominal_steps += contenders * (ROUNDS[stage] - prior_target)
        prior_target = ROUNDS[stage]
    first_increment = len(ARMS) * (ROUNDS['round1'] - FOUNDATION)
    return {'version': 'v5015-lab1', 'seed': seed, 'envs': envs,
            'foundation': FOUNDATION, 'rounds': ROUNDS,
            'arms': [asdict(a) for a in ARMS],
            'nominal_steps': nominal_steps,
            'review_points': ['after common 20M foundation', 'after four 80M branches (+60M each)',
                              'optional: two 100M finalists', 'optional: one 120M finalist'],
            'recommended': 'emu ladder --stage foundation, inspect, then emu ladder --stage auto --stop-after round1',
            'round1_incremental_steps': first_increment,
            'through_round1_total_steps': FOUNDATION + first_increment,
            'automatic_promotion': 'eligible-only, score-ranked in --stage auto; manual by default',
            'auto_stop_if_ineligible': True}
