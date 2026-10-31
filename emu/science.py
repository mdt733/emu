"""One pre-registered v5033 experiment catalogue: ONE intervention per arm.

This module does not start runs, edit checkpoints, or change actor inputs.
Research arm -1 means old/historical source behaviour. Arm 00 is the
explicit unchanged 260M continuation control. All 24 identities are saved
in TrainingConfig and must be rechecked against the resolved CLI plan.
"""
from __future__ import annotations

from dataclasses import replace

ARM_NAMES = (
    "control",
    "step_promotion_1mm",
    "step_retreat_1mm",
    "step_twelve_evidence",
    "ascent_contact_1p5",
    "ascent_contact_3",
    "ascent_contact_4",
    "ascent_contact_3_persistent",
    "downhill_cautious_0p10",
    "downhill_births_double",
    "downhill_signed_yaw",
    "downhill_overspeed",
    "steering_focus",
    "fast_rehearsal_40pct",
    "fast_rehearsal_10pct",
    "velocity_integral_1p5",
    "temporal_speed_0p25",
    "temporal_speed_0p80",
    "speed_envelope_0p8",
    "transition_focus_20pct",
    "aggressive_pushes",
    "resilience_focus_30pct",
    "motor_cost_0p03",
    "step_up_crouch_allowance",
)
ARM_COUNT = len(ARM_NAMES)

# Tunable existing knobs. New teaching/reward interventions are selected
# solely by science_arm inside their physical owners; no duplicated toggles.
_CONFIG_OVERRIDES = {
    4: dict(verified_ascent_reallocation=1.5),
    5: dict(verified_ascent_reallocation=3.0),
    6: dict(verified_ascent_reallocation=4.0),
    7: dict(verified_ascent_reallocation=3.0),
    12: dict(teaching_schedule="steering"),
    13: dict(athletic_rehearsal_frac=.40),
    14: dict(athletic_rehearsal_frac=.10),
    15: dict(velocity_i_multiplier=1.5),
    16: dict(temporal_vx_reward_share=.25),
    17: dict(temporal_vx_reward_share=.80),
    18: dict(vx_hi_max=.80),
    20: dict(frontier_push_safety_n=4.0, force_program="aggressive"),
    22: dict(motor_guidance_weight=.03),
}
# Arm 07 is a 3-frame persistence modification of arm 05. Others
# compare directly against the unchanged control in the same age window.
PARENT_COMPARATOR = {i: (5 if i == 7 else 0) for i in range(ARM_COUNT)}


def arm_name(arm: int) -> str:
    if not isinstance(arm, int) or not 0 <= arm < ARM_COUNT:
        raise ValueError("v5033 science_arm must be one of 00..23")
    return f"{arm:02d}_{ARM_NAMES[arm]}"


def configure_arm(config, arm: int):
    """Return the *actual saved* learner configuration for a science arm.

    Arguments from the common base recipe are first resolved normally by
    emu.run, then exactly one registered intervention modifies those values.
    Native reward/controller knobs are plain TrainingConfig fields; teacher
    and curriculum variants are controlled by the saved science_arm field.
    """
    arm_name(arm)
    if not (config.unified_capability_focus
            and config.observation_schema == "terrain_360"
            and config.terrain_lesson == "coherent-careful"):
        raise ValueError("v5033 requires unified terrain_360 coherent-careful actor")
    if config.verified_ascent_reallocation != 0.0:
        raise ValueError("v5033 base must use ASCENT 0.0; arm chooses the treatment")
    if config.temporal_vx_reward_share != .5:
        raise ValueError("v5033 base must use temporal reward share 0.5")
    if config.athletic_rehearsal_frac != .25:
        raise ValueError("v5033 base must use 25% athletic rehearsal")
    if config.velocity_i_multiplier != 1.0 or config.motor_guidance_weight != 0.0:
        raise ValueError("v5033 base must disable velocity-I and motor-cost")
    if config.teaching_schedule != "balanced" or config.force_program != "gentle":
        raise ValueError("v5033 base must be balanced/gentle")
    if abs(config.vx_hi_max - .6) > 1e-6:
        raise ValueError("v5033 base must have vx_hi_max 0.6")
    return replace(config, science_arm=arm, **_CONFIG_OVERRIDES.get(arm, {}))


def expected_fields(arm: int) -> dict:
    arm_name(arm)
    return {"science_arm": arm, **_CONFIG_OVERRIDES.get(arm, {})}
