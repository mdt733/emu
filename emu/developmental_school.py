"""Opt-in cold-school research: mild stairs as shared specialist context.

The 13-arm late-school screen found apparent stair gains paired with
regression of clean recovery/slope missions. It did NOT establish that
mixed training cures forgetting or that the 32mm uphill barrier is
physically feasible. Keep original foundation, one actor, PPO, servo,
observation and independent up/down difficulty teacher.
"""
from __future__ import annotations

# Exactly capability_focus.FOCUS_NAMES index order. These are ADDITIONAL
# chances within each role's ordinary birth cohort, NOT whole-batch shares.
# STEP retains its separate Academy.
CROSS_STAIR_ROLE_PROBABILITIES=(.08,.12,.15,0.,.10,.06)
# Shared lips are capped at these metres per role AND by already-earned
# direction-specific frontiers; downhill 55mm cannot invade recovery.
CROSS_STAIR_ROLE_CAP_M=(.012,.016,.018,0.,.014,.008)


def cross_stair_exposure_blend(*,step:int,enabled:bool,
                               start_steps:int=90_000_000,
                               ramp_steps:int=20_000_000)->float:
    """Age ramps exposure 0→1, never stair difficulty or reward value."""
    if (type(step) is not int or type(start_steps) is not int
            or type(ramp_steps) is not int or step<0 or
            start_steps<0 or ramp_steps<=0 or type(enabled) is not bool):
        raise ValueError("Invalid developmental school exposure schedule")
    if not enabled:
        return 0.
    return min(1.,max(0.,(step-start_steps)/ramp_steps))


def cross_stair_plan(blend:float)->dict[str,float]:
    """Maximum extra birth chance in each EXISTING specialist cohort."""
    from .capability_focus import FOCUS_NAMES
    if not 0.<=blend<=1.:
        raise ValueError("Invalid cross-specialty stair blend")
    return {name:blend*prob for name,prob in zip(
        FOCUS_NAMES,CROSS_STAIR_ROLE_PROBABILITIES)}
