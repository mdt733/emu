"""Single-policy specialist allocation and boundary probes for EMU.

Only teacher-side task selection. Focus is deliberately NOT a neural input:
all cohorts train the same actor, and every cohort still encounters the
other physical challenges at below-frontier severity.

All functions below are pure, apart from the caller's JAX random keys.
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple

FOCUS_NAMES = ("command", "resilience", "slope", "step", "transition", "control")
FOCUS_FRACTIONS = (.20, .20, .20, .20, .10, .10)
COMMAND, RESILIENCE, SLOPE, STEP, TRANSITION, CONTROL = range(6)


# The cold-start gait has first refusal through 20M world steps. Thereafter
# the only time-driven quantity is TEACHING SHARE; physical targets always
# advance/retreat on local completed-episode evidence, never on a clock.
# Every allocated cohort becomes actively focused after the 10M exposure
# blend. Its other skills remain in easier practice INSIDE the same world.
# Thus a 20% command allocation really means ~20% command-focused worlds.
SPECIALIST_ACTIVE_FRACTION = 1.0
SPECIALIST_BLEND_STEPS = 10_000_000
NONFOCUS_STRENGTH = 0.50
CONTROL_STRENGTH = 0.25


class TeachingProfile(NamedTuple):
    """Host-only lesson settings; never actor inputs or a second PPO."""
    name: str
    specialist_fraction: float
    adaptive_specialists: bool


def teaching_profile(*, step: int, foundation_steps: int,
                     steady_difficulty: float) -> TeachingProfile:
    """Two profiles in one training system: foundation then growth.

    Foundation is the ACTUAL main-style adaptive regime governor, including
    the initial 70-update steady-only allocation and later ordinary
    transition/resilience worlds. Replacing this with 90% command/10% grace
    would be a new experiment, NOT an exact reference to main.

    The growth profile gradually enables specialist teaching. The schedule
    controls teaching SHARE only, not target difficulty or physical mastery.
    """
    import math
    if not math.isfinite(float(steady_difficulty)):
        raise ValueError('steady difficulty must be finite')
    if step < 0 or foundation_steps < 0:
        raise ValueError('negative training step or foundation boundary')
    if step <= foundation_steps:
        return TeachingProfile('foundation', 0.0, False)
    fraction = SPECIALIST_ACTIVE_FRACTION * min(
        1.0, max(0.0, (step-foundation_steps)/SPECIALIST_BLEND_STEPS))
    return TeachingProfile('growth', fraction, True)


def growth_mix(*, step: int, foundation_steps: int,
               steady_difficulty: float) -> float:
    """Existing task-packet interface; share the same two-profile decision."""
    return teaching_profile(
        step=step, foundation_steps=foundation_steps,
        steady_difficulty=steady_difficulty).specialist_fraction


def focus_indices(num_envs: int):
    """Stable balanced specialist identity across 2,048 parallel worlds.

    Role is world-local *emphasis*, not exclusive content. Deterministic
    assignment avoids JAX recompilation on every PPO batch and lets cohorts
    maintain their own independent physical/evidence denominators.
    """
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    counts = np.floor(np.asarray(FOCUS_FRACTIONS) * num_envs).astype(int)
    for i in np.argsort(-(np.asarray(FOCUS_FRACTIONS) * num_envs-counts))[:num_envs-int(counts.sum())]:
        counts[i] += 1
    return np.repeat(np.arange(len(FOCUS_NAMES),dtype=np.int32),counts)


def diamond_commands(jax, jnp, keys, vx_hi, vx_lo, vy_max, yaw_max,
                     *, probe_scale=1.0):
    """One signed L1/octrahedral command sample per world (vx,vy,yaw).

    ~25% interior, ~55% edge, ~20% measured above edge (up to 12%).
    Vertices and faces are sampled deliberately, not rejected from a box;
    physical hardware safety remains an independent limit. A successful
    mixed-direction probe is evidence for that DIRECTION, not proof that
    each individual Cartesian maximum can now expand.
    """
    def one(key):
        k_type,k_axis,k_other,k_mass,k_sign,k_radius,k_mix=jax.random.split(key,7)
        kind=jax.random.uniform(k_type,())
        axis=jax.random.randint(k_axis,(),0,3)
        other=jax.random.randint(k_other,(),0,2)
        second=(axis+1+other)%3
        selector=jnp.arange(3)
        mask=jnp.where(kind<.27,selector==axis,
              jnp.where(kind<.81,(selector==axis)|(selector==second),
                        jnp.ones((3,),dtype=jnp.bool_)))
        # Positive random weights, normalized *after* selecting active axes.
        mass=jax.random.exponential(k_mass,(3,))*mask
        weights=mass/jnp.maximum(jnp.sum(mass),1e-6)
        sign=jnp.where(jax.random.uniform(k_sign,(3,))<.5,-1.0,1.0)
        r=jax.random.uniform(k_radius,())
        u=jax.random.uniform(k_mix,())
        radius=jnp.where(r<.25,.3+.55*u,
                jnp.where(r<.80,.85+.15*u,1.0+.12*u))
        # The mixed-axes frontier may grow independently. Pure velocity
        # axis limits still have their own real-world physical boundaries.
        radius=radius*jnp.where(kind<.27,1.0,probe_scale)
        unit=weights*sign*radius
        return jnp.asarray((
            jnp.where(unit[0]>=0,unit[0]*vx_hi,unit[0]*jnp.abs(vx_lo)),
            unit[1]*vy_max,unit[2]*yaw_max),dtype=jnp.float32)
    return jax.vmap(one)(keys)


def safe_command_scale(jnp, fraction, confidence):
    """Cold-start ramp before probing the full evidence-controlled frontier.

    This controls difficulty, *not* a permanent physical ceiling. Probes
    still extend ~12% beyond the currently reachable frontier.
    """
    return jnp.clip(.18+.82*jnp.maximum(fraction,confidence),.18,1.0)


def control_effort_weights(jnp, focus, *, control=.055, general=.006):
    """Small mechanical proxy lesson everywhere, stronger in easy controls.

    Real shaft work and braking are not electrical battery consumption.
    The lesson should *never* replace survival/velocity tracking reward.
    """
    return jnp.where(focus<0,0.0,jnp.where(focus==CONTROL,control,general)).astype(jnp.float32)
