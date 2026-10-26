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

# These probabilities choose a new episode's teacher; local difficulty
# targets remain driven by physical evidence, never schedule age.
TEACHING_SCHEDULES = ('balanced', 'staged', 'terrain-early', 'steering')
STAGED_START = (.40,.25,.05,.00,.10,.20)
STAGED_MIDDLE = (.25,.20,.25,.05,.10,.15)
STAGED_LATE = (.20,.15,.20,.20,.10,.15)
# A genuine geometric-teaching intervention (slope+step = 65% vs 40%).
# Distinct from balanced throughout the growth ramp, not merely +2% slope.
TERRAIN_EARLY = (.16,.09,.35,.30,.05,.05)
# A targeted directional-control arm: preserve substantial terrain practice,
# while funding more commands without starving resilience and transitions.
STEERING = (.35,.15,.20,.15,.10,.05)


def attention_weights(*, schedule: str, step: int,
                      foundation_steps: int) -> tuple[float,...]:
    """Smooth six-way birth distribution, not a proficiency gate."""
    if schedule not in TEACHING_SCHEDULES:
        raise ValueError('unknown teaching allocation: '+str(schedule))
    if step < 0 or foundation_steps < 0:
        raise ValueError('negative step or foundation boundary')
    if schedule == 'balanced' or step <= foundation_steps:
        return tuple(FOCUS_FRACTIONS)
    if schedule == 'terrain-early':
        return tuple(TERRAIN_EARLY)
    if schedule == 'steering':
        return tuple(STEERING)
    elapsed=step-foundation_steps
    before,after=(STAGED_START,STAGED_MIDDLE) if elapsed<=20_000_000 else (STAGED_MIDDLE,STAGED_LATE)
    t=min(1.,max(0.,(elapsed if elapsed<=20_000_000 else elapsed-20_000_000)/20_000_000))
    return tuple((1.-t)*x+t*y for x,y in zip(before,after))
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


# Fractions among ordinary (non-grace) worlds. Each recipe still shares
# the exact actor, PPO, baseline rewards and existing three-regime sampler.
# Grace uses some of the preallocated CONTROL worlds, not another network.
FOUNDATION_RECIPES = {
    'main': (None, 0.0),
    'command-heavy': ((.90,.05,.05), 0.0),
    'curious': ((.84,.08,.08), .50),  # ~5% of all worlds control/grace
}


class TeachingProfile(NamedTuple):
    """Host-only lesson settings; never actor inputs or a second PPO."""
    name: str
    specialist_fraction: float
    adaptive_specialists: bool
    regime_mix: tuple[float,float,float] | None = None
    foundation_control_mix: float = 0.0


def teaching_profile(*, step: int, foundation_steps: int,
                     steady_difficulty: float,
                     foundation_mix: str = 'main') -> TeachingProfile:
    """Two profiles in one training system: foundation then growth.

    Foundation may use the original adaptive regime allocation or a chosen
    command-heavy/curious allocation. Its original eight locomotion payments
    and PPO learner are unchanged. The curious recipe includes a small
    focused grace cohort, still required to actually walk.
    Growth starts the same adaptive six-specialty controller in all cases.
    """
    import math
    if not math.isfinite(float(steady_difficulty)):
        raise ValueError('steady difficulty must be finite')
    if step < 0 or foundation_steps < 0:
        raise ValueError('negative training step or foundation boundary')
    if foundation_mix not in FOUNDATION_RECIPES:
        raise ValueError('unknown foundation mix: '+str(foundation_mix))
    if step <= foundation_steps:
        regime,control=FOUNDATION_RECIPES[foundation_mix]
        return TeachingProfile('foundation', 0.0, False, regime, control)
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
                     *, probe_scale=1.0, steering_attention=0.0):
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
        # Opt-in steering curricula sample more pure vy/yaw commands,
        # while legacy balanced/staged/terrain runs stay bitwise intact.
        weights=jnp.asarray([1.,1.+2.*steering_attention,
                              1.+2.*steering_attention],dtype=jnp.float32)
        biased_axis=jax.random.categorical(k_axis,jnp.log(weights))
        uniform_axis=jax.random.randint(k_axis,(),0,3)
        axis=jnp.where(steering_attention>0.,biased_axis,uniform_axis)
        other=jax.random.randint(k_other,(),0,2)
        second=(axis+1+other)%3
        selector=jnp.arange(3)
        vertex_fraction=.27+.18*steering_attention
        mask=jnp.where(kind<vertex_fraction,selector==axis,
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
