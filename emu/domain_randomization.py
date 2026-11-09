"""Per-world MuJoCo/MJX physics-domain randomization.

This module owns manufacturing, friction, mass and servo uncertainty. It
changes batched model leaves, never the six-servo topology or actor inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

from .physical_variants import SERVO_RATED_CONTINUOUS_TORQUE_NM

# MJX model-level domain variations; physical topology remains fixed.


# Physical manufacturing envelopes at radius=1. The real prism carpet
# owns gradients and discontinuities independently of manufacturing spread.
BODY_MASS_GLOBAL_FRAC = 0.08
BODY_MASS_LOCAL_FRAC = 0.04
JOINT_DAMPING_FRAC = 0.18
JOINT_FRICTIONLOSS_FRAC = 0.25
JOINT_ARMATURE_FRAC = 0.10
ACTUATOR_KP_FRAC = 0.12
ACTUATOR_TORQUE_FRAC = 0.15
SERVO_ZERO_MAX_DEG = 1.5
GRAVITY_MAG_FRAC = 0.015



@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Optional per-world robot inertia and servo/joint uncertainty.

    No synthetic terrain incline, rotated plane, or tilted gravity.
    """
    enabled: bool = False
    robot_radius: float = 0.0
    hardware_radius: float = 0.0
    near_nominal_fraction: float = 0.35
    hard_tail_fraction: float = 0.08
    seed_offset: int = 5014

    def normalized(self) -> "DomainRandomizationConfig":
        return DomainRandomizationConfig(
            enabled=bool(self.enabled),
            robot_radius=max(0.0, min(1.0, float(self.robot_radius))),
            hardware_radius=max(0.0, min(1.0, float(self.hardware_radius))),
            near_nominal_fraction=max(0.0, min(0.90, float(self.near_nominal_fraction))),
            hard_tail_fraction=max(0.0, min(0.40, float(self.hard_tail_fraction))),
            seed_offset=int(self.seed_offset),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


def _severity(jax, key, radius: float, near_nominal_fraction: float, hard_tail_fraction: float):
    """Sample one 0..radius severity with a useful training mixture.

    A healthy fraction remains close to nominal so robustification does not
    erase the clean gait. A small hard tail prevents a curriculum frontier from
    becoming a cliff edge.
    """
    k_bucket, k_value = jax.random.split(key)
    bucket = jax.random.uniform(k_bucket, ())
    u = jax.random.uniform(k_value, ())
    near = 0.25 * u
    normal = u
    hard = 0.75 + 0.25 * u
    value = jax.numpy.where(
        bucket < near_nominal_fraction,
        near,
        jax.numpy.where(bucket > 1.0 - hard_tail_fraction, hard, normal),
    )
    return value * radius


def build_batched_model(model, *, ids: dict[str, Any], num_envs: int, seed: int,
                        config: DomainRandomizationConfig):
    """Optional robot/servo variation. Keep exact pre-cleanup RNG key indexes.

    Only selected MJX model leaves are batched across the physical worlds.
    Terrain geometry stays the same real-prism carpet in every environment.
    """
    cfg = config.normalized()
    if not cfg.enabled or max(cfg.robot_radius, cfg.hardware_radius) <= 0.0:
        return model, None, {
            'enabled': False,
            'gravity_unit_world': None,
            'config': cfg.as_dict(),
        }

    import jax
    import jax.numpy as jnp

    rngs = jax.random.split(jax.random.PRNGKey(int(seed) + cfg.seed_offset), int(num_envs))
    act_dofs = jnp.asarray(ids['act_qvel'], dtype=jnp.int32)
    act_ids = jnp.asarray(ids['actuators'], dtype=jnp.int32)

    @jax.vmap
    def one(rng):
        keys = jax.random.split(rng, 18)
        r = _severity(jax, keys[0], cfg.robot_radius,
                      cfg.near_nominal_fraction, cfg.hard_tail_fraction)
        h = _severity(jax, keys[2], cfg.hardware_radius,
                      cfg.near_nominal_fraction, cfg.hard_tail_fraction)
        gmag = 9.81 * (1.0 + r * jax.random.uniform(
            keys[7], (), minval=-GRAVITY_MAG_FRAC, maxval=GRAVITY_MAG_FRAC))
        gravity = jnp.stack((jnp.zeros_like(gmag),
                             jnp.zeros_like(gmag), -gmag))

        # Robot inertial uncertainty.  Link mass variation is a standard MJX
        # sim2real randomizer; coherent *dimension* changes live in the separate
        # morphology-bank path because geometry has derived collision constants.
        global_mass = 1.0 + r * jax.random.uniform(keys[8], (), minval=-BODY_MASS_GLOBAL_FRAC, maxval=BODY_MASS_GLOBAL_FRAC)
        local_mass = 1.0 + r * jax.random.uniform(keys[9], model.body_mass.shape, minval=-BODY_MASS_LOCAL_FRAC, maxval=BODY_MASS_LOCAL_FRAC)
        body_mass = model.body_mass * global_mass * local_mass
        body_mass = body_mass.at[0].set(model.body_mass[0])  # world body stays exactly zero

        # The driven hinges have NO velocity-dependent passive damping.
        # The same 18% hardware spread now applies to the actively capped
        # servo kv term (actuator_biasprm[2]), never unbounded dof damping.
        # Passive shin-flex damping is independent of this sampling.
        damping = model.dof_damping
        frictionloss = model.dof_frictionloss
        armature = model.dof_armature
        damping_scale = 1.0 + h * jax.random.uniform(keys[10], (act_dofs.shape[0],), minval=-JOINT_DAMPING_FRAC, maxval=JOINT_DAMPING_FRAC)
        friction_scale = 1.0 + h * jax.random.uniform(keys[11], (act_dofs.shape[0],), minval=-JOINT_FRICTIONLOSS_FRAC, maxval=JOINT_FRICTIONLOSS_FRAC)
        armature_scale = 1.0 + h * jax.random.uniform(keys[12], (act_dofs.shape[0],), minval=-JOINT_ARMATURE_FRAC, maxval=JOINT_ARMATURE_FRAC)
        damping = damping.at[act_dofs].set(model.dof_damping[act_dofs])
        frictionloss = frictionloss.at[act_dofs].set(model.dof_frictionloss[act_dofs] * friction_scale)
        armature = armature.at[act_dofs].set(model.dof_armature[act_dofs] * armature_scale)

        gainprm = model.actuator_gainprm
        biasprm = model.actuator_biasprm
        forcerange = model.actuator_forcerange
        kp_scale = 1.0 + h * jax.random.uniform(keys[13], (act_ids.shape[0],), minval=-ACTUATOR_KP_FRAC, maxval=ACTUATOR_KP_FRAC)
        torque_scale = 1.0 + h * jax.random.uniform(keys[14], (act_ids.shape[0],), minval=-ACTUATOR_TORQUE_FRAC, maxval=ACTUATOR_TORQUE_FRAC)
        zero = math.radians(SERVO_ZERO_MAX_DEG) * h * jax.random.uniform(keys[15], (act_ids.shape[0],), minval=-1.0, maxval=1.0)
        base_kp = model.actuator_gainprm[act_ids, 0]
        new_kp = base_kp * kp_scale
        gainprm = gainprm.at[act_ids, 0].set(new_kp)
        # Position actuator: force = kp*ctrl + bias0 - kp*q + ...
        biasprm = biasprm.at[act_ids, 1].set(model.actuator_biasprm[act_ids, 1] * kp_scale)
        biasprm = biasprm.at[act_ids, 2].set(
            model.actuator_biasprm[act_ids, 2] * damping_scale)
        biasprm = biasprm.at[act_ids, 0].set(model.actuator_biasprm[act_ids, 0] + new_kp * zero)
        # Manufacturing variation may WEAKEN a servo. It must NEVER
        # strengthen any of the six actuators beyond the rated 12V ceiling,
        # even if sampled multiplier >1. MuJoCo/MJX enforces this force cap.
        varied_limits = model.actuator_forcerange[act_ids] * torque_scale[:, None]
        varied_limits = jnp.clip(
            varied_limits,
            -SERVO_RATED_CONTINUOUS_TORQUE_NM,
            SERVO_RATED_CONTINUOUS_TORQUE_NM)
        forcerange = forcerange.at[act_ids].set(varied_limits)

        return (gravity, body_mass, damping, frictionloss, armature,
                gainprm, biasprm, forcerange, r, h)

    (gravity, body_mass, damping, frictionloss, armature,
     gainprm, biasprm, forcerange, robot_severity, hardware_severity) = one(rngs)
    replacements = {
        'opt.gravity': gravity,
        'body_mass': body_mass,
        'dof_damping': damping,
        'dof_frictionloss': frictionloss,
        'dof_armature': armature,
        'actuator_gainprm': gainprm,
        'actuator_biasprm': biasprm,
        'actuator_forcerange': forcerange,
    }
    batched = model.tree_replace(replacements)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({name: 0 for name in replacements})
    gravity_unit = gravity / jnp.maximum(jnp.linalg.norm(gravity, axis=1, keepdims=True), 1e-6)
    return batched, in_axes, {
        'enabled': True,
        'config': cfg.as_dict(),
        'gravity_unit_world': gravity_unit,
        'robot_severity': robot_severity,
        'hardware_severity': hardware_severity,
    }


def doctrine_dict() -> dict[str, float]:
    """Physical ceilings recorded in docs/run provenance."""
    return {
        'BODY_MASS_GLOBAL_FRAC': BODY_MASS_GLOBAL_FRAC,
        'BODY_MASS_LOCAL_FRAC': BODY_MASS_LOCAL_FRAC,
        'JOINT_DAMPING_FRAC': JOINT_DAMPING_FRAC,
        'JOINT_FRICTIONLOSS_FRAC': JOINT_FRICTIONLOSS_FRAC,
        'JOINT_ARMATURE_FRAC': JOINT_ARMATURE_FRAC,
        'ACTUATOR_KP_FRAC': ACTUATOR_KP_FRAC,
        'ACTUATOR_TORQUE_FRAC': ACTUATOR_TORQUE_FRAC,
        'SERVO_ZERO_MAX_DEG': SERVO_ZERO_MAX_DEG,
        'GRAVITY_MAG_FRAC': GRAVITY_MAG_FRAC,
    }

