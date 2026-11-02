"""Per-world MuJoCo/MJX physics-domain randomization.

This module owns manufacturing, friction, mass and servo uncertainty. It
changes batched model leaves, never the six-servo topology or actor inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np

# MJX model-level domain variations; physical topology remains fixed.


# Physical envelopes at radius=1.  Training should normally grow the three
# radii gradually rather than exposing the whole box from step zero.
MAX_SLOPE_DEG = 6.0
GROUND_FRICTION_FRAC = 0.35
FOOT_FRICTION_FRAC = 0.20
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
    """Narrow experiment surface for the v5014 domain envelope.

    ``robot_radius`` controls inertial/manufacturing-style variation,
    ``terrain_radius`` controls rotated-gravity/contact-friction variation and
    ``hardware_radius`` controls actuator/joint variation.  The concrete
    parameter ranges above are doctrine, not dozens of new experiment knobs.
    """

    enabled: bool = False
    robot_radius: float = 0.0
    terrain_radius: float = 0.0
    terrain_mode: str = 'gravity'  # gravity=legacy approximation; contact=real per-world planes
    hardware_radius: float = 0.0
    near_nominal_fraction: float = 0.35
    hard_tail_fraction: float = 0.08
    seed_offset: int = 5014

    def normalized(self) -> "DomainRandomizationConfig":
        if self.terrain_mode not in ('gravity','contact'):
            raise ValueError('terrain_mode must be gravity or contact')
        return DomainRandomizationConfig(
            enabled=bool(self.enabled),
            robot_radius=max(0.0, min(1.0, float(self.robot_radius))),
            terrain_radius=max(0.0, min(1.0, float(self.terrain_radius))),
            terrain_mode=self.terrain_mode,
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


def contact_plane_quaternion(angle_deg, uphill_xy=(1.0, 0.0)):
    """Quaternion of a real MuJoCo ground plane, gravity staying vertical.

    Plane normal = (-sin(angle)*ux, -sin(angle)*uy, cos(angle)).
    Thus walking along (ux,uy) climbs by +tan(angle) per horizontal metre.
    Changing a WORLD geom before mjx.put_model preserves an immutable
    MJX contact model; it does not rotate gravity or add task channels.
    """
    if not np.isfinite(angle_deg) or not 0.0 <= angle_deg <= 5.0:
        raise ValueError('contact slope must lie in [0,5] degrees')
    uphill=np.asarray(uphill_xy,dtype=float)
    if uphill.shape != (2,) or not np.isfinite(uphill).all() or np.linalg.norm(uphill)<1e-6:
        raise ValueError('contact uphill vector must be finite, nonzero xy')
    ux,uy=uphill/np.linalg.norm(uphill)
    a=math.radians(float(angle_deg))
    return np.asarray((math.cos(a/2),math.sin(a/2)*uy,
                       -math.sin(a/2)*ux,0.0),dtype=np.float64)


def _gravity_from_slope(jnp, roll, pitch, magnitude):
    """Gravity vector equivalent to a plane tilted under a robot-local frame.

    The collision plane remains geometrically flat; rotating gravity is exactly
    the same physics after a rigid coordinate rotation for an infinite plane,
    and avoids mutating geom transforms/AABBs inside a JIT.
    """
    sr, cr = jnp.sin(roll), jnp.cos(roll)
    sp, cp = jnp.sin(pitch), jnp.cos(pitch)
    return jnp.stack([
        -magnitude * sp * cr,
        +magnitude * sr,
        -magnitude * cp * cr,
    ])


def build_batched_model(model, *, ids: dict[str, Any], num_envs: int, seed: int,
                        config: DomainRandomizationConfig):
    """Return ``(model, in_axes, domain_info)`` for one vector environment.

    This follows MJX's intended domain-randomization pattern: only selected JAX
    leaves gain a leading environment dimension and ``in_axes`` marks those
    leaves as vmapped. Structural leaves remain shared.
    """
    cfg = config.normalized()
    if not cfg.enabled or max(cfg.robot_radius, cfg.terrain_radius, cfg.hardware_radius) <= 0.0:
        return model, None, {
            'enabled': False,
            'gravity_unit_world': None,
            'config': cfg.as_dict(),
        }

    import jax
    import jax.numpy as jnp

    rngs = jax.random.split(jax.random.PRNGKey(int(seed) + cfg.seed_offset), int(num_envs))
    floor_id = int(ids['floor_geom'])
    foot_geom_ids = jnp.asarray(ids['foot_geoms'], dtype=jnp.int32)
    act_dofs = jnp.asarray(ids['act_qvel'], dtype=jnp.int32)
    act_ids = jnp.asarray(ids['actuators'], dtype=jnp.int32)

    @jax.vmap
    def one(rng):
        keys = jax.random.split(rng, 18)
        r = _severity(jax, keys[0], cfg.robot_radius, cfg.near_nominal_fraction, cfg.hard_tail_fraction)
        t = _severity(jax, keys[1], cfg.terrain_radius, cfg.near_nominal_fraction, cfg.hard_tail_fraction)
        h = _severity(jax, keys[2], cfg.hardware_radius, cfg.near_nominal_fraction, cfg.hard_tail_fraction)

        # Terrain/contact. Randomize floor and each foot separately so the
        # controller cannot assume left/right traction symmetry.
        geom_friction = model.geom_friction
        floor_scale = 1.0 + t * jax.random.uniform(keys[3], (), minval=-GROUND_FRICTION_FRAC, maxval=GROUND_FRICTION_FRAC)
        foot_scale = 1.0 + t * jax.random.uniform(keys[4], (2,), minval=-FOOT_FRICTION_FRAC, maxval=FOOT_FRICTION_FRAC)
        geom_friction = geom_friction.at[floor_id].set(model.geom_friction[floor_id] * floor_scale)
        geom_friction = geom_friction.at[foot_geom_ids].set(model.geom_friction[foot_geom_ids] * foot_scale[:, None])

        gmag = 9.81 * (1.0 + r * jax.random.uniform(
            keys[7], (), minval=-GRAVITY_MAG_FRAC, maxval=GRAVITY_MAG_FRAC))
        if cfg.terrain_mode == 'contact':
            # One physically inclined plane per model/world. Sampling a
            # single polar angle enforces the actual 5-degree total cap
            # (independent roll+pitch maxima could exceed it diagonally).
            angle = math.radians(5.0) * t * jax.random.uniform(keys[5], ())
            azimuth = jax.random.uniform(keys[6], (), minval=-jnp.pi, maxval=jnp.pi)
            half = angle * .5
            floor_quat = jnp.stack((
                jnp.cos(half), jnp.sin(half)*jnp.sin(azimuth),
                -jnp.sin(half)*jnp.cos(azimuth), jnp.zeros_like(angle)))
            floor_normal = jnp.stack((
                -jnp.sin(angle)*jnp.cos(azimuth),
                -jnp.sin(angle)*jnp.sin(azimuth), jnp.cos(angle)))
            gravity = jnp.stack((jnp.zeros_like(gmag),
                                 jnp.zeros_like(gmag), -gmag))
            roll = angle*jnp.sin(azimuth)   # diagnostic projections only
            pitch = angle*jnp.cos(azimuth)
        else:
            # Preserve the original sampled tilted-gravity recipe exactly.
            slope = math.radians(MAX_SLOPE_DEG) * t
            roll = jax.random.uniform(keys[5], (), minval=-slope, maxval=slope)
            pitch = jax.random.uniform(keys[6], (), minval=-slope, maxval=slope)
            gravity = _gravity_from_slope(jnp, roll, pitch, gmag)
            floor_quat = model.geom_quat[floor_id]
            floor_normal = jnp.asarray((0.,0.,1.))

        # Robot inertial uncertainty.  Link mass variation is a standard MJX
        # sim2real randomizer; coherent *dimension* changes live in the separate
        # morphology-bank path because geometry has derived collision constants.
        global_mass = 1.0 + r * jax.random.uniform(keys[8], (), minval=-BODY_MASS_GLOBAL_FRAC, maxval=BODY_MASS_GLOBAL_FRAC)
        local_mass = 1.0 + r * jax.random.uniform(keys[9], model.body_mass.shape, minval=-BODY_MASS_LOCAL_FRAC, maxval=BODY_MASS_LOCAL_FRAC)
        body_mass = model.body_mass * global_mass * local_mass
        body_mass = body_mass.at[0].set(model.body_mass[0])  # world body stays exactly zero

        # Joint/servo spread. Only the six actuated DoFs are changed; passive
        # shin-flex physics remains a separate physical component.
        damping = model.dof_damping
        frictionloss = model.dof_frictionloss
        armature = model.dof_armature
        damping_scale = 1.0 + h * jax.random.uniform(keys[10], (act_dofs.shape[0],), minval=-JOINT_DAMPING_FRAC, maxval=JOINT_DAMPING_FRAC)
        friction_scale = 1.0 + h * jax.random.uniform(keys[11], (act_dofs.shape[0],), minval=-JOINT_FRICTIONLOSS_FRAC, maxval=JOINT_FRICTIONLOSS_FRAC)
        armature_scale = 1.0 + h * jax.random.uniform(keys[12], (act_dofs.shape[0],), minval=-JOINT_ARMATURE_FRAC, maxval=JOINT_ARMATURE_FRAC)
        damping = damping.at[act_dofs].set(model.dof_damping[act_dofs] * damping_scale)
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
        biasprm = biasprm.at[act_ids, 0].set(model.actuator_biasprm[act_ids, 0] + new_kp * zero)
        forcerange = forcerange.at[act_ids].set(model.actuator_forcerange[act_ids] * torque_scale[:, None])

        return (geom_friction, gravity, body_mass, damping, frictionloss, armature,
                gainprm, biasprm, forcerange, floor_quat, floor_normal,
                r, t, h, roll, pitch)

    (geom_friction, gravity, body_mass, damping, frictionloss, armature,
     gainprm, biasprm, forcerange, floor_quat, floor_normal,
     robot_severity, terrain_severity, hardware_severity, slope_roll,
     slope_pitch) = one(rngs)

    replacements = {
        'geom_friction': geom_friction,
        'opt.gravity': gravity,
        'body_mass': body_mass,
        'dof_damping': damping,
        'dof_frictionloss': frictionloss,
        'dof_armature': armature,
        'actuator_gainprm': gainprm,
        'actuator_biasprm': biasprm,
        'actuator_forcerange': forcerange,
    }
    if cfg.terrain_mode == 'contact':
        replacements['geom_quat'] = jax.vmap(
            lambda q: model.geom_quat.at[floor_id].set(q))(floor_quat)
    batched = model.tree_replace(replacements)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({name: 0 for name in replacements})
    gravity_unit = gravity / jnp.maximum(jnp.linalg.norm(gravity, axis=1, keepdims=True), 1e-6)

    return batched, in_axes, {
        'enabled': True,
        'config': cfg.as_dict(),
        'gravity_unit_world': gravity_unit,
        'contact_normal_world': floor_normal if cfg.terrain_mode == 'contact' else None,
        # Kept on device; callers may summarize with device_get outside JIT.
        'robot_severity': robot_severity,
        'terrain_severity': terrain_severity,
        'hardware_severity': hardware_severity,
        'slope_roll_rad': slope_roll,
        'slope_pitch_rad': slope_pitch,
    }


def doctrine_dict() -> dict[str, float]:
    """Physical ceilings recorded in docs/run provenance."""
    return {
        'MAX_SLOPE_DEG': MAX_SLOPE_DEG,
        'GROUND_FRICTION_FRAC': GROUND_FRICTION_FRAC,
        'FOOT_FRICTION_FRAC': FOOT_FRICTION_FRAC,
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

