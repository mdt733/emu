"""The simulated universe: robot physics, task packets and sensors.

`EmuMJXEnv` advances many simultaneous MuJoCo worlds and supplies physical
observations to PPO. The robot's joints, springs and motor limits are defined
in robot.xml; the DomainRandomizationConfig and build_batched_model code below supply per-world variations.

The TaskSampler below generates lessons for this environment
without duplicating JAX, configuration or RNG state. This file owns the
physical world: reset, motor delivery, disturbances, stepping, sensor packets
and measured events. rewards.py owns the scalar teaching payment.

Teacher-only task fields never become privileged actor sensor channels.
See README.md for the teaching-to-physics overview.
"""
from __future__ import annotations

from typing import Any, NamedTuple
import xml.etree.ElementTree as ET
import numpy as np

from .rewards import training_reward, apply_early_fall_penalty, REWARD_ONLY_METRICS
from .capability_focus import (focus_indices, diamond_commands,
                               control_effort_weights, COMMAND, RESILIENCE,
                               SLOPE, STEP, TRANSITION, CONTROL)
from .directional import (simulate_tripod_ranges, fit_tripod_ground,
                          fit_forward_array_ground,
                          simulate_downward_optical_flow, estimate_flow_velocity,
)
from .contracts import (SIM_DT, CONTROL_SUBSTEPS, CONTROL_DT, ACTION_DIM, ACTUATED_JOINTS, BASE_OBS_DIM, BASE_OBS_FIELDS, HISTORY_LEN, OBS_DIM, OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN, TERRAIN_TOF_PITCH_DEG, TERRAIN_TOF_YAW_DEG, TERRAIN_TOF_MIN_M, TERRAIN_TOF_MAX_M, actor_observation_fields, sensor_observation_contract, COMBO_P, PHASE_RATE_MAX_HZ, PITCH_CMD_MAX_RAD, FIELD_EDGE_SCALES, SEED_VX_LO, SEED_VX_HI, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS,
                       FIELD_NODE_STRIDE, FIELD_EDGE_STRIDE, FIELD_NODE_START, FIELD_EDGE_START, TASK_PARAM_DIM,
                       TERRAIN_TASK_FIELDS, TERRAIN_TASK_PARAM_DIM, UNIFIED_TERRAIN_FIELDS, UNIFIED_TASK_PARAM_DIM, TASK_PARAM_FIELDS, P)

# Deployable actor observation excludes training-only
# teacher signals (regime id, exact contact/fall classifiers, phase grading and
# step-evidence composites) from the policy input. Every remaining field has a
# plausible real-robot source: IMU, servo telemetry, forward kinematics, shin
# flex Hall sensors, optical-flow/ToF velocity estimation, command/controller
# state. Reward/evaluation may still use perfect MuJoCo truth.
#
# The exact 55-channel order is the shared BASE_OBS_FIELDS contract.
# Fixed-size history used to model target-delivery latency. The buffer shape is
# independent of the configured latency so JIT shapes stay stable.
FOOT_BODIES = ['r_foot', 'l_foot']
BAD_BODIES = ['torso', 'r_thigh', 'l_thigh', 'r_shin', 'l_shin']
TARGET_SPAN = np.array([0.38, 0.48, 0.58, 0.38, 0.48, 0.58], dtype=np.float32)
RESET_JOINT_NOISE = 0.004
RESET_QVEL_NOISE = 0.003
RESET_FORWARD_VELOCITY_BLEND = 1.0
NEUTRAL_ROOT_Z = 0.2584
# Falling is diagnosed from body geometry/height rather than treating every
# non-foot contact as an immediate terminal. EMU is expected to recover through
# dynamic stepping, so termination should represent an unrecoverable fall rather
# than merely imperfect contact.
SOFT_BODY_LOW_Z = 0.035
BODY_LOW_REWARD_WEIGHT = 0.25
# Seed gait used as the project's basic capability reference. It is a slow
# moving gait, not a near-zero/static-standing task.
# RESILIENCE push parameters. External force is per-environment data, so it can
# vary cheaply in the shared-model MJX fast path. Mass/geometry/friction changes
# use batched MJX model leaves in v5014; geometry/topology still needs host compilation.
PUSH_PROB_PER_STEP = 0.0015   # tiny per-step chance a new push starts, at resilience_push_n == PUSH_MAX_FORCE_N
PUSH_MAX_FORCE_N = 0.35       # peak horizontal push force (newtons); robot is ~0.5kg, sized to impart a
                               # recoverable nudge (tens of mm/s), not an instant launch
PUSH_MIN_DURATION_STEPS = 2
PUSH_MAX_DURATION_STEPS = 8
# RESILIENCE ceilings for the episode's initial disturbance. These are
# deliberately recoverable perturbations: ~20 deg pose tilt, a few cm of
# position error, and a moderate angular-velocity kick rather than arbitrary
# fallen configurations.
POSE_TILT_MAX_RAD = 0.35     # rad (~20deg) -- RESILIENCE: initial roll/pitch lean at spawn
POSITION_NOISE_MAX_M = 0.02  # m -- RESILIENCE: initial xy/height offset at spawn
ANGVEL_NOISE_MAX_RAD_S = 1.5 # rad/s -- RESILIENCE: initial angular velocity at spawn
# STEADY and TRANSITION regime spawns still get a small taste of
# RESILIENCE's current envelope (pose tilt, position offset, angular-
# velocity kick) -- this fraction of it, not the full scaled amount
# RESILIENCE itself gets. Couples the regimes through an earned quantity
# rather than a second hand-tuned constant, without asking every regime to
# also be a resilience test at full strength.
STEADY_RESILIENCE_BORROW_FRAC = 0.15
# Regime-specific reward shaping. These are tunable design choices rather than
# hard physical limits; validate them against held-out behaviour before treating
# them as settled constants.
#   TRANSITION_TRACKING_BOOST: how much more heavily tracking error is
#     weighted specifically WHILE a resampled command is still ramping in
#     (ramp_frac<1), rewarding fast AND accurate convergence in the same
#     signal -- slow convergence keeps error large for longer, inaccurate
#     convergence keeps it large regardless of speed. 1.0 would mean no
#     special treatment at all (transitions score identically to holding).
#   RESILIENCE_TOLERANCE_MULT: how much wider the overspeed/underspeed dead
#     zone is during RESILIENCE. Physics already treats recovery specially
#     (push force, initial-state noise); without this, the reward's speed-
#     tracking penalty wouldn't, fighting the exact deviation needed to
#     catch balance.
#   RESILIENCE_TRACKING_RELIEF: multiplies the WHOLE weighted-error sum
#     (not just the overspeed/underspeed dead zone) during RESILIENCE,
#     softening the tracking exponential's sharpness overall -- real room
#     to deviate for balance recovery without the tracking term collapsing
#     toward zero the moment a push lands.

# Opt-in G3 platform. These are collision dimensions, NOT a terrain teacher.
# Episode height/direction/yaw live in mjx.Data.mocap_pos/mocap_quat, which
# can change at reset without rebuilding the shared model or JIT structure.
TERRAIN_PLATFORM_HALF_LENGTH_M = 0.90
TERRAIN_PLATFORM_HALF_WIDTH_M = 0.48
TERRAIN_PLATFORM_HALF_DEPTH_M = 0.15
TERRAIN_PLATFORM_EDGE_X_M = 0.25


def add_episode_platform(robot_xml: str) -> str:
    """Augment the existing robot model with one finite collision platform.

    A mocap body's real geom is moved only at episode boundaries.  It has
    no degrees of freedom and introduces no hidden channels to the policy.
    The on-disk robot.xml and default training configuration are untouched.
    """
    root = ET.parse(robot_xml).getroot()
    world = root.find('worldbody')
    if world is None:
        raise ValueError('robot.xml has no worldbody')
    # CRITICAL: a MuJoCo ground *plane* is an infinite half-space.
    # A platform buried BELOW it is not "out of contact": without
    # collision filtering the plane/box pair is deeply interpenetrating,
    # even in an episode where the terrain is disabled. That static pair
    # must never consume constraint/contact slots or perturb normal gait.
    #
    # Bit 1 = floor, bit 2 = terrain. Existing collision-enabled
    # robot geoms (feet and torso) can touch BOTH support types, but
    # passive non-contacting shin/thigh rods remain non-contacting.
    # Mask original robot geoms ONLY in this opt-in compiled model;
    # the on-disk robot and flat champion physics stay untouched.
    for geom_name in ('torso_box', 'r_foot_ball', 'l_foot_ball'):
        contact_geom = root.find(f".//geom[@name='{geom_name}']")
        if contact_geom is None:
            raise ValueError(f'missing robot contact geom {geom_name}')
        contact_geom.set('contype', '3')
        contact_geom.set('conaffinity', '3')
    platform = ET.SubElement(world, 'body', {
        'name': 'episode_terrain_platform', 'mocap': 'true',
        'pos': '0 0 -1',
    })
    ET.SubElement(platform, 'geom', {
        'name': 'episode_terrain_surface', 'type': 'box',
        'size': f'{TERRAIN_PLATFORM_HALF_LENGTH_M} {TERRAIN_PLATFORM_HALF_WIDTH_M} {TERRAIN_PLATFORM_HALF_DEPTH_M}',
        'contype': '2', 'conaffinity': '2',
        'friction': '1.5 0.005 0.0001', 'rgba': '0.65 0.50 0.24 1',
    })
    return ET.tostring(root, encoding='unicode')


# Unified terrain is a pair of adjacent *solid* convex slabs. Their tops
# exactly coincide when step height is zero, and differ by h along the common
# surface normal when a real step is active. Two boxes avoid six open mesh
# triangles being convex-hulled into a phantom ramp by MuJoCo.
UNIFIED_SLAB_HALF_LENGTH_M = 4.0
UNIFIED_SLAB_HALF_WIDTH_M = 3.0
UNIFIED_SLAB_HALF_DEPTH_M = .25


def add_unified_terrain(robot_xml: str) -> str:
    """Replace the infinite floor with movable lower + higher solid slabs.

    The old XML/main is NEVER modified. Floor and platform are two mocap
    collision boxes, not mesh-triangle soup. Distinct contact bits permit
    feet/torso vs either slab, while excluding slab-slab self contacts.
    Their poses are sampled per episode through mjx.Data, with NO per-episode
    model rebuild and no privileged actor-input channels.
    """
    root=ET.parse(robot_xml).getroot()
    world=root.find('worldbody')
    if world is None:
        raise ValueError('robot.xml requires worldbody')
    floor=world.find("geom[@name='floor']")
    if floor is None or floor.get('type') != 'plane':
        raise ValueError('unified physical terrain expects named plane floor')
    # Preserve original floor's contact friction and appearance.
    friction=floor.get('friction','1.5 0.005 0.0001')
    rgba=floor.get('rgba','0.2 0.25 0.3 1')
    world.remove(floor)
    for geom_name in ('torso_box','r_foot_ball','l_foot_ball'):
        geom=root.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise ValueError(f'missing contact geometry {geom_name}')
        geom.set('contype','3')
        geom.set('conaffinity','3')
    size=(f'{UNIFIED_SLAB_HALF_LENGTH_M} {UNIFIED_SLAB_HALF_WIDTH_M} '
          f'{UNIFIED_SLAB_HALF_DEPTH_M}')
    ground=ET.SubElement(world,'body',dict(name='unified_ground',mocap='true',
                                            pos='0 0 -0.25'))
    ET.SubElement(ground,'geom',dict(name='floor',type='box',size=size,
                                     contype='1',conaffinity='1',
                                     friction=friction,rgba=rgba))
    elevated=ET.SubElement(world,'body',dict(name='episode_terrain_platform',
                                              mocap='true',pos='0 0 -2'))
    ET.SubElement(elevated,'geom',dict(name='episode_terrain_surface',
                                      type='box',size=size,
                                      contype='2',conaffinity='2',
                                      friction=friction,rgba='0.65 0.50 0.24 1'))
    return ET.tostring(root,encoding='unicode')


class EnvState(NamedTuple):
    data: Any
    hist: Any
    rng: Any
    task: Any
    specialist_active: Any  # teacher-only birth focus, not an actor observation
    prev_action: Any
    # Reward-only previous foot pose for true stance-slip measurement.
    prev_foot_pos: Any
    # Previous raw target. Current hardware doctrine models one control-frame
    # command-delivery delay, so no configurable latency ring is carried.
    prev_raw_target: Any
    phase: Any
    step_i: Any
    ep_return: Any
    ep_len: Any
    ep_vx_sum: Any
    ep_cmd_vx_sum: Any
    ep_bad_sum: Any
    ep_support_sum: Any
    # Episode-so-far accumulated step_evidence, for the reward's anti-
    # freezing gate (see _reward). A running accumulator, not the
    # instantaneous per-step value -- gating on the instant value would
    # punish the normal stance phase of a genuine walking gait (feet are
    # legitimately grounded most of any real step cycle); this gates on
    # "has this episode shown a sustained pattern of stepping", which a
    # real gait clears easily and true freezing does not.
    ep_step_evidence_sum: Any
    ep_command_quality_sum: Any   # physical vx/vy/yaw tracking across entire episode
    terrain_cross_streak: Any
    terrain_crossed: Any
    # Ascent-only high-water marks, teacher state, never actor observations.
    # Prevent repeated lip-rocking and foot-tapping from farming rewards.
    terrain_ascent_root_peak: Any
    terrain_ascent_foot_peak: Any
    # The command actually being tracked this step ("cmd_now", written
    # into task[:,VX/VY/YAW] each step in _step_impl) ramps from
    # cmd_ramp_from toward cmd_target over cmd_ramp_frac in [0,1],
    # incremented by cmd_ramp_speed each step; cmd_hold_left counts down to
    # the next mid-episode resample. See _step_impl and _sample_cmd_target.
    cmd_target: Any
    cmd_ramp_from: Any
    cmd_ramp_frac: Any
    cmd_ramp_speed: Any
    cmd_hold_left: Any
    # Active push-perturbation force (N, world-frame xy) and how many more
    # steps it continues. Applied to the torso via data.xfrc_applied
    # *before* the physics substep loop in _step_impl -- applying it after
    # the physics step and clearing it before the next one would give it
    # zero actual effect, so this ordering is deliberate, not incidental.
    push_force: Any
    push_steps_left: Any
    # The v5016 opt-in sensor queue persists independent acquisition and
    # delivery timing. Legacy policies leave this at None.
    sensor_state: Any = None


class SensorSample(NamedTuple):
    """One raw sensor snapshot; no estimated grade or velocity stored."""
    tof_ranges_m: Any
    tof_beam_valid: Any
    terrain_ranges_m: Any
    terrain_valid: Any
    flow_rad_s: Any
    flow_valid: Any
    gyro_body_rad_s: Any
    tof_age_frames: Any
    flow_age_frames: Any


class SensorState(NamedTuple):
    """Four-frame acquisition ring (oldest first), plus episode gyro bias."""
    samples: SensorSample
    gyro_bias_rad_s: Any
    current_gyro_body_rad_s: Any


class PhysicalFrame(NamedTuple):
    """One applied servo frame; teacher commands are advanced separately."""
    data: Any
    action: Any
    raw_target: Any
    delivered_ctrl: Any
    push_force: Any
    push_steps_left: Any
    diverged: Any
    next_keys: Any
    reset_keys: Any
    resample_keys: Any


class StepOut(NamedTuple):
    state: EnvState
    obs: Any
    # Observation of the physical next state BEFORE auto-reset. PPO uses
    # this for exact time-limit bootstrapping; `obs` above is the normal
    # post-reset observation when done=True.
    bootstrap_obs: Any
    reward: Any
    done: Any
    metrics: dict[str, Any]


def horizontal_tracking_motion(xp, rotation_world, linear_world, angular_world):
    """Body-heading velocity across the contact plane, independent of torso pitch.

    A pitched chassis must not redefine what a requested forward or lateral
    ground speed means. Coordinates use the simulator's flat contact plane:
    local +X gives forward heading, and local +Y gives left. For v5015 only;
    actor IMU/body-frame observations retain their original sensor contract.
    """
    heading_xy = rotation_world[:, :2, 0]
    heading_xy = heading_xy / xp.maximum(
        xp.linalg.norm(heading_xy, axis=1, keepdims=True), 1e-6)
    hx, hy = heading_xy[:, 0], heading_xy[:, 1]
    vx = hx * linear_world[:, 0] + hy * linear_world[:, 1]
    vy = -hy * linear_world[:, 0] + hx * linear_world[:, 1]
    yaw_rate = angular_world[:, 2]
    return vx, vy, yaw_rate


# Read-only motor observables, distinct from scoring. Joint positions and
# velocities are true MJX physics values; actuator_force is the actual position
# actuator output (N m) after actuator limits, NOT servo electrical current.
# These are per-control-frame *instantaneous* estimates, not joule integration,
# winding temperature, battery watts or a calibrated thermal model.
SERVO_RATED_CONTINUOUS_TORQUE_NM = 0.638  # STS3215 nominal rating, robot.xml


def servo_effort_metrics(xp, torque_nm, joint_rate_rad_s, stall_limit_nm):
    """Six-actuator nominal shaft demand from MuJoCo actuator force and qvel.

    Shapes: [worlds, 6] inputs -> [worlds] named diagnostics.
    Drive/brake are mechanical power sign splits; braking isn't regeneration.
    A high fraction over rated torque indicates load exposure, not overheating.
    """
    power = torque_nm * joint_rate_rad_s
    torque_abs = xp.abs(torque_nm)
    return {
        'servo_shaft_abs_power_w': xp.sum(xp.abs(power), axis=1),
        'servo_shaft_drive_power_w': xp.sum(xp.maximum(power, 0.0), axis=1),
        'servo_shaft_brake_power_w': xp.sum(xp.maximum(-power, 0.0), axis=1),
        'servo_over_rated_frac': xp.mean((torque_abs > SERVO_RATED_CONTINUOUS_TORQUE_NM).astype(torque_nm.dtype), axis=1),
        'servo_peak_torque_limit_frac': xp.max(torque_abs / xp.maximum(stall_limit_nm, 1e-6), axis=1),
    }


def joint_spring_energy(xp, joint_position_rad, spring_ref_rad, stiffness_nm_rad):
    """Stored elastic potential: sum(0.5 * stiffness * (q - springref)^2).

    Includes six driven hinge springs and both passive shin-flex springs.
    This is a *state* in joules, not a net transfer from springs to motion.
    For the nominal XML model, stiffness is not domain-randomised.
    """
    energy = 0.5 * stiffness_nm_rad * (joint_position_rad - spring_ref_rad) ** 2
    return {
        'spring_potential_energy_j': xp.sum(energy, axis=1),
        'shin_spring_energy_j': xp.sum(energy[:, -2:], axis=1),
    }


# Commands/tasks share the same MJX physical-world state and RNG.
from types import SimpleNamespace
from typing import Any, NamedTuple
from .contracts import (
    CONTROL_DT, COMBO_P, PHASE_RATE_MAX_HZ, SEED_VX_HI, SEED_VX_LO,
    FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS, FIELD_NODE_STRIDE,
    FIELD_EDGE_STRIDE, FIELD_NODE_START, FIELD_EDGE_START,
    FIELD_EDGE_SCALES, P,
)

# Command-transition dynamics (see _sample_cmd_target / TRANSITION's own
# difficulty path in _transition_hold_and_jump).
TRANSITION_HOLD_MIN_STEPS = 45.0   # shortest command hold, reached when min_hold_s bottoms out
TRANSITION_HOLD_MAX_STEPS = 700.0  # longest hold, the starting value (~never resamples mid-episode)
TRANSITION_RAMP_MIN_STEPS = 4.0    # near-instant target change, the hardest end
TRANSITION_RAMP_MAX_STEPS = 120.0  # gentle ramp, the starting/easy value
CARRYOVER_MAX_PROB = 0.35          # at TRANSITION's hardest point, up to this fraction of resets carry over qvel
# RESILIENCE-regime episodes get the full current push ceiling -- that's
# the regime actually testing it. STEADY/TRANSITION get a reduced fraction
# instead of the same full level, so ordinary walking practice stays
# representative of normal operation rather than being perturbed as hard
# as a dedicated recovery drill.
REGULAR_RESILIENCE_FRAC = 0.25
# Curriculum ceilings are written in physical units. Each regime's 0..1
# difficulty scalar interpolates between a seed value and these limits.
VX_HI_MAX = 0.45             # m/s -- STEADY: vx_hi walks from SEED_VX_HI up to this
VX_LO_MIN = -0.15            # m/s -- STEADY: vx_lo walks from SEED_VX_LO down to this
YAW_MAX_RAD_S_LIMIT = 1.20   # rad/s -- STEADY: yaw_max walks from 0 up to this
VY_MAX_MPS_LIMIT = 0.18     # m/s -- STEADY: vy_max walks from 0 up to this
HEIGHT_RANGE_MAX_M = 0.018   # m -- POSTURE (global): height_range walks from 0 up to
                              # this; pitch/roll/step-height ranges derive from the
                              # same posture_difficulty, since they're a genuinely
                              # coupled "how much does static pose vary" bundle, not
                              # four independently-meaningful physical limits.
TRANSITION_HOLD_MAX_S = TRANSITION_HOLD_MAX_STEPS * CONTROL_DT  # ~14.0s -- TRANSITION:
                                                                  # easiest end of hold-time
TRANSITION_HOLD_MIN_S = TRANSITION_HOLD_MIN_STEPS * CONTROL_DT  # ~0.9s -- hardest end
# Short reward-settle window after reset. Termination remains active; only
# tracking score avoids over-interpreting a few frames of spawn transient.
# STEADY and RESILIENCE regimes never resample their command --
# cmd_hold_left is initialized to this for them, safely larger than any
# real episode length (max_episode_steps is ~700), so the resample branch
# in _step_impl simply never fires. int32-safe (max ~2.1 billion).
INFINITE_HOLD_STEPS = 1_000_000_000
# Per-environment task-vector schema.  The actor only sees the command/sensor
# subset exposed by _base_obs; regime, birth difficulty and rehearsal are
# teacher-only state.  Resilience is a unitless disturbance fraction and is
# converted to Newtons exactly once in _step_impl.
TASK_FIELDS = (
    "regime", "vx", "vy", "yaw", "height", "pitch", "roll", "step_height",
    "resilience", "phase_rate",
    # Immutable-at-birth diagnostics: score an episode against the difficulty
    # that created it, not the possibly newer governor state when it ends.
    "birth_steady_d", "birth_transition_d", "birth_resilience_d", "birth_posture_d",
    "athletic_rehearsal",  # teacher-only full-envelope rehearsal tag
    "sensor_scale",        # observation corruption scale; never an actor channel
    "field_taught",        # teacher-only provenance; never actor/reward input
)
T = SimpleNamespace(**{name: i for i, name in enumerate(TASK_FIELDS)})
TASK_DIM = len(TASK_FIELDS)


def command_phase_rate(vx, vy, vx_hi_max, phase_speed_gain, base_phase_rate):
    """Sampled gait cadence: base 0.85–1.20 Hz plus birth-speed gain, max 2.6 Hz.

    Keep pitch-command limits out of this calculation. This function also gives
    CPU-only contract tests a way to catch invalid JAX argument signatures
    before a multi-hour MJX GPU compile.
    """
    import jax.numpy as jnp
    speed_fraction = jnp.clip(jnp.sqrt(vx * vx + vy * vy) / jnp.maximum(0.05, vx_hi_max), 0.0, 1.5)
    return jnp.minimum(PHASE_RATE_MAX_HZ, base_phase_rate + phase_speed_gain * speed_fraction)



class CommandFrame(NamedTuple):
    """The next command request, its ramp state, and transition diagnostics."""
    task: Any                 # advanced command for the next observation
    stored_task: Any          # next state, including teacher-only provenance
    ramp_frac: Any
    transition_active: Any
    cmd_resampled: Any
    target_next: Any
    ramp_from_next: Any
    ramp_frac_next: Any
    ramp_speed_next: Any
    hold_left_next: Any


class CommandEnvelope(NamedTuple):
    """Velocity/yaw command limits; full and reduced share the same units."""
    vx_hi: Any
    vx_lo: Any
    vy: Any
    yaw: Any


class TaskSampler:
    """Stateless lesson-generation methods consumed by EmuMJXEnv.

    Requires the existing environment's JAX context, num_envs, physical
    command ceilings, attitude profile and optional capability-field knobs.
    It does not store a second copy of that configuration or model state.
    """

    def _lerp(self, seed, ceiling, difficulty):
        """The one function every regime's difficulty scalar goes through
        to become a real physical range -- plain linear interpolation from
        seed to ceiling (see the module doctrine for why one shared scalar
        with no per-component rate table is enough)."""
        jnp = self.jnp
        return seed + jnp.clip(difficulty, 0.0, 1.0) * (ceiling - seed)

    def _field_node_table(self, params):
        return params[FIELD_NODE_START:FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE].reshape((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE))

    def _field_edge_table(self, params):
        return params[FIELD_EDGE_START:FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE].reshape((FIELD_EDGE_ANCHORS, FIELD_EDGE_STRIDE))

    def _weighted_anchor_index(self, key, weights):
        jax, jnp = self.jax, self.jnp
        w = jnp.maximum(weights, 0.0)
        total = jnp.sum(w)
        u = jax.random.uniform(key, ()) * jnp.maximum(total, 1e-6)
        idx = jnp.sum(u > jnp.cumsum(w)).astype(jnp.int32)
        return jnp.clip(idx, 0, w.shape[0]-1), total

    def _command_envelopes(self, params) -> tuple[CommandEnvelope, CommandEnvelope]:
        """One source for birth and transition target bounds.

        A physical frontier, when enabled, wins over the 0–1 steady
        difficulty interpolation. The reduced envelope keeps the original
        regime_command_frac lerp back toward the two seed velocity bounds.
        This is deterministic JAX maths: NO extra random key is consumed.
        """
        jnp = self.jnp
        steady_d = params[P.steady_difficulty]
        command_frac = jnp.clip(params[P.regime_command_frac], 0.0, 1.0)
        # The unified actor first receives main's ordinary steady-difficulty
        # command interpolation. The newly-earned physical limits gradually
        # take over after the foundation. Legacy actor maths is untouched.
        vx_hi = self._lerp(SEED_VX_HI, self.vx_hi_max, steady_d)
        vx_lo = self._lerp(SEED_VX_LO, self.vx_lo_min, steady_d)
        yaw_max = self._lerp(0.0, self.yaw_max_rad_s, steady_d)
        vy_max = self._lerp(0.0, self.vy_max_mps, steady_d)
        if self.open_frontier:
            if self.unified_capability_focus:
                mix = jnp.clip(params[P.specialist_mix], 0., 1.)
                # A newly seeded specialist frontier (.06 m/s, etc.) is
                # initially SMALLER than main's already-earned 20M command
                # experience. Never replace that successful foundation with
                # an easier limit during the specialist fade-in. Only let
                # specialist evidence EXTEND each signed main-style axis.
                vx_hi = jnp.maximum(vx_hi,
                    self._lerp(SEED_VX_HI,
                        jnp.maximum(SEED_VX_HI,params[P.frontier_vx_hi]),mix))
                vx_lo = jnp.minimum(vx_lo,
                    self._lerp(SEED_VX_LO,
                        jnp.minimum(SEED_VX_LO,params[P.frontier_vx_lo]),mix))
                yaw_max = jnp.maximum(yaw_max,
                    mix*jnp.maximum(0.,params[P.frontier_yaw]))
                vy_max = jnp.maximum(vy_max,
                    mix*jnp.maximum(0.,params[P.frontier_vy]))
            else:
                vx_hi = jnp.maximum(SEED_VX_HI, params[P.frontier_vx_hi])
                vx_lo = jnp.minimum(SEED_VX_LO, params[P.frontier_vx_lo])
                yaw_max = jnp.maximum(0.0, params[P.frontier_yaw])
                vy_max = jnp.maximum(0.0, params[P.frontier_vy])
        full = CommandEnvelope(vx_hi, vx_lo, vy_max, yaw_max)
        reduced = CommandEnvelope(
            SEED_VX_HI + command_frac * (vx_hi - SEED_VX_HI),
            SEED_VX_LO + command_frac * (vx_lo - SEED_VX_LO),
            command_frac * vy_max, command_frac * yaw_max,
        )
        return full, reduced

    def _sample_task(self, keys, params):
        """Draws each env's regime (STEADY/TRANSITION/RESILIENCE) and that
        regime's command envelope, built from one difficulty scalar via
        _lerp. TRANSITION/RESILIENCE spawn at a fraction of STEADY's
        current envelope (regime_command_frac) rather than a range of
        their own. Initial-state noise (pose tilt, position, angular
        velocity) for RESILIENCE is NOT sampled here: those are one-time
        reset perturbations, not an ongoing target that needs to persist
        for the whole episode the way vx/vy/yaw/posture do, so they're
        computed directly in _reset_with_keys from resilience_difficulty
        and this task's own regime column instead of bloating TASK_DIM
        with fields that are only ever read once.
        """
        jax, jnp = self.jax, self.jnp
        steady_d = params[P.steady_difficulty]
        posture_d = params[P.posture_difficulty]
        resilience_d = params[P.resilience_difficulty]
        # The SAME physical limits now govern birth and later transitions.
        full, reduced = self._command_envelopes(params)
        vx_hi, vx_lo, vy_max, yaw_max = full
        vx_hi_reduced, vx_lo_reduced, vy_max_reduced, yaw_max_reduced = reduced

        # POSTURE (global, every regime): height/pitch/roll/step-height are
        # a coupled bundle, all driven by the one posture_difficulty
        # fraction directly.
        height_range = self._lerp(0.0, HEIGHT_RANGE_MAX_M, posture_d)
        pitch_range = self.pitch_limit_rad * posture_d
        roll_range = 0.055 * posture_d
        step_height_range = 0.030 * jnp.maximum(0.05, posture_d)

        # RESILIENCE difficulty remains unitless here. Non-resilience
        # regimes still get a small AMBIENT fraction of it
        # (REGULAR_RESILIENCE_FRAC), and _step_impl converts that fraction
        # to Newtons exactly once at the force-application site.

        regime_probs = jnp.asarray([params[P.regime_frac_steady], params[P.regime_frac_transition],
                                    params[P.regime_frac_resilience]], dtype=jnp.float32)
        regime_probs = jnp.maximum(regime_probs, 0.0)
        regime_probs = regime_probs / jnp.maximum(jnp.sum(regime_probs), 1e-6)
        regime_cdf = jnp.cumsum(regime_probs)

        def one(k):
            # Independent draws, one purpose each -- no key is reused
            # for two different quantities. A reused key here isn't just
            # "correlated" output, it can be an exact bit-for-bit collision
            # depending on how two different split-counts happen to overlap,
            # which is why every draw below gets its own key rather than
            # sharing one across quantities.
            # Keep split arity stable so deleting a dead task feature does not
            # reshuffle unrelated stochastic streams under the same seed.
            k_regime, k_vx, k_vy, k_yaw, k_height, k_pitch, k_roll, k_step, k_phase, _reserved, k_athletic, k_combo, k_axis, k_radius, k_field_use, k_field_pick = jax.random.split(k, 16)
            u = jax.random.uniform(k_regime, ())
            regime = jnp.sum(u > regime_cdf).astype(jnp.float32)
            # Full-envelope rehearsal preserves the proven athletic distribution
            # while the adaptive frontier explores beyond it.
            is_athletic = jax.random.uniform(k_athletic, ()) < jnp.clip(params[P.athletic_rehearsal_frac], 0.0, 0.5)
            # Rehearsal is STEADY so it never contaminates transition/push tests.
            regime = jnp.where(is_athletic, 0.0, regime)
            is_steady = regime < 0.5
            is_transition = jnp.logical_and(regime > 0.5, regime < 1.5)
            is_narrow = jnp.logical_not(is_steady)  # TRANSITION or RESILIENCE: both use the reduced envelope

            use_vx_hi = jnp.where(is_narrow, vx_hi_reduced, vx_hi)
            use_vx_lo = jnp.where(is_narrow, vx_lo_reduced, vx_lo)
            use_vy = jnp.where(is_narrow, vy_max_reduced, vy_max)
            use_yaw = jnp.where(is_narrow, yaw_max_reduced, yaw_max)
            vx = jax.random.uniform(k_vx, (), minval=use_vx_lo, maxval=use_vx_hi)
            vy = jax.random.uniform(k_vy, (), minval=-use_vy, maxval=use_vy)
            yaw = jax.random.uniform(k_yaw, (), minval=-use_yaw, maxval=use_yaw)

            # Command-combination curriculum: a configurable fraction of
            # samples are true mixed-axis commands constrained by a p-norm
            # blob; the remainder are axis-isolation samples so each boundary
            # remains measurable. Mixed samples sometimes probe just beyond the
            # current blob radius, which is how local shape expansion gets data.
            if self.combo_mix_frac > 0.0:
                combo = jax.random.uniform(k_combo, ()) < self.combo_mix_frac
                axis = jax.random.randint(k_axis, (), 0, 3)
                vx_scale = jnp.where(vx >= 0.0, jnp.maximum(use_vx_hi, 1e-4), jnp.maximum(-use_vx_lo, 1e-4))
                nx = vx / vx_scale
                ny = vy / jnp.maximum(use_vy, 1e-4)
                nz = yaw / jnp.maximum(use_yaw, 1e-4)
                norm = jnp.power(jnp.abs(nx)**COMBO_P + jnp.abs(ny)**COMBO_P + jnp.abs(nz)**COMBO_P, 1.0/COMBO_P)
                combo_radius = jnp.where(self.open_frontier, jnp.maximum(0.10, params[P.frontier_combo_radius]), 1.0)
                # Mixed samples cover the blob interior and a thin just-outside
                # probe band instead of living only on the mathematical boundary.
                radial = 0.35 + (0.65 + self.combo_probe_frac) * jax.random.uniform(k_radius, ())
                probe = combo_radius * radial
                scale = jnp.where(norm > 1e-6, probe / jnp.maximum(norm, 1e-6), 1.0)
                vx_m, vy_m, yaw_m = vx*scale, vy*scale, yaw*scale
                vx_a = jnp.where(axis == 0, vx, 0.0)
                vy_a = jnp.where(axis == 1, vy, 0.0)
                yaw_a = jnp.where(axis == 2, yaw, 0.0)
                # Preserve the seed-band forward motion when isolating vy/yaw;
                # this unstable robot is not trained to stand statically.
                vx_a = jnp.where(axis == 0, vx_a, jnp.clip(0.5*(SEED_VX_LO+SEED_VX_HI), use_vx_lo, use_vx_hi))
                vx = jnp.where(combo, vx_m, vx_a)
                vy = jnp.where(combo, vy_m, vy_a)
                yaw = jnp.where(combo, yaw_m, yaw_a)
            # A bounded fraction of ordinary births
            # may be drawn from the capability field's current teaching packet.
            # Anchors are clipped to the governor's current envelope; the field
            # chooses *where* to spend practice while the governor still owns
            # global safety/frontier expansion.
            node_table = self._field_node_table(params)
            ni, node_total = self._weighted_anchor_index(k_field_pick, node_table[:, 3])
            node_cmd = node_table[ni, :3]
            node_recovery = jnp.clip(node_table[ni, 4], 0.0, 1.0)
            field_use = jnp.logical_and(
                jax.random.uniform(k_field_use, ()) < jnp.clip(params[P.field_node_frac], 0.0, 0.95),
                node_total > 1e-6)
            field_use = jnp.logical_and(field_use, jnp.logical_not(is_athletic))
            nvx = jnp.clip(node_cmd[0], use_vx_lo, use_vx_hi)
            nvy = jnp.clip(node_cmd[1], -use_vy, use_vy)
            nyaw = jnp.clip(node_cmd[2], -use_yaw, use_yaw)
            vx = jnp.where(field_use, nvx, vx)
            vy = jnp.where(field_use, nvy, vy)
            yaw = jnp.where(field_use, nyaw, yaw)
            height = jax.random.uniform(k_height, (), minval=-height_range, maxval=height_range)
            pitch = jax.random.uniform(k_pitch, (), minval=-pitch_range, maxval=pitch_range)
            roll = jax.random.uniform(k_roll, (), minval=-roll_range, maxval=roll_range)
            step_height = 0.012 + jax.random.uniform(k_step, (), minval=0.0, maxval=step_height_range)
            base_phase_rate = jax.random.uniform(k_phase, (), minval=0.85, maxval=1.20)
            phase_rate = command_phase_rate(vx, vy, self.vx_hi_max,
                                            self.phase_speed_gain, base_phase_rate)
            # Keep disturbance intensity unitless in the task vector. Convert it
            # to Newtons only where the external force is applied.
            # The field may choose a locally useful recovery severity for a
            # resilience birth, but never above the governor's globally-earned
            # resilience difficulty.  This turns the recovery layer into an
            # actual curriculum signal while preserving the governor as the
            # physical safety ceiling.
            if self.field_teach_recovery:
                birth_resilience_d = jnp.where(
                    jnp.logical_and(field_use, regime > 1.5),
                    jnp.minimum(node_recovery, jnp.clip(resilience_d, 0.0, 1.0)),
                    jnp.clip(resilience_d, 0.0, 1.0))
            else:
                birth_resilience_d = jnp.clip(resilience_d, 0.0, 1.0)
            if self.open_frontier:
                resilience_frac = jnp.clip(params[P.frontier_push_n] / jnp.maximum(self.push_max_force_n, 1e-6), 0.0, 1.0)
                if self.unified_capability_focus:
                    # Main's foundation scaled pushes with earned resilience
                    # difficulty and a 0.70 N physical maximum. Unified has
                    # 3.2 N *headroom* but must not use its whole frontier
                    # merely because open-frontier sampling is enabled.
                    foundation_frac = birth_resilience_d * (
                        .70 / jnp.maximum(self.push_max_force_n, 1e-6))
                    resilience_frac = self._lerp(
                        foundation_frac, resilience_frac,
                        jnp.clip(params[P.specialist_mix], 0., 1.))
            else:
                resilience_frac = birth_resilience_d
            resilience_task = jnp.where(regime > 1.5, jnp.maximum(resilience_frac, 0.08 / jnp.maximum(self.push_max_force_n, 1e-6)),
                                        resilience_frac * REGULAR_RESILIENCE_FRAC)
            sensor_scale_task = jnp.maximum(0.0, params[P.frontier_sensor_scale])
            # Full athletic rehearsal samples the proven control
            # envelope directly. The three axes are intentionally independent:
            # this is memory rehearsal of the old rectangular distribution, not
            # evidence for the new p-norm frontier.
            vx_ath = jax.random.uniform(k_vx, (), minval=-0.20, maxval=0.60)
            vy_ath = jax.random.uniform(k_vy, (), minval=-0.22, maxval=0.22)
            yaw_ath = jax.random.uniform(k_yaw, (), minval=-1.50, maxval=1.50)
            vx = jnp.where(is_athletic, vx_ath, vx)
            vy = jnp.where(is_athletic, vy_ath, vy)
            yaw = jnp.where(is_athletic, yaw_ath, yaw)
            height = jnp.where(is_athletic, 0.0, height)
            pitch = jnp.where(is_athletic, 0.0, pitch)
            roll = jnp.where(is_athletic, 0.0, roll)
            step_height = jnp.where(is_athletic, 0.012, step_height)
            resilience_task = jnp.where(is_athletic, 0.0, resilience_task)

            # Capability-atlas command probes use exact physical commands,
            # bypassing random envelope sampling without changing the actor.
            vx = jnp.where(jnp.isfinite(params[P.probe_cmd_vx]), params[P.probe_cmd_vx], vx)
            vy = jnp.where(jnp.isfinite(params[P.probe_cmd_vy]), params[P.probe_cmd_vy], vy)
            yaw = jnp.where(jnp.isfinite(params[P.probe_cmd_yaw]), params[P.probe_cmd_yaw], yaw)
            cols = [regime, vx, vy, yaw, height, pitch, roll, step_height, resilience_task, phase_rate,
                    steady_d, params[P.transition_difficulty], birth_resilience_d, posture_d,
                    is_athletic.astype(jnp.float32), sensor_scale_task, field_use.astype(jnp.float32)]
            assert len(cols) == TASK_DIM, f"task vector has {len(cols)} columns, TASK_DIM says {TASK_DIM}"
            return jnp.asarray(cols, dtype=jnp.float32)
        return jax.vmap(one)(keys)

    def _sample_cmd_target(self, keys, params, current_cmd=None):
        """Draws a fresh (vx, vy, yaw) TRANSITION-regime target.
        Only ever called for envs currently in TRANSITION (see _step_impl's
        resample gate) -- STEADY and RESILIENCE hold their one spawn target
        for the whole episode, so there's nothing to redraw for them. The
        envelope is scaled TWICE: once by regime_command_frac (transition
        never fights steady's own full command difficulty at the same
        time -- see module doctrine), then again by jump_frac, which is 0
        early in transition_difficulty's own range and 1 from its midpoint
        on -- see _transition_hold_and_jump for why jump size and hold time
        are two separately-progressing halves of ONE scalar, not one axis
        moving both at once.
        """
        jax, jnp = self.jax, self.jnp
        _, jump_frac = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            jump_frac = jnp.asarray(1.0, dtype=jnp.float32)
        _, reduced = self._command_envelopes(params)
        vx_hi, vx_lo, vy_max, yaw_max = reduced
        vx_hi_j = SEED_VX_HI + jump_frac * (vx_hi - SEED_VX_HI)
        vx_lo_j = SEED_VX_LO + jump_frac * (vx_lo - SEED_VX_LO)
        yaw_max_j = jump_frac * yaw_max
        vy_max_j = jump_frac * vy_max

        node_table = self._field_node_table(params)
        edge_table = self._field_edge_table(params)
        if current_cmd is None:
            current_cmd = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)

        def one(k, cur):
            k1, k2, k3, k_node_use, k_node_pick, k_edge_use, k_edge_pick = jax.random.split(k, 7)
            vx = jax.random.uniform(k1, (), minval=vx_lo_j, maxval=vx_hi_j)
            vy = jax.random.uniform(k2, (), minval=-vy_max_j, maxval=vy_max_j)
            yaw = jax.random.uniform(k3, (), minval=-yaw_max_j, maxval=yaw_max_j)
            base = jnp.stack([vx, vy, yaw])

            ni, node_total = self._weighted_anchor_index(k_node_pick, node_table[:, 3])
            node_target = node_table[ni, :3]
            use_node = jnp.logical_and(jax.random.uniform(k_node_use, ()) < jnp.clip(params[P.field_node_frac], 0.0, 0.95), node_total > 1e-6)

            # Graph anchors are local source->target maneuvers.  Prefer anchors
            # whose source is near this env's current command, without adding a
            # high-dimensional table to the actor or policy network.
            scale = jnp.asarray(FIELD_EDGE_SCALES, dtype=jnp.float32)
            dist2 = jnp.sum(((edge_table[:, :3] - cur[None, :]) / scale[None, :]) ** 2, axis=1)
            local_w = jnp.maximum(edge_table[:, 6], 0.0) * jnp.exp(-0.5 * dist2 / (0.45**2))
            ei, edge_total = self._weighted_anchor_index(k_edge_pick, local_w)
            edge_target = edge_table[ei, 3:6]
            use_edge = jnp.logical_and(jax.random.uniform(k_edge_use, ()) < jnp.clip(params[P.field_edge_frac], 0.0, 0.95), edge_total > 1e-6)

            target = jnp.where(use_node, node_target, base)
            target = jnp.where(use_edge, edge_target, target)  # graph has priority
            target = target.at[0].set(jnp.clip(target[0], vx_lo_j, vx_hi_j))
            target = target.at[1].set(jnp.clip(target[1], -vy_max_j, vy_max_j))
            target = target.at[2].set(jnp.clip(target[2], -yaw_max_j, yaw_max_j))
            target = target.at[0].set(jnp.where(jnp.isfinite(params[P.probe_cmd_vx]), params[P.probe_cmd_vx], target[0]))
            target = target.at[1].set(jnp.where(jnp.isfinite(params[P.probe_cmd_vy]), params[P.probe_cmd_vy], target[1]))
            target = target.at[2].set(jnp.where(jnp.isfinite(params[P.probe_cmd_yaw]), params[P.probe_cmd_yaw], target[2]))
            field_taught = jnp.logical_or(use_node, use_edge).astype(jnp.float32)
            return target, field_taught
        return jax.vmap(one)(keys, current_cmd)

    def _transition_hold_and_jump(self, transition_difficulty):
        """Map one transition difficulty scalar onto jump size and frequency.

        Difficulty 0..0.5 increases command-jump magnitude while keeping a
        generous hold time. Difficulty 0.5..1.0 keeps the jump range at maximum
        and shortens the hold time. This preserves one curriculum decision while
        avoiding simultaneous movement of both transition dimensions.
        """
        jnp = self.jnp
        d = jnp.clip(transition_difficulty, 0.0, 1.0)
        jump_frac = jnp.clip(d / 0.5, 0.0, 1.0)
        hold_frac = jnp.clip((d - 0.5) / 0.5, 0.0, 1.0)
        min_hold_s = TRANSITION_HOLD_MAX_S - hold_frac * (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
        return min_hold_s, jump_frac

    def _init_transition_state(self, task, params, hold_keys=None):
        """(cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed,
        cmd_hold_left) for a freshly (re)sampled task. frac=1.0 means
        "already arrived" -- task's own VX/VY/YAW is exactly the value to
        start at, no ramp-in needed. Only TRANSITION-regime envs ever get a
        finite hold; STEADY and RESILIENCE get a hold effectively longer
        than any episode (INFINITE_HOLD_STEPS), because neither regime
        resamples its command mid-episode -- see the module doctrine for
        why RESILIENCE holds one target throughout rather than also
        transitioning. `hold_keys` is None only for the one-time env-
        creation reset in _reset_impl, where per-env draw diversity doesn't
        matter (every real episode reset goes through _reset_with_keys
        instead, which always passes real keys).
        """
        jax, jnp = self.jax, self.jnp
        # v5015 adds a continuously changing chassis-pitch target to the
        # existing task ramps. The learned field remains 3D (vx,vy,yaw).
        columns = ([T.vx, T.vy, T.yaw, T.pitch] if self.attitude_profile == 'v5015'
                   else [T.vx, T.vy, T.yaw])
        cmd_target = task[:, columns]
        cmd_ramp_from = cmd_target
        cmd_ramp_frac = jnp.ones((self.num_envs,), dtype=jnp.float32)
        cmd_ramp_speed = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        is_transition = jnp.logical_and(task[:, T.regime] > 0.5, task[:, T.regime] < 1.5)
        min_hold_s, _ = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P.frontier_transition_hold_s])
        hold_min = min_hold_s / CONTROL_DT  # steps
        if hold_keys is None:
            transition_hold = jnp.full((self.num_envs,), hold_min * 0.8, dtype=jnp.int32)
        else:
            hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(hold_keys)
            transition_hold = jnp.round(hold_min * (0.6 + 0.4 * hold_draw)).astype(jnp.int32)
        cmd_hold_left = jnp.where(is_transition, transition_hold, INFINITE_HOLD_STEPS)
        return cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left

    def _advance_commands(self, state, params, resample_keys) -> CommandFrame:
        """Advance the lesson and prepare the command for the NEXT actor step.

        The caller must score this frame against state.task, not the result
        returned here; the actor cannot react to information it hasn't seen.
        """
        jax, jnp = self.jax, self.jnp
        # Command-transition dynamics: ramps the *tracked* command
        # (task[:, VX/VY/YAW], which every other function below reads)
        # smoothly toward an occasionally-resampled target instead of
        # Command hold/ramp timing tightens with transition difficulty. The
        # A transition is defined by the PRE-advance ramp state.  This is
        # crucial at the digital limit: when ramp_speed==1 a newly-resampled
        # command advances 0->1 in one control frame, so testing ramp_frac<1
        # only *after* the advance made the sharpest maneuver invisible.
        transition_active = state.cmd_ramp_frac < 1.0
        cmd_resampled = transition_active & (state.cmd_ramp_frac <= 1e-6) & (state.cmd_ramp_speed > 0.0)
        ramp_frac = jnp.clip(state.cmd_ramp_frac + state.cmd_ramp_speed, 0.0, 1.0)
        cmd_now = state.cmd_ramp_from + ramp_frac[:, None] * (state.cmd_target - state.cmd_ramp_from)
        task = state.task.at[:, T.vx].set(cmd_now[:, 0]).at[:, T.vy].set(cmd_now[:, 1]).at[:, T.yaw].set(cmd_now[:, 2])
        if self.attitude_profile == 'v5015':
            task = task.at[:, T.pitch].set(cmd_now[:, 3])

        hold_left = state.cmd_hold_left - 1
        should_resample = hold_left <= 0
        rk_target, rk_hold, rk_ramp = self._split_batch(resample_keys, 3)
        # Only TRANSITION-regime envs ever have a finite cmd_hold_left (see
        # _init_transition_state) -- STEADY/RESILIENCE's INFINITE_HOLD_STEPS
        # means should_resample is always False for them, so new_target/
        # new_hold below are computed for every env (vmap has no branch)
        # but only ever actually USED where should_resample is True, i.e.
        # only ever for TRANSITION envs.
        new_target, new_field_taught = self._sample_cmd_target(rk_target, params, cmd_now[:, :3])
        if self.attitude_profile == 'v5015':
            pitch_keys = jax.vmap(lambda k: jax.random.fold_in(k, 5015))(rk_target)
            pitch_range = self.pitch_limit_rad * jnp.clip(task[:, T.birth_posture_d], 0.0, 1.0)
            pitch_draws = jax.vmap(lambda k: jax.random.uniform(k, (), minval=-1.0, maxval=1.0))(pitch_keys)
            new_target = jnp.concatenate((new_target, (pitch_draws * pitch_range)[:, None]), axis=1)
        hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_hold)
        ramp_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_ramp)
        min_hold_s, _ = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P.frontier_transition_hold_s])
        hold_min = min_hold_s / CONTROL_DT  # steps
        new_hold = jnp.maximum(1.0, jnp.round(hold_min * (0.6 + 0.4 * hold_draw))).astype(jnp.int32)
        if self.open_frontier:
            # Explicit acceleration frontier: ramp duration is a real count of
            # 20ms control frames and may shrink toward one frame, the natural
            # digital limit. No arbitrary locomotion-speed ceiling is involved.
            ramp_min = jnp.maximum(1.0, params[P.frontier_ramp_steps])
            new_ramp_steps = jnp.maximum(1.0, jnp.round(ramp_min * (0.75 + 0.50 * ramp_draw)))
        else:
            trans_frac = (TRANSITION_HOLD_MAX_S - min_hold_s) / (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
            ramp_min = TRANSITION_RAMP_MIN_STEPS + (1.0 - trans_frac) * (TRANSITION_RAMP_MAX_STEPS - TRANSITION_RAMP_MIN_STEPS)
            new_ramp_steps = jnp.maximum(1.0, jnp.round(ramp_min * (0.5 + 0.5 * ramp_draw)))
        new_ramp_speed = 1.0 / new_ramp_steps

        cmd_target_next = jnp.where(should_resample[:, None], new_target, state.cmd_target)
        cmd_ramp_from_next = jnp.where(should_resample[:, None], cmd_now, state.cmd_ramp_from)
        cmd_ramp_frac_next = jnp.where(should_resample, 0.0, ramp_frac)
        cmd_ramp_speed_next = jnp.where(should_resample, new_ramp_speed, state.cmd_ramp_speed)
        cmd_hold_left_next = jnp.where(should_resample, new_hold, hold_left).astype(jnp.int32)
        # Provenance follows the newly-selected transition target from the next
        # state onward.  It never enters actor observation or reward.
        task_for_state = task.at[:, T.field_taught].set(
            jnp.where(should_resample, new_field_taught, task[:, T.field_taught]))

        return CommandFrame(
            task=task, stored_task=task_for_state,
            ramp_frac=ramp_frac, transition_active=transition_active,
            cmd_resampled=cmd_resampled, target_next=cmd_target_next,
            ramp_from_next=cmd_ramp_from_next,
            ramp_frac_next=cmd_ramp_frac_next,
            ramp_speed_next=cmd_ramp_speed_next,
            hold_left_next=cmd_hold_left_next,
        )

# MJX model-level domain variations; physical topology remains fixed.
from dataclasses import dataclass, asdict
from typing import Any
import math


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

class EmuMJXEnv(TaskSampler):
    """Batched robot universe, with task sampling inherited from TaskSampler.

    The inherited sampler is deliberately stateless: it uses this environment's
    existing JAX context and physical limits, so no parallel teacher state
    can drift from the simulated world.
    """

    def __init__(self, robot_xml: str, num_envs: int, seed: int = 733, max_episode_steps: int = 700,
                 push_max_force_n: float | None = None,
                 vx_hi_max: float = VX_HI_MAX, vx_lo_min: float = VX_LO_MIN,
                 vy_max_mps: float = VY_MAX_MPS_LIMIT, yaw_max_rad_s: float = YAW_MAX_RAD_S_LIMIT,
                 phase_speed_gain: float = 0.0,
                 sensor_noise_scale: float = 1.0, open_frontier: bool = False,
                 combo_mix_frac: float = 0.0, combo_probe_frac: float = 0.08,
                 field_teach_recovery: bool = True,
                 recovery_tilt_max_deg: float | None = None,
                 recovery_position_max_m: float | None = None,
                 recovery_linvel_max_mps: float = 0.0,
                 recovery_angvel_max_rad_s: float | None = None,
                 recovery_joint_max_deg: float = 0.0,
                 domain_config: DomainRandomizationConfig | None = None,
                 attitude_profile: str = 'v5014', pitch_command_max_deg: float = 8.0214,
                 motor_guidance_weight: float = 0.0,
                 contact_slope_deg: float = 0.0,
                 contact_uphill_xy: tuple[float, float] = (1.0, 0.0),
                 observation_schema: str = OBS_SCHEMA_LEGACY,
                 tof_period_frames: int = 2,
                 flow_period_frames: int = 1,
                 tof_delay_frames: int = 1,
                 flow_delay_frames: int = 1,
                 tof_dropout_prob: float = 0.01,
                 flow_dropout_prob: float = 0.02,
                 gyro_bias_std_rad_s: float = 0.015,
                 terrain_step_max_m: float = 0.0,
                 paired_terrain_heights: bool = False,
                 unified_capability_focus: bool = False):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx
        if attitude_profile not in ('v5014', 'v5015'):
            raise ValueError(f'unknown attitude profile {attitude_profile!r}')
        if not (0.0 < pitch_command_max_deg <= 45.0):
            raise ValueError('pitch command maximum must be within (0, 45] degrees')
        if not np.isfinite(motor_guidance_weight) or not 0.0 <= motor_guidance_weight <= 0.10:
            raise ValueError('motor_guidance_weight must be in [0, 0.10]')
        self.motor_guidance_weight = float(motor_guidance_weight)
        self.observation_schema = observation_schema
        self.base_obs_fields = actor_observation_fields(observation_schema)
        self.base_obs_dim = len(self.base_obs_fields)
        self.obs_dim = self.base_obs_dim * HISTORY_LEN
        # These quantities are in *control* frames (20 ms), not physics
        # substeps. Check on construction rather than in an opaque JAX trace.
        for name, value in (('tof_period_frames', tof_period_frames),
                            ('flow_period_frames', flow_period_frames)):
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError(f'{name} must be an integer in [1,20]')
        for name, value in (('tof_delay_frames', tof_delay_frames),
                            ('flow_delay_frames', flow_delay_frames)):
            if type(value) is not int or not 0 <= value < HISTORY_LEN:
                raise ValueError(f'{name} must be an integer in [0,3]')
        for name, value in (('tof_dropout_prob', tof_dropout_prob),
                            ('flow_dropout_prob', flow_dropout_prob)):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be a probability in [0,1]')
        if not np.isfinite(gyro_bias_std_rad_s) or gyro_bias_std_rad_s < 0.0:
            raise ValueError('gyro_bias_std_rad_s must be nonnegative')
        self.tof_period_frames = tof_period_frames
        self.flow_period_frames = flow_period_frames
        self.tof_delay_frames = tof_delay_frames
        self.flow_delay_frames = flow_delay_frames
        self.tof_dropout_prob = float(tof_dropout_prob)
        self.flow_dropout_prob = float(flow_dropout_prob)
        self.gyro_bias_std_rad_s = float(gyro_bias_std_rad_s)
        self.sensor_contract = (sensor_observation_contract({
            'tof_period_frames': tof_period_frames,
            'flow_period_frames': flow_period_frames,
            'tof_delay_frames': tof_delay_frames,
            'flow_delay_frames': flow_delay_frames,
            'tof_dropout_prob': self.tof_dropout_prob,
            'flow_dropout_prob': self.flow_dropout_prob,
            'gyro_bias_std_rad_s': self.gyro_bias_std_rad_s,
            'sensor_noise_scale': float(sensor_noise_scale),
        }, schema=observation_schema, array_only=unified_capability_focus)
            if observation_schema in (OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN) else None)
        self.attitude_profile = attitude_profile
        self.pitch_limit_rad = (PITCH_CMD_MAX_RAD if attitude_profile == 'v5014'
                                else float(np.deg2rad(pitch_command_max_deg)))
        self.num_envs = int(num_envs)
        self.unified_capability_focus = bool(unified_capability_focus)
        self.focus_id = jnp.asarray(focus_indices(self.num_envs))
        self.max_episode_steps = int(max_episode_steps)
        self.robot_xml = str(robot_xml)
        # Experimental G3 only. Default zero uses the exact old XML path,
        # collision topology and JAX shapes. Never activate mid-checkpoint.
        if not np.isfinite(terrain_step_max_m) or not 0 <= terrain_step_max_m <= .08:
            raise ValueError('terrain_step_max_m must be in [0, 0.08] metres')
        self.terrain_step_max_m = float(terrain_step_max_m)
        self.paired_terrain_heights = bool(paired_terrain_heights)
        self.terrain_pair_seed = int(seed)
        self.mj_model = (
            mujoco.MjModel.from_xml_string(add_unified_terrain(self.robot_xml))
            if self.unified_capability_focus else
            mujoco.MjModel.from_xml_string(add_episode_platform(self.robot_xml))
            if self.terrain_step_max_m > 0 else
            mujoco.MjModel.from_xml_path(self.robot_xml)
        )
        self.unified_ground_mocap_id = -1
        if self.unified_capability_focus:
            if self.observation_schema != OBS_SCHEMA_TERRAIN or self.terrain_step_max_m <= 0:
                raise ValueError('unified terrain requires terrain_360 and real step geometry')
            ground_body=mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,
                                          'unified_ground')
            self.unified_ground_mocap_id=int(self.mj_model.body_mocapid[ground_body])
            if self.unified_ground_mocap_id < 0:
                raise RuntimeError('unified ground must have a real movable mocap body')
        self.terrain_step_mocap_id = -1
        self.terrain_step_geom_id = -1
        if self.terrain_step_max_m:
            platform_body = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'episode_terrain_platform')
            self.terrain_step_mocap_id = int(self.mj_model.body_mocapid[platform_body])
            self.terrain_step_geom_id = mujoco.mj_name2id(
                self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'episode_terrain_surface')
            if self.terrain_step_mocap_id < 0:
                raise RuntimeError('physical episode terrain requires a mocap body')
        self.mj_model.opt.timestep = SIM_DT
        # A static real plane and per-world real contact-domain models
        # share the same floor geom and forward-kinematics path. Both require
        # disabling MuJoCo's compiled identity-frame shortcut BEFORE put_model.
        self.domain_config = (domain_config or DomainRandomizationConfig()).normalized()
        self.contact_slope_deg = float(contact_slope_deg)
        if self.terrain_step_max_m and not self.unified_capability_focus and (
                self.contact_slope_deg != 0 or
                (self.domain_config.enabled and self.domain_config.terrain_radius > 0)):
            raise ValueError('legacy finite-step G3 probe cannot mix with inclined terrain')
        if self.unified_capability_focus and (
                self.contact_slope_deg != 0 or self.domain_config.terrain_radius>0):
            raise ValueError('unified dynamic slope is episode sampled; disable old model-level incline randomizers')
        self.contact_uphill_xy = tuple(np.asarray(contact_uphill_xy,dtype=float))
        orientation = contact_plane_quaternion(self.contact_slope_deg,
                                               self.contact_uphill_xy)
        real_floor_domain = (self.domain_config.enabled
                             and self.domain_config.terrain_mode == 'contact'
                             and self.domain_config.terrain_radius > 0)
        if self.contact_slope_deg > 0 and (real_floor_domain or
                (self.domain_config.enabled and self.domain_config.terrain_radius > 0)):
            raise ValueError('shared real contact slope cannot mix with randomized terrain')
        if self.contact_slope_deg > 0 or real_floor_domain:
            floor_id=mujoco.mj_name2id(
                self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
            if floor_id < 0 or self.mj_model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError('real contact slope requires a named plane geom floor')
            if self.contact_slope_deg > 0:
                self.mj_model.geom_quat[floor_id] = orientation
            # A compiled world geom at zero pos/identity quat usually has
            # geom_sameframe=BODY. Leaving that compiler shortcut in place
            # can IGNORE the rotated geom_quat completely, in CPU and MJX.
            # MuJoCo explicitly permits changing sameframe from nonzero->0.
            self.mj_model.geom_sameframe[floor_id] = 0
        # Performance hint from MJX docs: keep contacts explicit/simple. The XML
        # owns the complete robot physics, including actuated-joint armature and
        # Coulomb friction; thin thigh/shin rods are deliberately non-contacting.
        self.mjx_model_template = mjx.put_model(self.mj_model)
        self.ids = self._resolve_ids()
        self.mjx_model, self.model_in_axes, self.domain_info = build_batched_model(
            self.mjx_model_template, ids=self.ids, num_envs=self.num_envs, seed=int(seed),
            config=self.domain_config)
        gravity_unit = self.domain_info.get('gravity_unit_world')
        self.gravity_unit_world = (
            gravity_unit if gravity_unit is not None
            else jnp.tile(jnp.asarray([[0.0, 0.0, -1.0]], dtype=jnp.float32), (self.num_envs, 1))
        )
        # Same world-specific normal drives spawn clearance and actual MJX
        # plane collision; no independent 'virtual hill' for the trainer.
        sampled_normal = self.domain_info.get('contact_normal_world')
        if sampled_normal is None:
            a = math.radians(self.contact_slope_deg)
            u = np.asarray(self.contact_uphill_xy, dtype=float)
            u = u / np.linalg.norm(u)
            nominal = [-math.sin(a)*u[0], -math.sin(a)*u[1], math.cos(a)]
            sampled_normal = jnp.tile(jnp.asarray([nominal],dtype=jnp.float32),
                                      (self.num_envs,1))
        self.floor_normal_world = sampled_normal
        self.key0 = jax.random.PRNGKey(int(seed))

        # Per-instance versions of a few module-level profile defaults
        # (None = use the module constant unchanged). Each is read
        # via self.* at its use site below instead of the bare module name.
        self.push_max_force_n = float(push_max_force_n) if push_max_force_n is not None else PUSH_MAX_FORCE_N
        self.vx_hi_max = float(vx_hi_max)
        self.vx_lo_min = float(vx_lo_min)
        self.vy_max_mps = float(vy_max_mps)
        self.yaw_max_rad_s = float(yaw_max_rad_s)
        self.phase_speed_gain = float(phase_speed_gain)
        self.sensor_noise_scale = float(sensor_noise_scale)
        self.open_frontier = bool(open_frontier)
        self.combo_mix_frac = float(max(0.0, min(1.0, combo_mix_frac)))
        self.combo_probe_frac = float(max(0.0, combo_probe_frac))
        self.field_teach_recovery = bool(field_teach_recovery)
        # One resilience difficulty scalar owns recovery
        # progression, while these physical maxima define what "difficulty=1"
        # means for a particular experiment family. Defaults reproduce the
        # baseline exactly; linear/joint disorder are opt-in.
        self.recovery_tilt_max_rad = float(np.deg2rad(recovery_tilt_max_deg)) if recovery_tilt_max_deg is not None else POSE_TILT_MAX_RAD
        self.recovery_position_max_m = float(recovery_position_max_m) if recovery_position_max_m is not None else POSITION_NOISE_MAX_M
        self.recovery_linvel_max_mps = max(0.0, float(recovery_linvel_max_mps))
        self.recovery_angvel_max_rad_s = float(recovery_angvel_max_rad_s) if recovery_angvel_max_rad_s is not None else ANGVEL_NOISE_MAX_RAD_S
        self.recovery_joint_max_rad = max(0.0, float(np.deg2rad(recovery_joint_max_deg)))
        self.target_span = jnp.asarray(TARGET_SPAN, dtype=jnp.float32)
        self.home_target = jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
        self.act_qpos = jnp.asarray(self.ids['act_qpos'], dtype=jnp.int32)
        self.act_qvel = jnp.asarray(self.ids['act_qvel'], dtype=jnp.int32)
        # Resolve the nominal stall limit from the actual loaded XML instead
        # of duplicating six force ceilings in training code.
        limits = np.asarray(self.mj_model.actuator_forcerange, dtype=np.float32)
        if limits.shape != (ACTION_DIM, 2):
            raise ValueError('Expected exactly six force-limited robot actuators')
        # Hardware domain randomisation can vary actuator forcerange per
        # world. Use the actual model limits, not the nominal XML ceiling.
        self.servo_stall_torque_nm = jnp.max(
            jnp.abs(self.mjx_model.actuator_forcerange), axis=-1)
        # The six actuated hinge springs and two passive shin-flex springs.
        # MuJoCo exposes each hinge's spring reference via qpos_spring.
        spring_names = (*ACTUATED_JOINTS, 'r_shin_flex', 'l_shin_flex')
        joints = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in spring_names]
        if any(j < 0 for j in joints):
            raise ValueError('Missing a joint needed for spring-energy diagnostics')
        joint_pos = np.asarray(self.mj_model.jnt_qposadr)[joints].astype(np.int32)
        self.spring_qpos = jnp.asarray(joint_pos)
        self.spring_reference = jnp.asarray(
            np.asarray(self.mj_model.qpos_spring)[joint_pos], dtype=jnp.float32)
        self.spring_stiffness = jnp.asarray(
            np.asarray(self.mj_model.jnt_stiffness)[joints], dtype=jnp.float32)
        self.act_joint_range = jnp.asarray(self.ids['act_joint_range'], dtype=jnp.float32)
        self.shin_qpos = jnp.asarray(self.ids['shin_qpos'], dtype=jnp.int32)
        self.shin_qvel = jnp.asarray(self.ids['shin_qvel'], dtype=jnp.int32)
        self.env_index = jnp.arange(self.num_envs, dtype=jnp.float32)
        self.foot_bodies = jnp.asarray(self.ids['feet_bodies'], dtype=jnp.int32)
        self.bad_bodies = jnp.asarray(self.ids['bad_bodies'], dtype=jnp.int32)
        self.torso_id = int(self.ids['torso'])

        # JIT compile per env-count.  This is the compile boundary that makes the
        # trainer fast after the first warm-up update.
        # MJX kernels operate on one mjx.Data at a time.  Our environment
        # stores a leading env dimension on every Data field, so forward/step
        # must be explicitly vmapped over that leading dimension.  Calling
        # mjx.forward/step directly on a batched Data makes MJX interpret the
        # batch dimension as part of qpos/qvel and crashes with e.g.
        # "q has length 2048, expected 15".
        self.forward_batched = jax.vmap(mjx.forward, in_axes=(self.model_in_axes, 0))
        self.step_batched = jax.vmap(mjx.step, in_axes=(self.model_in_axes, 0))

        self.reset_jit = jax.jit(self._reset_impl)
        self.step_jit = jax.jit(self._step_impl)

    def _resolve_ids(self) -> dict[str, Any]:
        mujoco = self.mujoco
        def jid(n: str) -> int:
            i = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if i < 0: raise RuntimeError(f'Missing joint {n}')
            return int(i)
        def bid(n: str) -> int:
            i = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            if i < 0: raise RuntimeError(f'Missing body {n}')
            return int(i)
        def gid(n: str) -> int:
            i = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if i < 0: raise RuntimeError(f'Missing geom {n}')
            return int(i)
        def aid(n: str) -> int:
            i = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if i < 0: raise RuntimeError(f'Missing actuator {n}')
            return int(i)
        act_j = [jid(n) for n in ACTUATED_JOINTS]
        shin_j = [jid('r_shin_flex'), jid('l_shin_flex')]
        return {
            'torso': bid('torso'),
            'feet_bodies': [bid(n) for n in FOOT_BODIES],
            'bad_bodies': [bid(n) for n in BAD_BODIES],
            'act_qpos': np.array([self.mj_model.jnt_qposadr[j] for j in act_j], dtype=np.int32),
            'act_qvel': np.array([self.mj_model.jnt_dofadr[j] for j in act_j], dtype=np.int32),
            'act_joint_range': np.asarray(self.mj_model.jnt_range[act_j], dtype=np.float32),
            'shin_qpos': np.array([self.mj_model.jnt_qposadr[j] for j in shin_j], dtype=np.int32),
            'shin_qvel': np.array([self.mj_model.jnt_dofadr[j] for j in shin_j], dtype=np.int32),
            'floor_geom': gid('floor'),
            'foot_geoms': np.array([gid('r_foot_ball'), gid('l_foot_ball')], dtype=np.int32),
            'actuators': np.array([aid(n) for n in ACTUATED_JOINTS], dtype=np.int32),
        }

    def params_array(self, params: dict[str, float] | None = None):
        # Defaults put a fresh env at "all steady, zero difficulty
        # everywhere" -- the bootstrapped-from-nothing starting point.
        # Order here MUST match the P_* named indices above.
        jnp = self.jnp
        params = params or {}
        base = []
        for name, default in TASK_PARAM_FIELDS:
            if name == 'frontier_sensor_scale':
                default = self.sensor_noise_scale
            base.append(float(params.get(name, default)))
        node = np.asarray(params.get('field_node_anchors', np.zeros((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE), np.float32)), dtype=np.float32)
        edge = np.asarray(params.get('field_edge_anchors', np.zeros((FIELD_EDGE_ANCHORS, FIELD_EDGE_STRIDE), np.float32)), dtype=np.float32)
        node_pad = np.zeros((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE), np.float32)
        edge_pad = np.zeros((FIELD_EDGE_ANCHORS, FIELD_EDGE_STRIDE), np.float32)
        if node.ndim == 2:
            rr=min(FIELD_NODE_ANCHORS,node.shape[0]); cc=min(FIELD_NODE_STRIDE,node.shape[1]); node_pad[:rr,:cc]=node[:rr,:cc]
        if edge.ndim == 2:
            rr=min(FIELD_EDGE_ANCHORS,edge.shape[0]); cc=min(FIELD_EDGE_STRIDE,edge.shape[1]); edge_pad[:rr,:cc]=edge[:rr,:cc]
        packed = np.concatenate([
            np.asarray(base, np.float32),
            np.asarray([float(params.get('field_node_frac', 0.0))], np.float32),
            node_pad.reshape(-1),
            np.asarray([float(params.get('field_edge_frac', 0.0))], np.float32),
            edge_pad.reshape(-1),
        ])
        assert packed.shape[0] == TASK_PARAM_DIM, (packed.shape, TASK_PARAM_DIM)
        if self.terrain_step_mocap_id >= 0:
            # Explicit versioned task extension. Flat runs keep their old
            # exact shape and P-index semantics, even on this feature branch.
            terrain = np.asarray(
                [float(params.get(name, default)) for name, default in TERRAIN_TASK_FIELDS],
                dtype=np.float32)
            if not (np.isfinite(terrain).all() and
                    0.0 <= terrain[0] <= 1.0 and
                    0.0 <= terrain[1] <= self.terrain_step_max_m):
                raise ValueError('terrain exposure must be [0,1] and height must be [0, terrain_step_max_m]')
            packed = np.concatenate((packed, terrain))
            assert packed.shape[0] == TERRAIN_TASK_PARAM_DIM
            if self.unified_capability_focus:
                slope = np.asarray(
                    [float(params.get(name, default))
                     for name,default in UNIFIED_TERRAIN_FIELDS],dtype=np.float32)
                if (not np.isfinite(slope[:2]).all() or not 0.<=slope[0]<=1. or
                        not 0.<=slope[1]<=15.0):
                    raise ValueError('slope exposure must be [0,1], probe cap 15 degrees')
                fixed_pitch,fixed_roll=slope[2],slope[3]
                if np.isfinite(fixed_pitch) != np.isfinite(fixed_roll):
                    raise ValueError('explicit terrain pitch and roll must be provided together')
                if np.isfinite(fixed_pitch):
                    g=np.hypot(np.tan(np.deg2rad(fixed_pitch)),
                               np.tan(np.deg2rad(fixed_roll)))
                    if not np.isfinite(g) or np.rad2deg(np.arctan(g))>15.0:
                        raise ValueError('physical terrain total inclination exceeds 15-degree exam cap')
                if np.isfinite(slope[4]) and not 0.<=slope[4]<=self.terrain_step_max_m:
                    raise ValueError('fixed step height outside compiled physical range')
                if np.isfinite(slope[5]) and abs(slope[5])>45.0:
                    raise ValueError('fixed edge yaw outside +/-45-degree exam range')
                if not np.isfinite(slope[6]) or not 0. <= slope[6] <= 1.:
                    raise ValueError('specialist_mix must be between zero and one')
                if not np.all(np.isfinite(slope[~np.isnan(slope)])):
                    raise ValueError('invalid physical terrain override')
                packed=np.concatenate((packed,slope))
                assert packed.shape[0] == UNIFIED_TASK_PARAM_DIM
        return jnp.asarray(packed, dtype=jnp.float32)

    def reset(self, params: dict[str, float] | None = None):
        p = self.params_array(params)
        state = self.reset_jit(self.key0, p)
        return state, self.obs(state)

    def step(self, state: EnvState, action, params: dict[str, float] | Any) -> StepOut:
        p = self.params_array(params) if isinstance(params, dict) or params is None else params
        return self.step_jit(state, action, p)

    def obs(self, state: EnvState):
        return state.hist.reshape((self.num_envs, self.obs_dim))

    # ---------------- JAX-only implementation below ----------------

    def _split_batch(self, keys, n: int):
        jax = self.jax
        split = jax.vmap(lambda k: jax.random.split(k, n))(keys)
        return [split[:, i, :] for i in range(n)]

    def _make_batched_data(self):
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        def one(_):
            # Data topology is created from the unbatched template. Batched
            # model leaves are supplied separately through vmap(in_axes).
            return mjx.make_data(self.mjx_model_template)
        return jax.vmap(one)(jnp.arange(self.num_envs))

    def _reset_impl(self, key, params):
        """Reset the full vector batch and decorrelate timeout phase.

        Episode content is delegated to _reset_with_keys so initial and later
        resets follow the same sampling path. ``step_i`` is then staggered
        across the batch to prevent successful episodes from timing out as one
        synchronized cohort. True episode age remains ``ep_len`` and starts at
        zero for reward shaping.
        """
        jax, jnp = self.jax, self.jnp
        env_key, stagger_key = jax.random.split(key)
        state = self._reset_with_keys(jax.random.split(env_key, self.num_envs), params)
        step_i = jax.random.randint(stagger_key, (self.num_envs,), 0, self.max_episode_steps, dtype=jnp.int32)
        return state._replace(step_i=step_i)

    def _step_impl(self, state: EnvState, action, params) -> StepOut:
        """Advance a control frame and publish a truthful PPO transition.

        Physics first; lesson timing second; then score the command visible
        when the actor chose this action, record the next observation, and
        reset only worlds whose episodes ended.
        """
        jnp = self.jnp

        # 1. The robot moves in the physical world.
        physics = self._simulate_physics(state, action, params)
        data = physics.data
        action = physics.action
        ctrl = physics.delivered_ctrl
        raw_target = physics.raw_target

        step_i = state.step_i + 1
        phase = (state.phase + CONTROL_DT * state.task[:, T.phase_rate] * 2.0 * jnp.pi) % (2.0 * jnp.pi)

        # 2. The teacher advances the NEXT request; it cannot rewrite the
        # instruction the actor already saw when choosing this action.
        lesson = self._advance_commands(state, params, physics.resample_keys)
        task = lesson.task
        task_for_state = lesson.stored_task
        ramp_frac = lesson.ramp_frac
        transition_active = lesson.transition_active
        cmd_resampled = lesson.cmd_resampled
        physics_diverged = physics.diverged

        # 3. Measure the physical result, then reward the PREVIOUS command.
        # The ramped command is for the next observation, never this payment.
        reward_task = state.task
        metrics = self._metrics(data, reward_task, action, state.prev_action, phase, state.prev_foot_pos)
        # Gate freezing against previous activity, not this one stance frame.
        stepping_activity_so_far = state.ep_step_evidence_sum / jnp.maximum(1.0, state.ep_len)
        # ramp_frac tracks command-change difficulty. Real episode age is
        # ep_len; step_i is deliberately staggered for timeout diversity.
        metrics = {**metrics, 'age_s': (state.ep_len + 1.0) * CONTROL_DT,
                  'stepping_activity_so_far': stepping_activity_so_far, 'ramp_frac': ramp_frac}
        reward, terms = training_reward(
            metrics, reward_task[:, T.regime],
            attitude_profile=self.attitude_profile, xp=jnp,
            motor_guidance_weight=self.motor_guidance_weight,
            motor_guidance_per_world=(control_effort_weights(jnp,
                    jnp.where(state.specialist_active,self.focus_id,-1))
                if self.unified_capability_focus else None))
        timeout = step_i >= self.max_episode_steps
        # Even after rollback, a diverged world must terminate and respawn.
        terminal_fail = jnp.logical_or(metrics['terminal'] > 0.5, physics_diverged)
        done = jnp.logical_or(terminal_fail, timeout)
        # Penalise premature falls, not successful timeouts. Use true age
        # (ep_len), never the intentionally staggered timeout clock (step_i).
        episode_age_steps = state.ep_len + 1.0
        reward, fall_payment = apply_early_fall_penalty(
            reward, terminal_fail, episode_age_steps, self.max_episode_steps, xp=jnp)

        ep_return = state.ep_return + reward
        ep_len = state.ep_len + 1.0
        ep_vx_sum = state.ep_vx_sum + metrics['vx']
        ep_cmd_vx_sum = state.ep_cmd_vx_sum + metrics['cmd_vx']
        ep_bad_sum = state.ep_bad_sum + metrics['bad_contact']
        ep_support_sum = state.ep_support_sum + metrics['support']
        ep_step_evidence_sum = state.ep_step_evidence_sum + metrics['step_evidence']
        # Teacher-only episode quality: using one good final frame would let
        # the robot coast into a high velocity for a moment and 'pass'.
        vx_e=metrics['vx']-metrics['cmd_vx']
        vy_e=metrics['vy']-metrics['cmd_vy']
        yaw_e=metrics['yaw_rate']-metrics['cmd_yaw']
        tracking_frame=jnp.exp(-(35.*vx_e*vx_e+28.*vy_e*vy_e+2.0*yaw_e*yaw_e))
        ep_command_quality_sum = state.ep_command_quality_sum + tracking_frame * (
            .50 + .50*jnp.clip(metrics['support'],0.,1.)) * (
            1. - jnp.clip(metrics['bad_contact'],0.,1.))
        # G4 examiner/teacher: evidence comes exclusively from actual
        # root travel AND far-side MJX foot contacts. No privileged terrain
        # height or approach position is ever passed to the actor.
        terrain_streak, terrain_crossed, terrain_active = self._terrain_crossing(
            state, data, metrics)
        terrain_cross_new = terrain_crossed & ~state.terrain_crossed
        terrain_up = terrain_active & ((self.env_index % 2) == 0)
        terrain_down = terrain_active & ((self.env_index % 2) == 1)
        # Physical step severity: displacement between the two actual
        # surface collider centres projected onto their shared normal.
        # Never infer it from the requested teacher height or flat fallback.
        if self.unified_capability_focus:
            base_id=int(self.ids['floor_geom'])
            upper_id=int(self.terrain_step_geom_id)
            support_n=data.geom_xmat[:,base_id,:].reshape(
                (self.num_envs,3,3))[:,:,2]
            physical_height=jnp.abs(jnp.sum((
                data.geom_xpos[:,upper_id,:]-data.geom_xpos[:,base_id,:])
                * support_n,axis=1))
            physical_height=jnp.where(terrain_active,physical_height,0.)
            ratio=physical_height/jnp.maximum(params[P.terrain_height_m],.0001)
            step_near=terrain_active & (ratio>=.78) & (ratio<=1.05)
            step_easy=terrain_active & (ratio>=.36) & (ratio<.68)
        else:
            physical_height=jnp.zeros_like(reward)
            step_near=terrain_active & False
            step_easy=terrain_active & False
        ascent_payment = jnp.zeros_like(reward)
        root_peak = state.terrain_ascent_root_peak
        foot_peak = state.terrain_ascent_foot_peak
        if self.observation_schema == OBS_SCHEMA_TERRAIN:
            # A single bounded foothold/progress lesson for ASCENT, before
            # the very sparse far-side completion. Both high-water marks are
            # reset at the next episode birth; no pay for rocking backwards.
            ascent_payment, root_peak, foot_peak = self._terrain_ascent_guidance(
                state, data, metrics, terrain_up)
            reward = reward + ascent_payment
            ep_return = ep_return + ascent_payment
            # Completion still requires real, sustained far-side MJX support.
            reward = reward + 10.0 * terrain_cross_new.astype(jnp.float32)
            ep_return = ep_return + 10.0 * terrain_cross_new.astype(jnp.float32)

        # 4. Publish the NEXT actor observation; preserve the pre-reset
        # observation separately for PPO bootstrap.
        sensors = (self._advance_sensor_state(
            state.sensor_state, data, task, phase, physics.next_keys,
            state.ep_len + 1) if self.observation_schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN)
            else None)
        base = self._base_obs(data, task, action, phase, ctrl, sensors)
        hist = jnp.concatenate([state.hist[:, 1:, :], base[:, None, :]], axis=1)
        bootstrap_obs = hist.reshape((self.num_envs, self.obs_dim))
        # carryover_qvel is this step's own (already divergence-safe)
        # qvel -- see the docstring on _reset_with_keys.
        reset_state = self._reset_with_keys(physics.reset_keys, params, carryover_qvel=data.qvel)

        def choose(reset_val, step_val):
            if hasattr(step_val, 'shape') and len(step_val.shape) > 0 and step_val.shape[0] == self.num_envs:
                mask = done.reshape((self.num_envs,) + (1,) * (len(step_val.shape) - 1))
                return jnp.where(mask, reset_val, step_val)
            return step_val

        stepped = EnvState(data, hist, physics.next_keys, task_for_state, state.specialist_active, action, data.xpos[:, self.foot_bodies, :], raw_target, phase, step_i,
                           ep_return, ep_len, ep_vx_sum, ep_cmd_vx_sum, ep_bad_sum, ep_support_sum, ep_step_evidence_sum, ep_command_quality_sum,
                           terrain_streak,terrain_crossed,root_peak,foot_peak,
                           lesson.target_next, lesson.ramp_from_next, lesson.ramp_frac_next, lesson.ramp_speed_next, lesson.hold_left_next,
                           physics.push_force, physics.push_steps_left, sensors)
        new_state = self.jax.tree_util.tree_map(choose, reset_state, stepped)
        obs = self.obs(new_state)
        denom = jnp.maximum(ep_len, 1.0)
        # 5. Report independent facts, splitting ramping from steady holding.
        ramping = transition_active.astype(jnp.float32)
        resampled = cmd_resampled.astype(jnp.float32)
        steady = 1.0 - ramping
        vx_err = jnp.abs(metrics['vx'] - metrics['cmd_vx'])
        vy_err = jnp.abs(metrics['vy'] - metrics['cmd_vy'])
        yaw_err = jnp.abs(metrics['yaw_rate'] - metrics['cmd_yaw'])
        # Normalized L1 radius at the actual episode's requested command.
        # This is evidence context, not an observation/reward. For COMMAND
        # specialists the target remains fixed for the full episode.
        fx=jnp.where(metrics['cmd_vx']>=0.,
                     jnp.maximum(.01,params[P.frontier_vx_hi]),
                     jnp.maximum(.01,jnp.abs(params[P.frontier_vx_lo])))
        unit_x=jnp.abs(metrics['cmd_vx'])/fx
        unit_y=jnp.abs(metrics['cmd_vy'])/jnp.maximum(.05,params[P.frontier_vy])
        unit_z=jnp.abs(metrics['cmd_yaw'])/jnp.maximum(.10,params[P.frontier_yaw])
        unit_radius=unit_x+unit_y+unit_z
        if self.unified_capability_focus:
            # Read the ACTUAL tilted COLLISION geom, not requested lesson
            # severity, world gravity or a guessed angle. This is teacher
            # evidence and never enters the neural observation.
            Rfloor=data.geom_xmat[:,int(self.ids['floor_geom']),:].reshape(
                (self.num_envs,3,3))
            floor_normal=Rfloor[:,:,2]
            gx=-floor_normal[:,0]/jnp.maximum(.01,floor_normal[:,2])
            gy=-floor_normal[:,1]/jnp.maximum(.01,floor_normal[:,2])
            physical_slope=jnp.rad2deg(jnp.arctan(
                jnp.sqrt(gx*gx+gy*gy)))
            slope_role=state.specialist_active & (self.focus_id==SLOPE)
        else:
            gx=jnp.zeros_like(unit_x);gy=jnp.zeros_like(unit_x)
            physical_slope=jnp.zeros_like(unit_x)
            slope_role=jnp.zeros((self.num_envs,),dtype=jnp.bool_)
        # Smaller-than-seed incline lessons remain meaningful retreat
        # territory; do not let a stale .25-degree recognition threshold
        # erase their completed-episode evidence.
        has_slope=slope_role & (physical_slope>=.10)
        # Severity bands use the REAL collision geom, never a teacher's
        # claimed task difficulty. Easy failures and near-target successes
        # are different lessons. These masks only affect host-side evidence.
        slope_fraction=physical_slope/jnp.maximum(params[P.slope_max_deg],.1)
        slope_near=has_slope & (slope_fraction>=.78) & (slope_fraction<=1.05)
        slope_easy=has_slope & (slope_fraction>=.36) & (slope_fraction<.68)
        along=jnp.abs(gx)>=jnp.abs(gy)
        slope_directions=(
            ('uphill',has_slope & along & (gx>=0.)),
            ('downhill',has_slope & along & (gx<0.)),
            ('cross_left',has_slope & ~along & (gy>=0.)),
            ('cross_right',has_slope & ~along & (gy<0.)))
        slope_pass=(
            timeout & ~terminal_fail & (ep_len>=120.) &
            (ep_command_quality_sum/jnp.maximum(ep_len,1.)>=.60) &
            (ep_support_sum/jnp.maximum(ep_len,1.)>.10) &
            (ep_bad_sum/jnp.maximum(ep_len,1.)<.35) &
            (ep_vx_sum/jnp.maximum(ep_len,1.)>.025))
        out_metrics = {
            **{k: v for k, v in metrics.items() if k not in REWARD_ONLY_METRICS},
            # Same physical slip value used by the gait reward; publish a
            # read-only diagnostic for the independent frozen examiner.
            # It is NOT an actor input, reward change or curriculum result.
            'exam_stance_slip_proxy':metrics['stance_slip'],
            'physical_slope_deg':physical_slope,
            'physical_slope_gx':gx,
            'physical_slope_gy':gy,
            'slope_specialist_active':has_slope.astype(jnp.float32),
            **{f'slope_episode_attempt_{name}':(done & active).astype(jnp.float32)
               for name,active in slope_directions},
            **{f'slope_episode_success_{name}':(slope_pass & active).astype(jnp.float32)
               for name,active in slope_directions},
            **{f'slope_episode_{band}_{outcome}_{name}':
               (mask & active & condition).astype(jnp.float32)
               for name,active in slope_directions
               for band,mask in (('near',slope_near),('easy',slope_easy))
               for outcome,condition in (('attempt',done),
                                         ('success',slope_pass))},
            'command_probe_radius':unit_radius,
            'command_probe_weight_x':unit_x/jnp.maximum(1e-5,unit_radius),
            'command_probe_weight_y':unit_y/jnp.maximum(1e-5,unit_radius),
            'command_probe_weight_z':unit_z/jnp.maximum(1e-5,unit_radius),
            'transition_frac': ramping,
            'cmd_resampled': resampled,
            # One maneuver verdict per source-to-target transition. These
            # diagnostics never enter actor observations or reward.
            'target_cmd_vx': state.cmd_target[:, 0],
            'target_cmd_vy': state.cmd_target[:, 1],
            'target_cmd_yaw': state.cmd_target[:, 2],
            'transition_ramp_speed': state.cmd_ramp_speed,
            'transition_vx_err': vx_err * ramping, 'transition_vy_err': vy_err * ramping,
            'transition_yaw_err': yaw_err * ramping,
            'steady_vx_err': vx_err * steady, 'steady_vy_err': vy_err * steady,
            'steady_yaw_err': yaw_err * steady,
            **{f'reward_{k}': v for k, v in terms.items()},
            'reward_weighted_fall_penalty': fall_payment,
            'episode_done': done.astype(jnp.float32),
            'specialist_focus': jnp.where(state.specialist_active,
                                           self.focus_id,-1).astype(jnp.float32),
            'episode_command_quality': ep_command_quality_sum/jnp.maximum(ep_len,1.),
            'episode_length': ep_len,
            'episode_return': ep_return,
            'episode_fell': terminal_fail.astype(jnp.float32),
            'episode_timeout': timeout.astype(jnp.float32),
            'terrain_episode_attempt': (done & terrain_active).astype(jnp.float32),
            'terrain_episode_success': (timeout & ~terminal_fail & terrain_active & terrain_crossed).astype(jnp.float32),
            'terrain_cross_new':terrain_cross_new.astype(jnp.float32),
            'physical_step_height_m': physical_height,
            **{f'terrain_episode_{band}_{outcome}_{direction}':
                (mask & cohort & condition).astype(jnp.float32)
                for direction,cohort in (('up',terrain_up),('down',terrain_down))
                for band,mask in (('near',step_near),('easy',step_easy))
                for outcome,condition in (
                    ('attempt',done),
                    ('success',timeout & ~terminal_fail & terrain_crossed))},
            'terrain_ascent_guidance':ascent_payment,
            # Teacher-only direction: actor observations remain unchanged.
            'terrain_active_up': terrain_up.astype(jnp.float32),
            'terrain_active_down': terrain_down.astype(jnp.float32),
            'terrain_episode_attempt_up': (done & terrain_up).astype(jnp.float32),
            'terrain_episode_attempt_down': (done & terrain_down).astype(jnp.float32),
            'terrain_episode_success_up': (timeout & ~terminal_fail & terrain_up & terrain_crossed).astype(jnp.float32),
            'terrain_episode_success_down': (timeout & ~terminal_fail & terrain_down & terrain_crossed).astype(jnp.float32),
            'terrain_episode_fall_up': (terminal_fail & terrain_up).astype(jnp.float32),
            'terrain_episode_fall_down': (terminal_fail & terrain_down).astype(jnp.float32),
            # Direction + approach speed cohorts: teacher-only counters, not
            # actor observations. Absolute task vx is the commanded speed.
            **{f'terrain_episode_{outcome}_up_{label}': jnp.logical_and(
                condition,terrain_up & (
                    (metrics['cmd_vx']>=lo) & (metrics['cmd_vx']<hi))
                ).astype(jnp.float32)
               for label,lo,hi in (
                   ('slow',.10,.16),('medium',.16,.23),('fast',.23,.31))
               for outcome,condition in (
                   ('attempt',(done)),
                   ('success',(timeout & ~terminal_fail & terrain_crossed)),
                   ('fall',(terminal_fail)))},
            'episode_vx_mean': ep_vx_sum / denom,
            'episode_cmd_vx_mean': ep_cmd_vx_sum / denom,
            'episode_bad_contact': ep_bad_sum / denom,
            'episode_support': ep_support_sum / denom,
        }
        return StepOut(
            new_state, jnp.nan_to_num(obs), jnp.nan_to_num(bootstrap_obs), jnp.nan_to_num(reward), done.astype(jnp.float32),
            # Defense in depth: the rollback above is the real fix for
            # physics divergence specifically, this is just insurance in
            # case some other numeric path ever goes non-finite -- cheap,
            # and guarantees NaN can never reach a PPO batch/gradient from
            # this function.
            {k: jnp.nan_to_num(v) for k, v in out_metrics.items()},
        )

    def _simulate_physics(self, state: EnvState, action, params) -> PhysicalFrame:
        """Deliver the last servo target, apply pushes, and advance MJX.

        Returns post-step finite physics, event-specific RNG keys and the next
        raw servo command. Diverged worlds are rolled back *before* metrics
        read them, then marked for a real reset by _step_impl.
        """
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        action = jnp.clip(action, -1.0, 1.0)
        raw_target = self.home_target + action * self.target_span
        raw_target = jnp.clip(raw_target, -self.target_span, self.target_span)
        # One control-frame command-delivery delay: servos receive the previous
        # raw target, matching the current hardware doctrine.
        ctrl = state.prev_raw_target
        # The requested position target is delivered directly. MuJoCo's geared
        # servo dynamics, not an artificial software slew limiter, determine
        # how fast the joint can move.
        data = state.data.replace(ctrl=state.data.ctrl.at[:, :ACTION_DIM].set(ctrl))

        # Push perturbations (RESILIENCE), sized to be a recoverable nudge
        # rather than an instant launch. Applied to `data` BEFORE the
        # physics substep loop -- applying a push after stepping physics
        # and clearing it before the next step would give it zero actual
        # effect, so this ordering is deliberate. Uses the PER-EPISODE
        # resilience level (state.task's T.resilience column, set in
        # _sample_task) instead of the raw
        # global curriculum dial directly -- this is what makes
        # REGULAR_RESILIENCE_FRAC actually take effect on the physics, not
        # just on what the policy is told.
        resilience_frac = jnp.clip(state.task[:, T.resilience], 0.0, 1.0)
        resilience_push_n = resilience_frac * self.push_max_force_n
        # Split the per-environment RNG once, then dedicate each key to one
        # random purpose. Re-splitting the same unadvanced JAX key can reuse
        # identical key material. Independent splits deliberately
        # reproduces that older path for regression/A-B work only.
        push_rng, angle_key, mag_key, dur_key, next_keys, reset_keys, resample_keys = self._split_batch(state.rng, 7)
        push_roll = jax.vmap(lambda k: jax.random.uniform(k, ()))(push_rng)
        push_active = state.push_steps_left > 0
        # Capability-cloud/atlas probe: one controlled horizontal shove shortly
        # after birth. Magnitude and duration are explicit physical quantities;
        # direction remains random across envs so one cloud point estimates a
        # distribution of nuisance directions instead of adding push-angle as
        # yet another map dimension. Negative probe_push_n disables this path.
        probe_push_n = params[P.probe_push_n]
        probe_push_steps = jnp.maximum(1.0, params[P.probe_push_steps])
        probe_enabled = probe_push_n >= 0.0
        probe_start = jnp.logical_and(jnp.logical_not(push_active),
                                      jnp.logical_and(state.ep_len >= 5.0, state.ep_len < 6.0))
        random_start = jnp.logical_and(jnp.logical_not(push_active),
                                       push_roll < (PUSH_PROB_PER_STEP * resilience_frac))
        should_start = jnp.where(probe_enabled, probe_start, random_start)
        random_push_angle = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.0, maxval=2.0 * jnp.pi))(angle_key)
        probe_push_angle = (2.0 * jnp.pi / 8.0) * jnp.mod(self.env_index.astype(jnp.int32), 8).astype(jnp.float32)
        push_angle = jnp.where(probe_enabled, probe_push_angle, random_push_angle)
        random_mag = jax.vmap(lambda k, mx: jax.random.uniform(k, (), minval=0.0, maxval=mx))(mag_key, resilience_push_n)
        push_mag = jnp.where(probe_enabled, probe_push_n, random_mag)
        random_dur = jnp.round(jax.vmap(lambda k: jax.random.uniform(
            k, (), minval=PUSH_MIN_DURATION_STEPS, maxval=PUSH_MAX_DURATION_STEPS))(dur_key))
        push_dur = jnp.where(probe_enabled, probe_push_steps, random_dur)
        new_push_force = jnp.stack([push_mag * jnp.cos(push_angle), push_mag * jnp.sin(push_angle),
                                     jnp.zeros_like(push_mag)], axis=1)
        push_force = jnp.where(should_start[:, None], new_push_force,
                                jnp.where(push_active[:, None], state.push_force, 0.0))
        push_steps_left = jnp.where(should_start, push_dur, jnp.maximum(0.0, state.push_steps_left.astype(jnp.float32) - 1.0)).astype(jnp.int32)
        xfrc = jnp.zeros_like(data.xfrc_applied).at[:, self.torso_id, 0:3].set(push_force)
        data = data.replace(xfrc_applied=xfrc)

        def substep(_, d):
            return self.step_batched(self.mjx_model, d)
        data = jax.lax.fori_loop(0, CONTROL_SUBSTEPS, substep, data)

        # Non-finite MJX qpos/qvel can bypass every fall comparison (NaN
        # compares false), poison the shared PPO gradient and survive forever.
        # Restore that world's last finite state *before* measuring anything;
        # _step_impl will then terminate it and perform a normal reset.
        physics_diverged = jnp.logical_or(
            jnp.any(~jnp.isfinite(data.qpos), axis=1),
            jnp.any(~jnp.isfinite(data.qvel), axis=1),
        )
        def _rollback_diverged(pre, post):
            if hasattr(post, 'shape') and len(post.shape) > 0 and post.shape[0] == self.num_envs:
                mask = physics_diverged.reshape((self.num_envs,) + (1,) * (len(post.shape) - 1))
                return jnp.where(mask, pre, post)
            return post
        data = jax.tree_util.tree_map(_rollback_diverged, state.data, data)
        return PhysicalFrame(
            data=data, action=action, raw_target=raw_target,
            delivered_ctrl=ctrl, push_force=push_force,
            push_steps_left=push_steps_left, diverged=physics_diverged,
            next_keys=next_keys, reset_keys=reset_keys,
            resample_keys=resample_keys,
        )

    def _quat_mul(self, q1, q2):
        """Hamilton product, (w,x,y,z) convention (MuJoCo's qpos order).
        Used once: composing a small initial roll and pitch tilt for
        RESILIENCE spawns. Exact composition, not a small-angle linear
        approximation -- POSE_TILT_MAX_RAD (~20deg) isn't small enough for
        that to be a safe shortcut."""
        jnp = self.jnp
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return jnp.stack([w, x, y, z], axis=-1)

    def _birth_physics(self, data, task, params, keys, carryover_qvel):
        """Create the physical initial pose/velocity for one sampled lesson.

        RESILIENCE samples a larger disturbance than ordinary practice;
        held-out probes use explicit magnitudes and stratified directions.
        All key streams and the free-joint velocity birth remain unchanged.
        This method does not sample the task or construct actor observations.
        """
        jax, jnp = self.jax, self.jnp
        (noise_keys, vnoise_keys, tilt_keys, pos_keys, angvel_keys,
         linvel_keys, joint_keys, carry_keys) = keys
        qpos, qvel = data.qpos, data.qvel

        regime = task[:, T.regime]
        is_resilience = regime > 1.5
        # Birth difficulty is episode-local. The governor owns
        # the ceiling; field-steered resilience births may intentionally train
        # below it at the locally useful severity for this command voxel.
        resilience_d = task[:, T.birth_resilience_d]

        # Initial position offset (xy + a bit of extra height wobble
        # on top of posture's own T.height). RESILIENCE gets the full
        # difficulty-scaled magnitude; STEADY/TRANSITION borrow a small
        # fixed fraction of it (see STEADY_RESILIENCE_BORROW_FRAC).
        pos_noise_full = self._lerp(0.0, self.recovery_position_max_m, resilience_d)
        pos_noise_mag = jnp.where(is_resilience, pos_noise_full, pos_noise_full * STEADY_RESILIENCE_BORROW_FRAC)
        pos_offset = jax.vmap(lambda k, m: jax.random.uniform(k, (3,), minval=-m, maxval=m))(pos_keys, pos_noise_mag)
        root_xy=pos_offset[:,:2]
        # Physical birth height is derived from the SAME per-world plane
        # normal that the collision engine uses (plane passes through zero).
        normal=self.floor_normal_world
        floor_z=-jnp.sum(root_xy*normal[:,:2],axis=1)/normal[:,2]
        qpos = qpos.at[:, 0:3].set(jnp.stack([
            root_xy[:,0],root_xy[:,1],
            NEUTRAL_ROOT_Z + task[:, T.height] + pos_offset[:, 2] + floor_z,
        ], axis=1))

        # Initial pose tilt (roll/pitch away from upright). Same
        # borrow-fraction coupling as position noise above. Composed via a
        # real quaternion product (_quat_mul), not a small-angle
        # approximation -- the ceiling (~20deg) is real enough that a linear
        # shortcut would be visibly wrong at the difficult end.
        tilt_full = self._lerp(0.0, self.recovery_tilt_max_rad, resilience_d)
        tilt_mag = jnp.where(is_resilience, tilt_full, tilt_full * STEADY_RESILIENCE_BORROW_FRAC)
        probe_tilt = params[P.probe_tilt_rad]
        tilt_mag = jnp.where(probe_tilt >= 0.0, probe_tilt, tilt_mag)
        def _sample_tilt(k, mag):
            kr, kp = jax.random.split(k)
            roll = jax.random.uniform(kr, (), minval=-mag, maxval=mag)
            pitch = jax.random.uniform(kp, (), minval=-mag, maxval=mag)
            return roll, pitch
        roll0, pitch0 = jax.vmap(_sample_tilt)(tilt_keys, tilt_mag)
        # Held-out recovery probes integrate nuisance direction without adding
        # another map axis: 8 evenly represented azimuth bins across the vector
        # batch. Training (probe_tilt < 0) keeps its ordinary random sampling.
        probe_bin = jnp.mod(self.env_index.astype(jnp.int32), 8)
        probe_angle = (2.0 * jnp.pi / 8.0) * probe_bin.astype(jnp.float32)
        roll0 = jnp.where(probe_tilt >= 0.0, probe_tilt * jnp.cos(probe_angle), roll0)
        pitch0 = jnp.where(probe_tilt >= 0.0, probe_tilt * jnp.sin(probe_angle), pitch0)
        zeros1 = jnp.zeros_like(roll0)
        q_roll = jnp.stack([jnp.cos(roll0 / 2), jnp.sin(roll0 / 2), zeros1, zeros1], axis=-1)
        q_pitch = jnp.stack([jnp.cos(pitch0 / 2), zeros1, jnp.sin(pitch0 / 2), zeros1], axis=-1)
        qpos = qpos.at[:, 3:7].set(self._quat_mul(q_pitch, q_roll))

        qnoise = jax.vmap(lambda k: jax.random.normal(k, (ACTION_DIM,), dtype=jnp.float32))(noise_keys) * RESET_JOINT_NOISE
        joint_full = self.recovery_joint_max_rad * resilience_d
        joint_mag = jnp.where(is_resilience, joint_full, joint_full * STEADY_RESILIENCE_BORROW_FRAC)
        joint_extra = jax.vmap(lambda k, m: jax.random.uniform(k, (ACTION_DIM,), minval=-m, maxval=m))(joint_keys, joint_mag)
        qnoise = qnoise + joint_extra
        probe_joint = params[P.probe_joint_rad]
        probe_joint_noise = jax.vmap(lambda k: jax.random.uniform(k, (ACTION_DIM,), minval=-1.0, maxval=1.0))(joint_keys) * jnp.maximum(probe_joint, 0.0)
        qnoise = jnp.where(probe_joint >= 0.0, probe_joint_noise, qnoise)
        vnoise = jax.vmap(lambda k: jax.random.normal(k, (self.mj_model.nv,), dtype=jnp.float32))(vnoise_keys) * RESET_QVEL_NOISE
        qj = self.home_target + qnoise
        qj = jnp.clip(qj, self.act_joint_range[:, 0] + 1e-3, self.act_joint_range[:, 1] - 1e-3)
        qpos = qpos.at[:, self.act_qpos].set(qj)
        qvel = vnoise
        # Dynamic-birth doctrine: every regime starts already moving at (or
        # near) its own commanded vx/vy/yaw-rate state, not first becoming
        # a statue -- STEADY/TRANSITION/RESILIENCE all ask "hold roughly
        # this state" rather than "stand still". Free-joint qvel layout:
        # [0:3] linear (x,y,z), [3:6] angular (x,y,z) -- yaw-rate is the
        # angular-z term.
        qvel = qvel.at[:, 0].set(qvel[:, 0] + RESET_FORWARD_VELOCITY_BLEND * task[:, T.vx])
        qvel = qvel.at[:, 1].set(qvel[:, 1] + RESET_FORWARD_VELOCITY_BLEND * task[:, T.vy])
        qvel = qvel.at[:, 5].set(qvel[:, 5] + RESET_FORWARD_VELOCITY_BLEND * task[:, T.yaw])
        probe_lin = params[P.probe_linvel_mps]
        lin_full = self.recovery_linvel_max_mps * resilience_d
        lin_mag = jnp.where(is_resilience, lin_full, lin_full * STEADY_RESILIENCE_BORROW_FRAC)
        train_lin = jax.vmap(lambda k, m: jax.random.uniform(k, (3,), minval=-m, maxval=m))(linvel_keys, lin_mag)
        train_lin = train_lin * jnp.asarray([1.0, 1.0, 0.35], dtype=jnp.float32)
        # Held-out probes use exact horizontal magnitude with 8 stratified
        # directions. This makes worst-direction recovery measurable while
        # keeping direction a nuisance variable rather than a cloud dimension.
        probe_lin_vec = jnp.stack([jnp.cos(probe_angle), jnp.sin(probe_angle), jnp.zeros_like(probe_angle)], axis=1) * jnp.maximum(probe_lin, 0.0)
        lin_kick = jnp.where(probe_lin >= 0.0, probe_lin_vec, train_lin)
        qvel = qvel.at[:, 0:3].set(qvel[:, 0:3] + lin_kick)
        # RESILIENCE's initial angular-velocity kick -- a disturbance to
        # catch, added ON TOP of the commanded yaw-rate blend just above,
        # not instead of it. Same borrow-fraction coupling as tilt/position.
        angvel_full = self._lerp(0.0, self.recovery_angvel_max_rad_s, resilience_d)
        angvel_mag = jnp.where(is_resilience, angvel_full, angvel_full * STEADY_RESILIENCE_BORROW_FRAC)
        probe_ang = params[P.probe_angvel_rad_s]
        train_angvel = jax.vmap(lambda k, m: jax.random.uniform(k, (3,), minval=-m, maxval=m))(angvel_keys, angvel_mag)
        yaw_sign = jnp.where((probe_bin % 2) == 0, 0.5, -0.5).astype(jnp.float32)
        probe_ang_dir = jnp.stack([jnp.cos(probe_angle), jnp.sin(probe_angle), yaw_sign], axis=1)
        probe_ang_dir = probe_ang_dir / jnp.maximum(jnp.linalg.norm(probe_ang_dir, axis=1, keepdims=True), 1e-6)
        probe_ang_vec = probe_ang_dir * jnp.maximum(probe_ang, 0.0)
        angvel_kick = jnp.where(probe_ang >= 0.0, probe_ang_vec, train_angvel)
        qvel = qvel.at[:, 3:6].set(qvel[:, 3:6] + angvel_kick)
        if carryover_qvel is not None:
            min_hold_s, _ = self._transition_hold_and_jump(params[P.transition_difficulty])
            trans_frac = (TRANSITION_HOLD_MAX_S - min_hold_s) / (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
            carry_prob = trans_frac * CARRYOVER_MAX_PROB
            do_carry = jax.vmap(lambda k: jax.random.uniform(k, ()))(carry_keys) < carry_prob
            qvel = jnp.where(do_carry[:, None], carryover_qvel, qvel)
        return data.replace(qpos=qpos, qvel=qvel)

    def _specialist_enabled(self, keys, params):
        """Earned specialist activation preserves main's cold steady gait.

        Source main began with a 70-update steady-only bootstrap, then grew
        difficulty by measured competence. In this unified cold lineage we
        use the SAME earned steady difficulty to blend emphasis gradually,
        rather than a hard 20M unlock or full specialist mix at step zero.
        The RNG is independently folded: no legacy task/sensor key shifts.
        """
        jax,jnp=self.jax,self.jnp
        if not self.unified_capability_focus:
            return jnp.zeros((self.num_envs,),dtype=jnp.bool_)
        # Host decides phase from absolute steps AND earned steady competence;
        # there is no second independent maturity schedule inside the world.
        mix=jnp.clip(params[P.specialist_mix],0.,1.)
        k=jax.vmap(lambda x:jax.random.fold_in(x,50240))(keys)
        return jax.vmap(lambda x:jax.random.uniform(x,()))(k)<mix

    def _specialist_task(self, task, keys, params):
        """Change difficulty emphasis, never policy or actor observation.

        The 2048 worlds share the same robot, contacts, neural weights, and
        optimizer. Commands sample the measured L1 boundary, while control
        teaches cheaper/smoother gait inside an easier reliable envelope.
        Real incline+step collision is a separate mandatory launch gate.
        """
        jnp=self.jnp
        active=self._specialist_enabled(keys,params)
        if not self.unified_capability_focus:
            return task,active
        jax=self.jax
        focus=self.focus_id
        is_command=active & (focus==COMMAND)
        is_resilience=active & (focus==RESILIENCE)
        is_transition=active & (focus==TRANSITION)
        is_control=active & (focus==CONTROL)
        # Command specialist gets a single steady target, so its *ended*
        # episode is meaningful evidence of that command direction.
        # Transition and resilience specialists keep their own existing
        # dynamics, with all worlds still carrying normal contact tasks.
        regime=jnp.where(is_command,0.,
               jnp.where(is_resilience,2.,
               jnp.where(is_transition,1.,
               jnp.where(is_control,0.,task[:,T.regime]))))
        task=task.at[:,T.regime].set(regime)
        ordinary=~(is_command|is_control)
        task=task.at[:,T.athletic_rehearsal].set(
            jnp.where(ordinary,task[:,T.athletic_rehearsal],0.))
        key_probe=jax.vmap(lambda k:jax.random.fold_in(k,50241))(keys)
        # In unified mode the governor's frontiers START SMALL and expand
        # on proven episodes; don't add a second hard-coded 20M ramp.
        span=jnp.asarray(1.0,dtype=jnp.float32)
        candidate=diamond_commands(
            jax,jnp,key_probe,
            params[P.frontier_vx_hi],params[P.frontier_vx_lo],
            params[P.frontier_vy],params[P.frontier_yaw],
            probe_scale=params[P.frontier_combo_radius])
        # The control reference must still WALK; no zero-velocity energy
        # farming. Other commands remain small but not all exactly zero.
        k_control=jax.vmap(lambda k:jax.random.fold_in(k,50242))(keys)
        u=jax.vmap(lambda k:jax.random.uniform(k,(3,)))(k_control)
        steady_hi=jnp.minimum(params[P.frontier_vx_hi],.06+.22*span)
        control_vx=.04+jnp.minimum(.15,steady_hi-.04)*u[:,0]
        control_vy=(u[:,1]*2-1)*params[P.frontier_vy]*span*.18
        control_yaw=(u[:,2]*2-1)*params[P.frontier_yaw]*span*.18
        desired=jnp.stack([task[:,T.vx],task[:,T.vy],task[:,T.yaw]],axis=-1)
        desired=jnp.where(is_command[:,None],candidate,desired)
        easy=jnp.stack((control_vx,control_vy,control_yaw),axis=-1)
        desired=jnp.where(is_control[:,None],easy,desired)
        old_speed=jnp.sqrt(task[:,T.vx]**2+task[:,T.vy]**2)
        new_speed=jnp.sqrt(desired[:,0]**2+desired[:,1]**2)
        base=jnp.clip(task[:,T.phase_rate]-
                      self.phase_speed_gain*jnp.clip(
                        old_speed/jnp.maximum(.05,self.vx_hi_max),0.,1.5),
                      .85,1.20)
        new_phase=command_phase_rate(desired[:,0],desired[:,1],
                                    self.vx_hi_max,self.phase_speed_gain,base)
        modified=is_command|is_control
        task=task.at[:,T.vx].set(desired[:,0])
        task=task.at[:,T.vy].set(desired[:,1])
        task=task.at[:,T.yaw].set(desired[:,2])
        task=task.at[:,T.phase_rate].set(
            jnp.where(modified,new_phase,task[:,T.phase_rate]))
        # Resilience focused worlds get current earned push severity. Control
        # stays gentler in disturbed starts; neither makes pushes vanish
        # from ordinary/terrain specialists.
        resilience=jnp.where(is_resilience,
             jnp.clip(params[P.frontier_push_n]/jnp.maximum(.05,self.push_max_force_n),.02,1.),
             task[:,T.resilience])
        resilience=jnp.where(is_control,resilience*.35,resilience)
        task=task.at[:,T.resilience].set(resilience)
        # Control keeps modest variations in posture even during the early
        # gait-efficiency lesson, rather than hiding those capabilities.
        for column in (T.height,T.pitch,T.roll):
            task=task.at[:,column].set(jnp.where(
                is_control,.25*task[:,column],task[:,column]))
        task=task.at[:,T.birth_resilience_d].set(
            jnp.where(is_resilience,params[P.resilience_difficulty],
                      jnp.where(is_control,
                                .35*params[P.resilience_difficulty],
                                task[:,T.birth_resilience_d])))
        return task,active

    def _unified_terrain(self, data, keys, params):
        """Solid pair of independently sampled inclined contact surfaces.

        Local top faces share n, tangent e/w; the discontinuity is along
        their common normal. Both slabs always exist in model topology.
        A dormant upper slab is buried and the lower slab spans the world;
        non-dormant upper/lower slabs abut exactly at the sharp lip.
        Changing ONLY mocap_pos/quat at reset keeps JAX compilation fixed.
        """
        jax,jnp=self.jax,self.jnp
        def draw(key):
            ks=jax.random.split(jax.random.fold_in(key,50245),6)
            return jnp.stack([jax.random.uniform(ks[i],()) for i in range(6)])
        u=jax.vmap(draw)(keys)
        focused=self._specialist_enabled(keys,params)
        focus=self.focus_id
        step_dose=jnp.where(focused & (focus==STEP),4.0,.25)
        slope_dose=jnp.where(focused & (focus==SLOPE),4.0,.25)
        # The governor may seed gentle physical severity on the exact 20M
        # boundary; the host-owned fade ensures the NEXT worlds do not
        # instantly get 15% obstacle births before gait has transitioned.
        mix=jnp.clip(params[P.specialist_mix],0.,1.)
        step_p=jnp.minimum(1.0,params[P.terrain_exposure]*step_dose*mix)
        slope_p=jnp.minimum(1.0,params[P.slope_exposure]*slope_dose*mix)
        has_fixed_height=jnp.isfinite(params[P.terrain_fixed_height_m])
        step_active=jnp.where(
            has_fixed_height,params[P.terrain_fixed_height_m]>1e-6,
            (u[:,0]<step_p)&(params[P.terrain_height_m]>1e-6))
        slope_active=(u[:,1]<slope_p) & (params[P.slope_max_deg]>1e-6)
        slope_mag=(.30+.70*u[:,2])*params[P.slope_max_deg]
        slope_rad=jnp.deg2rad(jnp.where(slope_active,slope_mag,0.))
        # Azimuth 0,180deg: longitudinal uphill/downhill. 90,270deg:
        # sideways camber. Explicit pitch/roll overrides serve the
        # reproducible CPU/held-out examiner and do not alter actor inputs.
        azimuth=2.*jnp.pi*u[:,3]
        fixed=jnp.isfinite(params[P.terrain_pitch_deg]) & jnp.isfinite(
            params[P.terrain_roll_deg])
        gx=jnp.where(fixed,jnp.tan(jnp.deg2rad(params[P.terrain_pitch_deg])),
                     jnp.tan(slope_rad)*jnp.cos(azimuth))
        gy=jnp.where(fixed,jnp.tan(jnp.deg2rad(params[P.terrain_roll_deg])),
                     jnp.tan(slope_rad)*jnp.sin(azimuth))
        normal=jnp.stack((-gx,-gy,jnp.ones_like(gx)),axis=1)
        normal=normal/jnp.linalg.norm(normal,axis=1,keepdims=True)
        # Shortest quaternion taking world +Z into physical support normal.
        qt=jnp.stack((1.+normal[:,2],-normal[:,1],normal[:,0],
                      jnp.zeros_like(gx)),axis=1)
        qt=qt/jnp.linalg.norm(qt,axis=1,keepdims=True)
        edge_yaw=jnp.where(jnp.isfinite(params[P.terrain_edge_angle_deg]),
                           jnp.deg2rad(params[P.terrain_edge_angle_deg]),
                           (u[:,4]-.5)*.70)
        qy=jnp.stack((jnp.cos(edge_yaw/2),jnp.zeros_like(gx),
                      jnp.zeros_like(gx),jnp.sin(edge_yaw/2)),axis=1)
        q=self._quat_mul(qt,qy)
        qw,qx,qyy,qz=q[:,0],q[:,1],q[:,2],q[:,3]
        direction=jnp.stack((1.-2.*(qyy*qyy+qz*qz),
                             2.*(qx*qyy+qw*qz),
                             2.*(qx*qz-qw*qyy)),axis=1)
        edge=jnp.stack((jnp.full_like(gx,TERRAIN_PLATFORM_EDGE_X_M),
                        jnp.zeros_like(gx),
                        gx*TERRAIN_PLATFORM_EDGE_X_M),axis=1)
        # Up: left lower / right upper; down: left upper / right lower.
        down=(self.env_index.astype(jnp.int32)%2)==1
        side=jnp.where(down,-1.,1.)[:,None]
        L=UNIFIED_SLAB_HALF_LENGTH_M
        D=UNIFIED_SLAB_HALF_DEPTH_M
        floor_active=edge-side*L*direction-D*normal
        floor_dormant=-D*normal
        floor_pos=jnp.where(step_active[:,None],floor_active,floor_dormant)
        ceiling=jnp.clip(params[P.terrain_height_m],0.,self.terrain_step_max_m)
        # Height is the physical normal displacement, not an impossible
        # world-vertical offset that would force the two faces to overlap.
        height=jnp.where(has_fixed_height,
                         params[P.terrain_fixed_height_m],
                         ceiling*(.30+.70*u[:,5]))
        upper_pos=edge+side*L*direction+(height[:,None]-D)*normal
        buried=jnp.asarray((0.,0.,-2.),dtype=jnp.float32)
        upper_pos=jnp.where(step_active[:,None],upper_pos,buried)
        floor_i=self.unified_ground_mocap_id
        upper_i=self.terrain_step_mocap_id
        return data.replace(
            mocap_pos=data.mocap_pos.at[:,floor_i,:].set(floor_pos).at[:,upper_i,:].set(upper_pos),
            mocap_quat=data.mocap_quat.at[:,floor_i,:].set(q).at[:,upper_i,:].set(q),
            qpos=data.qpos.at[:,2].add(
                jnp.where(step_active & down,height*normal[:,2],0.))
        )

    def _episode_terrain(self, data, keys, params):
        """Sample real episode terrain with an exactly flat dormant state.

        A zero terrain exposure or height buries the *same* physical box
        below the ground. Up/down pairs, height and edge angle only matter
        for exposed worlds. The task parameters affect geometry through the
        actual MJX reset, not an actor-visible difficulty label.
        """
        if self.terrain_step_mocap_id < 0:
            return data
        if self.unified_capability_focus:
            return self._unified_terrain(data,keys,params)
        jax, jnp = self.jax, self.jnp
        k = jax.vmap(lambda x: jax.random.fold_in(x, 5023))(keys)
        def choose(key):
            kh, ka, ke = jax.random.split(key, 3)
            magnitude = 0.3 + 0.7*jax.random.uniform(kh, ())
            angle = jax.random.uniform(ka, (), minval=-0.35, maxval=0.35)
            selection_u = jax.random.uniform(ke, ())
            return magnitude, angle, selection_u
        magnitude, angle, selection_u = jax.vmap(choose)(k)
        if self.unified_capability_focus:
            # Not exclusive: steps are emphasized in 20% of worlds, while
            # *all* specialists and even control see low-dose real edges.
            focus=self.focus_id
            active_focus=self._specialist_enabled(keys,params)
            dose=jnp.where(active_focus & (focus==STEP),4.0,.25)
            probability=jnp.clip(params[P.terrain_exposure]*dose,0.,1.)
        else:
            probability=params[P.terrain_exposure]
        selected = selection_u < probability
        if self.paired_terrain_heights:
            # Independent frozen exam only. Each adjacent UP/DOWN pair has
            # identical real height AND edge yaw. Live curriculum keeps
            # its per-episode severity sampling unchanged.
            pair_index = jnp.arange(self.num_envs, dtype=jnp.int32)//2
            pair_keys = jax.vmap(lambda x: jax.random.fold_in(
                jax.random.PRNGKey(self.terrain_pair_seed), x))(pair_index)
            magnitude = .3 + .7*jax.vmap(
                lambda x: jax.random.uniform(x, ()))(pair_keys)
            angle = jax.vmap(lambda x: jax.random.uniform(
                jax.random.fold_in(x, 19), (), minval=-.35, maxval=.35))(pair_keys)
        ceiling = jnp.clip(params[P.terrain_height_m], 0.0, self.terrain_step_max_m)
        active = selected & (ceiling > 1e-6)
        height = jnp.where(active, ceiling * magnitude, 0.0)
        # Stratified even/odd up/down worlds in the exposed subset, so a
        # small batch with 100% exposure still sees both physical types.
        down = (self.env_index.astype(jnp.int32) % 2) == 1
        ca, sa = jnp.cos(angle), jnp.sin(angle)
        side = jnp.where(down, -1.0, 1.0)
        mid_x = TERRAIN_PLATFORM_EDGE_X_M + side*TERRAIN_PLATFORM_HALF_LENGTH_M*ca
        mid_y = side*TERRAIN_PLATFORM_HALF_LENGTH_M*sa
        center = jnp.stack((
            mid_x, mid_y, height - TERRAIN_PLATFORM_HALF_DEPTH_M), axis=1)
        # No ghost contacts or ray returns during the first 20M flat steps:
        # absent terrain remains buried ~85cm below the real contact plane.
        buried = jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32)
        center = jnp.where(active[:, None], center, buried)
        zeros = jnp.zeros_like(angle)
        yaw_quat = jnp.stack((jnp.cos(angle/2), zeros, zeros,
                              jnp.sin(angle/2)), axis=1)
        idx = self.terrain_step_mocap_id
        return data.replace(
            mocap_pos=data.mocap_pos.at[:, idx, :].set(center),
            mocap_quat=data.mocap_quat.at[:, idx, :].set(yaw_quat),
            qpos=data.qpos.at[:, 2].add(jnp.where(active & down, height, 0.0)),
        )

    def _reset_with_keys(self, keys, params, carryover_qvel=None):
        # Same as reset_impl but accepts a key per env so done envs can be reset inside jit.
        # carryover_qvel, when given, is the just-ended episode's actual
        # final qvel (already rolled back to finite by _step_impl's
        # physics-divergence guard before this is ever called) --
        # episode-boundary state carryover blends it in with probability
        # scaled by how close transition_difficulty is to its hardest end,
        # so training practices "already moving, now asked for something
        # new" instead of only ever "starting from a fresh birth state".
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        # Ten single-purpose keys. Keep the persisted next RNG separate from
        # every draw made during this reset, and give resilience pose/position/
        # angular-velocity perturbations independent draws.
        (next_keys, noise_keys, vnoise_keys, phase_keys, task_keys, hold_keys,
         carry_keys, tilt_keys, pos_keys, angvel_keys, linvel_keys, joint_keys) = self._split_batch(keys, 12)
        data = self._make_batched_data()
        task = self._sample_task(task_keys, params)
        task, specialist_active = self._specialist_task(task, keys, params)
        data = self._birth_physics(
            data, task, params,
            (noise_keys, vnoise_keys, tilt_keys, pos_keys, angvel_keys,
             linvel_keys, joint_keys, carry_keys),
            carryover_qvel,
        )
        # One episode-local collision course. The standard flat path above
        # is bitwise untouched whenever terrain_step_max_m is zero.
        data = self._episode_terrain(data, keys, params)
        ctrl = data.ctrl
        ctrl = ctrl.at[:, :ACTION_DIM].set(self.home_target)
        data = data.replace(ctrl=ctrl)
        data = self.forward_batched(self.mjx_model, data)
        phase = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.0, maxval=2.0 * jnp.pi))(phase_keys)
        prev_action = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        prev_foot_pos = data.xpos[:, self.foot_bodies, :]
        prev_raw_target = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        sensors = (self._initial_sensor_state(data, task, phase, noise_keys)
                   if self.observation_schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN) else None)
        base = self._base_obs(data, task, prev_action, phase, prev_action, sensors)
        hist = jnp.repeat(base[:, None, :], HISTORY_LEN, axis=1)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left = self._init_transition_state(task, params, hold_keys)
        push_force0 = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        return EnvState(data, hist, next_keys, task, specialist_active, prev_action, prev_foot_pos, prev_raw_target, phase, izeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                         zeros, izeros,jnp.zeros((self.num_envs,),dtype=jnp.bool_),
                         jnp.full((self.num_envs,),-.12,dtype=jnp.float32), zeros,
                         cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left, push_force0, izeros, sensors)

    def _rotation_body_to_world(self, data):
        return data.xmat[:, self.torso_id, :].reshape((self.num_envs, 3, 3))

    def _body_packets(self, data):
        jnp = self.jnp
        R = self._rotation_body_to_world(data)
        RT = jnp.swapaxes(R, 1, 2)
        projected_gravity = jnp.einsum('nij,nj->ni', RT, self.gravity_unit_world)
        body_vel = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 0:3])
        body_ang = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 3:6])
        return projected_gravity, body_vel, body_ang

    def ground_sensor_probe(self, data, range_errors_m=None):
        """Provisional three-ToF + IMU surface observation, not actor truth.

        Floor geometry is used ONLY to synthesize beam returns. The inverse
        estimate sees measured ranges and projected IMU gravity, never the
        MuJoCo ground normal. Intentionally diagnostic until versioned
        v5016 actor input, latency and optical-flow models are accepted.
        """
        jnp=self.jnp
        R=self._rotation_body_to_world(data)
        torso=data.xpos[:,self.torso_id,:]
        f=int(self.ids['floor_geom'])
        floor_n=jnp.reshape(data.geom_xmat[:,f],(self.num_envs,3,3))[:,:,2]
        floor_o=data.geom_xpos[:,f,:]
        projected_gravity,_,_=self._body_packets(data)
        if self.unified_capability_focus:
            ranges4,valid4=self._terrain_4x4_ranges(data)
            fit=fit_forward_array_ground(jnp,ranges4,valid4,projected_gravity)
            # Legacy diagnostic shape only; zero 3-ToF cells are NOT actor
            # observations, and no independent tripod rays were sampled.
            return dict(
                ranges_m=jnp.zeros((self.num_envs,3),dtype=jnp.float32),
                beam_valid=jnp.zeros((self.num_envs,3),dtype=jnp.bool_),
                valid=fit['valid'],
                optical_depth_m=fit['optical_depth_m'],
                forward_grade_rad=fit['ground_forward_grade_rad'],
                sideways_grade_rad=fit['ground_sideways_grade_rad'],
                visibility_confidence=fit['ground_visibility_confidence'])
        ranges,beam_valid=simulate_tripod_ranges(
            jnp,torso,R,floor_o,floor_n,range_errors_m=range_errors_m)
        estimate=fit_tripod_ground(jnp,ranges,projected_gravity)
        return dict(ranges_m=ranges,beam_valid=beam_valid,**estimate)

    def optical_flow_probe(self, data, *, range_errors_m=None,
                           flow_errors_rad_s=None, gyro_errors_rad_s=None):
        """Hardware-facing diagnostic: ToF-scaled, gyro-compensated image flow.

        The forward camera model may read privileged MJX velocity/plane
        geometry to synthesize a ray and image flow. The INVERSE estimator
        receives measured flow, three-ToF fitted depth and local-frame gyro
        only. This is not yet the production actor observation pipeline.
        """
        jnp=self.jnp
        R=self._rotation_body_to_world(data)
        torso=data.xpos[:,self.torso_id,:]
        if self.unified_capability_focus:
            # Flow scale is the measured surface distance from the actual
            # downward camera, never a projected infinite plane.
            down_ray=(-R[:,:,2])[:,None,:]
            depth_floor=self._ray_box_distance(
                data,int(self.ids['floor_geom']),torso,down_ray)[:,0]
            depth_upper=self._ray_box_distance(
                data,int(self.terrain_step_geom_id),torso,down_ray)[:,0]
            true_depth=jnp.minimum(depth_floor,depth_upper)
            true_depth=jnp.where(jnp.isfinite(true_depth)&
                                  (true_depth>0.)&(true_depth<.60),true_depth,0.)
        else:
            floor=int(self.ids['floor_geom'])
            normal=jnp.reshape(data.geom_xmat[:,floor],(self.num_envs,3,3))[:,:,2]
            origin=data.geom_xpos[:,floor,:]
            beam=-R[:,:,2]
            denom=jnp.sum(beam*normal,axis=-1)
            distance=jnp.sum((origin-torso)*normal,axis=-1)
            true_depth=distance/jnp.where(jnp.abs(denom)>1e-6,denom,-1.)
            true_depth=jnp.where(denom<-.3,true_depth,0.)
        _,body_velocity,_=self._body_packets(data)
        # MuJoCo freejoint qvel[3:6] is ALREADY body-local; do not apply
        # the second world-to-body rotation used by the legacy 220 actor.
        body_gyro=data.qvel[:,3:6]
        measured_flow,flow_valid=simulate_downward_optical_flow(
            jnp,body_velocity,body_gyro,true_depth,
            flow_error_rad_s=flow_errors_rad_s)
        ground=self.ground_sensor_probe(data,range_errors_m=range_errors_m)
        measured_gyro=body_gyro+(jnp.asarray(gyro_errors_rad_s)
                                 if gyro_errors_rad_s is not None else 0.)
        estimate=estimate_flow_velocity(
            jnp,measured_flow,measured_gyro,ground['optical_depth_m'],
            flow_valid=flow_valid,tof_valid=ground['valid'])
        return dict(flow_rad_s=measured_flow,flow_return_valid=flow_valid,
                    measured_gyro_body_rad_s=measured_gyro,
                    tof_ranges_m=ground['ranges_m'],
                    tof_beam_valid=ground['beam_valid'],
                    tof_estimated_depth_m=ground['optical_depth_m'],
                    ground_measurement_valid=ground['valid'],
                    ground_forward_grade_rad=ground['forward_grade_rad'],
                    ground_sideways_grade_rad=ground['sideways_grade_rad'],
                    ground_visibility_confidence=ground['visibility_confidence'],
                    **estimate)

    def _ray_box_distance(self, data, geom_id, origin, rays):
        """Full 3-D ray/OBB slab test, including pitch, roll and edge yaw."""
        jnp=self.jnp
        R=data.geom_xmat[:,geom_id,:].reshape((self.num_envs,3,3))
        centre=data.geom_xpos[:,geom_id,:]
        if origin.ndim == 3:
            local_o=jnp.einsum('eki,eij->ekj',origin-centre[:,None,:],R)
        else:
            local_o=jnp.einsum('ei,eij->ej',origin-centre,R)[:,None,:]
        local_d=jnp.einsum('eki,eij->ekj',rays,R)
        half=jnp.asarray(self.mj_model.geom_size[geom_id,:3],dtype=jnp.float32)
        parallel=jnp.abs(local_d)<1e-7
        safe=jnp.where(parallel,1.,local_d)
        t1=(-half-local_o)/safe
        t2=(half-local_o)/safe
        inside=jnp.abs(local_o)<=half
        near=jnp.max(jnp.where(
            parallel,jnp.where(inside,-jnp.inf,jnp.inf),jnp.minimum(t1,t2)),axis=-1)
        far=jnp.min(jnp.where(
            parallel,jnp.where(inside,jnp.inf,-jnp.inf),jnp.maximum(t1,t2)),axis=-1)
        return jnp.where(far>=jnp.maximum(near,0.),
                         jnp.where(near>0.,near,far),jnp.inf)

    def _terrain_4x4_ranges(self, data, errors_m=None):
        """16 real slant rays, closest floor/finite box; foot occlusion masks cells.

        No height-map truth, per-ray CPU raycasts, or actor-visible collision IDs.
        A fixed-topology JAX slab/sphere intersection works on batched MJX data.
        """
        jnp = self.jnp
        beam_dirs = jnp.asarray([
            [math.cos(math.radians(p))*math.cos(math.radians(y)),
             math.cos(math.radians(p))*math.sin(math.radians(y)),
             -math.sin(math.radians(p))]
            for p in TERRAIN_TOF_PITCH_DEG for y in TERRAIN_TOF_YAW_DEG],
            dtype=jnp.float32)
        R = self._rotation_body_to_world(data)
        origin = data.xpos[:, self.torso_id, :]
        ray = jnp.einsum('eij,kj->eki', R, beam_dirs)
        if self.unified_capability_focus:
            # Both measured slant ranges are to actual collidable boxes.
            # Infinite-plane fallback would create phantom upper/lower
            # support beyond the sharp step discontinuity.
            floor_t=self._ray_box_distance(
                data,int(self.ids['floor_geom']),origin,ray)
            box_t=self._ray_box_distance(
                data,int(self.terrain_step_geom_id),origin,ray)
            terrain_t=jnp.minimum(floor_t,box_t)
        else:
            floor = int(self.ids['floor_geom'])
            normal = data.geom_xmat[:, floor, :].reshape((self.num_envs,3,3))[:,:,2]
            floor_orig = data.geom_xpos[:, floor, :]
            denom = jnp.einsum('eki,ei->ek', ray, normal)
            plane_t = jnp.einsum('ei,ei->e',floor_orig-origin,normal)[:,None] / jnp.where(
                jnp.abs(denom)>1e-7,denom,1.)
            plane_t = jnp.where(denom< -1e-6,plane_t,jnp.inf)

            if self.terrain_step_mocap_id >= 0:
                i = self.terrain_step_mocap_id
                centre = data.mocap_pos[:,i,:]
                q = data.mocap_quat[:,i,:]
                ca = 1.-2.*q[:,3]**2
                sa = 2.*q[:,0]*q[:,3]
                offset = origin-centre
                local_o = jnp.stack((
                    ca*offset[:,0]+sa*offset[:,1],
                    -sa*offset[:,0]+ca*offset[:,1],offset[:,2]),axis=-1)
                local_d = jnp.stack((
                    ca[:,None]*ray[:,:,0]+sa[:,None]*ray[:,:,1],
                    -sa[:,None]*ray[:,:,0]+ca[:,None]*ray[:,:,1],
                    ray[:,:,2]),axis=-1)
                half = jnp.asarray((TERRAIN_PLATFORM_HALF_LENGTH_M,
                                    TERRAIN_PLATFORM_HALF_WIDTH_M,
                                    TERRAIN_PLATFORM_HALF_DEPTH_M),dtype=jnp.float32)
                parallel = jnp.abs(local_d)<1e-7
                safe = jnp.where(parallel,1.,local_d)
                first = (-half-local_o[:,None,:])/safe
                last = (half-local_o[:,None,:])/safe
                inside = jnp.abs(local_o[:,None,:])<=half
                entering = jnp.where(parallel,jnp.where(inside,-jnp.inf,jnp.inf),
                                     jnp.minimum(first,last))
                leaving = jnp.where(parallel,jnp.where(inside,jnp.inf,-jnp.inf),
                                    jnp.maximum(first,last))
                near = jnp.max(entering,axis=-1)
                far = jnp.min(leaving,axis=-1)
                box_t = jnp.where(far>=jnp.maximum(near,0.),
                                  jnp.where(near>0.,near,far),jnp.inf)
                terrain_t = jnp.minimum(plane_t,box_t)
            else:
                terrain_t = plane_t

        foot_ids = jnp.asarray(self.ids['foot_geoms'],dtype=jnp.int32)
        foot_centres = data.geom_xpos[:,foot_ids,:]
        diff = origin[:,None,:]-foot_centres
        b = jnp.einsum('eki,eji->ekj',ray,diff)
        radii = jnp.asarray(self.mj_model.geom_size[np.asarray(self.ids['foot_geoms']),0])
        c = jnp.sum(diff*diff,axis=-1)-radii[None,:]**2
        disc = b*b-c[:,None,:]
        near = -b-jnp.sqrt(jnp.maximum(0.,disc))
        far = -b+jnp.sqrt(jnp.maximum(0.,disc))
        hit = jnp.where(near>1e-6,near,far)
        occlusion_t = jnp.min(jnp.where((disc>=0.)&(hit>1e-6),hit,jnp.inf),axis=-1)
        valid = ((terrain_t>=TERRAIN_TOF_MIN_M)&(terrain_t<=TERRAIN_TOF_MAX_M)&
                 (occlusion_t>=terrain_t))
        result = terrain_t+(errors_m if errors_m is not None else 0.)
        return (jnp.where(valid,jnp.clip(result,TERRAIN_TOF_MIN_M,
                                        TERRAIN_TOF_MAX_M),0.).astype(jnp.float32),
                valid)

    def _unified_sensor_sample(self, data, task, keys, gyro_bias):
        """Array-only sensor: 4x4 ranges, optical flow and gyro, NO tripod.

        Physics truth is used only to produce the simulated optical signal,
        exactly as in old sensor forward synthesis. Actual actor readout
        uses the delayed array fit; no privileged slope or exact root velocity.
        """
        jax,jnp=self.jax,self.jnp
        scale=jnp.where(task[:,T.sensor_scale]>=0.,
                        task[:,T.sensor_scale],self.sensor_noise_scale)
        noise=jax.vmap(lambda k:jax.random.normal(
            jax.random.fold_in(k,5016),(8,),dtype=jnp.float32))(keys)
        four_noise=jax.vmap(lambda k:jax.random.normal(
            jax.random.fold_in(k,5024),(16,),dtype=jnp.float32))(keys)
        four,valid=self._terrain_4x4_ranges(
            data,errors_m=scale[:,None]*.002*four_noise)
        R=self._rotation_body_to_world(data)
        torso=data.xpos[:,self.torso_id,:]
        down=(-R[:,:,2])[:,None,:]
        floor_depth=self._ray_box_distance(
            data,int(self.ids['floor_geom']),torso,down)[:,0]
        upper_depth=self._ray_box_distance(
            data,int(self.terrain_step_geom_id),torso,down)[:,0]
        depth=jnp.minimum(floor_depth,upper_depth)
        depth=jnp.where(jnp.isfinite(depth)&(depth>=.035)&(depth<=.60),
                        depth,0.)
        _,vbody,_=self._body_packets(data)
        physical_gyro=data.qvel[:,3:6]
        flow,flow_valid=simulate_downward_optical_flow(
            jnp,vbody,physical_gyro,depth,
            flow_error_rad_s=scale[:,None]*.035*noise[:,3:5])
        measured_gyro=physical_gyro+gyro_bias+scale[:,None]*.025*noise[:,5:8]
        zero_ranges=jnp.zeros((self.num_envs,3),dtype=jnp.float32)
        return SensorSample(
            zero_ranges,jnp.zeros_like(zero_ranges,dtype=jnp.bool_),
            four,valid,flow,flow_valid,measured_gyro,
            jnp.zeros((self.num_envs,),dtype=jnp.int32),
            jnp.zeros((self.num_envs,),dtype=jnp.int32))

    def _sensor_sample(self, data, task, keys, gyro_bias):
        """Synthesize raw camera/ToF/IMU readings with independent JAX noise.

        Simulator truth is used only in forward measurement synthesis. The
        later inverse estimator is deliberately fed this sample ring, never
        the simulated ground normal or qvel. Fold-in does not perturb
        physics/lesson RNG streams or the old 220-input behavior.
        """
        if self.unified_capability_focus:
            return self._unified_sensor_sample(data,task,keys,gyro_bias)
        jax, jnp = self.jax, self.jnp
        scale = jnp.where(task[:, T.sensor_scale] >= 0.,
                          task[:, T.sensor_scale], self.sensor_noise_scale)
        noise = jax.vmap(lambda k: jax.random.normal(
            jax.random.fold_in(k, 5016), (8,), dtype=jnp.float32))(keys)
        probe = self.optical_flow_probe(
            data, range_errors_m=scale[:, None] * 0.0015 * noise[:, :3],
            flow_errors_rad_s=scale[:, None] * 0.035 * noise[:, 3:5],
            gyro_errors_rad_s=gyro_bias + scale[:, None] * 0.025 * noise[:, 5:8])
        ages = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        if self.observation_schema == OBS_SCHEMA_TERRAIN:
            four_noise = jax.vmap(lambda k: jax.random.normal(
                jax.random.fold_in(k,5024),(16,),dtype=jnp.float32))(keys)
            ranges4, valid4 = self._terrain_4x4_ranges(
                data,errors_m=scale[:,None]*.002*four_noise)
        else:
            ranges4 = jnp.zeros((self.num_envs,16),dtype=jnp.float32)
            valid4 = jnp.zeros((self.num_envs,16),dtype=jnp.bool_)
        return SensorSample(
            probe['tof_ranges_m'], probe['tof_beam_valid'],
            ranges4,valid4,
            probe['flow_rad_s'], probe['flow_return_valid'],
            probe['measured_gyro_body_rad_s'], ages, ages)

    def _initial_sensor_state(self, data, task, phase, keys):
        """Start an episode with a real first sample, not fabricated flat data."""
        jax, jnp = self.jax, self.jnp
        bias_noise = jax.vmap(lambda k: jax.random.normal(
            jax.random.fold_in(k, 5017), (3,), dtype=jnp.float32))(keys)
        bias = bias_noise * self.gyro_bias_std_rad_s
        sample = self._sensor_sample(data, task, keys, bias)
        samples = jax.tree_util.tree_map(
            lambda a: jnp.repeat(a[:, None, ...], HISTORY_LEN, axis=1),
            sample)
        return SensorState(samples, bias, sample.gyro_body_rad_s)

    def _advance_sensor_state(self, state, data, task, phase, keys, age_frames):
        """Acquire at separate cadences, hold between updates, and queue delay.

        Dropouts make the *new* sample invalid; an old value may still be
        in transit, but an invalid one eventually reaches the policy with
        zero confidence. Clock is episode age, not staggered timeout step_i.
        """
        jax, jnp = self.jax, self.jnp
        raw = self._sensor_sample(data, task, keys, state.gyro_bias_rad_s)
        old = jax.tree_util.tree_map(lambda a: a[:, -1, ...], state.samples)
        step = jnp.asarray(age_frames, dtype=jnp.int32)
        tof_due = (step % self.tof_period_frames) == 0
        flow_due = (step % self.flow_period_frames) == 0
        # Dedicated fold-ins prevent correlated failure events and never
        # change how physical pushes and curriculum resampling are drawn.
        tof_failed = jax.vmap(lambda k: jax.random.uniform(
            jax.random.fold_in(k, 5018), ()))(keys) < self.tof_dropout_prob
        flow_failed = jax.vmap(lambda k: jax.random.uniform(
            jax.random.fold_in(k, 5019), ()))(keys) < self.flow_dropout_prob
        tof_valid = raw.tof_beam_valid & ~tof_failed[:, None]
        terrain_valid = raw.terrain_valid & ~tof_failed[:, None]
        flow_valid = raw.flow_valid & ~flow_failed
        latest = SensorSample(
            jnp.where(tof_due[:, None], jnp.where(tof_valid, raw.tof_ranges_m, 0.), old.tof_ranges_m),
            jnp.where(tof_due[:, None], tof_valid, old.tof_beam_valid),
            jnp.where(tof_due[:,None],jnp.where(terrain_valid,raw.terrain_ranges_m,0.),old.terrain_ranges_m),
            jnp.where(tof_due[:,None],terrain_valid,old.terrain_valid),
            jnp.where(flow_due[:, None], jnp.where(flow_valid[:, None], raw.flow_rad_s, 0.), old.flow_rad_s),
            jnp.where(flow_due, flow_valid, old.flow_valid),
            jnp.where(flow_due[:, None], raw.gyro_body_rad_s, old.gyro_body_rad_s),
            jnp.where(tof_due, 0, old.tof_age_frames + 1),
            jnp.where(flow_due, 0, old.flow_age_frames + 1))
        queued = jax.tree_util.tree_map(
            lambda history, value: jnp.concatenate(
                (history[:, 1:, ...], value[:, None, ...]), axis=1),
            state.samples, latest)
        return SensorState(queued, state.gyro_bias_rad_s, raw.gyro_body_rad_s)

    def _sensor_readout(self, state, gravity_body):
        """Invert delayed measured signals, including their validity/age.

        Flow is gyro-compensated with the IMU sample from the SAME camera
        acquisition frame. The current actor gyro remains current (50 Hz).
        ToF is independently delayed and the gravity estimate is current.
        """
        jnp = self.jnp
        s = state.samples
        tf = HISTORY_LEN - 1 - self.tof_delay_frames
        ff = HISTORY_LEN - 1 - self.flow_delay_frames
        if self.unified_capability_focus:
            ground=fit_forward_array_ground(
                jnp,s.terrain_ranges_m[:,tf,:],s.terrain_valid[:,tf,:],
                gravity_body)
            flow_valid=s.flow_valid[:,ff]
            estimate=estimate_flow_velocity(
                jnp,s.flow_rad_s[:,ff,:],s.gyro_body_rad_s[:,ff,:],
                ground['optical_depth_m'],flow_valid=flow_valid,
                tof_valid=ground['depth_valid'])
            tof_age=s.tof_age_frames[:,tf]
            flow_age=s.flow_age_frames[:,ff]
            freshness=jnp.clip(
                1.-.25*(tof_age.astype(jnp.float32)/self.tof_period_frames+
                         flow_age.astype(jnp.float32)/self.flow_period_frames),
                .25,1.)
            return dict(
                measured_gyro_body_rad_s=state.current_gyro_body_rad_s,
                velocity_body_xy_mps=estimate['velocity_body_xy_mps'],
                velocity_confidence=estimate['velocity_confidence']*freshness,
                tof_estimated_depth_m=ground['optical_depth_m'],
                ground_forward_grade_rad=ground['ground_forward_grade_rad'],
                ground_sideways_grade_rad=ground['ground_sideways_grade_rad'],
                ground_visibility_confidence=ground['ground_visibility_confidence']*
                    jnp.clip(1.-.25*tof_age.astype(jnp.float32)/
                             self.tof_period_frames,.25,1.),
                ground_valid=ground['valid'],
                terrain_ranges_m=s.terrain_ranges_m[:,tf,:],
                terrain_valid=s.terrain_valid[:,tf,:],
                flow_valid=flow_valid,
                tof_age_frames=tof_age,flow_age_frames=flow_age)
        ranges = s.tof_ranges_m[:, tf, :]
        beams_valid = jnp.all(s.tof_beam_valid[:, tf, :], axis=1)
        ground = fit_tripod_ground(jnp, ranges, gravity_body)
        ground_valid = ground['valid'] & beams_valid
        flow_valid = s.flow_valid[:, ff]
        estimate = estimate_flow_velocity(
            jnp, s.flow_rad_s[:, ff, :], s.gyro_body_rad_s[:, ff, :],
            ground['optical_depth_m'],
            flow_valid=flow_valid, tof_valid=ground_valid)
        # Confidence records age/staleness, not an invented probability of
        # geometric accuracy. An outright invalid reading remains confidence 0.
        tof_age = s.tof_age_frames[:, tf]
        flow_age = s.flow_age_frames[:, ff]
        freshness = jnp.clip(
            1. - 0.25 * (tof_age.astype(jnp.float32) /
                         self.tof_period_frames +
                         flow_age.astype(jnp.float32) /
                         self.flow_period_frames), .25, 1.)
        grade_freshness = jnp.clip(
            1. - 0.25 * tof_age.astype(jnp.float32) /
            self.tof_period_frames, .25, 1.)
        return dict(
            measured_gyro_body_rad_s=state.current_gyro_body_rad_s,
            velocity_body_xy_mps=estimate['velocity_body_xy_mps'],
            velocity_confidence=estimate['velocity_confidence'] * freshness,
            tof_estimated_depth_m=jnp.where(ground_valid,
                                              ground['optical_depth_m'], 0.),
            ground_forward_grade_rad=jnp.where(ground_valid,
                                               ground['forward_grade_rad'], 0.),
            ground_sideways_grade_rad=jnp.where(ground_valid,
                                                ground['sideways_grade_rad'], 0.),
            ground_visibility_confidence=jnp.where(
                ground_valid, ground['visibility_confidence'] * grade_freshness, 0.),
            ground_valid=ground_valid,
            terrain_ranges_m=s.terrain_ranges_m[:,tf,:],
            terrain_valid=s.terrain_valid[:,tf,:],
            flow_valid=flow_valid,
            tof_age_frames=tof_age, flow_age_frames=flow_age)

    def _floor_signed_heights(self, data, positions):
        """Per-world signed distance to the *actual* collision plane.

        positions is [worlds,points,3]. Flat ground preserves exact world-Z
        semantics; tilted-gravity trials still have a flat contact surface.
        This is privileged simulator geometry, NEVER an actor observation.
        """
        floor = int(self.ids['floor_geom'])
        normal = self.jnp.reshape(data.geom_xmat[:,floor], (self.num_envs,3,3))[:,:,2]
        origin = data.geom_xpos[:,floor,:]
        if self.unified_capability_focus:
            origin=origin+UNIFIED_SLAB_HALF_DEPTH_M*normal
        return self.jnp.sum((positions-origin[:,None,:])*normal[:,None,:],axis=-1)

    def _support_signed_heights(self, data, positions):
        """Teacher-only distance above real local support (plane OR box top).

        This does NOT go to the 220/232 policy. The approximate top-surface
        classifier is valid away from vertical side contacts; crossing
        acceptance must still inspect real contact pairs and trajectory.
        """
        flat = self._floor_signed_heights(data, positions)
        if self.unified_capability_focus:
            jnp=self.jnp
            geom=int(self.terrain_step_geom_id)
            R=data.geom_xmat[:,geom,:].reshape((self.num_envs,3,3))
            delta=positions-data.geom_xpos[:,geom,None,:]
            local=jnp.einsum('epi,eij->epj',delta,R)
            half=self.mj_model.geom_size[geom,:3]
            inside=(jnp.abs(local[:,:,0])<=half[0]) & (
                    jnp.abs(local[:,:,1])<=half[1])
            active=data.mocap_pos[:,self.terrain_step_mocap_id,2]>-1.
            return jnp.where(inside & active[:,None],
                             local[:,:,2]-half[2],flat)
        if self.terrain_step_mocap_id < 0:
            return flat
        jnp = self.jnp
        i = self.terrain_step_mocap_id
        center = data.mocap_pos[:, i, :]
        quat = data.mocap_quat[:, i, :]
        c = 1.0 - 2.0*quat[:, 3]**2
        si = 2.0*quat[:, 0]*quat[:, 3]
        delta = positions[:, :, :2] - center[:, None, :2]
        local_x = delta[:, :, 0]*c[:, None] + delta[:, :, 1]*si[:, None]
        local_y = -delta[:, :, 0]*si[:, None] + delta[:, :, 1]*c[:, None]
        inside = ((jnp.abs(local_x) < TERRAIN_PLATFORM_HALF_LENGTH_M) &
                  (jnp.abs(local_y) < TERRAIN_PLATFORM_HALF_WIDTH_M))
        top = center[:, 2] + TERRAIN_PLATFORM_HALF_DEPTH_M
        elevated = inside & (top[:, None] > 0.001)
        return jnp.where(elevated, positions[:, :, 2]-top[:, None], flat)

    def _terrain_crossing(self, state, data, metrics):
        """Sustained physical far-side crossing, teacher telemetry only.

        Real foot-floor/box contacts come from MJX. A step-down is not a
        zero-height reading: its target support is the actual lower floor.
        The initial edge location is known ONLY to this independent examiner.
        """
        jnp = self.jnp
        if self.terrain_step_mocap_id < 0:
            return state.terrain_cross_streak*0,state.terrain_crossed & False,(
                jnp.zeros((self.num_envs,),dtype=jnp.bool_))
        from .directional import contact_pair_mask
        i = self.terrain_step_mocap_id
        centre=data.mocap_pos[:,i,:]
        q=data.mocap_quat[:,i,:]
        down=(self.env_index.astype(jnp.int32)%2)==1
        side=jnp.where(down,-1.,1.)
        if self.unified_capability_focus:
            R=data.geom_xmat[:,int(self.terrain_step_geom_id),:].reshape(
                (self.num_envs,3,3))
            axis=R[:,:,0]
            across=R[:,:,1]
            half_length=UNIFIED_SLAB_HALF_LENGTH_M
            edge=centre-side[:,None]*half_length*axis
            delta=data.qpos[:,:3]-edge
            forward=jnp.sum(delta*axis,axis=1)
            lateral=jnp.sum(delta*across,axis=1)
        else:
            ca=1.0-2.0*q[:,3]**2
            sa=2.0*q[:,0]*q[:,3]
            axis=jnp.stack((ca,sa),axis=1)
            edge=centre[:,:2]-side[:,None]*TERRAIN_PLATFORM_HALF_LENGTH_M*axis
            delta=data.qpos[:,:2]-edge
            forward=jnp.sum(delta*axis,axis=1)
            lateral=-delta[:,0]*sa+delta[:,1]*ca
        contact=getattr(data,'_impl',data).contact
        feet=tuple(int(x) for x in self.ids['foot_geoms'])
        box_geom=int(self.terrain_step_geom_id)
        floor=int(self.ids['floor_geom'])
        box_contact=contact_pair_mask(jnp,contact.geom1,contact.geom2,
                                      contact.dist,box_geom,feet)
        floor_contact=contact_pair_mask(jnp,contact.geom1,contact.geom2,
                                        contact.dist,floor,feet)
        supported=jnp.any(jnp.where(down[:,None],floor_contact,box_contact),axis=1)
        active=centre[:,2]>(-1. if self.unified_capability_focus else -.5)
        good=(active & (forward>.10) & (jnp.abs(lateral)<.27) &
              supported & (metrics['vx']>.02) &
              (metrics['bad_contact']<.5) & (metrics['terminal']<.5))
        streak=jnp.where(good,state.terrain_cross_streak+1,0)
        crossed=state.terrain_crossed | (streak>=5)
        return streak,crossed,active

    def _terrain_ascent_guidance(self, state, data, metrics, terrain_up):
        """Small, non-repeatable physical ascent teaching; NEVER an actor input.

        Up to 6.6 progress + 2.4 foot-placement reward PER EPISODE.
        Pay only near the real lip, when commanded forward and stable. Progress
        needs new root travel and a real MJX floor-foot support pair; foot
        placement needs the OTHER foot actually grounded and the swing ball
        bottom above the upper platform. Merely lifting in place, moving while
        falling, backwards rocking or an unsupported leap cannot farm rewards.
        Real crossing remains scored independently in _terrain_crossing.
        """
        jnp = self.jnp
        if self.terrain_step_mocap_id < 0:
            return (jnp.zeros_like(metrics['vx']), state.terrain_ascent_root_peak,
                    state.terrain_ascent_foot_peak)
        from .directional import contact_pair_mask
        center = data.mocap_pos[:, self.terrain_step_mocap_id, :]
        q = data.mocap_quat[:, self.terrain_step_mocap_id, :]
        if self.unified_capability_focus:
            R=data.geom_xmat[:,int(self.terrain_step_geom_id),:].reshape(
                (self.num_envs,3,3))
            axis,across,normal=R[:,:,0],R[:,:,1],R[:,:,2]
            edge=center-UNIFIED_SLAB_HALF_LENGTH_M*axis
            delta=data.qpos[:,:3]-edge
            forward=jnp.sum(delta*axis,axis=1)
            lateral=jnp.sum(delta*across,axis=1)
            moved=jnp.sum((data.qpos[:,:3]-state.data.qpos[:,:3])*axis,axis=1)
        else:
            axis = jnp.stack((1.-2.*q[:,3]**2, 2.*q[:,0]*q[:,3]), axis=1)
            edge = center[:,:2] - TERRAIN_PLATFORM_HALF_LENGTH_M*axis
            delta = data.qpos[:,:2] - edge
            forward = jnp.sum(delta*axis, axis=1)
            lateral = -delta[:,0]*axis[:,1] + delta[:,1]*axis[:,0]
            moved = jnp.sum((data.qpos[:,:2]-state.data.qpos[:,:2])*axis, axis=1)

        c = getattr(data, '_impl', data).contact
        feet = tuple(int(x) for x in self.ids['foot_geoms'])
        floor_pairs = contact_pair_mask(
            jnp, c.geom1, c.geom2, c.dist, int(self.ids['floor_geom']), feet)
        floor_stance = jnp.any(floor_pairs, axis=1)
        healthy = (terrain_up & (metrics['terminal']<.5) &
                   (metrics['bad_contact']<.5) & floor_stance &
                   (metrics['cmd_vx']>=.08) &
                   (jnp.abs(metrics['cmd_vy'])<.08) &
                   (jnp.abs(metrics['cmd_yaw'])<.5) &
                   (jnp.abs(lateral)<.22) & (forward>-.13))

        # Positive *new* root progress, up to 10cm past the lip. It must
        # actually move forward during this frame, while a foot is grounded.
        eligible_root = healthy & (moved>.0005) & (forward<.12)
        root_now = jnp.clip(forward, -.12, .10)
        root_peak = jnp.where(
            eligible_root, jnp.maximum(state.terrain_ascent_root_peak, root_now),
            state.terrain_ascent_root_peak)
        root_gain = root_peak - state.terrain_ascent_root_peak

        # A forward-reaching airborne foot, safely above the actual ledge
        # top, with the opposite foot's floor contact confirmed by MJX.
        # This signal is monotonic; 1,000 taps at the edge pay at most once.
        fp = data.xpos[:, self.foot_bodies, :]
        if self.unified_capability_focus:
            foot_delta=fp-edge[:,None,:]
            foot_forward=jnp.sum(foot_delta*axis[:,None,:],axis=2)
            foot_lateral=jnp.sum(foot_delta*across[:,None,:],axis=2)
            # Measured surface-normal clearance above the ACTUAL inclined
            # upper top face; world Z would reward fake footholds on camber.
            foot_clearance=jnp.sum((fp-center[:,None,:])*normal[:,None,:],
                                   axis=2)-UNIFIED_SLAB_HALF_DEPTH_M-.018
        else:
            foot_delta = fp[:,:,:2] - edge[:,None,:]
            foot_forward = jnp.sum(foot_delta*axis[:,None,:], axis=2)
            foot_lateral = (-foot_delta[:,:,0]*axis[:,None,1] +
                            foot_delta[:,:,1]*axis[:,None,0])
            upper_z = center[:,2]+TERRAIN_PLATFORM_HALF_DEPTH_M
            foot_bottom = fp[:,:,2] - .018  # 18mm ball radius
            foot_clearance=foot_bottom-upper_z[:,None]
        opposite_stance = jnp.flip(floor_pairs, axis=1)
        reachable = ((~floor_pairs) & opposite_stance &
                     (foot_clearance>.004) &
                     (jnp.abs(foot_lateral)<.22))
        reach = jnp.where(reachable, jnp.clip(foot_forward+.06, 0., .12), 0.)
        best_reach = jnp.max(reach, axis=1)
        eligible_foot = healthy & (forward<.06) & (forward>-.11)
        foot_peak = jnp.where(
            eligible_foot, jnp.maximum(state.terrain_ascent_foot_peak, best_reach),
            state.terrain_ascent_foot_peak)
        foot_gain = foot_peak - state.terrain_ascent_foot_peak
        return 30.*root_gain + 20.*foot_gain, root_peak, foot_peak

    def _metrics(self, data, task, action, prev_action, phase, prev_foot_pos):
        """Measure physical facts (not a score or the deployable actor packet).

        This dictionary holds the body/foot truth needed to calculate the
        teaching reward and independently inspect locomotion. Short-lived
        reward-only inputs are stripped before host-side rollout reporting.
        No curriculum decisions or learning updates occur here.
        """
        jnp = self.jnp
        # 1. Body motion and posture relative to the physical world.
        projected_gravity, body_vel, body_ang = self._body_packets(data)
        gravity_z = -projected_gravity[:, 2]
        # Gravity-relative chassis pitch: this remains meaningful on a slope and
        # is exactly the operator semantics used by the browser/physical control
        # surface. Positive command requests nose-up/back lean relative to local
        # gravity, not relative to the world's nominal flat plane.
        # v5015 movement targets are horizontal world motion relative to
        # the robot's ground-heading, NOT body-local X/Y speed. Otherwise
        # the same walk earns a different vx score as the camera tilts.
        if self.attitude_profile == 'v5015':
            motion_vx, motion_vy, motion_yaw = horizontal_tracking_motion(
                jnp, self._rotation_body_to_world(data),
                data.qvel[:, 0:3], data.qvel[:, 3:6])
        else:
            motion_vx, motion_vy, motion_yaw = body_vel[:, 0], body_vel[:, 1], body_ang[:, 2]
        body_pitch = jnp.arctan2(projected_gravity[:, 0], -projected_gravity[:, 2])
        body_roll = jnp.arctan2(-projected_gravity[:, 1], -projected_gravity[:, 2])
        pitch_err = body_pitch - task[:, T.pitch]
        roll_err = body_roll - task[:, T.roll]
        foot_pos = data.xpos[:, self.foot_bodies, :]
        # Height means clearance above the actual contact plane, not world Z.
        # Otherwise a robot walking uphill appears higher and its grounded
        # uphill foot can be incorrectly rewarded as a swing foot.
        height = self._support_signed_heights(data,data.xpos[:,self.torso_id:self.torso_id+1,:])[:,0]
        foot_ground_distance = self._support_signed_heights(data,foot_pos)
        foot_mid = jnp.mean(foot_pos, axis=1)
        torso_pos = data.xpos[:, self.torso_id, :]
        torso_to_feet = torso_pos - foot_mid
        horiz_to_feet = jnp.linalg.norm(torso_to_feet[:, :2], axis=1)
        vertical_to_feet = jnp.maximum(0.025, torso_to_feet[:, 2])
        foot_mid_angle = jnp.arctan2(horiz_to_feet, vertical_to_feet)
        foot_mid_cone = jnp.clip((0.4363323 - foot_mid_angle) / 0.4363323, 0.0, 1.0)
        # Robot lean comes from the feet-to-chassis line, never the IMU pitch.
        # Both feet count even in swing: this is a diagnostic, not a contact polygon.
        torso_R = self._rotation_body_to_world(data)
        heading_xy = torso_R[:, :2, 0]
        heading_xy = heading_xy / jnp.maximum(1e-6, jnp.linalg.norm(heading_xy, axis=1, keepdims=True))
        delta_xy = torso_to_feet[:, :2]
        robot_lean_forward = jnp.arctan2(jnp.sum(delta_xy * heading_xy, axis=1), torso_to_feet[:, 2])
        robot_lean_side = jnp.arctan2(-delta_xy[:, 0] * heading_xy[:, 1] + delta_xy[:, 1] * heading_xy[:, 0], torso_to_feet[:, 2])
        height_error = height - (NEUTRAL_ROOT_Z + task[:, T.height])
        # v5015: healthy support and requested attitude, not a level camera.
        posture_ok_5015 = ((foot_mid_angle < 0.6108652) &
                            (jnp.abs(pitch_err) < 0.3141593) &
                            (jnp.abs(roll_err) < 0.2792527) &
                            (height_error > -0.040))
        posture_ok = (posture_ok_5015 if self.attitude_profile == 'v5015'
                      else (-projected_gravity[:, 2] > 0.93969262))
        attitude_quality = jnp.exp(-8.0 * (pitch_err * pitch_err + roll_err * roll_err))
        # 2. Contact, gait and fall proxies. Foot height indicates nearby
        # ground; it is NOT a verified load-bearing contact impulse.
        foot_contacts = jnp.sum((foot_ground_distance < 0.036).astype(jnp.float32), axis=1)
        support = jnp.clip(foot_contacts / 2.0, 0.0, 1.0)
        bad_z = self._support_signed_heights(data,data.xpos[:,self.bad_bodies,:])
        # This model has no knee/shin sentinel contact geoms, so a blunt
        # body-origin proxy alone is too strict a termination condition --
        # only actual height/tilt/rate collapse may terminate. Body-low
        # stays visible as a pressure signal (below) so strict validation
        # could be reintroduced later without poisoning early training.
        body_low_soft = jnp.any(bad_z[:, 1:] < SOFT_BODY_LOW_Z, axis=1).astype(jnp.float32)
        low_height_soft = (height < 0.130).astype(jnp.float32)
        bad_contact = jnp.maximum(BODY_LOW_REWARD_WEIGHT * body_low_soft, low_height_soft)
        swing_clearance = jnp.maximum(0.0, jnp.max(foot_ground_distance, axis=1) - 0.018)
        q = data.qpos[:, self.act_qpos]
        joint_pressure = jnp.mean(jnp.maximum(0.0, jnp.abs(q - self.home_target) / (self.target_span * 1.20) - 0.72), axis=1)
        scissor = jnp.maximum(0.0, jnp.abs(q[:, 0] - q[:, 3]) - 0.62)
        action_delta = jnp.linalg.norm(action - prev_action, axis=1)
        # 3. Servo/spring diagnostics: measured mechanical proxies, NOT
        # payments. PPO currently penalises action changes and joint-limit
        # pressure instead of direct torque, shaft power or electrical usage.
        # Passive telemetry only. It never enters the deployable 220-input
        # observation or the scalar PPO reward.
        servo_effort = servo_effort_metrics(
            jnp, data.actuator_force[:, :ACTION_DIM],
            data.qvel[:, self.act_qvel], self.servo_stall_torque_nm)
        spring_energy = joint_spring_energy(
            jnp, data.qpos[:, self.spring_qpos],
            self.spring_reference, self.spring_stiffness)
        # Reward-only stance-foot slip proxy. MuJoCo truth is allowed in the
        # reward: consecutive
        # foot poses estimate tangential speed and only near-floor feet count.
        foot_vel_xy = (data.xpos[:, self.foot_bodies, :2] - prev_foot_pos[:, :, :2]) / CONTROL_DT
        foot_speed = jnp.linalg.norm(foot_vel_xy, axis=2)
        stance_w = jnp.clip((0.040 - foot_ground_distance) / 0.020, 0.0, 1.0)
        stance_slip = jnp.sum(foot_speed * stance_w, axis=1) / jnp.maximum(1.0, jnp.sum(stance_w, axis=1))
        # 4. Termination conditions and estimated stepping evidence.
        fall_height = height < 0.145
        fall_tilt = gravity_z < 0.35
        fall_roll_rate = jnp.abs(body_ang[:, 0]) > 9.0
        fall_pitch_rate = jnp.abs(body_ang[:, 1]) > 9.0
        terminal = jnp.logical_or(fall_height, fall_tilt)
        # Do not terminate on the body-low proxy in fast seed training.
        terminal = jnp.logical_or(terminal, fall_roll_rate)
        terminal = jnp.logical_or(terminal, fall_pitch_rate)
        s_phase = jnp.sin(phase)
        r_expected = jnp.maximum(0.0, s_phase)
        l_expected = jnp.maximum(0.0, -s_phase)
        r_lift = jnp.clip((foot_ground_distance[:, 0] - 0.018) / jnp.maximum(0.008, task[:, T.step_height]), 0.0, 1.0)
        l_lift = jnp.clip((foot_ground_distance[:, 1] - 0.018) / jnp.maximum(0.008, task[:, T.step_height]), 0.0, 1.0)
        phase_sync = jnp.clip(1.0 - 0.5 * (jnp.abs(r_lift - r_expected) + jnp.abs(l_lift - l_expected)), 0.0, 1.0)
        step_evidence = jnp.clip((jnp.maximum(r_lift, l_lift) - 0.10) / 0.55, 0.0, 1.0)
        alternating_lift = jnp.abs(r_lift - l_lift)
        organic_step_shape = jnp.clip(0.50 * jnp.maximum(r_lift, l_lift) + 0.30 * phase_sync * step_evidence + 0.20 * alternating_lift, 0.0, 1.0)
        # 5. Publish measurements in the existing stable field names.
        # rewards.py owns the list of short-lived fields to omit from the
        # post-step telemetry dictionary; diagnostics keep the others.
        return {
            'terminal': terminal.astype(jnp.float32),
            'fall_height': fall_height.astype(jnp.float32),
            'fall_tilt': fall_tilt.astype(jnp.float32),
            'fall_roll_rate': fall_roll_rate.astype(jnp.float32),
            'fall_pitch_rate': fall_pitch_rate.astype(jnp.float32),
            'regime_steady': (task[:, T.regime] < 0.5).astype(jnp.float32),
            'regime_transition': jnp.logical_and(task[:, T.regime] > 0.5, task[:, T.regime] < 1.5).astype(jnp.float32),
            'regime_resilience': (task[:, T.regime] > 1.5).astype(jnp.float32),
            'athletic_rehearsal': task[:, T.athletic_rehearsal],
            'field_taught': task[:, T.field_taught],
            'regime_steady_frontier': jnp.logical_and(task[:, T.regime] < 0.5, task[:, T.athletic_rehearsal] < 0.5).astype(jnp.float32),
            'regime_transition_frontier': jnp.logical_and(jnp.logical_and(task[:, T.regime] > 0.5, task[:, T.regime] < 1.5), task[:, T.athletic_rehearsal] < 0.5).astype(jnp.float32),
            'regime_resilience_frontier': jnp.logical_and(task[:, T.regime] > 1.5, task[:, T.athletic_rehearsal] < 0.5).astype(jnp.float32),
            'birth_steady_d': task[:, T.birth_steady_d],
            'birth_transition_d': task[:, T.birth_transition_d],
            'birth_resilience_d': task[:, T.birth_resilience_d],
            'birth_posture_d': task[:, T.birth_posture_d],
            'vx': motion_vx, 'vy': motion_vy, 'yaw_rate': motion_yaw,
            'cmd_vx': task[:, T.vx], 'cmd_vy': task[:, T.vy], 'cmd_yaw': task[:, T.yaw],
            'cmd_pitch': task[:, T.pitch], 'body_pitch': body_pitch,
            'cmd_roll': task[:, T.roll], 'body_roll': body_roll,
            'robot_lean_forward': robot_lean_forward,
            'robot_lean_side': robot_lean_side,
            'structural_attitude_ok': posture_ok.astype(jnp.float32),
            'attitude_quality': attitude_quality,
            'height': height, 'target_height': NEUTRAL_ROOT_Z + task[:, T.height],
            # Signed measured-minus-commanded posture error; negative is too low.
            # Diagnostic only: no change to PPO observation, action or reward.
            'height_error': height_error,
            'gravity_z': gravity_z,
            'foot_mid_angle_deg': foot_mid_angle * 57.29578,
            'foot_mid_cone': foot_mid_cone,
            'phase_sync': phase_sync,
            'step_evidence': step_evidence,
            'organic_step_shape': organic_step_shape,
            'bad_contact': bad_contact, 'support': support,
            'stance_slip': stance_slip,
            'swing_clearance': swing_clearance, 'target_step_height': jnp.maximum(0.008, task[:, T.step_height]),
            'action_delta_norm': action_delta,
            'joint_limit_pressure': joint_pressure,
            'scissor_pressure': scissor,
            **servo_effort,
            **spring_energy,
        }

    def _base_obs(self, data, task, prev_action, phase, delivered_ctrl, sensor_state=None):
        """Deployable actor observation: sensors/estimators, not teacher truth."""
        jnp = self.jnp
        projected_gravity, body_vel, body_ang = self._body_packets(data)
        q = data.qpos[:, self.act_qpos]
        qv = data.qvel[:, self.act_qvel]
        shin_flex = data.qpos[:, self.shin_qpos]
        shin_rate = data.qvel[:, self.shin_qvel]
        n0 = jnp.sin(1.73 * phase + 0.6180339 * self.env_index)
        n1 = jnp.sin(2.31 * phase + 1.4142136 * self.env_index + 0.7)
        ns = jnp.where(task[:, T.sensor_scale] >= 0.0, task[:, T.sensor_scale], self.sensor_noise_scale)
        grav_est = projected_gravity + ns[:, None] * 0.008 * jnp.stack([n0, n1, -0.5*n0], axis=1)
        gyro_est = body_ang + ns[:, None] * 0.025 * jnp.stack([n1, n0, -n1], axis=1)
        q_est = q + ns[:, None] * 0.0015 * n0[:, None]
        qv_est = qv + ns[:, None] * 0.015 * n1[:, None]
        R = self._rotation_body_to_world(data)
        RT = jnp.swapaxes(R, 1, 2)
        torso = data.xpos[:, self.torso_id, :]
        foot_rel_world = data.xpos[:, self.foot_bodies, :] - torso[:, None, :]
        foot_rel = jnp.einsum('nij,nkj->nki', RT, foot_rel_world).reshape((self.num_envs, 6))
        # Simulator counterpart of a fused optical-flow + ToF + IMU/leg estimate.
        # Exact root vxy is never handed straight to the actor.
        vel_est = body_vel[:, :2] + ns[:, None] * 0.018 * jnp.stack([n0, n1], axis=1)
        # Approximate height-sensor reading over the physical ground plane.
        # The sensor noise remains the legacy observer model; this is not
        # evidence that hardware can recover true floor geometry perfectly.
        ground_height = self._floor_signed_heights(data,torso[:,None,:])
        tof_height = ground_height + ns[:, None] * 0.0025 * n1[:, None]
        vel_conf = jnp.clip(1.0 - 0.18*ns, 0.2, 1.0)[:, None].astype(jnp.float32)
        shin_flex_est = shin_flex + ns[:, None] * 0.0010 * jnp.stack([n0, n1], axis=1)
        shin_rate_est = shin_rate + ns[:, None] * 0.010 * jnp.stack([n1, n0], axis=1)
        shin_load_est = jnp.clip(jnp.abs(shin_flex_est) / 0.10, 0.0, 1.5)
        cmd = task[:, [T.vx, T.vy, T.yaw, T.height, T.pitch, T.roll, T.step_height]]
        phase_pack = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=1)
        base = jnp.concatenate([grav_est, gyro_est, q_est, qv_est, prev_action, delivered_ctrl,
                                foot_rel, vel_est, tof_height, vel_conf, cmd, phase_pack,
                                shin_flex_est, shin_rate_est, shin_load_est], axis=1)
        if self.observation_schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN):
            # No direct access to simulated true velocity, floor normal or
            # even instantaneous raw ToF/flow readings: the persistent sensor
            # ring owns acquisition cadence, delay, dropout and gyro bias.
            if sensor_state is None:
                raise ValueError('sim2real_232 requires a sensor state')
            sensor = self._sensor_readout(sensor_state, grav_est)
            vel_i = BASE_OBS_FIELDS.index('vel_x')
            height_i = BASE_OBS_FIELDS.index('tof_height')
            conf_i = BASE_OBS_FIELDS.index('vel_confidence')
            base = (base.at[:, 3:6].set(sensor['measured_gyro_body_rad_s'])
                        .at[:, vel_i:vel_i+2].set(sensor['velocity_body_xy_mps'])
                        .at[:, height_i].set(sensor['tof_estimated_depth_m'])
                        .at[:, conf_i].set(sensor['velocity_confidence']))
            terrain = jnp.stack((
                sensor['ground_forward_grade_rad'],
                sensor['ground_sideways_grade_rad'],
                sensor['ground_visibility_confidence']), axis=1)
            base = jnp.concatenate((base, terrain), axis=1)
            if self.observation_schema == OBS_SCHEMA_TERRAIN:
                base = jnp.concatenate((base,sensor['terrain_ranges_m'],
                    sensor['terrain_valid'].astype(jnp.float32)),axis=1)
        assert base.shape[1] == self.base_obs_dim, (
            f"_base_obs produced {base.shape[1]} columns, "
            f"{self.observation_schema} requires {self.base_obs_dim}")
        return jnp.nan_to_num(base, nan=0.0, posinf=10.0, neginf=-10.0).astype(jnp.float32)
