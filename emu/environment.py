"""The simulated universe: robot physics, task packets and sensors.

`EmuMJXEnv` advances many simultaneous MuJoCo worlds and supplies physical
observations to PPO. The robot's joints, springs and motor limits are defined
in robot.xml. Per-world model variation has one owner in domain_randomization.py.

The TaskSampler below generates lessons for this environment
without duplicating JAX, configuration or RNG state. This file owns the
physical world: reset, motor delivery, disturbances, stepping, sensor packets
and measured events. rewards.py owns the scalar teaching payment.

Read from top to bottom:
  1. Fixed robot/contact contracts, terrain XML construction and state packets.
  2. TaskSampler: teacher requests and command transitions (NOT actor truth).
  3. EmuMJXEnv: the single batched reset/physics/observation implementation.
     Within it: motor/physics -> lesson/world reset -> sensor simulation
     -> supported terrain evidence -> reward facts and deployable observations.
No hidden alternate physics engine and no per-experiment environment subclass.

Teacher-only task fields never become privileged actor sensor channels.
See README.md for the teaching-to-physics overview.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple
import math
import xml.etree.ElementTree as ET
import numpy as np

from .domain_randomization import DomainRandomizationConfig, build_batched_model
from .rewards import (training_reward, apply_early_fall_penalty,
                      REWARD_ONLY_METRICS)
from .capability_focus import (focus_indices, diamond_commands,
                               control_effort_weights, COMMAND, RESILIENCE,
                               SLOPE, STEP, TRANSITION, CONTROL,
                               NONFOCUS_STRENGTH, CONTROL_STRENGTH)
from .terrain_carpet_mjx import (
    update_tile_mocaps, validate_prism_only_geoms,
    add_carpet_geoms, BODY_NAMES as CARPET_BODY_NAMES,
    GEOM_NAMES as CARPET_GEOM_NAMES, TILE_COUNT as CARPET_TILE_COUNT,
    TILE_M as CARPET_TILE_M, TILE_SIZE_Z_M as CARPET_SIZE_Z_M,
    midpoint_anchor as carpet_midpoint_anchor,
    PROFILE_HALF as CARPET_PROFILE_HALF,
    role_envelope as carpet_role_envelope, make_profiles as carpet_make_profiles,
    exact_exam_lip_profiles as carpet_exam_lip_profiles,
    generate_window as carpet_generate_window, sample_surface as carpet_sample_surface,
    ray_surface as carpet_ray_surface, VISIT_HISTORY as CARPET_VISIT_HISTORY,
    unseen_tile as carpet_unseen_tile, note_confirmed_tile as carpet_note_confirmed_tile,
    signed_height_over_support as carpet_signed_support,
    supported_tile_foot_pairs as carpet_top_foot_pairs,
    shares_physical_edge as carpet_shared_edge,
)
from .carpet_prism_mjx import (
    batch_update_prism_mocaps,
    PRISM_COUNT, PRISM_BODY_NAMES, PRISM_GEOM_NAMES, PRISM_DEPTH_M, PrismMeshBuilder)
from .directional import (fit_forward_array_ground,
                          simulate_downward_optical_flow, estimate_flow_velocity,
)
from .contracts import (
    SIM_DT, CONTROL_SUBSTEPS, CONTROL_DT, ACTION_DIM, ACTUATED_JOINTS,
    HISTORY_LEN, OBS_SCHEMA_TERRAIN, TERRAIN_TOF_PITCH_DEG,
    TERRAIN_TOF_YAW_DEG, TERRAIN_TOF_MIN_M,
    TERRAIN_TOF_MAX_M, actor_observation_fields, sensor_observation_contract, COMBO_P,
    PHASE_RATE_MAX_HZ, FIELD_EDGE_SCALES, SEED_VX_LO, SEED_VX_HI,
    FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS, FIELD_NODE_STRIDE, FIELD_EDGE_STRIDE,
    FIELD_NODE_START, FIELD_EDGE_START, TASK_PARAM_DIM, TERRAIN_TASK_FIELDS,
    TERRAIN_TASK_PARAM_DIM, UNIFIED_TERRAIN_FIELDS, UNIFIED_TASK_PARAM_DIM,
    TASK_PARAM_FIELDS, P
)

# Deployable actor observation excludes training-only
# teacher signals (regime id, exact contact/fall classifiers, phase grading and
# step-evidence composites) from the policy input. Every remaining field has a
# plausible real-robot source: IMU, servo telemetry, forward kinematics, shin
# flex Hall sensors, optical-flow/ToF velocity estimation, command/controller
# state. Reward/evaluation may still use perfect MuJoCo truth.
#
# Channel order is fixed by the 360-input terrain sensor observation contract.
# A fixed-size four-frame history models measurement and actuation latency;
# its JIT shape cannot vary between training and the frozen examiner.
FOOT_BODIES = ['r_foot', 'l_foot']
BAD_BODIES = ['torso', 'r_thigh', 'l_thigh', 'r_shin', 'l_shin']
TARGET_SPAN = np.array([0.38, 0.48, 0.58, 0.38, 0.48, 0.58], dtype=np.float32)
RESET_JOINT_NOISE = 0.004
RESET_QVEL_NOISE = 0.003
RESET_FORWARD_VELOCITY_BLEND = 1.0
from .physical_variants import neutral_hip_height
NEUTRAL_ROOT_Z = neutral_hip_height()  # exact selected robot geometry, not 26cm hardcode
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


class EnvState(NamedTuple):
    data: Any
    hist: Any
    rng: Any
    task: Any
    specialist_active: Any  # teacher-only birth focus, not an actor observation
    focus_id: Any  # sampled per-episode teaching identity; never an actor input
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
    terrain_crossed: Any  # episode-level confirmed crossing evidence
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
    sensor_state: Any
    velocity_i_integral: Any  # two controller axes, not actor inputs
    raw_velocity_command: Any  # true operator request
    # NEW v5030 reward-only rolling 25-frame history (never actor input).
    # This is a fresh source/checkpoint shape: no cross-SHA resume.
    vx_reward_history: Any
    cmd_vx_reward_history: Any
    vx_reward_count: Any
    # Teacher-only rolling three-platform progress; lip count is unbounded.
    academy_active: Any
    academy_lip_index: Any
    academy_streak: Any
    academy_crossings: Any
    academy_up_crossings: Any
    academy_down_crossings: Any
    academy_foothold_seen: Any
    # v5051 five-frame simultaneous bilateral real-top support after a lip.
    academy_transfer_streak: Any
    academy_transfer_seen: Any
    academy_transfer_armed: Any  # true only after verified novel signed lip
    # One-time 50cm post-transfer progress check in original heading.
    academy_progress_origin_xy: Any
    academy_progress_heading_rad: Any
    academy_progress_active: Any
    academy_progress_paid: Any
    # One physical attempt verdict per lip. No repeated near-edge farming.
    academy_approach_seen: Any
    academy_approach_lip_m: Any  # signed physical lip retained until failure verdict
    academy_approach_frontier_m: Any  # its difficulty reference at approach time
    terrain_attempt_lip_m: Any  # first ordinary STEP encounter retained to episode end
    terrain_attempt_frontier_m: Any  # frozen severity denominator for that attempt
    terrain_attempt_tile_xy: Any  # exact first approached target tile ID
    terrain_first_cross_verified: Any  # success is tied to THAT first tile
    academy_birth_lip_m: Any  # actual FIRST +X seam signed height, sampled at reset
    academy_birth_frontier_m: Any  # earned up/down height WHEN the lip was generated
    academy_finish_streak: Any  # three safe frames after BOTH verified upper feet
    # v5052 fixed world-frame target beyond FIRST upcoming real tile lip.
    # Teacher-only episode state: never present in the 360 actor packet.
    academy_goal_xy: Any
    academy_goal_lip_armed: Any
    academy_goal_foot_streak: Any
    academy_goal_foot_seen: Any
    academy_goal_transfer_ready: Any
    academy_goal_stable_streak: Any
    academy_goal_completed: Any
    # v5041 physics-only rolling tile state. All four mutable tiles
    # belong to this same PPO world batch, not a second simulator.
    carpet_hfields: Any
    carpet_center: Any
    carpet_seed: Any
    carpet_xprofile: Any
    carpet_yprofile: Any
    carpet_up: Any
    carpet_down: Any
    carpet_slope: Any
    carpet_confirmed: Any
    carpet_seen_cells: Any  # teacher anti-rocking: last 64 confirmed positions
    carpet_seen_write: Any


class SensorSample(NamedTuple):
    """One real 4x4 range/flow/gyro snapshot, not a tripod emulation."""
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
from .physical_variants import (
    SERVO_RATED_CONTINUOUS_TORQUE_NM,
    SERVO_PEAK_STALL_TORQUE_NM,
    SERVO_NOLOAD_RAD_S,
)


def servo_effort_metrics(xp, torque_nm, joint_rate_rad_s, actuator_limit_nm):
    """Six-actuator nominal shaft demand from MuJoCo actuator force and qvel.

    Shapes: [worlds, 6] inputs -> [worlds] named diagnostics.
    Drive/brake are mechanical power sign splits; braking isn't regeneration.
    Hardware rated torque is now a HARD MuJoCo/MJX actuator force limit.
    Nonzero over-rated exposure means the physical contract is broken.
    """
    power = torque_nm * joint_rate_rad_s
    torque_abs = xp.abs(torque_nm)
    speed_abs = xp.abs(joint_rate_rad_s)
    # READ-ONLY: the XML enforces rated torque, not an electric speed curve.
    # A passively driven/backdriven joint can exceed a motor's no-load speed;
    # this is a reality-check warning, NOT a calibrated motor failure.
    speed_ratio = speed_abs / SERVO_NOLOAD_RAD_S
    torque_ratio = torque_abs / xp.maximum(actuator_limit_nm, 1e-6)
    # The speculative linear electrical envelope uses the datasheet STALL
    # intercept, NOT the new hard rated actuator limit. Neither the
    # datasheet line nor the speed warning controls motor power.
    electrical_stall_ratio = torque_abs / SERVO_PEAK_STALL_TORQUE_NM
    return {
        'servo_shaft_abs_power_w': xp.sum(xp.abs(power), axis=1),
        'servo_shaft_drive_power_w': xp.sum(xp.maximum(power, 0.0), axis=1),
        'servo_shaft_brake_power_w': xp.sum(xp.maximum(-power, 0.0), axis=1),
        'servo_over_rated_frac': xp.mean((torque_abs > SERVO_RATED_CONTINUOUS_TORQUE_NM + 1e-6).astype(torque_nm.dtype), axis=1),
        'servo_peak_torque_limit_frac': xp.max(torque_ratio, axis=1),
        'servo_speed_over_noload_frac': xp.mean(
            (speed_ratio > 1.).astype(torque_nm.dtype), axis=1),
        'servo_peak_speed_ratio': xp.max(speed_ratio, axis=1),
        'servo_linear_envelope_warning_frac': xp.mean(
            ((electrical_stall_ratio + speed_ratio > 1.) &
             (power > 0.)).astype(torque_nm.dtype), axis=1),
    }


def joint_spring_energy(xp, joint_position_rad, spring_ref_rad, stiffness_nm_rad):
    """Stored elastic potential: sum(0.5 * stiffness * (q - springref)^2).

    Includes six driven hinge springs and both passive shin-flex springs,
    plus the extra two lateral carbon-flex axes when the v5059 body is used.
    This is a *state* in joules, not a net transfer from springs to motion.
    For the nominal XML model, stiffness is not domain-randomised.
    """
    energy = 0.5 * stiffness_nm_rad * (joint_position_rad - spring_ref_rad) ** 2
    return {
        'spring_potential_energy_j': xp.sum(energy, axis=1),
        'shin_spring_energy_j': xp.sum(energy[:, 6:], axis=1),
    }


# Commands/tasks share the same MJX physical-world state and RNG.

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



def velocity_i_step(xp,requested,observed,confidence,previous,integral,
                    *,dt=0.02,ramp_s=0.4,multiplier=2.,
                    forward_limit=.40,reverse_limit=.12,lateral_limit=.14):
    """Vectorized controller: normalized signed shortfall ramps 0->1 in ramp_s."""
    if not (dt>0 and .1<=ramp_s<=5. and 1.<=multiplier<=3. and
            min(forward_limit,reverse_limit,lateral_limit)>0):
        raise ValueError('invalid velocity-I parameters')
    if multiplier==1.:
        return requested,xp.zeros_like(integral)
    req=xp.asarray(requested)
    est=xp.asarray(observed)
    confid=xp.asarray(confidence)
    valid=(confid>=.35) & xp.all(xp.isfinite(est),axis=1)
    active=xp.abs(req)>=.025
    changed=(req*previous<0)|(xp.abs(req-previous)>xp.maximum(.05,.5*xp.abs(previous)))
    shortfall=xp.clip((req-est)*xp.sign(req)/xp.maximum(xp.abs(req),.04),-1.,1.)
    candidate=xp.clip(integral+shortfall*(dt/ramp_s),0.,1.)
    state=xp.where(changed|~active,0.,
                   xp.where(valid[:,None],candidate,integral*.90))
    demand=req*(1.+(multiplier-1.)*state)
    demand=xp.stack((xp.clip(demand[:,0],-reverse_limit,forward_limit),
                     xp.clip(demand[:,1],-lateral_limit,lateral_limit)),axis=-1)
    return demand,state


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


# Teacher-side commands, specialist lessons and ramp timing. This class
# samples requests; it never overrides the policy's six motor actions.
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
            # Genuine high-speed *steady* practice, not a second actor or hard
            # 0.60m/s ceiling: once the signed evidence frontier grows,
            # rehearsal can probe 15% beyond it. The floor supports it.
            athletic_hi=jnp.maximum(.60,1.15*params[P.frontier_vx_hi])
            # Teach forward speed without saturating the tracking reward
            # on simultaneous extreme turn/side requests. Broad commands
            # stay in COMMAND and TRANSITION specialists.
            vx_ath = jax.random.uniform(k_vx, (), minval=.06,
                                        maxval=athletic_hi)
            vy_ath = jax.random.uniform(k_vy, (), minval=-.05, maxval=.05)
            yaw_ath = jax.random.uniform(k_yaw, (), minval=-.25, maxval=.25)
            vx = jnp.where(is_athletic, vx_ath, vx)
            vy = jnp.where(is_athletic, vy_ath, vy)
            yaw = jnp.where(is_athletic, yaw_ath, yaw)
            height = jnp.where(is_athletic, 0.0, height)
            pitch = jnp.where(is_athletic, 0.0, pitch)
            roll = jnp.where(is_athletic, 0.0, roll)
            step_height = jnp.where(is_athletic, 0.012, step_height)
            resilience_task = jnp.where(is_athletic, 0.0, resilience_task)

            # Zero-command research births are reserved AFTER specialist
            # task assignment in _reset_with_keys. Here STEP/CONTROL command
            # rewriting would otherwise silently erase sampled stillness.
            # Capability-atlas command probes use exact physical commands,
            # bypassing random envelope sampling without changing the actor.
            vx = jnp.where(jnp.isfinite(params[P.probe_cmd_vx]), params[P.probe_cmd_vx], vx)
            vy = jnp.where(jnp.isfinite(params[P.probe_cmd_vy]), params[P.probe_cmd_vy], vy)
            yaw = jnp.where(jnp.isfinite(params[P.probe_cmd_yaw]), params[P.probe_cmd_yaw], yaw)
            # The actual rehearsal command must set the gait clock;
            # _sample_task calculated phase before the athletic override.
            phase_rate=jnp.where(is_athletic,
                command_phase_rate(vx,vy,self.vx_hi_max,
                                   self.phase_speed_gain,base_phase_rate),
                phase_rate)
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
        # The active 360-input policy ramps chassis pitch alongside the
        # three learned command-field axes (vx, vy, yaw).
        columns = [T.vx, T.vy, T.yaw, T.pitch]
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
        # Transition-focused episodes push command-change timing and
        # acceleration, not the maximum walking envelope at the same time.
        # Reduce the NEXT requested velocity too, not only its birth task.
        trans_focus=state.specialist_active & (state.focus_id==TRANSITION)
        centre=jnp.asarray((.04,0.,0.),dtype=jnp.float32)
        new_target=jnp.where(trans_focus[:,None],
                   centre+NONFOCUS_STRENGTH*(new_target-centre),
                   new_target)
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

# One production MJX environment, one six-servo robot, one vectorised JAX
# stepping path. Keep its physical model and deployment sensor contract here.
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
                 attitude_profile: str = 'v5015', pitch_command_max_deg: float = 8.0214,
                 motor_guidance_weight: float = 0.0,
                 observation_schema: str = OBS_SCHEMA_TERRAIN,
                 tof_period_frames: int = 2,
                 flow_period_frames: int = 1,
                 tof_delay_frames: int = 1,
                 flow_delay_frames: int = 1,
                 tof_dropout_prob: float = 0.01,
                 flow_dropout_prob: float = 0.02,
                 gyro_bias_std_rad_s: float = 0.015,
                 terrain_step_max_m: float = 0.0,
                 unified_capability_focus: bool = False,
                 velocity_i_ramp_s: float = 0.4,
                 velocity_i_multiplier: float = 1.0,
                 retention_share: float = 0.0,
                 terrain_lesson: str = 'mixed',
                 temporal_vx_reward_share: float = 0.0,
                 stillness_birth_frac: float = 0.0,
                 science_gate_mode: str = "legacy",
                 science_phase_lowcmd_scale: float = 1.0,
                 science_phase_sync_strength: float = 1.0,
                 science_crouch_allowance_m: float = 0.0,
                 science_stride_efficiency_weight: float = 0.0,
                 science_supported_cross_bonus: float = 0.0,
                 science_foothold_bonus: float = 0.0,
                 science_bilateral_support_bonus: float = 0.0,
                 science_followthrough_bonus: float = 0.0,
                 science_sensor_teacher_bonus: float = 0.0,
                 science_step_goal_weight: float = 0.0,
                 science_step_velocity_relief: float = 0.0,
                 science_step_approach_m: float = 0.0,
                 science_step_finish_on_transfer: bool = False,
                 academy_stream: bool = False,
                 terrain_carpet: bool = False,
                 carpet_transition_cap_deg: float = 3.0,
                 terrain_carpet_prisms: bool = False,
                 academy_height_cap_m: float = 0.150,
                 academy_course_episode_frames: int = 1200,
                 reset_forward_capacity: int = 128,
                 feet_only_contacts: bool = True,
                 solver_iterations: int | None = None,
                 solver_ls_iterations: int | None = None,
                 exam_fixture_steps_m=None,
                 exam_fixture_grades_deg=None,
                 exam_fixture_kicks_mps=None):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
        # The active simulator has ONE real 360-input, eight-prism topology.
        # Historical 220/232, box, finite-step and three-slab experiments
        # are preserved in Git, not executable compatibility branches.
        if not (terrain_carpet_prisms and terrain_carpet and
                unified_capability_focus and observation_schema==OBS_SCHEMA_TERRAIN):
            raise ValueError('Current EMU requires unified terrain_360 and true MJX prisms')

        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx
        # Read-only heterogeneous physical examination, not a trainable
        # curriculum knob. Training and old fixed exams leave this None.
        fixture=(exam_fixture_steps_m,exam_fixture_grades_deg,
                 exam_fixture_kicks_mps)
        if any(value is not None for value in fixture):
            if exam_fixture_steps_m is None:
                raise ValueError("Exam fixture needs explicit per-world signed heights")
            arrays=[np.asarray(value if value is not None else
                      np.zeros(num_envs),dtype=np.float32) for value in fixture]
            if any(v.shape!=(num_envs,) or not np.isfinite(v).all()
                   for v in arrays):
                raise ValueError("Exam fixture must have finite [num_envs] arrays")
            if (np.max(np.abs(arrays[0]))>.08 or
                np.max(np.abs(arrays[1]))>15. or
                np.max(np.abs(arrays[2]))>.6):
                raise ValueError("Exam physical step/grade/kick exceeds design envelope")
            self.exam_fixture_steps=jnp.asarray(arrays[0])
            self.exam_fixture_grades=jnp.asarray(arrays[1])
            self.exam_fixture_kicks=jnp.asarray(arrays[2])
        else:
            self.exam_fixture_steps=None
            self.exam_fixture_grades=None
            self.exam_fixture_kicks=None
        if attitude_profile != 'v5015':
            raise ValueError('The active 360-input prism trainer requires v5015 attitude')
        if not (0.0 < pitch_command_max_deg <= 45.0):
            raise ValueError('pitch command maximum must be within (0, 45] degrees')
        if not np.isfinite(motor_guidance_weight) or not 0.0 <= motor_guidance_weight <= 0.10:
            raise ValueError('motor_guidance_weight must be in [0, 0.10]')
        self.motor_guidance_weight = float(motor_guidance_weight)
        if not np.isfinite(stillness_birth_frac) or not 0. <= stillness_birth_frac <= .25:
            raise ValueError("stillness_birth_frac must be in [0,0.25]")
        self.stillness_birth_frac = float(stillness_birth_frac)
        if science_gate_mode not in ("legacy","zero-only","off"):
            raise ValueError("Invalid v5051 gate mode")
        limits=dict(science_phase_lowcmd_scale=(0.,1.),
                    science_phase_sync_strength=(0.,1.),
                    science_crouch_allowance_m=(0.,.05),
                    science_stride_efficiency_weight=(0.,.12),
                    science_supported_cross_bonus=(0.,15.),
                    science_foothold_bonus=(0.,8.),
                    science_bilateral_support_bonus=(0.,10.),
                    science_followthrough_bonus=(0.,15.),
                    science_sensor_teacher_bonus=(0.,8.),
                    science_step_goal_weight=(0.,.50),
                    science_step_velocity_relief=(0.,1.),
                    science_step_approach_m=(0.,.35))
        candidate=locals()
        for name,(lo,hi) in limits.items():
            value=float(candidate[name])
            if not np.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f"Invalid v5051 treatment {name}")
            setattr(self,name,value)
        if 0. < self.science_step_approach_m < .15:
            raise ValueError('near-lip Academy birth requires safe 0.15..0.35m or 0=off')
        if type(science_step_finish_on_transfer) is not bool:
            raise ValueError("step finish switch must be boolean")
        if science_step_finish_on_transfer and not academy_stream:
            raise ValueError("step finish requires streamed real-prism Academy")
        self.science_step_finish_on_transfer=science_step_finish_on_transfer
        self.science_gate_mode=science_gate_mode
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
        # There is only one deployable sensor geometry and one history width.
        self.sensor_contract = sensor_observation_contract({
            'tof_period_frames': tof_period_frames,
            'flow_period_frames': flow_period_frames,
            'tof_delay_frames': tof_delay_frames,
            'flow_delay_frames': flow_delay_frames,
            'tof_dropout_prob': self.tof_dropout_prob,
            'flow_dropout_prob': self.flow_dropout_prob,
            'gyro_bias_std_rad_s': self.gyro_bias_std_rad_s,
            'sensor_noise_scale': float(sensor_noise_scale),
        }, schema=OBS_SCHEMA_TERRAIN, array_only=True)
        self.attitude_profile = attitude_profile
        self.pitch_limit_rad = float(np.deg2rad(pitch_command_max_deg))
        self.num_envs = int(num_envs)
        # A compact batch of terminated worlds needs the full MJX forward
        # solve. Everybody else reuses the already-stepped physical state.
        # 0 opts into the old full-width reference implementation for A/B.
        if type(reset_forward_capacity) is not int or reset_forward_capacity < 0:
            raise ValueError('reset_forward_capacity must be a non-negative int')
        self.reset_forward_capacity = min(reset_forward_capacity, self.num_envs)
        if type(feet_only_contacts) is not bool:
            raise ValueError('feet_only_contacts must be bool')
        self.feet_only_contacts = feet_only_contacts
        self.unified_capability_focus = bool(unified_capability_focus)
        self.focus_id = jnp.asarray(focus_indices(self.num_envs))
        self.max_episode_steps = int(max_episode_steps)
        self.robot_xml = str(robot_xml)
        # Experimental G3 only. Default zero uses the exact old XML path,
        # collision topology and JAX shapes. Never activate mid-checkpoint.
        if not np.isfinite(terrain_step_max_m) or not 0 <= terrain_step_max_m <= .08:
            raise ValueError('terrain_step_max_m must be in [0, 0.08] metres')
        self.terrain_step_max_m = float(terrain_step_max_m)
        if type(academy_stream) is not bool or (
                academy_stream and not unified_capability_focus):
            raise ValueError('streaming Academy requires unified terrain_360')
        if not .003 <= academy_height_cap_m <= .25:
            raise ValueError('Academy frontier aspiration must be 3mm-25cm')
        if terrain_carpet and (not unified_capability_focus or
                observation_schema!=OBS_SCHEMA_TERRAIN or terrain_step_max_m<=0):
            raise ValueError('terrain carpet needs unified terrain_360 and physical topology')
        if (not np.isfinite(carpet_transition_cap_deg) or
                not 0.<=carpet_transition_cap_deg<=15.):
            raise ValueError('carpet transition-angle cap must be 0..15deg')
        self.carpet_transition_cap_deg=float(carpet_transition_cap_deg)
        if terrain_carpet_prisms and not terrain_carpet:
            raise ValueError('prism contacts require opt-in terrain_carpet')
        self.terrain_carpet=bool(terrain_carpet)
        self.terrain_carpet_prisms=bool(terrain_carpet_prisms)
        self.academy_stream=bool(academy_stream)
        self.academy_height_cap_m=float(academy_height_cap_m)
        if (type(academy_course_episode_frames) is not int or
            not 700<=academy_course_episode_frames<=1500):
            raise ValueError('Academy course episode time must be 700..1500 frames')
        self.academy_course_episode_frames=int(academy_course_episode_frames)
        # Solver caps are part of the physical training identity. This
        # experimental branch lets a run opt in explicitly, without changing
        # robot.xml or the default physics of existing source checkouts.
        # The same override is passed to all held-out exam worlds.
        for name,value in (("solver_iterations",solver_iterations),
                           ("solver_ls_iterations",solver_ls_iterations)):
            if value is not None and (type(value) is not int or not 1<=value<=100):
                raise ValueError(f"{name} must be None or integer 1..100")
        source_xml=ET.parse(self.robot_xml).getroot()
        if solver_iterations is not None or solver_ls_iterations is not None:
            option=source_xml.find("option")
            if option is None:
                option=ET.SubElement(source_xml,"option")
            if solver_iterations is not None:
                option.set("iterations",str(solver_iterations))
            if solver_ls_iterations is not None:
                option.set("ls_iterations",str(solver_ls_iterations))
        self.mj_model=mujoco.MjModel.from_xml_string(
            add_carpet_geoms(ET.tostring(source_xml,encoding='unicode'),
                             prisms=True,feet_only_contacts=self.feet_only_contacts))
        self.carpet_mocap_ids=tuple(int(self.mj_model.body_mocapid[
            mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,n)])
            for n in CARPET_BODY_NAMES)
        self.carpet_geom_ids=tuple(int(mujoco.mj_name2id(
            self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,n))
            for n in CARPET_GEOM_NAMES)
        if (any(v<0 for v in self.carpet_mocap_ids)
            or any(v<0 for v in self.carpet_geom_ids)):
            raise RuntimeError('Missing four independent source tile samples')
        self.carpet_prism_mocap_ids=tuple(int(
            self.mj_model.body_mocapid[self.mujoco.mj_name2id(
                self.mj_model,self.mujoco.mjtObj.mjOBJ_BODY,name)])
            for name in PRISM_BODY_NAMES)
        self.carpet_prism_geom_ids=tuple(int(self.mujoco.mj_name2id(
            self.mj_model,self.mujoco.mjtObj.mjOBJ_GEOM,name))
            for name in PRISM_GEOM_NAMES)
        if (len(self.carpet_prism_mocap_ids)!=PRISM_COUNT or
            any(v<0 for v in (*self.carpet_prism_mocap_ids,
                               *self.carpet_prism_geom_ids))):
            raise RuntimeError('8 moving convex prism geoms missing')
        if any(self.mj_model.geom_contype[g]!=4
               for g in self.carpet_prism_geom_ids):
            raise RuntimeError('Prisms must be physical bit-4 colliders')
        validate_prism_only_geoms(self.mujoco, self.mj_model)
        # Only two spherical feet may produce real terrain contact in
        # the reduced solver. The torso still has its physical mass/inertia,
        # but non-foot ground strikes are diagnosed geometrically below.
        torso_geom=mujoco.mj_name2id(
            self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'torso_box')
        self.torso_geom=torso_geom
        self.foot_radius_m = float(self.mj_model.geom_size[
            mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,
                               'r_foot_ball'),0])
        self.rod_radius_m = float(self.mj_model.geom_size[
            mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,
                               'r_shin_upper'),0])
        self.torso_half_extents=jnp.asarray(
            self.mj_model.geom_size[torso_geom,:3],dtype=jnp.float32)
        self.shin_lower_bodies=jnp.asarray([
            mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,n)
            for n in ('r_shin_lower','l_shin_lower')],dtype=jnp.int32)
        if self.feet_only_contacts:
            floor_geom=mujoco.mj_name2id(
                self.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
            if any(self.mj_model.geom_contype[g] or
                   self.mj_model.geom_conaffinity[g]
                   for g in (torso_geom,floor_geom)):
                raise RuntimeError('Only the feet may collide with the carpet')
        assert self.mj_model.nhfield==CARPET_TILE_COUNT,self.mj_model.nhfield
        assert list(self.mj_model.hfield_nrow)==[2]*CARPET_TILE_COUNT
        assert list(self.mj_model.hfield_ncol)==[2]*CARPET_TILE_COUNT
        self.mj_model.opt.timestep = SIM_DT
        # Terrain angles belong to the moving prisms, not an alternative
        # rotated-floor or tilted-gravity teaching world. Robot/servo
        # manufacturing randomization may still use batched model leaves.
        self.domain_config = (domain_config or DomainRandomizationConfig()).normalized()
        # Performance hint from MJX docs: keep contacts explicit/simple. The XML
        # owns the complete robot physics, including actuated-joint armature and
        # Coulomb friction; thin thigh/shin rods are deliberately non-contacting.
        self.mjx_model_template = mjx.put_model(self.mj_model)
        self.carpet_prism_builder=PrismMeshBuilder(
            self.mj_model,self.mjx_model_template,self.carpet_prism_geom_ids)
        self.ids = self._resolve_ids()
        self.mjx_model, self.model_in_axes, self.domain_info = build_batched_model(
            self.mjx_model_template, ids=self.ids, num_envs=self.num_envs, seed=int(seed),
            config=self.domain_config)
        h=self.mjx_model_template.hfield_data
        if h.shape!=(CARPET_TILE_COUNT*4,):
            raise RuntimeError(f'2-triangle carpet hfield has wrong storage: {h.shape}')
        self.mjx_model=self.mjx_model.tree_replace({
            'hfield_data':jnp.broadcast_to(h,(self.num_envs,h.shape[0]))})
        axes=(self.model_in_axes if self.model_in_axes is not None else
              jax.tree_util.tree_map(lambda _:None,self.mjx_model_template))
        self.model_in_axes=axes.tree_replace({'hfield_data':0})
        builder=self.carpet_prism_builder
        mesh=self.mjx_model._impl.mesh_convex
        batch=builder.batch_template(jnp,self.num_envs)
        merged=tuple(batch[i] if any(
            m['mesh_id']==i for m in builder.meta) else mesh[i]
            for i in range(len(mesh)))
        self.mjx_model=self.mjx_model.replace(
            _impl=self.mjx_model._impl.replace(mesh_convex=merged))
        self.model_in_axes=self.model_in_axes.replace(
            _impl=self.model_in_axes._impl.replace(
                mesh_convex=builder.in_axes(self.mjx_model_template)))
        gravity_unit = self.domain_info.get('gravity_unit_world')
        self.gravity_unit_world = (
            gravity_unit if gravity_unit is not None
            else jnp.tile(jnp.asarray([[0.0, 0.0, -1.0]], dtype=jnp.float32), (self.num_envs, 1))
        )
        # The underlying floor is flat. Actual slopes come from the eight
        # prism contacts; no synthetic gravity/plane incline is substituted.
        self.floor_normal_world = jnp.tile(
            jnp.asarray([[0., 0., 1.]], dtype=jnp.float32),
            (self.num_envs, 1))
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
        if (not math.isfinite(velocity_i_ramp_s) or
                not 0.1 <= velocity_i_ramp_s <= 5.0 or
                not math.isfinite(velocity_i_multiplier) or
                not 1.0 <= velocity_i_multiplier <= 3.0 or
                (velocity_i_multiplier > 1.0 and
                 (not unified_capability_focus or observation_schema != OBS_SCHEMA_TERRAIN))):
            raise ValueError('velocity I requires unified terrain_360, ramp 0.1..5s, multiplier 1..3')
        self.velocity_i_ramp_s = float(velocity_i_ramp_s)
        self.velocity_i_multiplier = float(velocity_i_multiplier)
        if terrain_lesson not in ('mixed','coherent','coherent-direct','coherent-careful') or not 0.<=retention_share<=.4:
            raise ValueError('invalid recovery-first experiment contract')
        self.terrain_lesson = terrain_lesson
        self.retention_share = float(retention_share)
        if (not np.isfinite(temporal_vx_reward_share)
                or not 0.0<=temporal_vx_reward_share<=1.0
                or (temporal_vx_reward_share>0 and not self.unified_capability_focus)):
            raise ValueError('temporal vx reward requires unified terrain and share 0..1')
        self.temporal_vx_reward_share=float(temporal_vx_reward_share)
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
        # Source-XML passive viscous torque-speed approximation, NOT motor
        # electrical power. It dissipates energy even at zero servo command.
        # Use the live MJX model's last dimension so this diagnostic is
        # also correct for per-world randomized damping, if enabled.
        # Shape is [6] for static motor physics or [world,6].
        self.act_joint_viscous_damping = jnp.take(
            self.mjx_model.dof_damping, self.act_qvel, axis=-1)
        # FAIL CLOSED if this source loads a peak-stall-torque XML, or
        # someone disables MuJoCo force limiting. This is a physics
        # invariant, not optional PPO shaping or a logging convention.
        limits = np.asarray(self.mj_model.actuator_forcerange, dtype=np.float32)
        limited = np.asarray(self.mj_model.actuator_forcelimited)
        if (limits.shape != (ACTION_DIM, 2) or
                limited.shape != (ACTION_DIM,) or
                not np.all(limited == 1) or
                not np.all(np.isfinite(limits)) or
                not np.all(limits[:, 0] < 0.) or
                not np.all(limits[:, 1] > 0.) or
                np.max(np.abs(limits)) > SERVO_RATED_CONTINUOUS_TORQUE_NM + 1e-5):
            raise ValueError('C018 actuator force contract violated: all six motors '
                             'must be force-limited to <= 0.980665 Nm')
        # Per-world domain randomization can weaken, never exceed rated.
        self.servo_actuator_torque_limit_nm = jnp.max(
            jnp.abs(self.mjx_model.actuator_forcerange), axis=-1)
        # Motor and passive rod springs: the carbon version additionally
        # bends laterally at the shin midpoint (without new actor channels).
        spring_names = (*ACTUATED_JOINTS, 'r_shin_flex', 'l_shin_flex')
        if mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_JOINT,
                             'r_shin_flex_roll') >= 0:
            spring_names += ('r_shin_flex_roll','l_shin_flex_roll')
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
        # Symmetric v5056 passive compression struts are spatial tendons, NOT
        # linear joint springs. Real tendon lengths are updated by MJX, and
        # stored energy must include those springs even when all servo power
        # is zero. Read physical stiffness/rest lengths from the compiled XML.
        knee_struts = ('r_knee_support', 'l_knee_support')
        tendon_ids = [mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_TENDON, name)
            for name in knee_struts]
        if any(t < 0 for t in tendon_ids):
            raise ValueError('Missing symmetric knee-compression struts in robot.xml')
        self.knee_strut_ids = jnp.asarray(tendon_ids, dtype=jnp.int32)
        self.knee_strut_k = jnp.asarray(
            np.asarray(self.mj_model.tendon_stiffness)[tendon_ids],
            dtype=jnp.float32)
        # MuJoCo springlength has TWO limits: for compression springs
        # both are equal, but [0, L] defines a tension-only cable.
        spring_limits=np.asarray(self.mj_model.tendon_lengthspring)[tendon_ids]
        self.knee_strut_rest_m = jnp.asarray(spring_limits[:,0],dtype=jnp.float32)
        self.knee_strut_rest_max_m = jnp.asarray(spring_limits[:,1],dtype=jnp.float32)
        flex_joint=mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_JOINT,
                                    'r_shin_flex')
        flex_max=float(self.mj_model.jnt_range[flex_joint,1])
        # The two existing Hall-load inputs are normalized measured bending
        # proxies, not privileged torque. Retain historic .10rad mapping.
        self.shin_sensor_fullscale_rad = (.10 if flex_max<=np.deg2rad(12.)
                                         else flex_max)
        self.act_joint_range = jnp.asarray(self.ids['act_joint_range'], dtype=jnp.float32)
        self.shin_qpos = jnp.asarray(self.ids['shin_qpos'], dtype=jnp.int32)
        self.shin_qvel = jnp.asarray(self.ids['shin_qvel'], dtype=jnp.int32)
        flex_names=('r_shin_flex','l_shin_flex')
        if mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_JOINT,
                             'r_shin_flex_roll')>=0:
            flex_names+=('r_shin_flex_roll','l_shin_flex_roll')
        flex_ids=[mujoco.mj_name2id(self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,name) for name in flex_names]
        self.shin_all_qpos=jnp.asarray(
            np.asarray(self.mj_model.jnt_qposadr)[flex_ids],dtype=jnp.int32)
        self.shin_all_limit_rad=jnp.asarray(
            np.max(np.abs(np.asarray(self.mj_model.jnt_range)[flex_ids]),axis=1),
            dtype=jnp.float32)
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
        # One complete parameter packet: control, terrain, six specialist weights.
        terrain = np.asarray(
            [float(params.get(name, default)) for name, default in TERRAIN_TASK_FIELDS],
            dtype=np.float32)
        if not (np.isfinite(terrain).all() and
                0.0 <= terrain[0] <= 1.0 and
                0.0 <= terrain[1] <= self.terrain_step_max_m):
            raise ValueError('terrain exposure must be [0,1] and height must be [0, terrain_step_max_m]')
        packed = np.concatenate((packed, terrain))
        assert packed.shape[0] == TERRAIN_TASK_PARAM_DIM
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
            # Float32 tan/atan round-trip makes a genuine 15-degree
            # single-axis grade appear as 15.000001. Tolerate
            # numerical error, never a meaningful over-limit slope.
            if not np.isfinite(g) or np.rad2deg(np.arctan(g))>15.0 + 1e-5:
                raise ValueError('physical terrain total inclination exceeds 15-degree exam cap')
        if np.isfinite(slope[4]) and not 0.<=slope[4]<=self.terrain_step_max_m:
            raise ValueError('fixed step height outside compiled physical range')
        if np.isfinite(slope[5]) and abs(slope[5])>45.0:
            raise ValueError('fixed edge yaw outside +/-45-degree exam range')
        if not np.isfinite(slope[6]) or not 0. <= slope[6] <= 1.:
            raise ValueError('specialist_mix must be between zero and one')
        if not np.isfinite(slope[7]) or not 0. <= slope[7] <= 1.:
            raise ValueError('foundation_control_mix must be between zero and one')
        focus_weights=slope[8:14]
        if (not np.isfinite(focus_weights).all() or
            (focus_weights<0.).any() or
            (focus_weights.sum()>0. and not np.isclose(
                focus_weights.sum(),1.,atol=1e-4))):
            raise ValueError('focus weights must be nonnegative and sum to one')
        if (not np.isfinite(slope[14]) or
            slope[14]<0. or slope[14]>1.):
            raise ValueError('steering attention must be within [0,1]')
        if (not np.isfinite(slope[15]) or
            slope[15]<0. or slope[15]>1.):
            raise ValueError('academy birth probability must be within [0,1]')
        if (not np.isfinite(slope[16:18]).all() or
            np.any(slope[16:18]<.001) or
            np.any(slope[16:18]>self.academy_height_cap_m)):
            raise ValueError('Academy UP/DOWN lip frontiers exceed physical compiled cap')
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
        # v5051 causal experiment: preserve valid unit-circle phase
        # channels and policy architecture while slowing the *clock* only
        # for genuine near-zero requests. Commanded walking is unchanged.
        phase_rate=state.task[:, T.phase_rate]
        if self.science_phase_lowcmd_scale < 1.:
            raw_cmd=state.raw_velocity_command
            low_speed=jnp.linalg.norm(raw_cmd,axis=1)
            fraction=jnp.clip(low_speed/.10,0.,1.)
            factor=(self.science_phase_lowcmd_scale +
                    (1.-self.science_phase_lowcmd_scale)*fraction)
            factor=jnp.where(jnp.abs(state.task[:,T.yaw])<=.10,
                             factor,1.)
            phase_rate=phase_rate*factor
        phase = (state.phase + CONTROL_DT * phase_rate * 2.0 * jnp.pi) % (2.0 * jnp.pi)

        # 2. The teacher advances the NEXT request; it cannot rewrite the
        # instruction the actor already saw when choosing this action.
        # Train and assess ORIGINAL operator velocities; the boosted
        # controller request is never passed to the reward/teacher.
        source_task = state.task
        if self.velocity_i_multiplier > 1.0:
            source_task = (source_task.at[:,T.vx].set(state.raw_velocity_command[:,0])
                                     .at[:,T.vy].set(state.raw_velocity_command[:,1]))
        lesson = self._advance_commands(
            state._replace(task=source_task) if self.velocity_i_multiplier > 1.0 else state,
            params, physics.resample_keys)
        task = lesson.task
        task_for_state = lesson.stored_task
        ramp_frac = lesson.ramp_frac
        transition_active = lesson.transition_active
        cmd_resampled = lesson.cmd_resampled
        physics_diverged = physics.diverged

        # 3. Measure the physical result, then reward the PREVIOUS command.
        # The ramped command is for the next observation, never this payment.
        reward_task = source_task
        metrics = self._metrics(data, reward_task, action, state.prev_action, phase,
                                state.prev_foot_pos,carpet_heights=state.carpet_hfields)
        # Reward-only true cohort identity: crouch tolerance cannot become
        # a free survival advantage for all flat/command/resilience worlds.
        step_focus=(state.specialist_active & (state.focus_id==STEP))
        # Fixed birth-to-destination goal: 60cm beyond first +X prism
        # boundary. No moving carrot, reset on every episode birth only.
        from .step_goal import signed_distance_progress
        goal_delta=signed_distance_progress(
            jnp,state.academy_goal_xy,
            state.data.qpos[:,:2],data.qpos[:,:2])
        metrics={**metrics,
                 'science_step_focus':step_focus.astype(jnp.float32),
                 # These TWO teacher-only fields never reach the actor.
                 'science_step_goal_active':(
                     step_focus & state.academy_active).astype(jnp.float32),
                 'science_step_goal_delta_m':goal_delta}
        # Gate freezing against previous activity, not this one stance frame.
        stepping_activity_so_far = state.ep_step_evidence_sum / jnp.maximum(1.0, state.ep_len)
        # ramp_frac tracks command-change difficulty. Real episode age is
        # ep_len; step_i is deliberately staggered for timeout diversity.
        metrics = {**metrics, 'age_s': (state.ep_len + 1.0) * CONTROL_DT,
                  'stepping_activity_so_far': stepping_activity_so_far, 'ramp_frac': ramp_frac}
        # v5030 B: exact trailing 25-frame 0.5s *reward-only* forward
        # history. Use PREVIOUS request; never contaminate with next task.
        # Clear at every transition/ramp so command changes don't look like
        # free success, and reset on episode birth via the state reset path.
        history_ready=(state.task[:,T.regime]<.5) & (
            state.cmd_ramp_frac>=.999) & ~physics_diverged
        slot=jnp.mod(state.step_i,25).astype(jnp.int32)
        idx=jnp.arange(self.num_envs)
        hist_v=jnp.where(history_ready[:,None],
                         state.vx_reward_history,0.)
        hist_c=jnp.where(history_ready[:,None],
                         state.cmd_vx_reward_history,0.)
        hist_v=hist_v.at[idx,slot].set(
            jnp.where(history_ready,metrics['vx'],0.))
        hist_c=hist_c.at[idx,slot].set(
            jnp.where(history_ready,metrics['cmd_vx'],0.))
        hist_n=jnp.where(history_ready,
                          jnp.minimum(state.vx_reward_count+1,25),0)
        temporal=(hist_v.sum(axis=1)/jnp.maximum(1,hist_n),
                  hist_c.sum(axis=1)/jnp.maximum(1,hist_n),
                  self.temporal_vx_reward_share*(
                      hist_n.astype(jnp.float32)/25.))
        reward, terms = training_reward(
            metrics, reward_task[:, T.regime],
            xp=jnp,
            temporal_vx=(temporal if self.temporal_vx_reward_share>0. else None),
            motor_guidance_weight=self.motor_guidance_weight,
            allow_zero_command_balance=self.stillness_birth_frac > 0.,
            science_gate_mode=self.science_gate_mode,
            science_phase_sync_strength=self.science_phase_sync_strength,
            science_crouch_allowance_m=self.science_crouch_allowance_m,
            science_stride_efficiency_weight=self.science_stride_efficiency_weight,
            science_step_goal_weight=self.science_step_goal_weight,
            science_step_velocity_relief=self.science_step_velocity_relief,
            motor_guidance_per_world=(jnp.where(
                    state.academy_active,0.,
                    control_effort_weights(jnp,
                      jnp.where(state.specialist_active,state.focus_id,-1)))
                if self.unified_capability_focus else None))
        # Academy traverses a repeating physical ring up to its own timeout,
        # while ordinary roles keep their 700-frame episode clock.
        timeout=jnp.where(state.academy_active,
            state.ep_len+1>=self.academy_course_episode_frames,
            step_i>=self.max_episode_steps)
        terminal_fail=jnp.logical_or(metrics['terminal']>.5,physics_diverged)
        terminal_or_timeout=jnp.logical_or(terminal_fail,timeout)
        academy=self._carpet_events(state,data,metrics,terminal_or_timeout,params)
        done=terminal_or_timeout
        episode_age_steps=state.ep_len+1.
        # Extending course TIME must not silently add +30 early-fall
        # penalty points and drown out novice ascent/foothold discovery.
        # Preserve the proven 700-frame gait penalty economics; ages beyond
        # 700 naturally receive the original flat base penalty only.
        reward,fall_payment=apply_early_fall_penalty(
            reward,terminal_fail,episode_age_steps,self.max_episode_steps,xp=jnp)

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
        # Real prisms credit each new supported lip, not only the first
        # crossing of a rolling-carpet episode.
        terrain_crossed = state.terrain_crossed | academy['ordinary_step_new']
        terrain_cross_new = academy['ordinary_step_new']
        reward=reward+10.0*terrain_cross_new.astype(jnp.float32)
        ep_return=ep_return+10.0*terrain_cross_new.astype(jnp.float32)
        reward=reward+academy['reward']
        ep_return=ep_return+academy['reward']
        # v5051: sparse event payments only; actual contact/top geometry
        # is evaluated before each event and duplicate tiles are guarded.
        # The bilateral event is *top support*, not measured CoM load.
        if (self.science_supported_cross_bonus > 0. or
                self.science_foothold_bonus > 0. or
                self.science_bilateral_support_bonus > 0. or
                self.science_followthrough_bonus > 0. or
                self.science_sensor_teacher_bonus > 0.):
            true_lip=(academy['ordinary_step_new'] | academy['cross_new'])
            extra=(self.science_supported_cross_bonus*
                   true_lip.astype(jnp.float32) +
                   self.science_foothold_bonus*
                   academy['foothold_new'].astype(jnp.float32) +
                   self.science_bilateral_support_bonus*
                   academy['transfer_new'].astype(jnp.float32) +
                   self.science_followthrough_bonus*
                   academy['progress_new'].astype(jnp.float32))
            if self.science_sensor_teacher_bonus > 0.:
                sensor_now=self._sensor_readout(state.sensor_state,
                                               self._body_packets(data)[0])
                valid=jnp.asarray(sensor_now['terrain_valid'],dtype=jnp.bool_)
                ranges=sensor_now['terrain_ranges_m']
                visible=jnp.any(valid & (ranges>.045) & (ranges<.70),axis=1)
                extra=extra+self.science_sensor_teacher_bonus*(
                    visible & academy['foothold_new']).astype(jnp.float32)
            reward=reward+extra
            ep_return=ep_return+extra

        # v5052: an ACTUAL goal task ends only at the fixed position
        # 60cm beyond the original lip, after true two-foot support transfer
        # plus five stable arrival frames. No instantaneous teleport or
        # physical-goal bonus from coordinate changes alone.
        # Destination coordinates are reward-only, never actor inputs.
        goal_transfer_ready=state.academy_goal_transfer_ready
        goal_lip_armed=state.academy_goal_lip_armed
        goal_foot_streak=state.academy_goal_foot_streak
        goal_foot_seen=state.academy_goal_foot_seen
        goal_stable_streak=state.academy_goal_stable_streak
        goal_completed=state.academy_goal_completed
        if self.science_step_goal_weight > 0. or self.science_step_finish_on_transfer:
            from .step_goal import (
                arrival_quality, stable_arrival, short_climb_quality,
                CLIMB_STABLE_FRAMES)
            rot=self._rotation_body_to_world(data)
            heading=jnp.arctan2(rot[:,1,0],rot[:,0,0])
            # The recovered feet must belong to the FIRST assigned
            # destination tile, not an arbitrary later lip in the carpet.
            destination_cell=jnp.floor(
                state.academy_goal_xy/CARPET_TILE_M).astype(jnp.int32)
            observed_cell=jnp.floor(
                data.qpos[:,:2]/CARPET_TILE_M).astype(jnp.int32)
            # A dynamic gait may never hold BOTH feet together for
            # five frames. Verify each independently after the SAME
            # authentically supported first lip. Neither a coordinate
            # crossing nor an unrelated later tile can arm this.
            in_goal_tile=jnp.all(observed_cell==destination_cell,axis=1)
            goal_lip_armed=(goal_lip_armed |
                            (academy['cross_new'] & in_goal_tile))
            foot_ok=(academy['goal_top_feet'] &
                     goal_lip_armed[:,None] & in_goal_tile[:,None])
            goal_foot_streak=jnp.where(
                foot_ok,goal_foot_streak+1,0)
            goal_foot_seen=(goal_foot_seen |
                            (goal_foot_streak>=5))
            goal_transfer_ready=(goal_transfer_ready |
                                 jnp.all(goal_foot_seen,axis=1))
            strict=arrival_quality(
                jnp,state.academy_goal_xy,data.qpos[:,:2],heading,
                metrics['gravity_z'],metrics['support'],
                metrics['bad_contact'],metrics['stance_slip'],
                goal_transfer_ready)
            strict=(strict & state.academy_active & step_focus &
                    ~terminal_fail & ~timeout)
            strict=strict & (self.science_step_goal_weight>0.)
            goal_stable_streak,goal_new=stable_arrival(
                jnp,strict,goal_stable_streak,goal_completed)
            goal_completed=goal_completed|goal_new
            # v5054: end the nursery lesson as soon as both feet have
            # independently held true first-tile top contact and the body
            # is safely 10cm past the *real* lip for three frames. Not a
            # one-foot touchdown, not a 60cm parking maneuver.
            short=short_climb_quality(
                jnp,state.academy_goal_xy,data.qpos[:,:2],heading,
                metrics['gravity_z'],metrics['support'],metrics['bad_contact'],
                goal_transfer_ready,state.academy_birth_lip_m)
            # At least one foot must STILL be physically supported
            # on the verified upper tile when we declare completion.
            short=(short & jnp.any(academy['goal_top_feet'],axis=1) &
                   goal_lip_armed & in_goal_tile &
                   state.academy_active & step_focus &
                   ~terminal_fail & ~timeout &
                   self.science_step_finish_on_transfer)
            finish_streak=jnp.where(
                short,state.academy_finish_streak+1,0)
            finish_new=finish_streak>=CLIMB_STABLE_FRAMES
            # Success is a TRUE absorbing PPO terminal, not a timeout.
            # A separate sparse payment distinguishes partial foothold
            # shaping from the completed whole-foot transfer lesson.
            bonus=(20.*self.science_step_goal_weight*
                   goal_new.astype(jnp.float32) +
                   8.*finish_new.astype(jnp.float32))
            reward=reward+bonus
            ep_return=ep_return+bonus
            done=done | goal_new | finish_new
        else:
            goal_new=jnp.zeros_like(done)
            finish_new=jnp.zeros_like(done)
            finish_streak=state.academy_finish_streak
        academy={**academy,'completed':goal_new,'finish_new':finish_new}

        # 4. Publish the NEXT actor observation; preserve the pre-reset
        # observation separately for PPO bootstrap.
        sensors = self._advance_sensor_state(
            state.sensor_state, data, task, phase, physics.next_keys,
            state.ep_len + 1, carpet_heights=state.carpet_hfields)
        integral_next = state.velocity_i_integral
        raw_next = task[:,[T.vx,T.vy]]
        if self.velocity_i_multiplier > 1.0:
            sensed = self._sensor_readout(sensors,self._body_packets(data)[0])
            commanded, integral_next = velocity_i_step(
                jnp,raw_next,sensed['velocity_body_xy_mps'],
                sensed['velocity_confidence'],state.raw_velocity_command,
                state.velocity_i_integral,dt=CONTROL_DT,
                ramp_s=self.velocity_i_ramp_s,
                multiplier=self.velocity_i_multiplier,
                forward_limit=max(self.vx_hi_max,1.4),
                reverse_limit=max(abs(self.vx_lo_min),.55),
                lateral_limit=max(self.vy_max_mps,.55))
            task_for_state=(task_for_state.at[:,T.vx].set(commanded[:,0])
                                          .at[:,T.vy].set(commanded[:,1]))
        base = self._base_obs(data, task_for_state, action, phase, ctrl, sensors)
        hist = jnp.concatenate([state.hist[:, 1:, :], base[:, None, :]], axis=1)
        bootstrap_obs = hist.reshape((self.num_envs, self.obs_dim))
        # Recycle the physical prism tiles only after both feet clear.
        data,carpet_hfields,carpet_center=self._carpet_recycle(state,data)
        # carryover_qvel is this step's own (already divergence-safe)
        # qvel -- see the docstring on _reset_with_keys.
        reset_state = self._reset_with_keys(
            physics.reset_keys, params, carryover_qvel=data.qvel,
            reset_mask=done)

        def choose(reset_val, step_val):
            if hasattr(step_val, 'shape') and len(step_val.shape) > 0 and step_val.shape[0] == self.num_envs:
                mask = done.reshape((self.num_envs,) + (1,) * (len(step_val.shape) - 1))
                return jnp.where(mask, reset_val, step_val)
            return step_val

        stepped = state._replace(
            data=data,
            hist=hist,
            rng=physics.next_keys,
            task=task_for_state,
            prev_action=action,
            prev_foot_pos=data.xpos[:, self.foot_bodies, :],
            prev_raw_target=raw_target,
            phase=phase,
            step_i=step_i,
            ep_return=ep_return,
            ep_len=ep_len,
            ep_vx_sum=ep_vx_sum,
            ep_cmd_vx_sum=ep_cmd_vx_sum,
            ep_bad_sum=ep_bad_sum,
            ep_support_sum=ep_support_sum,
            ep_step_evidence_sum=ep_step_evidence_sum,
            ep_command_quality_sum=ep_command_quality_sum,
            terrain_crossed=terrain_crossed,
            cmd_target=lesson.target_next,
            cmd_ramp_from=lesson.ramp_from_next,
            cmd_ramp_frac=lesson.ramp_frac_next,
            cmd_ramp_speed=lesson.ramp_speed_next,
            cmd_hold_left=lesson.hold_left_next,
            push_force=physics.push_force,
            push_steps_left=physics.push_steps_left,
            sensor_state=sensors,
            velocity_i_integral=integral_next,
            raw_velocity_command=raw_next,
            vx_reward_history=hist_v,
            cmd_vx_reward_history=hist_c,
            vx_reward_count=hist_n,
            academy_lip_index=academy['lip_index'],
            academy_streak=academy['streak'],
            academy_crossings=academy['crossings'],
            academy_up_crossings=academy['up_count'],
            academy_down_crossings=academy['down_count'],
            academy_foothold_seen=academy['foothold_seen'],
            academy_transfer_streak=academy['transfer_streak'],
            academy_transfer_seen=academy['transfer_seen'],
            academy_transfer_armed=academy['transfer_armed'],
            academy_progress_origin_xy=academy['progress_origin_xy'],
            academy_progress_heading_rad=academy['progress_heading_rad'],
            academy_progress_active=academy['progress_active'],
            academy_progress_paid=academy['progress_paid'],
            academy_approach_seen=academy['approach_seen'],
            academy_approach_lip_m=academy['approach_lip_m'],
            academy_approach_frontier_m=academy['approach_frontier_m'],
            terrain_attempt_lip_m=academy['terrain_attempt_lip_m'],
            terrain_attempt_frontier_m=academy['terrain_attempt_frontier_m'],
            terrain_attempt_tile_xy=academy['terrain_attempt_tile_xy'],
            terrain_first_cross_verified=academy['terrain_first_cross_verified'],
            academy_birth_lip_m=state.academy_birth_lip_m,
            academy_birth_frontier_m=state.academy_birth_frontier_m,
            academy_finish_streak=finish_streak,
            academy_goal_xy=state.academy_goal_xy,
            academy_goal_lip_armed=goal_lip_armed,
            academy_goal_foot_streak=goal_foot_streak,
            academy_goal_foot_seen=goal_foot_seen,
            academy_goal_transfer_ready=goal_transfer_ready,
            academy_goal_stable_streak=goal_stable_streak,
            academy_goal_completed=goal_completed,
            carpet_hfields=carpet_hfields,
            carpet_center=carpet_center,
            carpet_confirmed=academy['confirmed_tile'],
            carpet_seen_cells=academy['seen_cells'],
            carpet_seen_write=academy['seen_write'],
        )
        new_state = self.jax.tree_util.tree_map(choose, reset_state, stepped)
        obs = self.obs(new_state)
        # Telemetry is a read-only projection of physical facts and the
        # pre-reset episode state. It never affects PPO reward or actuation.
        out_metrics = self._step_report(
            state, stepped, data, params, metrics, terms, academy,
            timeout=timeout, terminal_fail=terminal_fail, done=done,
            transition_active=transition_active,
            cmd_resampled=cmd_resampled, fall_payment=fall_payment)
        return StepOut(
            new_state, jnp.nan_to_num(obs), jnp.nan_to_num(bootstrap_obs), jnp.nan_to_num(reward), done.astype(jnp.float32),
            # Defense in depth: the rollback above is the real fix for
            # physics divergence specifically, this is just insurance in
            # case some other numeric path ever goes non-finite -- cheap,
            # and guarantees NaN can never reach a PPO batch/gradient from
            # this function.
            {k: jnp.nan_to_num(v) for k, v in out_metrics.items()},
        )

    def _step_report(self, state: EnvState, stepped: EnvState, data, params,
                     metrics, terms, academy, *, timeout, terminal_fail, done,
                     transition_active, cmd_resampled, fall_payment):
        """Report physically measured episode/terrain evidence, not rewards.

        Runs on JAX arrays as part of the existing jit. `stepped` contains
        accumulated values from the frame *before* auto-reset; using the
        masked `new_state` would misclassify terminating episodes.
        """
        jnp = self.jnp
        ep_len = stepped.ep_len
        ep_return = stepped.ep_return
        ep_vx_sum = stepped.ep_vx_sum
        ep_cmd_vx_sum = stepped.ep_cmd_vx_sum
        ep_bad_sum = stepped.ep_bad_sum
        ep_support_sum = stepped.ep_support_sum
        ep_command_quality_sum = stepped.ep_command_quality_sum
        # Historical 'any lip crossed' remains a separate contact diagnostic.
        # Promotion success requires the stored FIRST attempted target tile.
        terrain_crossed = stepped.terrain_first_cross_verified
        terrain_cross_new = academy['ordinary_step_new']
        # Preserve the signed first encounter until the episode closes:
        # a success at frame 200 must not vanish at frame 700 timeout.
        ordinary_lip=stepped.terrain_attempt_lip_m
        terrain_active=(state.specialist_active &
            (state.focus_id==STEP) & ~state.academy_active &
            (jnp.abs(ordinary_lip)>.0015))
        # Near/easy support evidence uses the actual signed step size.
        # Ascents and descents are independently measured against their OWN
        # learned frontier. Late 150%-challenge lips cannot fake promotion.
        # A lip's height and severity *reference* must come from the same
        # encounter. Reclassifying an old 3mm attempt against a newly-grown
        # 4mm frontier biases the promotion denominator.
        academy_verdict_height=jnp.where(
            self.science_step_finish_on_transfer,
            state.academy_birth_lip_m,academy['verdict_height_m'])
        academy_ref=jnp.where(
            self.science_step_finish_on_transfer,
            state.academy_birth_frontier_m,
            academy['verdict_frontier_m'])
        academy_ref=jnp.where(
            jnp.isfinite(params[P.terrain_fixed_height_m]),
            jnp.abs(params[P.terrain_fixed_height_m]),academy_ref)
        academy_relative_height=(jnp.abs(academy_verdict_height) /
                                 jnp.maximum(academy_ref,.001))
        academy_near=((academy_relative_height>=.78) &
                      (academy_relative_height<=1.05))
        academy_easy=((academy_relative_height>=.36) &
                      (academy_relative_height<.68))
        terrain_up=terrain_active & (ordinary_lip>0.)
        terrain_down=terrain_active & (ordinary_lip<0.)
        # Step severity is measured from physical shared-edge contacts,
        # never the teacher's requested lip height or a retired slab.
        physical_height=jnp.abs(ordinary_lip)
        ratio=physical_height/jnp.maximum(stepped.terrain_attempt_frontier_m,.0001)
        step_near=terrain_active & (ratio>=.78) & (ratio<=1.05)
        step_easy=terrain_active & (ratio>=.36) & (ratio<.68)
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
        # Actual local triangle gradients of the two real top planes.
        # Do not average a fictitious quadrilateral or inspect a retired slab.
        cx=jnp.floor(data.qpos[:,0]/CARPET_TILE_M).astype(jnp.int32)
        cy=jnp.floor(data.qpos[:,1]/CARPET_TILE_M).astype(jnp.int32)
        slot=jnp.mod(cx,2)+2*jnp.mod(cy,2)
        hv=state.carpet_hfields.reshape((self.num_envs,CARPET_TILE_COUNT,4))
        h=jnp.take_along_axis(hv,slot[:,None,None],axis=1)[:,0,:]
        u=data.qpos[:,0]/CARPET_TILE_M-cx.astype(jnp.float32)
        v=data.qpos[:,1]/CARPET_TILE_M-cy.astype(jnp.float32)
        gx=jnp.where(u+v<=1.,h[:,1]-h[:,0],h[:,3]-h[:,2])*CARPET_SIZE_Z_M/CARPET_TILE_M
        gy=jnp.where(u+v<=1.,h[:,2]-h[:,0],h[:,3]-h[:,1])*CARPET_SIZE_Z_M/CARPET_TILE_M
        physical_slope=jnp.rad2deg(jnp.arctan(jnp.sqrt(gx*gx+gy*gy)))
        slope_role=state.specialist_active & (state.focus_id==SLOPE)
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
        # Two DISTINCT physical skills: a supported moving gait on the
        # incline (even with drift), and accurate 3-axis command tracking.
        # Neither may be inferred from merely being born on a slope.
        # Still require meaningful forward travel so standing motionless
        # cannot automatically push the teacher to harder terrain.
        slope_stable=(
            timeout & ~terminal_fail & (ep_len>=120.) &
            (ep_support_sum/jnp.maximum(ep_len,1.)>.10) &
            (ep_bad_sum/jnp.maximum(ep_len,1.)<.35) &
            (ep_vx_sum/jnp.maximum(ep_len,1.)>.01))
        slope_pass=(
            slope_stable &
            (ep_command_quality_sum/jnp.maximum(ep_len,1.)>=.60) &
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
            **{f'slope_episode_stable_{name}':(slope_stable & active).astype(jnp.float32)
               for name,active in slope_directions},
            **{f'slope_episode_{band}_{outcome}_{name}':
               (mask & active & condition).astype(jnp.float32)
               for name,active in slope_directions
               for band,mask in (('near',slope_near),('easy',slope_easy))
               for outcome,condition in (('attempt',done),
                                         ('success',slope_pass),
                                         ('stable',slope_stable))},
            'command_probe_radius':unit_radius,
            'command_probe_weight_x':unit_x/jnp.maximum(1e-5,unit_radius),
            'command_probe_weight_y':unit_y/jnp.maximum(1e-5,unit_radius),
            'command_probe_weight_z':unit_z/jnp.maximum(1e-5,unit_radius),
            'transition_frac': ramping,
            'cmd_resampled': resampled,
            # One maneuver verdict per source-to-target transition. These
            # diagnostics never enter actor observations or reward.
            'policy_cmd_vx': state.task[:,T.vx],
            'policy_cmd_vy': state.task[:,T.vy],
            'velocity_i_x': state.velocity_i_integral[:,0],
            'velocity_i_y': state.velocity_i_integral[:,1],
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
                                           state.focus_id,-1).astype(jnp.float32),
            'episode_command_quality': ep_command_quality_sum/jnp.maximum(ep_len,1.),
            'episode_length': ep_len,
            # A genuinely completed early stair task is an absorbing
            # SUCCESS, not an early survival failure. Keep actual episode
            # length honest; only the governor's competence denominator
            # receives the equivalent successful 350-frame credit.
            'episode_competence_length':jnp.where(
                academy['finish_new'],jnp.maximum(ep_len,350.),ep_len),
            'episode_return': ep_return,
            'episode_fell': terminal_fail.astype(jnp.float32),
            'episode_timeout': timeout.astype(jnp.float32),
            'terrain_episode_attempt': (done & terrain_active).astype(jnp.float32),
            'terrain_episode_success': (timeout & ~terminal_fail & terrain_active & terrain_crossed).astype(jnp.float32),
            'terrain_cross_new':terrain_cross_new.astype(jnp.float32),
            'academy_active':state.academy_active.astype(jnp.float32),
            'academy_lip_index':academy['lip_index'].astype(jnp.float32),
            'academy_lip_height_m':academy['height_m'],
            'academy_verdict_lip_height_m':academy_verdict_height,
            'academy_verdict_frontier_m':academy_ref,
            'terrain_encounter_frontier_m':stepped.terrain_attempt_frontier_m,
            'academy_finish_new':academy['finish_new'].astype(jnp.float32),
            'academy_cross_new':academy['cross_new'].astype(jnp.float32),
            # A four-crossing milestone, not a terminal course completion.
            'academy_course_new':(
                academy['cross_new'] &
                (jnp.mod(academy['lip_index'],4)==0)
            ).astype(jnp.float32),
            'academy_cross_up':academy['up_new'].astype(jnp.float32),
            'academy_lip_attempt_new':jnp.where(
                self.science_step_finish_on_transfer,
                done & state.academy_active &
                (jnp.abs(state.academy_birth_lip_m)>.0015),
                academy['attempt_new']).astype(jnp.float32),
            **{f'academy_{band}_{outcome}_{direction}':
                (band_mask & verdict &
                 direction_mask).astype(jnp.float32)
                for band,band_mask in (('near',academy_near),
                                       ('easy',academy_easy))
                for outcome,verdict in (
                    ('attempts',jnp.where(
                        self.science_step_finish_on_transfer,
                        done & state.academy_active &
                        (jnp.abs(state.academy_birth_lip_m)>.0015),
                        academy['attempt_new'])),
                    ('successes',jnp.where(
                        self.science_step_finish_on_transfer,
                        academy['finish_new'],academy['cross_new'])))
                for direction,direction_mask in (
                    ('up',academy_verdict_height>.0015),
                    ('down',academy_verdict_height<-.0015))},
            'academy_cross_down':academy['down_new'].astype(jnp.float32),
            'academy_clearance_new':academy['clearance_new'].astype(jnp.float32),
            'academy_foothold_new':academy['foothold_new'].astype(jnp.float32),
            'academy_transfer_new':academy['transfer_new'].astype(jnp.float32),
            'academy_followthrough_new':academy['progress_new'].astype(jnp.float32),
            'academy_reward':academy['reward'],
            # v5052 actual destination completion is separate from legacy
            # Academy "any supported lip before timeout" summaries.
            'academy_goal_complete_new':(
                stepped.academy_goal_completed &
                ~state.academy_goal_completed).astype(jnp.float32),
            'academy_goal_remaining_m':jnp.linalg.norm(
                state.academy_goal_xy-data.qpos[:,:2],axis=1),
            # Raw post-step physical pose, BEFORE any automatic reset.
            # Frozen actor exams need this fifth stable arrival frame:
            # their ordinary position_xy sample is PRE-step and misses
            # the last frame when goal completion ends the episode.
            'physical_position_xy_after_step':data.qpos[:,:2],
            'physical_heading_rad_after_step':jnp.arctan2(
                self._rotation_body_to_world(data)[:,1,0],
                self._rotation_body_to_world(data)[:,0,0]),
            'academy_goal_sequential_feet_ready':(
                stepped.academy_goal_transfer_ready).astype(jnp.float32),
            'academy_episode_attempt':(done & state.academy_active).astype(jnp.float32),
            # Separate first-tile proof gates at EPISODE END. These are
            # read-only diagnostics, not actor observations or reward flags.
            # A valid upper-foot transfer can be lost to the *teacher's*
            # first-lip arming gate before the short-finish check.
            'academy_episode_first_lip_armed':(
                done & state.academy_active &
                stepped.academy_goal_lip_armed).astype(jnp.float32),
            'academy_episode_two_foot_ready':(
                done & state.academy_active &
                stepped.academy_goal_transfer_ready).astype(jnp.float32),
            'academy_episode_complete':jnp.where(
                self.science_step_finish_on_transfer,
                academy['finish_new'],
                done & ~terminal_fail & state.academy_active &
                (academy['crossings']>0)).astype(jnp.float32),
            'academy_episode_crossings':jnp.where(done,
                 academy['crossings'].astype(jnp.float32),0.),
            'academy_episode_up_crossings':jnp.where(done,
                 academy['up_count'].astype(jnp.float32),0.),
            'academy_episode_down_crossings':jnp.where(done,
                 academy['down_count'].astype(jnp.float32),0.),
            'physical_step_height_m': physical_height,
            **{f'terrain_episode_{band}_{outcome}_{direction}':
                (mask & cohort & condition).astype(jnp.float32)
                for direction,cohort in (('up',terrain_up),('down',terrain_down))
                for band,mask in (('near',step_near),('easy',step_easy))
                for outcome,condition in (
                    ('attempt',done),
                    ('success',timeout & ~terminal_fail & terrain_crossed))},
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
        return out_metrics

    # ---- Physics and the actuators' actual delivered forces ----
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

        dynamic_model=self._carpet_dynamic_model(state.carpet_hfields)
        def substep(_, d):
            return self.step_batched(dynamic_model, d)
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

    def _focus_for_birth(self, keys, params):
        """Sample one role when an episode begins, not each physics frame.

        With no schedule weights supplied, preserve the older fixed world
        role map for held-out exams and legacy physical acceptance tests.
        With weights, zero-probability roles can never be sampled.
        """
        # Zero weights fall back to a deterministic fixed focus map in
        # held-out tests, never to a second physical or policy model.
        jax,jnp=self.jax,self.jnp
        w=params[P.focus_weight_command:P.focus_weight_control+1]
        logits=jnp.where(w>0.,jnp.log(jnp.maximum(w,1e-30)),-1e9)
        drawn=jax.vmap(lambda k:jax.random.categorical(
            jax.random.fold_in(k,50246),logits))(keys).astype(jnp.int32)
        return jnp.where(jnp.sum(w)>0.,drawn,self.focus_id)

    def _specialist_enabled(self, keys, params, focus_id=None):
        """Sample the host-chosen specialist teaching share at episode birth.

        The host holds this at zero for the first 20M cold gait steps, then
        linearly activates every allocated cohort over 10M. This is exposure
        scheduling, NOT a declaration of mastery; each physical target
        continues moving on its OWN completed-episode evidence.
        RNG streams remain independent of the legacy task/sensor sampler.
        """
        jax,jnp=self.jax,self.jnp
        # The host supplies the sole training-age exposure blend; there is
        # no separate age or cross-specialist gate inside the world.
        mix=jnp.clip(params[P.specialist_mix],0.,1.)
        k=jax.vmap(lambda x:jax.random.fold_in(x,50240))(keys)
        sampled=jax.vmap(lambda x:jax.random.uniform(x,()))(k)
        ordinary_focus=sampled<mix
        # When the normal specialist mix is zero (foundation), an optional
        # small slice of CONTROL worlds can practise graceful useful walking.
        # Slope/step/advanced command cohorts remain completely inactive.
        focus=self.focus_id if focus_id is None else focus_id
        grace=(focus==CONTROL) & (
            sampled<jnp.clip(params[P.foundation_control_mix],0.,1.))
        return ordinary_focus | grace

    def _specialist_task(self, task, keys, params, focus_id=None):
        """Change difficulty emphasis, never policy or actor observation.

        The 2048 worlds share the same robot, contacts, neural weights, and
        optimizer. Commands sample the measured L1 boundary, while control
        teaches cheaper/smoother gait inside an easier reliable envelope.
        Real incline+step collision is a separate mandatory launch gate.
        """
        jnp=self.jnp
        focus=self.focus_id if focus_id is None else focus_id
        active=self._specialist_enabled(keys,params,focus)
        # Rehearsal preserves its genuine fast command; it is a cohort
        # of the SAME policy, never another actor or terrain implementation.
        active=active & (task[:,T.athletic_rehearsal]<.5)
        jax=self.jax
        is_step=active & (focus==STEP)
        is_slope=active & (focus==SLOPE)
        coherent=self.terrain_lesson in ('coherent','coherent-direct','coherent-careful')
        is_coherent_terrain=(is_step|is_slope) & coherent
        direct_terrain=is_coherent_terrain & (self.terrain_lesson=='coherent-direct')
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
               jnp.where(is_control|is_coherent_terrain,0.,task[:,T.regime]))))
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
            probe_scale=params[P.frontier_combo_radius],
            steering_attention=params[P.steering_attention])
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
        # A real step needs forward intent to earn lip-progress reward.
        # Keep the genuine physical obstacle but take unrelated turn,
        # stop/reversal and resilience tasks out of the initial lesson.
        if coherent:
            kc=jax.vmap(lambda k:jax.random.fold_in(k,50321))(keys)
            v=jax.vmap(lambda k:jax.random.uniform(k,()))(kc)
            straight_vx=.10+.12*v
            straight=jnp.stack((straight_vx,
                jnp.zeros_like(straight_vx),jnp.zeros_like(straight_vx)),axis=-1)
            desired=jnp.where(is_coherent_terrain[:,None],straight,desired)
        # A single policy retains moving lower-speed expertise in a
        # persistent CONTROL rehearsal cohort; it must not learn to stand
        # motionless to avoid pushes or save energy.
        if self.retention_share>0.:
            kr=jax.vmap(lambda k:jax.random.fold_in(k,50322))(keys)
            q=jax.vmap(lambda k:jax.random.uniform(k,(4,)))(kr)
            rehearsal_vx=jnp.where(q[:,0]<.20,-(.04+.08*q[:,1]),
                                   .05+.19*q[:,1])
            rehearsal=jnp.stack((rehearsal_vx,
                (q[:,2]*2.-1.)*.07,(q[:,3]*2.-1.)*.30),axis=-1)
            desired=jnp.where(is_control[:,None],rehearsal,desired)
        # Every specialty practises the OTHER commands, but at half of
        # their sampled excursion around a small forward-walking reference.
        # This is not a separate command frontier and does not change the
        # target/edge probes of COMMAND-focused worlds.
        # Isolated v5027 treatment: preserve coherent STEP/SLOPE operator
        # intent, rather than attenuating sampled 0.10-0.22m/s toward
        # +0.04m/s. The original "coherent" option remains bit-for-bit
        # comparable; all other specialists retain their previous sampler.
        auxiliary=active & ~(is_command|is_control|direct_terrain)
        centre=jnp.asarray((.04,0.,0.),dtype=jnp.float32)
        desired=jnp.where(auxiliary[:,None],
                          centre+NONFOCUS_STRENGTH*(desired-centre),desired)
        old_speed=jnp.sqrt(task[:,T.vx]**2+task[:,T.vy]**2)
        new_speed=jnp.sqrt(desired[:,0]**2+desired[:,1]**2)
        base=jnp.clip(task[:,T.phase_rate]-
                      self.phase_speed_gain*jnp.clip(
                        old_speed/jnp.maximum(.05,self.vx_hi_max),0.,1.5),
                      .85,1.20)
        new_phase=command_phase_rate(desired[:,0],desired[:,1],
                                    self.vx_hi_max,self.phase_speed_gain,base)
        # Auxiliary command changes must update gait phase rate too.
        # Direct coherent terrain targets also need the correct gait clock.
        modified=is_command|is_control|auxiliary|direct_terrain
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
        resilience=jnp.where(active & ~is_resilience,
                             resilience*NONFOCUS_STRENGTH,resilience)
        resilience=jnp.where(is_control,resilience*(
            CONTROL_STRENGTH/NONFOCUS_STRENGTH),resilience)
        if self.retention_share>0.:
            # The rehearsal set includes manageable real shoves; not
            # simply undisturbed standing or huge 4N shocks.
            small=.10+.60*q[:,2]
            resilience=jnp.where(is_control,
                jnp.minimum(small/jnp.maximum(.70,self.push_max_force_n),
                            jnp.maximum(.05,resilience)),resilience)
        task=task.at[:,T.resilience].set(resilience)
        # All active specialists have easier posture variation; control is
        # easier still, so its loss/effort lessons reward *doing* real work.
        for column in (T.height,T.pitch,T.roll):
            task=task.at[:,column].set(jnp.where(
                is_control,CONTROL_STRENGTH*task[:,column],
                jnp.where(active,NONFOCUS_STRENGTH*task[:,column],
                          task[:,column])))
        birth=task[:,T.birth_resilience_d]
        birth=jnp.where(is_resilience,params[P.resilience_difficulty],
                        jnp.where(active,birth*NONFOCUS_STRENGTH,birth))
        birth=jnp.where(is_control,
                        params[P.resilience_difficulty]*CONTROL_STRENGTH,birth)
        task=task.at[:,T.birth_resilience_d].set(birth)
        return task,active

    # ---- Teacher-only physical terrain placement at episode birth ----
    def _academy_birth(self, keys, params, focus_id,
                       specialist_active, athletic_rehearsal):
        """Assign a real prism STEP cohort; old three-slab bodies are gone.

        Preserve the original salted 8-draw RNG assignment for reproducible
        Academy birth fractions, but no longer create unused platform poses.
        """
        jnp,jax=self.jnp,self.jax
        if not self.academy_stream:
            return jnp.zeros((self.num_envs,),dtype=jnp.bool_)
        u=jax.vmap(lambda k:jax.random.uniform(
            jax.random.fold_in(k,5040),(8,),dtype=jnp.float32))(keys)
        active=(specialist_active & (focus_id==STEP) &
                (athletic_rehearsal<.5) &
                (u[:,0]<params[P.academy_episode_probability]))
        return active

    def _carpet_dynamic_model(self,heights):
        """Rebuild convex prism vertices/normals per environment, fixed topology."""
        model=self.mjx_model.tree_replace({'hfield_data':heights})
        meshes=self.carpet_prism_builder.deform(
            self.jnp,heights,self.mjx_model_template._impl.mesh_convex)
        model=model.replace(_impl=model._impl.replace(mesh_convex=meshes))
        return model

    def _carpet_birth(self,data,keys,params,focus_id,specialist_active,
                      task,birth_root_z,academy_active):
        """The same four real triangle-pair tiles for ALL six roles.

        A foundation episode has zero terrain difficulty. Every specialist
        reads the *same* generated physics, with different frontiers from
        one existing PPO teacher. The terrain model is never rebuilt.
        """
        jnp,jax=self.jnp,self.jax
        seeds=jax.vmap(lambda k:jax.random.randint(
            jax.random.fold_in(k,5041),(),0,2**31-1,
            dtype=jnp.int32))(keys)
        # Use the SAME Academy assignment that the teacher, reward and
        # timetable use. Sampling the same PRNG key with a DIFFERENT shape
        # (scalar versus 8 draws) is not guaranteed to return that assignment.
        academy=academy_active
        up=jnp.where(academy,params[P.academy_height_up_m],
                           jnp.maximum(params[P.terrain_height_m],
                                       params[P.academy_height_up_m]))
        down=jnp.where(academy,params[P.academy_height_down_m],
                           jnp.maximum(params[P.terrain_height_m],
                                       params[P.academy_height_down_m]))
        slope,up,down=carpet_role_envelope(
            jnp,focus_id,specialist_active,task[:,T.athletic_rehearsal],
            params[P.slope_max_deg],up,down,academy,
            params[P.slope_exposure],params[P.terrain_exposure],seeds)
        xprof,yprof=carpet_make_profiles(jnp,seeds,up,down)
        if self.exam_fixture_steps is not None:
            # No sampled ±X sign, jitter or extra lip. Exact 0.75m seam,
            # optional continuous signed grade, unchanged real colliders.
            xprof,yprof=carpet_exam_lip_profiles(
                jnp,self.exam_fixture_steps)
            up=jnp.zeros_like(up)
            down=jnp.zeros_like(down)
            slope=jnp.zeros_like(slope)
        xy=jnp.floor(data.qpos[:,:2]/CARPET_TILE_M).astype(jnp.int32)
        window=carpet_midpoint_anchor(jnp,data.qpos[:,:2])
        centers,heights,_=carpet_generate_window(
            jnp,window[:,0],window[:,1],seeds,xprof,yprof,up,down,slope,
            transition_limit_deg=self.carpet_transition_cap_deg,
            exact_grade_x=self.exam_fixture_grades)
        pos=data.mocap_pos
        pos=update_tile_mocaps(
            jnp,pos,self.carpet_mocap_ids,centers)
        # Two physical half-tile prisms share one tile centre.
        prism_z=centers.at[:,:,2].add(.5*CARPET_SIZE_Z_M)
        pos=batch_update_prism_mocaps(
            jnp,pos,self.carpet_prism_mocap_ids,prism_z,
            jnp.ones((self.num_envs,CARPET_TILE_COUNT),dtype=jnp.bool_))
        data=data.replace(mocap_pos=pos)
        top=carpet_sample_surface(jnp,data,self.carpet_mocap_ids,
                  heights,data.qpos[:,:3][:,None,:])[:,0]
        top=jnp.where(jnp.isfinite(top),top,0.)
        data=data.replace(qpos=data.qpos.at[:,2].set(birth_root_z+top))
        visited=jnp.full((self.num_envs,CARPET_VISIT_HISTORY,2),
                         -(2**30),dtype=jnp.int32)
        visited=visited.at[:,0,:].set(xy)
        return data,dict(
            carpet_hfields=heights,carpet_center=window,carpet_seed=seeds,
            carpet_xprofile=xprof,carpet_yprofile=yprof,
            carpet_up=up,carpet_down=down,carpet_slope=slope,
            carpet_confirmed=xy,
            carpet_seen_cells=visited,
            carpet_seen_write=jnp.ones((self.num_envs,),dtype=jnp.int32))

    def _carpet_recycle(self,state,data):
        """Midpoint-driven 2x2 window: recycle only foot-clear prisms.

        The requested neighbour switches at the robot's halfway point in
        each 75cm tile, not at the terrain edge. The CURRENT tile always
        remains; both feet must clear any outgoing square's actual solid
        footprint before its two prism halves move together. Oscillating
        near a midpoint only reassigns distant empty tiles.
        """
        jnp=self.jnp
        desired=carpet_midpoint_anchor(jnp,data.qpos[:,:2])
        # The old cell-only guard missed a subtle but real collision:
        # a ball-foot centre can be inside the NEW 2x2 footprint while
        # its 18mm sphere still touches an abandoned prism overhang.
        # Use the actual rotated collider AABBs as a conservative exclusion.
        from .terrain_carpet_mjx import feet_clear_of_abandoned_prisms
        feet_xy=data.xpos[:,self.foot_bodies,:2]
        foot_cell=jnp.floor(feet_xy/CARPET_TILE_M).astype(jnp.int32)
        foot_inside_new_window=jnp.all(
            (foot_cell>=desired[:,None,:]) &
            (foot_cell<=desired[:,None,:]+1),axis=(1,2))
        foot_geom_ids=np.asarray(self.ids['foot_geoms'],dtype=np.int32)
        foot_radii=jnp.asarray(self.mj_model.geom_size[foot_geom_ids,0])
        foot_safe=foot_inside_new_window&feet_clear_of_abandoned_prisms(
            jnp,state.carpet_center,desired,feet_xy,foot_radii)
        # A controller should never jump two full 75cm tiles in 20ms.
        # Refuse such a teleport even if neither foot is detected.
        near=jnp.all(jnp.abs(desired-state.carpet_center)<=1,axis=1)
        move=foot_safe&near&jnp.any(desired!=state.carpet_center,axis=1)
        new_center=jnp.where(move[:,None],desired,state.carpet_center)
        centers,hfields,indices=carpet_generate_window(
            jnp,new_center[:,0],new_center[:,1],state.carpet_seed,
            state.carpet_xprofile,state.carpet_yprofile,
            state.carpet_up,state.carpet_down,state.carpet_slope,
            transition_limit_deg=self.carpet_transition_cap_deg,
            exact_grade_x=self.exam_fixture_grades)
        pos=data.mocap_pos
        # Identical slot geometry for every world; only retired 2x2
        # row/column slots move. Never move support under a foot.
        old_cx,old_cy=state.carpet_center[:,0],state.carpet_center[:,1]
        from .terrain_carpet_mjx import slot_indices
        old_x,old_y=slot_indices(jnp,old_cx,old_cy)
        change=move[:,None] & jnp.any(
            indices!=jnp.stack((old_x,old_y),axis=-1),axis=-1)
        pos=update_tile_mocaps(
            jnp,pos,self.carpet_mocap_ids,centers,change)
        physical_centres=centers.at[:,:,2].add(.5*CARPET_SIZE_Z_M)
        pos=batch_update_prism_mocaps(
            jnp,pos,self.carpet_prism_mocap_ids,physical_centres,change)
        hfield=jnp.where(move[:,None],hfields,state.carpet_hfields)
        return data.replace(mocap_pos=pos),hfield,new_center

    def _forward_reset_data(self, model, data, reset_mask):
        """Full-accuracy MJX forward, only for worlds actually terminating.

        At every control step the old code ran an entire 2048-world
        mjx.forward (and an expensive 14-DoF constraint solve) just to
        throw away almost every row with jnp.where(done,...). Here all
        reset sampling remains bit-for-bit unchanged; only the expensive
        *forward* runs on a compact fixed-size batch of completed worlds.

        Compaction uses a fixed JAX shape to avoid re-JIT on varying falls.
        If the number of deaths exceeds capacity, take the original full
        physics path, exactly (no dropped resets, no dead-world shortage).
        Model leaves with in_axes=0 are packed alongside each world's data,
        so domain-randomized actuator and mass physics remain per-world.
        A scalar lax.cond (not a vmapped one) genuinely skips the full
        solve. The 0 setting restores the old reference path for A/B.
        """
        if reset_mask is None or self.reset_forward_capacity == 0:
            return self.forward_batched(model, data)
        jax,jnp=self.jax,self.jnp
        cap=self.reset_forward_capacity
        count=jnp.sum(reset_mask.astype(jnp.int32))

        def packed(_):
            indices=jnp.nonzero(reset_mask,size=cap,fill_value=0)[0]
            # Inactive padded indices may repeat zero. Their values are
            # discarded by the final episode-done selection; active rows
            # are unique and receive precisely their own MJX model leaves.
            partial_model=jax.tree_util.tree_map(
                lambda leaf,axis: jnp.take(leaf,indices,axis=0)
                if axis == 0 else leaf,
                model,self.model_in_axes,
                is_leaf=lambda x: x is None)
            partial_data=jax.tree_util.tree_map(
                lambda leaf: jnp.take(leaf,indices,axis=0),data)
            forwarded=self.forward_batched(partial_model,partial_data)
            return jax.tree_util.tree_map(
                lambda original,small: original.at[indices].set(small),
                data,forwarded)

        def some_reset(_):
            return jax.lax.cond(
                count<=cap,packed,
                lambda _:self.forward_batched(model,data),operand=None)

        return jax.lax.cond(
            count>0,some_reset,lambda _:data,operand=None)

    def _reset_with_keys(self, keys, params, carryover_qvel=None,
                         reset_mask=None):
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
        focus_id = self._focus_for_birth(keys, params)
        task, specialist_active = self._specialist_task(
            task, keys, params, focus_id)
        if self.stillness_birth_frac > 0.:
            # Reserve real zero-command episodes AFTER Academy/coherent
            # STEP and CONTROL command rewriting. The earlier placement
            # inside _sample_task was erased by _specialist_task in most
            # worlds. This is ONE optional 10% birth cohort, not a second
            # policy or a privileged observation input. It takes practice
            # away from other cohorts by the requested fraction.
            still_keys=jax.vmap(
                lambda key:jax.random.fold_in(key,504801))(keys)
            still_draw=jax.vmap(
                lambda key:jax.random.uniform(key,()))(still_keys)
            still=(still_draw < self.stillness_birth_frac) & (
                task[:,T.athletic_rehearsal] < .5)
            for col in (T.vx,T.vy,T.yaw,T.height,T.pitch,T.roll,
                        T.resilience,T.birth_resilience_d,T.field_taught):
                task=task.at[:,col].set(jnp.where(still,0.,task[:,col]))
            task=task.at[:,T.regime].set(jnp.where(still,0.,task[:,T.regime]))
            focus_id=jnp.where(still,CONTROL,focus_id)
            specialist_active=jnp.where(still,False,specialist_active)
        data = self._birth_physics(
            data, task, params,
            (noise_keys, vnoise_keys, tilt_keys, pos_keys, angvel_keys,
             linvel_keys, joint_keys, carry_keys),
            carryover_qvel,
        )
        if self.exam_fixture_steps is not None:
            # Every frozen trial begins 22cm before the first real seam,
            # centred in Y. Kicks are velocity initial conditions, not
            # phantom impulses/reward flags. Non-exam births unaffected.
            qpos=data.qpos.at[:,0].set(.53).at[:,1].set(.375)
            qvel=data.qvel.at[:,0].add(self.exam_fixture_kicks)
            data=data.replace(qpos=qpos,qvel=qvel)
        # One episode-local collision course. The standard flat path above
        # is bitwise untouched whenever terrain_step_max_m is zero.
        academy_active=self._academy_birth(
            keys,params,focus_id,specialist_active,
            task[:,T.athletic_rehearsal])
        if self.science_step_approach_m > 0.:
            # Academic stair nursery: Academy-only births land a fixed safe
            # distance BEFORE the next authentic 75cm prism seam. The
            # physical profile (including its signed up/down lip), ToF and
            # foot contacts are then recomputed from the new birth position
            # by the ordinary _carpet_birth + MJX forward path.
            # Other roles and the reset RNG are strictly unchanged.
            from .terrain_carpet_mjx import academy_near_lip_spawn_x
            near_x=academy_near_lip_spawn_x(
                jnp,data.qpos[:,0],self.science_step_approach_m)
            data=data.replace(qpos=data.qpos.at[:,0].set(
                jnp.where(academy_active,near_x,data.qpos[:,0])))
        birth_root_z=data.qpos[:,2]
        data,carpet=self._carpet_birth(
            data,keys,params,focus_id,specialist_active,task,
            birth_root_z,academy_active)
        ctrl = data.ctrl
        ctrl = ctrl.at[:, :ACTION_DIM].set(self.home_target)
        data = data.replace(ctrl=ctrl)
        dynamic_model=self._carpet_dynamic_model(carpet['carpet_hfields'])
        data = self._forward_reset_data(dynamic_model, data, reset_mask)
        # Actual signed FIRST +X edge: these are the physical prism sources
        # already used to generate collisions, not a requested frontier or
        # the transient seam height at an eventual end-of-episode frame.
        cx=jnp.floor(data.qpos[:,0]/CARPET_TILE_M).astype(jnp.int32)
        cy=jnp.floor(data.qpos[:,1]/CARPET_TILE_M).astype(jnp.int32)
        src_slot=jnp.mod(cx,2)+2*jnp.mod(cy,2)
        dst_slot=jnp.mod(cx+1,2)+2*jnp.mod(cy,2)
        corners=carpet['carpet_hfields'].reshape((self.num_envs,CARPET_TILE_COUNT,4))
        src=jnp.take_along_axis(corners,src_slot[:,None,None],axis=1)[:,0,:]
        dst=jnp.take_along_axis(corners,dst_slot[:,None,None],axis=1)[:,0,:]
        mz=jnp.stack(tuple(data.mocap_pos[:,m,2]
                           for m in self.carpet_mocap_ids),axis=1)
        z_src=jnp.take_along_axis(mz,src_slot[:,None],axis=1)[:,0]
        z_dst=jnp.take_along_axis(mz,dst_slot[:,None],axis=1)[:,0]
        birth_lip=z_dst-z_src+.5*CARPET_SIZE_Z_M*(
            dst[:,0]+dst[:,2]-src[:,1]-src[:,3])
        phase = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.0, maxval=2.0 * jnp.pi))(phase_keys)
        prev_action = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        prev_foot_pos = data.xpos[:, self.foot_bodies, :]
        prev_raw_target = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        sensors = self._initial_sensor_state(
            data, task, phase, noise_keys, carpet_heights=carpet['carpet_hfields'])
        base = self._base_obs(data, task, prev_action, phase, prev_action, sensors)
        hist = jnp.repeat(base[:, None, :], HISTORY_LEN, axis=1)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        from .step_goal import destination_from_birth
        goal_xy=destination_from_birth(jnp,data.qpos[:,:2])
        cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left = self._init_transition_state(task, params, hold_keys)
        push_force0 = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        return EnvState(
            data=data,
            hist=hist,
            rng=next_keys,
            task=task,
            specialist_active=specialist_active,
            focus_id=focus_id,
            prev_action=prev_action,
            prev_foot_pos=prev_foot_pos,
            prev_raw_target=prev_raw_target,
            phase=phase,
            step_i=izeros,
            ep_return=zeros,
            ep_len=zeros,
            ep_vx_sum=zeros,
            ep_cmd_vx_sum=zeros,
            ep_bad_sum=zeros,
            ep_support_sum=zeros,
            ep_step_evidence_sum=zeros,
            ep_command_quality_sum=zeros,
            terrain_crossed=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            cmd_target=cmd_target,
            cmd_ramp_from=cmd_ramp_from,
            cmd_ramp_frac=cmd_ramp_frac,
            cmd_ramp_speed=cmd_ramp_speed,
            cmd_hold_left=cmd_hold_left,
            push_force=push_force0,
            push_steps_left=izeros,
            sensor_state=sensors,
            velocity_i_integral=jnp.zeros((self.num_envs,2),dtype=jnp.float32),
            raw_velocity_command=task[:,[T.vx,T.vy]],
            vx_reward_history=jnp.zeros((self.num_envs,25),dtype=jnp.float32),
            cmd_vx_reward_history=jnp.zeros((self.num_envs,25),dtype=jnp.float32),
            vx_reward_count=izeros,
            academy_active=academy_active,
            academy_lip_index=izeros,
            academy_streak=izeros,
            academy_crossings=izeros,
            academy_up_crossings=izeros,
            academy_down_crossings=izeros,
            academy_foothold_seen=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_transfer_streak=izeros,
            academy_transfer_seen=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_transfer_armed=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_progress_origin_xy=jnp.zeros((self.num_envs,2),dtype=jnp.float32),
            academy_progress_heading_rad=zeros,
            academy_progress_active=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_progress_paid=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_approach_seen=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_approach_lip_m=zeros,
            academy_approach_frontier_m=zeros,
            terrain_attempt_lip_m=zeros,
            terrain_attempt_frontier_m=zeros,
            terrain_attempt_tile_xy=jnp.full((self.num_envs,2),-(2**30),dtype=jnp.int32),
            terrain_first_cross_verified=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_birth_lip_m=birth_lip,
            academy_birth_frontier_m=jnp.where(
                birth_lip>0.,params[P.academy_height_up_m],
                params[P.academy_height_down_m]),
            academy_finish_streak=izeros,
            academy_goal_xy=goal_xy,
            academy_goal_lip_armed=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_goal_foot_streak=jnp.zeros((self.num_envs,2),dtype=jnp.int32),
            academy_goal_foot_seen=jnp.zeros((self.num_envs,2),dtype=jnp.bool_),
            academy_goal_transfer_ready=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            academy_goal_stable_streak=izeros,
            academy_goal_completed=jnp.zeros((self.num_envs,),dtype=jnp.bool_),
            carpet_hfields=carpet['carpet_hfields'],
            carpet_center=carpet['carpet_center'],
            carpet_seed=carpet['carpet_seed'],
            carpet_xprofile=carpet['carpet_xprofile'],
            carpet_yprofile=carpet['carpet_yprofile'],
            carpet_up=carpet['carpet_up'],
            carpet_down=carpet['carpet_down'],
            carpet_slope=carpet['carpet_slope'],
            carpet_confirmed=carpet['carpet_confirmed'],
            carpet_seen_cells=carpet['carpet_seen_cells'],
            carpet_seen_write=carpet['carpet_seen_write'],
        )

    def _rotation_body_to_world(self, data):
        return data.xmat[:, self.torso_id, :].reshape((self.num_envs, 3, 3))

    # ---- Sensor simulation and delayed deployable observations ----
    def _body_packets(self, data):
        jnp = self.jnp
        R = self._rotation_body_to_world(data)
        RT = jnp.swapaxes(R, 1, 2)
        projected_gravity = jnp.einsum('nij,nj->ni', RT, self.gravity_unit_world)
        body_vel = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 0:3])
        body_ang = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 3:6])
        return projected_gravity, body_vel, body_ang

    def _terrain_4x4_ranges(self, data, errors_m=None, carpet_heights=None,
                            *, include_downward_depth=False):
        """16 ToF rays, optionally fused with a downward optical-flow ray.

        The 17th ray is never exposed as an actor ToF channel.
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
        # The same two physical triangles and vertical prism sidewall
        # used by MJX contacts. Never substitute a box or infinite plane.
        # Both sensors sample identical physical prisms. Intersect 16+1
        # rays once rather than recreating four tile/wall intersections
        # separately for the optical-flow height sensor.
        surface_rays=(jnp.concatenate([ray,(-R[:,:,2])[:,None,:]],axis=1)
                      if include_downward_depth else ray)
        all_t=carpet_ray_surface(
            jnp,data,self.carpet_mocap_ids,carpet_heights,
            origin,surface_rays,prism_bottom_depth_m=PRISM_DEPTH_M)
        terrain_t=all_t[:,:16] if include_downward_depth else all_t
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
        four=jnp.where(valid,jnp.clip(result,TERRAIN_TOF_MIN_M,
                                      TERRAIN_TOF_MAX_M),0.).astype(jnp.float32)
        if include_downward_depth:
            return four,valid,all_t[:,16]
        return four,valid

    def _sensor_sample(self, data, task, keys, gyro_bias, carpet_heights=None):
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
        four,valid,depth=self._terrain_4x4_ranges(
            data,errors_m=scale[:,None]*.002*four_noise,
            carpet_heights=carpet_heights,include_downward_depth=True)
        depth=jnp.where(jnp.isfinite(depth)&(depth>=.035)&(depth<=.60),
                        depth,0.)
        _,vbody,_=self._body_packets(data)
        physical_gyro=data.qvel[:,3:6]
        flow,flow_valid=simulate_downward_optical_flow(
            jnp,vbody,physical_gyro,depth,
            flow_error_rad_s=scale[:,None]*.035*noise[:,3:5])
        measured_gyro=physical_gyro+gyro_bias+scale[:,None]*.025*noise[:,5:8]
        ages=jnp.zeros((self.num_envs,),dtype=jnp.int32)
        return SensorSample(
            four,valid,flow,flow_valid,measured_gyro,ages,ages)

    def _initial_sensor_state(self, data, task, phase, keys, carpet_heights=None):
        """Start an episode with a real first sample, not fabricated flat data."""
        jax, jnp = self.jax, self.jnp
        bias_noise = jax.vmap(lambda k: jax.random.normal(
            jax.random.fold_in(k, 5017), (3,), dtype=jnp.float32))(keys)
        bias = bias_noise * self.gyro_bias_std_rad_s
        sample = self._sensor_sample(
            data, task, keys, bias, carpet_heights=carpet_heights)
        samples = jax.tree_util.tree_map(
            lambda a: jnp.repeat(a[:, None, ...], HISTORY_LEN, axis=1),
            sample)
        return SensorState(samples, bias, sample.gyro_body_rad_s)

    def _advance_sensor_state(self, state, data, task, phase, keys, age_frames,
                              carpet_heights=None):
        """Acquire at separate cadences, hold between updates, and queue delay.

        Dropouts make the *new* sample invalid; an old value may still be
        in transit, but an invalid one eventually reaches the policy with
        zero confidence. Clock is episode age, not staggered timeout step_i.
        """
        jax, jnp = self.jax, self.jnp
        raw = self._sensor_sample(
            data, task, keys, state.gyro_bias_rad_s,
            carpet_heights=carpet_heights)
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
        terrain_valid = raw.terrain_valid & ~tof_failed[:, None]
        flow_valid = raw.flow_valid & ~flow_failed
        latest = SensorSample(
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
        """Invert delayed 4x4 rays + flow using the exact same IMU frames.

        The original three-ToF model is deleted: it was never read by the
        actual 360-input policy. World truth is not leaked to this estimator.
        """
        jnp=self.jnp
        samples=state.samples
        tf=HISTORY_LEN-1-self.tof_delay_frames
        ff=HISTORY_LEN-1-self.flow_delay_frames
        ground=fit_forward_array_ground(
            jnp,samples.terrain_ranges_m[:,tf,:],
            samples.terrain_valid[:,tf,:],gravity_body)
        flow_valid=samples.flow_valid[:,ff]
        estimate=estimate_flow_velocity(
            jnp,samples.flow_rad_s[:,ff,:],
            samples.gyro_body_rad_s[:,ff,:],
            ground['optical_depth_m'],
            flow_valid=flow_valid,tof_valid=ground['depth_valid'])
        tof_age=samples.tof_age_frames[:,tf]
        flow_age=samples.flow_age_frames[:,ff]
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
            terrain_ranges_m=samples.terrain_ranges_m[:,tf,:],
            terrain_valid=samples.terrain_valid[:,tf,:],
            flow_valid=flow_valid,
            tof_age_frames=tof_age,flow_age_frames=flow_age)

    def _carpet_real_top_height(self,data,positions,carpet_heights=None):
        """Teacher-only support height from the SAME actual prism vertices."""
        if carpet_heights is None:
            raise ValueError('Prism top needs live physical vertex samples')
        return carpet_sample_surface(
            self.jnp,data,self.carpet_mocap_ids,carpet_heights,positions)

    def _support_signed_heights(self,data,positions,carpet_heights=None):
        """Signed height over actual prism top, never a phantom floor."""
        top=self._carpet_real_top_height(data,positions,carpet_heights)
        return carpet_signed_support(self.jnp,positions,top)

    def _carpet_events(self,state,data,metrics,done,params):
        """Genuine supported tile-to-tile crossings, separately for UP/DOWN.

        Every role can cross the rolling 2x2 world. STEP focus receives evidence
        only when an actual lip exists; no expert privileged geometry enters
        neural observations. Five consecutive post-lip top contacts are
        required, not just a coordinate change or the robot being airborne.
        """
        jnp=self.jnp
        zero=jnp.zeros((self.num_envs,),dtype=jnp.float32)
        false=jnp.zeros((self.num_envs,),dtype=jnp.bool_)
        root=jnp.floor(data.qpos[:,:2]/CARPET_TILE_M).astype(jnp.int32)
        previous=state.carpet_confirmed
        delta=root-previous
        moved=jnp.any(delta!=0,axis=1)
        near=jnp.all(jnp.abs(delta)<=1,axis=1)
        shared_lip=carpet_shared_edge(jnp,previous,root)
        target_slot=jnp.mod(root[:,0],2)+2*jnp.mod(root[:,1],2)
        previous_slot=jnp.mod(previous[:,0],2)+2*jnp.mod(previous[:,1],2)
        height_map=state.carpet_hfields.reshape((self.num_envs,CARPET_TILE_COUNT,4))
        mocaps=jnp.stack(tuple(data.mocap_pos[:,m,:]
                                for m in self.carpet_mocap_ids),axis=1)
        # The physical height of the shared discontinuity, not the rise
        # of the slope over the TILE_M span: compare BOTH sides at the edge.
        before=jnp.take_along_axis(height_map,
             previous_slot[:,None,None],axis=1)[:,0,:]
        after=jnp.take_along_axis(height_map,
             target_slot[:,None,None],axis=1)[:,0,:]
        pos_before=jnp.take_along_axis(mocaps,
             previous_slot[:,None,None],axis=1)[:,0,:]
        pos_after=jnp.take_along_axis(mocaps,
             target_slot[:,None,None],axis=1)[:,0,:]
        x_cross=delta[:,0]!=0
        x_pos=delta[:,0]>0
        y_pos=delta[:,1]>0
        # Sample at the middle of the crossing edge, paired with the
        # matching neighbor edge. The two triangle surfaces meet at
        # identical smooth vertex heights, except deliberate step offset.
        oldedge=jnp.where(x_cross,
            jnp.where(x_pos,.5*(before[:,1]+before[:,3]),
                             .5*(before[:,0]+before[:,2])),
            jnp.where(y_pos,.5*(before[:,2]+before[:,3]),
                             .5*(before[:,0]+before[:,1])))
        newedge=jnp.where(x_cross,
            jnp.where(x_pos,.5*(after[:,0]+after[:,2]),
                             .5*(after[:,1]+after[:,3])),
            jnp.where(y_pos,.5*(after[:,0]+after[:,1]),
                             .5*(after[:,2]+after[:,3])))
        rise=(pos_after[:,2]-pos_before[:,2]+
              CARPET_SIZE_Z_M*(newedge-oldedge))
        # Sidewall collision is not a supported *top* foothold. Require
        # the SAME foot's MJX contact normal to face upward, its sphere
        # bottom to match the target tile's actual triangle elevation,
        # and its XY position to be inside that target tile.
        contact=getattr(data,'_impl',data).contact
        foot_geoms=tuple(int(x) for x in self.ids['foot_geoms'])
        foot_xyz=data.geom_xpos[:,foot_geoms,:]
        foot_bottom=(foot_xyz[:,:,2]-jnp.asarray(
            self.mj_model.geom_size[list(foot_geoms),0])[None,:])
        physical_foot_top=self._carpet_real_top_height(
            data,foot_xyz,state.carpet_hfields)
        tile_geoms=tuple(self.carpet_prism_geom_ids[2*i:2*i+2]
                         for i in range(CARPET_TILE_COUNT))
        top_contacts_by_tile=jnp.stack(tuple(
            jnp.any(jnp.stack(tuple(
                carpet_top_foot_pairs(
                    jnp,contact.geom1,contact.geom2,contact.dist,
                    contact.frame,g,foot_geoms,foot_bottom,physical_foot_top)
                for g in pair),axis=1),axis=1)
            for pair in tile_geoms),axis=1)
        target_pairs=jnp.take_along_axis(
            top_contacts_by_tile,
            jnp.broadcast_to(target_slot[:,None,None],
                             (self.num_envs,1,2)),axis=1)[:,0,:]
        foot_cell=jnp.floor(foot_xyz[:,:,:2]/
                            CARPET_TILE_M).astype(jnp.int32)
        on_target_foot=jnp.any(
            jnp.all(foot_cell==root[:,None,:],axis=2)&target_pairs,axis=1)
        healthy=(metrics['terminal']<.5)&(metrics['bad_contact']<.5)
        eligible=moved&near&on_target_foot&healthy
        streak=jnp.where(eligible,state.academy_streak+1,0)
        confirmed=eligible&(streak>=5)
        fresh=carpet_unseen_tile(
            jnp,root,state.carpet_seen_cells)
        paid=confirmed&fresh&shared_lip
        # The shared helper records ONLY novel globally indexed
        # confirmed cells; repeat backtracking never consumes a ring slot.
        seen_cells,seen_write=carpet_note_confirmed_tile(
            jnp,root,state.carpet_seen_cells,
            state.carpet_seen_write,confirmed)
        raised=rise>.0015
        lowered=rise<-.0015
        lip=raised|lowered
        step_focus=state.specialist_active & (state.focus_id==STEP)
        academy_focus=step_focus&state.academy_active
        approached=moved&near&shared_lip&lip&step_focus&fresh
        seen=state.academy_approach_seen|approached
        # Latch the signed real rise while the seam actually exists.
        # At success or timeout the current body cell may have changed and
        # the transient 'height_m' is commonly zero. Keep the unresolved
        # Academy approach separately from the ordinary episode's FIRST lip.
        new_approach=approached & ~state.academy_approach_seen
        approach_lip=jnp.where(
            new_approach,rise,state.academy_approach_lip_m)
        reference=jnp.where(rise>0.,
                            params[P.academy_height_up_m],
                            params[P.academy_height_down_m])
        approach_frontier=jnp.where(
            new_approach,reference,state.academy_approach_frontier_m)
        ordinary_first=approached & ~state.academy_active & (
            jnp.abs(state.terrain_attempt_lip_m)<=.0015)
        terrain_attempt_lip=jnp.where(
            ordinary_first,rise,state.terrain_attempt_lip_m)
        terrain_attempt_frontier=jnp.where(
            ordinary_first,params[P.terrain_height_m],
            state.terrain_attempt_frontier_m)
        attempted_target=jnp.where(
            ordinary_first[:,None],root,state.terrain_attempt_tile_xy)
        new_foothold=(approached&eligible&~state.academy_foothold_seen)
        # For TRUE two-foot support, each foot must occupy the confirmed
        # new tile, have an authentic upward-facing prism contact, and
        # stay there together for five uninterrupted frames. A single
        # leading-foot contact is not a trailing-foot recovery.
        both_top=jnp.all(jnp.all(foot_cell==root[:,None,:],axis=2) &
                         target_pairs,axis=1)
        same_confirmed=jnp.all(root==previous,axis=1)
        # A prior crossing somewhere in the episode is NOT sufficient.
        # Arm the transfer exam only after an authentically supported
        # *new signed lip*, never after a smooth tile or later flat stance.
        transfer_ok=(academy_focus & state.academy_transfer_armed &
                     same_confirmed & both_top & healthy)
        transfer_streak=jnp.where(transfer_ok,
                                   state.academy_transfer_streak+1,0)
        new_transfer=(transfer_ok & (transfer_streak>=5) &
                      ~state.academy_transfer_seen)
        transfer_seen=jnp.where(confirmed,False,
                                state.academy_transfer_seen|new_transfer)
        transfer_streak=jnp.where(confirmed,0,transfer_streak)
        transfer_armed=jnp.where(
            confirmed,paid & academy_focus & lip,
            state.academy_transfer_armed & ~new_transfer)
        # Newly confirmed bilateral support creates a single rewardable
        # continuation episode. Heading is latched at transfer, not
        # silently redefined as the robot walks or turns around.
        R=self._rotation_body_to_world(data)
        yaw=jnp.arctan2(R[:,1,0],R[:,0,0])
        progress_origin=jnp.where(new_transfer[:,None],
                                  data.qpos[:,:2],
                                  state.academy_progress_origin_xy)
        progress_heading=jnp.where(new_transfer,yaw,
                                    state.academy_progress_heading_rad)
        angle=yaw-progress_heading
        heading_error=jnp.abs(jnp.arctan2(jnp.sin(angle),jnp.cos(angle)))
        follow_active=jnp.where(
            confirmed,False,
            jnp.where(new_transfer,True,
                      state.academy_progress_active &
                      (heading_error<=.35) & ~done))
        travel=jnp.sum(
            (data.qpos[:,:2]-progress_origin) *
            jnp.stack((jnp.cos(progress_heading),
                       jnp.sin(progress_heading)),axis=1),axis=1)
        fresh_paid=jnp.where(confirmed,False,state.academy_progress_paid)
        new_progress=(follow_active & ~fresh_paid &
                      (travel>=.50) & (metrics['cmd_vx']>=.05) &
                      (metrics['support']>=.5) & healthy)
        follow_paid=fresh_paid|new_progress
        # No repeated foothold farm after 5-frame verification.
        reward=(3.*new_foothold.astype(jnp.float32)+
                10.*(paid&academy_focus&lip).astype(jnp.float32))
        up_new=paid&academy_focus&raised
        down_new=paid&academy_focus&lowered
        # A later DIFFERENT lip must never certify the first attempted
        # near-frontier step as successful, even if the episode survives.
        first_target_paid=(paid & step_focus & ~state.academy_active &
                           lip & jnp.all(root==attempted_target,axis=1))
        first_cross_verified=(state.terrain_first_cross_verified |
                              first_target_paid)
        return dict(
            confirmed_tile=jnp.where(confirmed[:,None],root,previous),
            seen_cells=seen_cells,seen_write=seen_write,
            lip_index=state.academy_lip_index+
                (paid&academy_focus&lip).astype(jnp.int32),
            streak=jnp.where(confirmed,0,streak),
            crossings=state.academy_crossings+
                (paid&academy_focus&lip).astype(jnp.int32),
            up_count=state.academy_up_crossings+up_new.astype(jnp.int32),
            down_count=state.academy_down_crossings+down_new.astype(jnp.int32),
            cross_new=paid&academy_focus&lip,
            # Standard STEP worlds before 60M earn normal supported-crossing
            # credit; Academy worlds use the separate independent frontiers.
            ordinary_step_new=paid&step_focus&~state.academy_active&lip,
            up_new=up_new,down_new=down_new,
            attempt_new=((paid&academy_focus&lip) |
                (done&academy_focus&seen)),
            approach_seen=jnp.where(confirmed,False,seen),
            approach_lip_m=jnp.where(confirmed,0.,approach_lip),
            approach_frontier_m=jnp.where(confirmed,0.,approach_frontier),
            terrain_attempt_lip_m=terrain_attempt_lip,
            terrain_attempt_frontier_m=terrain_attempt_frontier,
            terrain_attempt_tile_xy=attempted_target,
            terrain_first_cross_verified=first_cross_verified,
            verdict_height_m=jnp.where(
                paid & academy_focus & lip,rise,
                jnp.where(done & academy_focus & seen,approach_lip,0.)),
            verdict_frontier_m=jnp.where(
                paid & academy_focus & lip,reference,
                jnp.where(done & academy_focus & seen,approach_frontier,0.)),
            foothold_seen=jnp.where(confirmed,False,
                state.academy_foothold_seen|new_foothold),
            clearance_new=false,foothold_new=new_foothold,
            transfer_new=new_transfer,
            # Separate v5052 stage: each foot's independently real top
            # contact on this root tile (never a sidewall or virtual hfield).
            # Teacher goal code additionally checks the ASSIGNED first tile.
            goal_top_feet=(jnp.all(foot_cell==root[:,None,:],axis=2)
                           & target_pairs & healthy[:,None]),
            transfer_streak=transfer_streak,
            transfer_seen=transfer_seen,
            transfer_armed=transfer_armed,
            progress_new=new_progress,
            progress_origin_xy=progress_origin,
            progress_heading_rad=progress_heading,
            progress_active=follow_active,
            progress_paid=follow_paid,
            reward=reward,height_m=jnp.where(
                moved&fresh&shared_lip,rise,zero),
            completed=false)

    # ---- Teacher/evaluator-only supported crossing evidence ----
    # ---- Physical facts for reward and report, not hidden actor inputs ----
    def _metrics(self, data, task, action, prev_action, phase, prev_foot_pos,
                 carpet_heights=None):
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
        # Command tracking measures horizontal world motion relative to
        # the robot's heading, not tilted body-local velocity.
        motion_vx, motion_vy, motion_yaw = horizontal_tracking_motion(
            jnp, self._rotation_body_to_world(data),
            data.qvel[:, 0:3], data.qvel[:, 3:6])
        body_pitch = jnp.arctan2(projected_gravity[:, 0], -projected_gravity[:, 2])
        body_roll = jnp.arctan2(-projected_gravity[:, 1], -projected_gravity[:, 2])
        pitch_err = body_pitch - task[:, T.pitch]
        roll_err = body_roll - task[:, T.roll]
        foot_pos = data.xpos[:, self.foot_bodies, :]
        # Height means clearance above the actual terrain, not world Z.
        # Batch the repeated terrain queries. For the new foot-only solver,
        # additionally sample the actual lowest oriented torso-box corner,
        # knees, thigh rods and two shin sections. This is a *failure
        # sentinel*, never a fake load-bearing contact force.
        bad_body_pos=data.xpos[:,self.bad_bodies,:]
        if self.feet_only_contacts:
            mat=data.geom_xmat[:,self.torso_geom,:].reshape(
                (self.num_envs,3,3))
            lowest_local=-jnp.sign(mat[:,2,:])*self.torso_half_extents
            torso_low=data.geom_xpos[:,self.torso_geom,:]+jnp.einsum(
                'nij,nj->ni',mat,lowest_local)
            knees=bad_body_pos[:,3:5,:]
            shin_middle=data.xpos[:,self.shin_lower_bodies,:]
            thigh_mid=.5*(bad_body_pos[:,1:3,:]+knees)
            shin_low_mid=.5*(shin_middle+foot_pos)
            extra=jnp.concatenate((
                torso_low[:,None,:],thigh_mid,shin_middle,shin_low_mid),
                axis=1)
            # One 14-point query is cheaper than repeated 1/2/5/7-point
            # queries, using exactly the live four-triangle-prism surfaces.
            signed=self._support_signed_heights(
                data,jnp.concatenate((bad_body_pos,foot_pos,extra),axis=1),
                carpet_heights)
            bad_z=signed[:,:5]
            foot_ground_distance=signed[:,5:7]
            virtual_z=signed[:,7:]
            # Noncontacting rods may be 8mm legacy proxies or the actual
            # 1mm-radius carbon shafts. Only their real radius applies here.
            leg_z=jnp.concatenate(
                (bad_z[:,3:5],virtual_z[:,1:]),axis=1)-self.rod_radius_m
            virtual_clearance=jnp.minimum(
                virtual_z[:,0],jnp.min(leg_z,axis=1))
            virtual_strike=virtual_clearance<-.005
            virtual_hard=virtual_clearance<-.020
            height=bad_z[:,0]
        else:
            # Exact original multi-query topology for meaningful A/B
            # against the 8-prism torso-contact GPU reference.
            height=self._support_signed_heights(
                data,data.xpos[:,self.torso_id:self.torso_id+1,:],
                carpet_heights)[:,0]
            foot_ground_distance=self._support_signed_heights(
                data,foot_pos,carpet_heights)
            bad_z=self._support_signed_heights(
                data,bad_body_pos,carpet_heights)
            virtual_clearance=jnp.ones((self.num_envs,),dtype=jnp.float32)
            virtual_strike=jnp.zeros((self.num_envs,),dtype=jnp.bool_)
            virtual_hard=jnp.zeros((self.num_envs,),dtype=jnp.bool_)
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
        # Healthy support and requested attitude, not a level camera.
        posture_ok = ((foot_mid_angle < 0.6108652) &
                      (jnp.abs(pitch_err) < 0.3141593) &
                      (jnp.abs(roll_err) < 0.2792527) &
                      (height_error > -0.040))
        attitude_quality = jnp.exp(-8.0 * (pitch_err * pitch_err + roll_err * roll_err))
        # 2. Contact, gait and fall proxies. Foot height indicates nearby
        # ground; it is NOT a verified load-bearing contact impulse.
        # The contact proxy uses sole clearance, not a hard-coded 18mm
        # sphere centre. Otherwise a 10mm foot receives fake support.
        foot_clearance = foot_ground_distance - self.foot_radius_m
        foot_contacts = jnp.sum((foot_clearance < .018).astype(jnp.float32), axis=1)
        support = jnp.clip(foot_contacts / 2.0, 0.0, 1.0)
        # This model has no knee/shin sentinel contact geoms, so a blunt
        # body-origin proxy alone is too strict a termination condition --
        # only actual height/tilt/rate collapse may terminate. Body-low
        # stays visible as a pressure signal (below) so strict validation
        # could be reintroduced later without poisoning early training.
        body_low_soft = jnp.any(bad_z[:, 1:] < SOFT_BODY_LOW_Z, axis=1).astype(jnp.float32)
        low_height_soft = (height < 0.130).astype(jnp.float32)
        bad_contact = jnp.maximum(BODY_LOW_REWARD_WEIGHT * body_low_soft, low_height_soft)
        if self.feet_only_contacts:
            # Penalise real geometric intrusion into the solid even when
            # the knee/thigh/torso is not permitted to produce an MJX impulse.
            bad_contact=jnp.maximum(
                bad_contact,jnp.where(virtual_strike,.80,0.))
        swing_clearance = jnp.maximum(0.0, jnp.max(foot_clearance, axis=1))
        q = data.qpos[:, self.act_qpos]
        # IMPORTANT: ±25deg carbon hinge limits are provisional. A policy
        # leaning on those numerical stops is not hardware-feasible walking.
        shin_angles=jnp.abs(data.qpos[:,self.shin_all_qpos])
        shin_near_stop=jnp.any(
            shin_angles >= .90*self.shin_all_limit_rad[None,:],axis=1
        ).astype(jnp.float32)
        joint_pressure = jnp.mean(jnp.maximum(0.0, jnp.abs(q - self.home_target) / (self.target_span * 1.20) - 0.72), axis=1)
        scissor = jnp.maximum(0.0, jnp.abs(q[:, 0] - q[:, 3]) - 0.62)
        action_delta = jnp.linalg.norm(action - prev_action, axis=1)
        # 3. Servo/spring diagnostics: measured mechanical proxies, NOT
        # payments. PPO currently penalises action changes and joint-limit
        # pressure instead of direct torque, shaft power or electrical usage.
        # Passive telemetry only. It never enters the deployable 360-input
        # observation or the scalar PPO reward.
        servo_effort = servo_effort_metrics(
            jnp, data.actuator_force[:, :ACTION_DIM],
            data.qvel[:, self.act_qvel], self.servo_actuator_torque_limit_nm)
        spring_energy = joint_spring_energy(
            jnp, data.qpos[:, self.spring_qpos],
            self.spring_reference, self.spring_stiffness)
        # Full spatial-tendon spring physics: old [L,L] is compression
        # support, new [0,L] is a tension-only knee return cable. Account
        # for each without crediting nonexistent slack-side spring energy.
        tendon_length=jnp.take(data.ten_length,self.knee_strut_ids,axis=1)
        strut_compression=jnp.maximum(0.,self.knee_strut_rest_m-tendon_length)
        strut_extension=jnp.maximum(0.,tendon_length-self.knee_strut_rest_max_m)
        strut_force=self.knee_strut_k*(strut_compression+strut_extension)
        strut_energy=.5*jnp.sum(self.knee_strut_k*(
            strut_compression**2+strut_extension**2),axis=1)
        spring_energy = {
            **spring_energy,
            'spring_potential_energy_j': (
                spring_energy['spring_potential_energy_j'] + strut_energy),
            'knee_strut_energy_j': strut_energy,
            'knee_strut_compression_m_right': strut_compression[:, 0],
            'knee_strut_compression_m_left': strut_compression[:, 1],
            'knee_support_extension_m_right':strut_extension[:,0],
            'knee_support_extension_m_left':strut_extension[:,1],
            'knee_support_tension_force_n_right':self.knee_strut_k[0]*strut_extension[:,0],
            'knee_support_tension_force_n_left':self.knee_strut_k[1]*strut_extension[:,1],
            'knee_strut_force_n_right': strut_force[:, 0],
            'knee_strut_force_n_left': strut_force[:, 1],
        }
        model_damping_dissipation_w = jnp.sum(
            self.act_joint_viscous_damping *
            data.qvel[:, self.act_qvel]**2, axis=-1)
        # Distinguish exhausted ACTION position range from actual MOTOR
        # force saturation at the two hip-rolls. Values are fractions of
        # the HARD RATED motor force ceiling (not peak stall torque).
        # Passive joint damping is separately recorded and is NOT motor
        # force, current or temperature.
        motor_roll_r = jnp.abs(data.actuator_force[:, 0])
        motor_roll_l = jnp.abs(data.actuator_force[:, 3])
        hip_roll_motor = dict(
            hip_roll_motor_load_frac_right=(
                motor_roll_r / jnp.maximum(self.servo_actuator_torque_limit_nm[..., 0], 1e-6)),
            hip_roll_motor_load_frac_left=(
                motor_roll_l / jnp.maximum(self.servo_actuator_torque_limit_nm[..., 3], 1e-6)),
            # Physical joint-centering spring, separately from actuator force.
            hip_roll_spring_abs_nm_right=(
                jnp.abs(self.spring_stiffness[0] *
                    (data.qpos[:, self.spring_qpos[0]] -
                     self.spring_reference[0]))),
            hip_roll_spring_abs_nm_left=(
                jnp.abs(self.spring_stiffness[3] *
                    (data.qpos[:, self.spring_qpos[3]] -
                     self.spring_reference[3]))),
        )
        # Reward-only stance-foot slip proxy. MuJoCo truth is allowed in the
        # reward: consecutive
        # foot poses estimate tangential speed and only near-floor feet count.
        foot_vel_xy = (data.xpos[:, self.foot_bodies, :2] - prev_foot_pos[:, :, :2]) / CONTROL_DT
        foot_speed = jnp.linalg.norm(foot_vel_xy, axis=2)
        stance_w = jnp.clip((.022 - foot_clearance) / .020, 0.0, 1.0)
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
        if self.feet_only_contacts:
            terminal=jnp.logical_or(terminal,virtual_hard)
        s_phase = jnp.sin(phase)
        r_expected = jnp.maximum(0.0, s_phase)
        l_expected = jnp.maximum(0.0, -s_phase)
        r_lift = jnp.clip(foot_clearance[:, 0] / jnp.maximum(.008, task[:, T.step_height]), 0.0, 1.0)
        l_lift = jnp.clip(foot_clearance[:, 1] / jnp.maximum(.008, task[:, T.step_height]), 0.0, 1.0)
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
            'fall_virtual_contact':virtual_hard.astype(jnp.float32),
            'shin_flex_near_stop':shin_near_stop,
            'virtual_nonfoot_strike':virtual_strike.astype(jnp.float32),
            'virtual_nonfoot_clearance_m':virtual_clearance,
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
            **hip_roll_motor,
            'actuator_joint_viscous_damping_w': model_damping_dissipation_w,
        }

    # ---- Actor observation: sensor/command contract, no teacher geometry ----
    def _base_obs(self, data, task, prev_action, phase, delivered_ctrl, sensor_state=None):
        """ONE deployable 360-input array-only observation.

        The old routine built a 220-input synthetic ground/tripod packet,
        queried the disabled floor, and then scattered raw 360-sensor
        values over four of its columns. Build the real input directly,
        using the same acquisition ring/normalizer semantics and field
        ordering, without privileged ground-plane geometry.
        """
        if sensor_state is None:
            raise ValueError('360-input actor requires an acquired 4x4 sensor state')
        jnp=self.jnp
        gravity_body,_,_=self._body_packets(data)
        q=data.qpos[:,self.act_qpos]
        qv=data.qvel[:,self.act_qvel]
        shin_flex=data.qpos[:,self.shin_qpos]
        shin_rate=data.qvel[:,self.shin_qvel]
        n0=jnp.sin(1.73*phase+0.6180339*self.env_index)
        n1=jnp.sin(2.31*phase+1.4142136*self.env_index+.7)
        ns=jnp.where(task[:,T.sensor_scale]>=0.,
                     task[:,T.sensor_scale],self.sensor_noise_scale)
        grav_est=gravity_body+ns[:,None]*.008*jnp.stack(
            (n0,n1,-.5*n0),axis=1)
        q_est=q+ns[:,None]*.0015*n0[:,None]
        qv_est=qv+ns[:,None]*.015*n1[:,None]
        R=self._rotation_body_to_world(data)
        foot_rel_world=data.xpos[:,self.foot_bodies,:] - (
            data.xpos[:,self.torso_id,None,:])
        foot_rel=jnp.einsum(
            'nij,nkj->nki',jnp.swapaxes(R,1,2),foot_rel_world
        ).reshape((self.num_envs,6))
        shin_flex_est=shin_flex+ns[:,None]*.001*jnp.stack((n0,n1),axis=1)
        shin_rate_est=shin_rate+ns[:,None]*.010*jnp.stack((n1,n0),axis=1)
        shin_load_est=jnp.clip(jnp.abs(shin_flex_est)/self.shin_sensor_fullscale_rad,0.,1.5)
        sensor=self._sensor_readout(sensor_state,grav_est)
        cmd=task[:,[T.vx,T.vy,T.yaw,T.height,T.pitch,T.roll,T.step_height]]
        phases=jnp.stack((jnp.sin(phase),jnp.cos(phase)),axis=1)
        base=jnp.concatenate((
            grav_est,
            sensor['measured_gyro_body_rad_s'],
            q_est,qv_est,prev_action,delivered_ctrl,foot_rel,
            sensor['velocity_body_xy_mps'],
            sensor['tof_estimated_depth_m'][:,None],
            sensor['velocity_confidence'][:,None],
            cmd,phases,shin_flex_est,shin_rate_est,shin_load_est,
            jnp.stack((
                sensor['ground_forward_grade_rad'],
                sensor['ground_sideways_grade_rad'],
                sensor['ground_visibility_confidence']),axis=1),
            sensor['terrain_ranges_m'],
            sensor['terrain_valid'].astype(jnp.float32)),axis=1)
        assert base.shape[1]==self.base_obs_dim,(
            f'Array-only sensor produced {base.shape[1]} channels, '
            f'contract requires {self.base_obs_dim}')
        return jnp.nan_to_num(
            base,nan=0.,posinf=10.,neginf=-10.).astype(jnp.float32)
