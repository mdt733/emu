"""EMU MJX environment: physics state, task sampling, reward and termination.

The environment deliberately trains a dynamically unstable biped. There is no
static-standing prerequisite: episodes begin near a commanded moving state and
the policy learns to keep itself upright by stepping.

Training mixes three regimes in the same vectorised rollout:

- STEADY: hold a commanded vx/vy/yaw.
- TRANSITION: change commands during the episode.
- RESILIENCE: cope with rough starts and horizontal pushes while moving.

POSTURE is a global background difficulty dial rather than a fourth regime.
Each regime owns one 0..1 difficulty scalar; the scalar expands a coherent set
of physical task parameters. The training-side RegimeGovernor coordinates
simultaneous expansions with a shared push budget so several individually safe
changes cannot accidentally create one oversized task-distribution jump.
Retreats are intentionally unrestricted.

The fast path shares one MJX model across environments. Per-environment state,
commands and external forces vary freely, but true per-env mass/geometry/floor
friction randomisation is outside this version's architecture. See handover.md
and history.md for the project rationale and evidence.
"""
from __future__ import annotations

from typing import Any, NamedTuple
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
_VERSION_FILE = _PROJECT_ROOT / 'VERSION'
try:
    VERSION = _VERSION_FILE.read_text(encoding='utf-8').strip()
except OSError as exc:
    raise RuntimeError(f'Missing VERSION file beside environment.py: {_VERSION_FILE}') from exc
if not VERSION or not VERSION.startswith('v'):
    raise RuntimeError(f'Invalid VERSION value {VERSION!r}; expected e.g. v5007')

ROBOT_VERSION = f'emu_{VERSION}'
ENV_VERSION = f'emu-{VERSION}-self-mapping-trainer'
PROGRAM_VERSION = f'emu-mjx-ppo-{VERSION}-self-mapping-trainer'
REGIME_STEADY = 0
REGIME_TRANSITION = 1
REGIME_RESILIENCE = 2
NUM_REGIMES = 3
REGIME_NAMES = ('steady', 'transition', 'resilience')
ACTION_DIM = 6
# Deployable actor observation. v5006 deliberately removes training-only
# teacher signals (regime id, exact contact/fall classifiers, phase grading and
# step-evidence composites) from the policy input. Every remaining field has a
# plausible real-robot source: IMU, servo telemetry, forward kinematics, shin
# flex Hall sensors, optical-flow/ToF velocity estimation, command/controller
# state. Reward/evaluation may still use perfect MuJoCo truth.
#
# gravity(3)+gyro(3)+q(6)+qv_est(6)+prev_action(6)+prev_ctrl(6)+foot_rel(6)+
# vxy_est(2)+tof_height(1)+vel_conf(1)+cmd(7)+phase(2)+shin_flex(2)+
# shin_rate(2)+shin_load(2) = 55.
BASE_OBS_DIM = 55
HISTORY_LEN = 4
# Fixed-size history used to model target-delivery latency. The buffer shape is
# independent of the configured latency so JIT shapes stay stable.
MAX_COMM_LATENCY_STEPS = 4
OBS_DIM = BASE_OBS_DIM * HISTORY_LEN
SIM_DT = 0.005
CONTROL_SUBSTEPS = 4
CONTROL_DT = SIM_DT * CONTROL_SUBSTEPS

ACTUATED_JOINTS = ['r_hip_roll', 'r_hip_pitch', 'r_knee', 'l_hip_roll', 'l_hip_pitch', 'l_knee']
FOOT_BODIES = ['r_foot', 'l_foot']
BAD_BODIES = ['torso', 'r_thigh', 'l_thigh', 'r_shin', 'l_shin']
TARGET_SPAN = np.array([0.38, 0.48, 0.58, 0.38, 0.48, 0.58], dtype=np.float32)
RESET_JOINT_NOISE = 0.004
RESET_QVEL_NOISE = 0.003
RESET_FORWARD_VELOCITY_BLEND = 1.0
TARGET_RATE_LIMIT_RAD_S_DEFAULT = 1.75  # v5004 equivalent: 0.035 rad / 20 ms
NEUTRAL_ROOT_Z = 0.2584
# Falling is diagnosed from body geometry/height rather than treating every
# non-foot contact as an immediate terminal. EMU is expected to recover through
# dynamic stepping, so termination should represent an unrecoverable fall rather
# than merely imperfect contact.
SOFT_BODY_LOW_Z = 0.035
HARD_BODY_FLOOR_Z = 0.012
HARD_BODY_TORSO_GATE_Z = 0.190
BODY_LOW_REWARD_WEIGHT = 0.25
EARLY_FALL_BASE_PENALTY = 65.0
EARLY_FALL_TIME_PENALTY = 0.10
# Anti-freezing gate: reward should not favour a motionless balance solution.
# The threshold only requires evidence that feet are actually lifting; it does
# not prescribe a cadence. A short grace period avoids judging the first few
# frames of an episode.
STEPPING_ACTIVITY_TARGET = 0.15
STEPPING_GATE_GRACE_S = 1.0
# Command-transition dynamics (see _sample_cmd_target / TRANSITION's own
# difficulty path in _transition_hold_and_jump).
TRANSITION_HOLD_MIN_STEPS = 45.0   # shortest command hold, reached when min_hold_s bottoms out
TRANSITION_HOLD_MAX_STEPS = 700.0  # longest hold, the starting value (~never resamples mid-episode)
TRANSITION_RAMP_MIN_STEPS = 4.0    # near-instant target change, the hardest end
TRANSITION_RAMP_MAX_STEPS = 120.0  # gentle ramp, the starting/easy value
CARRYOVER_MAX_PROB = 0.35          # at TRANSITION's hardest point, up to this fraction of resets carry over qvel
# Seed gait used as the project's basic capability reference. It is a slow
# moving gait, not a near-zero/static-standing task.
SEED_VX_LO = 0.020
SEED_VX_HI = 0.060
# RESILIENCE push parameters. External force is per-environment data, so it can
# vary cheaply in the shared-model MJX fast path. Mass/geometry/friction changes
# are model parameters and therefore require a different batched-model design.
PUSH_PROB_PER_STEP = 0.0015   # tiny per-step chance a new push starts, at resilience_push_n == PUSH_MAX_FORCE_N
PUSH_MAX_FORCE_N = 0.35       # peak horizontal push force (newtons); robot is ~0.5kg, sized to impart a
                               # recoverable nudge (tens of mm/s), not an instant launch
PUSH_MIN_DURATION_STEPS = 2
PUSH_MAX_DURATION_STEPS = 8
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
# RESILIENCE ceilings for the episode's initial disturbance. These are
# deliberately recoverable perturbations: ~20 deg pose tilt, a few cm of
# position error, and a moderate angular-velocity kick rather than arbitrary
# fallen configurations.
POSE_TILT_MAX_RAD = 0.35     # rad (~20deg) -- RESILIENCE: initial roll/pitch lean at spawn
POSITION_NOISE_MAX_M = 0.02  # m -- RESILIENCE: initial xy/height offset at spawn
ANGVEL_NOISE_MAX_RAD_S = 1.5 # rad/s -- RESILIENCE: initial angular velocity at spawn
TRANSITION_HOLD_MAX_S = TRANSITION_HOLD_MAX_STEPS * CONTROL_DT  # ~14.0s -- TRANSITION:
                                                                  # easiest end of hold-time
TRANSITION_HOLD_MIN_S = TRANSITION_HOLD_MIN_STEPS * CONTROL_DT  # ~0.9s -- hardest end
# TRANSITION and RESILIENCE use a fraction of STEADY's current command
# envelope so the regime-specific challenge is not always combined with the
# hardest possible locomotion command. Exposed in params for controlled tests.
REGIME_COMMAND_FRAC_DEFAULT = 0.5
# Short reward-settle window after reset. Termination remains active; only
# tracking score avoids over-interpreting a few frames of spawn transient.
SETTLE_STEPS = 25
# STEADY and RESILIENCE regimes never resample their command --
# cmd_hold_left is initialized to this for them, safely larger than any
# real episode length (max_episode_steps is ~700), so the resample branch
# in _step_impl simply never fires. int32-safe (max ~2.1 billion).
INFINITE_HOLD_STEPS = 1_000_000_000
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
TRANSITION_TRACKING_BOOST = 1.5
RESILIENCE_TOLERANCE_MULT = 3.0
RESILIENCE_TRACKING_RELIEF = 0.5

# Named indices into the params array (see EmuMJXEnv.params_array below).
# Bare integer indices into this array have already been a real source of
# silent bugs when hand-edited; named constants make a copy/paste or
# off-by-one error loud instead of quiet. 14 fields: four difficulty
# scalars (0-1, one per regime plus posture) that this file's own sampling
# code expands into physical ranges via linear interpolation, three
# regime-allocation fractions, and the small set of diagnostic/assist
# fields below.
P_STEADY_DIFFICULTY = 0
P_TRANSITION_DIFFICULTY = 1
P_RESILIENCE_DIFFICULTY = 2
P_POSTURE_DIFFICULTY = 3
P_REGIME_FRAC_STEADY = 4
P_REGIME_FRAC_TRANSITION = 5
P_REGIME_FRAC_RESILIENCE = 6
P_REGIME_COMMAND_FRAC = 7
P_ROOT_Z_OFFSET = 8
P_ASSIST_SCALE = 9
P_ASSIST_HIP_AMP = 10
P_ASSIST_KNEE_AMP = 11
P_ASSIST_SIGN = 12
P_CORE_REHEARSAL_FRAC = 13
# v5007: permanent rehearsal of the full, already-proven v5005 athletic
# command envelope. Unlike core_rehearsal this is not a seed-band crutch; it
# exists only to prevent a live frontier from erasing old command skills.
P_ATHLETIC_REHEARSAL_FRAC = 14
# Adaptive physical frontiers. These are REAL units, not 0-1 completion
# fractions. They may expand without an arbitrary locomotion ceiling.
P_FRONTIER_VX_HI = 15
P_FRONTIER_VX_LO = 16
P_FRONTIER_VY = 17
P_FRONTIER_YAW = 18
P_FRONTIER_PUSH_N = 19
P_FRONTIER_SENSOR_SCALE = 20
P_FRONTIER_COMBO_RADIUS = 21
P_FRONTIER_TRANSITION_HOLD_S = 22
P_FRONTIER_RAMP_STEPS = 23
# v5008 recovery/capability-atlas probe controls. NaN/-1 defaults mean
# 'normal training behaviour'. These live in the task parameter vector rather
# than the actor observation and are used only by deterministic held-out probes.
P_PROBE_CMD_VX = 24
P_PROBE_CMD_VY = 25
P_PROBE_CMD_YAW = 26
P_PROBE_TILT_RAD = 27
P_PROBE_LINVEL_MPS = 28
P_PROBE_ANGVEL_RAD_S = 29
P_PROBE_JOINT_RAD = 30
P_PROBE_PUSH_N = 31
P_PROBE_PUSH_STEPS = 32
# v5011 event-based capability-field teaching packet.  These are TRAINER-side
# task-sampling hints, never actor observations.  Fixed-size tables keep the JAX
# parameter shape stable while their contents may change every PPO update.
FIELD_NODE_ANCHORS = 24
FIELD_EDGE_ANCHORS = 24
FIELD_NODE_STRIDE = 5   # vx, vy, yaw, weight, recommended recovery severity
FIELD_EDGE_STRIDE = 7   # source vx/vy/yaw, target vx/vy/yaw, weight
P_FIELD_NODE_FRAC = 33
P_FIELD_NODE_START = 34
P_FIELD_EDGE_FRAC = P_FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE
P_FIELD_EDGE_START = P_FIELD_EDGE_FRAC + 1
P_DIM = P_FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE

# Task-vector columns. TASK_RESILIENCE stores a unitless disturbance fraction;
# _step_impl converts that fraction to Newtons exactly once.
TASK_REGIME = 0
TASK_VX = 1
TASK_VY = 2
TASK_YAW = 3
TASK_HEIGHT = 4
TASK_PITCH = 5
TASK_ROLL = 6
TASK_STEP_HEIGHT = 7
TASK_RESILIENCE = 8
TASK_PHASE_RATE = 9
# Immutable-at-birth diagnostics. These let the trainer compare episode
# outcomes against the curriculum difficulty that actually CREATED the
# episode, rather than the (possibly much newer) difficulty current when it
# finally ends. TASK_CORE_REHEARSAL marks the permanent easy-practice slice.
TASK_BIRTH_STEADY_D = 10
TASK_BIRTH_TRANSITION_D = 11
TASK_BIRTH_RESILIENCE_D = 12
TASK_BIRTH_POSTURE_D = 13
TASK_CORE_REHEARSAL = 14
# Full-envelope rehearsal flag. Actor never sees this teacher-only tag.
TASK_ATHLETIC_REHEARSAL = 15
# Per-episode sensor-error scale used by the deployable observation model.
# This is NOT exposed as an observation channel; it only controls corruption.
TASK_SENSOR_SCALE = 16
TASK_DIM = 17


class EnvState(NamedTuple):
    data: Any
    hist: Any
    rng: Any
    task: Any
    prev_action: Any
    prev_ctrl: Any
    # Reward-only previous foot pose for true stance-slip measurement.
    prev_foot_pos: Any
    # Ring buffer of the last MAX_COMM_LATENCY_STEPS raw (pre-slew-limit)
    # targets, shape (num_envs, MAX_COMM_LATENCY_STEPS, ACTION_DIM), most
    # recent last. Always maintained regardless of comm_latency_steps, so
    # this field's shape never depends on that constructor argument.
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


class EmuMJXEnv:
    """A JAX/MJX vector environment with all envs stepped in one device batch."""

    def __init__(self, robot_xml: str, num_envs: int, seed: int = 733, max_episode_steps: int = 700,
                 actuator_realism: bool = False, comm_latency_steps: int = 0,
                 push_max_force_n: float | None = None, stepping_activity_target: float | None = None,
                 speed_tolerance: float | None = None, legacy_rng_reuse: bool = False,
                 vx_hi_max: float = VX_HI_MAX, vx_lo_min: float = VX_LO_MIN,
                 vy_max_mps: float = VY_MAX_MPS_LIMIT, yaw_max_rad_s: float = YAW_MAX_RAD_S_LIMIT,
                 target_rate_limit_rad_s: float = TARGET_RATE_LIMIT_RAD_S_DEFAULT,
                 disable_target_slew: bool = False,
                 phase_speed_gain: float = 0.0, phase_rate_max_hz: float = 2.6,
                 sensor_noise_scale: float = 1.0, open_frontier: bool = False,
                 combo_mix_frac: float = 0.0, combo_p: float = 1.35, combo_probe_frac: float = 0.08,
                 phase_obs_scale: float = 1.0,
                 recovery_tilt_max_deg: float | None = None,
                 recovery_position_max_m: float | None = None,
                 recovery_linvel_max_mps: float = 0.0,
                 recovery_angvel_max_rad_s: float | None = None,
                 recovery_joint_max_deg: float = 0.0):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.robot_xml = str(robot_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_xml)
        self.mj_model.opt.timestep = SIM_DT
        # Performance hint from MJX docs: keep contacts explicit/simple.  The XML
        # already disables shin-rod contact and excludes hip/torso overlaps.
        self.mjx_model = mjx.put_model(self.mj_model)
        self.ids = self._resolve_ids()
        self.key0 = jax.random.PRNGKey(int(seed))

        # actuator_realism: this robot's servo is a 1:345 GEARED actuator
        # (confirmed against the datasheet), not the direct-drive assumption
        # this XML's armature value implies. Reflected rotor inertia at the
        # output shaft is armature ~= N^2 * J_rotor; for a small coreless
        # rotor (J_rotor ~ 1e-7 kg.m^2) and N=345, that is
        # 345^2 * 1e-7 ~= 0.012 kg.m^2, double the XML's 0.006. A geared
        # train of this class also has real Coulomb friction the XML has
        # none of; ~0.08 N.m (roughly 4% of stall torque) is a reasonable
        # placeholder pending an actual bench measurement, same honest
        # caveat as every other actuator number here. A single GLOBAL
        # correction, not per-env -- every env gets the same corrected
        # physical constant, so this needs no batched-model machinery at
        # all, unlike domain randomisation would.
        self.actuator_realism = bool(actuator_realism)
        if self.actuator_realism:
            act_dofs = np.asarray(self.ids['act_qvel'])
            dof_armature = np.array(self.mjx_model.dof_armature, copy=True)
            dof_frictionloss = np.array(self.mjx_model.dof_frictionloss, copy=True)
            dof_armature[act_dofs] = 0.012
            dof_frictionloss[act_dofs] = 0.08
            self.mjx_model = self.mjx_model.replace(
                dof_armature=jnp.asarray(dof_armature, dtype=jnp.float32),
                dof_frictionloss=jnp.asarray(dof_frictionloss, dtype=jnp.float32))

        # comm_latency_steps: see _step_impl. 0 (default) means instant
        # target delivery; >0 delays the slew-limiter's reference by that
        # many steps, reading into EnvState.prev_raw_target as a ring
        # buffer up to MAX_COMM_LATENCY_STEPS deep.
        if not (0 <= int(comm_latency_steps) <= MAX_COMM_LATENCY_STEPS):
            raise ValueError(f'comm_latency_steps must be in [0, {MAX_COMM_LATENCY_STEPS}], '
                              f'got {comm_latency_steps}')
        self.comm_latency_steps = int(comm_latency_steps)

        # Per-instance, CLI-overridable versions of a few module-level
        # defaults (None = use the module constant unchanged). Each is read
        # via self.* at its use site below instead of the bare module name.
        self.push_max_force_n = float(push_max_force_n) if push_max_force_n is not None else PUSH_MAX_FORCE_N
        self.stepping_activity_target = (float(stepping_activity_target) if stepping_activity_target is not None
                                          else STEPPING_ACTIVITY_TARGET)
        self.speed_tolerance = float(speed_tolerance) if speed_tolerance is not None else 0.025
        self.vx_hi_max = float(vx_hi_max)
        self.vx_lo_min = float(vx_lo_min)
        self.vy_max_mps = float(vy_max_mps)
        self.yaw_max_rad_s = float(yaw_max_rad_s)
        self.target_rate_limit_rad_s = float(target_rate_limit_rad_s)
        self.disable_target_slew = bool(disable_target_slew)
        self.phase_speed_gain = float(phase_speed_gain)
        self.phase_rate_max_hz = float(phase_rate_max_hz)
        self.sensor_noise_scale = float(sensor_noise_scale)
        self.open_frontier = bool(open_frontier)
        self.combo_mix_frac = float(max(0.0, min(1.0, combo_mix_frac)))
        self.combo_p = float(max(1.0, combo_p))
        self.combo_probe_frac = float(max(0.0, combo_probe_frac))
        self.phase_obs_scale = float(max(0.0, min(1.0, phase_obs_scale)))
        # v5008 stage 5: one resilience difficulty scalar still owns recovery
        # progression, while these physical maxima define what "difficulty=1"
        # means for a particular experiment family. Defaults reproduce the
        # v5007/v5008 reference exactly; new linear/joint disorder are opt-in.
        self.recovery_tilt_max_rad = float(np.deg2rad(recovery_tilt_max_deg)) if recovery_tilt_max_deg is not None else POSE_TILT_MAX_RAD
        self.recovery_position_max_m = float(recovery_position_max_m) if recovery_position_max_m is not None else POSITION_NOISE_MAX_M
        self.recovery_linvel_max_mps = max(0.0, float(recovery_linvel_max_mps))
        self.recovery_angvel_max_rad_s = float(recovery_angvel_max_rad_s) if recovery_angvel_max_rad_s is not None else ANGVEL_NOISE_MAX_RAD_S
        self.recovery_joint_max_rad = max(0.0, float(np.deg2rad(recovery_joint_max_deg)))
        # A/B flag for an RNG-key-reuse bug fixed in _step_impl (push vs.
        # next/reset/resample) and _sample_task (roll vs. step_height) --
        # see both call sites for what was wrong and why. Off by default:
        # this is a correctness fix, not a design choice to weigh.
        self.legacy_rng_reuse = bool(legacy_rng_reuse)

        self.target_span = jnp.asarray(TARGET_SPAN, dtype=jnp.float32)
        self.home_target = jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
        self.act_qpos = jnp.asarray(self.ids['act_qpos'], dtype=jnp.int32)
        self.act_qvel = jnp.asarray(self.ids['act_qvel'], dtype=jnp.int32)
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
        self.forward_batched = jax.vmap(lambda d: mjx.forward(self.mjx_model, d))
        self.step_batched = jax.vmap(lambda d: mjx.step(self.mjx_model, d))

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
        }

    def params_array(self, params: dict[str, float] | None = None):
        # Defaults put a fresh env at "all steady, zero difficulty
        # everywhere" -- the bootstrapped-from-nothing starting point.
        # Order here MUST match the P_* named indices above.
        jnp = self.jnp
        params = params or {}
        base = [
            float(params.get('steady_difficulty', 0.0)),           # P_STEADY_DIFFICULTY
            float(params.get('transition_difficulty', 0.0)),       # P_TRANSITION_DIFFICULTY
            float(params.get('resilience_difficulty', 0.0)),       # P_RESILIENCE_DIFFICULTY
            float(params.get('posture_difficulty', 0.0)),          # P_POSTURE_DIFFICULTY
            float(params.get('regime_frac_steady', 1.0)),          # P_REGIME_FRAC_STEADY
            float(params.get('regime_frac_transition', 0.0)),      # P_REGIME_FRAC_TRANSITION
            float(params.get('regime_frac_resilience', 0.0)),      # P_REGIME_FRAC_RESILIENCE
            float(params.get('regime_command_frac', REGIME_COMMAND_FRAC_DEFAULT)),  # P_REGIME_COMMAND_FRAC
            # Diagnostic/assist extras: let diagnostics sweep support
            # height and a tiny phase scaffold without touching the
            # observation/action contract or adding reward complexity.
            float(params.get('root_z_offset', 0.0)),               # P_ROOT_Z_OFFSET
            float(params.get('assist_scale', 0.0)),                # P_ASSIST_SCALE
            float(params.get('assist_hip_amp', 0.14)),             # P_ASSIST_HIP_AMP
            float(params.get('assist_knee_amp', 0.22)),            # P_ASSIST_KNEE_AMP
            float(params.get('assist_sign', 1.0)),                 # P_ASSIST_SIGN
            float(params.get('core_rehearsal_frac', 0.0)),         # P_CORE_REHEARSAL_FRAC
            float(params.get('athletic_rehearsal_frac', 0.0)),     # P_ATHLETIC_REHEARSAL_FRAC
            float(params.get('frontier_vx_hi', 0.60)),              # P_FRONTIER_VX_HI
            float(params.get('frontier_vx_lo', -0.20)),             # P_FRONTIER_VX_LO
            float(params.get('frontier_vy', 0.22)),                 # P_FRONTIER_VY
            float(params.get('frontier_yaw', 1.50)),                # P_FRONTIER_YAW
            float(params.get('frontier_push_n', 0.70)),             # P_FRONTIER_PUSH_N
            float(params.get('frontier_sensor_scale', getattr(self, 'sensor_noise_scale', 1.5))), # P_FRONTIER_SENSOR_SCALE
            float(params.get('frontier_combo_radius', 1.0)),          # P_FRONTIER_COMBO_RADIUS
            float(params.get('frontier_transition_hold_s', TRANSITION_HOLD_MIN_S)), # P_FRONTIER_TRANSITION_HOLD_S
            float(params.get('frontier_ramp_steps', TRANSITION_RAMP_MIN_STEPS)),    # P_FRONTIER_RAMP_STEPS
            float(params.get('probe_cmd_vx', np.nan)),              # P_PROBE_CMD_VX
            float(params.get('probe_cmd_vy', np.nan)),              # P_PROBE_CMD_VY
            float(params.get('probe_cmd_yaw', np.nan)),             # P_PROBE_CMD_YAW
            float(params.get('probe_tilt_rad', -1.0)),              # P_PROBE_TILT_RAD
            float(params.get('probe_linvel_mps', -1.0)),            # P_PROBE_LINVEL_MPS
            float(params.get('probe_angvel_rad_s', -1.0)),          # P_PROBE_ANGVEL_RAD_S
            float(params.get('probe_joint_rad', -1.0)),             # P_PROBE_JOINT_RAD
            float(params.get('probe_push_n', -1.0)),                 # P_PROBE_PUSH_N
            float(params.get('probe_push_steps', 5.0)),              # P_PROBE_PUSH_STEPS
        ]
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
        assert packed.shape[0] == P_DIM, (packed.shape, P_DIM)
        return jnp.asarray(packed, dtype=jnp.float32)

    def reset(self, params: dict[str, float] | None = None):
        p = self.params_array(params)
        state = self.reset_jit(self.key0, p)
        return state, self.obs(state)

    def step(self, state: EnvState, action, params: dict[str, float] | Any) -> StepOut:
        p = self.params_array(params) if isinstance(params, dict) or params is None else params
        return self.step_jit(state, action, p)

    def obs(self, state: EnvState):
        return state.hist.reshape((self.num_envs, OBS_DIM))

    # ---------------- JAX-only implementation below ----------------

    def _split_batch(self, keys, n: int):
        jax = self.jax
        split = jax.vmap(lambda k: jax.random.split(k, n))(keys)
        return [split[:, i, :] for i in range(n)]

    def _make_batched_data(self):
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        def one(_):
            return mjx.make_data(self.mjx_model)
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

    def _lerp(self, seed, ceiling, difficulty):
        """The one function every regime's difficulty scalar goes through
        to become a real physical range -- plain linear interpolation from
        seed to ceiling (see the module doctrine for why one shared scalar
        with no per-component rate table is enough)."""
        jnp = self.jnp
        return seed + jnp.clip(difficulty, 0.0, 1.0) * (ceiling - seed)

    def _field_node_table(self, params):
        return params[P_FIELD_NODE_START:P_FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE].reshape((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE))

    def _field_edge_table(self, params):
        return params[P_FIELD_EDGE_START:P_FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE].reshape((FIELD_EDGE_ANCHORS, FIELD_EDGE_STRIDE))

    def _weighted_anchor_index(self, key, weights):
        jax, jnp = self.jax, self.jnp
        w = jnp.maximum(weights, 0.0)
        total = jnp.sum(w)
        u = jax.random.uniform(key, ()) * jnp.maximum(total, 1e-6)
        idx = jnp.sum(u > jnp.cumsum(w)).astype(jnp.int32)
        return jnp.clip(idx, 0, w.shape[0]-1), total

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
        steady_d = params[P_STEADY_DIFFICULTY]
        posture_d = params[P_POSTURE_DIFFICULTY]
        resilience_d = params[P_RESILIENCE_DIFFICULTY]
        command_frac = jnp.clip(params[P_REGIME_COMMAND_FRAC], 0.0, 1.0)

        # STEADY's full current envelope -- every regime's command draw is
        # some fraction of THIS, never a separately-scheduled range of its
        # own (see module doctrine: isolate which axis is under test from
        # the aggregate difficulty of everything else).
        if self.open_frontier:
            # v5006: physical frontier values are live curriculum state, not
            # percentages of a programmer-declared final ceiling.
            vx_hi = jnp.maximum(SEED_VX_HI, params[P_FRONTIER_VX_HI])
            vx_lo = jnp.minimum(SEED_VX_LO, params[P_FRONTIER_VX_LO])
            yaw_max = jnp.maximum(0.0, params[P_FRONTIER_YAW])
            vy_max = jnp.maximum(0.0, params[P_FRONTIER_VY])
        else:
            vx_hi = self._lerp(SEED_VX_HI, self.vx_hi_max, steady_d)
            vx_lo = self._lerp(SEED_VX_LO, self.vx_lo_min, steady_d)
            yaw_max = self._lerp(0.0, self.yaw_max_rad_s, steady_d)
            vy_max = self._lerp(0.0, self.vy_max_mps, steady_d)
        # TRANSITION/RESILIENCE's own envelope: the same shape, anchored at
        # each axis's own seed point (0 for yaw/vy, SEED_VX_* for vx) and
        # scaled toward the current full envelope by command_frac. frac=0
        # is "right at seed", frac=1 is "exactly steady's own envelope".
        vx_hi_reduced = SEED_VX_HI + command_frac * (vx_hi - SEED_VX_HI)
        vx_lo_reduced = SEED_VX_LO + command_frac * (vx_lo - SEED_VX_LO)
        yaw_max_reduced = command_frac * yaw_max
        vy_max_reduced = command_frac * vy_max

        # POSTURE (global, every regime): height/pitch/roll/step-height are
        # a coupled bundle, all driven by the one posture_difficulty
        # fraction directly.
        height_range = self._lerp(0.0, HEIGHT_RANGE_MAX_M, posture_d)
        pitch_range = 0.08 * posture_d
        roll_range = 0.055 * posture_d
        step_height_range = 0.030 * jnp.maximum(0.05, posture_d)

        # RESILIENCE difficulty remains unitless here. Non-resilience
        # regimes still get a small AMBIENT fraction of it
        # (REGULAR_RESILIENCE_FRAC), and _step_impl converts that fraction
        # to Newtons exactly once at the force-application site.

        regime_probs = jnp.asarray([params[P_REGIME_FRAC_STEADY], params[P_REGIME_FRAC_TRANSITION],
                                    params[P_REGIME_FRAC_RESILIENCE]], dtype=jnp.float32)
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
            k_regime, k_vx, k_vy, k_yaw, k_height, k_pitch, k_roll, k_step, k_phase, k_rehearsal, k_athletic, k_combo, k_axis, k_radius, k_field_use, k_field_pick = jax.random.split(k, 16)
            u = jax.random.uniform(k_regime, ())
            regime = jnp.sum(u > regime_cdf).astype(jnp.float32)
            is_rehearsal = jax.random.uniform(k_rehearsal, ()) < jnp.clip(params[P_CORE_REHEARSAL_FRAC], 0.0, 0.5)
            # v5007 full-envelope rehearsal is mutually exclusive with the
            # seed-band core rehearsal. It preserves the known v5005 command
            # distribution while the adaptive frontier explores beyond it.
            is_athletic = jnp.logical_and(
                jnp.logical_not(is_rehearsal),
                jax.random.uniform(k_athletic, ()) < jnp.clip(params[P_ATHLETIC_REHEARSAL_FRAC], 0.0, 0.5))
            # Both rehearsal cohorts are STEADY so they never contaminate the
            # transition/push difficulty experiments.
            regime = jnp.where(jnp.logical_or(is_rehearsal, is_athletic), 0.0, regime)
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

            # v5006 command-combination curriculum. A configurable fraction of
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
                norm = jnp.power(jnp.abs(nx)**self.combo_p + jnp.abs(ny)**self.combo_p + jnp.abs(nz)**self.combo_p, 1.0/self.combo_p)
                combo_radius = jnp.where(self.open_frontier, jnp.maximum(0.10, params[P_FRONTIER_COMBO_RADIUS]), 1.0)
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
            # v5011 self-mapping trainer: a bounded fraction of ordinary births
            # may be drawn from the capability field's current teaching packet.
            # Anchors are clipped to the governor's current envelope; the field
            # chooses *where* to spend practice while the governor still owns
            # global safety/frontier expansion.
            node_table = self._field_node_table(params)
            ni, node_total = self._weighted_anchor_index(k_field_pick, node_table[:, 3])
            node_cmd = node_table[ni, :3]
            node_recovery = jnp.clip(node_table[ni, 4], 0.0, 1.0)
            field_use = jnp.logical_and(
                jax.random.uniform(k_field_use, ()) < jnp.clip(params[P_FIELD_NODE_FRAC], 0.0, 0.95),
                node_total > 1e-6)
            field_use = jnp.logical_and(field_use, jnp.logical_not(jnp.logical_or(is_rehearsal, is_athletic)))
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
            speed_fraction = jnp.clip(jnp.sqrt(vx*vx + vy*vy) / jnp.maximum(0.05, self.vx_hi_max), 0.0, 1.5)
            phase_rate = jnp.minimum(self.phase_rate_max_hz, base_phase_rate + self.phase_speed_gain * speed_fraction)
            # Keep disturbance intensity unitless in the task vector. Convert it
            # to Newtons only where the external force is applied.
            # The field may choose a locally useful recovery severity for a
            # resilience birth, but never above the governor's globally-earned
            # resilience difficulty.  This turns the recovery layer into an
            # actual curriculum signal while preserving the governor as the
            # physical safety ceiling.
            birth_resilience_d = jnp.where(
                jnp.logical_and(field_use, regime > 1.5),
                jnp.minimum(node_recovery, jnp.clip(resilience_d, 0.0, 1.0)),
                jnp.clip(resilience_d, 0.0, 1.0))
            if self.open_frontier:
                resilience_frac = jnp.clip(params[P_FRONTIER_PUSH_N] / jnp.maximum(self.push_max_force_n, 1e-6), 0.0, 1.0)
            else:
                resilience_frac = birth_resilience_d
            resilience_task = jnp.where(regime > 1.5, jnp.maximum(resilience_frac, 0.08 / jnp.maximum(self.push_max_force_n, 1e-6)),
                                        resilience_frac * REGULAR_RESILIENCE_FRAC)
            sensor_scale_task = jnp.maximum(0.0, params[P_FRONTIER_SENSOR_SCALE])
            # Full athletic rehearsal samples the proven v5005/v5006 control
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

            # Permanent core rehearsal overrides every difficulty-bearing
            # component but keeps the same dynamic-birth doctrine: non-zero
            # seed-band forward command, not standing still.
            vx = jnp.where(is_rehearsal, jax.random.uniform(k_vx, (), minval=SEED_VX_LO, maxval=SEED_VX_HI), vx)
            vy = jnp.where(is_rehearsal, 0.0, vy)
            yaw = jnp.where(is_rehearsal, 0.0, yaw)
            # Capability-atlas command probes use exact physical commands,
            # bypassing random envelope sampling without changing the actor.
            vx = jnp.where(jnp.isfinite(params[P_PROBE_CMD_VX]), params[P_PROBE_CMD_VX], vx)
            vy = jnp.where(jnp.isfinite(params[P_PROBE_CMD_VY]), params[P_PROBE_CMD_VY], vy)
            yaw = jnp.where(jnp.isfinite(params[P_PROBE_CMD_YAW]), params[P_PROBE_CMD_YAW], yaw)
            height = jnp.where(is_rehearsal, 0.0, height)
            pitch = jnp.where(is_rehearsal, 0.0, pitch)
            roll = jnp.where(is_rehearsal, 0.0, roll)
            step_height = jnp.where(is_rehearsal, 0.012, step_height)
            resilience_task = jnp.where(is_rehearsal, 0.0, resilience_task)

            cols = [regime, vx, vy, yaw, height, pitch, roll, step_height, resilience_task, phase_rate,
                    steady_d, params[P_TRANSITION_DIFFICULTY], birth_resilience_d, posture_d, is_rehearsal.astype(jnp.float32),
                    is_athletic.astype(jnp.float32), sensor_scale_task]
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
        steady_d = params[P_STEADY_DIFFICULTY]
        command_frac = jnp.clip(params[P_REGIME_COMMAND_FRAC], 0.0, 1.0)
        _, jump_frac = self._transition_hold_and_jump(params[P_TRANSITION_DIFFICULTY])
        if self.open_frontier:
            jump_frac = jnp.asarray(1.0, dtype=jnp.float32)
        if self.open_frontier:
            vx_hi_full = jnp.maximum(SEED_VX_HI, params[P_FRONTIER_VX_HI])
            vx_lo_full = jnp.minimum(SEED_VX_LO, params[P_FRONTIER_VX_LO])
            yaw_full = jnp.maximum(0.0, params[P_FRONTIER_YAW])
            vy_full = jnp.maximum(0.0, params[P_FRONTIER_VY])
            vx_hi = SEED_VX_HI + command_frac * (vx_hi_full - SEED_VX_HI)
            vx_lo = SEED_VX_LO + command_frac * (vx_lo_full - SEED_VX_LO)
            yaw_max = command_frac * yaw_full
            vy_max = command_frac * vy_full
        else:
            vx_hi = SEED_VX_HI + command_frac * (self._lerp(SEED_VX_HI, self.vx_hi_max, steady_d) - SEED_VX_HI)
            vx_lo = SEED_VX_LO + command_frac * (self._lerp(SEED_VX_LO, self.vx_lo_min, steady_d) - SEED_VX_LO)
            yaw_max = command_frac * self._lerp(0.0, self.yaw_max_rad_s, steady_d)
            vy_max = command_frac * self._lerp(0.0, self.vy_max_mps, steady_d)
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
            use_node = jnp.logical_and(jax.random.uniform(k_node_use, ()) < jnp.clip(params[P_FIELD_NODE_FRAC], 0.0, 0.95), node_total > 1e-6)

            # Graph anchors are local source->target maneuvers.  Prefer anchors
            # whose source is near this env's current command, without adding a
            # high-dimensional table to the actor or policy network.
            scale = jnp.asarray([0.90, 0.55, 3.0], dtype=jnp.float32)
            dist2 = jnp.sum(((edge_table[:, :3] - cur[None, :]) / scale[None, :]) ** 2, axis=1)
            local_w = jnp.maximum(edge_table[:, 6], 0.0) * jnp.exp(-0.5 * dist2 / (0.45**2))
            ei, edge_total = self._weighted_anchor_index(k_edge_pick, local_w)
            edge_target = edge_table[ei, 3:6]
            use_edge = jnp.logical_and(jax.random.uniform(k_edge_use, ()) < jnp.clip(params[P_FIELD_EDGE_FRAC], 0.0, 0.95), edge_total > 1e-6)

            target = jnp.where(use_node, node_target, base)
            target = jnp.where(use_edge, edge_target, target)  # graph has priority
            target = target.at[0].set(jnp.clip(target[0], vx_lo_j, vx_hi_j))
            target = target.at[1].set(jnp.clip(target[1], -vy_max_j, vy_max_j))
            target = target.at[2].set(jnp.clip(target[2], -yaw_max_j, yaw_max_j))
            target = target.at[0].set(jnp.where(jnp.isfinite(params[P_PROBE_CMD_VX]), params[P_PROBE_CMD_VX], target[0]))
            target = target.at[1].set(jnp.where(jnp.isfinite(params[P_PROBE_CMD_VY]), params[P_PROBE_CMD_VY], target[1]))
            target = target.at[2].set(jnp.where(jnp.isfinite(params[P_PROBE_CMD_YAW]), params[P_PROBE_CMD_YAW], target[2]))
            return target
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
        cmd_target = task[:, [TASK_VX, TASK_VY, TASK_YAW]]
        cmd_ramp_from = cmd_target
        cmd_ramp_frac = jnp.ones((self.num_envs,), dtype=jnp.float32)
        cmd_ramp_speed = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        is_transition = jnp.logical_and(task[:, TASK_REGIME] > 0.5, task[:, TASK_REGIME] < 1.5)
        min_hold_s, _ = self._transition_hold_and_jump(params[P_TRANSITION_DIFFICULTY])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P_FRONTIER_TRANSITION_HOLD_S])
        hold_min = min_hold_s / CONTROL_DT  # steps
        if hold_keys is None:
            transition_hold = jnp.full((self.num_envs,), hold_min * 0.8, dtype=jnp.int32)
        else:
            hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(hold_keys)
            transition_hold = jnp.round(hold_min * (0.6 + 0.4 * hold_draw)).astype(jnp.int32)
        cmd_hold_left = jnp.where(is_transition, transition_hold, INFINITE_HOLD_STEPS)
        return cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left

    def _birth_assist_target(self, task, phase, params):
        """Small optional CPG scaffold for diagnostics only.

        Ball feet are not statically stable; a zero-action standing pose
        always height-collapses, so the default birth path is organic PPO
        discovery from a forward-moving seed instead. This assist profile
        exists only to diagnose sign/phase/mechanical authority if a long
        organic run stays flat -- assist_scale defaults to 0.0 (off) in
        real training.
        """
        jnp = self.jnp
        scale = jnp.clip(params[P_ASSIST_SCALE], 0.0, 1.0)
        hip_amp = jnp.clip(params[P_ASSIST_HIP_AMP], 0.0, 0.40)
        knee_amp = jnp.clip(params[P_ASSIST_KNEE_AMP], 0.0, 0.55)
        sign = jnp.where(params[P_ASSIST_SIGN] >= 0.0, 1.0, -1.0)
        s = jnp.sin(phase)
        # Alternating hip-pitch and knee-lift scaffold.  Roll joints are left to
        # the policy/springs so the assist cannot hide side-balance failure.
        r_swing = jnp.maximum(0.0, s)
        l_swing = jnp.maximum(0.0, -s)
        r_hip = sign * hip_amp * s
        l_hip = -sign * hip_amp * s
        r_knee = knee_amp * r_swing
        l_knee = knee_amp * l_swing
        zeros = jnp.zeros_like(s)
        assist = jnp.stack([zeros, r_hip, r_knee, zeros, l_hip, l_knee], axis=1)
        return scale * assist

    def _step_impl(self, state: EnvState, action, params) -> StepOut:
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        action = jnp.clip(action, -1.0, 1.0)
        assist = self._birth_assist_target(state.task, state.phase, params)
        raw_target = self.home_target + action * self.target_span + assist
        raw_target = jnp.clip(raw_target, -self.target_span, self.target_span)
        # Optional, fixed, always-on communication latency: the simplest
        # honest model of "the servo hasn't received the newest target yet"
        # is that the slew-limiter chases an EARLIER step's commanded
        # target, not this step's -- how much earlier is comm_latency_steps.
        # comm_latency_steps is a plain Python int fixed at env construction
        # (not a per-step JAX value), so this is static indexing, resolved
        # at trace time -- no dynamic_slice needed. index=-1 (most recent
        # buffer slot) is last step's raw_target; more negative indices
        # reach further back. comm_latency_steps=0 means no delay at all
        # (the buffer is filled but never read).
        if self.comm_latency_steps > 0:
            slew_target = state.prev_raw_target[:, MAX_COMM_LATENCY_STEPS - self.comm_latency_steps, :]
        else:
            slew_target = raw_target
        max_delta = (self.target_rate_limit_rad_s * CONTROL_DT) * (0.75 + 0.25 * jnp.clip(state.task[:, TASK_RESILIENCE], 0.0, 1.0))
        slew_ctrl = state.prev_ctrl + jnp.clip(slew_target - state.prev_ctrl, -max_delta[:, None], max_delta[:, None])
        # v5007 can remove the artificial software slew layer entirely. In
        # that mode the requested position target is delivered directly and
        # MuJoCo's geared-servo dynamics decide how fast the joint can move.
        ctrl = slew_target if self.disable_target_slew else slew_ctrl
        data = state.data.replace(ctrl=state.data.ctrl.at[:, :ACTION_DIM].set(ctrl))

        # Push perturbations (RESILIENCE), sized to be a recoverable nudge
        # rather than an instant launch. Applied to `data` BEFORE the
        # physics substep loop -- applying a push after stepping physics
        # and clearing it before the next step would give it zero actual
        # effect, so this ordering is deliberate. Uses the PER-EPISODE
        # resilience level (state.task's TASK_RESILIENCE column, set in
        # _sample_task) instead of the raw
        # global curriculum dial directly -- this is what makes
        # REGULAR_RESILIENCE_FRAC actually take effect on the physics, not
        # just on what the policy is told.
        resilience_frac = jnp.clip(state.task[:, TASK_RESILIENCE], 0.0, 1.0)
        resilience_push_n = resilience_frac * self.push_max_force_n
        # Split the per-environment RNG once, then dedicate each key to one
        # random purpose. Re-splitting the same unadvanced JAX key can reuse
        # identical key material. The compatibility flag intentionally
        # reproduces that older path for regression/A-B work only.
        if self.legacy_rng_reuse:
            push_rng, angle_key, mag_key, dur_key = self._split_batch(state.rng, 4)
        else:
            push_rng, angle_key, mag_key, dur_key, next_keys, reset_keys, resample_keys = self._split_batch(state.rng, 7)
        push_roll = jax.vmap(lambda k: jax.random.uniform(k, ()))(push_rng)
        push_active = state.push_steps_left > 0
        # Capability-cloud/atlas probe: one controlled horizontal shove shortly
        # after birth. Magnitude and duration are explicit physical quantities;
        # direction remains random across envs so one cloud point estimates a
        # distribution of nuisance directions instead of adding push-angle as
        # yet another map dimension. Negative probe_push_n disables this path.
        probe_push_n = params[P_PROBE_PUSH_N]
        probe_push_steps = jnp.maximum(1.0, params[P_PROBE_PUSH_STEPS])
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
            return self.step_batched(d)
        data = jax.lax.fori_loop(0, CONTROL_SUBSTEPS, substep, data)

        # Physics occasionally goes non-finite (NaN/Inf in qpos/qvel) on a
        # long run -- rare, exact trigger not pinned down, but worth
        # guarding against unconditionally rather than assuming it can't
        # happen. Every fall/termination check below (fall_height,
        # fall_tilt, fall_roll_rate, fall_pitch_rate) is a plain threshold
        # comparison, and IEEE-754 defines ANY comparison against NaN as
        # False. That makes a diverged env UNKILLABLE without this guard:
        # it can never be flagged as fallen, so it would silently time out
        # every episode forever, poisoning every batch-wide statistic --
        # and once such a transition enters a PPO update, the shared
        # network's gradient (a batch-wide reduction) goes NaN and the
        # optimizer writes that into every weight, permanently.
        #
        # Fix: detect non-finite qpos/qvel immediately, from the raw
        # physics state, before anything else touches it. Roll the affected
        # envs' `data` back to their pre-step (still-finite) state -- so
        # every metric/reward/observation computed below for them is clean
        # -- and force those envs into `terminal_fail` below, so they get a
        # real reset (like any other fall) instead of running broken.
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
        step_i = state.step_i + 1
        phase = (state.phase + CONTROL_DT * state.task[:, TASK_PHASE_RATE] * 2.0 * jnp.pi) % (2.0 * jnp.pi)

        # Command-transition dynamics: ramps the *tracked* command
        # (task[:, VX/VY/YAW], which every other function below reads)
        # smoothly toward an occasionally-resampled target instead of
        # Command hold/ramp timing tightens with transition difficulty. The
        # compatibility flag can deliberately reproduce the older shared-key
        # resampling path for controlled A/B work; the default uses keys from
        # the single top-of-step split.
        if self.legacy_rng_reuse:
            next_keys, reset_keys, resample_keys = self._split_batch(state.rng, 3)
        ramp_frac = jnp.clip(state.cmd_ramp_frac + state.cmd_ramp_speed, 0.0, 1.0)
        cmd_now = state.cmd_ramp_from + ramp_frac[:, None] * (state.cmd_target - state.cmd_ramp_from)
        task = state.task.at[:, TASK_VX].set(cmd_now[:, 0]).at[:, TASK_VY].set(cmd_now[:, 1]).at[:, TASK_YAW].set(cmd_now[:, 2])

        hold_left = state.cmd_hold_left - 1
        should_resample = hold_left <= 0
        rk_target, rk_hold, rk_ramp = self._split_batch(resample_keys, 3)
        # Only TRANSITION-regime envs ever have a finite cmd_hold_left (see
        # _init_transition_state) -- STEADY/RESILIENCE's INFINITE_HOLD_STEPS
        # means should_resample is always False for them, so new_target/
        # new_hold below are computed for every env (vmap has no branch)
        # but only ever actually USED where should_resample is True, i.e.
        # only ever for TRANSITION envs.
        new_target = self._sample_cmd_target(rk_target, params, cmd_now)
        hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_hold)
        ramp_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_ramp)
        min_hold_s, _ = self._transition_hold_and_jump(params[P_TRANSITION_DIFFICULTY])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P_FRONTIER_TRANSITION_HOLD_S])
        hold_min = min_hold_s / CONTROL_DT  # steps
        new_hold = jnp.maximum(1.0, jnp.round(hold_min * (0.6 + 0.4 * hold_draw))).astype(jnp.int32)
        if self.open_frontier:
            # Explicit acceleration frontier: ramp duration is a real count of
            # 20ms control frames and may shrink toward one frame, the natural
            # digital limit. No arbitrary locomotion-speed ceiling is involved.
            ramp_min = jnp.maximum(1.0, params[P_FRONTIER_RAMP_STEPS])
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

        # Score this action against the command it actually OBSERVED. `task`
        # above is the next-step command after advancing the transition ramp;
        # using it here punishes an action for information not present in its
        # input. The next observation still receives `task`, below.
        reward_task = state.task
        metrics = self._metrics(data, reward_task, action, state.prev_action, phase, state.prev_foot_pos)
        # Stepping-activity-so-far, injected into metrics before _reward
        # runs so the reward's anti-freezing gate can see it. Uses the
        # PRE-this-step accumulator (how the episode has looked up to but
        # not including now) -- correct, not an off-by-one: a step
        # currently in a normal stance phase shouldn't be judged on its own
        # instantaneous zero-lift, only on the episode's broader pattern.
        stepping_activity_so_far = state.ep_step_evidence_sum / jnp.maximum(1.0, state.ep_len)
        # ramp_frac (computed above, moving cmd_now toward cmd_target) is
        # also injected into metrics here, alongside age_s/
        # stepping_activity_so_far, so _reward can weight TRANSITION's
        # tracking term differently while a command change is actually in
        # flight -- training directly on "how well and how quickly it
        # transitions", not just measuring it after the fact.
        # Episode AGE comes from ep_len, not step_i: step_i is the timeout clock
        # and is staggered on full resets (see _reset_impl); ep_len always
        # starts at 0, so settle/stepping-gate grace stay honest either way.
        metrics = {**metrics, 'age_s': (state.ep_len + 1.0) * CONTROL_DT,
                  'stepping_activity_so_far': stepping_activity_so_far, 'ramp_frac': ramp_frac}
        reward, terms = self._reward(metrics, reward_task[:, TASK_REGIME])
        timeout = step_i >= self.max_episode_steps
        # physics_diverged forces a real termination even though the
        # rollback above already made this step's own metrics/reward clean --
        # without this, a once-diverged env would just keep silently
        # rolling back to the same pre-divergence state forever, never
        # resetting, never producing a usable episode again.
        terminal_fail = jnp.logical_or(metrics['terminal'] > 0.5, physics_diverged)
        done = jnp.logical_or(terminal_fail, timeout)
        # A fall must make the whole short episode unprofitable, especially
        # near reset, or a policy can learn to launch forward, get paid for
        # short-lived velocity tracking, then die. Timeouts are not
        # punished; they mean the agent survived the full episode.
        # Episode AGE is ep_len, not the deliberately staggered timeout
        # clock step_i. Using step_i made otherwise-identical early falls pay
        # different penalties depending on the random timeout offset.
        episode_age_steps = state.ep_len + 1.0
        early_fall_penalty = EARLY_FALL_BASE_PENALTY + EARLY_FALL_TIME_PENALTY * jnp.maximum(0.0, self.max_episode_steps - episode_age_steps)
        reward = jnp.where(terminal_fail, reward - early_fall_penalty, reward)

        ep_return = state.ep_return + reward
        ep_len = state.ep_len + 1.0
        ep_vx_sum = state.ep_vx_sum + metrics['vx']
        ep_cmd_vx_sum = state.ep_cmd_vx_sum + metrics['cmd_vx']
        ep_bad_sum = state.ep_bad_sum + metrics['bad_contact']
        ep_support_sum = state.ep_support_sum + metrics['support']
        ep_step_evidence_sum = state.ep_step_evidence_sum + metrics['step_evidence']

        base = self._base_obs(data, task, action, phase, metrics, ctrl)
        hist = jnp.concatenate([state.hist[:, 1:, :], base[:, None, :]], axis=1)
        bootstrap_obs = hist.reshape((self.num_envs, OBS_DIM))
        # carryover_qvel is this step's own (already divergence-safe)
        # qvel -- see the docstring on _reset_with_keys.
        reset_state = self._reset_with_keys(reset_keys, params, carryover_qvel=data.qvel)

        def choose(reset_val, step_val):
            if hasattr(step_val, 'shape') and len(step_val.shape) > 0 and step_val.shape[0] == self.num_envs:
                mask = done.reshape((self.num_envs,) + (1,) * (len(step_val.shape) - 1))
                return jnp.where(mask, reset_val, step_val)
            return step_val

        # Shift the ring buffer -- drop the oldest slot, append this
        # step's raw_target at the end (index -1 = most recent, matching
        # what the read side above expects). Must stay shape-matched with
        # reset_state's own prev_raw_target (see reset()) since tree_map's
        # per-env `choose` below relies on both sides of the tree having
        # the same shape at this position.
        raw_target_history = jnp.concatenate([state.prev_raw_target[:, 1:, :], raw_target[:, None, :]], axis=1)
        stepped = EnvState(data, hist, next_keys, task, action, ctrl, data.xpos[:, self.foot_bodies, :], raw_target_history, phase, step_i,
                           ep_return, ep_len, ep_vx_sum, ep_cmd_vx_sum, ep_bad_sum, ep_support_sum, ep_step_evidence_sum,
                           cmd_target_next, cmd_ramp_from_next, cmd_ramp_frac_next, cmd_ramp_speed_next, cmd_hold_left_next,
                           push_force, push_steps_left)
        new_state = self.jax.tree_util.tree_map(choose, reset_state, stepped)
        obs = self.obs(new_state)
        denom = jnp.maximum(ep_len, 1.0)
        # Command-TRANSITION tracking metrics. Everything logged before
        # this (precision_*, overspeed/underspeed, reward_command_tracking) is
        # averaged over whole episodes, so it can't tell "holds a steady
        # command well" from "follows a CHANGING command well" -- and the
        # second is the actual controllability goal. `ramping` marks the steps
        # where the tracked command is mid-ramp toward a freshly resampled
        # target (ramp_frac < 1, computed above); errors are split by that
        # mask so summarize_rollout can report tracking error DURING
        # transitions vs. during steady holds separately.
        ramping = (ramp_frac < 1.0).astype(jnp.float32)
        steady = 1.0 - ramping
        vx_err = jnp.abs(metrics['vx'] - metrics['cmd_vx'])
        vy_err = jnp.abs(metrics['vy'] - metrics['cmd_vy'])
        yaw_err = jnp.abs(metrics['yaw_rate'] - metrics['cmd_yaw'])
        out_metrics = {
            **metrics,
            'transition_frac': ramping,
            # v5011 maneuver-event instrumentation.  These describe the target
            # and commanded ramp rate associated with the command the actor is
            # currently observing.  They are training diagnostics only; they do
            # not enter the actor observation or reward.  The capability field
            # uses them to issue ONE verdict per complete source->target
            # maneuver rather than frame-by-frame pseudo trials.
            'target_cmd_vx': state.cmd_target[:, 0],
            'target_cmd_vy': state.cmd_target[:, 1],
            'target_cmd_yaw': state.cmd_target[:, 2],
            'transition_ramp_speed': state.cmd_ramp_speed,
            'transition_vx_err': vx_err * ramping, 'transition_vy_err': vy_err * ramping,
            'transition_yaw_err': yaw_err * ramping,
            'steady_vx_err': vx_err * steady, 'steady_vy_err': vy_err * steady,
            'steady_yaw_err': yaw_err * steady,
            **{f'reward_{k}': v for k, v in terms.items()},
            'episode_done': done.astype(jnp.float32),
            'episode_length': ep_len,
            'episode_return': ep_return,
            'episode_fell': terminal_fail.astype(jnp.float32),
            'episode_timeout': timeout.astype(jnp.float32),
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
        qpos = data.qpos
        qvel = data.qvel
        ctrl = data.ctrl

        regime = task[:, TASK_REGIME]
        is_resilience = regime > 1.5
        is_rehearsal = task[:, TASK_CORE_REHEARSAL] > 0.5
        # Birth difficulty is episode-local in v5011.  The governor owns
        # the ceiling; field-steered resilience births may intentionally train
        # below it at the locally useful severity for this command voxel.
        resilience_d = task[:, TASK_BIRTH_RESILIENCE_D]

        # Initial position offset (xy + a bit of extra height wobble
        # on top of posture's own TASK_HEIGHT). RESILIENCE gets the full
        # difficulty-scaled magnitude; STEADY/TRANSITION borrow a small
        # fixed fraction of it (see STEADY_RESILIENCE_BORROW_FRAC).
        pos_noise_full = self._lerp(0.0, self.recovery_position_max_m, resilience_d)
        pos_noise_mag = jnp.where(is_rehearsal, 0.0, jnp.where(is_resilience, pos_noise_full, pos_noise_full * STEADY_RESILIENCE_BORROW_FRAC))
        pos_offset = jax.vmap(lambda k, m: jax.random.uniform(k, (3,), minval=-m, maxval=m))(pos_keys, pos_noise_mag)
        qpos = qpos.at[:, 0:3].set(jnp.stack([
            pos_offset[:, 0], pos_offset[:, 1],
            NEUTRAL_ROOT_Z + params[P_ROOT_Z_OFFSET] + task[:, TASK_HEIGHT] + pos_offset[:, 2],
        ], axis=1))

        # Initial pose tilt (roll/pitch away from upright). Same
        # borrow-fraction coupling as position noise above. Composed via a
        # real quaternion product (_quat_mul), not a small-angle
        # approximation -- the ceiling (~20deg) is real enough that a linear
        # shortcut would be visibly wrong at the difficult end.
        tilt_full = self._lerp(0.0, self.recovery_tilt_max_rad, resilience_d)
        tilt_mag = jnp.where(is_rehearsal, 0.0, jnp.where(is_resilience, tilt_full, tilt_full * STEADY_RESILIENCE_BORROW_FRAC))
        probe_tilt = params[P_PROBE_TILT_RAD]
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
        joint_mag = jnp.where(is_rehearsal, 0.0, jnp.where(is_resilience, joint_full, joint_full * STEADY_RESILIENCE_BORROW_FRAC))
        joint_extra = jax.vmap(lambda k, m: jax.random.uniform(k, (ACTION_DIM,), minval=-m, maxval=m))(joint_keys, joint_mag)
        qnoise = qnoise + joint_extra
        probe_joint = params[P_PROBE_JOINT_RAD]
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
        qvel = qvel.at[:, 0].set(qvel[:, 0] + RESET_FORWARD_VELOCITY_BLEND * task[:, TASK_VX])
        qvel = qvel.at[:, 1].set(qvel[:, 1] + RESET_FORWARD_VELOCITY_BLEND * task[:, TASK_VY])
        qvel = qvel.at[:, 5].set(qvel[:, 5] + RESET_FORWARD_VELOCITY_BLEND * task[:, TASK_YAW])
        probe_lin = params[P_PROBE_LINVEL_MPS]
        lin_full = self.recovery_linvel_max_mps * resilience_d
        lin_mag = jnp.where(is_rehearsal, 0.0, jnp.where(is_resilience, lin_full, lin_full * STEADY_RESILIENCE_BORROW_FRAC))
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
        angvel_mag = jnp.where(is_rehearsal, 0.0, jnp.where(is_resilience, angvel_full, angvel_full * STEADY_RESILIENCE_BORROW_FRAC))
        probe_ang = params[P_PROBE_ANGVEL_RAD_S]
        train_angvel = jax.vmap(lambda k, m: jax.random.uniform(k, (3,), minval=-m, maxval=m))(angvel_keys, angvel_mag)
        yaw_sign = jnp.where((probe_bin % 2) == 0, 0.5, -0.5).astype(jnp.float32)
        probe_ang_dir = jnp.stack([jnp.cos(probe_angle), jnp.sin(probe_angle), yaw_sign], axis=1)
        probe_ang_dir = probe_ang_dir / jnp.maximum(jnp.linalg.norm(probe_ang_dir, axis=1, keepdims=True), 1e-6)
        probe_ang_vec = probe_ang_dir * jnp.maximum(probe_ang, 0.0)
        angvel_kick = jnp.where(probe_ang >= 0.0, probe_ang_vec, train_angvel)
        qvel = qvel.at[:, 3:6].set(qvel[:, 3:6] + angvel_kick)
        if carryover_qvel is not None:
            min_hold_s, _ = self._transition_hold_and_jump(params[P_TRANSITION_DIFFICULTY])
            trans_frac = (TRANSITION_HOLD_MAX_S - min_hold_s) / (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
            carry_prob = trans_frac * CARRYOVER_MAX_PROB
            do_carry = jax.vmap(lambda k: jax.random.uniform(k, ()))(carry_keys) < carry_prob
            # The core-rehearsal cohort is deliberately a pristine anchor: no
            # transition carry-over, no resilience perturbation, no harder
            # command envelope. Its whole job is to keep rehearsing the seed
            # gait while the frontier distributions move underneath it.
            do_carry = jnp.logical_and(do_carry, jnp.logical_not(is_rehearsal))
            qvel = jnp.where(do_carry[:, None], carryover_qvel, qvel)
        ctrl = ctrl.at[:, :ACTION_DIM].set(self.home_target)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = self.forward_batched(data)
        phase = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.0, maxval=2.0 * jnp.pi))(phase_keys)
        prev_action = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        prev_ctrl = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        prev_foot_pos = data.xpos[:, self.foot_bodies, :]
        prev_raw_target = jnp.zeros((self.num_envs, MAX_COMM_LATENCY_STEPS, ACTION_DIM), dtype=jnp.float32)
        base = self._base_obs(data, task, prev_action, phase)
        hist = jnp.repeat(base[:, None, :], HISTORY_LEN, axis=1)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left = self._init_transition_state(task, params, hold_keys)
        push_force0 = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        return EnvState(data, hist, next_keys, task, prev_action, prev_ctrl, prev_foot_pos, prev_raw_target, phase, izeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                         cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left, push_force0, izeros)

    def _rotation_body_to_world(self, data):
        return data.xmat[:, self.torso_id, :].reshape((self.num_envs, 3, 3))

    def _body_packets(self, data):
        jnp = self.jnp
        R = self._rotation_body_to_world(data)
        RT = jnp.swapaxes(R, 1, 2)
        projected_gravity = jnp.einsum('nij,j->ni', RT, jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32))
        body_vel = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 0:3])
        body_ang = jnp.einsum('nij,nj->ni', RT, data.qvel[:, 3:6])
        return projected_gravity, body_vel, body_ang

    def _metrics(self, data, task, action, prev_action, phase=None, prev_foot_pos=None):
        jnp = self.jnp
        projected_gravity, body_vel, body_ang = self._body_packets(data)
        gravity_z = -projected_gravity[:, 2]
        height = data.xpos[:, self.torso_id, 2]
        foot_pos = data.xpos[:, self.foot_bodies, :]
        foot_z = foot_pos[:, :, 2]
        foot_mid = jnp.mean(foot_pos, axis=1)
        torso_pos = data.xpos[:, self.torso_id, :]
        torso_to_feet = torso_pos - foot_mid
        horiz_to_feet = jnp.linalg.norm(torso_to_feet[:, :2], axis=1)
        vertical_to_feet = jnp.maximum(0.025, torso_to_feet[:, 2])
        foot_mid_angle = jnp.arctan2(horiz_to_feet, vertical_to_feet)
        foot_mid_cone = jnp.clip((0.4363323 - foot_mid_angle) / 0.4363323, 0.0, 1.0)
        foot_contacts = jnp.sum((foot_z < 0.036).astype(jnp.float32), axis=1)
        support = jnp.clip(foot_contacts / 2.0, 0.0, 1.0)
        bad_z = data.xpos[:, self.bad_bodies, 2]
        # This model has no knee/shin sentinel contact geoms, so a blunt
        # body-origin proxy alone is too strict a termination condition --
        # only actual height/tilt/rate collapse may terminate. Body-low
        # stays visible as a pressure signal (below) so strict validation
        # could be reintroduced later without poisoning early training.
        body_low_soft = jnp.any(bad_z[:, 1:] < SOFT_BODY_LOW_Z, axis=1).astype(jnp.float32)
        body_floor_hard = jnp.logical_and(
            jnp.any(bad_z[:, 1:] < HARD_BODY_FLOOR_Z, axis=1),
            height < HARD_BODY_TORSO_GATE_Z,
        ).astype(jnp.float32)
        low_height_soft = (height < 0.130).astype(jnp.float32)
        bad_contact = jnp.maximum(BODY_LOW_REWARD_WEIGHT * body_low_soft, low_height_soft)
        swing_clearance = jnp.maximum(0.0, jnp.max(foot_z, axis=1) - 0.018)
        q = data.qpos[:, self.act_qpos]
        joint_pressure = jnp.mean(jnp.maximum(0.0, jnp.abs(q - self.home_target) / (self.target_span * 1.20) - 0.72), axis=1)
        scissor = jnp.maximum(0.0, jnp.abs(q[:, 0] - q[:, 3]) - 0.62)
        action_delta = jnp.linalg.norm(action - prev_action, axis=1)
        # Reward-only true stance-foot slip proxy. v5004 used chassis
        # horizontal velocity here, accidentally making fast translation itself
        # mildly undesirable. MuJoCo truth is allowed in the reward: consecutive
        # foot poses estimate tangential speed and only near-floor feet count.
        if prev_foot_pos is None:
            stance_slip = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        else:
            foot_vel_xy = (data.xpos[:, self.foot_bodies, :2] - prev_foot_pos[:, :, :2]) / CONTROL_DT
            foot_speed = jnp.linalg.norm(foot_vel_xy, axis=2)
            stance_w = jnp.clip((0.040 - foot_z) / 0.020, 0.0, 1.0)
            stance_slip = jnp.sum(foot_speed * stance_w, axis=1) / jnp.maximum(1.0, jnp.sum(stance_w, axis=1))
        fall_height = height < 0.145
        fall_tilt = gravity_z < 0.35
        # Seed-stage fast model: proxy floor contact is diagnostic only.
        # Real non-foot contact enforcement belongs in the later validation model.
        fall_bad_contact = jnp.zeros_like(body_floor_hard, dtype=bool)
        fall_roll_rate = jnp.abs(body_ang[:, 0]) > 9.0
        fall_pitch_rate = jnp.abs(body_ang[:, 1]) > 9.0
        terminal = jnp.logical_or(fall_height, fall_tilt)
        # Do not terminate on the body-low proxy in fast seed training.
        terminal = jnp.logical_or(terminal, fall_roll_rate)
        terminal = jnp.logical_or(terminal, fall_pitch_rate)
        overspeed = jnp.maximum(0.0, body_vel[:, 0] - task[:, TASK_VX])
        underspeed = jnp.maximum(0.0, task[:, TASK_VX] - body_vel[:, 0])
        if phase is None:
            phase = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        s_phase = jnp.sin(phase)
        r_expected = jnp.maximum(0.0, s_phase)
        l_expected = jnp.maximum(0.0, -s_phase)
        r_lift = jnp.clip((foot_z[:, 0] - 0.018) / jnp.maximum(0.008, task[:, TASK_STEP_HEIGHT]), 0.0, 1.0)
        l_lift = jnp.clip((foot_z[:, 1] - 0.018) / jnp.maximum(0.008, task[:, TASK_STEP_HEIGHT]), 0.0, 1.0)
        phase_sync = jnp.clip(1.0 - 0.5 * (jnp.abs(r_lift - r_expected) + jnp.abs(l_lift - l_expected)), 0.0, 1.0)
        step_evidence = jnp.clip((jnp.maximum(r_lift, l_lift) - 0.10) / 0.55, 0.0, 1.0)
        alternating_lift = jnp.abs(r_lift - l_lift)
        organic_step_shape = jnp.clip(0.50 * jnp.maximum(r_lift, l_lift) + 0.30 * phase_sync * step_evidence + 0.20 * alternating_lift, 0.0, 1.0)
        return {
            'terminal': terminal.astype(jnp.float32),
            'fall_height': fall_height.astype(jnp.float32),
            'fall_tilt': fall_tilt.astype(jnp.float32),
            'fall_bad_contact': fall_bad_contact.astype(jnp.float32),
            'body_low_soft': body_low_soft,
            'body_floor_hard': body_floor_hard,
            'fall_roll_rate': fall_roll_rate.astype(jnp.float32),
            'fall_pitch_rate': fall_pitch_rate.astype(jnp.float32),
            'overspeed': overspeed, 'underspeed': underspeed,
            'regime_steady': (task[:, TASK_REGIME] < 0.5).astype(jnp.float32),
            'regime_transition': jnp.logical_and(task[:, TASK_REGIME] > 0.5, task[:, TASK_REGIME] < 1.5).astype(jnp.float32),
            'regime_resilience': (task[:, TASK_REGIME] > 1.5).astype(jnp.float32),
            'core_rehearsal': task[:, TASK_CORE_REHEARSAL],
            'athletic_rehearsal': task[:, TASK_ATHLETIC_REHEARSAL],
            'regime_steady_frontier': jnp.logical_and(jnp.logical_and(task[:, TASK_REGIME] < 0.5, task[:, TASK_CORE_REHEARSAL] < 0.5), task[:, TASK_ATHLETIC_REHEARSAL] < 0.5).astype(jnp.float32),
            'regime_transition_frontier': jnp.logical_and(jnp.logical_and(jnp.logical_and(task[:, TASK_REGIME] > 0.5, task[:, TASK_REGIME] < 1.5), task[:, TASK_CORE_REHEARSAL] < 0.5), task[:, TASK_ATHLETIC_REHEARSAL] < 0.5).astype(jnp.float32),
            'regime_resilience_frontier': jnp.logical_and(jnp.logical_and(task[:, TASK_REGIME] > 1.5, task[:, TASK_CORE_REHEARSAL] < 0.5), task[:, TASK_ATHLETIC_REHEARSAL] < 0.5).astype(jnp.float32),
            'birth_steady_d': task[:, TASK_BIRTH_STEADY_D],
            'birth_transition_d': task[:, TASK_BIRTH_TRANSITION_D],
            'birth_resilience_d': task[:, TASK_BIRTH_RESILIENCE_D],
            'birth_posture_d': task[:, TASK_BIRTH_POSTURE_D],
            'vx': body_vel[:, 0], 'vy': body_vel[:, 1], 'yaw_rate': body_ang[:, 2],
            'roll_rate': body_ang[:, 0], 'pitch_rate': body_ang[:, 1],
            'cmd_vx': task[:, TASK_VX], 'cmd_vy': task[:, TASK_VY], 'cmd_yaw': task[:, TASK_YAW],
            'height': height, 'target_height': NEUTRAL_ROOT_Z + task[:, TASK_HEIGHT],
            'gravity_z': gravity_z,
            'foot_mid_angle': foot_mid_angle,
            'foot_mid_angle_deg': foot_mid_angle * 57.29578,
            'foot_mid_cone': foot_mid_cone,
            'phase_sync': phase_sync,
            'step_evidence': step_evidence,
            'organic_step_shape': organic_step_shape,
            # Positive means torso/root is ahead of the midpoint between the feet;
            # negative means it is behind.  This is not full COM, but it is cheap,
            # batched, and exactly the signal we need when HOME zero-action drifts
            # forward/backward before PPO has a vote.
            'torso_foot_x': data.xpos[:, self.torso_id, 0] - foot_mid[:, 0],
            'torso_foot_y': data.xpos[:, self.torso_id, 1] - foot_mid[:, 1],
            'foot_mean_z': jnp.mean(foot_z, axis=1),
            'foot_contacts': foot_contacts, 'bad_contact': bad_contact, 'support': support,
            'stance_slip': stance_slip,
            'swing_clearance': swing_clearance, 'target_step_height': jnp.maximum(0.008, task[:, TASK_STEP_HEIGHT]),
            'action_delta_norm': action_delta,
            'joint_limit_pressure': joint_pressure,
            'scissor_pressure': scissor,
            'regime_id': task[:, TASK_REGIME],
        }

    def _reward(self, m, regime):
        jnp = self.jnp
        # Every regime spawns already moving toward a real commanded state
        # (see the module doctrine and _reset_with_keys), so there's no
        # standing-still special case here -- what's regime-conditioned
        # below is real, targeted shaping, not a special-cased behavior:
        is_transition = jnp.logical_and(regime > 0.5, regime < 1.5).astype(jnp.float32)
        is_resilience = (regime > 1.5).astype(jnp.float32)
        valid_alive = 1.0 - m['terminal']
        vx_err = m['vx'] - m['cmd_vx']; vy_err = m['vy'] - m['cmd_vy']; yaw_err = m['yaw_rate'] - m['cmd_yaw']
        # Forward overspeed must not be rewarded almost as strongly as clean
        # tracking, or a policy can discover a forward launch into terminal
        # contact -- imperfect motion is fine, a ballistic "close enough,
        # then faceplant" hack is not. RESILIENCE gets a wider dead zone
        # here specifically -- see RESILIENCE_TOLERANCE_MULT's own comment
        # for why.
        speed_tol = jnp.where(is_resilience > 0.5, self.speed_tolerance * RESILIENCE_TOLERANCE_MULT, self.speed_tolerance)
        overspeed = jnp.maximum(0.0, m['vx'] - m['cmd_vx'] - speed_tol)
        underspeed = jnp.maximum(0.0, m['cmd_vx'] - m['vx'] - speed_tol)
        # TRANSITION weights the WHOLE error sum harder specifically while a
        # command is still ramping in; RESILIENCE weights it softer
        # throughout. Both multiply the same underlying formula rather than
        # branching into regime-specific shapes, so the three regimes stay
        # one recognizable, comparably-tunable reward, not three diverging
        # ones.
        ramping = (m['ramp_frac'] < 1.0).astype(jnp.float32)
        tracking_gain = 1.0 + is_transition * ramping * (TRANSITION_TRACKING_BOOST - 1.0)
        tracking_gain = tracking_gain * jnp.where(is_resilience > 0.5, RESILIENCE_TRACKING_RELIEF, 1.0)
        command_tracking = jnp.exp(-tracking_gain * (62.0 * vx_err * vx_err + 14.0 * vy_err * vy_err + 4.5 * yaw_err * yaw_err
                                                      + 120.0 * overspeed * overspeed + 40.0 * underspeed * underspeed))
        # Episode-start settle grace -- every regime spawns close to a real
        # commanded state, so a few frames of residual spawn noise
        # (position/tilt/velocity) resolving shouldn't score as bad
        # tracking before the robot has had a chance to actually settle.
        # Neutral (1.0), not penalized, before settle -- NOT excluded from
        # termination, a fall during settle is still real.
        settled = (m['age_s'] > SETTLE_STEPS * CONTROL_DT).astype(jnp.float32)
        command_tracking = settled * command_tracking + (1.0 - settled) * 1.0
        upright = jnp.clip((m['gravity_z'] - 0.35) / 0.65, 0.0, 1.0)
        height = jnp.exp(-80.0 * (m['height'] - m['target_height']) ** 2)
        # Feet-midpoint cone: the chassis may be dynamically falling forward,
        # but if the feet midpoint to chassis vector exceeds about 25 degrees
        # the policy is no longer catching the fall.  This is shaping, not a
        # static-stand gate.
        upright_height = 0.52 * upright + 0.28 * height + 0.20 * m['foot_mid_cone']
        feet_only_contact = (1.0 - m['bad_contact']) * (0.35 + 0.65 * m['support'])
        stance_anti_slip = jnp.exp(-6.0 * m['stance_slip'])
        raw_swing = jnp.clip(m['swing_clearance'] / jnp.maximum(0.006, m['target_step_height']), 0.0, 1.0)
        # Step shape should emerge organically.  Once either foot starts lifting,
        # give a small bonus for matching the left/right phase signal and not
        # dragging both feet.  No step evidence means no harsh penalty, just no
        # extra step-shape payment.
        swing_clearance = jnp.clip(0.55 * raw_swing + 0.45 * m['organic_step_shape'], 0.0, 1.0)
        servo_smoothness = jnp.exp(-0.22 * m['action_delta_norm'])
        joint_safety = jnp.exp(-(5.0 * m['joint_limit_pressure'] + 2.0 * m['scissor_pressure']))
        # Every regime is a real "must actually walk" episode from the
        # first settled step onward.
        locomotion_maturity = jnp.clip((m['age_s'] - 0.20) / 0.45, 0.0, 1.0)
        # Anti-freezing gate. A policy can learn to balance on one leg and
        # avoid stepping entirely, rather than walk -- confirmed directly
        # by watching a real run, and the reward arithmetic explains
        # exactly why that's attractive. At small commanded speeds, a
        # robot standing perfectly frozen already scores valid_alive=1,
        # upright_height~1, feet_only_contact~1, stance_anti_slip~1,
        # servo_smoothness~1 (it directly rewards LOW action delta, i.e.
        # stillness), and even command_tracking stays high since the
        # tracking error from a near-zero commanded speed is small. The
        # only one of 8 terms that actually requires motion is
        # swing_clearance (weight 0.16 out of ~4.94 total) -- worked out to
        # an estimated reward of ~3.56 for doing nothing, higher than the
        # +3.2 a genuinely well-walking robot earns. Not a hypothesis: this
        # is the real formula and real arithmetic.
        #
        # Fix: gate upright_height and feet_only_contact's contribution to
        # the final sum -- the two biggest "free when frozen" terms -- on
        # `stepping_activity_so_far` (an episode-long running average of
        # step_evidence, NOT the instantaneous per-step value, which would
        # incorrectly punish the normal stance phase of a real gait). No
        # gating for the first second of an episode (not enough data to
        # judge a pattern yet, same spirit as survival_maturity's own
        # ramp-in above); after that, a 30% floor plus up to 70% bonus for
        # demonstrated stepping, not a hard cliff -- a policy that's
        # stepping only a little still gets substantial credit, only
        # *total, sustained* non-stepping is strongly discouraged.
        stepping_factor = jnp.clip(m['stepping_activity_so_far'] / self.stepping_activity_target, 0.0, 1.0)
        # Every regime commands real, non-zero movement, so there's no
        # "standing still is correct" case to exempt from this gate.
        stepping_gate = jnp.where(m['age_s'] > STEPPING_GATE_GRACE_S, 0.30 + 0.70 * stepping_factor, 1.0)
        terms = {
            'valid_alive': valid_alive,
            'command_tracking': command_tracking,
            'upright_height': upright_height,
            'feet_only_contact': feet_only_contact,
            'stance_anti_slip': stance_anti_slip,
            'swing_clearance': swing_clearance,
            'servo_smoothness': servo_smoothness,
            'joint_safety': joint_safety,
        }
        # Reward remains the same 8-term doctrine, but survival/contact now gate
        # most locomotion payment.  This keeps the reward pure while making
        # short terminal episodes mathematically unattractive.
        locomotion_quality = locomotion_maturity * command_tracking * upright_height * (0.35 + 0.65 * feet_only_contact) * (0.30 + 0.70 * m['foot_mid_cone'])
        reward = (0.58*valid_alive + 1.55*locomotion_quality + 0.68*upright_height*stepping_gate + 0.50*feet_only_contact*stepping_gate +
                  0.18*stance_anti_slip + 0.16*swing_clearance + 0.12*servo_smoothness + 0.17*joint_safety)
        return reward, terms

    def _base_obs(self, data, task, prev_action, phase, metrics=None, prev_ctrl=None):
        """Deployable actor observation: sensors/estimators, not teacher truth."""
        jnp = self.jnp
        projected_gravity, body_vel, body_ang = self._body_packets(data)
        q = data.qpos[:, self.act_qpos]
        qv = data.qvel[:, self.act_qvel]
        shin_flex = data.qpos[:, self.shin_qpos]
        shin_rate = data.qvel[:, self.shin_qvel]
        n0 = jnp.sin(1.73 * phase + 0.6180339 * self.env_index)
        n1 = jnp.sin(2.31 * phase + 1.4142136 * self.env_index + 0.7)
        ns = jnp.where(task[:, TASK_SENSOR_SCALE] >= 0.0, task[:, TASK_SENSOR_SCALE], self.sensor_noise_scale)
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
        tof_height = torso[:, 2:3] + ns[:, None] * 0.0025 * n1[:, None]
        vel_conf = jnp.clip(1.0 - 0.18*ns, 0.2, 1.0)[:, None].astype(jnp.float32)
        shin_flex_est = shin_flex + ns[:, None] * 0.0010 * jnp.stack([n0, n1], axis=1)
        shin_rate_est = shin_rate + ns[:, None] * 0.010 * jnp.stack([n1, n0], axis=1)
        shin_load_est = jnp.clip(jnp.abs(shin_flex_est) / 0.10, 0.0, 1.5)
        cmd = task[:, [TASK_VX, TASK_VY, TASK_YAW, TASK_HEIGHT, TASK_PITCH, TASK_ROLL, TASK_STEP_HEIGHT]]
        phase_pack = self.phase_obs_scale * jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=1)
        if prev_ctrl is None:
            prev_ctrl = prev_action
        base = jnp.concatenate([grav_est, gyro_est, q_est, qv_est, prev_action, prev_ctrl,
                                foot_rel, vel_est, tof_height, vel_conf, cmd, phase_pack,
                                shin_flex_est, shin_rate_est, shin_load_est], axis=1)
        assert base.shape[1] == BASE_OBS_DIM, f"_base_obs produced {base.shape[1]} columns, BASE_OBS_DIM says {BASE_OBS_DIM}"
        return jnp.nan_to_num(base, nan=0.0, posinf=10.0, neginf=-10.0).astype(jnp.float32)
