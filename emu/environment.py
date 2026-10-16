"""The simulated universe: robot physics, task packets and sensors.

`EmuMJXEnv` advances many simultaneous MuJoCo worlds and supplies physical
observations to PPO. The robot's joints, springs and motor limits are defined
in robot.xml; domain_randomization.py supplies per-world model variations.

KNOWN DEBT: this module currently also samples curriculum tasks and calls
the pure training reward in rewards.py. Those are *integration boundaries*,
not reasons to hide the physical state or create a second trainer. Future
simplification will make the physics/reset/sensor path readable first.
"""
from __future__ import annotations

from typing import Any, NamedTuple
from types import SimpleNamespace
import numpy as np

from .domain_randomization import DomainRandomizationConfig, build_batched_model
from .rewards import training_reward, apply_early_fall_penalty
from .contracts import (SIM_DT, CONTROL_SUBSTEPS, CONTROL_DT, ACTION_DIM, ACTUATED_JOINTS, BASE_OBS_DIM, BASE_OBS_FIELDS, HISTORY_LEN, OBS_DIM, COMBO_P, PHASE_RATE_MAX_HZ, PITCH_CMD_MAX_RAD, FIELD_EDGE_SCALES, SEED_VX_LO, SEED_VX_HI, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS,
                       FIELD_NODE_STRIDE, FIELD_EDGE_STRIDE, FIELD_NODE_START, FIELD_EDGE_START, TASK_PARAM_DIM,
                       TASK_PARAM_FIELDS, P)

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
# Anti-freezing gate: reward should not favour a motionless balance solution.
# The threshold only requires evidence that feet are actually lifting; it does
# not prescribe a cadence. A short grace period avoids judging the first few
# frames of an episode.
# These quantities exist only long enough to compute reward. Keeping them out of
# StepOut.metrics avoids carrying/transferring eight per-env arrays that no
# curriculum, evaluator, capability map or logger consumes.
REWARD_ONLY_METRICS = frozenset({
    'target_height', 'stance_slip', 'target_step_height', 'action_delta_norm',
    'joint_limit_pressure', 'scissor_pressure', 'stepping_activity_so_far', 'ramp_frac',
})
# Command-transition dynamics (see _sample_cmd_target / TRANSITION's own
# difficulty path in _transition_hold_and_jump).
TRANSITION_HOLD_MIN_STEPS = 45.0   # shortest command hold, reached when min_hold_s bottoms out
TRANSITION_HOLD_MAX_STEPS = 700.0  # longest hold, the starting value (~never resamples mid-episode)
TRANSITION_RAMP_MIN_STEPS = 4.0    # near-instant target change, the hardest end
TRANSITION_RAMP_MAX_STEPS = 120.0  # gentle ramp, the starting/easy value
CARRYOVER_MAX_PROB = 0.35          # at TRANSITION's hardest point, up to this fraction of resets carry over qvel
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
# Short reward-settle window after reset. Termination remains active; only
# tracking score avoids over-interpreting a few frames of spawn transient.
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


class EnvState(NamedTuple):
    data: Any
    hist: Any
    rng: Any
    task: Any
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


def command_phase_rate(vx, vy, vx_hi_max, phase_speed_gain, base_phase_rate):
    """Sampled gait cadence: base 0.85–1.20 Hz plus birth-speed gain, max 2.6 Hz.

    Keep pitch-command limits out of this calculation. This function also gives
    CPU-only contract tests a way to catch invalid JAX argument signatures
    before a multi-hour MJX GPU compile.
    """
    import jax.numpy as jnp
    speed_fraction = jnp.clip(jnp.sqrt(vx * vx + vy * vy) / jnp.maximum(0.05, vx_hi_max), 0.0, 1.5)
    return jnp.minimum(PHASE_RATE_MAX_HZ, base_phase_rate + phase_speed_gain * speed_fraction)


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


class EmuMJXEnv:
    """A JAX/MJX vector environment with all envs stepped in one device batch."""

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
                 attitude_profile: str = 'v5014', pitch_command_max_deg: float = 8.0214):
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
        self.attitude_profile = attitude_profile
        self.pitch_limit_rad = (PITCH_CMD_MAX_RAD if attitude_profile == 'v5014'
                                else float(np.deg2rad(pitch_command_max_deg)))
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.robot_xml = str(robot_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_xml)
        self.mj_model.opt.timestep = SIM_DT
        # Performance hint from MJX docs: keep contacts explicit/simple. The XML
        # owns the complete robot physics, including actuated-joint armature and
        # Coulomb friction; thin thigh/shin rods are deliberately non-contacting.
        self.mjx_model_template = mjx.put_model(self.mj_model)
        self.ids = self._resolve_ids()
        self.domain_config = (domain_config or DomainRandomizationConfig()).normalized()
        self.mjx_model, self.model_in_axes, self.domain_info = build_batched_model(
            self.mjx_model_template, ids=self.ids, num_envs=self.num_envs, seed=int(seed),
            config=self.domain_config)
        gravity_unit = self.domain_info.get('gravity_unit_world')
        self.gravity_unit_world = (
            gravity_unit if gravity_unit is not None
            else jnp.tile(jnp.asarray([[0.0, 0.0, -1.0]], dtype=jnp.float32), (self.num_envs, 1))
        )
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
        command_frac = jnp.clip(params[P.regime_command_frac], 0.0, 1.0)

        # STEADY's full current envelope -- every regime's command draw is
        # some fraction of THIS, never a separately-scheduled range of its
        # own (see module doctrine: isolate which axis is under test from
        # the aggregate difficulty of everything else).
        if self.open_frontier:
            # Physical frontier values are live curriculum state, not
            # percentages of a programmer-declared final ceiling.
            vx_hi = jnp.maximum(SEED_VX_HI, params[P.frontier_vx_hi])
            vx_lo = jnp.minimum(SEED_VX_LO, params[P.frontier_vx_lo])
            yaw_max = jnp.maximum(0.0, params[P.frontier_yaw])
            vy_max = jnp.maximum(0.0, params[P.frontier_vy])
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
        steady_d = params[P.steady_difficulty]
        command_frac = jnp.clip(params[P.regime_command_frac], 0.0, 1.0)
        _, jump_frac = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            jump_frac = jnp.asarray(1.0, dtype=jnp.float32)
        if self.open_frontier:
            vx_hi_full = jnp.maximum(SEED_VX_HI, params[P.frontier_vx_hi])
            vx_lo_full = jnp.minimum(SEED_VX_LO, params[P.frontier_vx_lo])
            yaw_full = jnp.maximum(0.0, params[P.frontier_yaw])
            vy_full = jnp.maximum(0.0, params[P.frontier_vy])
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

    def _step_impl(self, state: EnvState, action, params) -> StepOut:
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
        phase = (state.phase + CONTROL_DT * state.task[:, T.phase_rate] * 2.0 * jnp.pi) % (2.0 * jnp.pi)

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
        reward, terms = training_reward(
            metrics, reward_task[:, T.regime],
            attitude_profile=self.attitude_profile, xp=jnp)
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
        reward, fall_payment = apply_early_fall_penalty(
            reward, terminal_fail, episode_age_steps, self.max_episode_steps, xp=jnp)

        ep_return = state.ep_return + reward
        ep_len = state.ep_len + 1.0
        ep_vx_sum = state.ep_vx_sum + metrics['vx']
        ep_cmd_vx_sum = state.ep_cmd_vx_sum + metrics['cmd_vx']
        ep_bad_sum = state.ep_bad_sum + metrics['bad_contact']
        ep_support_sum = state.ep_support_sum + metrics['support']
        ep_step_evidence_sum = state.ep_step_evidence_sum + metrics['step_evidence']

        base = self._base_obs(data, task, action, phase, ctrl)
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

        stepped = EnvState(data, hist, next_keys, task_for_state, action, data.xpos[:, self.foot_bodies, :], raw_target, phase, step_i,
                           ep_return, ep_len, ep_vx_sum, ep_cmd_vx_sum, ep_bad_sum, ep_support_sum, ep_step_evidence_sum,
                           cmd_target_next, cmd_ramp_from_next, cmd_ramp_frac_next, cmd_ramp_speed_next, cmd_hold_left_next,
                           push_force, push_steps_left)
        new_state = self.jax.tree_util.tree_map(choose, reset_state, stepped)
        obs = self.obs(new_state)
        denom = jnp.maximum(ep_len, 1.0)
        # Command-TRANSITION tracking metrics. Whole-rollout summaries
        # cannot distinguish steady holding from following a changing command.
        # `ramping` marks the steps
        # where the tracked command is mid-ramp toward a freshly resampled
        # target (ramp_frac < 1, computed above); errors are split by that
        # mask so summarize_rollout can report tracking error DURING
        # transitions vs. during steady holds separately.
        ramping = transition_active.astype(jnp.float32)
        resampled = cmd_resampled.astype(jnp.float32)
        steady = 1.0 - ramping
        vx_err = jnp.abs(metrics['vx'] - metrics['cmd_vx'])
        vy_err = jnp.abs(metrics['vy'] - metrics['cmd_vy'])
        yaw_err = jnp.abs(metrics['yaw_rate'] - metrics['cmd_yaw'])
        out_metrics = {
            **{k: v for k, v in metrics.items() if k not in REWARD_ONLY_METRICS},
            'transition_frac': ramping,
            'cmd_resampled': resampled,
            # Maneuver-event instrumentation describes the target
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
            'reward_weighted_fall_penalty': fall_payment,
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
        qpos = qpos.at[:, 0:3].set(jnp.stack([
            pos_offset[:, 0], pos_offset[:, 1],
            NEUTRAL_ROOT_Z + task[:, T.height] + pos_offset[:, 2],
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
        ctrl = ctrl.at[:, :ACTION_DIM].set(self.home_target)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = self.forward_batched(self.mjx_model, data)
        phase = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.0, maxval=2.0 * jnp.pi))(phase_keys)
        prev_action = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        prev_foot_pos = data.xpos[:, self.foot_bodies, :]
        prev_raw_target = jnp.zeros((self.num_envs, ACTION_DIM), dtype=jnp.float32)
        base = self._base_obs(data, task, prev_action, phase, prev_action)
        hist = jnp.repeat(base[:, None, :], HISTORY_LEN, axis=1)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left = self._init_transition_state(task, params, hold_keys)
        push_force0 = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        return EnvState(data, hist, next_keys, task, prev_action, prev_foot_pos, prev_raw_target, phase, izeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                         cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left, push_force0, izeros)

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

    def _metrics(self, data, task, action, prev_action, phase, prev_foot_pos):
        jnp = self.jnp
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
        foot_contacts = jnp.sum((foot_z < 0.036).astype(jnp.float32), axis=1)
        support = jnp.clip(foot_contacts / 2.0, 0.0, 1.0)
        bad_z = data.xpos[:, self.bad_bodies, 2]
        # This model has no knee/shin sentinel contact geoms, so a blunt
        # body-origin proxy alone is too strict a termination condition --
        # only actual height/tilt/rate collapse may terminate. Body-low
        # stays visible as a pressure signal (below) so strict validation
        # could be reintroduced later without poisoning early training.
        body_low_soft = jnp.any(bad_z[:, 1:] < SOFT_BODY_LOW_Z, axis=1).astype(jnp.float32)
        low_height_soft = (height < 0.130).astype(jnp.float32)
        bad_contact = jnp.maximum(BODY_LOW_REWARD_WEIGHT * body_low_soft, low_height_soft)
        swing_clearance = jnp.maximum(0.0, jnp.max(foot_z, axis=1) - 0.018)
        q = data.qpos[:, self.act_qpos]
        joint_pressure = jnp.mean(jnp.maximum(0.0, jnp.abs(q - self.home_target) / (self.target_span * 1.20) - 0.72), axis=1)
        scissor = jnp.maximum(0.0, jnp.abs(q[:, 0] - q[:, 3]) - 0.62)
        action_delta = jnp.linalg.norm(action - prev_action, axis=1)
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
        stance_w = jnp.clip((0.040 - foot_z) / 0.020, 0.0, 1.0)
        stance_slip = jnp.sum(foot_speed * stance_w, axis=1) / jnp.maximum(1.0, jnp.sum(stance_w, axis=1))
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
        r_lift = jnp.clip((foot_z[:, 0] - 0.018) / jnp.maximum(0.008, task[:, T.step_height]), 0.0, 1.0)
        l_lift = jnp.clip((foot_z[:, 1] - 0.018) / jnp.maximum(0.008, task[:, T.step_height]), 0.0, 1.0)
        phase_sync = jnp.clip(1.0 - 0.5 * (jnp.abs(r_lift - r_expected) + jnp.abs(l_lift - l_expected)), 0.0, 1.0)
        step_evidence = jnp.clip((jnp.maximum(r_lift, l_lift) - 0.10) / 0.55, 0.0, 1.0)
        alternating_lift = jnp.abs(r_lift - l_lift)
        organic_step_shape = jnp.clip(0.50 * jnp.maximum(r_lift, l_lift) + 0.30 * phase_sync * step_evidence + 0.20 * alternating_lift, 0.0, 1.0)
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

    def _base_obs(self, data, task, prev_action, phase, delivered_ctrl):
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
        tof_height = torso[:, 2:3] + ns[:, None] * 0.0025 * n1[:, None]
        vel_conf = jnp.clip(1.0 - 0.18*ns, 0.2, 1.0)[:, None].astype(jnp.float32)
        shin_flex_est = shin_flex + ns[:, None] * 0.0010 * jnp.stack([n0, n1], axis=1)
        shin_rate_est = shin_rate + ns[:, None] * 0.010 * jnp.stack([n1, n0], axis=1)
        shin_load_est = jnp.clip(jnp.abs(shin_flex_est) / 0.10, 0.0, 1.5)
        cmd = task[:, [T.vx, T.vy, T.yaw, T.height, T.pitch, T.roll, T.step_height]]
        phase_pack = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=1)
        base = jnp.concatenate([grav_est, gyro_est, q_est, qv_est, prev_action, delivered_ctrl,
                                foot_rel, vel_est, tof_height, vel_conf, cmd, phase_pack,
                                shin_flex_est, shin_rate_est, shin_load_est], axis=1)
        assert base.shape[1] == len(BASE_OBS_FIELDS) == BASE_OBS_DIM, f"_base_obs produced {base.shape[1]} columns, contract says {BASE_OBS_DIM}"
        return jnp.nan_to_num(base, nan=0.0, posinf=10.0, neginf=-10.0).astype(jnp.float32)
