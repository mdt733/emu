"""Per-step teaching payments for the active EMU actor.

Eight core weighted terms, the fall penalty and optional motor/terrain
guidance modify PPO learning, not certified locomotion. Teacher-only
MuJoCo facts may inform payments but never enter actor observations.
An independent frozen-policy exam determines physical capability.

Shaft power and spring energy are simulator proxies, not measured
electrical draw or recovered power. Keep default payment arithmetic
stable so before/after experiments remain comparable.
"""

from .contracts import CONTROL_DT

# Scalar names correspond to `reward_weighted_<name>` in logged metrics.
PAYMENT_NAMES = (
    "alive", "locomotion", "posture", "contact", "anti_slip",
    "clearance", "smoothness", "joint_safety",
)
FALL_PAYMENT_NAME = "fall_penalty"
MOTOR_PAYMENT_NAME = "motor_cost"  # opt-in; old eight-payment contract stays fixed

# Physical measurements needed solely to calculate teaching payments. They
# must not be shipped in every PPO diagnostic packet or misread as evidence.
REWARD_ONLY_METRICS = frozenset({
    'target_height', 'stance_slip', 'target_step_height', 'action_delta_norm',
    'joint_limit_pressure', 'scissor_pressure', 'stepping_activity_so_far',
    'ramp_frac',
})

# Stable baseline reward constants: preserve values across matched studies.
EARLY_FALL_BASE_PENALTY = 65.0
EARLY_FALL_TIME_PENALTY = 0.10
STEPPING_ACTIVITY_TARGET = 0.15
SPEED_TOLERANCE = 0.025
STEPPING_GATE_GRACE_S = 1.0
SETTLE_STEPS = 25
TRANSITION_TRACKING_BOOST = 1.5
RESILIENCE_TOLERANCE_MULT = 3.0
RESILIENCE_TRACKING_RELIEF = 0.5

def terrain_progress_payment(metrics, *, xp, weight: float):
    """Bounded opt-in reward-only forward progress across a real obstacle.

    Only pays actual positive world-X displacement while a physical step
    episode is active, supported and upright. No oracle height/distance is
    written into actor observations. This is coaching, NOT course success.
    """
    if weight <= 0.:
        return xp.zeros_like(metrics['terminal'])
    dx = xp.clip(metrics['terrain_forward_delta_m'], 0., .012)
    eligible = ((metrics['terrain_height_birth_m'] > 0.) &
                (metrics['terrain_course_x'] >= -.05) &
                (metrics['terrain_course_x'] <= .42) &
                (metrics['cmd_vx'] >= .07) &
                (xp.abs(metrics['cmd_vy']) <= .08) &
                (xp.abs(metrics['cmd_yaw']) <= .50) &
                (metrics['support'] >= .25) &
                (metrics['gravity_z'] > .80) &
                (metrics['terminal'] < .5))
    forward = weight * xp.clip(dx / .012, 0., 1.) * eligible.astype(dx.dtype)
    # Modest one-shot completion reward for genuine box-foot contact and
    # supported first crossing; the environment supplies this REWARD-ONLY
    # truth, never an actor input. The existing zero-weight branch stays
    # bit-identical and can still beat the specialist on held-out exams.
    crossed = metrics.get('terrain_first_verified_crossing', xp.zeros_like(dx))
    completion = 4.0 * weight * xp.clip(crossed, 0., 1.)
    return forward + completion


def training_reward(m, regime, *, attitude_profile, xp, motor_guidance_weight=0.0, considered_reward=False):
    """Score one frame: commands → body/feet → gait maturity → payments.

    m contains physical truths measured by environment._metrics plus the
    previous episode activity and command ramp. `regime` labels the lesson;
    only a reduced, independent physics summary should inform curriculum.
    The returned terms distinguish diagnostic shaping from *weighted* pay.
    """
    jnp = xp
    is_transition = jnp.logical_and(regime > 0.5, regime < 1.5).astype(jnp.float32)
    is_resilience = (regime > 1.5).astype(jnp.float32)
    valid_alive = 1.0 - m['terminal']

    command_tracking = _command_tracking(m, is_transition, is_resilience, jnp)
    (upright_height, feet_only_contact, stance_anti_slip,
     swing_clearance, servo_smoothness, joint_safety) = _physical_qualities(
        m, attitude_profile, jnp)
    locomotion_maturity, stepping_gate = _gait_maturity(m, jnp)

    # Keep only the five unweighted shaping diagnostics used off-device.
    terms = {
        'valid_alive': valid_alive,
        'command_tracking': command_tracking,
        'upright_height': upright_height,
        'feet_only_contact': feet_only_contact,
        'swing_clearance': swing_clearance,
    }
    # Locomotion depends on real movement, body support and maturity.
    locomotion_quality = locomotion_maturity * command_tracking * upright_height * (0.35 + 0.65 * feet_only_contact) * (0.30 + 0.70 * m['foot_mid_cone'])
    if attitude_profile == 'v5015':
        remaining = jnp.maximum(0.0, -m['height_error'] - 0.012 -
                                0.018*jnp.clip(jnp.abs(m['cmd_pitch'])/0.7853982,0.0,1.0))
        locomotion_quality *= jnp.exp(-1200.0 * remaining * remaining)
    # Simulated actuator output is not verified motor current or electrical
    # power. This is a bounded teaching *hypothesis*, opt-in per study arm.
    # Zero cost weight exactly reproduces pre-v5016 scalar economics.
    motor_kwargs = {}
    if motor_guidance_weight > 0.0:
        overload = jnp.clip(m['servo_over_rated_frac'], 0.0, 1.0)
        shaft_motion = jnp.clip(m['servo_shaft_abs_power_w'] / 20.0, 0.0, 1.0)
        if considered_reward:
            # V5021: meaningful motor economics without a prescribed gait.
            # Existing physical telemetry: no new JAX kernels or sensors.
            # Quadratic overload punishes chronic operation above the nominal
            # continuous torque, yet still permits brief emergency peaks.
            # Shaft work and stance-foot velocity discourage foot hammering;
            # action-rate discourages target jitter. Every term is bounded.
            target_jerk = jnp.clip(m['action_delta_norm'] / 3.0, 0.0, 1.0)
            contact_harshness = jnp.clip(m['foot_impact_speed_mps'] / 0.5, 0.0, 1.0)
            motor_load_proxy = (
                4.0 * overload * overload +
                1.5 * shaft_motion * shaft_motion +
                0.75 * target_jerk +
                0.5 * contact_harshness
            )
        else:
            motor_load_proxy = 0.65 * overload + 0.35 * shaft_motion
        motor_kwargs = {'motor_cost': motor_guidance_weight * motor_load_proxy}
    reward, payments = weighted_training_terms(
        alive=valid_alive, locomotion=locomotion_quality,
        posture=upright_height, contact=feet_only_contact,
        anti_slip=stance_anti_slip, clearance=swing_clearance,
        smoothness=servo_smoothness, joint_safety=joint_safety,
        stepping_gate=stepping_gate, **motor_kwargs)
    # Always expose the new diagnostic to rollout CSV, even for controls;
    # the old eight-payment reducer remains mathematically identical.
    motor_payment = payments.get('motor_cost', jnp.zeros_like(valid_alive))
    return reward, {**terms, **{f'weighted_{name}': value for name, value in payments.items()},
                    'weighted_motor_cost': motor_payment}

def weighted_training_terms(*, alive, locomotion, posture, contact,
                            anti_slip, clearance, smoothness, joint_safety,
                            stepping_gate, motor_cost=None):
    """Return (unpenalised PPO reward, individually weighted payments).

    Array inputs may be NumPy or JAX arrays; this function is pure and does
    not know about training, physics, curriculum or evaluation.
    """
    terms = {
        'alive': 0.58 * alive,
        'locomotion': 1.55 * locomotion,
        'posture': 0.68 * posture * stepping_gate,
        'contact': 0.50 * contact * stepping_gate,
        'anti_slip': 0.18 * anti_slip,
        'clearance': 0.16 * clearance,
        'smoothness': 0.12 * smoothness,
        'joint_safety': 0.17 * joint_safety,
    }
    # Explicit left-to-right addition matches the pre-extraction expression.
    reward = (terms['alive'] + terms['locomotion'] + terms['posture'] +
              terms['contact'] + terms['anti_slip'] + terms['clearance'] +
              terms['smoothness'] + terms['joint_safety'])
    # This branch is absent when disabled. In particular do not even append
    # "+ 0" in the old floating-point reduction or change the old term set.
    if motor_cost is not None:
        terms['motor_cost'] = -motor_cost
        reward = reward + terms['motor_cost']
    return reward, terms


def _command_tracking(m, is_transition, is_resilience, jnp):
    """Follow velocity, yaw and chassis pitch requests without launch hacks."""
    vx_err = m['vx'] - m['cmd_vx']; vy_err = m['vy'] - m['cmd_vy']; yaw_err = m['yaw_rate'] - m['cmd_yaw']
    pitch_err = m['body_pitch'] - m['cmd_pitch']
    # Penalise both underspeed and ballistic overspeed; RESILIENCE gets
    # a wider tolerance while catching an imposed disturbance.
    speed_tol = jnp.where(is_resilience > 0.5, SPEED_TOLERANCE * RESILIENCE_TOLERANCE_MULT, SPEED_TOLERANCE)
    overspeed = jnp.maximum(0.0, m['vx'] - m['cmd_vx'] - speed_tol)
    underspeed = jnp.maximum(0.0, m['cmd_vx'] - m['vx'] - speed_tol)
    # One tracking formula: tougher while ramping, gentler in RESILIENCE.
    ramping = (m['ramp_frac'] < 1.0).astype(jnp.float32)
    tracking_gain = 1.0 + is_transition * ramping * (TRANSITION_TRACKING_BOOST - 1.0)
    tracking_gain = tracking_gain * jnp.where(is_resilience > 0.5, RESILIENCE_TRACKING_RELIEF, 1.0)
    command_tracking = jnp.exp(-tracking_gain * (62.0 * vx_err * vx_err + 14.0 * vy_err * vy_err + 4.5 * yaw_err * yaw_err
                                                  + 18.0 * pitch_err * pitch_err
                                                  + 120.0 * overspeed * overspeed + 40.0 * underspeed * underspeed))
    # Early frames score neutral tracking while initial pose noise settles;
    # a genuine fall still terminates and incurs its penalty.
    settled = (m['age_s'] > SETTLE_STEPS * CONTROL_DT).astype(jnp.float32)
    command_tracking = settled * command_tracking + (1.0 - settled) * 1.0
    return command_tracking


def _physical_qualities(m, attitude_profile, jnp):
    """Measure posture, supported feet, slip, movement shape and joint safety."""
    upright = jnp.clip((m['gravity_z'] - 0.35) / 0.65, 0.0, 1.0)
    height = jnp.exp(-80.0 * (m['height'] - m['target_height']) ** 2)
    # Body-to-feet angle is dynamic catch-up shaping, not static standing.
    if attitude_profile == 'v5015':
        # Require requested chassis pitch AND healthy gait height.
        pitch_allowance = 0.018 * jnp.clip(jnp.abs(m['cmd_pitch']) / 0.7853982, 0.0, 1.0)
        height_shortfall = jnp.maximum(0.0, -m['height_error'] - 0.012 - pitch_allowance)
        height = jnp.exp(-1000.0 * height_shortfall * height_shortfall)
        upright_height = 0.32 * m['attitude_quality'] + 0.48 * height + 0.20 * m['foot_mid_cone']
    else:
        upright_height = 0.52 * upright + 0.28 * height + 0.20 * m['foot_mid_cone']
    feet_only_contact = (1.0 - m['bad_contact']) * (0.35 + 0.65 * m['support'])
    stance_anti_slip = jnp.exp(-6.0 * m['stance_slip'])
    raw_swing = jnp.clip(m['swing_clearance'] / jnp.maximum(0.006, m['target_step_height']), 0.0, 1.0)
    # Reward emerging alternating foot lift, without prescribing foot timing.
    swing_clearance = jnp.clip(0.55 * raw_swing + 0.45 * m['organic_step_shape'], 0.0, 1.0)
    servo_smoothness = jnp.exp(-0.22 * m['action_delta_norm'])
    joint_safety = jnp.exp(-(5.0 * m['joint_limit_pressure'] + 2.0 * m['scissor_pressure']))
    return (upright_height, feet_only_contact, stance_anti_slip,
            swing_clearance, servo_smoothness, joint_safety)


def _gait_maturity(m, jnp):
    """Reduce free rewards for long periods without observable foot activity."""
    # Every regime practises moving; don't reward endless balancing.
    locomotion_maturity = jnp.clip((m['age_s'] - 0.20) / 0.45, 0.0, 1.0)
    # Upright/contact payment fades if the EPISODE-LONG stepping proxy
    # stays near zero. Do not gate individual stance frames or fresh births.
    stepping_factor = jnp.clip(m['stepping_activity_so_far'] / STEPPING_ACTIVITY_TARGET, 0.0, 1.0)
    # A 30% floor avoids a hard failure cliff once the grace period expires.
    stepping_gate = jnp.where(m['age_s'] > STEPPING_GATE_GRACE_S, 0.30 + 0.70 * stepping_factor, 1.0)
    return locomotion_maturity, stepping_gate


def apply_early_fall_penalty(reward, terminal_fail, episode_age_steps,
                             max_episode_steps, *, xp):
    """Same early-death penalty as accepted main. Timeouts receive no penalty.

    The second return value is a negative *reward contribution*, not a measured
    motor effort. It is recorded on the pre-reset frame for diagnostics.
    """
    amount = (EARLY_FALL_BASE_PENALTY +
              EARLY_FALL_TIME_PENALTY *
              xp.maximum(0.0, max_episode_steps - episode_age_steps))
    penalized = xp.where(terminal_fail, reward - amount, reward)
    payment = xp.where(terminal_fail, -amount, xp.zeros_like(reward))
    return penalized, payment
