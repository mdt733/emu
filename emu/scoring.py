"""One auditable ledger for the eight existing PPO reward payments.

These values are *training* reward contributions, not held-out promotion
scores or measured electrical energy. Keep order/associativity equivalent to
the original environment._reward expression when updating this function.
"""


def weighted_training_terms(*, alive, locomotion, posture, contact,
                            anti_slip, clearance, smoothness, joint_safety,
                            stepping_gate):
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
    return reward, terms


# Existing v5015 doctrine: moved, not retuned. All reward constants now live
# beside the calculations they govern instead of being mixed into task physics.
from .contracts import CONTROL_DT

EARLY_FALL_BASE_PENALTY = 65.0
EARLY_FALL_TIME_PENALTY = 0.10
STEPPING_ACTIVITY_TARGET = 0.15
SPEED_TOLERANCE = 0.025
STEPPING_GATE_GRACE_S = 1.0
SETTLE_STEPS = 25
TRANSITION_TRACKING_BOOST = 1.5
RESILIENCE_TOLERANCE_MULT = 3.0
RESILIENCE_TRACKING_RELIEF = 0.5

def training_reward(m, regime, *, attitude_profile, xp):
    jnp = xp
    # Every regime spawns already moving toward a real commanded state
    # (see the module doctrine and _reset_with_keys), so there's no
    # standing-still special case here -- what's regime-conditioned
    # below is real, targeted shaping, not a special-cased behavior:
    is_transition = jnp.logical_and(regime > 0.5, regime < 1.5).astype(jnp.float32)
    is_resilience = (regime > 1.5).astype(jnp.float32)
    valid_alive = 1.0 - m['terminal']
    vx_err = m['vx'] - m['cmd_vx']; vy_err = m['vy'] - m['cmd_vy']; yaw_err = m['yaw_rate'] - m['cmd_yaw']
    pitch_err = m['body_pitch'] - m['cmd_pitch']
    # Forward overspeed must not be rewarded almost as strongly as clean
    # tracking, or a policy can discover a forward launch into terminal
    # contact -- imperfect motion is fine, a ballistic "close enough,
    # then faceplant" hack is not. RESILIENCE gets a wider dead zone
    # here specifically -- see RESILIENCE_TOLERANCE_MULT's own comment
    # for why.
    speed_tol = jnp.where(is_resilience > 0.5, SPEED_TOLERANCE * RESILIENCE_TOLERANCE_MULT, SPEED_TOLERANCE)
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
                                                  + 18.0 * pitch_err * pitch_err
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
    if attitude_profile == 'v5015':
        # Reward the requested chassis orientation without allowing an
        # increasingly low gait to masquerade as improved locomotion.
        pitch_allowance = 0.018 * jnp.clip(jnp.abs(m['cmd_pitch']) / 0.7853982, 0.0, 1.0)
        height_shortfall = jnp.maximum(0.0, -m['height_error'] - 0.012 - pitch_allowance)
        height = jnp.exp(-1000.0 * height_shortfall * height_shortfall)
        upright_height = 0.32 * m['attitude_quality'] + 0.48 * height + 0.20 * m['foot_mid_cone']
    else:
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
    # Anti-freezing gate. Earlier reward versions made sustained balancing
    # without stepping too attractive because survival/upright/contact/
    # smoothness can all score well while motion evidence stays near zero.
    # The exact historical arithmetic belongs in docs/history.md; the live
    # invariant is simply that sustained non-stepping must not outscore
    # genuine locomotion.
    #
    # Gate upright_height and feet_only_contact's contribution to
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
    stepping_factor = jnp.clip(m['stepping_activity_so_far'] / STEPPING_ACTIVITY_TARGET, 0.0, 1.0)
    # Every regime commands real, non-zero movement, so there's no
    # "standing still is correct" case to exempt from this gate.
    stepping_gate = jnp.where(m['age_s'] > STEPPING_GATE_GRACE_S, 0.30 + 0.70 * stepping_factor, 1.0)
    # Only reward components that are actually inspected off-device cross
    # the telemetry boundary. The remaining terms still shape reward below.
    terms = {
        'valid_alive': valid_alive,
        'command_tracking': command_tracking,
        'upright_height': upright_height,
        'feet_only_contact': feet_only_contact,
        'swing_clearance': swing_clearance,
    }
    # Reward remains the same 8-term doctrine, but survival/contact now gate
    # most locomotion payment.  This keeps the reward pure while making
    # short terminal episodes mathematically unattractive.
    locomotion_quality = locomotion_maturity * command_tracking * upright_height * (0.35 + 0.65 * feet_only_contact) * (0.30 + 0.70 * m['foot_mid_cone'])
    if attitude_profile == 'v5015':
        remaining = jnp.maximum(0.0, -m['height_error'] - 0.012 -
                                0.018*jnp.clip(jnp.abs(m['cmd_pitch'])/0.7853982,0.0,1.0))
        locomotion_quality *= jnp.exp(-1200.0 * remaining * remaining)
    reward, payments = weighted_training_terms(
        alive=valid_alive, locomotion=locomotion_quality,
        posture=upright_height, contact=feet_only_contact,
        anti_slip=stance_anti_slip, clearance=swing_clearance,
        smoothness=servo_smoothness, joint_safety=joint_safety,
        stepping_gate=stepping_gate)
    return reward, {**terms, **{f'weighted_{name}': value for name, value in payments.items()}}


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
