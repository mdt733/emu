"""Numerical baseline contract for the existing eight reward payments.

This is CPU-only arithmetic evidence, not a MuJoCo stepping test.
"""
import numpy as np

from emu.scoring import weighted_training_terms


def test_weighted_reward_is_bitwise_equivalent_to_original_float32_expression():
    rng = np.random.default_rng(733)
    inputs = [rng.uniform(0, 1, 512).astype(np.float32) for _ in range(9)]
    alive, locomotion, posture, contact, anti_slip, clearance, smoothness, safety, gate = inputs
    original = (0.58*alive + 1.55*locomotion + 0.68*posture*gate +
                0.50*contact*gate + 0.18*anti_slip + 0.16*clearance +
                0.12*smoothness + 0.17*safety)
    actual, parts = weighted_training_terms(
        alive=alive, locomotion=locomotion, posture=posture, contact=contact,
        anti_slip=anti_slip, clearance=clearance, smoothness=smoothness,
        joint_safety=safety, stepping_gate=gate)
    assert set(parts) == {'alive', 'locomotion', 'posture', 'contact',
                          'anti_slip', 'clearance', 'smoothness', 'joint_safety'}
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(actual, sum(parts.values()))


def test_scoring_reward_reaches_rollout_ledger_for_both_profiles():
    """Exercise the real _reward method without MuJoCo, for both posture profiles."""
    from emu.scoring import training_reward, apply_early_fall_penalty
    from emu.rollout_metrics import summarize_rollout

    rng = np.random.default_rng(733)
    n = 48
    def u(a, b):
        return rng.uniform(a, b, n).astype(np.float32)
    measurements = dict(
        terminal=u(0, 1).round(), vx=u(-.2, .5), cmd_vx=u(.05, .4),
        vy=u(-.1, .1), cmd_vy=u(-.1, .1),
        yaw_rate=u(-.4, .4), cmd_yaw=u(-.3, .3),
        body_pitch=u(-.2, .2), cmd_pitch=u(-.25, .25),
        gravity_z=u(.5, 1.0), height=u(.20, .3),
        target_height=u(.22, .28), height_error=u(-.025, .025),
        attitude_quality=u(.3, 1), foot_mid_cone=u(.3, 1),
        bad_contact=u(0, .5), support=u(0, 1), stance_slip=u(0, .1),
        swing_clearance=u(0, .06), target_step_height=u(.02, .09),
        organic_step_shape=u(0, 1), action_delta_norm=u(0, 2),
        joint_limit_pressure=u(0, .5), scissor_pressure=u(0, .4),
        age_s=u(.01, 10), ramp_frac=u(0, 1),
        stepping_activity_so_far=u(0, 1),
    )
    names = ('alive', 'locomotion', 'posture', 'contact', 'anti_slip',
             'clearance', 'smoothness', 'joint_safety')
    for profile in ('v5014', 'v5015'):
        reward, terms = training_reward(measurements, u(0, 3), attitude_profile=profile, xp=np)
        np.testing.assert_allclose(
            reward, sum(terms[f'weighted_{name}'] for name in names),
            rtol=2e-6, atol=2e-6,
        )
        # The step implementation uses the same reward_{key} prefix.
        packet = {f'reward_{key}': value[None, :]
                  for key, value in terms.items()}
        packet['reward_weighted_fall_penalty'] = np.zeros((1, n), np.float32)
        reduced = summarize_rollout(packet)
        for name in (*names, 'fall_penalty'):
            np.testing.assert_allclose(
                reduced[f'reward_weighted_{name}'],
                np.mean(packet[f'reward_weighted_{name}']),
                rtol=1e-6,
            )


def test_early_fall_penalty_matches_accepted_main_and_exempts_timeouts():
    """Read-only oracle of the original environmental terminal arithmetic."""
    from emu.scoring import apply_early_fall_penalty
    rewards = np.array([0.2, 1.0, -0.3, 1.7], np.float32)
    fails = np.array([True, False, True, False])
    ages = np.array([1., 1., 500., 700.], np.float32)
    horizon = 700
    old_amount = 65.0 + 0.10 * np.maximum(0.0, horizon - ages)
    old_reward = np.where(fails, rewards - old_amount, rewards)
    expected_payment = np.where(fails, -old_amount, np.zeros_like(rewards))
    result, payment = apply_early_fall_penalty(rewards, fails, ages, horizon, xp=np)
    np.testing.assert_array_equal(result, old_reward)
    np.testing.assert_array_equal(payment, expected_payment)


def test_training_reward_and_terminal_payment_jit_compile_in_jax():
    import jax
    import jax.numpy as jnp
    from emu.scoring import training_reward, apply_early_fall_penalty

    m = dict(
        terminal=jnp.array([0., 1.]), vx=jnp.array([.2, .4]),
        cmd_vx=jnp.array([.2, .3]), vy=jnp.zeros(2), cmd_vy=jnp.zeros(2),
        yaw_rate=jnp.zeros(2), cmd_yaw=jnp.zeros(2),
        body_pitch=jnp.zeros(2), cmd_pitch=jnp.zeros(2),
        gravity_z=jnp.ones(2), height=jnp.full(2, .25),
        target_height=jnp.full(2, .25), height_error=jnp.zeros(2),
        attitude_quality=jnp.ones(2), foot_mid_cone=jnp.ones(2),
        bad_contact=jnp.zeros(2), support=jnp.ones(2),
        stance_slip=jnp.zeros(2), swing_clearance=jnp.array([.02, .0]),
        target_step_height=jnp.full(2, .04),
        organic_step_shape=jnp.zeros(2), action_delta_norm=jnp.zeros(2),
        joint_limit_pressure=jnp.zeros(2), scissor_pressure=jnp.zeros(2),
        age_s=jnp.array([1.0, 1.0]), ramp_frac=jnp.ones(2),
        stepping_activity_so_far=jnp.array([.2, 0.]),
    )
    @jax.jit
    def one_step(values):
        raw, parts = training_reward(values, jnp.zeros(2), attitude_profile='v5015', xp=jnp)
        out, fall = apply_early_fall_penalty(raw, values['terminal'] > .5,
                                             jnp.array([10., 10.]), 700, xp=jnp)
        return out, parts['weighted_locomotion'], fall

    reward, locomotion, fall = jax.device_get(one_step(m))
    assert np.isfinite(reward).all() and np.isfinite(locomotion).all()
    assert fall[0] == 0.0 and fall[1] < 0.0
