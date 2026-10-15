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


def test_environment_reward_calls_ledger_and_reports_weighted_payments():
    """Exercise the real _reward method without MuJoCo, for both posture profiles."""
    from types import SimpleNamespace
    from emu.environment import EmuMJXEnv
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
        environment = SimpleNamespace(jnp=np, attitude_profile=profile)
        reward, terms = EmuMJXEnv._reward(environment, measurements, u(0, 3))
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
