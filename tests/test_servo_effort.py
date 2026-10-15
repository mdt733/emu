"""Read-only physical-effort telemetry, not an electric/thermal servo model."""
import numpy as np

from emu.environment import servo_effort_metrics
from emu.rollout_metrics import summarize_rollout


def test_torque_times_joint_speed_splits_drive_and_braking_without_claiming_regen():
    tau = np.array([[0., 0., 0., 0., 0., 0.],
                    [1.0, -0.5, .2, 0., 0., 0.]], dtype=np.float32)
    omega = np.array([[5., 0., 0., 0., 0., 0.],
                      [2., 2., -5., 0., 0., 0.]], dtype=np.float32)
    limits = np.full(6, 1.913, dtype=np.float32)
    v = servo_effort_metrics(np, tau, omega, limits)
    # World 2: +2 W drive, -1 W and -1 W braking; no net electrical estimate.
    np.testing.assert_allclose(v['servo_shaft_abs_power_w'], [0, 4], atol=1e-6)
    np.testing.assert_allclose(v['servo_shaft_drive_power_w'], [0, 2], atol=1e-6)
    np.testing.assert_allclose(v['servo_shaft_brake_power_w'], [0, 2], atol=1e-6)
    np.testing.assert_allclose(v['servo_over_rated_frac'], [0, 1/6], atol=1e-6)
    np.testing.assert_allclose(v['servo_peak_torque_limit_frac'], [0, 1/1.913], atol=1e-6)


def test_stall_can_require_torque_without_mechanical_shaft_power():
    tau = np.full((2, 6), 1.913, dtype=np.float32)
    stationary = np.zeros_like(tau)
    result = servo_effort_metrics(np, tau, stationary, np.full(6, 1.913))
    np.testing.assert_array_equal(result['servo_shaft_abs_power_w'], [0., 0.])
    np.testing.assert_array_equal(result['servo_over_rated_frac'], [1., 1.])
    np.testing.assert_array_equal(result['servo_peak_torque_limit_frac'], [1., 1.])


def test_servo_indicators_flow_to_rollout_report_but_not_training_reward():
    import inspect
    from emu import scoring
    # Intentional non-dependency: rewarding low torque without an explicit
    # hypothesis would allow freezing to look energy-efficient.
    assert 'servo_' not in inspect.getsource(scoring.training_reward)
    fields = servo_effort_metrics(
        np, np.ones((2,6), dtype=np.float32),
        np.ones((2,6), dtype=np.float32),
        np.full(6, 1.913, dtype=np.float32),
    )
    row = summarize_rollout({k: v[None, :] for k,v in fields.items()})
    assert row['servo_shaft_abs_power_w'] == 6.0
    assert row['servo_over_rated_frac'] == 1.0


def test_effort_metrics_jax_jit_supports_vector_batches():
    import jax
    import jax.numpy as jnp
    @jax.jit
    def measured(tau, rate):
        return servo_effort_metrics(jnp, tau, rate, jnp.full((6,), 1.913))
    measured_values = jax.device_get(measured(jnp.ones((8,6)), jnp.ones((8,6))))
    assert measured_values['servo_shaft_drive_power_w'].shape == (8,)
    np.testing.assert_allclose(measured_values['servo_shaft_drive_power_w'], 6.)


def test_passive_spring_energy_matches_mujoco_hinge_formula():
    from emu.environment import joint_spring_energy
    q = np.array([[0., 0., 0., 0., 0., 0., .10, -.20],
                  [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)
    ref = np.zeros(8, dtype=np.float32)
    k = np.array([2., .48, .48, 2., .48, .48, 4., 4.], dtype=np.float32)
    readings = joint_spring_energy(np, q, ref, k)
    # Two compliant shins store 0.5 * 4 * (0.1^2 + 0.2^2) = 0.1 J.
    np.testing.assert_allclose(readings['shin_spring_energy_j'], [.10, 0], atol=1e-6)
    np.testing.assert_allclose(readings['spring_potential_energy_j'], [.10, 0], atol=1e-6)
    result = summarize_rollout({k: v[None, :] for k, v in readings.items()})
    assert abs(result['spring_potential_energy_j'] - .05) < 1e-6


def test_spring_energy_supports_jax_jit_and_joint_reference_offset():
    import jax
    import jax.numpy as jnp
    from emu.environment import joint_spring_energy

    @jax.jit
    def result(q):
        return joint_spring_energy(
            jnp, q, jnp.array([.1, -.2]), jnp.array([2., 4.]))

    q = jnp.array([[.1, -.2], [.3, -.1]], dtype=jnp.float32)
    readings = jax.device_get(result(q))
    np.testing.assert_allclose(readings['spring_potential_energy_j'],
                               [0, .06], atol=1e-6)
