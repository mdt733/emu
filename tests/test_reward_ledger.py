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
