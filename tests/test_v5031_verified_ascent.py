"""Only the new reward contract: no GPU or actor dependency."""
import numpy as np

from emu.rewards import verified_support_elevation_credit


def test_verified_transfer_requires_grounded_target_contact_and_matching_cog_motion():
    delta = np.array([.03, .03, .03, .03], dtype=np.float32)
    cog = np.array([.025, .025, -.025, .025], dtype=np.float32)
    contact = np.array([True, False, True, True])
    stable = np.array([True, True, True, False])
    credit, peak = verified_support_elevation_credit(
        delta, cog, contact, stable, np.zeros(4, np.float32),
        direction=1, xp=np, max_rise_m=.04, total_credit=3.,
        normalize_by_actual_height=True)
    np.testing.assert_allclose(credit, [2.5, 0., 0., 0.], rtol=1e-5)
    np.testing.assert_allclose(peak, [.025, 0., 0., 0.], rtol=1e-5)

    # A second contact on the same high-water mark cannot farm reward.
    again, peak2 = verified_support_elevation_credit(
        delta, cog, contact, stable, peak,
        direction=1, xp=np, max_rise_m=.04, total_credit=3.,
        normalize_by_actual_height=True)
    np.testing.assert_allclose(again, np.zeros(4), atol=1e-6)
    np.testing.assert_allclose(peak2, peak)


def test_verified_transfer_not_paid_on_flat_or_downward_without_down_contract():
    credit, peak = verified_support_elevation_credit(
        np.array([0., -.03]), np.array([.10, .10]),
        np.ones(2, dtype=bool), np.ones(2, dtype=bool),
        np.zeros(2), direction=1, xp=np, total_credit=3.,
        normalize_by_actual_height=True)
    np.testing.assert_array_equal(credit, [0., 0.])
    np.testing.assert_array_equal(peak, [0., 0.])


def test_verified_transfer_cannot_exceed_reallocated_budget():
    credit, peak = verified_support_elevation_credit(
        np.array([.03]), np.array([.50]),
        np.array([True]), np.array([True]),
        np.array([0.]), direction=1, xp=np,
        max_rise_m=.04, total_credit=3.,
        normalize_by_actual_height=True)
    np.testing.assert_allclose(credit, [3.])
    np.testing.assert_allclose(peak, [.03])
