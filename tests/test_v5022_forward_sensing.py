"""Phase-1 forward-beam geometry tests; no MuJoCo dependency.

These are sensor visibility tests, NOT MJX physics, PPO, or crossing tests.
"""
import numpy as np
from emu.forward_sensing import (
    FORWARD_TOF_MOUNTS_BODY_M, forward_ray_geometry,
    ray_box_distances, scan_box_course,
)


def course(*, edge=0.16, left=0.0, right=0.0):
    # Two genuinely different supporting top elevations. The boxes are
    # solid beneath the top surface, not an invisible marker on a plane.
    low = np.array([[-2., -1., -2.], [edge, -1., -2.]])
    high = np.array([[edge, 1., left], [2., 1., right]])
    low[0, 0] = -2.
    high[0, 0] = edge
    low[1, 0] = edge
    return low, high


def pose(worlds=1):
    return (np.tile(np.array([[0., 0., .2584]]), (worlds, 1)),
            np.tile(np.eye(3)[None, :, :], (worlds, 1, 1)))


def scan(*, edge=0.16, left=0.0, right=0.0, **kw):
    low, high = course(edge=edge, left=left, right=right)
    t, R = pose()
    return scan_box_course(np, t, R, low, high, **kw)


def test_forward_fan_is_normalized_and_reaches_ahead_of_feet():
    t, R = pose()
    origins, rays = forward_ray_geometry(np, t, R)
    assert origins.shape == rays.shape == (1, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0)
    assert all(m[0] > .076 for m in FORWARD_TOF_MOUNTS_BODY_M), 'ToF must not start inside torso box'
    assert (rays[0, :, 0] > 0.65).all() and (rays[0, :, 2] < -0.65).all()


def test_flat_seam_matches_unbroken_floor():
    a, valid_a = scan(edge=.16)
    b, valid_b = scan(edge=.50)
    assert valid_a.all() and valid_b.all()
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_riser_and_drop_visible_before_contact():
    base, _ = scan(edge=.16)
    rise, rise_ok = scan(edge=.16, right=.05)
    drop, drop_ok = scan(edge=.16, right=-.05)
    assert rise_ok.all() and drop_ok.all()
    assert np.all(rise < base - .025), (base, rise)
    assert np.all(drop > base + .025), (base, drop)


def test_remote_step_does_not_leak_height_through_geometry():
    base, _ = scan(edge=.60)
    up, _ = scan(edge=.60, right=.08)
    down, _ = scan(edge=.60, right=-.08)
    np.testing.assert_allclose(up, base)
    np.testing.assert_allclose(down, base)


def test_vertical_riser_occludes_top_of_high_obstacle():
    base, _ = scan(edge=.16)
    high, valid = scan(edge=.16, right=.20)
    assert valid.all()
    assert np.all(high < base - .08)


def test_dropout_and_range_limit_emit_invalid_not_flat():
    r, valid = scan(dropout=np.array([[True, False, False]]))
    assert not valid[0, 0] and r[0, 0] == 0
    assert valid[0, 1:].all()
    missing, good = scan(max_range_m=.20)
    assert not good.any() and not missing.any()


def test_parallel_and_missed_rays_are_finite_or_infinite_not_nan():
    origin = np.array([[[0., 0., 1.], [0., 1., 0.]]])
    dirs = np.array([[[1., 0., 0.], [0., 0., 1.]]])
    lo = np.array([[.2, -.2, -.2]])
    hi = np.array([[.4, .2, .2]])
    ranges = ray_box_distances(np, origin, dirs, lo, hi)
    assert np.isinf(ranges).all()
    assert not np.isnan(ranges).any()


def test_vectorized_worlds_can_have_different_step_heights():
    t, R = pose(worlds=2)
    lo = np.array([
        [[-2., -1., -2.], [.16, -1., -2.]],
        [[-2., -1., -2.], [.16, -1., -2.]]])
    hi = np.array([
        [[.16, 1., 0.], [2., 1., .05]],
        [[.16, 1., 0.], [2., 1., -.05]]])
    ranges, valid = scan_box_course(np, t, R, lo, hi)
    assert ranges.shape == valid.shape == (2, 3)
    assert valid.all()
    assert (ranges[0] < ranges[1]).all()


def test_jax_jit_matches_numpy_for_fixed_box_shapes():
    import jax
    import jax.numpy as jnp
    t, R = pose(worlds=2)
    lo = np.array([[-2., -1., -2.], [.16, -1., -2.]])
    hi = np.array([[.16, 1., 0.], [2., 1., .05]])
    expect, _ = scan_box_course(np, t, R, lo, hi)
    f = jax.jit(lambda p, rot: scan_box_course(
        jnp, p, rot, jnp.asarray(lo), jnp.asarray(hi)))
    got, good = f(jnp.asarray(t), jnp.asarray(R))
    assert np.asarray(good).all()
    np.testing.assert_allclose(np.asarray(got), expect, atol=1e-6)
