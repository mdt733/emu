"""Hardware-plausible one-module multizone ToF diagnostic contracts.

The five-ray response approximation is not a VL53L5CX driver emulator and
must not be treated as perfect measurements at a discontinuity.
"""
import numpy as np
import pytest
from emu.forward_sensing import (
    MULTIZONE_MOUNT_BODY_M, multizone_ray_geometry,
    scan_multizone_box_course,
)


def scene(height):
    return (
        np.array([[-2., -1., -1.], [.16, -1., -1.]]),
        np.array([[.16, 1., 0.], [2., 1., height]]),
    )


def frame(w=1):
    return (
        np.tile([[0., 0., .2584]], (w, 1)),
        np.tile(np.eye(3)[None, :, :], (w, 1, 1)),
    )


def get(height):
    t, R = frame()
    lo, hi = scene(height)
    return scan_multizone_box_course(np, t, R, lo, hi)


def test_one_physical_camera_shared_origin_and_4x4_zone_layout():
    t, R = frame(2)
    origins, direction = multizone_ray_geometry(np, t, R)
    assert origins.shape == direction.shape == (2, 16, 5, 3)
    assert MULTIZONE_MOUNT_BODY_M[0] > .076
    np.testing.assert_allclose(origins, origins[:, 0:1, 0:1, :], atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1., atol=1e-7)
    # The outer zones scan different horizontal and vertical angles.
    assert direction[0, 0, 0, 1] < direction[0, 3, 0, 1]
    assert direction[0, 0, 0, 2] > direction[0, 12, 0, 2]


def test_neutral_floor_reports_ranges_not_an_imagined_elevation_grid():
    out = get(0.0)
    assert out['ranges_m'].shape == (1, 4, 4)
    assert out['valid'].shape == (1, 4, 4)
    assert np.any(out['valid'])
    assert np.all(out['ranges_m'][out['valid']] > 0)
    assert np.all((out['visibility_confidence'] >= 0) &
                  (out['visibility_confidence'] <= 1))


def test_higher_and_lower_ground_produce_distinct_zone_patterns():
    flat = get(0.0)
    up = get(.07)
    down = get(-.07)
    common = flat['valid'] & up['valid'] & down['valid']
    # Some zones must see differences ahead of foot contact. This does not
    # imply every zone sees the discontinuity at this pose.
    assert common.any()
    assert (np.abs(up['ranges_m'][common] -
                   flat['ranges_m'][common]) > .015).any()
    assert (np.abs(down['ranges_m'][common] -
                   flat['ranges_m'][common]) > .015).any()


def test_dropout_and_occlusion_do_not_fabricate_a_flat_surface():
    t, R = frame()
    lo, hi = scene(.05)
    miss = np.ones((1, 4, 4), dtype=bool)
    result = scan_multizone_box_course(np, t, R, lo, hi, dropout=miss)
    assert not result['valid'].any()
    assert not result['ranges_m'].any()
    assert not result['visibility_confidence'].any()


def test_body_pitch_changes_visible_patch_with_no_privileged_compensation():
    t, R = frame()
    tilted = R.copy()
    theta = np.deg2rad(12)
    tilted[0] = np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.],
        [-np.sin(theta), 0., np.cos(theta)],
    ])
    lo, hi = scene(.05)
    a = scan_multizone_box_course(np, t, R, lo, hi)['ranges_m']
    b = scan_multizone_box_course(np, t, tilted, lo, hi)['ranges_m']
    assert not np.allclose(a, b)


def test_jax_zone_geometry_and_return_shapes_if_available():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    t, R = frame(2)
    lo, hi = scene(.05)
    expected = scan_multizone_box_course(np, t, R, lo, hi)
    fun = jax.jit(lambda p, rot: scan_multizone_box_course(
        jnp, p, rot, jnp.asarray(lo), jnp.asarray(hi)))
    observed = fun(jnp.asarray(t), jnp.asarray(R))
    for key in ('ranges_m', 'valid', 'visibility_confidence'):
        np.testing.assert_allclose(
            np.asarray(observed[key]), expected[key], atol=1e-5)
