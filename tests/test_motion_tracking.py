"""Independent NumPy checks for v5015 ground-heading velocity semantics."""
import math
import numpy as np
from emu.environment import horizontal_tracking_motion


def test_forward_speed_is_invariant_to_camera_pitch():
    speed = .32
    v = np.asarray([[speed, 0., 0.]], dtype=np.float64)
    w = np.zeros_like(v)
    for degrees in (0, 8, 25, 45):
        a = math.radians(degrees)
        c, s = math.cos(a), math.sin(a)
        R = np.asarray([[[c, 0, s], [0, 1, 0], [-s, 0, c]]])
        vx, vy, yaw = horizontal_tracking_motion(np, R, v, w)
        np.testing.assert_allclose(vx, [speed], atol=1e-12)
        np.testing.assert_allclose(vy, [0], atol=1e-12)
        np.testing.assert_allclose(yaw, [0], atol=1e-12)


def test_yaw_rotates_forward_and_lateral_heading():
    R = np.asarray([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float64)
    v = np.asarray([[0, .4, 0]], dtype=np.float64)
    w = np.asarray([[0, 0, .6]], dtype=np.float64)
    vx, vy, yaw = horizontal_tracking_motion(np, R, v, w)
    np.testing.assert_allclose(vx, [.4])
    np.testing.assert_allclose(vy, [0])
    np.testing.assert_allclose(yaw, [.6])


def test_sideways_speed_retains_sign_when_chassis_pitches():
    a = math.radians(45)
    R = np.asarray([[[math.cos(a), 0, math.sin(a)],
                     [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]]])
    vx, vy, _ = horizontal_tracking_motion(
        np, R, np.asarray([[0, -.20, 0]]), np.zeros((1, 3)))
    np.testing.assert_allclose(vx, [0], atol=1e-12)
    np.testing.assert_allclose(vy, [-.2], atol=1e-12)
