#!/usr/bin/env python3
"""Shared locomotion tracking doctrine for EMU v5012.

The capability field, held-out node calibration exam and frontier governor all
use the same physical error tolerances and the same *window-mean* idea.  They
may apply the contract to different cohorts, but a walking gait is never judged
by requiring every instantaneous frame to sit inside the tolerance box.
"""
from __future__ import annotations

import math
import numpy as np

VX_ABS_TOL = 0.08
VX_REL_TOL = 0.25
VY_ABS_TOL = 0.07
VY_REL_TOL = 0.35
YAW_ABS_TOL = 0.25
YAW_REL_TOL = 0.25
UPRIGHT_GRAVITY_Z_MIN = math.cos(math.radians(20.0))
SUPPORT_MIN = 0.05
BAD_CONTACT_FRACTION_MAX = 0.10
NODE_WINDOW_S = 1.00
NODE_SETTLE_DELAY_S = 0.50
RECOVERY_STABLE_S = 0.24
EDGE_STABLE_S = 0.24


def dynamic_tolerance(cmd, abs_floor: float, rel_frac: float):
    """NumPy dynamic tolerance used by host-side diagnostics/tests."""
    return np.maximum(float(abs_floor), float(rel_frac) * np.abs(np.asarray(cmd)))


def tracking_tolerances(vx_cmd, vy_cmd, yaw_cmd):
    return (
        dynamic_tolerance(vx_cmd, VX_ABS_TOL, VX_REL_TOL),
        dynamic_tolerance(vy_cmd, VY_ABS_TOL, VY_REL_TOL),
        dynamic_tolerance(yaw_cmd, YAW_ABS_TOL, YAW_REL_TOL),
    )


def node_window_success_numpy(cmd_vx, cmd_vy, cmd_yaw, vx, vy, yaw_rate,
                              gravity_z, support, bad_contact, fell=None,
                              axis: int = 0):
    """Return Bernoulli success for one complete command operating window.

    Success is based on absolute window-mean tracking error plus window-mean
    posture/support, not the fraction of individual gait frames inside a box.
    ``axis`` is the time axis; all remaining dimensions are preserved.
    """
    cvx=np.asarray(cmd_vx); cvy=np.asarray(cmd_vy); cy=np.asarray(cmd_yaw)
    vx=np.asarray(vx); vy=np.asarray(vy); yw=np.asarray(yaw_rate)
    gz=np.asarray(gravity_z); sup=np.asarray(support); bad=np.asarray(bad_contact)
    tx,ty,tz=tracking_tolerances(cvx,cvy,cy)
    ok=(
        np.abs(np.mean(vx-cvx,axis=axis)) <= np.mean(tx,axis=axis)
    ) & (
        np.abs(np.mean(vy-cvy,axis=axis)) <= np.mean(ty,axis=axis)
    ) & (
        np.abs(np.mean(yw-cy,axis=axis)) <= np.mean(tz,axis=axis)
    ) & (
        np.mean(gz,axis=axis) > UPRIGHT_GRAVITY_Z_MIN
    ) & (
        np.mean(sup,axis=axis) > SUPPORT_MIN
    ) & (
        np.mean(bad > 0.5,axis=axis) <= BAD_CONTACT_FRACTION_MAX
    )
    if fell is not None:
        ok = ok & (~np.any(np.asarray(fell) > 0.5,axis=axis))
    return ok
