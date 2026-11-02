"""Four real sharp-lip platforms for the existing STEP specialist.

Only physical ray measurements reach the policy; no terrain oracle does.
"""
from __future__ import annotations

ACADEMY_LIPS = 4
ACADEMY_INTERVAL_M = 0.35
ACADEMY_EDGE_X_M = 0.25
ACADEMY_MAX_EDGE_YAW_DEG = 3.0
ACADEMY_MIDDLE_HALF_LENGTH_M = ACADEMY_INTERVAL_M / 2
ACADEMY_MIDDLE_HALF_WIDTH_M = 16.0
ACADEMY_MIDDLE_HALF_DEPTH_M = 0.25
ACADEMY_MIDDLE_NAMES = tuple(f"academy_platform_{i}" for i in range(3))
ACADEMY_MIDDLE_GEOMS = tuple(f"academy_surface_{i}" for i in range(3))

# All six two-up/two-down orders, randomly chosen at each training birth.
ACADEMY_PATTERNS = (
    (+1,+1,-1,-1), (+1,-1,+1,-1),
    (-1,+1,-1,+1), (-1,-1,+1,+1),
    (+1,-1,-1,+1), (-1,+1,+1,-1),
)


def course_heights(xp, index, samples, difficulty_m):
    """Five real platforms from four independently sized signed lips.

    Shift the entire walk so the lowest plateau is zero, without clipping
    away real down steps. The maximum elevation span is 2*difficulty_m.
    'index' is the sampled course pattern ID in each world.
    """
    h = xp.maximum(.0015, difficulty_m * (.55 + .45*samples))
    signs = xp.asarray(ACADEMY_PATTERNS, dtype=h.dtype)[
        xp.mod(index.astype(xp.int32), len(ACADEMY_PATTERNS))]
    levels = xp.concatenate((xp.zeros_like(h[:, :1]),
                              xp.cumsum(signs*h, axis=1)), axis=1)
    return levels - xp.min(levels, axis=1, keepdims=True)


def expected_lip_directions(heights):
    """Signed true lip heights for the teacher and independent examiner."""
    return heights[:,1:] - heights[:,:-1]


def course_edge_positions(xp, yaw, length=ACADEMY_INTERVAL_M):
    """Four sharp lip locations along the physical course heading."""
    xs=ACADEMY_EDGE_X_M+xp.arange(ACADEMY_LIPS)*length
    return xp.stack((xp.cos(yaw)[:,None]*xs[None,:],
                     xp.sin(yaw)[:,None]*xs[None,:]),axis=-1)
