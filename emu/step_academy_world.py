"""Physical, fixed-topology four-lip course dimensions and height profiles.

The actor does not see these values: only actual ray returns and proprioception.
Extra colliders are ordinary MuJoCo geom boxes, not visual-only geometry.
The first/final slabs are reused from the existing two-slab environment;
three short intermediate platforms complete four successive sharp lips.
"""
from __future__ import annotations

ACADEMY_LIPS = 4
ACADEMY_INTERVAL_M = 0.35  # ~1.30 m final lip; traversable in standard 14 s bouts
ACADEMY_EDGE_X_M = 0.25
ACADEMY_MIDDLE_HALF_LENGTH_M = ACADEMY_INTERVAL_M / 2
ACADEMY_MIDDLE_HALF_WIDTH_M = 16.0
ACADEMY_MIDDLE_HALF_DEPTH_M = 0.25
ACADEMY_MIDDLE_NAMES = tuple(f"academy_platform_{i}" for i in range(3))
ACADEMY_MIDDLE_GEOMS = tuple(f"academy_surface_{i}" for i in range(3))


def course_heights(xp, index, samples, difficulty_m):
    """Five contiguous plateau heights giving four bona fide signed lips.

    index is [worlds], samples is [worlds,4] uniform(0,1), difficulty_m
    is a scalar/array with a strict per-lip upper bound. Four balanced
    patterns: UP-UP-DOWN-DOWN, UP-DOWN-UP-DOWN, DOWN-UP-DOWN-UP and
    DOWN-DOWN-UP-UP. No clipping of cumulative heights (which could
    silently erase a required step). All plateau heights >= 0.
    """
    # Never silently ignore a legitimate nursery retreat below 3mm.
    # Each real collision lip stays >=1.5mm so the signed crossing/contact
    # threshold can still distinguish it from numerical zero-height noise.
    h=xp.maximum(.0015, difficulty_m * (.55 + .45*samples))
    a,b,c,d=[h[:,i] for i in range(4)]
    z=xp.zeros_like(a)
    # Each transition is nonzero. Cumulative rise is limited to 2*difficulty.
    profiles=xp.stack((
        xp.stack((z,a,a+b,b,z),axis=1),
        xp.stack((z,a,z,c,z),axis=1),
        xp.stack((a,z,b,z,c),axis=1),
        xp.stack((a+b,b,z,c,c+d),axis=1),
    ),axis=1)
    route=xp.mod(index.astype(xp.int32),4)
    return xp.take_along_axis(profiles,route[:,None,None],axis=1)[:,0,:]


def expected_lip_directions(heights):
    """Signed height differences per course; teacher/evaluator only."""
    return heights[:,1:]-heights[:,:-1]


def course_edge_positions(xp, yaw, length=ACADEMY_INTERVAL_M):
    """Four world-space sharp edge points at configured 0.35m spacing."""
    xs=ACADEMY_EDGE_X_M+xp.arange(ACADEMY_LIPS)*length
    return xp.stack((xp.cos(yaw)[:,None]*xs[None,:],
                     xp.sin(yaw)[:,None]*xs[None,:]),axis=-1)
