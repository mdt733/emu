"""A recyclable three-platform course for the existing STEP specialist.

Three fixed MJX collision geoms rotate through current/next/ahead roles.
No height, direction or world index is injected into actor observations.
"""
from __future__ import annotations

ACADEMY_PLATFORMS = 3
ACADEMY_INTERVAL_M = 0.50
ACADEMY_EDGE_X_M = ACADEMY_INTERVAL_M / 2
ACADEMY_MAX_EDGE_YAW_DEG = 6.0
ACADEMY_PLATFORM_HALF_LENGTH_M = ACADEMY_INTERVAL_M / 2
ACADEMY_PLATFORM_HALF_WIDTH_M = 16.0
ACADEMY_PLATFORM_HALF_DEPTH_M = 0.25
ACADEMY_PLATFORM_NAMES = tuple(f"academy_platform_{i}" for i in range(ACADEMY_PLATFORMS))
ACADEMY_PLATFORM_GEOMS = tuple(f"academy_surface_{i}" for i in range(ACADEMY_PLATFORMS))
# Beyond normal locomotion geometry, never overlap disabled long floor slabs.
ACADEMY_BURIED_FLOOR_Z_M = -20.0


def signed_lip_height(xp, direction, sample, up_frontier_m, down_frontier_m,
                      progress=0.0):
    """Random sign, independent ascent/descent ability, adaptive episode challenge.

    0%, 50%, 100% through the episode -> 0.5x, 1.0x, 1.5x frontier.
    Small independent 0.85..1.15 jitter prevents a repeated exact height.
    """
    frontier = xp.where(direction > 0, up_frontier_m, down_frontier_m)
    scale = 0.5 + xp.clip(progress, 0.0, 1.0)
    height = xp.maximum(0.0015, frontier * scale * (0.85 + 0.30 * sample))
    return xp.where(direction > 0, height, -height)


def initial_heights(xp, signed_lips):
    """Cumulative real heights for the initial three plates (two transitions)."""
    return xp.concatenate((xp.zeros_like(signed_lips[:, :1]),
                           xp.cumsum(signed_lips, axis=1)), axis=1)


def platform_centers(xp, indices, yaw):
    """World-space centres along one randomly yawed course, no pattern repeat."""
    x = (indices * ACADEMY_INTERVAL_M)[:, :] if indices.ndim == 2 else indices * ACADEMY_INTERVAL_M
    return xp.stack((x * xp.cos(yaw)[:, None],
                     x * xp.sin(yaw)[:, None]), axis=-1)
