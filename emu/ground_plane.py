"""v5016 experimental downward-ranging ground-plane estimator.

Read-only hardware-shaped geometry. Three valid rays determine one local plane
but CANNOT detect which of the three returns is wrong; four or more hits are
required for meaningful outlier rejection. IMU gravity is in the same body frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GroundPlane:
    valid: bool
    reason: str
    ground_normal_body: tuple[float, float, float] | None = None
    slope_deg: float | None = None
    uphill_body: tuple[float, float, float] | None = None
    triangle_area_m2: float = 0.0


def estimate_ground_plane(
    origins_body, directions_body, ranges_m, gravity_body,
    *, min_area_m2: float = 0.0001, max_range_m: float = 0.6,
) -> GroundPlane:
    """Intersect three calibrated range beams with the ground and fit a plane.

    Normal points *away* from the ground relative to measured gravity.
    The uphill vector lies within the estimated ground plane. Neither a
    chassis pitch change on level terrain nor yaw should masquerade as a
    ground slope. Invalid samples are explicitly rejected; never fill them
    with a plausible zero-degree measurement.
    """
    origin = np.asarray(origins_body, dtype=np.float64)
    direction = np.asarray(directions_body, dtype=np.float64)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    gravity = np.asarray(gravity_body, dtype=np.float64)
    if origin.shape != (3, 3) or direction.shape != (3, 3) or ranges.shape != (3,) or gravity.shape != (3,):
        raise ValueError('expected three origins, three beams, three ranges and one gravity vector')
    if not (np.isfinite(origin).all() and np.isfinite(direction).all() and np.isfinite(gravity).all()):
        return GroundPlane(False, 'nonfinite ray geometry/gravity')
    if not np.isfinite(ranges).all() or np.any(ranges <= 0) or np.any(ranges > max_range_m):
        return GroundPlane(False, 'missing, nonpositive or out-of-range return')
    ray_norm = np.linalg.norm(direction, axis=1)
    g_norm = np.linalg.norm(gravity)
    if np.any(ray_norm < 1e-8) or g_norm < 1e-8:
        return GroundPlane(False, 'degenerate ray/gravity')
    ray_direction = direction / ray_norm[:, None]
    points = origin + ranges[:, None] * ray_direction
    n = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(n)
    area = 0.5 * length
    if length < 1e-10 or area < min_area_m2:
        return GroundPlane(False, 'nearly collinear ground returns', triangle_area_m2=float(area))
    n = n / length
    g = gravity / g_norm
    if np.dot(n, -g) < 0:
        n = -n
    cosine = float(np.dot(n, -g))
    if cosine < 0.1:
        return GroundPlane(False, 'ground normal nearly perpendicular to gravity', triangle_area_m2=float(area))
    tangential_down = g - np.dot(g, n) * n
    tangential_norm = float(np.linalg.norm(tangential_down))
    degrees = math.degrees(math.atan2(tangential_norm, cosine))
    uphill = (-tangential_down / tangential_norm) if tangential_norm > 1e-7 else None
    return GroundPlane(
        valid=True,
        reason='three-ray geometric fit only; no outlier redundancy',
        ground_normal_body=tuple(float(x) for x in n),
        slope_deg=degrees,
        uphill_body=tuple(float(x) for x in uphill) if uphill is not None else None,
        triangle_area_m2=float(area),
    )
