"""v5022 forward ToF geometry spike (NOT wired into the actor).

Three narrow, body-mounted, forward/downward beams. The *forward sensor*
returns the distance to the first visible physical surface, not privileged
terrain height or a perfect step classification. A beam can hit a vertical
riser, a horizontal platform, or nothing.

This small pure NumPy/JAX-compatible analytic box probe is a controlled
geometry reference. It DOES NOT establish MJX collision support, visibility
parity, sensor timing, or policy readiness. The production path must compare
these same rays with actual MJX scene geometry before connecting PPO.
"""
from __future__ import annotations

# Body X forward, Y left, Z up; provisional measured extrinsics.
# A second ToF fan complements, and does not replace, the v5016 down tripod.
FORWARD_TOF_MOUNTS_BODY_M = (
    (0.045, -0.047, -0.018),
    (0.045,  0.000, -0.018),
    (0.045,  0.047, -0.018),
)
FORWARD_TOF_BEAMS_BODY = (
    (0.70, -0.12, -0.70),
    (0.71,  0.00, -0.71),
    (0.70,  0.12, -0.70),
)
FORWARD_TOF_MIN_M = 0.035
FORWARD_TOF_MAX_M = 0.60


def forward_ray_geometry(xp, torso_world, body_to_world):
    """Return (origins, unit_directions), both [worlds,3 beams,xyz].

    The body rotation and position must be the *sensor mounting body* pose.
    Passing privileged contact heights or obstacle positions is unnecessary.
    """
    torso = xp.asarray(torso_world)
    rotation = xp.asarray(body_to_world)
    mounts = xp.asarray(FORWARD_TOF_MOUNTS_BODY_M)
    beams = xp.asarray(FORWARD_TOF_BEAMS_BODY)
    beams = beams / xp.sqrt(xp.sum(beams * beams, axis=-1, keepdims=True))
    origins = torso[:, None, :] + xp.einsum('wij,kj->wki', rotation, mounts)
    directions = xp.einsum('wij,kj->wki', rotation, beams)
    return origins, directions


def ray_box_distances(xp, origins, directions, box_min, box_max):
    """Nearest ray/surface distance in a union of solid axis-aligned boxes.

    Inputs: origins/directions [W,K,3]; box_min/max [B,3] or [W,B,3].
    Output: [W,K] distances; +inf means no intersection.

    Slab intersection includes all six faces, including vertical risers and
    the exposed edge of a lower platform. Parallel ray axes are treated
    explicitly rather than divided by zero. This model also permits body
    occlusion tests by including a physical blocking box.

    Box geometry is available ONLY to the sensor forward model / examiner;
    nothing here produces a privileged actor terrain-height label.
    """
    o = xp.asarray(origins)[:, :, None, :]
    d = xp.asarray(directions)[:, :, None, :]
    lo = xp.asarray(box_min)
    hi = xp.asarray(box_max)
    if lo.ndim == 2:
        lo = lo[None, :, :]
        hi = hi[None, :, :]
    lo = lo[:, None, :, :]
    hi = hi[:, None, :, :]
    parallel = xp.abs(d) <= 1e-8
    safe_d = xp.where(parallel, 1.0, d)
    t0 = (lo - o) / safe_d
    t1 = (hi - o) / safe_d
    enter = xp.max(xp.where(parallel, -xp.inf, xp.minimum(t0, t1)), axis=-1)
    leave = xp.min(xp.where(parallel, xp.inf, xp.maximum(t0, t1)), axis=-1)
    within_parallel = xp.all(
        (~parallel) | ((o >= lo) & (o <= hi)), axis=-1)
    meets = within_parallel & (leave >= xp.maximum(enter, 0.0)) & (leave >= 0.0)
    # A ray starting in a solid object meets the outgoing face instead.
    first = xp.where(enter >= 0.0, enter, leave)
    return xp.min(xp.where(meets, first, xp.inf), axis=-1)


def scan_box_course(xp, torso_world, body_to_world, box_min, box_max,
                    *, range_noise_m=None, dropout=None,
                    min_range_m=FORWARD_TOF_MIN_M,
                    max_range_m=FORWARD_TOF_MAX_M):
    """Return (three measured ranges [W,3], three validity bits [W,3]).

    Noise and dropped samples are explicit inputs so JAX callers can supply
    independent PRNG streams. Zero range always means invalid, never level
    ground. Timing, stale-sample age and confidence require a later async
    sensor queue; this function is only the instantaneous optical geometry.
    """
    origins, directions = forward_ray_geometry(xp, torso_world, body_to_world)
    distances = ray_box_distances(xp, origins, directions, box_min, box_max)
    if range_noise_m is not None:
        distances = distances + xp.asarray(range_noise_m)
    valid = ((distances >= min_range_m) & (distances <= max_range_m)
             & xp.isfinite(distances))
    if dropout is not None:
        valid = valid & ~xp.asarray(dropout, dtype=bool)
    return xp.where(valid, distances, 0.0), valid
