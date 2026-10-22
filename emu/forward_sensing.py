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

# Body X forward, Y left, Z up; provisional extrinsics: X=90 mm sits beyond the actual torso front face (76 mm).
# A second ToF fan complements, and does not replace, the v5016 down tripod.
FORWARD_TOF_MOUNTS_BODY_M = (
    (0.090, -0.060, -0.018),
    (0.090,  0.000, -0.018),
    (0.090,  0.060, -0.018),
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


# Candidate production hardware package: ONE front-facing 4x4 multizone ToF
# (e.g. VL53L5CX), mounted ahead of the torso. The 3-beam arrangement above
# is a comparison prototype, NOT the currently selected hardware layout.
# A single device shares ONE optical centre across all 16 zones.
# This 45x45 degree FOV approximation samples five rays within each zone;
# a real histogram sensor may return several targets, reject edges, or report
# a dominant return. Do not claim the simulated median is hardware truth.
MULTIZONE_MOUNT_BODY_M = (0.090, 0.0, -0.018)
MULTIZONE_DOWN_TILT_DEG = 50.0
MULTIZONE_CENTRES_DEG = (-16.875, -5.625, 5.625, 16.875)
MULTIZONE_SUBRAY_OFFSETS_DEG = ((0., 0.), (-2.8, 0.), (2.8, 0.),
                                (0., -2.8), (0., 2.8))


def multizone_ray_geometry(xp, torso_world, body_to_world):
    """16 physical zones × 5 sub-rays from one common front module.

    Result: origins/directions both [W, 16, 5, 3]. The optical centre
    follows the *actual* body pose and every zone shares that centre.
    A 4x4 sensor's measured range is not four rows of exact ground truth.
    """
    from math import radians
    mount = xp.asarray(MULTIZONE_MOUNT_BODY_M)
    R = xp.asarray(body_to_world)
    torso = xp.asarray(torso_world)
    angles = xp.asarray(MULTIZONE_CENTRES_DEG)
    delta = xp.asarray(MULTIZONE_SUBRAY_OFFSETS_DEG)
    azimuth = angles[None, :, None] + delta[None, None, :, 0]
    elevation = (MULTIZONE_DOWN_TILT_DEG +
                 angles[:, None, None] + delta[None, :, 1][None, ...])
    # Layout [4 vertical rows, 4 horizontal columns, 5 subrays].
    az = xp.broadcast_to(azimuth, (4, 4, 5))
    el = xp.broadcast_to(elevation, (4, 4, 5))
    x = xp.cos(xp.deg2rad(el)) * xp.cos(xp.deg2rad(az))
    y = xp.cos(xp.deg2rad(el)) * xp.sin(xp.deg2rad(az))
    z = -xp.sin(xp.deg2rad(el))
    dirs_body = xp.stack((x, y, z), axis=-1).reshape(16, 5, 3)
    dirs_world = xp.einsum("wij,ksj->wksi", R, dirs_body)
    origin = torso + xp.einsum("wij,j->wi", R, mount)
    return xp.broadcast_to(origin[:, None, None, :], dirs_world.shape), dirs_world


def scan_multizone_box_course(xp, torso_world, body_to_world,
                              box_min, box_max, *,
                              noise_m=None, dropout=None,
                              min_range_m=0.035, max_range_m=0.60):
    """Controlled spatial sensor model, not the deployable observation yet.

    Returns range/valid/visibility confidence arrays [W,4,4].
    A zone reports the median of >=3 valid sub-ray returns, not a secretly
    exact height. Broadly mixed depths reduce confidence. A missing range
    is explicitly invalid. Only the measurement simulator reads box truth;
    PPO must eventually receive just ranges, validity and freshness.
    """
    origins, dirs = multizone_ray_geometry(xp, torso_world, body_to_world)
    n = xp.asarray(torso_world).shape[0]
    rays = ray_box_distances(
        xp, origins.reshape(n, 80, 3), dirs.reshape(n, 80, 3),
        box_min, box_max).reshape(n, 16, 5)
    return reduce_multizone_measurements(
        xp, rays, noise_m=noise_m, dropout=dropout,
        min_range_m=min_range_m, max_range_m=max_range_m)



def ray_oriented_box_distances(xp, origins, directions,
                               box_centres_world, box_rotations_world,
                               box_half_sizes):
    """Intersect real oriented support boxes, retaining batch-world geometry.

    origins, directions: [W,R,3]
    box_centres_world: [B,3] or [W,B,3]
    box_rotations_world: [B,3,3] or [W,B,3,3] (local-to-world)
    box_half_sizes: [B,3] or [W,B,3]
    returns [W,R] along the normalized ray, +inf if none.

    Read these inputs from the actual MJX contact geometry / geom_xmat.
    Unlike a second analytic slope oracle, this follows the SAME solid
    support boxes that produce the robot's collisions. The sensor actor
    must only receive delayed noisy ranges + validity, never geom ids.
    """
    origins, directions = xp.asarray(origins), xp.asarray(directions)
    centres = xp.asarray(box_centres_world)
    rotation = xp.asarray(box_rotations_world)
    sizes = xp.asarray(box_half_sizes)
    if centres.ndim == 2:
        centres = centres[None]
    if rotation.ndim == 3:
        rotation = rotation[None]
    if sizes.ndim == 2:
        sizes = sizes[None]
    # R[j,i] takes local coordinate i to world coordinate j.
    # Its transpose maps the world ray into each actual box frame.
    delta = origins[:, :, None, :] - centres[:, None, :, :]
    local_origins = xp.einsum('wbji,wrbj->wrbi', rotation, delta)
    local_directions = xp.einsum('wbji,wrj->wrbi', rotation, directions)
    half = sizes[:, None, :, :]
    parallel = xp.abs(local_directions) <= 1e-8
    safe_d = xp.where(parallel, 1., local_directions)
    a = (-half - local_origins) / safe_d
    b = (half - local_origins) / safe_d
    entering = xp.max(xp.where(parallel, -xp.inf, xp.minimum(a, b)), axis=-1)
    exiting = xp.min(xp.where(parallel, xp.inf, xp.maximum(a, b)), axis=-1)
    parallel_ok = xp.all(
        (~parallel) | (xp.abs(local_origins) <= half), axis=-1)
    hit = parallel_ok & (exiting >= xp.maximum(entering, 0.)) & (exiting >= 0.)
    distance = xp.where(entering >= 0., entering, exiting)
    return xp.min(xp.where(hit, distance, xp.inf), axis=-1)



def reduce_multizone_measurements(xp, rays, *, noise_m=None, dropout=None,
                                  min_range_m=0.035, max_range_m=0.60):
    """Convert simulated [W,16,5] raw ray hits into one physical-zone packet.

    Approximate a multi-target optical receiver, never exact terrain heights.
    Sensor acquisition clock and delay still belong to the future queue.
    """
    n = xp.asarray(rays).shape[0]
    inside = xp.isfinite(rays) & (rays >= min_range_m) & (rays <= max_range_m)
    valid_count = xp.sum(inside, axis=-1)
    sorted_ranges = xp.sort(xp.where(inside, rays, xp.inf), axis=-1)
    mid_index = xp.maximum(0, (valid_count - 1) // 2).astype(int)
    median = xp.take_along_axis(
        sorted_ranges, mid_index[..., None], axis=-1)[..., 0]
    lowest = sorted_ranges[..., 0]
    highest_index = xp.maximum(0, valid_count - 1).astype(int)
    highest = xp.take_along_axis(
        sorted_ranges, highest_index[..., None], axis=-1)[..., 0]
    spread = xp.where(valid_count > 0, highest - lowest, 0.0)
    valid = valid_count >= 3
    if noise_m is not None:
        median = median + xp.asarray(noise_m).reshape(n, 16)
    valid = valid & xp.isfinite(median) & (
        median >= min_range_m) & (median <= max_range_m)
    if dropout is not None:
        valid = valid & ~xp.asarray(dropout, dtype=bool).reshape(n, 16)
    # Report VALIDITY separately; confidence is conservative measurement
    # quality/visibility, never a ground-truth probability of safe walking.
    confidence = xp.where(
        valid, (valid_count / 5.0) * xp.clip(1 - spread / 0.07, 0., 1.), 0.)
    return dict(
        ranges_m=xp.where(valid, median, 0.).reshape(n, 4, 4),
        valid=valid.reshape(n, 4, 4),
        visibility_confidence=confidence.reshape(n, 4, 4),
    )



def scan_multizone_oriented_boxes(xp, torso_world, body_to_world,
                                 box_centres_world, box_rotations_world,
                                 box_half_sizes, *, noise_m=None,
                                 dropout=None, min_range_m=0.035,
                                 max_range_m=0.60):
    """One front-mounted 4x4 sensor against the ACTUAL rotated solid support.

    This is the terrain-contact-aligned forward measurement simulator. The
    geometry comes from MJX geom_xpos / geom_xmat / geom_size, never directly
    from teacher course labels. Caller must handle samples/latency/dropout.
    """
    origin, direction = multizone_ray_geometry(xp, torso_world, body_to_world)
    n = xp.asarray(torso_world).shape[0]
    rays = ray_oriented_box_distances(
        xp, origin.reshape(n,80,3), direction.reshape(n,80,3),
        box_centres_world, box_rotations_world, box_half_sizes).reshape(n,16,5)
    return reduce_multizone_measurements(
        xp, rays, noise_m=noise_m, dropout=dropout,
        min_range_m=min_range_m, max_range_m=max_range_m)
