"""v5019 forward ToF: two virtual 3-zone modules, 5/10/15cm footprints per lane.

This module owns the sensor *forward model* and optional diagnostic inverse, not
training rewards, terrain curriculum, or oracle actor features. It treats a
ray as a calibrated narrow beam; two single-point ToF chips could NOT deliver
these six simultaneous zones. The eventual physical hardware must be two
multi-zone modules (or an explicitly timed scan) with measured extrinsics.

The first-hit model handles a real MuJoCo plane and a rotated collidable box,
including the near riser. MJX does not have a suitable general raycast for
this vectorized training path; no collision support is inferred from hfields.
Robot self-occlusion, surface reflectance and dynamic movers remain unsupported.
The deployable 280-input actor receives RAW range + validity, not an oracle
height or a ground-plane-dependent cooked height. The inverse below is only
for optional diagnosis; it is never used for the actor packet.
"""
from __future__ import annotations

import math

FORWARD_LANES_Y_M = (-.05, .05)              # right, left
FORWARD_LOOKAHEAD_X_M = (.05, .10, .15)      # footprint from torso origin
FORWARD_MOUNT_X_M = .035
FORWARD_MOUNT_Z_M = -.025
FORWARD_NOMINAL_DEPTH_M = .2334
FORWARD_MIN_RANGE_M = .035
FORWARD_MAX_RANGE_M = .60
FORWARD_NOISE_STD_M = .0015
FORWARD_QUANTIZATION_M = .001  # provisional 1mm measurement resolution
FORWARD_PROFILE_CLIP_M = .15
FORWARD_MODEL_REVISION = "two_three_zone_raw_slant_ranges_plane_box_v2"

# Explicit six-channel lane-major order; document geometry in the manifest.
FORWARD_MOUNTS_BODY_M = tuple(
    (FORWARD_MOUNT_X_M, side, FORWARD_MOUNT_Z_M)
    for side in FORWARD_LANES_Y_M for x in FORWARD_LOOKAHEAD_X_M
)
FORWARD_DIRECTIONS_BODY = tuple(
    tuple(v / math.sqrt(dx * dx + 1) for v in (dx, 0., -1.))
    for _side in FORWARD_LANES_Y_M
    for target in FORWARD_LOOKAHEAD_X_M
    for dx in ((target - FORWARD_MOUNT_X_M) / FORWARD_NOMINAL_DEPTH_M,)
)


def first_hit_ranges(xp, torso_xyz, body_rotation, floor_origin, floor_normal,
                     *, box_center=None, box_rotation=None, box_half_size=None,
                     errors_m=None):
    """Return six noisy first-hit lengths and validity, shape [worlds,6].

    Ground truth is permitted ONLY to synthesize range returns. An optional
    MuJoCo box intersects actual 3D rays (top, front or side) before the floor.
    No actual floor normal, root pose, or box height is returned to the actor.
    """
    mounts = xp.asarray(FORWARD_MOUNTS_BODY_M)
    beams = xp.asarray(FORWARD_DIRECTIONS_BODY)
    origins = torso_xyz[:, None, :] + xp.einsum('nij,kj->nki', body_rotation, mounts)
    dirs = xp.einsum('nij,kj->nki', body_rotation, beams)
    normal = floor_normal
    denom = xp.sum(dirs * normal[:, None, :], axis=-1)
    numer = xp.sum((floor_origin[:, None, :] - origins) * normal[:, None, :], axis=-1)
    plane_t = numer / xp.where(xp.abs(denom) > 1e-7, denom, 1.)
    plane_t = xp.where((denom < -.15) & (plane_t > 0.), plane_t, xp.inf)

    if box_center is not None:
        if box_rotation is None or box_half_size is None:
            raise ValueError('box rotation and half-size must accompany box centre')
        rel = origins - box_center[:, None, :]
        box_inverse = xp.swapaxes(box_rotation, 1, 2)
        q = xp.einsum('nij,nkj->nki', box_inverse, rel)
        d = xp.einsum('nij,nkj->nki', box_inverse, dirs)
        half = xp.asarray(box_half_size)
        # Slab ray/AABB first entry, finite interval in each of x/y/z.
        # Nearly parallel outside = no hit, not NaN / false side collision.
        parallel = xp.abs(d) < 1e-7
        outside_parallel = parallel & (xp.abs(q) > half + 1e-7)
        inv_d = 1. / xp.where(parallel, 1., d)
        lo = (-half - q) * inv_d
        hi = (half - q) * inv_d
        near = xp.where(parallel, -xp.inf, xp.minimum(lo, hi))
        far = xp.where(parallel, xp.inf, xp.maximum(lo, hi))
        entry = xp.max(near, axis=-1)
        exit_ = xp.min(far, axis=-1)
        valid = ((~xp.any(outside_parallel, axis=-1)) &
                 (entry <= exit_) & (entry > 0.) & (exit_ > 0.))
        box_t = xp.where(valid, entry, xp.inf)
        plane_t = xp.minimum(plane_t, box_t)

    ranges = plane_t if errors_m is None else plane_t + xp.asarray(errors_m)
    valid = (xp.isfinite(ranges) & (ranges >= FORWARD_MIN_RANGE_M) &
             (ranges <= FORWARD_MAX_RANGE_M))
    return xp.where(valid, ranges, 0.), valid


def relative_height_profile(xp, six_ranges, six_valid, tripod_ranges,
                            ground_normal_body, tripod_valid):
    """Invert six delayed rays into terrain height ABOVE measured tripod plane.

    Inverse receives only range lengths, validity, calibrated beam geometry,
    and tripod's estimated normal. Simulator floor origin and box geometry
    are NEVER passed into this function. Unknown returns zero height AND
    zero validity; a missing wall/occluded ray must not masquerade as flat.
    """
    mounts = xp.asarray(FORWARD_MOUNTS_BODY_M)
    beams = xp.asarray(FORWARD_DIRECTIONS_BODY)
    from .directional import TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY
    tripod_mounts = xp.asarray(TOF_TRIPOD_MOUNTS_BODY_M)
    tripod_beams = xp.asarray(TOF_TRIPOD_BEAMS_BODY)
    ref = tripod_mounts[0][None, :] + tripod_ranges[:, 0, None] * tripod_beams[0][None, :]
    hits = mounts[None, :, :] + six_ranges[:, :, None] * beams[None, :, :]
    signed = xp.sum((hits - ref[:, None, :]) * ground_normal_body[:, None, :], axis=-1)
    valid = six_valid & tripod_valid[:, None] & xp.isfinite(signed)
    heights = xp.where(valid, xp.clip(signed, -FORWARD_PROFILE_CLIP_M, FORWARD_PROFILE_CLIP_M), 0.)
    # This is validity, not a calibrated probability that a return is true.
    return heights, valid.astype(heights.dtype)
