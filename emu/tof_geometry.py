"""Single physical VL53L8CX 4x4 geometry contract.

45° x 45° FoV is an image-plane frustum, not a separable pitch/yaw fan.
One centre ray per zone approximates the 4x4 sensor's real multizone
histogram return (NOT a complete optical or multipath simulation).
All directions are BODY-FRAME unit vectors; the chassis orientation maps
them into world space. ST 4x4 zone order is row-major in our actor's view.
"""
from __future__ import annotations
import math

TOF_FOV_DEG = 45.0
TOF_TILT_DOWN_DEG = 70.0
TOF_YAW_DEG = 0.0
TOF_MOUNT_BODY_M = (0.0, 0.0, -0.030)
TOF_ZONE_SIDE = 4
# ST UM3109 §2.2: the receiving lens flips BOTH physical image axes.
# SPAD zone 0 is at array bottom-left but observes scene TOP-RIGHT;
# therefore, for the nominal package orientation shown in the datasheet,
# our actor's scene row-major [top-left..bottom-right] maps to raw IDs:
# [3,2,1,0; 7,6,5,4; 11,10,9,8; 15,14,13,12].
# A differently rotated sensor PCB requires a measured remap; do NOT
# silently treat ascending hardware indices as scene row-major.
TOF_HARDWARE_ZONE_IDS = tuple(4*row+(3-col)
                              for row in range(4) for col in range(4))

def tof4x4_rays_body() -> tuple[tuple[float, float, float], ...]:
    """Match the browser's physical square-frustum zone CENTRE directions."""
    t=math.radians(TOF_TILT_DOWN_DEG)
    c,s=math.cos(t),math.sin(t)
    half=math.tan(math.radians(TOF_FOV_DEG)/2)
    out=[]
    for row in range(TOF_ZONE_SIDE):
        for col in range(TOF_ZONE_SIDE):
            u=(col-1.5)*half/2
            v=(1.5-row)*half/2
            x,y,z=c+s*v,u,-s+c*v
            inv=1.0/math.sqrt(x*x+y*y+z*z)
            out.append((x*inv,y*inv,z*inv))
    return tuple(out)

TOF_RAYS_BODY = tof4x4_rays_body()

def returns_relative_to_feet(xp, ranges_m, valid, gravity_body, feet_body_m,
                             *, foot_radius_m=0.018):
    """Project measured 4x4 ToF ENDPOINTS relative to EACH possible stance foot.

    Shapes: ranges/valid [world,16], gravity [world,3], foot centres
    [world,2,3], output XYZ-like scalars [world,2,16].

    Inputs are deployable: delayed raw ToF, IMU gravity, joint-derived foot
    positions. No MJX mesh height, exact step, contact flag or target leaks in.

    HEIGHT IS THE RAY-HIT POINT, NOT GUARANTEED WALKABLE GROUND. A ray can
    hit a vertical riser. A separate footprint/support estimator must decide
    when a foot really anchors a local ground-height reference. In particular,
    shin spring deflection ALONE is not proof of stance. Return both possible
    references rather than silently guessing a loaded foot.

    Validity accompanies every output; invalid values are zero placeholders.
    A normalised gravity vector removes body pitch/roll. Camera mounting
    offset and the finite spherical-foot radius are included explicitly.
    No height is integrated across time: all coordinates are instantaneous.
    """
    d = xp.asarray(ranges_m)
    ok = xp.asarray(valid, dtype=xp.bool_)
    g = xp.asarray(gravity_body)
    feet = xp.asarray(feet_body_m)
    if d.ndim != 2 or d.shape[1] != 16:
        raise ValueError("ToF ranges must have shape [world,16]")
    if ok.shape != d.shape or g.shape != (d.shape[0], 3):
        raise ValueError("ToF validity/gravity shapes do not match worlds")
    if feet.shape != (d.shape[0], 2, 3):
        raise ValueError("Body-relative feet must have shape [world,2,3]")
    if not 0. < foot_radius_m < .1:
        raise ValueError("Expected physical ball-foot radius in metres")

    from .contracts import TERRAIN_TOF_MIN_M, TERRAIN_TOF_MAX_M
    up = -g / xp.maximum(xp.linalg.norm(g, axis=1, keepdims=True), 1e-7)
    gravity_valid = (xp.linalg.norm(g, axis=1) > .1) & xp.all(
        xp.isfinite(g), axis=1)
    ray_valid = (ok & xp.isfinite(d) &
                 (d >= TERRAIN_TOF_MIN_M) & (d <= TERRAIN_TOF_MAX_M))
    rays = xp.asarray(TOF_RAYS_BODY, dtype=d.dtype)
    mount = xp.asarray(TOF_MOUNT_BODY_M, dtype=d.dtype)
    hits = mount[None, None, :] + d[:, :, None] * rays[None, :, :]
    foot_bottom = feet - foot_radius_m * up[:, None, :]
    delta = hits[:, None, :, :] - foot_bottom[:, :, None, :]

    # Orthogonal robot-forward/side axes in the plane normal to gravity,
    # not chassis X/Y when the torso pitches into a climb.
    forward = xp.asarray((1., 0., 0.), dtype=d.dtype)[None, :]
    forward = forward - xp.sum(forward * up, axis=1, keepdims=True) * up
    forward = forward / xp.maximum(
        xp.linalg.norm(forward, axis=1, keepdims=True), 1e-7)
    sideways = xp.cross(up, forward)
    height = xp.sum(delta * up[:, None, None, :], axis=-1)
    ahead = xp.sum(delta * forward[:, None, None, :], axis=-1)
    lateral = xp.sum(delta * sideways[:, None, None, :], axis=-1)

    mask = (ray_valid[:, None, :] & gravity_valid[:, None, None] &
            xp.all(xp.isfinite(feet), axis=2)[:, :, None])
    return dict(
        return_height_over_foot_m=xp.where(mask, height, 0.),
        return_forward_from_foot_m=xp.where(mask, ahead, 0.),
        return_sideways_from_foot_m=xp.where(mask, lateral, 0.),
        return_valid=mask,
    )
