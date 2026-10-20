"""Terrain-relative *teacher truth*, not deployable robot perception.

P1 starts with a single stationary, horizontal-topped box under MuJoCo/MJX
collision. This is an explicit bounded geometry contract, not an invented
heightmap. Geometry and genuine foot/geom contact pairs are checked together:
a sphere pushing on a vertical riser is NOT credited as a supported foot on
the top. Near-contact pairs alone are still NOT measured normal force.

Only opt-in physics/reward evaluation may call this code. Never include its
output in the 220/232 input vectors; forward ToF/flow needs its own noisy,
delayed, occluded physical sensor model.
"""
from __future__ import annotations

from .directional import contact_pair_mask

# Conservative box-top geometric classification tolerances. A sphere centre
# whose vertical clearance is below its radius by >4mm is NOT on top.
TOP_PENETRATION_TOLERANCE_M = 0.004
TOP_SEPARATION_TOLERANCE_M = 0.010
BOX_FOOTPRINT_TOLERANCE_M = 0.003


def box_surface_truth(
    xp, *,
    foot_xyz, previous_foot_xyz, torso_xyz,
    floor_foot_signed, floor_torso_signed, floor_normal,
    box_center, box_rotation, box_half_size,
    contact_geom1, contact_geom2, contact_distance,
    floor_geom_id: int, box_geom_id: int, foot_geom_ids,
    foot_radius_m: float, control_dt: float,
):
    """Classify observed sphere/plane/box geometry and actual contact pairs.

    Arrays have leading world dimension N. Foot geometry is [N,2,3];
    floor_foot_signed [N,2], floor_torso_signed [N], normals [N,3],
    rotation [N,3,3], contact arrays [N,C]. Supports yawed static boxes,
    with a near-vertical top normal; the caller must reject tilting boxes.

    Signed clearance is *ball bottom* above the underlying support
    surface. A foot in the XY projection of the box but BELOW its top
    has negative raw clearance (reward later clips to zero); the
    vertical-riser collision is not counted as a load-bearing top.

    This function assumes the passed foot points are sphere geom centres,
    the box centre is a true MuJoCo geom pose, and real collision pairs
    refer to the two exact foot geoms. The support flag is a conservative
    contact + geometry candidate, not measured impulse or normal force.
    """
    if not (foot_radius_m > 0. and control_dt > 0.):
        raise ValueError('foot radius and control step must be positive')
    # World vector expressed in the box's rotated local XYZ basis:
    # coordinates = R^T(world - centre).
    delta = foot_xyz - box_center[:, None, :]
    local = xp.einsum('nfi,nij->nfj', delta, box_rotation)
    half = xp.asarray(box_half_size)
    footprint = (xp.abs(local[..., 0]) <= half[0] + BOX_FOOTPRINT_TOLERANCE_M) & (
        xp.abs(local[..., 1]) <= half[1] + BOX_FOOTPRINT_TOLERANCE_M)
    top_normal = box_rotation[:, :, 2]
    box_top_centre_distance = local[..., 2] - half[2]
    top_gap = box_top_centre_distance - foot_radius_m

    floor_pairs = contact_pair_mask(
        xp, contact_geom1, contact_geom2, contact_distance,
        floor_geom_id, foot_geom_ids)
    box_pairs = contact_pair_mask(
        xp, contact_geom1, contact_geom2, contact_distance,
        box_geom_id, foot_geom_ids)
    # This explicitly prevents side/riser collision from masquerading as
    # top support. Tall foot centre alone earns nothing without a real
    # box/ball contact pair. An edge contact is conservatively excluded
    # when the sphere centre lies outside the actual projected top area.
    on_top = (footprint & box_pairs &
              (top_gap >= -TOP_PENETRATION_TOLERANCE_M) &
              (top_gap <= TOP_SEPARATION_TOLERANCE_M))
    riser_or_side = box_pairs & ~on_top
    supported = floor_pairs | on_top

    foot_surface_signed = xp.where(
        footprint, box_top_centre_distance, floor_foot_signed)
    toe_clearance = foot_surface_signed - foot_radius_m
    torso_delta = torso_xyz - box_center
    torso_local = xp.einsum('ni,nij->nj', torso_delta, box_rotation)
    torso_above_box = ((xp.abs(torso_local[:, 0]) <= half[0]) &
                       (xp.abs(torso_local[:, 1]) <= half[1]))
    torso_signed = xp.where(
        torso_above_box, torso_local[:, 2] - half[2], floor_torso_signed)

    # Tangential slip uses each genuine supported contact's own surface
    # normal and full 3-D velocity: a moving foot dragging down a riser
    # does not count as stance. Collision pair must be present.
    normal = xp.where(on_top[..., None],
                      top_normal[:, None, :], floor_normal[:, None, :])
    foot_velocity = (foot_xyz - previous_foot_xyz) / control_dt
    normal_speed = xp.sum(foot_velocity * normal, axis=-1)
    tangential = foot_velocity - normal_speed[..., None] * normal
    tangent_speed = xp.linalg.norm(tangential, axis=-1)
    weights = supported.astype(foot_xyz.dtype)
    slip = xp.sum(weights * tangent_speed, axis=1) / xp.maximum(
        1., xp.sum(weights, axis=1))

    return {
        'torso_height': torso_signed,
        'foot_center_signed': foot_surface_signed,
        'toe_clearance_signed': toe_clearance,
        'foot_top_contact_candidate': on_top,
        'foot_floor_contact_candidate': floor_pairs,
        'foot_riser_contact': riser_or_side,
        'foot_supported_candidate': supported,
        'foot_in_box_footprint': footprint,
        'support_fraction': xp.mean(weights, axis=1),
        'stance_slip': slip,
        'swing_clearance': xp.maximum(
            0., xp.max(toe_clearance, axis=1)),
        # Signed minimum is deliberately preserved for riser auditing,
        # but it MUST NOT be interpreted as MuJoCo solver penetration.
        'min_toe_clearance_signed': xp.min(toe_clearance, axis=1),
    }
