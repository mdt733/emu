"""Terrain-relative *teacher truth*, not deployable robot perception.

P1 starts with a single stationary, horizontal-topped box under MuJoCo/MJX
collision. This is an explicit bounded geometry contract, not an invented
heightmap. Geometry and genuine foot/geom contact pairs are checked together:
a sphere pushing on a vertical riser is NOT credited as a supported foot on
the top. Both contact normal and actual contact point must match the top
face; near-contact pairs alone are still NOT measured normal force.

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
    floor_foot_signed, floor_torso_signed, floor_normal, floor_origin,
    box_center, box_rotation, box_half_size,
    contact_geom1, contact_geom2, contact_distance,
    contact_frame, contact_position,
    floor_geom_id: int, box_geom_id: int, foot_geom_ids,
    foot_radius_m: float, control_dt: float,
):
    """Classify observed sphere/plane/box geometry and actual contact pairs.

    Arrays have leading world dimension N. Foot geometry is [N,2,3];
    floor_foot_signed [N,2], floor_torso_signed [N], floor normals
    and floor origins [N,3], rotation [N,3,3], contact arrays [N,C].
    Supports yawed static boxes,
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
    if (contact_position.shape != (*contact_geom1.shape, 3) or
        contact_frame.shape[0:2] != contact_geom1.shape or
        contact_frame.size != contact_geom1.size * 9):
        raise ValueError('contact position and 9-component contact frame shape must agree with true contact pairs')
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

    # Root issue 2: MJX may list nearby pairs even with POSITIVE
    # separation within its broad contact buffer. Under our declared
    # default zero-margin robot/box geom convention they are NOT exerting
    # normal support. Pair presence alone (the former 2mm tolerance)
    # must never become teacher stance credit. Signed distance <= 0 is
    # a prerequisite, still NOT proof of positive normal FORCE.
    touching_distance = xp.where(
        contact_distance <= 0., contact_distance, xp.inf)
    floor_pairs = contact_pair_mask(
        xp, contact_geom1, contact_geom2, touching_distance,
        floor_geom_id, foot_geom_ids)
    box_pairs = contact_pair_mask(
        xp, contact_geom1, contact_geom2, touching_distance,
        box_geom_id, foot_geom_ids)
    # CRITICAL: proximity and near-top sphere-centre Z are NOT a proof of
    # contact with the horizontal face. On a box edge a sphere can collide
    # with a riser while its centre already reaches the top height.
    # MuJoCo/MJX contact.frame's first row is the contact NORMAL in world
    # coordinates; contact.pos is the collision point midpoint. Require
    # a nearly vertical normal AND a point on the actual top plane.
    # Use abs(dot) because geom1/geom2 contact ordering reverses the normal.
    normals = xp.reshape(contact_frame, (
        foot_xyz.shape[0], contact_geom1.shape[1], 3, 3))[:, :, 0, :]
    contact_local = xp.einsum(
        'nci,nij->ncj', contact_position - box_center[:, None, :],
        box_rotation)
    normal_is_top = xp.abs(
        xp.sum(normals * top_normal[:, None, :], axis=-1)) >= .88
    top_surface_point = (
        (xp.abs(contact_local[:, :, 2] - half[2]) <= .004) &
        (xp.abs(contact_local[:, :, 0]) <= half[0] + .002) &
        (xp.abs(contact_local[:, :, 1]) <= half[1] + .002))
    # Reuse the exact pair+distance semantics, but reject invalid
    # normal/location candidates by replacing their distance with +inf.
    top_distance = xp.where(
        normal_is_top & top_surface_point, touching_distance, xp.inf)
    top_pairs = contact_pair_mask(
        xp, contact_geom1, contact_geom2, top_distance,
        box_geom_id, foot_geom_ids)
    on_top = (footprint & top_pairs &
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
    # Root issue 3: body-over-obstacle XY is NOT body-on-elevated-support.
    # The old implementation lowered torso height reward by box height
    # simply because the torso projection was over the step, even with
    # BOTH feet still on flat ground. Conversely a foot on top after the
    # torso moved past the tiny box did not alter body-height reference.
    # Instead compute support-surface elevation from genuine feet with
    # top-facing contact vs flat floor; support-count weighting is
    # transparent but NOT contact-force/impulse weighting.
    surface_top_center = box_center + top_normal * half[2]
    box_height_above_floor = xp.sum(
        (surface_top_center - floor_origin) * floor_normal, axis=-1)
    supporting_feet = xp.sum(supported.astype(foot_xyz.dtype), axis=1)
    top_feet = xp.sum(on_top.astype(foot_xyz.dtype), axis=1)
    top_support_fraction = top_feet / xp.maximum(1., supporting_feet)
    support_reference_elevation = box_height_above_floor * top_support_fraction
    torso_signed = floor_torso_signed - support_reference_elevation

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
        'support_reference_elevation_m': support_reference_elevation,
        'foot_center_signed': foot_surface_signed,
        'toe_clearance_signed': toe_clearance,
        'foot_top_contact_candidate': on_top,
        'foot_floor_contact_candidate': floor_pairs,
        'foot_riser_contact': riser_or_side,
        'foot_supported_candidate': supported,
        'foot_in_box_footprint': footprint,
        'torso_over_box_footprint': torso_above_box,
        'box_contact_any':xp.any(box_pairs,axis=1),
        'support_fraction': xp.mean(weights, axis=1),
        'stance_slip': slip,
        'swing_clearance': xp.maximum(
            0., xp.max(toe_clearance, axis=1)),
        # Signed minimum is deliberately preserved for riser auditing,
        # but it MUST NOT be interpreted as MuJoCo solver penetration.
        'min_toe_clearance_signed': xp.min(toe_clearance, axis=1),
    }


def collidable_course(robot_source, dest, height_mm: int, *, yaw_deg: float = 0.):
    """Write the reproducible real MuJoCo course used by v5018 crossing exams.

    The 0mm control retains the same box geom, parked 5m off course. The
    routine refuses to overwrite an existing DIFFERENT course (important for
    checkpoint lineage and independent reproduction).
    """
    import math
    from pathlib import Path
    import xml.etree.ElementTree as ET

    if height_mm not in (0, 5, 10, 15, 20):
        raise ValueError('only declared 0, 5, 10, 15 or 20 mm pilot steps')
    if not math.isfinite(yaw_deg) or abs(yaw_deg) > 35:
        raise ValueError('bounded step yaw is -35 through +35 degrees')
    tree = ET.parse(Path(robot_source))
    world = tree.getroot().find('worldbody')
    if world is None:
        raise ValueError('canonical robot is missing worldbody')
    h = height_mm / 1000.
    x = .20 if height_mm else 5.
    thickness = h / 2. if h else .0025
    attributes = {
        'name': 'v5018_obstacle_step', 'type': 'box',
        'pos': f'{x} 0 {thickness}',
        'size': f'.04 .20 {thickness}',
        'friction': '1.5 .005 .0001',
    }
    if yaw_deg != 0.:
        attributes['euler'] = f'0 0 {yaw_deg}'
    ET.SubElement(world, 'geom', attributes)
    # Render to memory first; never damage an existing run's course file.
    return _write_reproducible_course(tree, Path(dest))


def _write_reproducible_course(tree, dest):
    from io import StringIO
    content = StringIO()
    tree.write(content, encoding='unicode', xml_declaration=True)
    encoded = content.getvalue().encode('utf-8')
    if dest.exists():
        if not dest.is_file() or dest.read_bytes() != encoded:
            raise RuntimeError(f'Course file differs from declared geometry: {dest}')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as out:
        out.write(encoded)
    return dest
