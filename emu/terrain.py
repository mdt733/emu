"""Physical terrain geometry and teacher-only contact/crossing evidence.

The current MJX course uses fixed-topology, per-world movable box geometry;
actual height and placement can vary without changing JAX tensor shapes.
Foot-on-riser collision is not box-top support, and a candidate contact
pair is not proof of positive normal force.

These functions may supervise reward, curriculum and frozen exams.
Actor observations are reconstructed solely from simulated physical sensors,
never from privileged terrain height, world geometry or collision labels.
"""
from __future__ import annotations

from .directional import contact_pair_mask

def box_crossing_position_flags(xp, *, foot_xyz, torso_xyz,
                                box_center, box_rotation, box_half_size):
    """Teacher-only per-world crossing geometry for a yawed physical box.

    foot_xyz: [N,2,3], torso_xyz/box_center: [N,3],
    box_rotation: [N,3,3], box_half_size: [3].
    Output is three [N] flags. NEVER compare [N,2] feet to an [N]
    threshold without an explicit [:,None] world-to-foot broadcast.

    This is only evidence of being past the box; the caller separately
    requires real supporting contacts, correct commands and upright gait.
    """
    half=xp.asarray(box_half_size)
    extent_x=(xp.abs(box_rotation[:,0,0])*half[0] +
              xp.abs(box_rotation[:,0,1])*half[1])
    beyond=box_center[:,0]+extent_x
    torso_far=torso_xyz[:,0]>=beyond+.18
    feet_far=xp.all(foot_xyz[:,:,0]>=(beyond+.04)[:,None],axis=1)
    lane_y=box_center[:,1]
    in_lane=((xp.abs(torso_xyz[:,1]-lane_y)<.11) &
             xp.all(xp.abs(foot_xyz[:,:,1]-lane_y[:,None])<.12,axis=1))
    return torso_far,feet_far,in_lane


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


# Fixed-shape box whose top moves with a
# per-episode MuJoCo mocap body. Avoid changing MJX geom_size/topology
# under JIT; every batched world gets an independently sampled height.
TERRAIN_FRONTIER_MAX_M = 0.150
TERRAIN_BOX_HALF_HEIGHT_M = 0.075


def adaptive_height_course(robot_source, dest, *, immutable: bool = True,
                           elevation_platform: bool = False):
    """Build one batched physical obstacle with fixed topology.

    Historical short hurdles remain the default. The opt-in 2m-long,
    1m-wide landing has its vertical edge moved forward for an ascent, or
    behind the robot for a descent; all geometry is real MuJoCo contact.
    Per-world mocap positions are sampled only at episode birth.
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET

    tree = ET.parse(Path(robot_source))
    world = tree.getroot().find('worldbody')
    if world is None:
        raise ValueError('canonical robot is missing worldbody')
    # A thick 2m platform geometrically penetrates the floor so its top
    # forms a real riser at small positive heights. Both terrain bodies are
    # kinematic; their *mutual* contact is unphysical and can consume MJX
    # contact slots that the feet need. Use disjoint collision categories
    # for plane (2) and platform (4), each accepting the default robot (1).
    # Foot<->floor and foot<->platform still collide: (1&1)!=0.
    if elevation_platform:
        floor = next((g for g in world.findall('geom')
                      if g.get('name') == 'floor'), None)
        if floor is None or floor.get('type') != 'plane':
            raise ValueError('raised-platform course requires canonical floor plane')
        floor.set('contype', '2')
        floor.set('conaffinity', '1')
    mocap = ET.SubElement(world, 'body', {
        'name': 'v5018_course_mocap', 'mocap': 'true', 'pos': '5 0 0'})
    attributes = {
        'name': 'v5018_obstacle_step', 'type': 'box',
        'pos': f'0 0 {-TERRAIN_BOX_HALF_HEIGHT_M}',
        'size': (f'1 .5 {TERRAIN_BOX_HALF_HEIGHT_M}' if elevation_platform
                 else f'.04 .20 {TERRAIN_BOX_HALF_HEIGHT_M}'),
        'friction': '1.5 .005 .0001'}
    if elevation_platform:
        attributes.update(contype='4', conaffinity='1')
    ET.SubElement(mocap, 'geom', attributes)
    return _write_reproducible_course(tree, Path(dest), immutable=immutable)


def adaptive_height_sample(xp, upper_m, uniform, selection, *, exposure=0.60, edge_fraction=.35):
    """Flat rehearsal + continuous interior samples + a frontier-height tail.

    The upper_m input is learned by the governor; NOT an actor channel.
    """
    high = xp.clip(upper_m, 0., TERRAIN_FRONTIER_MAX_M)
    exposure = xp.clip(exposure, 0., 1.)
    frac = xp.where(uniform < edge_fraction, .72 + .28 * uniform / xp.maximum(edge_fraction, 1.e-6),
                    (uniform - edge_fraction) / xp.maximum(1.-edge_fraction, 1.e-6))
    return xp.where(selection < exposure, high * frac, 0.)


def next_height_frontier(current_m: float, row: dict, *, safety_m: float,
                         growth_m: float = .001, min_exposure: float = .005):
    """Evidence-gated advancement/retreat of physical step difficulty.

    Contact alone does not earn growth. Require completed crossing samples,
    acceptable falls and actual exposure; absent readings cannot expand.
    """
    import math
    if not 0 <= current_m <= safety_m <= TERRAIN_FRONTIER_MAX_M:
        raise ValueError('invalid terrain physical frontier bounds')
    def measured(key):
        try:
            v = float(row[key])
            return v if math.isfinite(v) and v >= 0. else None
        except (ValueError, TypeError, KeyError):
            return None
    exposure = measured('terrain_edge_exposure')
    completed = measured('terrain_edge_done_count')
    # The governor pools TRUE terminal edge trials across multiple rollouts.
    # Requiring all episodes to finish in one 32-frame batch would trap a
    # healthy long-lived biped at the initial 3mm frontier forever.
    if completed is None or completed < 8:
        return current_m, 'hold_insufficient_completed_episodes'
    arrival = measured('terrain_edge_crossing')
    falls = measured('terrain_edge_fall_rate')
    contact = measured('terrain_edge_contact')
    if exposure is None or exposure < min_exposure:
        return current_m, 'hold_no_edge_coverage'
    if arrival is None or falls is None or contact is None:
        return current_m, 'hold_missing_physical_evidence'
    if arrival >= .25 and falls <= .35 and contact >= .0002:
        return min(safety_m, current_m + growth_m), 'expand_measured_crossing'
    if falls >= .65 and (contact >= .0002 or exposure >= .02):
        return max(.001, current_m - 2 * growth_m), 'retreat_sustained_falls'
    return current_m, 'hold_unproven'

def collidable_course(robot_source, dest, height_mm: int, *, yaw_deg: float = 0., immutable: bool = False):
    """Write the reproducible real MuJoCo course used by v5018 crossing exams.

    The 0mm control retains the same box geom, parked 5m off course.
    Training enables immutable=True: mismatched existing run geometry fails
    closed. The historical examiner keeps reusing temporary output paths.
    """
    import math
    from pathlib import Path
    import xml.etree.ElementTree as ET

    if type(height_mm) is not int or height_mm not in (0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150):
        raise ValueError('exam height must be a declared 0..150mm real step')
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
    return _write_reproducible_course(tree, Path(dest), immutable=immutable)


def _write_reproducible_course(tree, dest, *, immutable):
    from io import StringIO
    content = StringIO()
    tree.write(content, encoding='unicode', xml_declaration=True)
    encoded = content.getvalue().encode('utf-8')
    if dest.exists():
        if not dest.is_file():
            raise RuntimeError(f'Course target is not a file: {dest}')
        if dest.read_bytes() != encoded:
            if immutable:
                raise RuntimeError(f'Course file differs from declared geometry: {dest}')
            dest.write_bytes(encoded)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as out:
        out.write(encoded)
    return dest


def elevation_course(robot_source, dest, height_mm: int, *, direction: str,
                     edge_x_m: float = .25, landing_length_m: float = 2.,
                     width_m: float = 1., immutable: bool = True):
    """Physical elevation-transition exam fixture, not the hurdle curriculum.

    A long raised box over the real lower plane gives a genuine upper
    landing. Step-down starts on the box and travels toward its far edge.
    The canonical lower plane is never moved or replaced. The same named
    geometry is consumed by existing box-surface contact and ToF models.
    This fixture alone does not certify crossing or enable training.
    """
    import math
    from pathlib import Path
    import xml.etree.ElementTree as ET

    if direction not in ('up', 'down'):
        raise ValueError('elevation direction must be up or down')
    if type(height_mm) is not int or height_mm not in (5, 10, 20, 30, 40):
        raise ValueError('elevation exam height must be 5/10/20/30/40mm')
    if not all(math.isfinite(v) for v in (edge_x_m, landing_length_m, width_m)):
        raise ValueError('nonfinite elevation dimensions')
    if not .15 <= edge_x_m <= .50 or landing_length_m < 1. or width_m < .5:
        raise ValueError('elevation fixture needs safe approach and long, wide landing')
    tree = ET.parse(Path(robot_source))
    world = tree.getroot().find('worldbody')
    if world is None:
        raise ValueError('canonical robot is missing worldbody')
    half = landing_length_m / 2.
    centre = edge_x_m + (half if direction == 'up' else -half)
    h = height_mm / 1000.
    ET.SubElement(world, 'geom', {
        'name': 'v5018_obstacle_step', 'type': 'box',
        'pos': f'{centre} 0 {h/2}', 'size': f'{half} {width_m/2} {h/2}',
        'friction': '1.5 .005 .0001',
    })
    # CPU MuJoCo qpos0 and the environment's birth-height calculation both
    # put down-step starts on the physical upper surface.
    if direction == 'down':
        torso = world.find("body[@name='torso']")
        if torso is None:
            raise ValueError('canonical robot is missing torso')
        pos = [float(v) for v in torso.attrib['pos'].split()]
        pos[2] += h
        torso.set('pos', ' '.join(str(v) for v in pos))
    return _write_reproducible_course(tree, Path(dest), immutable=immutable)


def elevation_transition_update(
    xp, *, source_seen, contacted, landed, landing_x,
    supported_frames, post_landing_gait, crossed,
    source_support, obstacle_contact, target_support,
    torso_x, feet_x, edge_x, upright, gait, active,
):
    """Teacher/exam-only first-episode evidence for a true elevation change.

    Unlike the old hurdle test, the far end of a long platform is NOT the
    transition edge. We certify source-side support, physical box contact,
    both feet and torso past the transition, support on the destination
    elevation, then another 40mm of grounded motion with gait evidence.
    These privileged quantities must never enter actor observations.
    """
    source_seen = source_seen | (
        active & source_support & (torso_x < edge_x - .02))
    contacted = contacted | (active & obstacle_contact)
    past = (torso_x >= edge_x + .08) & xp.all(
        feet_x >= (edge_x + .02)[:, None], axis=1)
    confirmed_landing = (
        active & source_seen & contacted & past & target_support & upright)
    first_landing = confirmed_landing & ~landed
    landing_x = xp.where(first_landing, torso_x, landing_x)
    landed = landed | confirmed_landing
    # Keep cumulative destination-support frames rather than consecutive
    # frames: alternating biped feet may briefly enter swing phase.
    after_landing = active & landed & target_support & upright
    supported_frames = supported_frames + after_landing.astype(xp.int32)
    post_landing_gait = post_landing_gait + xp.where(
        after_landing, xp.maximum(gait, 0.), 0.)
    crossed = crossed | (
        confirmed_landing & (supported_frames >= 8) &
        (torso_x >= landing_x + .04) & (post_landing_gait >= 1.0))
    return (source_seen, contacted, landed, landing_x,
            supported_frames, post_landing_gait, crossed)
