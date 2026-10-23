"""v5022 physical terrain and deployable ToF forward models.

The teacher defines *solid* course geometry; the actor never receives that
geometry. Both the down tripod and front sensor intersect the SAME world
solids as MJX, with front feet included as visible occluders. Sensor latency
and asynchronous delivery belong to environment.py, not this pure JAX/NumPy
geometry layer. An optical-zone model is only an approximation of the real
VL53L8CX histogram algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

# ST VL53L8CX (nominal 65-degree diagonal, 45 x 45 degree square FoV).
# The 8x8-capable hardware is explicitly configured in 4x4 (16-zone) mode.
SENSOR_MODEL = "VL53L8CX"
SENSOR_RESOLUTION = (4, 4)
SENSOR_FOV_DEG = (45.0, 45.0)
SENSOR_MOUNT_BODY_M = (.090, .0, -.018)  # provisional; CAD clearance needed
SENSOR_DOWN_TILT_DEG = 50.0
SENSOR_RANGE_M = (.020, 1.5)  # deliberate training ROI, not rated max of 4m
# Centre + 4 quarter-zone rays shared with the browser reference.
# These are footprint surrogates, NOT actual ST histogram processing.
_ZONE_OFFSETS_DEG = ((0., 0.), (-2.8125, -2.8125),
                     (-2.8125, 2.8125), (2.8125, -2.8125),
                     (2.8125, 2.8125))


@dataclass(frozen=True)
class Surface:
    x0_m: float
    x1_m: float
    z0_m: float
    grade_rad: float

    def height_at(self, x_m: float) -> float:
        return self.z0_m + (x_m - self.x0_m) * math.tan(self.grade_rad)

    @property
    def z1_m(self) -> float:
        return self.height_at(self.x1_m)


@dataclass(frozen=True)
class Seam:
    x_m: float
    jump_m: float
    change_rad: float

    @property
    def discontinuous(self) -> bool:
        return abs(self.jump_m) > 1e-8


def course_surfaces(x0_m, z0_m, lengths_m, grades_deg, jumps_m=()):
    """Independent grade and signed jump: a crest is NOT a step."""
    lengths, grades, jumps = (tuple(map(float, a)) for a in
                              (lengths_m, grades_deg, jumps_m))
    if not lengths or len(lengths) != len(grades) or len(jumps) != len(lengths)-1:
        raise ValueError("terrain requires N lengths, N grades and N-1 jumps")
    if not all(map(math.isfinite, (float(x0_m), float(z0_m), *lengths,
                                   *grades, *jumps))):
        raise ValueError("nonfinite terrain geometry")
    if any(x <= 0 for x in lengths) or any(abs(x) >= 65 for x in grades):
        raise ValueError("invalid terrain segment length or angle")
    surfaces, seams = [], []
    x, z = float(x0_m), float(z0_m)
    for i, (length, grade_deg) in enumerate(zip(lengths, grades)):
        segment = Surface(x, x+length, z, math.radians(grade_deg))
        surfaces.append(segment)
        x, z = segment.x1_m, segment.z1_m
        if i < len(jumps):
            seams.append(Seam(x, jumps[i],
                              math.radians(grades[i+1]-grades[i])))
            z += jumps[i]
    return tuple(surfaces), tuple(seams)


def surface_box_geometry(surface: Surface, *, thickness_m=.12, half_width_m=.4):
    """MJCF solid BOX pose, half-sizes and wxyz rotation for surface contact."""
    if thickness_m <= 0 or half_width_m <= 0:
        raise ValueError("solid support needs positive thickness/width")
    a = surface.grade_rad
    xm = (surface.x0_m + surface.x1_m)*.5
    zm = surface.height_at(xm)
    pos = (xm + thickness_m*math.sin(a), 0.,
           zm-thickness_m*math.cos(a))
    size = ((surface.x1_m-surface.x0_m)/(2*math.cos(a)),
            half_width_m, thickness_m)
    quat = (math.cos(a*.5), 0., -math.sin(a*.5), 0.)
    return pos, size, quat


def ray_boxes(xp, origins, directions, centres, rotations, half_sizes):
    """Nearest ray intersection with [B] or [W,B] real oriented MJX boxes.

    origins/directions: [W,R,3]; model poses from data.geom_xpos/xmat/size.
    Returns [W,R] distances or +inf. No privileged actor feature here.
    """
    o, d = xp.asarray(origins), xp.asarray(directions)
    p, rot, size = map(xp.asarray, (centres, rotations, half_sizes))
    if p.ndim == 2:
        p = p[None]
    if rot.ndim == 3:
        rot = rot[None]
    if size.ndim == 2:
        size = size[None]
    delta = o[:, :, None, :] - p[:, None, :, :]
    local_o = xp.einsum("wbji,wrbj->wrbi", rot, delta)
    local_d = xp.einsum("wbji,wrj->wrbi", rot, d)
    half = size[:, None, :, :]
    parallel = xp.abs(local_d) < 1e-8
    safe = xp.where(parallel, 1., local_d)
    a, b = (-half-local_o)/safe, (half-local_o)/safe
    enter = xp.max(xp.where(parallel, -xp.inf, xp.minimum(a,b)), axis=-1)
    leave = xp.min(xp.where(parallel, xp.inf, xp.maximum(a,b)), axis=-1)
    in_axes = xp.all((~parallel) | (xp.abs(local_o) <= half), axis=-1)
    hit = in_axes & (leave >= xp.maximum(enter, 0.)) & (leave >= 0.)
    first = xp.where(enter >= 0., enter, leave)
    return xp.min(xp.where(hit, first, xp.inf), axis=-1)


def ray_spheres(xp, origins, directions, centres, radii):
    """Nearest actual visible foot-sphere intersection, batched per world.

    Spherical foot radii/centres come from *live* robot kinematics, not the
    future teacher course. If a foot moves into view it must occlude a zone.
    """
    o, d, c = map(xp.asarray, (origins, directions, centres))
    if c.ndim == 2:
        c = c[None]
    radius = xp.asarray(radii)
    delta = o[:, :, None, :] - c[:, None, :, :]
    b = xp.sum(delta*d[:, :, None, :], axis=-1)
    discr = b*b - (xp.sum(delta*delta, axis=-1) - radius*radius)
    root = xp.sqrt(xp.maximum(discr, 0.))
    near, far = -b-root, -b+root
    t = xp.where(near >= 0., near, far)
    return xp.min(xp.where((discr >= 0.) & (t >= 0.), t, xp.inf),
                  axis=-1)


def camera_rays(xp, torso_world, body_to_world, *,
                tilt_down_deg=SENSOR_DOWN_TILT_DEG,
                mount_body_m=SENSOR_MOUNT_BODY_M):
    """[W,16,5,3] origins and unit directions from ONE optical centre.

    Array is in spatial row-major order: upper/far rows first, columns
    right-to-left in ROBOT body Y. Physical driver orientation must be
    validated against ST zone numbering before robot deployment.
    """
    p, R = xp.asarray(torso_world), xp.asarray(body_to_world)
    coord = xp.asarray((-16.875, -5.625, 5.625, 16.875))
    sub = xp.asarray(_ZONE_OFFSETS_DEG)
    y = coord[None, :, None] + sub[None, None, :, 0]
    # Row 0 is the shallowest downward angle (farthest patch).
    down = tilt_down_deg + coord[:, None, None] + sub[None, :, 1][None]
    az = xp.broadcast_to(y, (4,4,5))
    el = xp.broadcast_to(down, (4,4,5))
    x = xp.cos(xp.deg2rad(el))*xp.cos(xp.deg2rad(az))
    lateral = xp.cos(xp.deg2rad(el))*xp.sin(xp.deg2rad(az))
    vertical = -xp.sin(xp.deg2rad(el))
    body_rays = xp.stack((x, lateral, vertical), axis=-1).reshape(16,5,3)
    direction = xp.einsum("wij,rsj->wrsi", R, body_rays)
    mount = xp.asarray(mount_body_m)
    origin = p+xp.einsum("wij,j->wi",R,mount)
    return xp.broadcast_to(origin[:,None,None,:], direction.shape), direction


def scan_forward(xp, torso_world, body_to_world,
                 box_centres, box_rotations, box_half_sizes,
                 *, foot_centres=None, foot_radii=.018,
                 noise_m=None, dropout=None,
                 tilt_down_deg=SENSOR_DOWN_TILT_DEG,
                 mount_body_m=SENSOR_MOUNT_BODY_M,
                 subrays=5):
    """Raw 4x4 front depth packet: range [W,16], validity [W,16].

    Rays intersect the real physical solid boxes and optionally moving
    feet. Foot occlusion is NOT magically removed: a nearby foot can
    become the first return. 1 / 4 / 5 subrays represent 16 / 64 / 80
    geometric samples; output is ALWAYS 16 zone ranges. A 60% valid
    subray threshold and lower median are not hardware guarantees.
    The environment must add capture/arrival age, delay and dropout.
    """
    if subrays not in (1, 4, 5):
        raise ValueError('subrays must be 1, 4 or 5')
    origin, direction = camera_rays(
        xp, torso_world, body_to_world, tilt_down_deg=tilt_down_deg,
        mount_body_m=mount_body_m)
    # Python and browser sample identical centre/quarter-zone directions.
    # All zone rays share the same optical centre. Broadcast one origin
    # regardless of 1/4/5 subray mode; do not retain five origins for four rays.
    origin = origin[:, :, :1, :]
    direction = (direction[:, :, :1, :] if subrays == 1 else
                 direction[:, :, 1:5, :] if subrays == 4 else direction)
    worlds = xp.asarray(torso_world).shape[0]
    origin = xp.broadcast_to(origin, direction.shape)
    o, d = origin.reshape(worlds, 16*subrays, 3), direction.reshape(worlds, 16*subrays, 3)
    ranges = ray_boxes(xp, o, d, box_centres,
                       box_rotations, box_half_sizes)
    if foot_centres is not None:
        ranges = xp.minimum(ranges, ray_spheres(
            xp, o, d, foot_centres, foot_radii))
    subranges = ranges.reshape(worlds,16,subrays)
    inside = (xp.isfinite(subranges) &
              (subranges >= SENSOR_RANGE_M[0]) &
              (subranges <= SENSOR_RANGE_M[1]))
    count = xp.sum(inside,axis=-1)
    sorted_r = xp.sort(xp.where(inside,subranges,xp.inf),axis=-1)
    mid = xp.maximum(0,(count-1)//2).astype(int)
    value = xp.take_along_axis(sorted_r,mid[...,None],axis=-1)[...,0]
    valid = (count >= (3*subrays + 4)//5) & xp.isfinite(value)
    if noise_m is not None:
        value = value+xp.asarray(noise_m)
    valid &= (value >= SENSOR_RANGE_M[0]) & (value <= SENSOR_RANGE_M[1])
    if dropout is not None:
        valid &= ~xp.asarray(dropout,dtype=bool)
    return xp.where(valid,value,0.),valid


def scan_tripod(xp, torso_world, body_to_world,
                box_centres, box_rotations, box_half_sizes,
                *, noise_m=None, dropout=None):
    """Legacy down-looking hardware geometry against the SAME solid course.

    This is opt-in for new terrain scenes; existing 232 policies retain
    their untouched original floor-plane sensor model.
    """
    from .directional import (
        TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY,
        TOF_MIN_RANGE_M, TOF_MAX_RANGE_M)
    p, R = xp.asarray(torso_world), xp.asarray(body_to_world)
    o = p[:,None,:]+xp.einsum("wij,kj->wki", R,
                             xp.asarray(TOF_TRIPOD_MOUNTS_BODY_M))
    d = xp.einsum("wij,kj->wki", R,
                  xp.asarray(TOF_TRIPOD_BEAMS_BODY))
    ranges = ray_boxes(xp,o,d,box_centres,box_rotations,box_half_sizes)
    if noise_m is not None:
        ranges = ranges+xp.asarray(noise_m)
    valid = (xp.isfinite(ranges) & (ranges >= TOF_MIN_RANGE_M) &
             (ranges <= TOF_MAX_RANGE_M))
    if dropout is not None:
        valid &= ~xp.asarray(dropout,dtype=bool)
    return xp.where(valid,ranges,0.),valid
