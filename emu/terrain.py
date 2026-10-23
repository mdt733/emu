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
SENSOR_MOUNT_BODY_M = (.0, .0, -.018)  # 0mm forward viewer candidate; CAD clearance TBD
SENSOR_DOWN_TILT_DEG = 70.0  # viewer candidate, not frozen hardware specification
SENSOR_RANGE_M = (.020, 1.5)  # deliberate training ROI, not rated max of 4m
# Centre + four quarter-zone rays, shared with viewer's square image plane.
# These are footprint surrogates, NOT hardware ST histogram sub-pixels.
_ZONE_OFFSETS = ((0., 0.), (-1., -1.), (-1., 1.),
                 (1., -1.), (1., 1.))

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


# A limited first campaign: real raised slab (up/down edges) and a real
# ramp/crest. Flat spawn is preserved; the legacy training model is untouched.
# Geoms deliberately share one topology in all MJX worlds for fast batching.
TERRAIN_COURSE_REVISION = "v5022_step_crest_r1"
TERRAIN_COURSE_NAMES = ("front_step", "front_crest_up",
                        "front_crest_top", "front_crest_down")
TERRAIN_COURSE_SUPPORTS = (
    # Named geoms, x0, x1, z0, z1, abs(y) <= half width.
    ("front_step", .44, .94, .05, .05, .35),
    ("front_crest_up", 1.16, 1.43, 0., .045, .35),
    ("front_crest_top", 1.43, 1.58, .045, .045, .35),
    ("front_crest_down", 1.58, 1.85, .045, 0., .35),
)


def training_course_xml(xml: str) -> str:
    """Inject a real MJX/MuJoCo contact course in a separate experiment.

    Uses the original infinite floor and four solid boxes. Steps have
    vertical risers; a crest has continuous sloped top contact. Not a
    modified gravity vector and not per-world ghost sensor terrain.
    """
    if not isinstance(xml,str) or "</worldbody>" not in xml:
        raise ValueError("expected valid robot worldbody XML")
    if any(f'name="{name}"' in xml for name in TERRAIN_COURSE_NAMES):
        raise ValueError("terrain course already inserted")
    geoms=[]
    for name,x0,x1,z0,z1,width in TERRAIN_COURSE_SUPPORTS:
        a=math.atan2(z1-z0,x1-x0)
        p,size,quat=surface_box_geometry(
            Surface(x0,x1,z0,a), thickness_m=.18,half_width_m=width)
        fmt=lambda seq:" ".join(f"{float(v):.9g}" for v in seq)
        geoms.append(
            f'<geom name="{name}" type="box" pos="{fmt(p)}" '
            f'size="{fmt(size)}" quat="{fmt(quat)}" '
            'friction="1.5 0.005 0.0001" rgba="0.33 0.55 0.48 1"/>'
        )
    return xml.replace("</worldbody>","\n".join(geoms)+"\n</worldbody>")


def course_ground_height(xp, points):
    """Reward-only ground support height below [W,P,3] points.

    The only raised surfaces are the same four physical boxes injected
    above. Actor range packets must use intersections, NOT this height
    oracle. The existing plane remains at ground z=0.
    """
    p=xp.asarray(points)
    x,y=p[...,0],p[...,1]
    result=xp.zeros_like(x)
    for _,x0,x1,z0,z1,width in TERRAIN_COURSE_SUPPORTS:
        z=z0+(x-x0)*(z1-z0)/(x1-x0)
        inside=(x>=x0)&(x<=x1)&(xp.abs(y)<=width)
        result=xp.maximum(result,xp.where(inside,z,0.))
    return result


def ray_plane(xp, origins, directions, plane_origin, plane_normal):
    """Slant distance [W,R] to the SAME world plane as MuJoCo collision."""
    o,d=xp.asarray(origins),xp.asarray(directions)
    base=xp.asarray(plane_origin)
    normal=xp.asarray(plane_normal)
    dot=xp.sum(d*normal[:,None,:],axis=-1)
    t=xp.sum((base[:,None,:]-o)*normal[:,None,:],axis=-1)/xp.where(
        xp.abs(dot)>1e-8,dot,1.)
    return xp.where((xp.abs(dot)>1e-8)&(t>=0.),t,xp.inf)


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
    # The detector has a square 45° by 45° image plane. Its two optical
    # axes rotate TOGETHER with mount tilt. Treating elevation and azimuth
    # as independent global pitch/yaw angles collapsed the horizontal FoV
    # to a V/line when the sensor approached straight down (90°).
    half = xp.tan(xp.deg2rad(xp.asarray(22.5)))
    coord = xp.asarray((-0.75, -0.25, 0.25, 0.75)) * half
    sub = xp.asarray(_ZONE_OFFSETS) * (half/8)
    u = coord[None, :, None] + sub[None, None, :, 0]
    v = -coord[:, None, None] + sub[None, None, :, 1]
    u = xp.broadcast_to(u, (4,4,5))
    v = xp.broadcast_to(v, (4,4,5))
    tilt = xp.deg2rad(xp.asarray(tilt_down_deg))
    c, st = xp.cos(tilt), xp.sin(tilt)
    denom = xp.sqrt(1 + u*u + v*v)
    forward = (c + st*v) / denom
    lateral = u / denom
    vertical = (-st + c*v) / denom
    body_rays = xp.stack((forward, lateral, vertical), axis=-1).reshape(16,5,3)
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
                 subrays=5, floor_origin=None, floor_normal=None):
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
    if floor_origin is not None:
        if floor_normal is None:
            raise ValueError('floor normal required with floor origin')
        ranges=xp.minimum(ranges,ray_plane(xp,o,d,floor_origin,floor_normal))
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
