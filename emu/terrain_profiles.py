"""Small v5022 terrain-course truth model, not deployable actor input.

Terrain is a sequence of SUPPORTING SURFACES. Each interval has an elevation
and grade; the boundary between intervals has a signed height jump.

Zero jump + nonzero grade change => crest, valley or slope transition.
Nonzero jump => step up/down, even if both sides have identical grade.

The model constructs actual rotated BOX geom parameters for MJX/CPU MuJoCo.
The teacher/examiner may inspect these profiles; the walking actor MAY NOT.
This is an engineering/geometry spike, not yet the training environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


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
    step_height_m: float
    grade_change_rad: float

    @property
    def discontinuous(self) -> bool:
        return abs(self.step_height_m) > 1e-8


def course_surfaces(x0_m: float, z0_m: float, lengths_m,
                    grades_deg, seam_jumps_m=()):
    """Build slopes and jumps without prescribing robot foot motions.

    Seam jumps occur at boundaries (not midway along slopes). No virtual
    surface is invented between different step heights.
    """
    lengths = tuple(map(float, lengths_m))
    grades = tuple(map(float, grades_deg))
    jumps = tuple(map(float, seam_jumps_m))
    if not lengths or len(lengths) != len(grades) or len(jumps) != len(lengths) - 1:
        raise ValueError("expected N lengths, N grades, and N-1 seam jumps")
    values = (x0_m, z0_m, *lengths, *grades, *jumps)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("terrain profile parameters must be finite")
    if not all(L > 0 for L in lengths):
        raise ValueError("terrain segment length must be positive")
    # Geometric representation cannot handle vertical slabs as angled top
    # surfaces. A vertical riser is represented as a jump, not 90° slope.
    if not all(abs(g) < 65 for g in grades):
        raise ValueError("slope surface must remain below 65 degrees")
    segments, seams = [], []
    x, z = float(x0_m), float(z0_m)
    for i, (L, angle) in enumerate(zip(lengths, grades)):
        slope = math.radians(angle)
        surface = Surface(x, x+L, z, slope)
        segments.append(surface)
        x, z = surface.x1_m, surface.z1_m
        if i < len(jumps):
            seams.append(Seam(x, jumps[i],
                              math.radians(grades[i+1] - grades[i])))
            z += jumps[i]
    return tuple(segments), tuple(seams)


def surface_box_geometry(surface: Surface, *,
                         thickness_m: float = .08, half_width_m: float = .40):
    """(geom centre xyz, half sizes, MuJoCo wxyz quaternion) for solid slab.

    Upper large face is exactly surface.height_at(x). The small end faces
    make step risers and exposed edges real collidable geometry. Independent
    thickness is explicitly a REPRESENTATION limit, not a learned ability.
    """
    a = surface.grade_rad
    if thickness_m <= 0 or half_width_m <= 0:
        raise ValueError("terrain box thickness and width must be positive")
    midx = (surface.x0_m + surface.x1_m) / 2
    midz = surface.height_at(midx)
    # local z face normal = (-sin(a), 0, cos(a)); quaternion rotates -a about Y.
    normal_x, normal_z = -math.sin(a), math.cos(a)
    center = (midx - thickness_m * normal_x, 0.,
              midz - thickness_m * normal_z)
    half_size = ((surface.x1_m - surface.x0_m) /
                 (2 * math.cos(a)), float(half_width_m), thickness_m)
    q = (math.cos(a/2), 0., -math.sin(a/2), 0.)
    return center, half_size, q


def profile_transition_kind(seam: Seam) -> str:
    if seam.step_height_m > 1e-8:
        return "step_up"
    if seam.step_height_m < -1e-8:
        return "step_down"
    if seam.grade_change_rad > 1e-8:
        return "slope_steepens"
    if seam.grade_change_rad < -1e-8:
        return "slope_flattens"
    return "level"
