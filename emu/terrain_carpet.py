"""Deterministic 3×3 polygon-terrain contract, independent of MuJoCo.

One physical concept for standing, command, slope, step, resilience and
control specialists. Tiles have two triangles, shared continuous slope
vertices and explicit constant-height vertical discontinuities on edges.

This module is a deterministic *terrain specification*. Collision-engine
integration is gated behind the separate MJX heightfield/contact spike;
nothing here silently changes the old v5040 training collisions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

TILE_M = 0.5
RAD = math.pi / 180.0
ROLES = ("command", "resilience", "slope", "step", "transition", "control")
CARRYOVER = 0.5


def clamp(n: float, low: float, high: float) -> float:
    return max(low, min(high, float(n)))


def hash01(i: int, j: int, seed: int, salt: int = 0) -> float:
    """Platform-stable 32-bit hash: same tile on every revisit."""
    mask = 0xffffffff
    x = (i * 374761393 + j * 668265263 + seed * 1442695041
         + salt * 1274126177) & mask
    x = ((x ^ (x >> 13)) * 1274126177) & mask
    return ((x ^ (x >> 16)) & mask) / 4294967296.0


@dataclass(frozen=True)
class Limits:
    seed: int = 735
    max_slope_deg: float = 8.0
    max_transition_deg: float = 3.0
    max_step_up_m: float = 0.015
    max_step_down_m: float = 0.015
    step_probability: float = 0.34

    def validated(self) -> "Limits":
        if not (0 <= self.seed < 2**31 and
                0 <= self.max_slope_deg <= 25 and
                0 <= self.max_transition_deg <= 15 and
                0 <= self.max_step_up_m <= .25 and
                0 <= self.max_step_down_m <= .25 and
                0 <= self.step_probability <= 1):
            raise ValueError(f"Invalid terrain limits: {self!r}")
        for number in (self.max_slope_deg, self.max_transition_deg,
                       self.max_step_up_m, self.max_step_down_m,
                       self.step_probability):
            if not math.isfinite(number):
                raise ValueError("Terrain limits must be finite")
        return self


def role_limits(role: str, *, seed: int, slope_deg: float, transition_deg: float,
                step_up_m: float, step_down_m: float,
                carryover: float = CARRYOVER) -> Limits:
    """Source-of-truth shared physical teaching envelope.

    Every role has some input from the other proven physical skills.
    The factor governs *allowed maxima*, not falsely claimed competence.
    """
    if role not in ROLES and role != "standing":
        raise ValueError(f"Unknown terrain specialist: {role}")
    if not (0 <= carryover <= 1):
        raise ValueError("carryover must be 0..1")
    # Physical role focus affects ONLY the requested terrain envelope;
    # the actor, reward, optimiser and sensor contracts are unchanged.
    primary = {
        "standing": (0., 0., 0.),
        "command": (.20, .16, .15),
        "resilience": (.70, .60, .65),
        "slope": (1., 1., carryover),
        "step": (carryover, carryover, 1.),
        "transition": (.65, 1., .55),
        "control": (.15, .10, .10),
    }
    a, t, s = primary[role]
    probs = {
        "standing": 0., "command": .10, "resilience": .35,
        "slope": .25, "step": .70, "transition": .32, "control": .08,
    }
    return Limits(
        seed=int(seed),max_slope_deg=float(slope_deg)*a,
        max_transition_deg=float(transition_deg)*t,
        max_step_up_m=float(step_up_m)*s,
        max_step_down_m=float(step_down_m)*s,
        step_probability=probs[role],
    ).validated()


def _smooth_height(x: float, y: float, limits: Limits) -> float:
    """Globally smooth, band-limited support height independent of tile cache.

    Derivative limits are deliberately conservative; every tile samples the
    same function at its corners, making continuous seams crack-free.
    """
    s = math.tan(limits.max_slope_deg * RAD)
    if s <= 0:
        return 0.
    f = .89
    st = math.tan(limits.max_transition_deg * RAD)
    a = hash01(0, 0, limits.seed, 31) * 2*math.pi
    p1 = hash01(0, 0, limits.seed, 32) * 2*math.pi
    p2 = hash01(0, 0, limits.seed, 33) * 2*math.pi
    p3 = hash01(0, 0, limits.seed, 34) * 2*math.pi
    amp = min(s*.37/f, st*.68/(TILE_M*f*f*2.1))
    return (s*.47*(math.cos(a)*x+math.sin(a)*y)
            +amp*(.72*math.sin(f*x+p1)*math.cos(.73*f*y+p2)
                  +.28*math.sin(.49*f*x+.83*f*y+p3)))


def _offset(i: int, j: int, limits: Limits) -> float:
    # Fixed geometry viewed from either direction must be the same world.
    # A conservative symmetric offset never exceeds either direction's
    # maximum. Independent UP/DOWN challenges require a *directional*
    # terrain birth generator, not a posthoc relabelling of a fixed step.
    cap = min(limits.max_step_up_m, limits.max_step_down_m)
    if cap <= 0 or hash01(i, j, limits.seed, 41) >= limits.step_probability:
        return 0.
    amount = (.22+.28*hash01(i, j, limits.seed, 42))*cap
    return (-1 if hash01(i, j, limits.seed, 43)<.5 else 1)*amount


def tile(i: int, j: int, limits: Limits):
    limits.validated()
    x = i*TILE_M
    y = j*TILE_M
    d = _offset(i, j, limits)
    vertices = ((x,y,_smooth_height(x,y,limits)+d),
                (x+TILE_M,y,_smooth_height(x+TILE_M,y,limits)+d),
                (x+TILE_M,y+TILE_M,_smooth_height(x+TILE_M,y+TILE_M,limits)+d),
                (x,y+TILE_M,_smooth_height(x,y+TILE_M,limits)+d))
    return {"index":[i,j], "offset_m":d, "vertices":vertices,
            "triangles":((0,1,2),(0,2,3))}


def neighbours(x: float, y: float, limits: Limits):
    i, j = math.floor(x/TILE_M), math.floor(y/TILE_M)
    return {"center":[i,j], "tiles":[tile(i+di,j+dj,limits)
            for dj in (-1,0,1) for di in (-1,0,1)]}


def edge(a, b):
    i,j=a["index"]
    k,l=b["index"]
    if abs(i-k)+abs(j-l)!=1:
        raise ValueError("Edge requires orthogonally adjacent squares")
    aa,bb=a["vertices"],b["vertices"]
    if k==i+1: left=(aa[1],aa[2]);right=(bb[0],bb[3])
    elif k==i-1: left=(aa[0],aa[3]);right=(bb[1],bb[2])
    elif l==j+1: left=(aa[3],aa[2]);right=(bb[0],bb[1])
    else: left=(aa[0],aa[1]);right=(bb[3],bb[2])
    for p,q in zip(left,right):
        if abs(p[0]-q[0])>1e-9 or abs(p[1]-q[1])>1e-9:
            raise AssertionError("Real-world tile XY edges must coincide")
    return {"rise_m":b["offset_m"]-a["offset_m"],
            "end_gaps_m":[p[2]-q[2] for p,q in zip(left,right)]}


def check_neighbourhood(world, limits: Limits):
    t=world["tiles"]
    byidx={tuple(v["index"]):v for v in t}
    max_rise=0.
    max_gap=0.
    seams=0
    for a in t:
        i,j=a["index"]
        for delta in ((1,0),(0,1)):
            b=byidx.get((i+delta[0],j+delta[1]))
            if b is None:continue
            e=edge(a,b)
            max_rise=max(max_rise,abs(e["rise_m"]))
            max_gap=max(max_gap,abs(e["end_gaps_m"][0]-e["end_gaps_m"][1]))
            seams+=abs(e["rise_m"])>1e-9
    if max_rise>min(limits.max_step_up_m,limits.max_step_down_m)+1e-8:
        raise AssertionError("Generated ledge exceeded the physical frontier")
    if max_gap>1e-8:
        raise AssertionError("Diagonal slope and step wall endpoints disagree")
    return {"tiles":len(t),"triangles":len(t)*2,
            "step_seams":int(seams),"largest_step_m":max_rise,
            "seam_error_m":max_gap}


def snapshot_geometry(*, role: str, x: float, y: float, limits: Limits):
    world=neighbours(x,y,limits)
    return {"schema":"emu_carpet_geometry_v1","role":role,
            "limits":asdict(limits),"window":world,
            "checks":check_neighbourhood(world,limits)}
