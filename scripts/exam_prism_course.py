"""Independent frozen-policy prism STEP examiner: raw contacts + actual triangle tops.

Never read reward, academy crossing flags, or teacher attempt classifications
when determining an independently supported physical traversal.
"""
from __future__ import annotations

import numpy as np

TILE_M = .50


def score_prism_crossings(frames, *, required_frames=5):
    """Count first-life, new-cell crossings with sustained true prism top contact.

    Expected arrays are PRE-step MJX observations, consistently time-aligned:
      carpet_root_cell            [time, world, 2] (global integer cells)
      carpet_prism_corner_z       [time, world, 9, 4] (metres, physical XY corners)
      carpet_raw_contact_pairs    [time, world, 18, 2]
      carpet_contact_vertical     [time, world, 18, 2] (abs normal Z, real MJX)
      foot_xyz                    [time, world, 2, 3] (sphere centres)
      initial_foot_radii_m        [2]
      done, fell                  [time, world]

    The source heights are independently compared against the 18 *physical*
    convex-mesh corner positions in the prism acceptance probe; reward flags
    are never used as scoring inputs.
    """
    from scripts.exam_step_course import first_episode_mask

    cells = np.asarray(frames["carpet_root_cell"], dtype=np.int32)
    tops = np.asarray(frames["carpet_prism_corner_z"], dtype=float)
    pairs = np.asarray(frames["carpet_raw_contact_pairs"], dtype=bool)
    vertical = np.asarray(frames["carpet_contact_vertical"], dtype=float)
    feet = np.asarray(frames["foot_xyz"], dtype=float)
    radii = np.asarray(frames["initial_foot_radii_m"], dtype=float)
    done = np.asarray(frames["done"], dtype=bool)
    fell = np.asarray(frames["fell"], dtype=bool)
    if cells.ndim != 3 or cells.shape[-1] != 2:
        raise ValueError("Prism trace needs [time, worlds, 2] root cells")
    time, worlds, _ = cells.shape
    shapes = {
        "carpet_prism_corner_z": (time, worlds, 9, 4),
        "carpet_raw_contact_pairs": (time, worlds, 18, 2),
        "carpet_contact_vertical": (time, worlds, 18, 2),
        "foot_xyz": (time, worlds, 2, 3),
        "done": (time, worlds),
        "fell": (time, worlds),
        "initial_foot_radii_m": (2,),
    }
    values = dict(carpet_prism_corner_z=tops,
                  carpet_raw_contact_pairs=pairs,
                  carpet_contact_vertical=vertical,
                  foot_xyz=feet, done=done, fell=fell,
                  initial_foot_radii_m=radii)
    for name, shape in shapes.items():
        if values[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {values[name].shape}")
    if not (1 <= required_frames <= 30):
        raise ValueError("required_frames must be 1..30")
    if not (np.isfinite(tops).all() and np.isfinite(vertical).all()
            and np.isfinite(feet).all() and np.isfinite(radii).all()
            and np.all((radii > 0.) & (radii < .10))):
        raise ValueError("Nonfinite or invalid physical prism trace")

    valid = first_episode_mask(done)

    def slot(cell):
        return int(cell[0] % 3 + 3 * (cell[1] % 3))

    def triangle_height(t, w, cell, xy):
        x = float(xy[0] / TILE_M - cell[0])
        y = float(xy[1] / TILE_M - cell[1])
        # A sphere centre should be over the actual physical tile footprint.
        inside = (-.001 <= x <= 1.001 and -.001 <= y <= 1.001)
        if not inside:
            return 0., False
        a, b, c, d = tops[t, w, slot(cell)]
        if x + y <= 1.:
            z = a * (1. - x - y) + b * x + c * y
        else:
            z = d * (x + y - 1.) + b * (1. - y) + c * (1. - x)
        return float(z), True

    def edge_xy(a, b):
        dx, dy = int(b[0] - a[0]), int(b[1] - a[1])
        if abs(dx) + abs(dy) != 1:
            raise ValueError("Only orthogonally adjacent tiles share a lip")
        if dx:
            x = max(a[0], b[0]) * TILE_M
            y = (a[1] + .5) * TILE_M
        else:
            x = (a[0] + .5) * TILE_M
            y = max(a[1], b[1]) * TILE_M
        return np.array([x, y], dtype=float)

    per_world = []
    for w in range(worlds):
        confirmed = tuple(int(v) for v in cells[0, w])
        visited = {confirmed}
        candidate = None
        streak = 0
        observed = []
        smooth = 0
        for t in range(time):
            if not valid[t, w]:
                break
            target = tuple(int(v) for v in cells[t, w])
            dx = target[0] - confirmed[0]
            dy = target[1] - confirmed[1]
            if abs(dx) + abs(dy) != 1 or target in visited:
                candidate, streak = None, 0
                continue
            if candidate != target:
                candidate, streak = target, 0
            target_slot = slot(target)
            supported = False
            for f in range(2):
                # Two triangles per tile, each a REAL convex MJX collider.
                contact_ok = any(
                    pairs[t, w, 2 * target_slot + tri, f] and
                    vertical[t, w, 2 * target_slot + tri, f] >= .70
                    for tri in (0, 1))
                if not contact_ok:
                    continue
                surface, inside = triangle_height(
                    t, w, target, feet[t, w, f, :2])
                if inside and abs(
                    feet[t, w, f, 2] - radii[f] - surface) <= .009:
                    supported = True
                    break
            if not supported or fell[t, w]:
                streak = 0
                continue
            streak += 1
            if streak < required_frames:
                continue
            edge = edge_xy(confirmed, target)
            old_height, ok_old = triangle_height(t, w, confirmed, edge)
            new_height, ok_new = triangle_height(t, w, target, edge)
            if not (ok_old and ok_new):
                raise AssertionError("Measured physical lip edge is missing")
            rise = new_height - old_height
            if abs(rise) > .0015:
                observed.append(dict(
                    from_cell=list(confirmed), to_cell=list(target),
                    signed_height_cm=round(rise * 100., 4), frame=t))
            else:
                smooth += 1
            confirmed = target
            visited.add(confirmed)
            candidate, streak = None, 0
        per_world.append(dict(
            world=w,
            independent_up=sum(e["signed_height_cm"] > 0 for e in observed),
            independent_down=sum(e["signed_height_cm"] < 0 for e in observed),
            smooth_grade_transitions=smooth,
            independently_supported_lips=observed))
    return dict(
        up=sum(w["independent_up"] for w in per_world),
        down=sum(w["independent_down"] for w in per_world),
        total=sum(w["independent_up"] + w["independent_down"] for w in per_world),
        worlds=per_world)
