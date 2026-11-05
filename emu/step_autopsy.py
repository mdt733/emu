"""v5051 independent mission autopsy on raw frozen-policy prism traces.

Read-only scorer. Every positive result must survive the existing independent
five-frame actual top-contact examiner; reward flags are never inputs.
The available trace does not include fore-aft CoM load or true step detection:
those stages remain UNKNOWN, never silently marked passed.
"""
from __future__ import annotations

import numpy as np

from .contracts import CONTROL_DT
from .prism_exam import first_episode_mask, score_prism_crossings
from .terrain_carpet_mjx import TILE_M


def _top(z, cell, point):
    """Physical triangle top interpolated from four actual prism corner heights."""
    x, y = np.asarray(point, float) / TILE_M - np.asarray(cell, float)
    if not (-.001 <= x <= 1.001 and -.001 <= y <= 1.001):
        return None
    a, b, c, d = map(float, z)
    return (a * (1-x-y) + b*x + c*y if x+y <= 1
            else d * (x+y-1) + b*(1-y) + c*(1-x))


def _slot(cell):
    return int(cell[0] % 2 + 2 * (cell[1] % 2))


def inspect_missions(frames, *, minimum_progress_m=.50, heading_limit_rad=.35,
                     support_frames=5, max_followup_s=10.):
    """Return per-eligible-verified-contact post-lip mission evidence.

    This is NOT a fixed-height obstacle generator. It uses observed signed
    prism rises and first-life contacts from the existing examiner, then asks
    if BOTH feet landed on upper tile and if body continued 50cm in the
    original heading. A zero eligible denominator is 'not measured'.
    """
    if not (0. < minimum_progress_m <= 2. and
            0. < heading_limit_rad <= 3.14 and
            1 <= support_frames <= 30 and 0. < max_followup_s <= 30.):
        raise ValueError("Invalid mission scoring protocol")
    crossings = score_prism_crossings(frames, required_frames=support_frames)
    xyz = np.asarray(frames["foot_xyz"], float)
    xy = np.asarray(frames["position_xy"], float)
    yaw = np.unwrap(np.asarray(frames["heading_rad"], float), axis=0)
    cells = np.asarray(frames["carpet_root_cell"], int)
    corners = np.asarray(frames["carpet_prism_corner_z"], float)
    pairs = np.asarray(frames["carpet_raw_contact_pairs"], bool)
    vertical = np.asarray(frames["carpet_contact_vertical"], float)
    radii = np.asarray(frames["initial_foot_radii_m"], float)
    alive = first_episode_mask(frames["done"])
    fell = np.asarray(frames["fell"], bool)
    time, worlds = xy.shape[:2]
    if xyz.shape != (time, worlds, 2, 3) or yaw.shape != (time, worlds):
        raise ValueError("Physical trace shape mismatch")
    max_frames = int(max_followup_s / CONTROL_DT)
    result = []
    for w, row in enumerate(crossings["worlds"]):
        for event in row["independently_supported_lips"]:
            t0 = int(event["frame"])
            target = tuple(event["to_cell"])
            source = tuple(event["from_cell"])
            slot = _slot(target)
            # Ensure target-foot support uses actual contact with one of the
            # two prisms, correct vertical normal, actual top and foot radius.
            def on_target(t, foot):
                if tuple(cells[t,w]) != target:
                    # Root may leave the target tile after completed motion.
                    # Foot contact is nonetheless tied to target prism slot.
                    pass
                if not np.any(pairs[t,w,2*slot:2*slot+2,foot] &
                              (vertical[t,w,2*slot:2*slot+2,foot] >= .70)):
                    return False
                h = _top(corners[t,w,slot], target, xyz[t,w,foot,:2])
                return (h is not None and
                        abs(xyz[t,w,foot,2] - radii[foot] - h) <= .009)
            stop = min(time, t0 + max_frames + 1)
            first_death = np.flatnonzero(~alive[t0:stop,w] | fell[t0:stop,w])
            if len(first_death):
                stop = t0 + int(first_death[0])
            # The original crossing guarantees >=1 foot top contact for
            # five frames, not both feet or a physically loaded CoM transfer.
            landed = [None, None]
            current = [0, 0]
            for t in range(t0,stop):
                for f in range(2):
                    current[f] = current[f]+1 if on_target(t,f) else 0
                    if current[f] >= support_frames and landed[f] is None:
                        landed[f] = t
            both = all(x is not None for x in landed)
            recovered = max(landed) if both else None
            completed = False
            best_progress = 0.
            max_heading = 0.
            if recovered is not None:
                bearing = np.array([np.cos(yaw[t0,w]), np.sin(yaw[t0,w])])
                for t in range(recovered,stop):
                    forward = float((xy[t,w]-xy[t0,w]) @ bearing)
                    heading_error = abs(float(np.arctan2(
                        np.sin(yaw[t,w]-yaw[t0,w]),
                        np.cos(yaw[t,w]-yaw[t0,w]))))
                    best_progress = max(best_progress,forward)
                    max_heading = max(max_heading,heading_error)
                    if forward >= minimum_progress_m and heading_error <= heading_limit_rad:
                        # Heading at the finish is insufficient if heading
                        # has been grossly wrong in intermediate frames.
                        interval = yaw[t0:t+1,w]-yaw[t0,w]
                        wrapped = np.abs(np.arctan2(np.sin(interval),np.cos(interval)))
                        if np.max(wrapped) <= heading_limit_rad:
                            completed = True
                            break
            if completed:
                verdict = "completed_mission"
            elif not both:
                verdict = "trailing_foot_not_verified"
            elif stop < time and (fell[stop,w] or not alive[stop,w]):
                verdict = "fell_after_support"
            else:
                verdict = "insufficient_controlled_progress"
            result.append(dict(
                world=w, frame=t0,
                from_cell=list(source),to_cell=list(target),
                signed_actual_height_cm=float(event["signed_height_cm"]),
                certified_upper_contact=True,
                both_feet_upper_support=bool(both),
                trailing_recovery_frame=recovered,
                post_contact_progress_m=round(best_progress,4),
                max_heading_error_rad=round(max_heading,4),
                complete=bool(completed), verdict=verdict,
                detection="unmeasured",
                swing_riser_clearance="unmeasured",
                weight_transfer="unmeasured",
            ))
    return dict(
        protocol=dict(minimum_progress_m=minimum_progress_m,
                      heading_limit_rad=heading_limit_rad,
                      support_frames=support_frames,
                      max_followup_s=max_followup_s,
                      note="Requires sufficiently long traces; positive certification only. "
                           "Unknown detection/clearance/load-transfer stages are NOT passed."),
        eligible_supported_contacts=len(result),
        completed_missions=sum(x["complete"] for x in result),
        events=result,
    )
