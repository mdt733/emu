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
            both_streak = 0
            recovered = None
            for t in range(t0,stop):
                support_now = [on_target(t,f) for f in range(2)]
                for f in range(2):
                    current[f] = current[f]+1 if support_now[f] else 0
                    if current[f] >= support_frames and landed[f] is None:
                        landed[f] = t
                both_streak = both_streak + 1 if all(support_now) else 0
                if both_streak >= support_frames and recovered is None:
                    recovered = t
            # Distinct landings at separate times do NOT prove bilateral
            # recovery. Require simultaneous sustained upper-tile support.
            both = recovered is not None
            completed = False
            best_progress = 0.
            max_heading = 0.
            if recovered is not None:
                bearing = np.array([np.cos(yaw[t0,w]), np.sin(yaw[t0,w])])
                for t in range(recovered,stop):
                    forward = float((xy[t,w]-xy[recovered,w]) @ bearing)
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


def inspect_destinations(frames):
    """Independent frozen-actor exam of v5052's 60cm-past-first-lip task.

    The physical evidence comes from raw trace positions, first-episode
    prismatic *top* contacts and independent five-frame bilateral recovery.
    NEVER read an Academy reward or any teacher success boolean. This does
    not certify a guaranteed fixed-height wall: signed lip rise is measured.
    """
    from .step_goal import (arrival_quality, destination_from_birth,
                            GOAL_STABLE_FRAMES)
    # First sample is the true birth pose; the *post-step* pose is
    # independently recorded before the terminal auto-reset. Its frame
    # alignment matches physical support/tilt and lets a successful 5th
    # stable frame be verified even when completion instantly ends the run.
    birth_xy=np.asarray(frames["position_xy"],dtype=float)
    xy=np.asarray(frames.get(
        "physical_position_xy_after_step",frames["position_xy"]),dtype=float)
    yaw=np.asarray(frames.get(
        "physical_heading_rad_after_step",frames["heading_rad"]),dtype=float)
    alive=first_episode_mask(frames["done"])
    fell=np.asarray(frames["fell"],dtype=bool)
    gravity=np.asarray(frames["terrain_gravity_z"],dtype=float)
    support=np.asarray(frames["terrain_support"],dtype=float)
    bad=np.asarray(frames["terrain_bad_contact"],dtype=float)
    slip=np.asarray(frames["stance_slip_proxy"],dtype=float)
    target=destination_from_birth(np,birth_xy[0])
    origin=np.floor(birth_xy[0]/TILE_M).astype(int)
    events=inspect_missions(frames)["events"]
    rows=[]
    for w in range(xy.shape[1]):
        matches=[e for e in events
                 if e["world"]==w and
                 e["from_cell"]==origin[w].tolist() and
                 e["to_cell"]==[int(origin[w,0])+1,int(origin[w,1])]]
        verified=next((e for e in matches
                       if e["both_feet_upper_support"] and
                       e["trailing_recovery_frame"] is not None),None)
        start=(int(verified["trailing_recovery_frame"])
               if verified is not None else len(xy))
        streak=0
        completed=False
        arrived_frame=None
        # Only the first life is valid: a later reset could reach this
        # position while the original stair attempt already failed.
        life=np.flatnonzero(alive[:,w])
        closest=float(np.min(np.linalg.norm(
            xy[life,w]-target[w],axis=1))) if len(life) else None
        for t in range(start,len(xy)):
            if not alive[t,w] or fell[t,w]:
                break
            # Both position and health are now post-step physical
            # samples. The independent crossing scorer still verifies
            # raw pre-step prism contacts and first-life world identity.
            qualifies=bool(arrival_quality(
                np,target[w:w+1],xy[t,w:w+1],yaw[t,w:w+1]-
                yaw[0,w:w+1],gravity[t,w:w+1],support[t,w:w+1],
                bad[t,w:w+1],slip[t,w:w+1],
                np.array([verified is not None]))[0])
            streak=streak+1 if qualifies else 0
            if streak>=GOAL_STABLE_FRAMES:
                completed=True
                arrived_frame=t
                break
        rows.append(dict(
            world=w,
            goal_xy_m=[round(float(x),4) for x in target[w]],
            matched_first_lip_contacts=len(matches),
            first_lip_bilateral_verified=verified is not None,
            first_lip_actual_height_cm=(
                float(verified["signed_actual_height_cm"])
                if verified is not None else None),
            closest_to_goal_m=(
                round(closest,4) if closest is not None else None),
            stable_arrival=completed,
            arrival_frame=arrived_frame,
        ))
    return dict(
        protocol=dict(goal="60cm past first positive-X tile edge",
                      stable_frames=GOAL_STABLE_FRAMES,
                      origin="birth position, fixed for episode",
                      contact="independent first-life real-prism upper-top contact",
                      note="Not a guaranteed-height stair; true geometry measured."),
        completed_missions=sum(row["stable_arrival"] for row in rows),
        uphill_completed=sum(row["stable_arrival"] and
                             row["first_lip_actual_height_cm"] is not None and
                             row["first_lip_actual_height_cm"]>=.15
                             for row in rows),
        downhill_completed=sum(row["stable_arrival"] and
                               row["first_lip_actual_height_cm"] is not None and
                               row["first_lip_actual_height_cm"]<=-.15
                               for row in rows),
        physically_verified_bilateral=sum(
            row["first_lip_bilateral_verified"] for row in rows),
        worlds=rows,
    )
