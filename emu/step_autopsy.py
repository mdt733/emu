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
    """Independent v5052 goal: sequential real foot landings + stable arrival.

    The strict historical scorer separately asks for both feet together.
    A dynamically walking biped may land them at different times, so this
    task verifies FIVE consecutive genuine physical top-contact frames
    *per foot*, after the first signed lip, without requiring overlap.
    No Academy event/reward/completion flag is ever an input.
    """
    from .step_goal import (
        arrival_quality, destination_from_birth, GOAL_STABLE_FRAMES)
    birth=np.asarray(frames["position_xy"],float)
    xy=np.asarray(frames.get("physical_position_xy_after_step",
                             frames["position_xy"]),float)
    yaw=np.asarray(frames.get("physical_heading_rad_after_step",
                              frames["heading_rad"]),float)
    alive=first_episode_mask(frames["done"])
    fell=np.asarray(frames["fell"],bool)
    gravity=np.asarray(frames["terrain_gravity_z"],float)
    support=np.asarray(frames["terrain_support"],float)
    bad=np.asarray(frames["terrain_bad_contact"],float)
    slip=np.asarray(frames["stance_slip_proxy"],float)
    xyz=np.asarray(frames["foot_xyz"],float)
    corners=np.asarray(frames["carpet_prism_corner_z"],float)
    pairs=np.asarray(frames["carpet_raw_contact_pairs"],bool)
    vertical=np.asarray(frames["carpet_contact_vertical"],float)
    radii=np.asarray(frames["initial_foot_radii_m"],float)
    target=destination_from_birth(np,birth[0])
    origin=np.floor(birth[0]/TILE_M).astype(int)
    contacts=score_prism_crossings(frames,required_frames=5)["worlds"]
    rows=[]
    for w in range(xy.shape[1]):
        from_cell=origin[w].tolist()
        to_cell=[int(origin[w,0])+1,int(origin[w,1])]
        encounters=[e for e in contacts[w]["independently_supported_lips"]
                    if e["from_cell"]==from_cell and e["to_cell"]==to_cell
                    and abs(e["signed_height_cm"])>=.15]
        # Do not accept a later unrelated tile as the actual first step.
        first=encounters[0] if encounters else None
        slot=_slot(to_cell)
        seen=[False,False]
        contact_streak=[0,0]
        ready_frame=None
        arrived_frame=None
        arrive_streak=0
        if first is not None:
            for t in range(int(first["frame"]),len(xy)):
                if not alive[t,w] or fell[t,w]:
                    break
                for foot in range(2):
                    actual_top=_top(corners[t,w,slot],to_cell,
                                    xyz[t,w,foot,:2])
                    contact=bool(np.any(
                        pairs[t,w,2*slot:2*slot+2,foot] &
                        (vertical[t,w,2*slot:2*slot+2,foot]>=.70)))
                    placed=bool(
                        contact and actual_top is not None and
                        abs(xyz[t,w,foot,2]-radii[foot]-actual_top)<=.009)
                    contact_streak[foot]=(contact_streak[foot]+1
                                          if placed else 0)
                    seen[foot]=seen[foot] or contact_streak[foot]>=5
                if all(seen) and ready_frame is None:
                    ready_frame=t
                if ready_frame is None:
                    continue
                stable=bool(arrival_quality(
                    np,target[w:w+1],xy[t,w:w+1],
                    yaw[t,w:w+1]-yaw[0,w:w+1],
                    gravity[t,w:w+1],support[t,w:w+1],
                    bad[t,w:w+1],slip[t,w:w+1],
                    np.array([True]))[0])
                arrive_streak=arrive_streak+1 if stable else 0
                if arrive_streak>=GOAL_STABLE_FRAMES:
                    arrived_frame=t
                    break
        life=np.flatnonzero(alive[:,w])
        closest=(float(np.min(np.linalg.norm(xy[life,w]-target[w],axis=1)))
                 if len(life) else None)
        signed=float(first["signed_height_cm"]) if first is not None else None
        rows.append(dict(
            world=w,goal_xy_m=[round(float(x),4) for x in target[w]],
            matched_first_lip_contacts=len(encounters),
            first_lip_actual_height_cm=signed,
            first_lip_each_foot_verified=all(seen),
            independent_foot_contact_frames=5,
            first_lip_simultaneous_bilateral_verified=False,
            first_lip_sequential_ready_frame=ready_frame,
            closest_to_goal_m=(round(closest,4) if closest is not None else None),
            stable_arrival=arrived_frame is not None,
            arrival_frame=arrived_frame,
        ))
    # The original stricter bilateral test still runs as mission_autopsy.
    return dict(
        protocol=dict(goal="60cm after first +X physical tile lip",
                      stable_frames=GOAL_STABLE_FRAMES,
                      each_foot_contact_frames=5,
                      foot_support="independent consecutive top contacts per foot",
                      first_life_only=True,
                      note="No simultaneous double stance required. Measured "
                           "signed geometry, not a guaranteed-height riser."),
        completed_missions=sum(x["stable_arrival"] for x in rows),
        uphill_completed=sum(x["stable_arrival"] and
                             x["first_lip_actual_height_cm"] is not None and
                             x["first_lip_actual_height_cm"]>=.15
                             for x in rows),
        downhill_completed=sum(x["stable_arrival"] and
                               x["first_lip_actual_height_cm"] is not None and
                               x["first_lip_actual_height_cm"]<=-.15
                               for x in rows),
        physically_verified_sequential_feet=sum(
            x["first_lip_each_foot_verified"] for x in rows),
        worlds=rows,
    )
