"""Physical post-lip mission certification for the existing prism examiner.

No rewards, Academy flags, privileged actor inputs or additional simulation.
The caller supplies first-life, pre-step MJX traces and independently
confirmed, physically supported crossings from score_prism_crossings.
An upper-foot contact alone is NEVER counted as a complete climb.

A score here is a conservative observed mission, not proof of reachability.
Unconfirmed approaches require a separate attempt examiner; this module
does not manufacture failed attempts from missing crossing events.
"""
from __future__ import annotations

import math
import numpy as np


def _surface_at(tops, tile, xy, tile_m):
    """Real two-triangle prism top within an exact global source tile."""
    u, v = (np.asarray(xy) / tile_m - np.asarray(tile, dtype=float))
    if min(u, v) < -.001 or max(u, v) > 1.001:
        return None
    slot = int(tile[0] % 2 + 2 * (tile[1] % 2))
    a, b, c, d = tops[slot]
    if u + v <= 1.:
        return float(a * (1. - u - v) + b * u + c * v)
    return float(d * (u + v - 1.) + b * (1. - v) + c * (1. - u))


def score_post_lip_missions(frames, crossings, *,
                            tile_m=.75, advance_m=.50,
                            heading_tolerance_deg=30.,
                            lateral_tolerance_m=.25,
                            trailing_support_frames=3):
    """Examine recovery/continued travel AFTER independent upper support.

    Results are per *observed supported crossing*, not per approached wall.
    A complete event needs the OTHER foot to obtain sustained valid top
    support on the destination physical tile, followed by >= advance_m
    directed travel with acceptable heading and lateral deviation, before
    a fall/timeout. No future time steps after the first episode count.
    Only positive operator-forward trials qualify as course certificates.
    """
    xy = np.asarray(frames["position_xy"], dtype=float)
    yaw = np.asarray(frames["heading_rad"], dtype=float)
    feet = np.asarray(frames["foot_xyz"], dtype=float)
    tops = np.asarray(frames["carpet_prism_corner_z"], dtype=float)
    pairs = np.asarray(frames["carpet_raw_contact_pairs"], dtype=bool)
    vertical = np.asarray(frames["carpet_contact_vertical"], dtype=float)
    radii = np.asarray(frames["initial_foot_radii_m"], dtype=float)
    done = np.asarray(frames["done"], dtype=bool)
    fell = np.asarray(frames["fell"], dtype=bool)
    t_count, worlds, _ = xy.shape
    if (xy.shape != (t_count, worlds, 2)
            or yaw.shape != (t_count, worlds)
            or feet.shape != (t_count, worlds, 2, 3)
            or tops.shape != (t_count, worlds, 4, 4)
            or pairs.shape != (t_count, worlds, 8, 2)
            or vertical.shape != pairs.shape
            or radii.shape != (2,)
            or done.shape != (t_count, worlds)
            or fell.shape != done.shape
            or len(crossings["worlds"]) != worlds):
        raise ValueError("Incompatible physical prism mission trace shapes")
    if not (np.isfinite(xy).all() and np.isfinite(yaw).all()
            and np.isfinite(feet).all() and np.isfinite(tops).all()
            and np.isfinite(vertical).all() and np.isfinite(radii).all()):
        raise ValueError("Nonfinite physical mission trace")
    if not (tile_m > 0 and advance_m > 0 and
            0 < heading_tolerance_deg <= 180 and lateral_tolerance_m > 0
            and 1 <= trailing_support_frames <= 20):
        raise ValueError("Invalid physical mission thresholds")

    earlier_done = np.concatenate(
        (np.zeros((1, worlds), dtype=np.int32),
         np.cumsum(done[:-1], axis=0)), axis=0)
    first_life = earlier_done == 0
    reports = []
    for w in range(worlds):
        # All ordinary panels request constant forward speed and zero yaw.
        # Record the initial BODY heading as the intended world direction;
        # this is conservative if the actor turns off-course before a lip.
        ref = float(yaw[0, w])
        forward = np.array([math.cos(ref), math.sin(ref)])
        sideways = np.array([-forward[1], forward[0]])
        world_events = []
        for crossing in crossings["worlds"][w]["independently_supported_lips"]:
            t0 = int(crossing["frame"])
            lead = int(crossing["leading_foot"])
            if not (0 <= t0 < t_count and lead in (0, 1)):
                raise ValueError("Invalid crossing frame or leading foot")
            other = 1 - lead
            tile = tuple(int(x) for x in crossing["to_cell"])
            slot = int(tile[0] % 2 + 2 * (tile[1] % 2))
            streak = 0
            recovered_at = None
            completed_at = None
            progressed_at = None
            max_directed = 0.
            ending_heading_error = None
            ending_lateral = None
            last_t = t0
            for t in range(t0, t_count):
                if not first_life[t, w] or fell[t, w]:
                    break
                last_t = t
                # Recycled prisms reuse slot numbers. Require the other
                # sphere to lie above the EXACT destination global tile.
                surface = _surface_at(tops[t, w], tile,
                                      feet[t, w, other, :2], tile_m)
                top_contact = (surface is not None and
                    abs(float(feet[t, w, other, 2] - radii[other] - surface)) <= .009 and
                    any(pairs[t, w, 2 * slot + tri, other] and
                        vertical[t, w, 2 * slot + tri, other] >= .70
                        for tri in (0, 1)))
                streak = streak + 1 if top_contact else 0
                if streak >= trailing_support_frames and recovered_at is None:
                    recovered_at = t
                offset = xy[t, w] - xy[t0, w]
                progress = float(offset @ forward)
                max_directed = max(max_directed, progress)
                if progress < advance_m or progressed_at is not None:
                    continue
                progressed_at = t
                heading_error = abs(math.degrees(math.atan2(
                    math.sin(float(yaw[t, w]) - ref),
                    math.cos(float(yaw[t, w]) - ref))))
                lateral = abs(float(offset @ sideways))
                ending_heading_error = round(heading_error, 3)
                ending_lateral = round(lateral, 4)
                # A recovery after progress is permitted only if a LATER
                # frame still demonstrates controlled forward progress.
                if (recovered_at is not None and recovered_at <= t and
                    heading_error <= heading_tolerance_deg and
                    lateral <= lateral_tolerance_m):
                    completed_at = t
                    break
            if completed_at is None and recovered_at is not None:
                # If the rear foot landed only AFTER passing the distance
                # mark, allow later qualifying frames to complete the mission.
                for t in range(recovered_at, last_t + 1):
                    offset = xy[t, w] - xy[t0, w]
                    progress = float(offset @ forward)
                    heading_error = abs(math.degrees(math.atan2(
                        math.sin(float(yaw[t, w]) - ref),
                        math.cos(float(yaw[t, w]) - ref))))
                    lateral = abs(float(offset @ sideways))
                    if (progress >= advance_m and
                        heading_error <= heading_tolerance_deg and
                        lateral <= lateral_tolerance_m):
                        completed_at = t
                        ending_heading_error = round(heading_error, 3)
                        ending_lateral = round(lateral, 4)
                        break
            fell_after = bool(np.any(fell[t0:last_t + 2, w]))
            if completed_at is not None:
                outcome = "complete_controlled_50cm"
            elif fell_after:
                outcome = "fell_after_upper_support"
            elif recovered_at is None:
                outcome = "upper_support_without_trailing_recovery"
            elif max_directed < advance_m:
                outcome = "trailing_recovered_without_50cm_progress"
            else:
                outcome = "distance_reached_but_path_control_failed"
            world_events.append({
                **crossing,
                "trailing_foot": other,
                "trailing_support_frame": recovered_at,
                "first_50cm_frame": progressed_at,
                "controlled_completion_frame": completed_at,
                "max_directed_progress_m": round(max_directed, 4),
                "heading_error_at_first_50cm_deg": ending_heading_error,
                "lateral_error_at_first_50cm_m": ending_lateral,
                "fell_after_upper_support": fell_after,
                "outcome": outcome,
            })
        reports.append({"world": w, "supported_event_count": len(world_events),
                        "completed": sum(x["controlled_completion_frame"] is not None
                                         for x in world_events),
                        "missions": world_events})
    return {
        "thresholds": {"directed_progress_m": advance_m,
                       "heading_error_deg": heading_tolerance_deg,
                       "lateral_error_m": lateral_tolerance_m,
                       "trailing_top_support_frames": trailing_support_frames},
        "supported_events": sum(x["supported_event_count"] for x in reports),
        "completed": sum(x["completed"] for x in reports),
        "worlds": reports,
    }
