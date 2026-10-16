"""Read-only v5016 footfall and uninterrupted-episode examination.

Input contacts are actual simulator floor-to-foot collision-pair proximity
flags, not old foot-height support estimates. They do not prove load-bearing
normal force: touchdown-derived strides remain a *conservative estimate*.
No training reward, curriculum or old checkpoint schema depends on this module.
"""
from __future__ import annotations

import numpy as np

from .directional import examine_course, incline_from_tilted_gravity


def contact_pair_mask(xp, contact_geom1, contact_geom2, contact_distance,
                      floor_geom: int, foot_geoms, *, tolerance_m=0.002):
    """[worlds, contacts] -> [worlds, two feet] active floor/foot geometry pairs.

    MJX pads some contact slots; a nonnegative near-contact distance may still
    indicate a geometric pair, not confirmed impulse. Use the conservative
    near-zero distance threshold and report it as contact-pair evidence.
    """
    result = []
    for foot in foot_geoms:
        pair = (((contact_geom1 == floor_geom) & (contact_geom2 == foot)) |
                ((contact_geom2 == floor_geom) & (contact_geom1 == foot)))
        result.append(xp.any(pair & (contact_distance <= tolerance_m), axis=-1))
    return xp.stack(result, axis=-1)


def count_contact_strides(body_xy, feet_xyz, contacts, *, settle_frames=0,
                          min_air_frames=3, min_swing_m=.012,
                          min_clearance_m=.004, min_body_progress_m=.025):
    """Conservative alternating touchdown pair count, NOT foot-lift count.

    Lift -> >=3 airborne frames with clearance -> touchdown, with horizontal
    foot relocation and torso displacement, must alternate feet. A completed
    pair counts once. Repeated tapping in place and sliding without airborne
    intervals cannot qualify. Course obedience is graded separately.
    """
    body = np.asarray(body_xy, dtype=float)
    feet = np.asarray(feet_xyz, dtype=float)
    ct = np.asarray(contacts, dtype=bool)
    n = len(body)
    if body.shape != (n, 2) or feet.shape != (n, 2, 3) or ct.shape != (n, 2) or n < 2:
        raise ValueError('expected [frames,2] root, [frames,2,3] feet and [frames,2] contacts')
    if not np.isfinite(body).all() or not np.isfinite(feet).all():
        raise ValueError('nonfinite foot trajectory')
    if not 0 <= settle_frames < n:
        raise ValueError('invalid settle window')
    swing_start = [None, None]
    max_clearance = [0.0, 0.0]
    air_frames = [0, 0]
    qualified = []
    # Use the actual foot-ball radius from robot.xml. Foot center at 0.018 m
    # is touching flat plane at z=0; keep this fact explicit and test it.
    for k in range(max(1, settle_frames), n):
        for foot in (0, 1):
            if ct[k-1, foot] and not ct[k, foot]:
                swing_start[foot] = k-1
                air_frames[foot] = 1
                max_clearance[foot] = max(0.0, feet[k, foot, 2] - .018)
            elif not ct[k, foot] and swing_start[foot] is not None:
                air_frames[foot] += 1
                max_clearance[foot] = max(max_clearance[foot],
                                          float(feet[k, foot, 2] - .018))
            elif not ct[k-1, foot] and ct[k, foot]:
                start = swing_start[foot]
                if start is not None and air_frames[foot] >= min_air_frames:
                    foot_travel = np.linalg.norm(feet[k, foot, :2] - feet[start, foot, :2])
                    body_travel = np.linalg.norm(body[k] - body[start])
                    if (foot_travel >= min_swing_m and
                            body_travel >= min_body_progress_m and
                            max_clearance[foot] >= min_clearance_m):
                        qualified.append((k, foot))
                swing_start[foot] = None
                air_frames[foot] = 0
                max_clearance[foot] = 0.0
    pairs = 0
    last = None
    for time, foot in qualified:
        if last is not None and foot != last[1]:
            if np.linalg.norm(body[time] - body[last[0]]) >= min_body_progress_m:
                pairs += 1
                last = None
                continue
        last = (time, foot)
    return pairs


def evaluate_trace_arrays(trace, gravity_world, *, dt=.02, settle_frames=25):
    """Reduce bounded, real-MJX pre-reset frame samples into per-world verdicts.

    trace contains [horizon, worlds, ...] pre-step poses/contacts/commands and
    [worlds,...] final pose. Any done truncates *before* its auto-reset. A
    premature timeout also invalidates full-horizon survival.
    """
    keys = ('position_xy','heading_rad','foot_xyz','floor_contact',
            'command','done','fell')
    arrays = {key: np.asarray(trace[key]) for key in keys}
    horizon, worlds, _ = arrays['position_xy'].shape
    if horizon < 2 or arrays['command'].shape != (horizon, worlds, 3):
        raise ValueError('incorrect trajectory dimensions')
    g = np.asarray(gravity_world, dtype=float)
    if g.shape != (worlds, 3):
        raise ValueError('gravity vector count must equal number of worlds')
    findings = []
    for w in range(worlds):
        term = np.flatnonzero(arrays['done'][:, w] > .5)
        stop = int(term[0]) if len(term) else horizon
        failed = bool(np.any(arrays['fell'][:stop+1, w] > .5)) if len(term) else False
        result = {'world': w, 'frames_recorded': stop,
                  'first_done_step': (stop+1) if len(term) else None,
                  'first_fall': failed, 'full_horizon_survival': len(term)==0}
        if stop <= settle_frames + 2:
            result.update(course_pass=False, walking_pass=False, contact_stride_pairs=0,
                          invalid_reason='insufficient uninterrupted frames')
            findings.append(result)
            continue
        # Record positions from the old state *before* env.step auto-resets it.
        if len(term):
            final_pos = arrays['position_xy'][stop,w]
            final_heading = arrays['heading_rad'][stop,w]
            final_feet = arrays['foot_xyz'][stop,w]
            final_contacts = arrays['floor_contact'][stop,w]
        else:
            final_pos = np.asarray(trace['last_position_xy'])[w]
            final_heading = np.asarray(trace['last_heading_rad'])[w]
            final_feet = np.asarray(trace['last_foot_xyz'])[w]
            final_contacts = np.asarray(trace['last_floor_contact'])[w]
        pos = np.concatenate((arrays['position_xy'][:stop,w],
                              final_pos[None]),axis=0)
        head = np.concatenate((arrays['heading_rad'][:stop,w],
                               np.asarray([final_heading])),axis=0)
        feet = np.concatenate((arrays['foot_xyz'][:stop,w],
                               final_feet[None]),axis=0)
        contacts = np.concatenate((arrays['floor_contact'][:stop,w],
                                   final_contacts[None]),axis=0)
        commands = arrays['command'][:stop,w]
        slope, uphill = incline_from_tilted_gravity(g[w])
        strides = count_contact_strides(pos,feet,contacts,settle_frames=settle_frames)
        # The grading function rejects early terminations. The stride count
        # stays reported but cannot override an invalid course or fall.
        course = examine_course(pos,head,commands,dt=dt,
                    settle_frames=settle_frames, fell=bool(len(term)),
                    completed_strides=strides,
                    uphill_world_xy=uphill,slope_deg=slope)
        result.update(course)
        if 'issued_yaw_rate' in trace:
            issued = np.asarray(trace['issued_yaw_rate'])[:stop, w]
            operator_yaw = arrays['command'][:stop, w, 2]
            delta = issued - operator_yaw
            result['mean_abs_controller_correction_rad_s'] = float(
                np.mean(np.abs(delta[settle_frames:])))
            result['peak_abs_controller_correction_rad_s'] = float(
                np.max(np.abs(delta[settle_frames:])))
            result['issued_yaw_mae_vs_operator_rad_s'] = float(
                np.mean(np.abs(delta[settle_frames:])))
        if 'estimated_heading_rad' in trace:
            estimated = np.asarray(trace['estimated_heading_rad'])[:stop, w]
            true_relative = np.unwrap(head[:-1]) - head[0]
            # Ideal simulated body gyro is not identical to world yaw rate
            # when body roll/pitch are nonzero. Keep this as a diagnostic.
            result['gyro_heading_error_mae_deg'] = float(np.degrees(
                np.mean(np.abs(estimated[settle_frames:] -
                               true_relative[settle_frames:]))))
        result.update(contact_stride_pairs=strides,measured_slope_deg=slope,
                      invalid_reason=('terminated before horizon' if len(term) else None))
        findings.append(result)
    return findings
