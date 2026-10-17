"""v5016 directional-control oracle and optional *outer* heading controller.

Pure CPU geometry: no MuJoCo, PPO or checkpoint dependence. The world-frame
examiner does not grant course credit by following the robot's current heading.
Callers must supply one uninterrupted episode (NEVER concatenate auto-resets).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def wrap_radians(angle):
    """Shortest signed angle, including correct ±pi boundary handling."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def heading_rate_request(xp, requested_rate, target_heading, estimated_heading,
                         *, kp, max_rate):
    """Shared NumPy/JAX P law; caller owns heading integration and sensor validity.

    This does not act on servos. It returns a bounded *policy yaw command*
    and the shortest signed target-vs-estimate heading error.
    """
    error = xp.arctan2(xp.sin(target_heading - estimated_heading),
                      xp.cos(target_heading - estimated_heading))
    return xp.clip(requested_rate + kp * error, -max_rate, max_rate), error


@dataclass(frozen=True)
class HeadingState:
    target_rad: float = 0.0  # unwrapped, integrated operator command
    has_reference: bool = False


def heading_hold_step(
    state: HeadingState, measured_yaw_rad: float, requested_yaw_rate: float,
    *, dt: float = 0.02, kp: float = 0.75, max_rate: float = 0.9,
    measurement_valid: bool = True,
) -> tuple[HeadingState, float, float]:
    """Integrate requested yaw; use P feedback to correct *heading* drift.

    Returned yaw rate is a requested command to an existing walking policy,
    never a direct motor action. A missing/invalid attitude estimate disables
    feedback and re-anchors on the next valid sample to prevent a kick.
    """
    if not all(math.isfinite(v) for v in (dt, kp, max_rate)) or dt <= 0 or kp < 0 or max_rate <= 0:
        raise ValueError('heading controller requires finite positive dt/rate and nonnegative gain')
    cmd = float(requested_yaw_rate)
    if not math.isfinite(cmd):
        raise ValueError('nonfinite operator yaw command')
    cmd = max(-max_rate, min(max_rate, cmd))
    valid = bool(measurement_valid) and math.isfinite(float(measured_yaw_rad))
    if not valid:
        return HeadingState(state.target_rad, False), cmd, 0.0
    measured = float(measured_yaw_rad)
    base = state.target_rad if state.has_reference else measured
    target = base + cmd * dt
    corrected, error = heading_rate_request(
        np, cmd, target, measured, kp=kp, max_rate=max_rate)
    corrected, error = float(corrected), float(error)
    return HeadingState(target, True), corrected, error


def incline_from_tilted_gravity(gravity_world_xyz):
    """Return (angle_deg, uphill_unit_xy) for a flat XY plane with tilted gravity.

    Only appropriate for the current gravity-equivalent infinite planar slope.
    The uphill unit is WORLD-fixed, independent of the robot's yaw. For flat
    ground, the uphill bearing is undefined, and returned as None.
    """
    g = np.asarray(gravity_world_xyz, dtype=np.float64)
    if g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) < 1e-9 or g[2] >= 0:
        raise ValueError('gravity must be finite, downward and three-dimensional')
    horizontal = np.linalg.norm(g[:2])
    angle = math.degrees(math.atan2(horizontal, -g[2]))
    if horizontal < np.linalg.norm(g) * 1e-8:
        return 0.0, None
    return angle, -g[:2] / horizontal


def _rolling_max_abs_error(values, frames):
    """Worst complete-window MEAN ABSOLUTE error: signs cannot cancel."""
    if len(values) == 0:
        return None
    width = max(1, min(int(frames), len(values)))
    return float(np.max(np.convolve(np.abs(values), np.ones(width) / width, mode='valid')))


@dataclass(frozen=True)
class CourseLimits:
    min_progress_ratio: float = 0.60
    max_heading_mae_deg: float = 20.0
    max_yaw_rate_mae_rad_s: float = 0.30
    max_path_rmse_m: float = 0.12
    max_idle_drift_m: float = 0.04
    min_completed_strides: int = 2


def examine_course(
    positions_xy, heading_rad, commands, *,
    dt: float = 0.02, settle_frames: int = 0,
    fell: bool = False, reset_flags=None, completed_strides: int | None = None,
    uphill_world_xy=None, slope_deg: float = 0.0,
    limits: CourseLimits = CourseLimits(),
):
    """Evaluate a commanded trajectory against a WORLD-FIXED requested course.

    commands[i] = (body-vx, body-vy, yaw-rate) issued during step i.
    positions/headings have N+1 pre-reset states, commands N rows.
    The desired course integrates commands in the DESIRED heading frame,
    never the robot's (potentially cheated) actual heading.

    course_pass is a directional/command test. walking_pass additionally
    requires real separately-verified complete strides; a foot-lift proxy
    MUST NOT be passed as completed_strides.
    """
    pos = np.asarray(positions_xy, dtype=np.float64)
    heading = np.asarray(heading_rad, dtype=np.float64)
    cmd = np.asarray(commands, dtype=np.float64)
    if cmd.ndim != 2 or cmd.shape[1] != 3 or not len(cmd):
        raise ValueError('commands must be nonempty [steps,3]')
    steps = len(cmd)
    if pos.shape != (steps + 1, 2) or heading.shape != (steps + 1,):
        raise ValueError('positions and heading must contain exactly N+1 pre-reset states')
    if not np.isfinite(pos).all() or not np.isfinite(heading).all() or not np.isfinite(cmd).all():
        raise ValueError('nonfinite trajectory or commands cannot be scored')
    if not math.isfinite(dt) or dt <= 0 or not (0 <= settle_frames < steps):
        raise ValueError('invalid dt/settle window')
    if not math.isfinite(slope_deg) or not (0 <= slope_deg < 89):
        raise ValueError('invalid nonnegative slope magnitude')
    if reset_flags is not None and np.any(np.asarray(reset_flags)):
        raise ValueError('reset or teleport within an episode: split trajectories first')
    if completed_strides is not None and (not isinstance(completed_strides, int) or completed_strides < 0):
        raise ValueError('completed_strides must be independently measured nonnegative integer')

    gradient = None
    if uphill_world_xy is not None:
        gradient = np.asarray(uphill_world_xy, dtype=np.float64)
        if gradient.shape != (2,) or not np.isfinite(gradient).all() or np.linalg.norm(gradient) < 1e-8:
            raise ValueError('invalid uphill basis')
        gradient = gradient / np.linalg.norm(gradient)
    if slope_deg > 1e-7 and gradient is None:
        raise ValueError('slope trials require a world-fixed uphill unit')

    # Heading reference follows the operator yaw-rate integrator, not actual yaw.
    desired_yaw = heading[0] + np.concatenate(([0.0], np.cumsum(cmd[:, 2] * dt)))
    ct, st = np.cos(desired_yaw[:-1]), np.sin(desired_yaw[:-1])
    planned_step = dt * np.column_stack((
        ct * cmd[:, 0] - st * cmd[:, 1],
        st * cmd[:, 0] + ct * cmd[:, 1],
    ))
    actual_step = np.diff(pos, axis=0)
    i = int(settle_frames)
    planned = planned_step[i:]
    actual = actual_step[i:]
    intended_dist = float(np.linalg.norm(planned, axis=1).sum())
    active = np.linalg.norm(planned, axis=1) > 1e-9
    if np.any(active):
        tangents = planned[active] / np.linalg.norm(planned[active], axis=1)[:, None]
        signed_course_m = float(np.sum(np.einsum('ij,ij->i', actual[active], tangents)))
        ratio = signed_course_m / intended_dist
    else:
        signed_course_m, ratio = 0.0, None

    ideal_pos = pos[i] + np.vstack((np.zeros((1, 2)), np.cumsum(planned, axis=0)))
    path_rmse = float(np.sqrt(np.mean(np.sum((pos[i:] - ideal_pos) ** 2, axis=1))))
    idle_drift = float(np.max(np.linalg.norm(pos[i:] - pos[i], axis=1)))
    heading_error = np.arctan2(np.sin(heading[i:] - desired_yaw[i:]),
                              np.cos(heading[i:] - desired_yaw[i:]))
    heading_mae = math.degrees(float(np.mean(np.abs(heading_error))))
    heading_p95 = math.degrees(float(np.percentile(np.abs(heading_error), 95)))
    observed_yaw_rate = np.arctan2(np.sin(np.diff(heading)),
                                  np.cos(np.diff(heading))) / dt
    yaw_error = observed_yaw_rate[i:] - cmd[i:, 2]
    yaw_mae = float(np.mean(np.abs(yaw_error)))
    net = pos[-1] - pos[i]
    uphill_progress = float(np.dot(net, gradient)) if gradient is not None else None
    planned_uphill = float(np.dot(planned.sum(axis=0), gradient)) if gradient is not None else None
    altitude_equivalent = uphill_progress * math.sin(math.radians(slope_deg)) if gradient is not None else None

    # Uphill qualification only when uphill is requested; downhill and
    # cross-slope commands must be evaluated against THEIR requested course.
    climbs_requested = planned_uphill is not None and planned_uphill > 0.03
    slope_obeyed = not climbs_requested or (
        uphill_progress is not None and uphill_progress >= 0.60 * planned_uphill)
    moving_obeyed = ((ratio is not None and ratio >= limits.min_progress_ratio)
                     if intended_dist > 1e-8 else idle_drift <= limits.max_idle_drift_m)
    course_pass = bool(not fell and slope_obeyed and moving_obeyed
                       and path_rmse <= limits.max_path_rmse_m
                       and heading_mae <= limits.max_heading_mae_deg
                       and yaw_mae <= limits.max_yaw_rate_mae_rad_s)
    gait_verified = completed_strides is not None and completed_strides >= limits.min_completed_strides
    return {
        'course_pass': course_pass,
        'walking_pass': bool(course_pass and gait_verified),
        'gait_verified': bool(gait_verified),
        'requested_distance_m': intended_dist,
        'signed_course_progress_m': signed_course_m,
        'progress_ratio': ratio,
        'path_rmse_m': path_rmse,
        'idle_drift_m': idle_drift,
        'heading_mae_deg': heading_mae,
        'heading_p95_deg': heading_p95,
        'yaw_rate_mae_rad_s': yaw_mae,
        'yaw_rate_signed_mean_error': float(np.mean(yaw_error)),
        'yaw_abs_error_1s': _rolling_max_abs_error(yaw_error, round(1 / dt)),
        'yaw_abs_error_3s': _rolling_max_abs_error(yaw_error, round(3 / dt)),
        'uphill_progress_m': uphill_progress,
        'requested_uphill_progress_m': planned_uphill,
        'equivalent_elevation_m': altitude_equivalent,
        'fell': bool(fell),
        'completed_strides': completed_strides,
        'steps_scored': steps - i,
    }

def examine_command_transitions(positions_xy, heading_rad, commands, *,
                                dt=0.02, settle_frames=0, stable_frames=3):
    """Independently measure acquisition/braking/reversal response times.

    Motion is projected into the *requested* world-fixed course frame, not
    redefined by where the robot actually turned. Response is reached only
    after three consecutive control frames. None means not observed before
    the next command change/episode end; it does NOT mean instant success.
    """
    pos=np.asarray(positions_xy,dtype=float)
    head=np.asarray(heading_rad,dtype=float)
    cmd=np.asarray(commands,dtype=float)
    n=len(cmd)
    if n < 2 or pos.shape != (n+1,2) or head.shape != (n+1,) or cmd.shape != (n,3):
        raise ValueError('transition exam requires N+1 poses, N three-axis commands')
    if not np.isfinite(pos).all() or not np.isfinite(head).all() or not np.isfinite(cmd).all():
        raise ValueError('nonfinite transition trace')
    if not np.isfinite(dt) or dt <= 0 or not 0 <= settle_frames < n:
        raise ValueError('invalid transition time/window')
    if stable_frames < 1:
        raise ValueError('positive stable frame count required')
    planned_yaw=head[0]+np.concatenate(([0.],np.cumsum(cmd[:,2]*dt)))
    c,s=np.cos(planned_yaw[:-1]),np.sin(planned_yaw[:-1])
    requested=np.column_stack((c*cmd[:,0]-s*cmd[:,1],
                               s*cmd[:,0]+c*cmd[:,1]))
    observed=np.diff(pos,axis=0)/dt
    norm=np.linalg.norm(requested,axis=1)
    transitions=[]
    boundary=np.flatnonzero(np.any(np.abs(np.diff(cmd[:,:2],axis=0)) > 1e-6,axis=1))+1
    for idx in boundary:
        if idx < settle_frames:
            continue
        prev,now=norm[idx-1],norm[idx]
        if now < .008 and prev >= .025:
            kind='brake_to_stop'
        elif prev < .008 and now >= .025:
            kind='rest_to_walk'
        elif prev >= .025 and now >= .025 and np.dot(requested[idx-1],requested[idx]) < -.50*prev*now:
            kind='reverse_or_redirect'
        elif now > prev*1.25 and now >= .025:
            kind='accelerate'
        elif now < prev*.8:
            kind='decelerate'
        else:
            kind='command_change'
        later=boundary[boundary>idx]
        end=int(later[0]) if len(later) else n
        end=min(end,idx+round(2/dt))
        response=None
        for j in range(idx,end-stable_frames+1):
            interval=observed[j:j+stable_frames]
            if now < .008:
                reached=np.all(np.linalg.norm(interval,axis=1)<=max(.015,.2*prev))
            else:
                direction=requested[idx]/now
                reached=np.all(interval @ direction >= .7*now)
            if reached:
                response=float((j-idx)*dt)
                break
        row=dict(start_step=int(idx),kind=kind,
                 target_speed_mps=float(now),
                 response_latency_s=response,
                 response_observed=response is not None)
        if now < .008:
            last=min(n,idx+max(1,round(.5/dt)))
            row['braking_path_first_0p5s_m']=float(
                np.sum(np.linalg.norm(np.diff(pos[idx:last+1],axis=0),axis=1)))
        transitions.append(row)
    timings=[r['response_latency_s'] for r in transitions if r['response_observed']]
    braking=[r['braking_path_first_0p5s_m'] for r in transitions
             if 'braking_path_first_0p5s_m' in r]
    return {'maneuvers':transitions,
            'maneuver_count':len(transitions),
            'maneuver_response_observed':len(timings),
            'maneuver_mean_response_s':float(np.mean(timings)) if timings else None,
            'maneuver_unobserved_count':len(transitions)-len(timings),
            'braking_path_first_0p5s_m':float(np.mean(braking)) if braking else None}


# Contact-pair footfalls, signed course progress, one episode at a time.
import numpy as np



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


def examine_actuator_trace(trace, world, *, start, stop, dt, terminal):
    """Read-only per-servo load and work evidence from the existing MJX trace.

    Torque is actual MuJoCo actuator output, not motor current. A squared
    relative-torque integral is an UNCALIBRATED heating-exposure proxy.
    It is neither temperature nor a claimed safe duty limit.
    """
    required = ('servo_torque_nm', 'servo_joint_rate_rad_s',
                'servo_peak_limit_nm', 'servo_shaft_abs_power_w',
                'servo_shaft_drive_power_w', 'servo_shaft_brake_power_w',
                'spring_potential_energy_j', 'shin_spring_energy_j')
    if not all(k in trace for k in required):
        return None
    # Pre-step physics at index k reflects the preceding control interval.
    # At the first terminal sample the state is still pre-reset; at later
    # samples it belongs to a fresh episode and must never be included.
    torque = np.asarray(trace['servo_torque_nm'], dtype=float)
    speed = np.asarray(trace['servo_joint_rate_rad_s'], dtype=float)
    final_index = stop + 1 if terminal else len(torque)
    lo = max(1, start + 1)
    tau = torque[lo:final_index, world, :]
    omega = speed[lo:final_index, world, :]
    if not terminal and 'last_servo_torque_nm' in trace and 'last_servo_joint_rate_rad_s' in trace:
        # Only unbroken survivors can use the final state; a terminal
        # env.step has already auto-reset its state by this point.
        tau = np.concatenate((tau,
            np.asarray(trace['last_servo_torque_nm'], dtype=float)[world:world+1]), axis=0)
        omega = np.concatenate((omega,
            np.asarray(trace['last_servo_joint_rate_rad_s'], dtype=float)[world:world+1]), axis=0)
    if not len(tau):
        return None
    limit = np.asarray(trace['servo_peak_limit_nm'], dtype=float)
    ceiling = limit[world] if limit.ndim == 2 else limit
    if tau.shape[1] != 6 or omega.shape != tau.shape or np.asarray(ceiling).shape != (6,):
        raise ValueError('physical exam expected six aligned servo channels')
    if not np.isfinite(tau).all() or not np.isfinite(omega).all() or not np.isfinite(ceiling).all() or np.any(ceiling <= 0):
        raise ValueError('nonfinite or invalid actuator measurements')
    rating = 0.638  # nominal STS3215 rating; hardware confirmation outstanding
    power = tau * omega
    positive = np.maximum(power, 0)
    negative = np.maximum(-power, 0)
    over = np.abs(tau) > rating
    duration = len(tau) * dt
    return {
        'actuator_sample_frames': int(len(tau)),
        'actuator_sample_duration_s': float(duration),
        'servo_peak_abs_torque_nm': np.max(np.abs(tau), axis=0).tolist(),
        'servo_p95_abs_torque_nm': np.percentile(np.abs(tau), 95, axis=0).tolist(),
        'servo_over_continuous_fraction': np.mean(over, axis=0).tolist(),
        'servo_over_continuous_seconds': (np.sum(over, axis=0) * dt).tolist(),
        'servo_peak_limit_utilization': np.max(np.abs(tau) / ceiling, axis=0).tolist(),
        'servo_thermal_exposure_proxy_s': (np.sum((tau / rating) ** 2, axis=0) * dt).tolist(),
        'servo_positive_shaft_work_j': (np.sum(positive, axis=0) * dt).tolist(),
        'servo_braking_shaft_work_j': (np.sum(negative, axis=0) * dt).tolist(),
        'servo_measured_interval_abs_shaft_work_j': float(np.sum(np.abs(power)) * dt),
        'servo_all_sample_over_continuous_fraction': float(np.mean(over)),
        'thermal_proxy_definition': 'integral (shaft torque / 0.638 Nm)^2 dt, uncalibrated; no cooling or temperature model',
        'sampling_caveat': ('one-frame delayed pre-step force/velocity; '
                            'terminal control step excluded from the scored path; '
                            'final state appended for uninterrupted survivors'),
    }


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
        result.update(examine_command_transitions(
            pos,head,commands,dt=dt,settle_frames=settle_frames))
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
        physical = examine_actuator_trace(
            trace, w, start=settle_frames, stop=stop, dt=dt,
            terminal=bool(len(term)))
        if physical is not None:
            # Use the already captured POST-step, PRE-reset simulator metrics
            # for the actual scored control intervals (including termination).
            first_step = int(settle_frames)
            # Course scoring stops at the pre-reset pose recorded on the
            # terminating frame, so its actual terminating transition is
            # excluded. Integrate only the same scored control intervals.
            last_step = stop if len(term) else horizon
            def series(name):
                return np.asarray(trace[name], dtype=float)[first_step:last_step, w]
            if last_step > first_step:
                watts = series('servo_shaft_abs_power_w')
                drive = series('servo_shaft_drive_power_w')
                brake = series('servo_shaft_brake_power_w')
                stored = series('spring_potential_energy_j')
                shin = series('shin_spring_energy_j')
                if all(np.isfinite(x).all() for x in (watts, drive, brake, stored, shin)):
                    physical['step_integrated_abs_shaft_work_j'] = float(np.sum(watts) * dt)
                    physical['step_integrated_drive_shaft_work_j'] = float(np.sum(drive) * dt)
                    physical['step_integrated_brake_shaft_work_j'] = float(np.sum(brake) * dt)
                    physical['spring_potential_mean_j'] = float(np.mean(stored))
                    physical['spring_potential_range_j'] = float(np.ptp(stored))
                    physical['shin_spring_potential_mean_j'] = float(np.mean(shin))
                    physical['spring_return_not_inferred'] = True
            # The scored foot path is a contact-conditioned geometric slip
            # proxy, not a force-plate measurement of sole friction.
            scored_feet = feet[settle_frames:]
            scored_contacts = contacts[settle_frames:]
            foot_delta = np.linalg.norm(np.diff(scored_feet[:, :, :2], axis=0), axis=2)
            grounded = scored_contacts[:-1] & scored_contacts[1:]
            if np.any(grounded):
                physical['contact_pair_slip_path_m'] = float(np.sum(foot_delta[grounded]))
                physical['contact_pair_slip_speed_mean_m_s'] = float(np.mean(foot_delta[grounded] / dt))
                physical['contact_pair_sample_count'] = int(np.sum(grounded))
            else:
                physical['contact_pair_slip_path_m'] = None
                physical['contact_pair_slip_speed_mean_m_s'] = None
                physical['contact_pair_sample_count'] = 0
            result['physical_evidence'] = physical
            # Mechanical positive work normalized by observed signed forward
            # progress is meaningful only for a genuinely progressing course.
            progress = course['signed_course_progress_m']
            # Complete scored-interval aggregate work is preferred to
            # per-servo samples when calculating the work/progress ratio.
            # Negative, zero or trivial progress must never look efficient.
            drive_j = physical.get('step_integrated_drive_shaft_work_j')
            abs_j = physical.get('step_integrated_abs_shaft_work_j')
            result['positive_shaft_work_per_progress_m_j_m'] = (
                float(drive_j / progress)
                if progress > 0.01 and drive_j is not None else None)
            result['absolute_shaft_work_per_progress_m_j_m'] = (
                float(abs_j / progress)
                if progress > 0.01 and abs_j is not None else None)
        result.update(contact_stride_pairs=strides,measured_slope_deg=slope,
                      invalid_reason=('terminated before horizon' if len(term) else None))
        findings.append(result)
    return findings


# Optional three-ray ground-plane experiment (read-only geometry).
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GroundPlane:
    valid: bool
    reason: str
    ground_normal_body: tuple[float, float, float] | None = None
    slope_deg: float | None = None
    uphill_body: tuple[float, float, float] | None = None
    triangle_area_m2: float = 0.0


def estimate_ground_plane(
    origins_body, directions_body, ranges_m, gravity_body,
    *, min_area_m2: float = 0.0001, max_range_m: float = 0.6,
) -> GroundPlane:
    """Intersect three calibrated range beams with the ground and fit a plane.

    Normal points *away* from the ground relative to measured gravity.
    The uphill vector lies within the estimated ground plane. Neither a
    chassis pitch change on level terrain nor yaw should masquerade as a
    ground slope. Invalid samples are explicitly rejected; never fill them
    with a plausible zero-degree measurement.
    """
    origin = np.asarray(origins_body, dtype=np.float64)
    direction = np.asarray(directions_body, dtype=np.float64)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    gravity = np.asarray(gravity_body, dtype=np.float64)
    if origin.shape != (3, 3) or direction.shape != (3, 3) or ranges.shape != (3,) or gravity.shape != (3,):
        raise ValueError('expected three origins, three beams, three ranges and one gravity vector')
    if not (np.isfinite(origin).all() and np.isfinite(direction).all() and np.isfinite(gravity).all()):
        return GroundPlane(False, 'nonfinite ray geometry/gravity')
    if not np.isfinite(ranges).all() or np.any(ranges <= 0) or np.any(ranges > max_range_m):
        return GroundPlane(False, 'missing, nonpositive or out-of-range return')
    ray_norm = np.linalg.norm(direction, axis=1)
    g_norm = np.linalg.norm(gravity)
    if np.any(ray_norm < 1e-8) or g_norm < 1e-8:
        return GroundPlane(False, 'degenerate ray/gravity')
    ray_direction = direction / ray_norm[:, None]
    points = origin + ranges[:, None] * ray_direction
    n = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(n)
    area = 0.5 * length
    if length < 1e-10 or area < min_area_m2:
        return GroundPlane(False, 'nearly collinear ground returns', triangle_area_m2=float(area))
    n = n / length
    g = gravity / g_norm
    if np.dot(n, -g) < 0:
        n = -n
    cosine = float(np.dot(n, -g))
    if cosine < 0.1:
        return GroundPlane(False, 'ground normal nearly perpendicular to gravity', triangle_area_m2=float(area))
    tangential_down = g - np.dot(g, n) * n
    tangential_norm = float(np.linalg.norm(tangential_down))
    degrees = math.degrees(math.atan2(tangential_norm, cosine))
    uphill = (-tangential_down / tangential_norm) if tangential_norm > 1e-7 else None
    return GroundPlane(
        valid=True,
        reason='three-ray geometric fit only; no outlier redundancy',
        ground_normal_body=tuple(float(x) for x in n),
        slope_deg=degrees,
        uphill_body=tuple(float(x) for x in uphill) if uphill is not None else None,
        triangle_area_m2=float(area),
    )
