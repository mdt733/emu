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
    linear_change=np.any(np.abs(np.diff(cmd[:,:2],axis=0))>1e-6,axis=1)
    angular_change=np.abs(np.diff(cmd[:,2]))>1e-6
    boundary=np.flatnonzero(linear_change|angular_change)+1
    actual_yaw=np.arctan2(np.sin(np.diff(head)),np.cos(np.diff(head)))/dt
    for idx in boundary:
        if idx < settle_frames:
            continue
        next_bound=boundary[boundary>idx]
        end=int(next_bound[0]) if len(next_bound) else n
        end=min(end,idx+round(2/dt))
        # Multiple physical requests can change on one frame; measure the
        # linear and angular response separately, rather than pretending a
        # pure turn isn't a maneuver.
        if linear_change[idx-1]:
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
            row=dict(start_step=int(idx),kind=kind,response_axis='translation',
                     target_speed_mps=float(now),response_latency_s=response,
                     response_observed=response is not None)
            if now < .008:
                last=min(n,idx+max(1,round(.5/dt)))
                row['braking_path_first_0p5s_m']=float(
                    np.sum(np.linalg.norm(np.diff(pos[idx:last+1],axis=0),axis=1)))
            transitions.append(row)
        if angular_change[idx-1]:
            previous=float(cmd[idx-1,2])
            target=float(cmd[idx,2])
            if abs(target)<.008 and abs(previous)>=.025:
                kind='turn_stop'
            elif abs(previous)<.008 and abs(target)>=.025:
                kind='turn_start'
            elif target*previous<0.:
                kind='turn_reverse'
            else:
                kind='turn_rate_change'
            response=None
            for j in range(idx,end-stable_frames+1):
                rates=actual_yaw[j:j+stable_frames]
                if abs(target)<.008:
                    reached=np.all(np.abs(rates)<=max(.04,.2*abs(previous)))
                else:
                    reached=np.all(rates*np.sign(target)>=.70*abs(target))
                if reached:
                    response=float((j-idx)*dt)
                    break
            transitions.append(dict(
                start_step=int(idx),kind=kind,response_axis='yaw',
                target_yaw_rate_rad_s=target,response_latency_s=response,
                response_observed=response is not None))
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
                          foot_clearance_m=None,
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
    # Signed ball-bottom clearance against the actual plane is supplied by
    # the examiner; older flat-world calls retain their original semantics.
    clearance = (np.asarray(foot_clearance_m,dtype=float)
                 if foot_clearance_m is not None else feet[:,:,2]-.018)
    if clearance.shape != (n,2) or not np.isfinite(clearance).all():
        raise ValueError('invalid foot-ground clearance shape/values')
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
                max_clearance[foot] = max(0.0, clearance[k, foot])
            elif not ct[k, foot] and swing_start[foot] is not None:
                air_frames[foot] += 1
                max_clearance[foot] = max(max_clearance[foot],
                                          float(clearance[k, foot]))
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
    # Consecutive overload is more informative than isolated spikes.
    # Break each boolean channel into contiguous runs in actual frame time.
    def max_true_run(mask):
        padded = np.concatenate(([False], np.asarray(mask,dtype=bool), [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return float(np.max(ends-starts) * dt) if len(starts) else 0.
    longest = [max_true_run(over[:,k]) for k in range(6)]
    near_peak = np.abs(tau) >= .90*ceiling
    return {
        'actuator_sample_frames': int(len(tau)),
        'actuator_sample_duration_s': float(duration),
        'servo_peak_abs_torque_nm': np.max(np.abs(tau), axis=0).tolist(),
        'servo_p95_abs_torque_nm': np.percentile(np.abs(tau), 95, axis=0).tolist(),
        'servo_over_continuous_fraction': np.mean(over, axis=0).tolist(),
        'servo_over_continuous_seconds': (np.sum(over, axis=0) * dt).tolist(),
        'servo_longest_continuous_overload_s': longest,
        'servo_near_peak_fraction': np.mean(near_peak, axis=0).tolist(),
        'servo_p95_abs_joint_speed_rad_s': np.percentile(np.abs(omega),95,axis=0).tolist(),
        'servo_peak_abs_joint_speed_rad_s': np.max(np.abs(omega),axis=0).tolist(),
        'servo_peak_limit_utilization': np.max(np.abs(tau) / ceiling, axis=0).tolist(),
        'servo_thermal_exposure_proxy_s': (np.sum((tau / rating) ** 2, axis=0) * dt).tolist(),
        'servo_positive_shaft_work_j': (np.sum(positive, axis=0) * dt).tolist(),
        'servo_braking_shaft_work_j': (np.sum(negative, axis=0) * dt).tolist(),
        'servo_measured_interval_abs_shaft_work_j': float(np.sum(np.abs(power)) * dt),
        'servo_all_sample_over_continuous_fraction': float(np.mean(over)),
        'thermal_proxy_definition': 'integral (MuJoCo actuator force / 0.638 Nm)^2 dt, uncalibrated; no cooling or temperature model',
        'actuator_work_definition': ('integral of MuJoCo position actuator_force times '
            'joint speed, before separate joint damping, spring and Coulomb losses; '
            'not net transmission work or measured electrical energy'),
        'sampling_caveat': ('one-frame delayed pre-step force/velocity; '
                            'terminal control step excluded from the scored path; '
                            'final state appended for uninterrupted survivors'),
    }


def examine_spring_cycle(trace, world, *, start, stop, terminal, contacts):
    """Resolve elastic storage and release on the SAME scored pose intervals.

    Positive/negative changes in 1/2*k*(q-reference)^2 are not useful work,
    net energy returned to the robot, or measured carbon-fibre damping.
    Track passive flex separately, and label when a release overlaps a
    continuously grounded foot. Never use post-reset spring states.
    """
    required = ('spring_joint_qpos_rad', 'spring_reference_rad',
                'spring_stiffness_nm_rad')
    if not all(k in trace for k in required):
        return {}
    states=np.asarray(trace['spring_joint_qpos_rad'],dtype=float)
    reference=np.asarray(trace['spring_reference_rad'],dtype=float)
    stiffness=np.asarray(trace['spring_stiffness_nm_rad'],dtype=float)
    if states.ndim != 3 or states.shape[2] != 8 or reference.shape != (8,) or stiffness.shape != (8,):
        raise ValueError('spring accounting expects eight actual MJX hinge channels')
    if np.any(stiffness < 0) or not all(np.isfinite(v).all() for v in (states,reference,stiffness)):
        raise ValueError('invalid spring positions or model parameters')
    # Pre-step states indexed 0..stop on a terminal episode. An unbroken
    # horizon instead ends at the true last state, not a repeated final
    # sample. start/stop refer to the course's pose, not reward timesteps.
    if terminal:
        pose=states[start:stop+1,world,:]
    else:
        if 'last_spring_joint_qpos_rad' not in trace:
            return {}
        last=np.asarray(trace['last_spring_joint_qpos_rad'],dtype=float)
        if last.shape != (states.shape[1],8):
            raise ValueError('final spring state dimension differs from trace')
        pose=np.concatenate((states[start:stop,world,:],last[world,None,:]),axis=0)
    contact=np.asarray(contacts,dtype=bool)
    if len(pose) < 2 or contact.shape != (len(pose),2):
        raise ValueError('scored contacts and spring pose endpoints misaligned')
    energy=.5*stiffness[None,:]*(pose-reference[None,:])**2
    delta=np.diff(energy,axis=0)
    charged=np.maximum(delta,0.)
    released=np.maximum(-delta,0.)
    # The first six are actuator-joint support springs; the last two
    # are physically passive right and left shin flex joints.
    passive_release=released[:,6:]
    grounded=contact[:-1] & contact[1:]
    stance=passive_release*grounded
    airborne=passive_release*(~grounded)
    stance_j=float(np.sum(stance))
    release_j=float(np.sum(passive_release))
    totals=dict(
        spring_interval_count=int(len(delta)),
        passive_shin_release_total_j=release_j,
        passive_shin_stance_release_total_j=stance_j,
        passive_shin_stance_release_fraction=(
            float(stance_j/release_j) if release_j > 1e-12 else None),
        spring_joint_charge_j=np.sum(charged,axis=0).tolist(),
        spring_joint_release_j=np.sum(released,axis=0).tolist(),
        spring_joint_net_storage_delta_j=np.sum(delta,axis=0).tolist(),
        passive_shin_charge_j=np.sum(charged[:,6:],axis=0).tolist(),
        passive_shin_release_j=np.sum(passive_release,axis=0).tolist(),
        passive_shin_stance_release_j=np.sum(stance,axis=0).tolist(),
        passive_shin_nonstance_release_j=np.sum(airborne,axis=0).tolist(),
        passive_shin_stance_interval_count=np.sum(grounded,axis=0).astype(int).tolist(),
        passive_shin_joint_energy_peak_j=np.max(energy[:,6:],axis=0).tolist(),
        passive_shin_joint_energy_start_end_j=[
            [float(energy[0,j]),float(energy[-1,j])] for j in (6,7)],
        spring_mechanism_caveat=(
            'geometric elastic potential changes at scored frame boundaries; '
            'stance overlap does not establish propulsion, recovered useful '
            'work, motor efficiency, or physical damping losses'),
    )
    return totals


def evaluate_trace_arrays(trace, gravity_world, *, dt=.02, settle_frames=25,
                          ground_mode='tilted_gravity'):
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
    if ground_mode not in ('tilted_gravity','inclined_contact_plane'):
        raise ValueError('unknown ground scoring mode')
    # Geometric ground must originate in the actual MJX Data, not from
    # requested angle labels; never silently "prove" a slope from a flag.
    floor_n = np.asarray(trace.get('floor_normal_world',
                        np.broadcast_to([0.,0.,1.],(horizon,worlds,3))),dtype=float)
    floor_o = np.asarray(trace.get('floor_origin_world',
                        np.zeros((horizon,worlds,3))),dtype=float)
    if (floor_n.shape!=(horizon,worlds,3) or floor_o.shape!=(horizon,worlds,3)
        or not np.isfinite(floor_n).all() or not np.isfinite(floor_o).all()):
        raise ValueError('invalid MJX floor geom transform trace')
    if ground_mode=='inclined_contact_plane' and (
        'floor_normal_world' not in trace or 'floor_origin_world' not in trace):
        raise ValueError('inclined contact exam requires actual MJX floor geometry')
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
        ground_n=floor_n[0,w]
        ground_o=floor_o[0,w]
        if (np.linalg.norm(ground_n)<1e-8 or ground_n[2]<=0
            or np.max(np.abs(floor_n[:stop,w]-ground_n))>1e-5
            or np.max(np.abs(floor_o[:stop,w]-ground_o))>1e-5):
            raise ValueError('physical floor normal/origin changed inside the scored episode')
        ground_n=ground_n/np.linalg.norm(ground_n)
        if ground_mode=='inclined_contact_plane':
            horizontal=np.linalg.norm(ground_n[:2])
            slope=math.degrees(math.atan2(horizontal,ground_n[2]))
            uphill=(-ground_n[:2]/horizontal) if horizontal>1e-8 else None
        else:
            slope,uphill=incline_from_tilted_gravity(g[w])
        foot_clearance=np.sum((feet-ground_o)*ground_n,axis=2)-.018
        strides = count_contact_strides(
            pos,feet,contacts,settle_frames=settle_frames,
            foot_clearance_m=foot_clearance)
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
            physical.update(examine_spring_cycle(
                trace,w,start=settle_frames,stop=stop,
                terminal=bool(len(term)),contacts=contacts[settle_frames:]))
            progress=course['signed_course_progress_m']
            release=physical.get('passive_shin_stance_release_total_j')
            physical['passive_shin_stance_release_per_progress_m_j_m'] = (
                float(release/progress) if release is not None and progress>0.01
                else None)
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
                    # Positive storage/release changes are energy-state
                    # movements, NOT calibrated useful work at the feet.
                    for prefix, energy in (('all_springs',stored),('passive_shins',shin)):
                        delta=np.diff(energy)
                        physical[prefix+'_charging_proxy_j'] = float(np.sum(np.maximum(delta,0.)))
                        physical[prefix+'_release_proxy_j'] = float(np.sum(np.maximum(-delta,0.)))
                    physical['spring_return_not_inferred'] = True
            # The scored foot path is a contact-conditioned geometric slip
            # proxy, not a force-plate measurement of sole friction.
            scored_feet = feet[settle_frames:]
            scored_contacts = contacts[settle_frames:]
            delta_3d=np.diff(scored_feet,axis=0)
            tangent_delta=delta_3d-np.sum(delta_3d*ground_n,axis=2)[:,:,None]*ground_n
            foot_delta = np.linalg.norm(tangent_delta,axis=2)
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
        # Grade is the requested course geometry, not a measured incline
        # recovered from a sensor or contact impulses.
        result.update(contact_stride_pairs=strides,configured_course_grade_deg=slope,
                      measured_ground_normal_xyz=ground_n.tolist(),
                      invalid_reason=('terminated before horizon' if len(term) else None))
        findings.append(result)
    return findings


# Three ToF modules below the torso: front-right, front-left, rear.
# Local coordinates are provisional mounting points, not a CAD claim.
# Calibrate their actual measured extrinsics before any physical deployment.
TOF_TRIPOD_MOUNTS_BODY_M = (
    (.035, -.050, -.025),
    (.035,  .050, -.025),
    (-.040, .000, -.025),
)
TOF_TRIPOD_BEAMS_BODY = ((0.,0.,-1.),)*3
TOF_MIN_RANGE_M = .035
TOF_MAX_RANGE_M = .60


def fit_forward_array_ground(xp, ranges, valid, gravity_body,
                             pitch_deg=(40.,60.,80.,100.),
                             yaw_deg=(-22.5,-7.5,7.5,22.5)):
    """Estimate grade, depth and *visibility*, solely from a 4x4 range array.

    One least-squares plane fit to the actual returned ray endpoints, with
    a residual rejection gate across all visible cells. It does not touch
    simulator normals or the old three physical ToF mounts. Mixed riser/top
    returns are NOT passed off as a smooth ground plane: low-confidence
    estimates become invalid while the original 16 raw rays remain available.

    Works in NumPy and JAX. Invalid rays cannot bias plane fitting.
    """
    rays=xp.asarray([
        (math.cos(math.radians(p))*math.cos(math.radians(y)),
         math.cos(math.radians(p))*math.sin(math.radians(y)),
         -math.sin(math.radians(p)))
        for p in pitch_deg for y in yaw_deg],dtype=xp.float32)
    d=xp.asarray(ranges)
    mask=xp.asarray(valid,dtype=xp.bool_) & xp.isfinite(d) & (d>=.04) & (d<=1.5)
    weight=mask.astype(xp.float32)
    points=rays[None,:,:]*xp.where(mask,d,0.)[:,:,None]
    x,y,z=points[:,:,0],points[:,:,1],points[:,:,2]
    design=xp.stack((x,y,xp.ones_like(x)),axis=-1)
    gram=xp.einsum('eki,ek,ekj->eij',design,weight,design)
    rhs=xp.einsum('eki,ek,ek->ei',design,weight,z)
    # Bias only the plane slopes to avoid singular layouts without inventing
    # an imaginary ground depth at the camera.
    reg=xp.asarray([.00005,.00005,.0000001],dtype=xp.float32)
    coefficients=xp.linalg.solve(
        gram+xp.eye(3,dtype=xp.float32)[None,:,:]*reg[None,None,:],rhs)
    a,b,c=coefficients[:,0],coefficients[:,1],coefficients[:,2]
    normal=xp.stack((-a,-b,xp.ones_like(a)),axis=1)
    normal=normal/xp.maximum(xp.linalg.norm(normal,axis=1,keepdims=True),1e-7)
    gravity=xp.asarray(gravity_body)
    up=-gravity/xp.maximum(xp.linalg.norm(gravity,axis=1,keepdims=True),1e-7)
    normal=xp.where(xp.sum(normal*up,axis=1,keepdims=True)<0,-normal,normal)
    forward=xp.asarray([1.,0.,0.],dtype=xp.float32)[None,:]-up[:,0:1]*up
    forward=forward/xp.maximum(xp.linalg.norm(forward,axis=1,keepdims=True),1e-7)
    sideways=xp.cross(up,forward)
    cosine=xp.sum(normal*up,axis=1)
    count=xp.sum(weight,axis=1)
    estimated=xp.einsum('eki,ei->ek',design,coefficients)
    residual=xp.sqrt(xp.sum(weight*(estimated-z)**2,axis=1)/xp.maximum(count,1.))
    depth=-c
    # At least six non-collinear returns. Planar residual <= 8mm avoids
    # interpreting a step riser and upper platform as one smooth slope.
    good=(count>=6.) & (residual<.008) & (cosine>.25) & (
        depth>=.035) & (depth<=.60) & (
        xp.linalg.norm(gravity,axis=1)>.1)
    confidence=xp.where(good,
        xp.clip(count/12.,0.,1.)*xp.clip(1.-residual/.008,0.,1.),0.)
    return dict(
        valid=good,
        ground_forward_grade_rad=xp.where(
            good,xp.arctan2(-xp.sum(normal*forward,axis=1),cosine),0.),
        ground_sideways_grade_rad=xp.where(
            good,xp.arctan2(-xp.sum(normal*sideways,axis=1),cosine),0.),
        ground_visibility_confidence=confidence,
        optical_depth_m=xp.where(good,depth,0.),
        fit_residual_m=residual,
        valid_count=count,
    )


def simulate_tripod_ranges(xp, torso_world, rotation_body_to_world,
                           floor_origin_world, floor_normal_world,
                           *, range_errors_m=None):
    """Sensor *forward model*: true collision geometry may only be read here.

    Three body-mounted, narrow downward rays intersect the fixed plane.
    Results are measured ray lengths, not the floor normal. No body/shin
    self-occlusion, reflectance, multipath or surface texture yet.
    """
    mounts=xp.asarray(TOF_TRIPOD_MOUNTS_BODY_M)
    beams=xp.asarray(TOF_TRIPOD_BEAMS_BODY)
    pos=torso_world[:,None,:]+xp.einsum('wij,kj->wki',rotation_body_to_world,mounts)
    dirs=xp.einsum('wij,kj->wki',rotation_body_to_world,beams)
    n=floor_normal_world
    denominator=xp.sum(dirs*n[:,None,:],axis=-1)
    numerator=xp.sum((floor_origin_world[:,None,:]-pos)*n[:,None,:],axis=-1)
    # Floor normals point upward. Near-grazing beams are invalid.
    safe=xp.where(xp.abs(denominator)>1e-6,denominator,-1.)
    ranges=numerator/safe
    if range_errors_m is not None:
        ranges=ranges+xp.asarray(range_errors_m)
    valid=(denominator<-.3)&xp.isfinite(ranges)&(ranges>=TOF_MIN_RANGE_M)&(ranges<=TOF_MAX_RANGE_M)
    return xp.where(valid,ranges,0.),valid


def fit_tripod_ground(xp, ranges_m, gravity_body, *,
                      mounts_body=TOF_TRIPOD_MOUNTS_BODY_M,
                      beams_body=TOF_TRIPOD_BEAMS_BODY,
                      min_area_m2=.0001):
    """Estimator shared by CPU and JAX; no simulator floor normal input.

    Uses only the three measured beam lengths, calibrated mounts, and IMU
    gravity estimate. Positive forward/side grades are uphill in the torso
    heading's gravity-levelled X/Y directions, expressed in radians.

    Three points ALWAYS define a plane: no residual-based quality claim is
    possible. Confidence here means valid geometry/visibility only, not
    probability of accurate terrain on broken, mixed or reflective ground.
    """
    ranges=xp.asarray(ranges_m)
    g=xp.asarray(gravity_body)
    origins=xp.asarray(mounts_body)
    dirs=xp.asarray(beams_body)
    length=xp.sqrt(xp.sum(dirs*dirs,axis=-1))
    dirs=dirs/xp.maximum(length[:,None],1e-8)
    points=origins[None,:,:]+ranges[:,:,None]*dirs[None,:,:]
    v1=points[:,1,:]-points[:,0,:]
    v2=points[:,2,:]-points[:,0,:]
    raw=xp.cross(v1,v2)
    area=.5*xp.sqrt(xp.sum(raw*raw,axis=-1))
    n=raw/xp.maximum(2*area[:,None],1e-8)
    up=-g/xp.maximum(xp.sqrt(xp.sum(g*g,axis=-1,keepdims=True)),1e-8)
    n=xp.where(xp.sum(n*up,axis=-1,keepdims=True)<0,-n,n)
    cosine=xp.sum(n*up,axis=-1)
    fwd=xp.broadcast_to(xp.asarray((1.,0.,0.)),up.shape)
    fwd=fwd-xp.sum(fwd*up,axis=-1,keepdims=True)*up
    fwd=fwd/xp.maximum(xp.sqrt(xp.sum(fwd*fwd,axis=-1,keepdims=True)),1e-8)
    side=xp.cross(up,fwd)
    fgrade=xp.arctan2(-xp.sum(n*fwd,axis=-1),cosine)
    sgrade=xp.arctan2(-xp.sum(n*side,axis=-1),cosine)
    valid=(xp.all(xp.isfinite(ranges)&(ranges>=TOF_MIN_RANGE_M)&
                  (ranges<=TOF_MAX_RANGE_M),axis=-1)
           &(area>=min_area_m2)&(cosine>.1)
           &(xp.sqrt(xp.sum(g*g,axis=-1))>.1)
           &(xp.sqrt(xp.sum(fwd*fwd,axis=-1))>.5))
    # Camera at the torso origin, looking along body -Z. Recover scale
    # from observed triangulated points, not from a privileged floor plane.
    signed_offset=xp.sum(n*points[:,0,:],axis=-1)
    optical_depth=-signed_offset/xp.where(xp.abs(n[:,2])>.1,n[:,2],1.)
    depth_valid=(xp.isfinite(optical_depth)
                 &(optical_depth>=TOF_MIN_RANGE_M)
                 &(optical_depth<=TOF_MAX_RANGE_M))
    valid=valid&depth_valid
    # A zero angle with valid=False MUST NOT be treated as flat ground.
    confidence=xp.where(valid,
        xp.clip(area/.00375,0.,1.)*xp.clip(cosine/.6,0.,1.),0.)
    return dict(forward_grade_rad=xp.where(valid,fgrade,0.),
                sideways_grade_rad=xp.where(valid,sgrade,0.),
                ground_normal_body=xp.where(valid[:,None],n,up),
                triangle_area_m2=area,valid=valid,visibility_confidence=confidence,
                optical_depth_m=xp.where(valid,optical_depth,0.))


def simulate_downward_optical_flow(xp, body_velocity_mps, body_gyro_rad_s,
                                   optical_depth_m, *, flow_error_rad_s=None):
    """Angular image flow at the optical axis of a down-looking camera.

    Forward optics uses TRUE simulator body velocity, body-local gyro and
    true ray/plane intersection depth to generate noisy *measurements*.
    No true velocity enters the inverse estimator. Optical boresight is
    body -Z; image horizontal axes are body X/Y. Units rad/s approximately.

    At the boresight, yaw-induced image translation is zero to first order,
    while roll/pitch rotation must be compensated. Does not model image
    texture, pixel quantization, FOV, self-occlusion, blur or sensor latency.
    """
    v=xp.asarray(body_velocity_mps)
    w=xp.asarray(body_gyro_rad_s)
    depth=xp.asarray(optical_depth_m)
    valid=(xp.isfinite(depth)&(depth>=TOF_MIN_RANGE_M)
           &(depth<=TOF_MAX_RANGE_M)
           &xp.all(xp.isfinite(v),axis=-1)&xp.all(xp.isfinite(w),axis=-1))
    safe=xp.maximum(depth,TOF_MIN_RANGE_M)
    angular=xp.stack((-v[:,0]/safe+w[:,1],
                      -v[:,1]/safe-w[:,0]),axis=-1)
    if flow_error_rad_s is not None:
        angular=angular+xp.asarray(flow_error_rad_s)
    valid=valid&xp.all(xp.isfinite(angular),axis=-1)
    return xp.where(valid[:,None],angular,0.),valid


def estimate_flow_velocity(xp, flow_rad_s, imu_gyro_body_rad_s,
                           tof_estimated_optical_depth_m, *,
                           flow_valid=None, tof_valid=None):
    """Invert a gyro-compensated downward optical-flow measurement.

    Returns *body-frame* horizontal vxy, never world velocity. The estimator
    receives measured angular image flow, gyro and tripod-derived distance,
    NOT the MuJoCo velocity or true floor normal used by the forward model.
    Invalid sources produce zero velocity AND false validity; a policy must
    not mistake this for a confident zero-speed measurement.
    """
    flow=xp.asarray(flow_rad_s)
    gyro=xp.asarray(imu_gyro_body_rad_s)
    depth=xp.asarray(tof_estimated_optical_depth_m)
    valid=(xp.isfinite(depth)&(depth>=TOF_MIN_RANGE_M)
           &(depth<=TOF_MAX_RANGE_M)
           &xp.all(xp.isfinite(flow),axis=-1)
           &xp.all(xp.isfinite(gyro),axis=-1))
    if flow_valid is not None:
        valid=valid&xp.asarray(flow_valid)
    if tof_valid is not None:
        valid=valid&xp.asarray(tof_valid)
    scale=xp.maximum(depth,TOF_MIN_RANGE_M)
    vxy=xp.stack((-scale*(flow[:,0]-gyro[:,1]),
                  -scale*(flow[:,1]+gyro[:,0])),axis=-1)
    return dict(velocity_body_xy_mps=xp.where(valid[:,None],vxy,0.),
                velocity_valid=valid,
                velocity_confidence=xp.where(valid,1.,0.))


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
    fitted = fit_tripod_ground(np,ranges[None,:],gravity[None,:],
                               mounts_body=origin,beams_body=direction,
                               min_area_m2=min_area_m2)
    area=float(fitted['triangle_area_m2'][0])
    if area<min_area_m2:
        return GroundPlane(False,'nearly collinear ground returns',
                           triangle_area_m2=area)
    if not bool(fitted['valid'][0]):
        return GroundPlane(False,'ground normal nearly perpendicular to gravity',
                           triangle_area_m2=area)
    n=np.asarray(fitted['ground_normal_body'][0])
    g=gravity/g_norm
    cosine=float(np.dot(n,-g))
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
