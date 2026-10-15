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
    error = wrap_radians(target - measured)
    corrected = max(-max_rate, min(max_rate, cmd + kp * error))
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
