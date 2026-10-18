// Browser-only command mapping. Policy observations keep their canonical axes.
// "split": left = forward/strafe, right = yaw/pitch (good for touch).
// "hybrid": left diagonal mixes forward, yaw and strafe; full sideways is strafe.
// Right stick continues to provide independent yaw trim and pitch.
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const JOYSTICK_MODES = ['split', 'hybrid'];

export function joystickToCommands(leftX, leftY, rightX, rightY, limits, mode='split') {
  if (!JOYSTICK_MODES.includes(mode)) throw new Error('Unknown joystick mode');
  const vx = leftY >= 0 ? leftY * limits.vx_max : (-leftY) * limits.vx_min;
  const turnBlend = mode === 'hybrid' ? 0.8 * clamp((Math.abs(leftY) - 0.12) / 0.60, 0, 1) : 0;
  const vy = -leftX * (1 - turnBlend) * limits.vy_abs;
  const yaw = -clamp(rightX + leftX * turnBlend, -1, 1) * limits.yaw_abs;
  // Pull back pitches the chassis back, push forward pitches it forward.
  const pitch = -rightY * limits.pitch_abs_rad;
  return {vx, vy, yaw: yaw || 0, pitch};
}

export const wrapRadians = radians => Math.atan2(Math.sin(radians), Math.cos(radians));

// A heading setpoint is an angle in world coordinates; the robot's neural
// actor expects a YAW RATE command in radians/s. A pure outer-loop P controller
// translates heading error to that rate, then clamps to the actor's taught
// yaw envelope. No direct-stick feedforward and no integral wind-up.
export const MAX_HEADING_ERROR_RAD = 1.0; // cap target lead at ~57° during a fall

export function headingHoldRate(requestedRate, headingTarget, actualHeading, kp, maxRate) {
  if (!Number.isFinite(actualHeading) || !Number.isFinite(headingTarget))
    return clamp(requestedRate, -maxRate, maxRate);
  const error = clamp(wrapRadians(headingTarget - actualHeading),
                      -MAX_HEADING_ERROR_RAD, MAX_HEADING_ERROR_RAD);
  return clamp(kp * error, -maxRate, maxRate);
}

/**
 * Integrate the operator's requested rotation into a world heading SETPOINT,
 * never the P-corrected yaw output. Releasing the stick holds the target.
 * With P disabled, the original direct yaw-rate steering is unchanged.
 */
export function nextYawCommand(requestedYaw, targetYaw, headingTarget,
                               actualHeading, dt, headingHold, kp, maxRate) {
  const maximumStep=Math.max(0,4*Math.min(Math.max(dt,0),.1));
  const difference=targetYaw-requestedYaw;
  const request=clamp(requestedYaw+clamp(difference,-maximumStep,maximumStep),
                      -maxRate,maxRate);
  if(!headingHold || !Number.isFinite(actualHeading)) {
    return {requestedYaw:request,appliedYaw:request,
            headingTarget:Number.isFinite(actualHeading)?wrapRadians(actualHeading):headingTarget};
  }
  // Keep the desired compass heading near the actual robot when it cannot
  // turn quickly enough; otherwise commands bank up an impossible full turn.
  const desired=wrapRadians(headingTarget+request*Math.max(0,dt));
  const boundedError=clamp(wrapRadians(desired-actualHeading),
                           -MAX_HEADING_ERROR_RAD,MAX_HEADING_ERROR_RAD);
  const target=wrapRadians(actualHeading+boundedError);
  return {requestedYaw:request,headingTarget:target,
          appliedYaw:headingHoldRate(request,target,actualHeading,kp,maxRate)};
}
