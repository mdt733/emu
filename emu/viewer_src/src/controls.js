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

// Heading target integrates the operator's requested turn rate, never the
// corrective rate (which would feed drift back into the setpoint).
// The error window prevents a struggling/fallen robot banking huge catch-up
// turns. All angles/rates use world-Z yaw in radians / radians per second.
export function headingHoldRate(requestedRate, headingTarget, actualHeading, kp, maxRate) {
  if (!Number.isFinite(actualHeading) || !Number.isFinite(headingTarget)) return requestedRate;
  const error = clamp(wrapRadians(headingTarget - actualHeading), -0.45, 0.45);
  return clamp(requestedRate + kp * error, -maxRate, maxRate);
}

/**
 * One viewer-only yaw control frame. Keep the operator/demo turn request in a
 * separate state channel from the rate handed to the neural actor.
 * Otherwise P corrections are mistaken for fresh turn requests on the next
 * frame, and the command can stick at the maximum yaw rate after release.
 */
export function nextYawCommand(requestedYaw, targetYaw, headingTarget,
                               actualHeading, dt, headingHold, kp, maxRate) {
  const maximumStep=Math.max(0,4*dt);
  const difference=targetYaw-requestedYaw;
  const request=clamp(requestedYaw+clamp(difference,-maximumStep,maximumStep),
                      -maxRate,maxRate);
  if(!headingHold || !Number.isFinite(actualHeading)) {
    return {requestedYaw:request,appliedYaw:request,
            headingTarget:Number.isFinite(actualHeading)?actualHeading:headingTarget};
  }
  // Avoid building up an unreachable heading if the robot has fallen or
  // lacks sufficient turning authority on unfamiliar terrain.
  const desired=wrapRadians(headingTarget+request*dt);
  const boundedError=clamp(wrapRadians(desired-actualHeading),-0.45,0.45);
  const target=wrapRadians(actualHeading+boundedError);
  return {requestedYaw:request,headingTarget:target,
          appliedYaw:headingHoldRate(request,target,actualHeading,kp,maxRate)};
}
