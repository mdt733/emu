// Browser-only command mapping. Policy observations keep their canonical axes.
// "split": left = forward/strafe, right = yaw/pitch (good for touch).
// "hybrid": left diagonal mixes forward, yaw and strafe; full sideways is strafe.
// Right stick continues to provide independent yaw trim and pitch.
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const JOYSTICK_MODES = ['split', 'hybrid'];

export function joystickToCommands(leftX, leftY, rightX, rightY, limits, mode='split') {
  if (!JOYSTICK_MODES.includes(mode)) throw new Error('Unknown joystick mode');
  const vx = leftY >= 0 ? leftY * limits.vx_max : (-leftY) * limits.vx_min;
  const turnBlend = mode === 'hybrid' ? clamp((Math.abs(leftY) - 0.12) / 0.60, 0, 1) : 0;
  const vy = -leftX * (1 - turnBlend) * limits.vy_abs;
  const yaw = -clamp(rightX + leftX * turnBlend, -1, 1) * limits.yaw_abs;
  // Pull back pitches the chassis back, push forward pitches it forward.
  const pitch = -rightY * limits.pitch_abs_rad;
  return {vx, vy, yaw, pitch};
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
