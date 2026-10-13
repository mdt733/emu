// Manual joystick/keyboard mapping for the browser viewer only.
// Do not invert the policy's canonical vy or autonomous demonstration commands.
export function joystickToCommands(leftX, leftY, rightX, rightY, limits) {
  return {
    vx: leftY >= 0 ? leftY * limits.vx_max : (-leftY) * limits.vx_min,
    // Physical right strafe is the negative vy direction for this model.
    vy: -leftX * limits.vy_abs,
    yaw: -rightX * limits.yaw_abs,
    pitch: rightY * limits.pitch_abs_rad,
  };
}
