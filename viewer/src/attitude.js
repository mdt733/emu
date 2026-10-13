// Two different geometric questions. The IMU is rigidly attached to the torso;
// whole-robot lean also depends on where the feet are relative to that torso.
export function measuredAttitude(torsoRotation, torsoPosition, rightFoot, leftFoot) {
  if (torsoRotation.length !== 9) throw new Error('expected MuJoCo 3x3 torso rotation');
  const [tx, ty, tz] = torsoPosition;
  const fx = (rightFoot[0] + leftFoot[0]) / 2;
  const fy = (rightFoot[1] + leftFoot[1]) / 2;
  const fz = (rightFoot[2] + leftFoot[2]) / 2;
  const dx = tx - fx, dy = ty - fy, dz = tz - fz;
  // Project body-forward (local +X) into the horizontal plane. Yaw is allowed;
  // roll and chassis pitch must not redefine the gravity-vertical reference.
  const hx = torsoRotation[0], hy = torsoRotation[3];
  const hnorm = Math.hypot(hx, hy);
  const fore = hnorm > 1e-6 ? (dx * hx + dy * hy) / hnorm : 0;
  // Positive lean is chassis-centre displacement in the facing direction.
  const robotLean = Math.atan2(fore, dz);
  // Same gravity-projected chassis-pitch convention as environment.py:
  // atan2(projected_gravity_x, -projected_gravity_z).
  // MuJoCo xmat is row-major. Positive pitch tips local +X downward.
  const chassisPitch = Math.atan2(-torsoRotation[6], torsoRotation[8]);
  return { robotLean, chassisPitch };
}
