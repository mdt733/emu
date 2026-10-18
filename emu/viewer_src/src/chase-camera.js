// Third-person, gravity-levelled chase camera. Pure geometry so movement
// can be verified without loading MuJoCo, WASM, or Three.js.
export const CHASE_DISTANCE_M = 0.85;
export const CHASE_CAMERA_HEIGHT_M = 0.44;
export const CHASE_LOOK_AHEAD_M = 0.24;

export function smoothFactor(dt, responseHz) {
  if (!Number.isFinite(dt) || !Number.isFinite(responseHz)) throw new Error('Invalid camera timing');
  return 1 - Math.exp(-Math.max(0, Math.min(dt, .1)) * Math.max(0, responseHz));
}

export function smoothHeading(current, target, amount) {
  if (![current, target, amount].every(Number.isFinite)) throw new Error('Invalid camera heading');
  // Heading is circular: from +179° to -179° is 2°, not a full revolution.
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  const next = current + delta * Math.max(0, Math.min(1, amount));
  return Math.atan2(Math.sin(next), Math.cos(next));
}

export function chasePose(torso, heading, terrainHeightAt = () => 0) {
  if (torso.length !== 3 || !torso.every(Number.isFinite) || !Number.isFinite(heading))
    throw new Error('Invalid chase camera pose');
  const dx = Math.cos(heading), dy = Math.sin(heading);
  const bx = torso[0] - CHASE_DISTANCE_M * dx;
  const by = torso[1] - CHASE_DISTANCE_M * dy;
  const cameraGround = Number(terrainHeightAt(bx, by)) || 0;
  const robotGround = Number(terrainHeightAt(torso[0], torso[1])) || 0;
  return {
    position: [bx, by, Math.max(torso[2] + CHASE_CAMERA_HEIGHT_M, cameraGround + 0.43)],
    target: [torso[0] + CHASE_LOOK_AHEAD_M * dx,
             torso[1] + CHASE_LOOK_AHEAD_M * dy,
             Math.max(torso[2] + 0.015, robotGround + 0.17)],
  };
}
