// Small pure geometry and timing helpers shared by the MuJoCo viewer.
// Keep rendering, navigation controls and the physical simulator separate.

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

// Training draws cadence ONCE per episode from [0.85,1.20] Hz plus
// the birth-command speed gain. It does not retune cadence at every turn.
export function sampleEpisodeCadence({vx, vy}, control, rng=Math.random) {
  const base = 0.85 + 0.35 * rng();
  const ref = Math.max(0.05, Number(control.phase_speed_reference_mps ?? 0.6));
  const speed = Math.hypot(Number(vx) || 0, Number(vy) || 0);
  const speedFraction = Math.max(0, Math.min(1.5, speed / ref));
  return Math.min(Number(control.phase_rate_max_hz ?? 2.6),
                  base + Number(control.phase_speed_gain ?? 0.75) * speedFraction);
}

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

// Browser spectator force sampling. The 50Hz trainer starts pushes with
// p=.0015*T.resilience per frame, and also samples disturbed RESET POSES.
// "training" preserves that rare steady-state event probability. "drill"
// intentionally DOES NOT represent the GPU curriculum: it stresses the
// recovery response with up to 2 events/s on a single viewer robot.
export const TRAINING_PUSH_PROB_PER_FRAME=0.0015;
export const CONTROL_HZ=50;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,Number(x)||0));
export const SHOVE_MODES=Object.freeze(['training','inspection','drill']);

export function shoveChance(intensity=0,multiplier=1){
  return clamp(TRAINING_PUSH_PROB_PER_FRAME*
    clamp(intensity,0,1)*clamp(multiplier,1,20),0,1);
}
export function shoveChanceForMode(intensity=0,mode='training'){
  const d=clamp(intensity,0,1);
  if(mode==='training')return shoveChance(d,1);
  if(mode==='inspection')return shoveChance(d,8);
  if(mode==='drill'){
    // A visible, frequent diagnostic recovery workout: above zero, about
    // 1..2 attempts/s as the resilience intensity grows. No impulses at 0.
    return d===0?0:clamp((1+d)/CONTROL_HZ,0,.08);
  }
  throw Error('Unknown shove preview mode: '+String(mode));
}
export function expectedShoveRate(intensity=0,mode='training'){
  return CONTROL_HZ*shoveChanceForMode(intensity,mode);
}
export function sampleShove({intensity=0,multiplier=1,mode=null,maxForceN=0,
                             active=false,manual=false,rng=Math.random}={}){
  if(active||!Number.isFinite(maxForceN)||maxForceN<=0)return null;
  const chance=mode===null?shoveChance(intensity,multiplier)
    :shoveChanceForMode(intensity,mode);
  if(!manual && !(chance>0&&rng()<chance))return null;
  // Trainer draws one random direction and a magnitude in [0, cap];
  // force is applied through genuine MuJoCo torso xfrc_applied.
  const angle=rng()*Math.PI*2;
  const magnitude=rng()*maxForceN;
  const frames=2+Math.floor(rng()*7);
  return {fx:magnitude*Math.cos(angle),fy:magnitude*Math.sin(angle),
    magnitude,angle,frames};
}
