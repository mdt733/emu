// Current browser-side specialist *scenario* selection. Physics geometry
// lives exclusively in sensor-lab-carpet.js (eight genuine 1m prisms).
import { CARPET_DEFAULT, carpetRoleLimits } from './training-carpet.js';
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));

export function carpetLearningStage(limits){
  if(limits.source!=='training')return 'illustrative';
  const step=Number(limits.step||0);
  const foundation=Number(limits.foundationSteps??20_000_000);
  if(step<foundation)return 'foundation';
  if(limits.academyStream && step>=Number(limits.academyIntroSteps??60_000_000))
    return 'multi-lip';
  return 'specialists';
}

export function carpetRepresentativeRole(limits,requested='auto'){
  if(requested!=='auto')return requested;
  const stage=carpetLearningStage(limits);
  if(stage==='foundation')return 'standing';
  if(stage==='multi-lip')return 'step';
  if(stage==='illustrative')return 'step';
  return 'slope'; // representative focus, NOT all concurrently active worlds
}

export function terrainRoleEnvelope(limits=CARPET_DEFAULT,requested='auto'){
  const role=carpetRepresentativeRole(limits,requested);
  const p=carpetRoleLimits(limits,role);
  // During actual foundation every specialist terrain-exposure is zero.
  if(carpetLearningStage(limits)==='foundation' && !limits.previewOverride){
    return {...p,slopeDeg:0,transitionDeg:0,stepUpM:0,stepDownM:0,
      role:'standing',stage:'foundation'};
  }
  // Show a representative EXPOSED birth. Exposure is a per-world
  // probability in the GPU sampler, not an obstacle-height multiplier.
  // This viewer does not replay which of 2048 worlds received terrain.
  const stage=carpetLearningStage(limits);
  const slopeRate=clamp(Number(limits.slopeExposure??1),0,1);
  const stepRate=clamp(Number(limits.stepExposure??1),0,1);
  // The UI selects ORDINARY vs ACADEMY before projecting this envelope.
  // Do not silently replace the ordinary STEP track just because 60M passed.
  const academyBirth=(role==='step'&&stage==='multi-lip'&&
    limits.stepTrack==='academy');
  const stepUp=p.stepUpM,stepDown=p.stepDownM;
  return {
    ...p,
    slopeDeg:slopeRate>0?p.slopeDeg:0,
    stepUpM:(stepRate>0||academyBirth)?stepUp:0,
    stepDownM:(stepRate>0||academyBirth)?stepDown:0,
    slopeExposureRate:slopeRate,stepExposureRate:stepRate,
    transitionDeg:Number(limits.transitionDeg??3),
    role,stage,
    // Explicit viewer-only specialist overrides are physical difficulty
    // AFTER curriculum/role scaling. This is never fed to the trainer.
    ...(limits.previewOverride?.terrain||{})};
}

