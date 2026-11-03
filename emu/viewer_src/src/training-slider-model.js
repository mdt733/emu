// Pure UI projection: effective role frontiers and reversible viewer-only
// overrides. These values are NOT the actor's weights or a learned ability.
import {specialistLesson} from './training-inspector.js';
const clip=(n,a,b)=>Math.min(b,Math.max(a,Number(n)||0));
export const SLIDER_FIELDS=Object.freeze([
  ['slope','tuneSlope','°',20,2],
  ['transition','tuneTransition','°',15,2],
  ['up','tuneUp',' mm',150,3],
  ['down','tuneDown',' mm',150,3],
  ['forward','tuneForward',' m/s',1.5,2],
  ['reverse','tuneReverse',' m/s',.8,2],
  ['lateral','tuneLateral',' m/s',1,2],
  ['yaw','tuneYaw',' rad/s',3,2],
  ['resilience','tuneResilience','',1,2],
  ['push','tunePush',' N',3,2],
]);
export function roleSliders(base,role='auto'){
  const l=specialistLesson(base,role);
  return {
    slope:l.terrain.slopeDeg,transition:l.terrain.transitionDeg,
    up:l.terrain.stepUpM*1000,down:l.terrain.stepDownM*1000,
    forward:l.caps.forward,reverse:l.caps.reverse,
    lateral:l.caps.lateral,yaw:l.caps.yaw,
    resilience:l.intensity,push:l.caps.push
  };
}
export function customRoleOverride(values){
  const t=Object.fromEntries(SLIDER_FIELDS.map(([key,,_,max])=>[
    key,clip(values[key],0,max)]));
  return {
    terrain:{slopeDeg:t.slope,transitionDeg:t.transition,
             stepUpM:t.up/1000,stepDownM:t.down/1000},
    command:{forward:t.forward,reverse:t.reverse,
             lateral:t.lateral,yaw:t.yaw,push:t.push},
    resilience:t.resilience
  };
}
