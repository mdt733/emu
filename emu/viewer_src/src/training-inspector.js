// Viewer-only specialist exercises. These never mutate EMU's learner.
// The live API supplies *measured curriculum envelopes*; each preview is a
// deterministic representative birth, not a stream from one of 2048 GPU envs.
import {carpetLearningStage,carpetRepresentativeRole,terrainRoleEnvelope} from './training-solid-carpet.js';

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const finite=(v,def=0)=>Number.isFinite(Number(v))?Number(v):def;
export const SPECIALIST_GUIDE=Object.freeze({
  standing:['Foundation','Small forward requests; learning to stand'],
  command:['Commands','Forward, reverse, strafe and yaw tracking'],
  resilience:['Resilience','Moving while recovering from horizontal shoves'],
  slope:['Slopes','Hold forward velocity on changing uphill/downhill grades'],
  step:['Steps','Approach and cross real rising/falling lips'],
  transition:['Transitions','Change speed and direction during a walk'],
  control:['Control & efficiency','Retain an economical, predictable gait']
});
export function specialistLesson(base,role='auto'){
  const active=carpetRepresentativeRole(base,role);
  const p=terrainRoleEnvelope(base,active);
  const override=base.previewOverride||{};
  const caps={
    forward:clamp(finite(base.vxHi,.06),.01,3),
    reverse:clamp(Math.abs(finite(base.vxLo,-.02)),0,2),
    lateral:clamp(Math.abs(finite(base.vyMax,.035)),0,2),
    yaw:clamp(Math.abs(finite(base.yawMax,.14)),0,5),
    push:clamp(finite(base.pushN,0),0,10),
    ...(override.command||{})
  };
  const foundation=carpetLearningStage(base)==='foundation'&&!base.previewOverride;
  const intensity=clamp(finite(override.resilience,
    finite(base.resilienceDifficulty,0) *
      (active==='resilience'?1:active==='standing'?0:.15)),0,1);
  // The teacher reports a global push frontier, but its configured
  // physical ceiling may be smaller. Show the *applied* limit.
  caps.push=override.command?.push!==undefined
    ?clamp(override.command.push,0,10)
    :clamp(Math.min(caps.push,Number(base.pushMaxForceN??caps.push)),0,10);
  return {role:active,stage:p.stage,terrain:p,caps,foundation,intensity,
    heading:SPECIALIST_GUIDE[active]?.[0]||'Training',
    description:SPECIALIST_GUIDE[active]?.[1]||'Representative challenge',
    // The curriculum can be far above the current trained skill. An
    // exercise uses only a bounded fraction of its global frontier.
    commandAt(seconds){
      const phase=Math.floor(Math.max(0,seconds)/7)%6;
      const low=Math.min(caps.forward,.08);
      const cruise=clamp(caps.forward*(active==='step'?.55:.65),low, .50);
      const easy=clamp(caps.forward*.42,.015,.25);
      if(active==='standing')return {vx:Math.min(caps.forward,.035),vy:0,yaw:0,pitch:0};
      if(active==='step')return {vx:clamp(caps.forward*.55,.045,.22),vy:0,yaw:0,pitch:0};
      if(active==='slope')return {vx:clamp(caps.forward*.58,.04,.28),vy:0,yaw:0,pitch:0};
      if(active==='resilience')return {vx:easy,vy:0,yaw:0,pitch:0};
      if(active==='control')return {vx:easy,vy:0,yaw:0,pitch:0};
      if(active==='transition')return [
        {vx:cruise,vy:0,yaw:0},{vx:easy,vy:0,yaw:0},
        {vx:0,vy:0,yaw:0},{vx:-Math.min(caps.reverse,.12),vy:0,yaw:0},
        {vx:cruise,vy:0,yaw:0},{vx:0,vy:0,yaw:Math.min(caps.yaw,.35)}
      ][phase];
      return [
        {vx:cruise,vy:0,yaw:0},
        {vx:Math.min(caps.forward,.04),vy:0,yaw:0},
        {vx:-Math.min(caps.reverse,.12),vy:0,yaw:0},
        {vx:easy,vy:Math.min(caps.lateral,.10),yaw:0},
        {vx:easy,vy:-Math.min(caps.lateral,.10),yaw:0},
        {vx:easy,vy:0,yaw:Math.min(caps.yaw,.45)}
      ][phase];
    }
  };
}

export function friendlyFrontiers(lesson){
  const p=lesson.terrain,c=lesson.caps;
  return {
    slope:p.slopeDeg.toFixed(1)+'°',
    steps:Math.round(p.stepUpM*1000)+' / '+Math.round(p.stepDownM*1000)+' mm',
    speed:'['+(-c.reverse).toFixed(2)+', +'+c.forward.toFixed(2)+'] m/s',
    lateral:'±'+c.lateral.toFixed(2)+' m/s',
    yaw:'±'+c.yaw.toFixed(2)+' rad/s',
    shove:c.push.toFixed(2)+' N'
  };
}
