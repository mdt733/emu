// Continuous-line tracker for EMU's existing vx/vy/yaw command interface.
// Route points are geometry, not intermediate parking obligations.
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const delta=(a,b)=>wrap(a-b);
const zero=()=>({vx:0,vy:0,yaw:0,pitch:0});
const mode=c=>Math.abs(c?.vx||0)>.045?(c.vx>0?'forward':'reverse'):
  Math.abs(c?.vy||0)>.035?'lateral':Math.abs(c?.yaw||0)>.035?'pivot':'terminal';

/** Group path samples into movement runs. A pivot or a forward/reverse cusp
 * must be honoured before tracking the following line.
 */
export function buildRoutePath(route,start=route?.start){
  if(!route?.steps?.length)throw new Error('Empty navigation route');
  const origin=start||route.steps[0];
  if(![origin.x,origin.y,origin.heading].every(Number.isFinite))
    throw new Error('Nonfinite navigation route origin');
  const runs=[];let from={...origin},run=null;
  for(const step of route.steps){
    const kind=mode(step.cmd);
    if(kind==='terminal')continue;
    if(kind!==run?.mode){
      run={mode:kind,start:{...from},end:{...from},length:0,segments:[]};
      runs.push(run);
    }
    const to={x:step.x,y:step.y,heading:step.heading},length=dist(from,to);
    run.segments.push({from:{...from},to,cmd:{...step.cmd},dt:step.dt||0,
      length,s0:run.length,s1:run.length+length});
    run.length+=length;run.end=to;from=to;
  }
  return {runs,goal:{...route.steps.at(-1)},start:{...origin}};
}
function sample(run,s){
  const value=clamp(s,0,run.length);
  for(const seg of run.segments){
    if(seg.length<1e-7||seg.s1<value-1e-8)continue;
    const t=clamp((value-seg.s0)/seg.length,0,1);
    return {pose:{x:seg.from.x+(seg.to.x-seg.from.x)*t,
      y:seg.from.y+(seg.to.y-seg.from.y)*t,
      heading:wrap(seg.from.heading+delta(seg.to.heading,seg.from.heading)*t)},
      segment:seg,t};
  }
  return {pose:run.end,segment:run.segments.at(-1),t:1};
}
/** A planned step holds the *exit* speed for its arc-length segment.
 * Interpolate from the previous profiled step so speed/yaw references don't
 * jump at every ~12 cm smoothed lattice sample. Avoid mixing manoeuvre modes.
 */
export function referenceAt(run,s){
  const point=sample(run,s),seg=point.segment;
  if(!seg)return {command:zero(),profiled:false};
  const i=run.segments.indexOf(seg),prior=i>0?run.segments[i-1].cmd:seg.cmd;
  const t=point.t??1,cmd={};
  for(const key of ['vx','vy','yaw','pitch'])
    cmd[key]=(prior[key]||0)*(1-t)+(seg.cmd[key]||0)*t;
  return {command:cmd,profiled:Boolean(seg.cmd.profiled),pose:point.pose,
    segment:seg};
}
export function projectOnRun(pose,run,previous=0){
  let best=null,score=Infinity;
  const min=Math.max(0,previous-.28),max=previous+1.20;
  for(const seg of run.segments){
    if(seg.length<1e-7||seg.s1<min||seg.s0>max)continue;
    const tx=(seg.to.x-seg.from.x)/seg.length;
    const ty=(seg.to.y-seg.from.y)/seg.length;
    const u=clamp(((pose.x-seg.from.x)*tx+(pose.y-seg.from.y)*ty)/
      seg.length,0,1);
    const s=seg.s0+u*seg.length;
    if(s<min-1e-8||s>max+1e-8)continue;
    const x=seg.from.x+tx*u*seg.length,y=seg.from.y+ty*u*seg.length;
    const d=Math.hypot(pose.x-x,pose.y-y),weight=d+.005*Math.abs(s-previous);
    if(weight<score){
      score=weight;best={x,y,s,distance:d,tangent:Math.atan2(ty,tx),
        crossTrack:tx*(pose.y-y)-ty*(pose.x-x),segment:seg};
    }
  }
  if(best)return best;
  return {x:run.end.x,y:run.end.y,s:run.length,
    crossTrack:dist(pose,run.end),distance:dist(pose,run.end),
    tangent:run.end.heading,segment:run.segments.at(-1)};
}
/** Intercept the actual LINE at up to 30 degrees. Slow as cross-track
 * error grows, giving the learned gait's yaw response time to correct.
 * Use nominal trajectory yaw as feedforward to preserve intended curves.
 */
export function trackRoutePath(pose,path,runIndex,progress,limits){
  const run=path.runs[runIndex];
  if(!run)return {command:zero(),crossTrack:0,progress:0,ready:true};
  const maxYaw=Math.max(0,Number(limits.yaw_abs)||0);
  if(run.mode==='pivot'){
    const error=delta(run.end.heading,pose.heading);
    const distance=dist(pose,run.end);
    return {command:{...zero(),yaw:clamp(1.9*error,-maxYaw*.85,maxYaw*.85)},
      crossTrack:distance,progress:0,mode:'pivot',
      headingError:error,ready:Math.abs(error)<.16&&distance<.23};
  }
  const near=projectOnRun(pose,run,progress);
  const s=Math.max(progress,near.s),remaining=Math.max(0,run.length-s);
  const speed=Math.hypot(pose.vx||0,pose.vy||0);
  const previewDistance=clamp(.14+.18*speed,.14,.33);
  const ahead=referenceAt(run,s+previewDistance);
  const current=referenceAt(run,s);
  const motion=current.command;
  const heading=delta(ahead.pose.heading,pose.heading);
  const c=Math.cos(pose.heading),sn=Math.sin(pose.heading);
  const wx=c*(pose.vx||0)-sn*(pose.vy||0);
  const wy=sn*(pose.vx||0)+c*(pose.vy||0);
  const rate=Math.cos(near.tangent)*wy-Math.sin(near.tangent)*wx;
  const nominalSpeed=Math.max(.12,Math.hypot(motion.vx||0,motion.vy||0));
  // Anticipate lateral momentum: relax steering BEFORE crossing the line.
  // This keeps the same 30-degree interception cap without zigzag chasing.
  const predictedCrossTrack=near.crossTrack+.40*rate;
  const intercept=clamp(Math.atan2(1.30*predictedCrossTrack,
    nominalSpeed+.20),-Math.PI/6,Math.PI/6);
  const desired=wrap(ahead.pose.heading-intercept);
  const angle=delta(desired,pose.heading);
  // Wider errors => slower forward (or reverse), never outright frozen.
  // Drift towards (or away from) the line calls for extra correction time.
  // Track the planner's arc-length-indexed speed, rather than inventing a
  // cruising speed from multiple throttles. Gait lag changes elapsed time,
  // not which section of the route the walker must currently follow.
  const profile=current.profiled;
  const key=run.mode==='lateral'?'vy':'vx';
  const signedReference=motion[key]||0;
  const direction=Math.sign(signedReference);
  const desiredSpeed=Math.abs(signedReference);
  const measured=Number.isFinite(pose[key])?pose[key]:0;
  const achievedAlongMode=direction*measured;
  // The neural gait is not an ideal kinematic actuator: small measured-speed
  // feedback compensates for lag, but never overrides the actor's limits.
  const speedTrim=profile?clamp(.38*(desiredSpeed-achievedAlongMode),-.13,.13):0;
  const maxCommand=run.mode==='lateral'?Math.max(0,Number(limits.vy_abs)||0)
    :direction>=0?Math.max(0,Number(limits.vx_max)||0)
    :Math.max(0,-Number(limits.vx_min)||0);
  // Safety slows the walker when noticeably far from the line; within the
  // normal ~12 cm corridor, the profiled feedforward remains unattenuated.
  const trackingError=Math.max(Math.abs(near.crossTrack),Math.abs(predictedCrossTrack));
  const safetyCap=clamp(1-Math.max(0,trackingError-.12)/.95,.30,1)*
    clamp(1-Math.max(0,Math.abs(angle)-.45)/1.25,.45,1);
  const legacyEnd=profile?1:clamp((remaining+.08)/.65,.19,1);
  const legacyScale=profile?1:clamp(1-trackingError/.95,.30,1);
  // Limited preview braking: do not drive faster than can reasonably slow
  // to the reference speed in the next few decimetres.
  const future=referenceAt(run,s+previewDistance);
  const nextSpeed=Math.abs(future.command[key]||0);
  const approachLimit=profile
    ?Math.sqrt(nextSpeed*nextSpeed+2*1.0*Math.min(previewDistance,remaining))
    :Infinity;
  const targetSpeed=Math.min(maxCommand,approachLimit,
    Math.max(0,desiredSpeed+speedTrim))*safetyCap*legacyEnd*legacyScale;
  const plannedYaw=(motion.yaw||0)*(desiredSpeed>.04
    ?Math.min(1,targetSpeed/desiredSpeed):1);
  // Measured yaw damping prevents continuing to turn after alignment.
  const measuredYaw=Number.isFinite(pose.yawRate)?pose.yawRate:plannedYaw;
  const yaw=clamp(plannedYaw+1.65*angle-.50*(measuredYaw-plannedYaw),
    -maxYaw*.86,maxYaw*.86);
  const cmd={vx:run.mode==='lateral'?0:direction*targetSpeed,
    vy:run.mode==='lateral'?direction*targetSpeed:0,yaw,pitch:0};
  if(run.mode==='lateral')
    cmd.yaw=clamp(plannedYaw+1.5*heading-.45*(measuredYaw-plannedYaw),
      -maxYaw*.80,maxYaw*.80);
  const endDistance=dist(pose,run.end);
  const endHeading=Math.abs(delta(run.end.heading,pose.heading));
  return {command:cmd,crossTrack:near.crossTrack,progress:s,
    headingError:angle,remaining,mode:run.mode,
    referenceSpeed:direction*desiredSpeed,observedSpeed:measured,
    speedError:desiredSpeed-achievedAlongMode,
    ready:remaining<.20&&endDistance<.23&&endHeading<.45};
}
/** Separate fine parking: EMU can reverse, pivot and make small side steps. */
export function parkAtGoal(pose,goal,limits,{positionTolerance=.19,headingTolerance=.28}={}){
  const dx=goal.x-pose.x,dy=goal.y-pose.y,distance=Math.hypot(dx,dy);
  const c=Math.cos(pose.heading),s=Math.sin(pose.heading);
  const ex=c*dx+s*dy,ey=-s*dx+c*dy;
  const eHeading=delta(goal.heading,pose.heading);
  const f=Math.max(0,Number(limits.vx_max)||0);
  const b=Math.max(0,-Number(limits.vx_min)||0);
  const lateral=Math.max(0,Number(limits.vy_abs)||0);
  const yawMax=Math.max(0,Number(limits.yaw_abs)||0);
  const speed=Math.hypot(pose.vx||0,pose.vy||0);
  if(distance<positionTolerance&&Math.abs(eHeading)>headingTolerance*.65)
    return {...zero(),yaw:clamp(1.8*eHeading,-yawMax*.8,yawMax*.8)};
  if(distance<positionTolerance&&Math.abs(eHeading)<headingTolerance&&speed<.23)
    return zero();
  let vx=clamp(.90*ex-.25*(pose.vx||0),-b*.60,f*.36);
  let vy=clamp(.85*ey-.24*(pose.vy||0),-lateral*.75,lateral*.75);
  const yaw=clamp(1.5*eHeading,-yawMax*.72,yawMax*.72);
  const e=Math.hypot(vx/Math.max(.01,f*.45),vy/Math.max(.01,lateral*.90));
  if(e>1){vx/=e;vy/=e;}
  return {vx,vy,yaw,pitch:0};
}
