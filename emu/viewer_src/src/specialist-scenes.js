// Read-only viewer laboratory scenes. These feed only the browser's CPU
// heightfields; never an actor observation, training curriculum or MJX learner.
// Each 1m square keeps two planar triangles and a real lip at x=1 for steps.
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const hash=(n)=>{let a=(n|0)^0x9e3779b9;
  a=Math.imul(a^(a>>>16),0x85ebca6b);
  a=Math.imul(a^(a>>>13),0xc2b2ae35);
  return ((a^(a>>>16))>>>0)/4294967296;};

export const SPECIALIST_SCENES=Object.freeze([
  ['meadow','Infinite Meadow · general movement'],
  ['command','Commands · ±vx, ±vy, ±yaw'],
  ['slope-up','Slopes · uphill'],
  ['slope-down','Slopes · downhill'],
  ['slope-left','Slopes · transverse left'],
  ['slope-right','Slopes · transverse right'],
  ['step-up','Steps · uphill Academy'],
  ['step-down','Steps · downhill Academy'],
  ['resilience','Resilience · shoves'],
  ['transition','Transitions · changes of command'],
  ['control','Control · efficient walking']
]);
export const isStepScene=mode=>mode==='step-up'||mode==='step-down';
export const isSlopeScene=mode=>mode.startsWith('slope-');
export const sceneRole=mode=>isStepScene(mode)?'step':
  isSlopeScene(mode)?'slope':
  ['command','resilience','transition','control'].includes(mode)?mode:'auto';

export function makeSpecialistScene(mode,{stepCm=5,gradeDeg=8,spawnCm=22.5,
  approachDeg=0,seed=735,trial=0,randomize=true}={}){
  if(mode==='meadow')return null;
  if(!SPECIALIST_SCENES.some(([key])=>key===mode))throw Error('Unknown specialist scene');
  if(![stepCm,gradeDeg,spawnCm,approachDeg,seed,trial].every(Number.isFinite))
    throw Error('Invalid specialist scene inputs');
  // B's actual Academy reset uses x=.70+.15*U and y=.45+.10*U.
  // At the x=1 seam this means uniform 15..30cm before the lip, and
  // ±5cm lateral offset. Keep independent seeded draws per trial.
  const xDraw=hash(seed+Math.imul(trial+1,137));
  const yDraw=hash(seed+Math.imul(trial+1,157)+13);
  const hDraw=hash(seed+Math.imul(trial+1,191)+71);
  const aDraw=hash(seed+Math.imul(trial+1,211)+37);
  const distanceM=randomize
    ?clamp((spawnCm+15*(xDraw-.5))/100,.15,.30)
    :clamp(spawnCm/100,.15,.38);
  const signedLipM=(mode==='step-down'?-1:1)*
    clamp((stepCm+(isStepScene(mode)&&randomize ? .5*(hDraw-.5) : 0))/100,.005,.08);
  const angle=clamp(gradeDeg,0,15)*Math.PI/180;
  const axis=(mode==='slope-left'||mode==='slope-right')?'y':'x';
  const inclineSign=(mode==='slope-down'||mode==='slope-right')?-1:1;
  const kind=isStepScene(mode)?'step':isSlopeScene(mode)?'slope':'flat';
  return Object.freeze({
    mode,kind,trial,distanceM,
    phaseRad:randomize?2*Math.PI*hash(seed+Math.imul(trial+1,317)+113):0,
    sideOffsetM:randomize ? .10*(yDraw-.5) : 0,
    lipM:kind==='step'?signedLipM:0,
    approachDeg:kind==='step'?clamp(approachDeg+
      (randomize?6*(aDraw-.5):0),-15,15):0,
    gradeDeg:kind==='slope'?clamp(gradeDeg,0,15):0,
    axis,grade:kind==='slope'?inclineSign*Math.tan(angle):0,
    edgeX:1,
    // A normal lesson is independent of a teacher reward or hidden PPO reset.
    horizonS:isStepScene(mode)?12:15
  });
}

export function specialistTileCorners(i,j,scenario){
  if(!scenario)return null;
  let elevation;
  if(scenario.kind==='step'){
    const h=i>=scenario.edgeX?scenario.lipM:0;
    elevation=()=>h;
  }else if(scenario.kind==='slope'){
    elevation=(x,y)=>scenario.grade*(scenario.axis==='y'?y:x);
  }else{
    elevation=()=>0;
  }
  return [elevation(i,j),elevation(i+1,j),
    elevation(i,j+1),elevation(i+1,j+1)];
}

export function supportedFirstLip({scenario,bodyX,feet,fallen}){
  // This is a browser OBSERVATION, not a certified frozen MJX crossing.
  // Require both feet physically on the target tile, each sole near its
  // true surface, and the body beyond the lip. Hold for five control frames.
  if(!scenario||scenario.kind!=='step'||fallen||!Number.isFinite(bodyX)||
     bodyX<scenario.edgeX+.08||!Array.isArray(feet)||feet.length!==2)
    return false;
  return feet.every(foot=>Number.isFinite(foot.x)&&
    Number.isFinite(foot.soleZ)&&Number.isFinite(foot.groundZ)&&
    foot.x>scenario.edgeX+.025&&
    foot.soleZ-foot.groundZ>=-.012&&foot.soleZ-foot.groundZ<=.020);
}


/** Viewer-only exercise instructions, bounded by the frozen actor's command limits.
 *  No historic curriculum/frontier parser and no privileged Academy goal.
 *  Physical scenery comes exclusively from makeSpecialistScene above.
 */
export function specialistExerciseCommand(mode,seconds,limits){
  const role=sceneRole(mode);
  const forward=clamp(Number(limits.vx_max||.06),.01,3);
  const reverse=clamp(Math.abs(Number(limits.vx_min||-.02)),0,2);
  const lateral=clamp(Math.abs(Number(limits.vy_abs||.035)),0,2);
  const yaw=clamp(Math.abs(Number(limits.yaw_abs||.14)),0,5);
  const phase=Math.floor(Math.max(0,seconds)/7)%6;
  const low=Math.min(forward,.08);
  const cruise=clamp(forward*(role==='step'?.55:.65),low,.5);
  const easy=clamp(forward*.42,.015,.25);
  const command=(vx,vy=0,yawRate=0)=>({vx,vy,yaw:yawRate,pitch:0});
  if(role==='step')return command(clamp(forward*.55,.045,.22));
  if(role==='slope')return command(clamp(forward*.58,.04,.28));
  if(role==='resilience'||role==='control')return command(easy);
  if(role==='transition')return [
    command(cruise),command(easy),command(0),command(-Math.min(reverse,.12)),
    command(cruise),command(0,0,Math.min(yaw,.35))
  ][phase];
  if(role==='command')return [
    command(cruise),command(Math.min(forward,.04)),
    command(-Math.min(reverse,.12)),
    command(easy,Math.min(lateral,.10)),
    command(easy,-Math.min(lateral,.10)),
    command(easy,0,Math.min(yaw,.45))
  ][phase];
  return command(0);
}
