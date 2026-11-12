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

export function makeSpecialistScene(mode,{stepCm=5,gradeDeg=8,spawnCm=22,
  seed=735,trial=0,randomize=true}={}){
  if(mode==='meadow')return null;
  if(!SPECIALIST_SCENES.some(([key])=>key===mode))throw Error('Unknown specialist scene');
  if(![stepCm,gradeDeg,spawnCm,seed,trial].every(Number.isFinite))
    throw Error('Invalid specialist scene inputs');
  const jitter=randomize?hash(seed+Math.imul(trial+1,137))*2-1:0;
  const distanceM=clamp((spawnCm+jitter*4)/100,.14,.40);
  const signedLipM=(mode==='step-down'?-1:1)*
    clamp((stepCm+(isStepScene(mode)?jitter*.5:0))/100,.005,.08);
  const angle=clamp(gradeDeg,0,15)*Math.PI/180;
  const axis=(mode==='slope-left'||mode==='slope-right')?'y':'x';
  const inclineSign=(mode==='slope-down'||mode==='slope-right')?-1:1;
  const kind=isStepScene(mode)?'step':isSlopeScene(mode)?'slope':'flat';
  return Object.freeze({
    mode,kind,trial,distanceM,
    sideOffsetM:randomize?jitter*.03:0,
    lipM:kind==='step'?signedLipM:0,
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
