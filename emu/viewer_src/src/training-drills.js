// Read-only, reproducible training-*inspired* CPU spectator scenarios.
// The actual trainer uses 8 deformable convex prisms in MJX. Here the same
// signed 1m tile seam is reconstructed with mutable MuJoCo heightfields.
// Never call this a bitwise GPU episode replay or a frozen independent exam.
export const DRILL_ROLES=Object.freeze([
  'step','command','resilience','slope','transition','control','standing'
]);
const clamp=(v,lo,hi)=>Math.min(hi,Math.max(lo,v));
const finite=(v,defaultValue)=>v!==''&&v!==null&&v!==undefined&&
  Number.isFinite(Number(v))?Number(v):defaultValue;

export function seededDraw(seed){
  let state=(Number(seed)>>>0)||1;
  return ()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;
    return state/4294967296;};
}

export function sampleTrainingDrill(limits,role,sequence=1){
  if(!DRILL_ROLES.includes(role))throw Error('Unknown specialist drill');
  if(!Number.isSafeInteger(sequence)||sequence<1)throw Error('Invalid drill sequence');
  const seed=(finite(limits.seed,735)>>>0);
  const draw=seededDraw((seed^Math.imul(sequence,0x9e3779b9)^
    Math.imul(DRILL_ROLES.indexOf(role)+1,0x85ebca6b))>>>0);
  const vxMax=clamp(finite(limits.vxHi,.6),.05,3);
  const vxMin=clamp(finite(limits.vxLo,-.2),-2,0);
  const vy=clamp(finite(limits.vyMax,.2),0,2);
  const yaw=clamp(finite(limits.yawMax,.8),0,5);
  const bound=(lo,hi)=>lo+(hi-lo)*draw();
  let rise=0,slopeDeg=0,stepKind='none',spawnX=.5,spawnY=.5;
  if(role==='step'){
    // Exact profile: x < 1m is source; x >= 1m is destination,
    // all y-cells have the SAME height. Match the h40 +X Academy
    // reset at 0.70..0.85 m; full body faces the seam.
    const downAllowed=finite(limits.schoolDescentUnlockedStep,-1)>=0;
    const downFraction=clamp(finite(limits.schoolDownShare,
      downAllowed?.1:0),0,.4);
    const downhill=downAllowed&&draw()<downFraction;
    if(downhill){
      rise=-clamp(finite(limits.academyStepDownM,.003),.003,.08);
      stepKind='downhill';
    }else{
      const top=clamp(finite(limits.academyStepUpM,.05),.05,.08);
      const piece=draw();
      // Refined curriculum: 50% anchor, 35% near current frontier,
      // 15% bridge. Before uncoached graduation the whole cohort
      // stays +50mm and descent is disabled.
      rise=piece<.50?.05:piece<.85?top:bound(.05,top);
      stepKind=rise<=.05001?'5cm retention':'frontier uphill';
    }
    rise=clamp(rise*(.96+.08*draw()),-.08,.08);
    spawnX=bound(.70,.85);
    spawnY=bound(.45,.55);
  }else if(role==='slope'){
    slopeDeg=(draw()<.5?-1:1)*bound(.25,.95)*
      clamp(finite(limits.slopeDeg,5),0,15);
  }else if(role==='transition'){
    slopeDeg=(draw()<.5?-1:1)*bound(.15,.55)*
      clamp(finite(limits.slopeDeg,5),0,15);
  }
  const forward=(role==='step'?clamp(vxMax*.55,.045,.22):
    role==='standing'?0:
    role==='control'?clamp(vxMax*.28,0,.22):
    role==='resilience'?clamp(vxMax*.40,.04,.25):
    role==='slope'?clamp(vxMax*.50,.04,.30):
    clamp(vxMax*bound(.28,.90),.03,vxMax));
  return {
    role,sequence,seed,stepKind,riseM:rise,slopeDeg,
    spawnX,spawnY,wallX:1,
    initialCommand:{vx:forward,vy:0,yaw:0,pitch:0},
    durationS:role==='step'?14:role==='standing'?10:18,
    caps:{vxMax,vxMin,vy,yaw},
    commandSeed:Math.floor(draw()*4294967295),
    // Up to the viewer to choose genuinely physical force previews.
    shoveIntensity:role==='resilience'?
      clamp(finite(limits.resilienceDifficulty,.4),0,1):0,
    shoveN:clamp(finite(limits.pushN,0),0,10),
    sourceStep:finite(limits.step,0),
    previewOnly:true
  };
}

export function drillTileCorners(i,j,episode){
  if(!episode)return null;
  const slope=Math.tan(episode.slopeDeg*Math.PI/180);
  const at=(x)=>((x>=episode.wallX?episode.riseM:0)+
    slope*(x-episode.spawnX));
  // hfield samples [southwest,southeast,northwest,northeast], 1m tiles.
  return [at(i),at(i+1),at(i),at(i+1)];
}

export function drillCommand(episode,elapsedS){
  const role=episode.role,c=episode.caps;
  if(role==='step'||role==='slope'||role==='resilience'||
     role==='control'||role==='standing')return episode.initialCommand;
  // Deterministic command bursts across the FULL demonstrated envelope;
  // not 7-second repeating gentle forward-only demonstration.
  const period=role==='command'?3:2.2;
  const block=Math.floor(Math.max(0,elapsedS)/period);
  const random=seededDraw((episode.commandSeed^Math.imul(block+1,2654435761))>>>0);
  const band=block%6;
  const v=(lo,hi)=>lo+(hi-lo)*random();
  if(role==='command'){
    if(band===0)return {vx:v(.70*c.vxMax,c.vxMax),vy:0,yaw:0,pitch:0};
    if(band===1)return {vx:v(c.vxMin,Math.min(0,c.vxMin*.3)),vy:0,yaw:0,pitch:0};
    if(band===2)return {vx:v(0,.35*c.vxMax),vy:v(.50*c.vy,c.vy),yaw:0,pitch:0};
    if(band===3)return {vx:v(0,.35*c.vxMax),vy:-v(.50*c.vy,c.vy),yaw:0,pitch:0};
    if(band===4)return {vx:.12*c.vxMax,vy:0,yaw:v(.5*c.yaw,c.yaw),pitch:0};
    return {vx:.12*c.vxMax,vy:0,yaw:-v(.5*c.yaw,c.yaw),pitch:0};
  }
  return [
    {vx:v(.65*c.vxMax,c.vxMax),vy:0,yaw:0,pitch:0},
    {vx:0,vy:0,yaw:0,pitch:0},
    {vx:v(c.vxMin,0),vy:0,yaw:0,pitch:0},
    {vx:v(.2*c.vxMax,.6*c.vxMax),vy:v(-c.vy,c.vy),yaw:0,pitch:0},
    {vx:v(.2*c.vxMax,.6*c.vxMax),vy:0,yaw:v(-c.yaw,c.yaw),pitch:0},
    {vx:0,vy:0,yaw:0,pitch:0}
  ][band];
}

export function drillCompletion(episode,pose,previous={right:0,left:0,stable:0}){
  const next={...previous},x=finite(pose.x,-100),y=finite(pose.y,100);
  if(episode.role!=='step')return {...next,complete:false};
  const eligible= x>=episode.wallX+.10&&x<episode.wallX+.75&&
    Math.abs(y-episode.spawnY)<.4&&pose.upright===true;
  // Both feet must have independently contacted the destination top.
  // This is a CPU contact-sample approximation to the 15-frame MJX gate.
  const foot=pose.footSupported||[false,false];
  next.right=foot[0]?Math.min(20,next.right+1):0;
  next.left=foot[1]?Math.min(20,next.left+1):0;
  const transfer=next.right>=3&&next.left>=3;
  next.stable=eligible&&transfer&&foot.some(Boolean)?next.stable+1:0;
  return {...next,complete:next.stable>=15,transfer};
}
