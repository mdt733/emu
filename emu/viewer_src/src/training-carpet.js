// v5041 Training Carpet: same browser MuJoCo heightfield and grass renderer
// as Terrain Park, different seeded curriculum-driven surface. Park unchanged.
import { generateTerrainPark, PARK_RADIUS_M } from './terrain.js';

export const CARPET_ROLES = Object.freeze(['standing','command','resilience','slope','step','transition','control']);
export const CARPET_TILE_M = .5;
export const CARPET_DEFAULT = Object.freeze({
  seed:735,slopeDeg:5,transitionDeg:2,stepUpM:.012,stepDownM:.016
});
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};

function hash(i,j,seed,salt=0){
  let h=(Math.imul(i,374761393)+Math.imul(j,668265263)+
         Math.imul(seed,1442695041)+Math.imul(salt,1274126177))|0;
  h=Math.imul(h^(h>>>13),1274126177);
  return ((h^(h>>>16))>>>0)/4294967296;
}

function csvFields(line){
  // Python's csv.DictWriter quotes arbitrary per-role events, which may
  // contain commas. Honor RFC4180 quotes and doubled literal quotes.
  const cols=[];let field='',quoted=false;
  for(let k=0;k<line.length;k++){
    const ch=line[k];
    if(ch==='"'){
      if(quoted&&line[k+1]==='"'){field+='"';k++;}
      else quoted=!quoted;
    }else if(ch===','&&!quoted){cols.push(field);field='';}
    else field+=ch;
  }
  cols.push(field);
  return !quoted?cols:null;
}
export function parseCarpetMetrics(text){
  if(typeof text!=='string'||text.length>5_000_000)throw Error('Invalid or oversized metrics.csv');
  const lines=text.replace(/\r/g,'').split('\n').filter(s=>s.trim());
  if(lines.length<2)throw Error('metrics.csv has no data rows');
  const names=csvFields(lines[0]);if(!names)throw Error('Malformed CSV header');
  const index=Object.fromEntries(names.map((n,i)=>[n.trim(),i]));
  if(index.step===undefined)throw Error('Missing step column');
  let best=null,bestStep=-1;
  for(const line of lines.slice(1)){
    const cols=csvFields(line);
    if(!cols||cols.length!==names.length)continue;
    const n=Number(cols[index.step]);
    if(Number.isFinite(n)&&n>=bestStep){bestStep=n;best=cols;}
  }
  if(!best)throw Error('No complete metrics rows');
  const row={};for(const [k,i] of Object.entries(index))row[k]=best[i];
  return {latest_metrics:row,training_step:bestStep};
}

export function trainingLimits(data){
  if(!data||typeof data!=='object')throw Error('Expected training metrics or snapshot JSON');
  const row=data.latest_metrics||data;
  const raw=data.measured_frontiers||{};
  const as=(x,defaultValue,lo,hi)=>{
    if(x==null||x==='')return defaultValue;
    const n=Number(x);return Number.isFinite(n)?clamp(n,lo,hi):defaultValue;
  };
  const slopeDeg=as(raw.slope_deg??row.slope_max_deg,CARPET_DEFAULT.slopeDeg,0,25);
  const cfg=data.training_config||{};
  const step=as(data.training_step??row.step??data.step,0,0,1e12);
  const ordinary=as(row.terrain_height_m,0,0,.25);
  const academyUp=as(row.academy_height_up_m,ordinary,0,.25);
  const academyDown=as(row.academy_height_down_m,ordinary,0,.25);
  // A STEP birth is not automatically an Academy birth at 60M.
  // _carpet_birth selects these PER EPISODE using academy_active:
  // ordinary STEP picks max(general terrain, Academy signed frontiers),
  // while the four-lip Academy uses its own independent UP/DOWN caps.
  // Keep BOTH so a spectator can explicitly inspect either cohort.
  const ordinaryStepUpM=as(raw.step_up_m,Math.max(ordinary,academyUp),0,.25);
  const ordinaryStepDownM=as(raw.step_down_m,Math.max(ordinary,academyDown),0,.25);
  const academyStepUpM=as(raw.academy_step_up_m,academyUp,0,.25);
  const academyStepDownM=as(raw.academy_step_down_m,academyDown,0,.25);
  const stepUpM=ordinaryStepUpM,stepDownM=ordinaryStepDownM;
  const reportedTransition=raw.slope_transition_deg??row.terrain_slope_transition_frontier_deg;
  const transitionMeasured=reportedTransition!=null&&reportedTransition!==''&&Number.isFinite(Number(reportedTransition));
  // The cap is a CONFIGURED physical geometry bound, not a newly
  // demonstrated "transition skill". Never present it as measured ability.
  const configuredTransition=cfg.carpet_transition_cap_deg;
  const transitionConfigured=Number.isFinite(Number(configuredTransition))&&configuredTransition!=null;
  const transitionDeg=transitionMeasured?as(reportedTransition,0,0,15)
    :transitionConfigured?as(configuredTransition,3,0,15)
    :Math.min(3,slopeDeg*.5);
  const seed=as(data.terrain_preview?.seed??cfg.seed??data.seed,735,0,2147483647);
  return {seed:Math.floor(seed),slopeDeg,transitionDeg,stepUpM,stepDownM,
    ordinaryStepUpM,ordinaryStepDownM,academyStepUpM,academyStepDownM,
    academyEligible:cfg.academy_stream===true &&
      step>=as(cfg.academy_intro_steps,60_000_000,0,1e12),
    // Actual global teacher-frontier targets. These ARE NOT achieved gait
    // velocities and do not imply every specialist gets maximum difficulty.
    vxHi:as(row.frontier_vx_hi,.06,0,3),
    vxLo:as(row.frontier_vx_lo,-.02,-2,0),
    vyMax:as(row.frontier_vy,.035,0,2),
    yawMax:as(row.frontier_yaw,.14,0,5),
    pushN:as(row.frontier_push_n,0,0,10),
    resilienceDifficulty:as(row.resilience_difficulty,0,0,1),
    slopeExposure:as(row.slope_exposure,1,0,1),
    stepExposure:as(row.terrain_exposure,1,0,1),
    pushMaxForceN:as(cfg.push_max_force_n,0.35,.01,10),
    step,run:data.run_name||data.source_provenance?.run_name||data.run_dir||'Imported training',
    foundationSteps:as(cfg.foundation_steps,20_000_000,0,1e12),
    academyIntroSteps:as(cfg.academy_intro_steps,60_000_000,0,1e12),
    academyStartSteps:as(cfg.academy_start_steps,80_000_000,0,1e12),
    academyRampSteps:as(cfg.academy_ramp_steps,20_000_000,0,1e12),
    academyStream:cfg.academy_stream===true,
    trainingCarpet:cfg.terrain_carpet===true,
    transitionMeasured,transitionConfigured,
    source:'training',physicsMode:data.physics_mode||'source curriculum metrics',
    warnings:data.warnings||[]};
}

export function carpetRoleLimits(b=CARPET_DEFAULT,role='step'){
  if(!CARPET_ROLES.includes(role))throw Error('Unknown terrain specialist');
  const factors={
    standing:[0,0,0,0],command:[.2,.16,.15,.1],
    resilience:[.7,.6,.65,.35],slope:[1,1,.5,.25],
    step:[.5,.5,1,.70],transition:[.65,1,.55,.32],
    control:[.15,.1,.1,.08]
  };
  const [s,t,p,frequency]=factors[role];
  return {seed:Math.floor(b.seed??735),
    slopeDeg:clamp((b.slopeDeg??5)*s,0,25),
    transitionDeg:clamp((b.transitionDeg??2)*t,0,15),
    stepUpM:clamp((b.stepUpM??.012)*p,0,.25),
    stepDownM:clamp((b.stepDownM??.016)*p,0,.25),
    frequency};
}

function smoothHeight(x,y,p){
  if(p.slopeDeg<=0)return 0;
  const slope=Math.tan(p.slopeDeg*Math.PI/180);
  const corner=Math.tan(p.transitionDeg*Math.PI/180);
  const f=.89,a=hash(0,0,p.seed,31)*Math.PI*2;
  const q1=hash(0,0,p.seed,32)*Math.PI*2;
  const q2=hash(0,0,p.seed,33)*Math.PI*2;
  const q3=hash(0,0,p.seed,34)*Math.PI*2;
  const amp=Math.min(slope*.37/f,corner*.68/(CARPET_TILE_M*f*f*2.1));
  return slope*.47*(Math.cos(a)*x+Math.sin(a)*y)+
    amp*(.72*Math.sin(f*x+q1)*Math.cos(.73*f*y+q2)+
         .28*Math.sin(.49*f*x+.83*f*y+q3));
}
function offset(i,j,p){
  const cap=Math.min(p.stepUpM,p.stepDownM);
  if(cap<=0||hash(i,j,p.seed,41)>=p.frequency)return 0;
  return (hash(i,j,p.seed,43)<.5?-1:1)*cap*(.22+.28*hash(i,j,p.seed,42));
}
function heightAt(x,y,p){
  const ix=Math.floor(x/CARPET_TILE_M),iy=Math.floor(y/CARPET_TILE_M);
  // Honest limited-resolution step approximation for browser hfield.
  // Tile faces carry different offsets; contact edges are rasterized
  // into finite triangles, not infinite vertical faces.
  const fade=smooth((Math.hypot(x,y)-.65)/.9);
  return fade*(smoothHeight(x,y,p)+offset(ix,iy,p));
}

export function generateTrainingCarpet(base=CARPET_DEFAULT,role='step',{
  radius=PARK_RADIUS_M,resolution=177,trees=true
}={}){
  if(resolution<33||resolution>257||resolution%2===0)throw Error('Invalid carpet grid size');
  const limits=carpetRoleLimits(base,role);
  const heights=new Float32Array(resolution*resolution);
  const cell=2*radius/(resolution-1);
  let lowest=Infinity,highest=-Infinity;
  for(let j=0;j<resolution;j++)for(let i=0;i<resolution;i++){
    const z=heightAt(-radius+i*cell,-radius+j*cell,limits);
    heights[j*resolution+i]=z;
    lowest=Math.min(lowest,z);highest=Math.max(highest,z);
  }
  // MuJoCo normalizes heightfield samples to [0,1]. Keep the *actual*
  // original world heights by moving the hfield geom to lowest instead.
  for(let i=0;i<heights.length;i++)heights[i]-=lowest;
  const maxHeight=Math.max(.001,highest-lowest);
  const sampleHeight=(x,y)=>{
    const fx=clamp((x+radius)/cell,0,resolution-1.00001);
    const fy=clamp((y+radius)/cell,0,resolution-1.00001);
    const i=Math.floor(fx),j=Math.floor(fy),u=fx-i,v=fy-j;
    const a=heights[j*resolution+i],b=heights[j*resolution+i+1];
    const c=heights[(j+1)*resolution+i],d=heights[(j+1)*resolution+i+1];
    return lowest+a*(1-u)*(1-v)+b*u*(1-v)+c*(1-u)*v+d*u*v;
  };
  const scenery=trees?generateTerrainPark({maxDegrees:1,seed:limits.seed,
    trees:true,features:false}):null;
  const treelist=(scenery?.trees||[]).map(t=>({...t,z:sampleHeight(t.x,t.y)}));
  return {mode:'carpet',radius,resolution,cell,heights,maxHeight,
    collisionOffsetZ:lowest,trees:treelist,features:[],sampleHeight,
    limits,role,seed:limits.seed,flatRadius:.65,challengeLevel:'training',
    tileSizeM:CARPET_TILE_M,preview:true};
}
