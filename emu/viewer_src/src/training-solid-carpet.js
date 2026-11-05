// Browser reconstruction of v5041's 9 REAL physical tilted-box carpet.
// This is a representative world at measured role/frontier, not a replay of
// the trainer's per-environment random seed or its moment-to-moment births.
// The box sizes, 50cm modulo-three slots, directional profile and top-plane
// construction mirror emu/terrain_carpet_mjx.py. No actor reward inputs.
import { CARPET_DEFAULT, carpetRoleLimits } from './training-carpet.js';

export const SOLID_TILE_M=.5;
export const SOLID_HALF_XY=.258;
export const SOLID_HALF_Z=.270;
export const SOLID_COUNT=9;
const PROFILE_HALF=128;
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const mod=(n,m)=>((n%m)+m)%m;
const radians=d=>d*Math.PI/180;

function hash(i,j,seed,salt){
  let x=(Math.imul(i,374761393)+Math.imul(j,668265263)+
         Math.imul(seed,1442695041)+Math.imul(salt,1274126177))|0;
  x=Math.imul(x^(x>>>13),1274126177);
  return ((x^(x>>>16))>>>0)/4294967296;
}

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

function directionalProfiles(seed,up,down){
  let x=0,y=0;const X=[],Y=[];
  for(let k=-PROFILE_HALF;k<PROFILE_HALF;k++){
    const difficulty=.95+.55*Math.min(1,Math.abs(k)/14);
    const sx=(hash(k,0,seed,101)<.5?.90*up:-.90*down)*
      (.75+.25*hash(k,0,seed,102))*difficulty;
    const smaller=Math.min(up,down);
    const sy=(hash(0,k,seed,103)<.5?.18*smaller:-.18*smaller)*
      (.3+.7*hash(0,k,seed,104));
    x+=sx;y+=sy;X.push(x);Y.push(y);
  }
  const x0=X[PROFILE_HALF],y0=Y[PROFILE_HALF];
  return {x:X.map(v=>v-x0),y:Y.map(v=>v-y0)};
}

export function createSolidCarpet(base=CARPET_DEFAULT,requested='auto'){
  const limits=terrainRoleEnvelope(base,requested);
  const seed=(limits.seed??735)|0;
  const profile=directionalProfiles(seed,limits.stepUpM,limits.stepDownM);
  const grade=Math.tan(radians(limits.slopeDeg));
  const maxTransition=Math.min(limits.slopeDeg,limits.transitionDeg);
  const angleTransition=Math.tan(radians(Math.max(0,maxTransition)));
  const f=1.15;
  const variable=Math.min(.65*grade,angleTransition/(f*SOLID_TILE_M*1.6));
  const smoothBase=Math.max(0,.92*grade-variable);
  const phase=hash(0,0,seed,140)*2*Math.PI;
  const p1=hash(0,0,seed,141)*2*Math.PI;
  const p2=hash(0,0,seed,142)*2*Math.PI;
  const cache=new Map();
  const at=(i,j)=>{
    if(!Number.isInteger(i)||!Number.isInteger(j))throw Error('Cell indices must be integers');
    const key=i+','+j;
    if(cache.has(key))return cache.get(key);
    const x=i*SOLID_TILE_M,y=j*SOLID_TILE_M;
    const meanLevel=profile.x[clamp(i+PROFILE_HALF,0,255)]+
      profile.y[clamp(j+PROFILE_HALF,0,255)]+
      (hash(i,j,seed,107)-.5)*Math.min(limits.stepUpM,limits.stepDownM)*.10;
    const vertex=(dx,dy)=>{
      const gx=x+dx,gy=y+dy;
      return meanLevel+smoothBase*(gx*Math.cos(phase)+gy*Math.sin(phase))+
        variable/f*Math.sin(f*gx+p1)*Math.cos(.85*f*gy+p2);
    };
    const a=vertex(0,0),b=vertex(SOLID_TILE_M,0);
    const c=vertex(0,SOLID_TILE_M),d=vertex(SOLID_TILE_M,SOLID_TILE_M);
    const top=(a+b+c+d)/4;
    const gx=.5*(b+d-a-c)/SOLID_TILE_M;
    const gy=.5*(c+d-a-b)/SOLID_TILE_M;
    const denom=Math.hypot(gx,gy,1);
    const normal=[-gx/denom,-gy/denom,1/denom];
    const w=Math.sqrt((1+normal[2])*.5);
    const quat=[w,-normal[1]/(2*w),normal[0]/(2*w),0];
    const tile={i,j,slot:mod(i,3)+3*mod(j,3),
      top,normal,quat,vertices:[a,b,c,d],
      center:[x+.25-SOLID_HALF_Z*normal[0],
              y+.25-SOLID_HALF_Z*normal[1],
              top-SOLID_HALF_Z*normal[2]],
      x:x+.25,y:y+.25};
    cache.set(key,tile);
    // Endless rolling terrain must not accumulate an unbounded world map.
    // Recomputing a returned tile is deterministic from global coordinates.
    if(cache.size>1024)cache.delete(cache.keys().next().value);
    return tile;
  };
  const sampleHeight=(x,y)=>{
    const i=Math.floor(x/SOLID_TILE_M),j=Math.floor(y/SOLID_TILE_M);
    const t=at(i,j),n=t.normal;
    return t.top-(n[0]*(x-t.x)+n[1]*(y-t.y))/n[2];
  };
  const window=(cx=0,cy=0)=>{
    const cells=[];
    for(let j=cy-1;j<=cy+1;j++)for(let i=cx-1;i<=cx+1;i++)
      cells.push(at(i,j));
    return cells.sort((a,b)=>a.slot-b.slot);
  };
  const boxes=window(0,0);
  return {mode:'carpet',physics:'nine-solid-boxes',role:limits.role,
    stage:limits.stage,limits,seed,tiles:boxes,window,
    sampleHeight,at,radius:11,cell:SOLID_TILE_M,
    trees:[],features:[],tileSizeM:SOLID_TILE_M,flatRadius:0,
    preview:false};
}

// Match the training recycler's crucial no-foot-on-discarded-support rule.
// A conservative XY overhang test is intentional: no support under a ball
// foot may teleport when a row/column of tiles is recycled.
export function safeCarpetShift(oldTiles,newTiles,feet,radius=.018){
  if(!Array.isArray(feet)||feet.length!==2)return false;
  const newKeys=new Set(newTiles.map(t=>t.i+','+t.j));
  for(const tile of oldTiles){
    if(newKeys.has(tile.i+','+tile.j))continue;
    // EXACT projected half-extent of the rotated SOLID box, including
    // its sideways-offset body centre. The trainer uses the same AABB
    // projection in feet_clear_of_abandoned_solids(). Checking only the
    // nominal tile centre is unsafe on a pitched or rolled support.
    const [w,x,y,z]=tile.quat;
    const rxx=1-2*(y*y+z*z),rxy=2*(x*y-w*z),rxz=2*(x*z+w*y);
    const ryx=2*(x*y+w*z),ryy=1-2*(x*x+z*z),ryz=2*(y*z-w*x);
    const halfX=(Math.abs(rxx)+Math.abs(rxy))*SOLID_HALF_XY+
      Math.abs(rxz)*SOLID_HALF_Z+radius+.002;
    const halfY=(Math.abs(ryx)+Math.abs(ryy))*SOLID_HALF_XY+
      Math.abs(ryz)*SOLID_HALF_Z+radius+.002;
    if(feet.some(p=>Math.abs(p[0]-tile.center[0])<=halfX&&
                     Math.abs(p[1]-tile.center[1])<=halfY))return false;
  }
  return true;
}
