// Browser-only physical terrain: generated *MuJoCo collision boxes*, not scenery.
// The flat starting circle is 2m. Beyond it, permitted local grades grow
// by 1 degree per metre (2° at 3m, etc), ending at the selected maximum.
// Deterministic multi-scale waves resemble smoothed midpoint displacement.
// A conservative derivative bound limits grades; tiles overlap to avoid gaps.
// Robot physics, policy and MJX training are never edited here.
export const GENTLE_SLOPE_DEGREES = 1;
export const COURSE_START_X = 0.85;
export const PARK_FLAT_RADIUS_M = 2;
export const PARK_MAX_DEGREES = 10;
const escapeFloat = n => Number(n.toFixed(5));
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const course = `
    <geom name="viewer_ramp_up" type="box" pos="1.25 0 0.0005" size="0.4 0.44 0.008" euler="0 -1 0" friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
    <geom name="viewer_ramp_top" type="box" pos="1.91 0 0.0075" size="0.26 0.44 0.008" friction="1.5 0.005 0.0001" rgba="0.28 0.56 0.51 1" />
    <geom name="viewer_ramp_down" type="box" pos="2.57 0 0.0005" size="0.4 0.44 0.008" euler="0 1 0" friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
`;

export function localSlopeLimitDeg(radius, maximum=PARK_MAX_DEGREES) {
  return clamp(radius - PARK_FLAT_RADIUS_M + 1, 0, clamp(maximum, 1, 20));
}
function smoothstep(t) { t=clamp(t,0,1); return t*t*(3-2*t); }
export function parkHeight(x,y, maximum=PARK_MAX_DEGREES, seed=733) {
  const r = Math.hypot(x,y);
  if (r <= PARK_FLAT_RADIUS_M) return 0;
  const phase = seed * 0.00713;
  // Smooth blend at inner edge. Max local derivative is conservatively
  // restricted by the final tile gradient projection below.
  const fade = smoothstep((r-PARK_FLAT_RADIUS_M)/1.5);
  const waves = Math.sin(x*1.6+phase)*Math.cos(y*1.3-phase)
    + 0.32*Math.sin(x*3.0+y*1.4+phase*1.7)
    + 0.5*Math.sin(x*0.65-y*0.7-phase);
  return fade * (0.12 + 0.075*waves) * clamp(localSlopeLimitDeg(r,maximum)/10,0,1);
}
export function generateTerrainPark({maxDegrees=PARK_MAX_DEGREES, seed=733, radius=5, cell=0.5}={}) {
  if (!Number.isFinite(maxDegrees) || maxDegrees < 1 || maxDegrees > 20) throw new Error('Terrain maximum must be 1–20 degrees');
  const count = 2*Math.ceil(radius/cell)+1;
  if (count*count>900) throw new Error('Terrain would exceed browser geometry budget');
  const start=-(count-1)*cell/2;
  const height=Array.from({length:count},(_,j)=>Array.from({length:count},(_,i)=>{
    const x=start+i*cell, y=start+j*cell;
    return Math.hypot(x,y)<=PARK_FLAT_RADIUS_M+cell*0.75?0:Math.max(0,parkHeight(x,y,maxDegrees,seed));
  }));
  // Project tile heights into neighbour grade constraints. The lower slope
  // tier wins across a boundary; fixes elevated/stepped edges around spawn.
  for(let pass=0;pass<16;pass++) {
    let changed=false;
    for(let j=0;j<count;j++)for(let i=0;i<count;i++){
      const x=start+i*cell,y=start+j*cell;
      if(Math.hypot(x,y)<=PARK_FLAT_RADIUS_M+cell*0.75){height[j][i]=0;continue;}
      const dirs=[[i-1,j],[i+1,j],[i,j-1],[i,j+1]];
      for(const [ii,jj] of dirs){
        if(ii<0||jj<0||ii>=count||jj>=count)continue;
        const nx=start+ii*cell,ny=start+jj*cell;
        const limit=Math.tan(Math.PI/180*Math.min(localSlopeLimitDeg(Math.hypot(x,y),maxDegrees),localSlopeLimitDeg(Math.hypot(nx,ny),maxDegrees)))*cell*0.65;
        const next=Math.min(height[j][i],height[jj][ii]+limit);
        if(next<height[j][i]-1e-9){height[j][i]=next;changed=true;}
      }
    }
    if(!changed)break;
  }
  const tiles=[];
  for(let j=0;j<count;j++)for(let i=0;i<count;i++){
    const x=start+i*cell,y=start+j*cell, r=Math.hypot(x,y);
    if(r<=PARK_FLAT_RADIUS_M+cell*0.25)continue;
    const at=(ii,jj)=>height[clamp(jj,0,count-1)][clamp(ii,0,count-1)];
    const dx=(at(i+1,j)-at(i-1,j))/(cell*(i===0||i===count-1?1:2));
    const dy=(at(i,j+1)-at(i,j-1))/(cell*(j===0||j===count-1?1:2));
    const maxGrade=Math.tan(Math.PI/180*localSlopeLimitDeg(r,maxDegrees));
    const scale=Math.min(1,maxGrade/Math.max(1e-9,Math.hypot(dx,dy)));
    const gx=dx*scale,gy=dy*scale;
    tiles.push({x,y,z:height[j][i],pitch:-Math.atan(gx)*180/Math.PI,roll:Math.atan(gy)*180/Math.PI,grade:Math.hypot(gx,gy),limit:maxGrade});
  }
  return {tiles,flatRadius:PARK_FLAT_RADIUS_M,maximum:maxDegrees,cell};
}
export function withTerrain(robotXml, mode='flat', options={}) {
  if(typeof robotXml!=='string'||!robotXml.includes('</worldbody>'))throw new Error('Cannot locate worldbody in policy robot XML');
  if(robotXml.includes('viewer_ramp_up')||robotXml.includes('viewer_park_'))throw new Error('Viewer terrain already inserted');
  if(mode==='flat')return robotXml;
  let content='';
  if(mode==='ramp')content=course;
  else if(mode==='park') {
    const {tiles,cell}=generateTerrainPark(options);
    const half=cell*0.505; // small overlap prevents falling into tile seams
    content=tiles.map((t,i)=>`<geom name="viewer_park_${i}" type="box" pos="${escapeFloat(t.x)} ${escapeFloat(t.y)} ${escapeFloat(t.z-0.016)}" size="${escapeFloat(half)} ${escapeFloat(half)} 0.016" euler="${escapeFloat(t.roll)} ${escapeFloat(t.pitch)} 0" friction="1.5 0.005 0.0001" rgba="0.23 0.47 0.39 1" />`).join('\n');
  } else throw new Error('Unknown terrain mode');
  return robotXml.replace('</worldbody>',`${content}\n</worldbody>`);
}
export const withGentleCourse=xml=>withTerrain(xml,'ramp');
