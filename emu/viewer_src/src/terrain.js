// Browser-only landscape: one continuous MuJoCo heightfield for collision
// and one matching Three.js triangle mesh for appearance. No training changes.
import {challengeCourseXml,generateChallengeCourse,CHALLENGE_RADIUS_M} from './challenge-course.js';
export const GENTLE_SLOPE_DEGREES = 1;
export const COURSE_START_X = 0.85;
export const PARK_FLAT_RADIUS_M = 2;
export const PARK_MAX_DEGREES = 10;
export const PARK_RADIUS_M = 11;
export const PARK_GRID = 129;
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const rounded = n => Number(n.toFixed(6));

const ramp = `
  <geom name="viewer_ramp_up" type="box" pos="1.25 0 0.0005"
        size="0.4 0.44 0.008" euler="0 -1 0" friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
  <geom name="viewer_ramp_top" type="box" pos="1.91 0 0.0075"
        size="0.26 0.44 0.008" friction="1.5 0.005 0.0001" rgba="0.28 0.56 0.51 1" />
  <geom name="viewer_ramp_down" type="box" pos="2.57 0 0.0005"
        size="0.4 0.44 0.008" euler="0 1 0" friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
`;

export function localSlopeLimitDeg(radius, maximum=PARK_MAX_DEGREES) {
  return clamp(radius - PARK_FLAT_RADIUS_M + 1, 0, clamp(maximum, 1, 20));
}
const smoothstep = t => {t=clamp(t,0,1);return t*t*(3-2*t);};
function rng32(seed) {
  let state = (seed | 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) | 0;
    return (state >>> 0) / 4294967296;
  };
}
export function parkHeight(x,y,maximum=PARK_MAX_DEGREES,seed=733) {
  const r=Math.hypot(x,y);
  if(r<=PARK_FLAT_RADIUS_M) return 0;
  const angle=localSlopeLimitDeg(r,maximum);
  const fade=smoothstep((r-PARK_FLAT_RADIUS_M)/1.8);
  const phase=seed*0.00713;
  const waves=0.36*Math.sin(x*0.9+phase)*Math.cos(y*0.8-phase)
    +0.27*Math.sin(1.8*x+0.7*y+phase*1.2)
    +0.19*Math.cos(2.4*y-0.8*x-phase*0.5)
    +0.13*Math.sin(x*3.3+y*2.1+phase*1.9);
  return Math.max(0,fade*0.027*angle*(0.85+waves));
}

// Produce a physical heightfield, not one collision box per tile.
// XML heightfield elevation is *top-to-bottom*, while MuJoCo model and
// Three.js geometry below use ascending world-Y rows, so XML rows are flipped.
// Lower neighbouring slope limits are applied across each connection, keeping
// spawn flat and preventing cliffs between radial difficulty bands.
export function generateTerrainPark({maxDegrees=PARK_MAX_DEGREES,seed=733,
                                     radius=PARK_RADIUS_M,resolution=PARK_GRID,
                                     trees=true,challengeRing=false}={}) {
  if(!Number.isFinite(maxDegrees)||maxDegrees<1||maxDegrees>20)
    throw new Error('Terrain maximum must be 1–20 degrees');
  if(!Number.isInteger(resolution)||resolution<17||resolution>257||resolution%2!==1)
    throw new Error('Terrain resolution must be odd, 17–257');
  const cell=2*radius/(resolution-1);
  const heights=new Float32Array(resolution*resolution);
  const index=(i,j)=>j*resolution+i;
  const pos=t=>-radius+t*cell;
  for(let j=0;j<resolution;j++)for(let i=0;i<resolution;i++){
    const x=pos(i),y=pos(j);
    // The challenge circle is OUTSIDE the spawn's 2m flat patch.
    // Preserve the actual seeded random hfield there; the obstacle
    // geoms sample the SAME final physical surface to set their Z.
    // Do not flatten the whole ring merely to make a 5mm box tidy.
    heights[index(i,j)]=parkHeight(x,y,maxDegrees,seed);
  }

  // Repeated one-sided projection: lower high spots until every grid edge
  // respects a conservative local gradient bound, including diagonals.
  const dirs=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,1],[-1,1],[1,-1]];
  for(let pass=0;pass<35;pass++){
    let changed=false;
    for(let j=0;j<resolution;j++)for(let i=0;i<resolution;i++){
      const k=index(i,j),r=Math.hypot(pos(i),pos(j));
      if(r<=PARK_FLAT_RADIUS_M){heights[k]=0;continue;}
      let maxHeight=heights[k];
      for(const [di,dj] of dirs){
        const ni=i+di,nj=j+dj;
        if(ni<0||nj<0||ni>=resolution||nj>=resolution) continue;
        const nr=Math.hypot(pos(ni),pos(nj));
        const deg=Math.min(localSlopeLimitDeg(r,maxDegrees),localSlopeLimitDeg(nr,maxDegrees));
        const bound=heights[index(ni,nj)]+Math.tan(deg*Math.PI/180)*cell*Math.hypot(di,dj)*0.55;
        maxHeight=Math.min(maxHeight,bound);
      }
      if(maxHeight<heights[k]-1e-7){heights[k]=maxHeight;changed=true;}
    }
    if(!changed)break;
  }
  // The max/min are explicit: the generated terrain does not depend on an
  // external file, texture or mutable global MuJoCo mesh state.
  const maximum=Math.max(0.001,...heights);
  const sampleHeight=(x,y)=>{
    const fx=clamp((x+radius)/cell,0,resolution-1.00001);
    const fy=clamp((y+radius)/cell,0,resolution-1.00001);
    const i=Math.floor(fx),j=Math.floor(fy),u=fx-i,v=fy-j;
    const a=heights[index(i,j)],b=heights[index(i+1,j)];
    const c=heights[index(i,j+1)],d=heights[index(i+1,j+1)];
    return a*(1-u)*(1-v)+b*u*(1-v)+c*(1-u)*v+d*u*v;
  };
  const treeList=[];
  if(trees) {
    if(challengeRing){
      // The original radial park's trees begin beyond 3.5m. Removing
      // trees from the 4m ring otherwise pushes EVERY trunk outside the
      // new arena camera and looks like we lost the forest again.
      // A small inner grove is visible within the 4m course, but its
      // trunks stay >1.5m inside the challenge and do not obstruct the
      // eight crossing lanes (or the centre -> first-station approach).
      for(const angle of [Math.PI/4,3*Math.PI/4,5*Math.PI/4,7*Math.PI/4]){
        const x=2.25*Math.cos(angle),y=2.25*Math.sin(angle);
        treeList.push({
          x,y,z:sampleHeight(x,y),trunkHeight:1.38,
          trunkRadius:.075,canopyRadius:.50,shade:.48,
        });
      }
    }
    const random=rng32(seed+9041);
    for(let attempt=0;attempt<650&&treeList.length<17;attempt++){
      const direction=random()*Math.PI*2;
      const distance=3.5+Math.sqrt(random())*(radius-4.15);
      const x=Math.cos(direction)*distance,y=Math.sin(direction)*distance;
      // Preserve the forest, but don't plant trunks on the lesson track
      // or where the robot approaches/lands beside a shallow step.
      if(challengeRing && Math.abs(distance-CHALLENGE_RADIUS_M)<1.08)continue;
      if(treeList.some(t=>Math.hypot(x-t.x,y-t.y)<1.55))continue;
      const trunkHeight=1.05+random()*0.85;
      treeList.push({x,y,z:sampleHeight(x,y),trunkHeight,
                     trunkRadius:0.065+random()*0.045,
                     canopyRadius:0.41+random()*0.28,
                     shade:random()});
    }
  }
  return {radius,resolution,cell,heights,maxHeight:maximum,trees:treeList,sampleHeight,
          maxDegrees,flatRadius:PARK_FLAT_RADIUS_M,seed,challengeRing};
}

function parkXml(park) {
  const {resolution,radius,maxHeight,heights,trees}=park;
  // The hfield parser normalizes the inline elevation itself. All elevations
  // are already nonnegative with a zero floor, so z=0 stays exactly at ground.
  const elevations=[];
  for(let j=resolution-1;j>=0;j--)for(let i=0;i<resolution;i++)
    elevations.push(rounded(heights[j*resolution+i]/maxHeight));
  const asset=`<hfield name="viewer_park_hfield" nrow="${resolution}" ncol="${resolution}"
    size="${radius} ${radius} ${rounded(maxHeight)} 0.20"
    elevation="${elevations.join(' ')}" />`;
  const terrain='<geom name="viewer_park_ground" type="hfield" hfield="viewer_park_hfield" pos="0 0 0" friction="1.5 0.005 0.0001" rgba="0.31 0.53 0.25 1" />';
  const trunks=trees.map((t,i)=>`<geom name="viewer_tree_trunk_${i}" type="cylinder"
    pos="${rounded(t.x)} ${rounded(t.y)} ${rounded(t.z+t.trunkHeight/2)}"
    size="${rounded(t.trunkRadius)} ${rounded(t.trunkHeight/2)}"
    friction="1.5 0.005 0.0001" rgba="0.29 0.18 0.10 1" />`).join('\n');
  return {asset,geometry:terrain+'\n'+trunks};
}

export function withTerrain(robotXml,mode='flat',options={}) {
  if(typeof robotXml!=='string'||!robotXml.includes('</worldbody>'))
    throw new Error('Cannot locate worldbody in policy robot XML');
  if(/viewer_ramp_up|viewer_park_hfield|viewer_tree_trunk_|viewer_challenge_/.test(robotXml))
    throw new Error('Viewer terrain already inserted');
  if(mode==='flat')return robotXml;
  if(mode==='ramp')return robotXml.replace('</worldbody>',ramp+'\n</worldbody>');
  if(mode!=='park'&&mode!=='course')throw new Error('Unknown terrain mode');
  // Challenge mode is the PARK plus actual collidable obstacle geometry,
  // not a replacement for the forest/grass. The park owns all ground
  // collision, and the original plane is removed exactly once.
  const course=mode==='course'
    ? options.course||generateChallengeCourse(options) : null;
  const park=options.park||generateTerrainPark({
    ...options,challengeRing:mode==='course'});
  const {asset,geometry}=parkXml(park);
  const challenge=course?challengeCourseXml(course,{groundHeightAt:park.sampleHeight}):'';
  // The source floor must not overlap the hfield at the spawn point: that
  // would double contacts and compromise the 120M actor's apparent walking.
  let xml=robotXml.replace(/<geom\s+name="floor"[^>]*\/>/,'');
  if(xml===robotXml)throw new Error('Expected the named MuJoCo floor plane');
  if(xml.includes('</asset>'))xml=xml.replace('</asset>',asset+'\n</asset>');
  else xml=xml.replace('<worldbody>','<asset>'+asset+'</asset>\n<worldbody>');
  return xml.replace('</worldbody>',geometry+(challenge?'\n'+challenge:'')+'\n</worldbody>');
}
export const withGentleCourse=xml=>withTerrain(xml,'ramp');

// Read the *actual generated ground* below the midpoint between the feet.
// This is terrain grade, NOT chassis pitch, commanded attitude, or a
// physically estimated IMU angle. The arrow points uphill RELATIVE to the
// robot's current heading: up=forward, right=robot right, left=robot left.
export function surfaceSlope(mode,park,x,y,robotHeading=0) {
  if (![x,y,robotHeading].every(Number.isFinite))
    return {supported:false,degrees:0,arrowDegrees:0};
  let gx=0,gy=0;
  if(mode==='park'||mode==='course') {
    if(!park||Math.abs(x)>=park.radius||Math.abs(y)>=park.radius)
      return {supported:false,degrees:0,arrowDegrees:0};
    const delta=Math.max(park.cell*0.5,0.05);
    const xl=Math.max(-park.radius,x-delta),xr=Math.min(park.radius,x+delta);
    const yl=Math.max(-park.radius,y-delta),yr=Math.min(park.radius,y+delta);
    gx=(park.sampleHeight(xr,y)-park.sampleHeight(xl,y))/(xr-xl);
    gy=(park.sampleHeight(x,yr)-park.sampleHeight(x,yl))/(yr-yl);
  } else if(mode==='ramp') {
    // Browser 1-degree course: up-ramp and down-ramp are aligned to world X.
    if(Math.abs(y)<0.44 && x>=COURSE_START_X && x<=COURSE_START_X+0.8)
      gx=Math.tan(Math.PI/180);
    else if(Math.abs(y)<0.44 && x>=2.17 && x<=2.97)
      gx=-Math.tan(Math.PI/180);
  } else if(mode!=='flat')
    throw new Error('Unknown terrain mode');
  const grade=Math.hypot(gx,gy);
  const degrees=Math.atan(grade)*180/Math.PI;
  const relative=Math.atan2(Math.sin(Math.atan2(gy,gx)-robotHeading),
                            Math.cos(Math.atan2(gy,gx)-robotHeading));
  return {supported:true,degrees,arrowDegrees:-relative*180/Math.PI,
          flat:degrees<0.05};
}
