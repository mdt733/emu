// Current 360-input array-only observer: optical flow, 4x4 actual terrain
// rays, IMU noise/delay. No retired tripod or 220/232-input sensor stack.
const MIN=.035,MAX=.6;
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const norm=a=>Math.sqrt(dot(a,a));
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const normalized=a=>{const n=Math.max(1e-8,norm(a));return a.map(x=>x/n);};
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const scale=(a,s)=>a.map(x=>x*s);
// MuJoCo xmat is row-major body to world, so columns map local basis vectors.
function bodyToWorld(R,v){
  return [R[0]*v[0]+R[1]*v[1]+R[2]*v[2],
          R[3]*v[0]+R[4]*v[1]+R[5]*v[2],
          R[6]*v[0]+R[7]*v[1]+R[8]*v[2]];
}

export function rayHeight(origin, direction, heightAt) {
  // Support comes from the REAL park or prism surface, not a ghost z=0
  // plane. Carpet is finite: outside its 2x2 tiles heightAt returns NaN.
  // A NaN must NEVER be interpreted as a closer terrain hit.
  const at=t=>origin[2]+direction[2]*t-
    heightAt(origin[0]+direction[0]*t,origin[1]+direction[1]*t);
  if(direction[2]>-.3) return 0;
  const near=at(MIN),far=at(MAX);
  if(!Number.isFinite(near)||near<0)return 0;
  let a=MIN,b=MAX;
  if(!Number.isFinite(far)){
    // The ray exits the finite fixture by MAX. Search for an earlier
    // supported crossing; otherwise report an invalid ToF cell.
    let found=false,last=MIN;
    for(let i=1;i<=16;i++){
      const t=MIN+(MAX-MIN)*i/16;
      const v=at(t);
      if(!Number.isFinite(v))return 0;
      if(v<=0){a=last;b=t;found=true;break;}
      last=t;
    }
    if(!found)return 0;
  }else if(far>0)return 0;
  for(let i=0;i<19;i++){
    const m=(a+b)/2,v=at(m);
    if(!Number.isFinite(v))return 0;
    if(v>0)a=m;else b=m;
  }
  return (a+b)/2;
}

function gaussian(){
  return Math.sqrt(-2*Math.log(Math.max(1e-8,Math.random())))*
    Math.cos(2*Math.PI*Math.random());
}

// Diagnostic-only ST VL53L8CX (8x8-capable package in native 4x4 mode).
// Four sub-rays per zone geometrically approximate a sampled 8x8 footprint;
// the real hardware does NOT return 64 measurements when configured as 4x4.
// No output enters the champion observation history.
export const DEPTH_FOV_DEG = 45;
export const DEPTH_ZONES = 4;
export const DEPTH_MAX_M = 1.5;
// 45° square angular frustum on a single sensor image plane. Independent
// image-plane axes prevent horizontal FoV collapsing at 90° downward tilt.
// A 4x4 zone occupies 1/4 of each image-plane extent. Each quarter-zone
// offset is 1/8 of the half-extent; these approximate footprints, not ST
// ranging/histogram sub-pixels.
const DEPTH_HALF = Math.tan(DEPTH_FOV_DEG*Math.PI/360);
const DEPTH_OFFSETS = [[0,0],[-1,-1],[-1,1],[1,-1],[1,1]];

function opticalRay(R, tilt, u, v, yaw=0) {
  const t=tilt*Math.PI/180, c=Math.cos(t), s=Math.sin(t);
  // The sensor looks forward (+X) until pitched down towards -Z.
  // Its horizontal image axis is body Y; vertical image axis is the
  // tilted local optical "up" vector. At tilt=90° the footprint is
  // square on horizontal flat ground, not a degenerate V/line.
  const length=1/Math.hypot(1,u,v);
  const x=(c+s*v)*length, y=u*length, z=(-s+c*v)*length;
  const a=yaw*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a);
  const fx=x*ca-y*sa,fy=x*sa+y*ca;
  return [
    R[0]*fx+R[1]*fy+R[2]*z,
    R[3]*fx+R[4]*fy+R[5]*z,
    R[6]*fx+R[7]*fy+R[8]*z,
  ];
}

export function depthProbeRays(torso, R,
  {forward=0,lateral=0,tilt=70,yaw=0,height=-.030}={}) {
  if (![forward,lateral,tilt,yaw,height].every(Number.isFinite) ||
      forward<0||forward>.20||Math.abs(lateral)>.12||
      tilt<0||tilt>90||Math.abs(yaw)>30)
    throw new Error('Depth mount requires forward 0–200mm, lateral ±120mm, tilt 0–90°, yaw ±30°');
  const mounted=[
    R[0]*forward+R[1]*lateral+R[2]*height,
    R[3]*forward+R[4]*lateral+R[5]*height,
    R[6]*forward+R[7]*lateral+R[8]*height,
  ];
  const origin=torso.map((v,i)=>v+mounted[i]);
  const directions=[];
  for(let row=0;row<4;row++)for(let col=0;col<4;col++){
    const zone=[];
    const u=(col-1.5)*DEPTH_HALF/2;
    const v=(1.5-row)*DEPTH_HALF/2;
    for(const [offU,offV] of DEPTH_OFFSETS)
      zone.push(opticalRay(R,tilt,u+offU*DEPTH_HALF/8,
                          v+offV*DEPTH_HALF/8,yaw));
    directions.push(zone);
  }
  return {origin,directions};
}

// Five-by-five BEAM BOUNDARIES outline the true square 4x4 zone footprint.
// On uneven ground the viewer casts each boundary ray onto actual geoms;
// edges with no real hit are left open, never drawn as imaginary surfaces.
export function depthGridRays(R,{tilt=70,yaw=0}={}) {
  if(!Number.isFinite(tilt)||tilt<0||tilt>90||
     !Number.isFinite(yaw)||Math.abs(yaw)>30)
    throw new Error('Depth grid tilt must be 0–90°');
  return Array.from({length:25},(_,i)=>{
    const row=Math.floor(i/5),col=i%5;
    const u=(col/2-1)*DEPTH_HALF;
    const v=(1-row/2)*DEPTH_HALF;
    return opticalRay(R,tilt,u,v,yaw);
  });
}

// Geometry-only ray/sphere test for the REAL 36mm foot collision balls.
// A painted foot image is not sensor occlusion; this uses MuJoCo geom centres.
export function footSphereHit(origin,dir,centre,radius=.018) {
  if(!Number.isFinite(radius)||radius<=0)return Infinity;
  const delta=sub(origin,centre),b=dot(dir,delta);
  const c=dot(delta,delta)-radius*radius,disc=b*b-c;
  if(disc<0)return Infinity;
  const root=Math.sqrt(disc),near=-b-root,far=-b+root;
  return near>1e-6?near:far>1e-6?far:Infinity;
}

export function readDepthProbe(pose, cast, {subrays=4, noise=0, dropout=0}={}) {
  // cast() returns a first physical intersection distance or
  // {distance,kind:'self'|'terrain'}; no hidden surface-height lookup.
  if(![1,4,5].includes(subrays))throw new Error('Expected 1, 4 or 5 depth rays per zone');
  const samples=subrays===1?[0]:subrays===4?[1,2,3,4]:[0,1,2,3,4];
  const ranges=[],valid=[],hitPoints=[],selfOccluded=[],footOccluded=[],footHitCounts=[],mixed=[];
  for(const zone of pose.directions){
    const values=[];
    for(const j of samples){
      const result=cast(pose.origin,zone[j],DEPTH_MAX_M);
      const distance=Number(typeof result==='number'?result:result?.distance);
      if(Number.isFinite(distance)&&distance>=.02&&distance<=DEPTH_MAX_M)
        values.push({distance,kind:result?.kind||'terrain',surface:result?.surface??null,ray:zone[j]});
    }
    values.sort((a,b)=>a.distance-b.distance);
    const coverage=values.length/samples.length;
    let ok=coverage>=.60;
    // Lower median: edge zones prefer an actual nearer surface instead of
    // averaging lower and upper platforms into a fictitious "ramp".
    const picked=ok?values[Math.floor((values.length-1)/2)]:null;
    let value=picked?picked.distance:0;
    if(ok && noise) value+=noise*(Math.random()-.5)*2;
    ok=ok&&value>=.02&&value<=DEPTH_MAX_M&&Math.random()>=dropout;
    ranges.push(ok?value:0);
    valid.push(ok);
    const footCount=values.filter(hit=>hit.kind==='foot').length;
    footOccluded.push(footCount>0);
    footHitCounts.push(footCount);
    selfOccluded.push(ok&&values.some(hit=>hit.kind==='self'||hit.kind==='foot'));
    // Raw slant-range variation happens even on perfectly FLAT ground;
    // a mixed-zone warning instead means rays actually hit different
    // visible objects/surfaces. This flag is diagnostic-only privilege.
    const distinct=new Set(values.map(v=>v.surface??v.kind));
    mixed.push(ok&&distinct.size>1);
    const direction=ok?picked.ray:zone[0];
    const length=ok?picked.distance:.55;
    hitPoints.push(pose.origin.map((p,k)=>p+length*direction[k]));
  }
  return {ranges,valid,origin:pose.origin,hitPoints,
    selfOccluded,footOccluded,footHitCounts,mixed};
}


/**
 * Diagnostic terrain elevation relative to a horizontal world-Z reference.
 *
 * A measured slant distance is NOT a vertical height: each zone has its own
 * direction and may select a different subray on an obstacle edge. Use the
 * actual chosen hit point, which also removes the flat-ground 20–30cm fan.
 * World z=0 is the explicit default flat-ground datum, not a learned
 * per-zone offset or a sensor channel granted to PPO.
 *
 * Invalid and self-occluded zones remain unknown (null), never "0cm terrain".
 * Values are not clipped; the UI clamps only its +/-15cm colour scale.
 */
export function terrainHeightOffsetsCm(reading, referenceZ=0) {
  if(!Number.isFinite(referenceZ))throw new Error('Invalid flat-ground reference');
  return reading.ranges.map((_,i)=>{
    if(!reading.valid[i]||reading.selfOccluded[i])return null;
    const z=reading.hitPoints[i]?.[2];
    return Number.isFinite(z)?100*(z-referenceZ):null;
  });
}


// Exact current actor ray geometry, derived from the physical inspector's
// one image-plane frustum rather than a second pitch/yaw formula.
const CURRENT_TOF_REV='vl53l8cx_array_plane_square45_below30_v3';
const IDENTITY=[1,0,0,0,1,0,0,0,1];
export const CURRENT_RAYS_BODY=depthProbeRays([0,0,0],IDENTITY,
  {tilt:70,forward:0,height:-.03,lateral:0,yaw:0})
  .directions.map(zone=>zone[0]);
export function terrainActorRays(R, rays=CURRENT_RAYS_BODY) {
  if(!Array.isArray(rays)||rays.length!==16||
     rays.some(v=>!Array.isArray(v)||v.length!==3))
    throw Error('Actor requires the 16 physical 4x4 directions');
  return rays.map(v=>bodyToWorld(R,v));
}

// First physical floor or world-box intersection, masked by nearest 18mm foot
// sphere. This mirrors environment._terrain_4x4_ranges at frozen inference.
// Other robot parts do NOT masquerade as a known terrain surface. The park's
// shared rendered heightfield is supplied as a fallback when no plane exists.
export function actorTerrainDistance(origin, ray, model, data, footGeomIds, heightAt=null) {
  let terrain=Infinity;
  const gpos=data.geom_xpos,gmat=data.geom_xmat,types=model.geom_type,
        sizes=model.geom_size, bodies=model.geom_bodyid;
  for(let g=0;g<Number(model.ngeom);g++) {
    const body=Number(bodies[g]);
    // Static terrain lives in world body 0. The v5041 nine-solid training
    // carpet uses *mocap* bodies, which are equally physical terrain and
    // must be included. Ordinary articulated robot bodies are never
    // allowed to pose as ground in the actor's 4x4 observation.
    if(body!==0 && Number(model.body_mocapid?.[body]??-1)<0)continue;
    const t=Number(types[g]);
    const ix=3*g,im=9*g;
    const delta=[origin[0]-gpos[ix],origin[1]-gpos[ix+1],origin[2]-gpos[ix+2]];
    if(t===0) { // plane: geom-xmat local Z is the upward world normal
      const nz=[gmat[im+2],gmat[im+5],gmat[im+8]];
      const denom=dot(ray,nz);
      if(denom < -1e-6){
        const dist=-dot(delta,nz)/denom;
        if(dist>0)terrain=Math.min(terrain,dist);
      }
    } else if(t===6) { // arbitrarily oriented finite box: slab intersection
      let near=-Infinity,far=Infinity;
      for(let axis=0;axis<3;axis++){
        const v=[gmat[im+axis],gmat[im+3+axis],gmat[im+6+axis]];
        const o=dot(delta,v),d=dot(ray,v),half=sizes[ix+axis];
        if(Math.abs(d)<1e-7) {
          if(Math.abs(o)>half){far=-Infinity;break;}
        } else {
          const ta=(-half-o)/d,tb=(half-o)/d;
          near=Math.max(near,Math.min(ta,tb));
          far=Math.min(far,Math.max(ta,tb));
        }
      }
      if(far>=Math.max(near,0)){
        const dist=near>0?near:far;
        if(dist>0)terrain=Math.min(terrain,dist);
      }
    }
  }
  if(heightAt) {
    const h=rayHeight(origin,ray,heightAt);
    if(h>0)terrain=Math.min(terrain,h);
  }
  let foot=Infinity;
  for(const g of footGeomIds){
    const i=g*3;
    const delta=[origin[0]-gpos[i],origin[1]-gpos[i+1],origin[2]-gpos[i+2]];
    const b=dot(ray,delta),c=dot(delta,delta)-sizes[i]*sizes[i],disc=b*b-c;
    if(disc<0)continue;
    const first=-b-Math.sqrt(disc),last=-b+Math.sqrt(disc),hit=first>1e-6?first:last;
    if(hit>1e-6)foot=Math.min(foot,hit);
  }
  return foot>=terrain && terrain>=.04 && terrain<=1.5 ? terrain : 0;
}

export class TerrainActorObserver {
  constructor(contract) {
    if(contract?.schema!=='terrain_360')throw new Error('Schema-9 terrain contract required');
    this.settings=contract.settings;
    if(contract.sensor_model_revision!==CURRENT_TOF_REV||
       !Array.isArray(contract.tof4x4_rays_body)||
       JSON.stringify(contract.tof4x4_mount_body_m)!==
         JSON.stringify([0,0,-.03]))
      throw Error('Only the current physical VL53L8CX geometry is supported');
    this.rays=contract.tof4x4_rays_body;
    this.mount=contract.tof4x4_mount_body_m;
    this.frame=0;
    this.ring=[];
  }
  sample(pose,model,data,footGeomIds,noisy=true) {
    const scale=noisy?this.settings.sensor_noise_scale:0;
    const h=pose.terrainHeightAt||null;
    const mounted=bodyToWorld(pose.R,this.mount);
    const origin=pose.torso.map((n,i)=>n+mounted[i]);
    const ranges=terrainActorRays(pose.R,this.rays).map(ray=>{
      const measured=actorTerrainDistance(origin,ray,model,data,footGeomIds,h);
      return measured ? clamp(measured+scale*.002*gaussian(),.04,1.5) : 0;
    });
    return {ranges,valid:ranges.map(x=>x>0)};
  }
  reset(pose,model,data,feet,noisy=true) {
    this.frame=0;
    const s=this.sample(pose,model,data,feet,noisy);
    this.ring=Array.from({length:4},()=>({ranges:s.ranges.slice(),valid:s.valid.slice()}));
  }
  advance(pose,model,data,feet,noisy=true) {
    this.frame++;
    const old=this.ring[3],st=this.settings;
    const due=this.frame%st.tof_period_frames===0;
    const missed=noisy&&Math.random()<st.tof_dropout_prob;
    const fresh=due?this.sample(pose,model,data,feet,noisy):old;
    const next={
      ranges:due?(missed?Array(16).fill(0):fresh.ranges.slice()):old.ranges.slice(),
      valid:due?(missed?Array(16).fill(false):fresh.valid.slice()):old.valid.slice()
    };
    this.ring.shift();this.ring.push(next);
  }
  read(){
    const s=this.ring[3-this.settings.tof_delay_frames];
    return {ranges:s.ranges,valid:s.valid};
  }
}


// Unified 360-input actor: no physical tripod ToFs. Infer one *optional*
// smooth ground-plane estimate from the same delayed 16 rays that the network
// receives directly. Reject mixed-level/riser fits rather than invent a slope.
export function fitArrayGround(ranges,valid,gravity,
                               raysBody=CURRENT_RAYS_BODY,mountZ=-.03){
  const invalid={valid:false,depth:0,forward:0,side:0,confidence:0,
                 residual:Infinity,count:0};
  const rays=terrainActorRays(IDENTITY,raysBody);
  const G=[[.00005,0,0],[0,.00005,0],[0,0,.0000001]],rhs=[0,0,0];
  let count=0;
  const points=[];
  for(let i=0;i<16;i++){
    const d=ranges[i];
    if(!valid[i]||!Number.isFinite(d)||d<.04||d>1.5)continue;
    const pt=scale(rays[i],d),v=[pt[0],pt[1],1];
    points.push(pt);count++;
    for(let j=0;j<3;j++){
      rhs[j]+=v[j]*pt[2];
      for(let k=0;k<3;k++)G[j][k]+=v[j]*v[k];
    }
  }
  if(count<6 || norm(gravity)<.1)return {...invalid,count};
  // Tiny, regularised 3x3 least-squares solve; no external libraries.
  const aug=G.map((row,i)=>[...row,rhs[i]]);
  for(let col=0;col<3;col++){
    let pivot=col;
    for(let r=col+1;r<3;r++)
      if(Math.abs(aug[r][col])>Math.abs(aug[pivot][col]))pivot=r;
    if(Math.abs(aug[pivot][col])<1e-9)return {...invalid,count};
    [aug[pivot],aug[col]]=[aug[col],aug[pivot]];
    const v=aug[col][col];
    for(let j=col;j<4;j++)aug[col][j]/=v;
    for(let r=0;r<3;r++){
      if(r===col)continue;
      const mul=aug[r][col];
      for(let j=col;j<4;j++)aug[r][j]-=mul*aug[col][j];
    }
  }
  const [a,b,c]=aug.map(row=>row[3]);
  const residual=Math.sqrt(points.reduce((acc,p)=>
    acc+(a*p[0]+b*p[1]+c-p[2])**2,0)/count);
  const up=scale(normalized(gravity),-1);
  let n=normalized([-a,-b,1]);
  if(dot(n,up)<0)n=scale(n,-1);
  const cosine=dot(n,up);
  const fwd=normalized(sub([1,0,0],scale(up,up[0])));
  const side=cross(up,fwd);
  const depth=-c;
  const planeValid=residual<.008&&cosine>.25&&depth>=.035&&depth<=.60
    &&Number.isFinite(depth);
  const steepIndices=[9,10,13,14],projected=[];
  for(const i of steepIndices)if(valid[i]&&Number.isFinite(ranges[i])&&ranges[i]>=.04)
    projected.push(ranges[i]*-rays[i][2]);
  const fallback=projected.length>=2?
    projected.reduce((a,b)=>a+b,0)/projected.length:0;
  // Fit/fallback samples originate 30mm below the optical-flow camera.
  // Express both depths in torso coordinates, as the Python trainer does.
  const opticalDepth=(planeValid?depth:fallback)-mountZ;
  const depthValid=(planeValid||projected.length>=2)&&
    opticalDepth>=.035&&opticalDepth<=.60;
  return {valid:planeValid,depthValid,depth:depthValid?opticalDepth:0,
    forward:planeValid?Math.atan2(-dot(n,fwd),cosine):0,
    side:planeValid?Math.atan2(-dot(n,side),cosine):0,
    confidence:planeValid?clamp(count/12,0,1)*clamp(1-residual/.008,0,1):0,
    residual,count};
}

export class ArrayFlowObserver {
  constructor(contract){
    if(contract?.sensor_model_revision!==CURRENT_TOF_REV||
       !Array.isArray(contract.tof4x4_rays_body))
      throw new Error('Expected the current physical 360-input array sensor');
    this.settings=contract.settings;this.ring=[];this.frame=0;this.bias=[0,0,0];
    this.rays=contract.tof4x4_rays_body;
    this.mount=contract.tof4x4_mount_body_m;
  }
  synth(pose,model,data,noisy=true){
    const st=this.settings,ns=noisy?st.sensor_noise_scale:0;
    const dir=bodyToWorld(pose.R,[0,0,-1]);
    const depth=actorTerrainDistance(pose.torso,dir,model,data,[],
                                    pose.terrainHeightAt||null);
    const gyro=pose.gyro.map((x,i)=>x+this.bias[i]+gaussian()*ns*.025);
    const good=depth>=.035&&depth<=.60;
    const flow=good?[
      -pose.vbody[0]/depth+pose.gyro[1]+gaussian()*ns*.035,
      -pose.vbody[1]/depth-pose.gyro[0]+gaussian()*ns*.035
    ]:[0,0];
    return {gyro,flow,flowGyro:gyro.slice(),flowValid:good,flowAge:0};
  }
  reset(pose,model,data,noisy=true){
    this.frame=0;
    this.bias=noisy?this.bias.map(()=>gaussian()*this.settings.gyro_bias_std_rad_s):[0,0,0];
    const sample=this.synth(pose,model,data,noisy);
    this.ring=Array.from({length:4},()=>({
      ...sample,gyro:sample.gyro.slice(),flow:sample.flow.slice(),
      flowGyro:sample.flowGyro.slice()
    }));
  }
  advance(pose,model,data,noisy=true){
    this.frame++;
    const st=this.settings,last=this.ring[3];
    const raw=this.synth(pose,model,data,noisy);
    const due=this.frame%st.flow_period_frames===0;
    const missed=noisy&&Math.random()<st.flow_dropout_prob;
    const next={
      gyro:raw.gyro,
      flow:due?(missed?[0,0]:raw.flow):last.flow.slice(),
      flowGyro:due?raw.flowGyro:last.flowGyro.slice(),
      flowValid:due?(raw.flowValid&&!missed):last.flowValid,
      flowAge:due?0:last.flowAge+1
    };
    this.ring.shift();this.ring.push(next);
  }
  read(gravity,array){
    const s=this.settings,flow=this.ring[3-s.flow_delay_frames];
    const ground=fitArrayGround(array.ranges,array.valid,gravity,
                               this.rays,this.mount[2]);
    const good=ground.depthValid&&flow.flowValid;
    const depth=ground.valid?ground.depth:0;
    const velocity=good?[
      -ground.depth*(flow.flow[0]-flow.flowGyro[1]),
      -ground.depth*(flow.flow[1]+flow.flowGyro[0])
    ]:[0,0];
    const fresh=clamp(1-.25*(flow.flowAge/s.flow_period_frames),.25,1);
    return {gyro:this.ring[3].gyro,velocity,
      velocityConfidence:good?fresh:0,depth,
      forwardGrade:ground.forward,sideGrade:ground.side,
      groundConfidence:ground.confidence};
  }
}
