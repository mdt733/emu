// Browser-side observation emulator for v5016's schema-8 sensor actor.
// One downward ray per ToF mount, centre optical flow, IMU bias, independent
// sampling clocks and delayed delivery. Not a substitute for physical sensors.
const MIN=.035,MAX=.6;
const MOUNTS=[[.035,-.05,-.025],[.035,.05,-.025],[-.04,0,-.025]];
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
  // Browser visual heightfield is not an infinite plane. Bisection gives a
  // plausible narrow ToF reading against its actual display surface.
  const at=t=>origin[2]+direction[2]*t-
    heightAt(origin[0]+direction[0]*t,origin[1]+direction[1]*t);
  if(direction[2]>-.3 || at(MIN)<0 || at(MAX)>0) return 0;
  let a=MIN,b=MAX;
  for(let i=0;i<19;i++){
    const m=(a+b)/2;
    if(at(m)>0)a=m;else b=m;
  }
  return (a+b)/2;
}

export function inferTripod(ranges,gravity){
  const invalid={valid:false,depth:0,forward:0,side:0,confidence:0};
  if(ranges.some(x=>!Number.isFinite(x)||x<MIN||x>MAX))return invalid;
  const points=MOUNTS.map((p,i)=>[p[0],p[1],p[2]-ranges[i]]);
  const raw=cross(sub(points[1],points[0]),sub(points[2],points[0]));
  const area=.5*norm(raw);
  if(area<.0001 || norm(gravity)<.1)return invalid;
  const up=scale(normalized(gravity),-1);
  let n=normalized(raw);
  if(dot(n,up)<0)n=scale(n,-1);
  const cosine=dot(n,up);
  let fwd=sub([1,0,0],scale(up,up[0]));
  const length=norm(fwd);
  if(length<.5 || cosine<=.1)return invalid;
  fwd=normalized(fwd);
  const side=cross(up,fwd);
  const depth=-dot(n,points[0])/(Math.abs(n[2])>.1?n[2]:1);
  if(!Number.isFinite(depth)||depth<MIN||depth>MAX)return invalid;
  return {valid:true,depth,
    forward:Math.atan2(-dot(n,fwd),cosine),
    side:Math.atan2(-dot(n,side),cosine),
    confidence:clamp(area/.00375,0,1)*clamp(cosine/.6,0,1)};
}

function gaussian(){
  return Math.sqrt(-2*Math.log(Math.max(1e-8,Math.random())))*
    Math.cos(2*Math.PI*Math.random());
}

export class SensorObserver {
  constructor(contract){
    if(contract?.schema!=='sim2real_232')throw new Error('Sensor contract missing');
    this.settings=contract.settings;
    this.ring=[];
    this.frame=0;
    this.bias=[0,0,0];
  }
  // pose = torso position, body-to-world R, world-surface heightAt,
  // local root translational velocity, local freejoint gyro.
  synth(pose,noisy=true){
    const st=this.settings;
    const ns=noisy?st.sensor_noise_scale:0;
    const dir=bodyToWorld(pose.R,[0,0,-1]);
    const ranges=MOUNTS.map(m=>{
      const xyz=bodyToWorld(pose.R,m);
      const start=pose.torso.map((v,i)=>v+xyz[i]);
      const measured=rayHeight(start,dir,pose.heightAt);
      if(!measured)return 0;
      const value=measured+gaussian()*ns*.0015;
      return value>=MIN&&value<=MAX?value:0;
    });
    const depth=rayHeight(pose.torso,dir,pose.heightAt);
    const gyro=pose.gyro.map((x,i)=>x+this.bias[i]+gaussian()*ns*.025);
    const flowValid=depth>=MIN&&depth<=MAX;
    const flow=flowValid?
      [-pose.vbody[0]/depth+pose.gyro[1]+gaussian()*ns*.035,
       -pose.vbody[1]/depth-pose.gyro[0]+gaussian()*ns*.035]:[0,0];
    return {ranges,beams:ranges.map(x=>x!==0),flow,flowValid,flowGyro:gyro,
            imuGyro:gyro,tofAge:0,flowAge:0};
  }
  reset(pose,noisy=true){
    this.frame=0;
    this.bias=noisy?this.bias.map(()=>gaussian()*this.settings.gyro_bias_std_rad_s):[0,0,0];
    const s=this.synth(pose,noisy);
    this.ring=Array.from({length:4},()=>({
      ...s,ranges:s.ranges.slice(),beams:s.beams.slice(),
      flow:s.flow.slice(),flowGyro:s.flowGyro.slice(),imuGyro:s.imuGyro.slice(),
    }));
  }
  advance(pose,noisy=true){
    this.frame++;
    const st=this.settings,raw=this.synth(pose,noisy);
    const prev=this.ring[3];
    const dueTof=this.frame%st.tof_period_frames===0;
    const dueFlow=this.frame%st.flow_period_frames===0;
    const tofFailed=noisy&&Math.random()<st.tof_dropout_prob;
    const flowFailed=noisy&&Math.random()<st.flow_dropout_prob;
    const sample={
      ranges:dueTof?(tofFailed?[0,0,0]:raw.ranges):prev.ranges.slice(),
      beams:dueTof?(tofFailed?[false,false,false]:raw.beams):prev.beams.slice(),
      flow:dueFlow?(flowFailed?[0,0]:raw.flow):prev.flow.slice(),
      flowValid:dueFlow?(raw.flowValid&&!flowFailed):prev.flowValid,
      flowGyro:dueFlow?raw.flowGyro:prev.flowGyro.slice(),
      imuGyro:raw.imuGyro,
      tofAge:dueTof?0:prev.tofAge+1,
      flowAge:dueFlow?0:prev.flowAge+1,
    };
    this.ring.shift();this.ring.push(sample);
  }
  read(gravity){
    const st=this.settings;
    const tof=this.ring[3-st.tof_delay_frames];
    const cam=this.ring[3-st.flow_delay_frames];
    const ground=inferTripod(tof.ranges,gravity);
    const valid=ground.valid&&tof.beams.every(Boolean);
    const velocityValid=valid&&cam.flowValid;
    const depth=valid?ground.depth:0;
    const vxy=velocityValid?[
      -depth*(cam.flow[0]-cam.flowGyro[1]),
      -depth*(cam.flow[1]+cam.flowGyro[0]),
    ]:[0,0];
    const fresh=clamp(1-.25*(tof.tofAge/st.tof_period_frames+
                               cam.flowAge/st.flow_period_frames),.25,1);
    const gradeFresh=clamp(1-.25*tof.tofAge/st.tof_period_frames,.25,1);
    return {gyro:this.ring[3].imuGyro,velocity:vxy,depth,
            velocityConfidence:velocityValid?fresh:0,
            forwardGrade:valid?ground.forward:0,
            sideGrade:valid?ground.side:0,
            groundConfidence:valid?ground.confidence*gradeFresh:0};
  }
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

function opticalRay(R, tilt, u, v) {
  const t=tilt*Math.PI/180, c=Math.cos(t), s=Math.sin(t);
  // The sensor looks forward (+X) until pitched down towards -Z.
  // Its horizontal image axis is body Y; vertical image axis is the
  // tilted local optical "up" vector. At tilt=90° the footprint is
  // square on horizontal flat ground, not a degenerate V/line.
  const length=1/Math.hypot(1,u,v);
  const x=(c+s*v)*length, y=u*length, z=(-s+c*v)*length;
  return [
    R[0]*x+R[1]*y+R[2]*z,
    R[3]*x+R[4]*y+R[5]*z,
    R[6]*x+R[7]*y+R[8]*z,
  ];
}

export function depthProbeRays(torso, R, {forward=0, tilt=70, height=-.018}={}) {
  if (![forward,tilt,height].every(Number.isFinite) || forward<0 ||
      forward>.20 || tilt<0 || tilt>90)
    throw new Error('Depth mount must be forward 0–200 mm, tilt 0–90°');
  const mounted=[
    R[0]*forward+R[1]*0+R[2]*height,
    R[3]*forward+R[4]*0+R[5]*height,
    R[6]*forward+R[7]*0+R[8]*height,
  ];
  const origin=torso.map((v,i)=>v+mounted[i]);
  const directions=[];
  for(let row=0;row<4;row++)for(let col=0;col<4;col++){
    const zone=[];
    const u=(col-1.5)*DEPTH_HALF/2;
    const v=(1.5-row)*DEPTH_HALF/2;
    for(const [offU,offV] of DEPTH_OFFSETS)
      zone.push(opticalRay(R,tilt,u+offU*DEPTH_HALF/8,
                          v+offV*DEPTH_HALF/8));
    directions.push(zone);
  }
  return {origin,directions};
}

// Five-by-five BEAM BOUNDARIES outline the true square 4x4 zone footprint.
// On uneven ground the viewer casts each boundary ray onto actual geoms;
// edges with no real hit are left open, never drawn as imaginary surfaces.
export function depthGridRays(R,{tilt=70}={}) {
  if(!Number.isFinite(tilt)||tilt<0||tilt>90)
    throw new Error('Depth grid tilt must be 0–90°');
  return Array.from({length:25},(_,i)=>{
    const row=Math.floor(i/5),col=i%5;
    const u=(col/2-1)*DEPTH_HALF;
    const v=(1-row/2)*DEPTH_HALF;
    return opticalRay(R,tilt,u,v);
  });
}

export function readDepthProbe(pose, cast, {subrays=4, noise=0, dropout=0}={}) {
  // cast() returns a first physical intersection distance or
  // {distance,kind:'self'|'terrain'}; no hidden surface-height lookup.
  if(![1,4,5].includes(subrays))throw new Error('Expected 1, 4 or 5 depth rays per zone');
  const samples=subrays===1?[0]:subrays===4?[1,2,3,4]:[0,1,2,3,4];
  const ranges=[],valid=[],hitPoints=[],selfOccluded=[],mixed=[];
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
    selfOccluded.push(ok&&values.some(hit=>hit.kind==='self'));
    // Raw slant-range variation happens even on perfectly FLAT ground;
    // a mixed-zone warning instead means rays actually hit different
    // visible objects/surfaces. This flag is diagnostic-only privilege.
    const distinct=new Set(values.map(v=>v.surface??v.kind));
    mixed.push(ok&&distinct.size>1);
    const direction=ok?picked.ray:zone[0];
    const length=ok?picked.distance:.55;
    hitPoints.push(pose.origin.map((p,k)=>p+length*direction[k]));
  }
  return {ranges,valid,origin:pose.origin,hitPoints,selfOccluded,mixed};
}
