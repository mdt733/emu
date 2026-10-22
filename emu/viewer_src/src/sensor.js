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
    if(!['sim2real_232','rolling_tof_312'].includes(contract?.schema))
      throw new Error('Supported 232/312 sensor contract missing');
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
