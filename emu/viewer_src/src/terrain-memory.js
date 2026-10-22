// Browser-local v5019 rolling V terrain memory. Schema-1/2 actors keep this
// as an optional diagnostic; schema-3 312-input actors MUST consume its
// exact 20 ordered readout channels on every frame.
// Geometry/calibrations approximate emu/rolling_tof.py and emu/forward_tof.py.
// No true MuJoCo ground height, box height or qvel enters the MEMORY update:
// pose.vbody is used ONLY to synthesize a noisy gyro-compensated optical flow.
import { inferTripod } from './sensor.js';

export const V_CELLS=51, V_START=-0.10, V_SPACING=.01, V_APEX=-.04;
export const V_SAMPLES=[5,10,15,20,30]; // -5,0,5,10,20cm; EXACT rolling_tof.py contract
const MOUNTS=[[.035,-.05,-.025],[.035,.05,-.025],[-.04,0,-.025]];
const FRONT=[[.035,-.05,-.025],[.035,.05,-.025]];
const RAYS=FRONT.map(()=>{const dx=(.20-.035)/.2334, l=Math.hypot(dx,1);return [dx/l,0,-1/l];});
const DOWN=[0,0,-1],MAX=.6,MIN=.035;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const times=(a,s)=>a.map(v=>v*s);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>times(a,1/Math.max(1e-8,Math.hypot(...a)));
const b2w=(r,v)=>[r[0]*v[0]+r[1]*v[1]+r[2]*v[2],
                   r[3]*v[0]+r[4]*v[1]+r[5]*v[2],
                   r[6]*v[0]+r[7]*v[1]+r[8]*v[2]];
const w2b=(r,v)=>[r[0]*v[0]+r[3]*v[1]+r[6]*v[2],
                   r[1]*v[0]+r[4]*v[1]+r[7]*v[2],
                   r[2]*v[0]+r[5]*v[1]+r[8]*v[2]];
function noise(){return Math.sqrt(-2*Math.log(Math.max(1e-8,Math.random())))*Math.cos(2*Math.PI*Math.random());}
function scan(origin,direction,raycast){
  const hit=raycast(origin,direction,MAX);
  const length=Number(hit?.range);
  if(!Number.isFinite(length)||length<MIN||length>MAX)return {valid:false,range:0,point:add(origin,times(direction,MAX)),origin};
  return {valid:true,range:length,point:add(origin,times(direction,length)),origin};
}
// First surface crossing, rather than a single endpoint: catches thin steps
// and ramps that could be missed by bisection against an infinite ground.
export function groundRayHit(origin,dir,heightAt,maxRange=.6) {
  const fn=t=>origin[2]+t*dir[2]-heightAt(origin[0]+t*dir[0],origin[1]+t*dir[1]);
  let prev=MIN,ph=fn(prev);
  if(!Number.isFinite(ph)||ph<=0)return null;
  for(let i=1;i<=100;i++){
    const cur=MIN+(maxRange-MIN)*i/100, ch=fn(cur);
    if(Number.isFinite(ch)&&ch<=0){
      let a=prev,b=cur;
      for(let j=0;j<16;j++){const m=(a+b)/2;if(fn(m)>0)a=m;else b=m;}
      return {range:(a+b)/2};
    }
    prev=cur;ph=ch;
  }
  return null;
}
function trackY(side,x,ax,ay) {
  return ay+((side===0?-.05:.05)-ay)*Math.max(0,x-ax)/Math.max(.08,.20-ax);
}
function newTrack(){return new Float32Array(V_CELLS);}
function interpolate(track,conf,x) {
  const f=(x-V_START)/V_SPACING;
  if(f<0||f>V_CELLS-1)return [0,0];
  const lo=Math.floor(f),hi=Math.min(lo+1,V_CELLS-1),w=f-lo;
  const a=(1-w)*conf[lo],b=w*conf[hi],c=a+b;
  return c>1e-5?[(track[lo]*a+track[hi]*b)/c,c]:[0,0];
}
export class RollingTerrainDiagnostic {
  constructor(){this.reset();}
  reset(){
    this.frame=0;this.heights=[newTrack(),newTrack()];
    this.confidence=[newTrack(),newTrack()];
    this.apexX=V_APEX;this.apexY=0;this.beams=[];
    this.lastScan=null;this.pending=null;this.groundAge=999;
    this.motion={vx:0,vy:0,yaw:0,confidence:0};
    this.activeSamples=0;this.freeze=false;
    this.lastGroundValid=false;
  }
  // Both tracks can move in opposite longitudinal directions under yaw.
  // Same inverse SE(2) warp + confidence-weighted cross-track projection as
  // emu/rolling_tof.py, not a global map or made-up interpolation.
  transport(dt,vx,vy,yaw,quality,newX=this.apexX,newY=this.apexY) {
    const nextH=[newTrack(),newTrack()],nextC=[newTrack(),newTrack()];
    const theta=yaw*dt,cs=Math.cos(theta),sn=Math.sin(theta);
    const trusted=quality>.10,dx=(trusted?vx:0)*dt,dy=(trusted?vy:0)*dt;
    const decay=Math.exp(-dt/.60)*clamp(1-Math.abs(yaw)*dt,0,1)*(.96+.04*clamp(quality,0,1));
    for(let lane=0;lane<2;lane++)for(let i=0;i<V_CELLS;i++){
      const x=V_START+i*V_SPACING,y=trackY(lane,x,newX,newY);
      const ox=cs*x-sn*y+dx,oy=sn*x+cs*y+dy;
      let sum=0, weighted=0,best=0;
      for(let oldLane=0;oldLane<2;oldLane++){
        const [h,c]=interpolate(this.heights[oldLane],this.confidence[oldLane],ox);
        const error=(oy-trackY(oldLane,ox,this.apexX,this.apexY))/.023;
        const wc=c*Math.exp(-.5*error*error);
        sum+=wc;weighted+=wc*h;best=Math.max(best,wc);
      }
      if(sum>1e-4&&Number.isFinite(weighted)){
        nextH[lane][i]=weighted/sum;
        nextC[lane][i]=clamp(best*decay,0,1);
      }
    }
    this.heights=nextH;this.confidence=nextC;this.apexX=newX;this.apexY=newY;
  }
  insert(lane,x,y,h,valid){
    if(!valid||!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(h)||x<V_START-.0005||x>.4005)return;
    for(let i=0;i<V_CELLS;i++){
      const cellX=V_START+i*V_SPACING;
      const dx=Math.abs(cellX-x);
      if(dx>=.020)continue;
      const dy=(y-trackY(lane,x,this.apexX,this.apexY))/.023;
      const weight=(1-dx/.020)*Math.exp(-.5*dy*dy);
      if(weight<1e-6)continue;
      const c=this.confidence[lane][i],v=this.heights[lane][i];
      this.heights[lane][i]=(v*c+h*weight)/Math.max(1e-8,c+weight);
      this.confidence[lane][i]=Math.max(c,weight);
    }
  }
  // Synthesise five *actual* sensor rays. Their lengths, not terrain truth,
  // are the only height observations passed to the rolling memory.
  acquire(pose,noisy,raycast) {
    const ns=noisy?1.5:0,all=[...FRONT,MOUNTS[0],MOUNTS[1],MOUNTS[2]];
    const dirs=[...RAYS,DOWN,DOWN,DOWN];
    const readings=all.map((m,i)=>{
      const origin=add(pose.torso,b2w(pose.R,m));
      const dir=b2w(pose.R,dirs[i]);
      let hit=scan(origin,dir,raycast);
      if(hit.valid){
        const r=hit.range+noise()*ns*.0015*(1+.5*hit.range/.6);
        hit=Number.isFinite(r)&&r>=MIN&&r<=MAX?
          {...hit,range:Math.round(r*1000)/1000,point:add(origin,times(dir,Math.round(r*1000)/1000))}:
          {...hit,valid:false,range:0};
      }
      if(noisy&&Math.random()<.01)hit={...hit,valid:false,range:0};
      return hit;
    });
    return {readings,gravity:w2b(pose.R,[0,0,-1])};
  }
  project(snapshot,currentGravity=null) {
    const a=snapshot.readings;
    const tripod=a.slice(2),valid=tripod.every(p=>p.valid);
    const ranges=tripod.map(p=>p.range);
    const gravity=currentGravity||snapshot.gravity;
    const ground=inferTripod(ranges,gravity);
    if(!valid||!ground.valid)return null;
    const up=unit(times(gravity,-1));
    const f=unit(sub([1,0,0],times(up,dot(up,[1,0,0]))));
    const side=cross(up,f);
    const rear=add(MOUNTS[2],times(DOWN,ranges[2]));
    const points=a.slice(0,2).map((hit,lane)=>{
      if(!hit.valid)return null;
      const p=add(FRONT[lane],times(RAYS[lane],hit.range));
      return {x:dot(p,f),y:dot(p,side),
              height:clamp(dot(sub(p,rear),up),-.15,.15)};
    });
    return {rear:{x:dot(rear,f),y:dot(rear,side)},points,ground};
  }
  // pose.vbody is used only here to emulate optical flow measurements, not
  // to move any map cells directly. Velocity consumed by transport comes
  // from an inverted noisy measured flow with gyro compensation.
  motionEstimate(pose,ground,noisy){
    const ns=noisy?1.5:0,depth=ground?.valid?ground.depth:0;
    const valid=depth>=MIN&&depth<=MAX&&(!noisy||Math.random()>=.02);
    const gyro=pose.gyro.map(z=>z+noise()*ns*.025);
    const flow=valid?[-pose.vbody[0]/depth+pose.gyro[1]+noise()*ns*.035,
                     -pose.vbody[1]/depth-pose.gyro[0]+noise()*ns*.035]:[0,0];
    return {vx:valid?-depth*(flow[0]-gyro[1]):0,
            vy:valid?-depth*(flow[1]+gyro[0]):0,
            yaw:gyro[2],confidence:valid?1:0};
  }
  update(pose,noisy,raycast,{freeze=false}={}){
    this.frame++;
    if(this.frame%2===0){
      this.pending=this.acquire(pose,noisy,raycast);
      this.beams=this.pending.readings; // latest raw real measured beam endpoints
    }
    // One 20ms control-frame latency: acquire on even frame, consume next.
    const ready=this.frame%2===1&&this.pending?this.pending:null;
    const currentGravity=w2b(pose.R,[0,0,-1]);
    const projected=ready?this.project(ready,currentGravity):null;
    if(ready){
      if(projected){this.lastScan=projected;this.groundAge=0;}
      else {this.lastScan=null;this.groundAge=999;}
    }else this.groundAge++;
    const motion=this.motionEstimate(
      pose,this.groundAge<=2?this.lastScan?.ground:null,noisy);
    this.motion=motion;
    this.lastGroundValid=Boolean(projected);
    if(freeze)return;
    // The delivered range was captured one control frame earlier.
    // Motion-compensate its MEASURED landing point to the current body frame;
    // otherwise the V would lag a centimetre at brisk speeds.
    const compensate=point=>{
      if(!point)return null;
      const t=motion.yaw*.02,c=Math.cos(t),si=Math.sin(t);
      const px=point.x-motion.vx*.02,py=point.y-motion.vy*.02;
      return {...point,x:c*px+si*py,y:-si*px+c*py};
    };
    const rear=projected?compensate(projected.rear):null;
    const points=projected?projected.points.map(compensate):null;
    this.transport(.02,motion.vx,motion.vy,motion.yaw,motion.confidence,
                   rear?.x??this.apexX,rear?.y??this.apexY);
    if(projected){
      // Rear ray defines zero reference shared across BOTH V branches.
      // When both lanes remember the rear patch, re-zero only on evidence.
      const left=interpolate(this.heights[0],this.confidence[0],rear.x);
      const right=interpolate(this.heights[1],this.confidence[1],rear.x);
      if(left[1]>.2&&right[1]>.2){
        const correction=clamp((left[0]+right[0])/2,-.08,.08)*clamp((left[1]+right[1])/2,0,1);
        for(let lane=0;lane<2;lane++)for(let i=0;i<V_CELLS;i++)
          if(this.confidence[lane][i]>1e-5)this.heights[lane][i]-=correction;
      }
      for(let lane=0;lane<2;lane++){
        const pt=points[lane];
        if(pt)this.insert(lane,pt.x,pt.y,pt.height,true);
        this.insert(lane,rear.x,rear.y,0,true);
      }
    }
    this.activeSamples=this.confidence.flatMap(x=>[...x]).filter(c=>c>.1).length;
  }
  values(){
    return {heights:this.heights.map(x=>Array.from(x)),
            confidence:this.confidence.map(x=>Array.from(x)),
            actorHeights:this.heights.flatMap(x=>V_SAMPLES.map(i=>x[i])),
            actorConfidence:this.confidence.flatMap(x=>V_SAMPLES.map(i=>x[i])),
            apex:[this.apexX,this.apexY],motion:{...this.motion},
            beams:this.beams.map((v,i)=>({...v,lane:i<2?i:2})),
            activeSamples:this.activeSamples,frame:this.frame,freeze:this.freeze};
  }
}
