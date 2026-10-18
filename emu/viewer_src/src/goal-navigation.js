// Browser-only pose navigation: time-weighted SE(2) lattice and closed-loop
// waypoint tracking. This plans *commands*, never neural weights. It is
// an estimated fastest route under a conservative velocity envelope,
// not proof of optimality or a learned capability/safety guarantee.
import { makeRng } from './live-demo.js';
export const ARENA_RADIUS_M = 4;
export const GOAL_POSITION_TOLERANCE_M = .19;
export const GOAL_HEADING_TOLERANCE_RAD = .28;
export const TREE_CLEARANCE_M = .53;
const GRID = .22, HEADINGS = 16, BOUND = 5.6;
const tau = 2*Math.PI;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const wrapAngle=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const angleError=(desired,current)=>wrapAngle(desired-current);
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
function treeCollision(x,y,trees=[]) {
  return trees.some(t=>Math.hypot(t.x-x,t.y-y)<TREE_CLEARANCE_M);
}
function inArena(p,trees,limit=BOUND){
  return Math.hypot(p.x,p.y)<limit && !treeCollision(p.x,p.y,trees);
}
export function sampleGoal(rng=Math.random,trees=[],from={x:0,y:0},radius=ARENA_RADIUS_M){
  for(let i=0;i<180;i++){
    // Uniform area, not uniform radius.
    const r=Math.sqrt(rng())*radius,a=tau*rng();
    const point={x:r*Math.cos(a),y:r*Math.sin(a),heading:wrapAngle(tau*rng()-Math.PI)};
    if(distance(point,from)>.95 && !treeCollision(point.x,point.y,trees))
      return point;
  }
  // Deterministic fallback if obstacles fill the centre.
  return {x:0,y:0,heading:wrapAngle(tau*rng()-Math.PI)};
}
function commandSpeeds(limits){
  const f=clamp(Number(limits.vx_max)*.72,0,.48);
  const b=clamp(-Number(limits.vx_min)*.72,0,.24);
  const s=clamp(Number(limits.vy_abs)*.75,0,.24);
  const w=clamp(Number(limits.yaw_abs)*.70,0,.95);
  if(!(f>.03&&w>.10))throw new Error('Policy command envelope is too small for navigation');
  return {f,b,s,w,fastest:Math.max(f,b,s)};
}
function primitives(limits){
  const {f,b,s,w}=commandSpeeds(limits),out=[];
  const add=(name,vx,vy,yaw,dt)=>{
    if(!Number.isFinite(dt)||dt<=0||dt>3||(!vx&&!vy&&!yaw))return;
    // Penalise simultaneous high effort slightly; always charge real
    // estimated elapsed time first.
    out.push({name,vx,vy,yaw,dt,cost:dt*(1+0.05*Math.abs(yaw)/w+
      0.035*Math.abs(vy)/(s||1))});
  };
  add('forward',f,0,0,.27/f);
  add('forward-left',f*.83,0,w*.72,clamp(.40/(w*.72),.35,1.1));
  add('forward-right',f*.83,0,-w*.72,clamp(.40/(w*.72),.35,1.1));
  if(b>.06){
    add('reverse',-b,0,0,.25/b);
    add('reverse-left',-b*.82,0,w*.72,clamp(.40/(w*.72),.35,1.1));
    add('reverse-right',-b*.82,0,-w*.72,clamp(.40/(w*.72),.35,1.1));
  }
  if(s>.07){
    add('strafe-left',0,s,0,.25/s);
    add('strafe-right',0,-s,0,.25/s);
    add('diagonal-left',f*.72,s*.70,0,.26/Math.hypot(f*.72,s*.70));
    add('diagonal-right',f*.72,-s*.70,0,.26/Math.hypot(f*.72,s*.70));
    add('side-arc-left',0,s*.85,w*.67,clamp(.39/(w*.67),.4,1.1));
    add('side-arc-right',0,-s*.85,-w*.67,clamp(.39/(w*.67),.4,1.1));
  }
  add('pivot-left',0,0,w,(tau/HEADINGS)/w);
  add('pivot-right',0,0,-w,(tau/HEADINGS)/w);
  return out;
}
export function integratePose(p,cmd,dt){
  const a=cmd.yaw*dt;
  // Analytic planar rigid-body twist. Correct for turning while moving,
  // including sideways movement; midpoint Euler would bias tight loops.
  const sn=Math.sin(a),cs=Math.cos(a);
  const forward=Math.abs(cmd.yaw)<1e-6?cmd.vx*dt:
    (cmd.vx*sn+cmd.vy*(cs-1))/cmd.yaw;
  const lateral=Math.abs(cmd.yaw)<1e-6?cmd.vy*dt:
    (cmd.vx*(1-cs)+cmd.vy*sn)/cmd.yaw;
  const ct=Math.cos(p.heading),st=Math.sin(p.heading);
  return {x:p.x+ct*forward-st*lateral,
          y:p.y+st*forward+ct*lateral,heading:wrapAngle(p.heading+a)};
}
function segmentSafe(p,cmd,dt,trees){
  const count=Math.max(3,Math.ceil(dt/.20));
  for(let i=1;i<=count;i++){
    if(!inArena(integratePose(p,cmd,dt*i/count),trees))return false;
  }
  return true;
}
class Heap {
  constructor(){this.a=[]}
  push(node){
    const a=this.a;let i=a.length;a.push(node);
    while(i>0){let p=(i-1)>>1;if(a[p].f<=node.f)break;a[i]=a[p];i=p;}
    a[i]=node;
  }
  pop(){
    const a=this.a,root=a[0],last=a.pop();
    if(a.length){
      let i=0;
      while(i*2+1<a.length){
        let ch=i*2+1,r=ch+1;
        if(r<a.length&&a[r].f<a[ch].f)ch=r;
        if(last.f<=a[ch].f)break;
        a[i]=a[ch];i=ch;
      }
      a[i]=last;
    }
    return root;
  }
  get length(){return this.a.length;}
}
function key(p){
  // Heading is periodic; ±pi must use the same bin.
  const ix=Math.round(p.x/GRID),iy=Math.round(p.y/GRID);
  const ih=((Math.round((wrapAngle(p.heading)+Math.PI)*HEADINGS/tau)
               %HEADINGS)+HEADINGS)%HEADINGS;
  return ix+','+iy+','+ih;
}
function heuristic(p,goal,speeds){
  const d=distance(p,goal);
  const da=Math.abs(angleError(goal.heading,p.heading));
  // Max rather than sum: walking and turning can happen together.
  return Math.max(d/Math.max(.05,speeds.fastest),da/Math.max(.1,speeds.w));
}
export function planPoseRoute(start,goal,limits,trees=[],options={}){
  if(![start.x,start.y,start.heading,goal.x,goal.y,goal.heading].every(Number.isFinite))
    throw new Error('Nonfinite navigation start/goal');
  const moves=primitives(limits),speed=commandSpeeds(limits);
  if(!inArena(goal,trees))return null;
  const maxNodes=clamp(Number(options.maxNodes)||6500,100,20000);
  const root={...start,g:0,f:heuristic(start,goal,speed)*1.20,
              parent:null,cmd:null,dt:0};
  const heap=new Heap(),best=new Map();
  heap.push(root);best.set(key(root),0);
  let expanded=0,winner=null,closest=root,bestH=heuristic(start,goal,speed);
  while(heap.length&&expanded<maxNodes){
    const p=heap.pop();
    if(p.g>(best.get(key(p))??Infinity)+1e-7)continue;
    expanded++;
    const h=heuristic(p,goal,speed);
    if(h<bestH){bestH=h;closest=p;}
    if(distance(p,goal)<.31&&Math.abs(angleError(goal.heading,p.heading))<.40){
      winner=p;break;
    }
    for(const cmd of moves){
      if(!segmentSafe(p,cmd,cmd.dt,trees))continue;
      const q=integratePose(p,cmd,cmd.dt),id=key(q);
      const g=p.g+cmd.cost;
      if(g>=(best.get(id)??Infinity)-1e-7)continue;
      best.set(id,g);
      heap.push({...q,g,f:g+1.20*heuristic(q,goal,speed),
                 parent:p,cmd:{vx:cmd.vx,vy:cmd.vy,yaw:cmd.yaw,pitch:0},
                 maneuver:cmd.name,dt:cmd.dt});
    }
  }
  const last=winner||closest;
  if(last===root)return null;
  const steps=[];for(let p=last;p.parent;p=p.parent)
    steps.push({x:p.x,y:p.y,heading:p.heading,cmd:p.cmd,
                maneuver:p.maneuver,dt:p.dt});
  steps.reverse();
  // A* searches quantized poses, so the final fine correction is closed
  // loop. Do not present an incomplete search as a solved fastest route.
  const solved=Boolean(winner);
  if(!solved&&!options.allowPartial)return null;
  steps.push({x:goal.x,y:goal.y,heading:goal.heading,
              cmd:{vx:0,vy:0,yaw:0,pitch:0},
              maneuver:'final parking correction',dt:0,final:true});
  return {steps,estimatedSeconds:last.g+
    heuristic(last,goal,speed),solved,expanded,
    modes:[...new Set(steps.map(x=>x.maneuver))]};
}
const safe=(v,a,b)=>clamp(Number.isFinite(v)?v:0,a,b);
export function followWaypoint(pose,waypoint,limits,{final=false}={}){
  const dx=waypoint.x-pose.x,dy=waypoint.y-pose.y;
  const ct=Math.cos(pose.heading),st=Math.sin(pose.heading);
  const fwd=ct*dx+st*dy,side=-st*dx+ct*dy;
  const err=angleError(waypoint.heading,pose.heading);
  const d=Math.hypot(dx,dy);
  const desired=waypoint.cmd||{vx:0,vy:0,yaw:0};
  // Combine position feedback and feedforward; no privileged simulator
  // truth enters the neural actor, and every command stays in-envelope.
  let vx=1.4*fwd+(final?0:.5*desired.vx);
  let vy=1.4*side+(final?0:.5*desired.vy);
  let yaw=2.2*err+(final?0:.15*desired.yaw);
  const maxF=Math.max(0,limits.vx_max),maxB=Math.max(0,-limits.vx_min);
  const maxS=Math.max(0,limits.vy_abs),maxYaw=Math.max(0,limits.yaw_abs);
  // Full-speed sideways+forward is not presumed safe: project onto
  // conservative combined-translation ellipse.
  const a=maxF*.78,b=maxB*.78,s=maxS*.80;
  const r=Math.hypot(vx/(vx>=0?Math.max(a,.01):Math.max(b,.01)),
                     vy/Math.max(s,.01));
  if(r>1){vx/=r;vy/=r;}
  const rate=safe(yaw,-maxYaw*.73,maxYaw*.73);
  // Slow before arriving, but don't multiply down alignment corrections.
  const brake=final?clamp(d/.55,.16,1):1;
  return {vx:safe(vx*brake,-maxB,maxF),vy:safe(vy*brake,-maxS,maxS),
          yaw:rate,pitch:0};
}
export class GoalNavigator {
  constructor(seed=5017){
    this.rng=makeRng(seed);this.active=false;
    this.goal=null;this.route=null;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.arrivals=0;
    this.replans=0;this.reason='Not started';this.replanDelay=0;
  }
  reset(){this.goal=null;this.route=null;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.reason='Ready for a target';
    this.replanDelay=0;this.replans=0;}
  setGoal(pose,goal,limits,trees=[]){
    const route=planPoseRoute(pose,goal,limits,trees);
    if(!route)return false;
    this.goal={...goal};this.route=route;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.replanDelay=0;
    this.reason=route.solved?'Navigating':'Partial route · replanning';
    return true;
  }
  nextGoal(pose,limits,trees=[]){
    for(let i=0;i<24;i++){
      const goal=sampleGoal(this.rng,trees,pose);
      const planned=this.setGoal(pose,goal,limits,trees);
      if(planned&&this.route.solved)return true;
    }
    this.reason='No fully planned target; adjust terrain/trees';
    this.goal=null;this.route=null;
    return false;
  }
  tick(dt,pose,limits,trees=[],apply=()=>{}){
    if(!this.active)return;
    if(!this.goal&&!this.nextGoal(pose,limits,trees)){
      apply({vx:0,vy:0,yaw:0,pitch:0});return;
    }
    if(!Number.isFinite(pose.x)||!Number.isFinite(pose.y)||
       !Number.isFinite(pose.heading))return;
    const delta=clamp(dt,0,.1);
    this.elapsed+=delta;
    const dist=distance(pose,this.goal),aerr=Math.abs(angleError(this.goal.heading,pose.heading));
    const speed=Math.hypot(pose.vx||0,pose.vy||0);
    if(dist<GOAL_POSITION_TOLERANCE_M&&aerr<GOAL_HEADING_TOLERANCE_RAD&&speed<.13){
      apply({vx:0,vy:0,yaw:0,pitch:0});
      this.arrivalHold+=delta;
      this.reason='Parking · holding pose';
      if(this.arrivalHold>.75){
        this.arrivals++;this.nextGoal(pose,limits,trees);
      }
      return;
    }
    this.arrivalHold=0;
    const steps=this.route.steps;
    while(this.index<steps.length-1){
      const p=steps[this.index];
      if(distance(pose,p)>.16||Math.abs(angleError(p.heading,pose.heading))>.34)break;
      this.index++;
    }
    const target=steps[this.index];
    const command=followWaypoint(pose,target,limits,{final:target.final});
    apply(command);
    this.reason=target.final?'Parking into final heading':target.maneuver;
    this.replanDelay+=delta;
    // Re-plan if significantly displaced from the planned next target.
    // Do not thrash the target from scratch every render frame.
    if(this.replanDelay>3.2 && (distance(pose,target)>.75||!this.route.solved)){
      this.replanDelay=0;
      const route=planPoseRoute(pose,this.goal,limits,trees,{maxNodes:7500});
      if(route){this.route=route;this.index=0;this.replans++;}
    }
  }
  status(pose){
    return {goal:this.goal,route:this.route,
      distance:this.goal?distance(pose,this.goal):0,
      headingError:this.goal?Math.abs(angleError(this.goal.heading,pose.heading)):0,
      elapsed:this.elapsed,arrivals:this.arrivals,replans:this.replans,
      index:this.index,reason:this.reason};
  }
}
