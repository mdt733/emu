// Experimental bidirectional kinodynamic A* over joystick manoeuvres.
// Search expands actual body-frame velocity/yaw commands in CONTINUOUS SE(2),
// not neighbours of a pixel grid. A coarse pose hash only suppresses near
// duplicate states. The robot's trained actor, MuJoCo and command delivery
// path are completely unchanged by this search prototype.
import {integratePose,wrapAngle} from './goal-navigation.js';
import {planCurvatureRoute} from './curvature-planner.js';

const TAU=2*Math.PI;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const deltaAngle=(a,b)=>Math.abs(wrapAngle(a-b));
// Three translation modes × nine yaw rates. This produces different
// physical turning RADII (vx / omega), rather than merely different grid
// directions. Heading and XY are continuous; bins only prune duplicates.
export const ARC_YAW_FRACTIONS=Object.freeze([-1,-.75,-.5,-.25,0,.25,.5,.75,1]);
export const ARC_TRANSLATION_MODES=Object.freeze(['forward','reverse','pivot']);
export const ALL_JOYSTICK_OPTIONS=ARC_YAW_FRACTIONS.length*ARC_TRANSLATION_MODES.length;

/** Sample a finite-duration, executable body twist from the NN's trained
 * command envelope. The biped CAN rotate in place: unlike the initial
 * prototype, pivot primitives are first-class. Sideways control belongs
 * to the near-goal follower, not a long-distance search edge.
 *
 * 9 forward arcs/straights + 9 reverse arcs/straights + 8 pivot speeds
 * (the ninth, neutral+straight, is a no-op and is not expanded).
 * This makes both vehicle-parking and in-place final heading feasible.
 */
export function arcPrimitives(limits,{duration=.55}={}){
  const maxF=Math.min(Math.max(0,Number(limits.vx_max)||0)*.90,.95);
  const maxB=Math.min(Math.max(0,-Number(limits.vx_min)||0)*.84,.48);
  const maxYaw=Math.min(Math.max(0,Number(limits.yaw_abs)||0)*.78,1.25);
  if(!(maxF>.08&&maxYaw>.1))throw new Error('Policy has insufficient forward/yaw authority');
  const results=[];
  for(const mode of ARC_TRANSLATION_MODES){
    const vx=mode==='forward'?maxF:mode==='reverse'?-maxB:0;
    for(const [index,fraction] of ARC_YAW_FRACTIONS.entries()){
      const yaw=maxYaw*fraction,executable=mode!=='pivot'||index!==4;
      const pivot=mode==='pivot';
      const radius=Math.abs(yaw)>1e-10?Math.abs(vx/yaw):Infinity;
      const curvature=Math.abs(vx)>1e-10?yaw/vx:null;
      const effort=Math.abs(fraction);
      results.push({
        id:results.length+1,mode,joystick:[Math.sign(vx),0,fraction],
        yawFraction:fraction,vx,vy:0,yaw,dt:duration,radius,curvature,
        executable,
        // Time first; small costs for high turn rate, reverse or pivot.
        // Do not penalise a tight parking turn more than many metres of
        // extra travel merely because it uses high angular velocity.
        cost:duration*(1+.045*effort+(mode==='reverse'?.10:0)+
          (pivot?.14:0))
      });
    }
  }
  return results;
}
// Retain the original API name for independent comparisons / callers.
export const joystickPrimitives=arcPrimitives;

export function reversePrimitive(primitive){
  // Invert the same finite-duration SE(2) twist exactly; never pretend
  // the backward search is a forward-only physical robot driving backward.
  // Backward edges are predecessor discovery; final trajectory replays
  // the ORIGINAL forward controls in their chronological order.
  return {vx:-primitive.vx,vy:-primitive.vy,yaw:-primitive.yaw,
          dt:primitive.dt};
}
const key=(pose)=>{
  // Quantization is ONLY a visited-state hash, not a stepping grid.
  const h=((Math.round((wrapAngle(pose.heading)+Math.PI)*24/TAU)%24)+24)%24;
  return Math.round(pose.x/.26)+','+Math.round(pose.y/.26)+','+h;
};
const valid=(p,trees)=>Math.hypot(p.x,p.y)<5.6&&
  trees.every(t=>Math.hypot(p.x-t.x,p.y-t.y)>.53);
function swept(start,primitive,backwards,trees){
  const cmd=backwards?reversePrimitive(primitive):primitive;
  for(let i=1;i<=7;i++){
    if(!valid(integratePose(start,cmd,primitive.dt*i/7),trees))return null;
  }
  return integratePose(start,cmd,primitive.dt);
}
class Heap {
  constructor(){this.a=[];}
  get length(){return this.a.length;}
  push(v){const a=this.a;let i=a.length;a.push(v);
    while(i>0){const p=(i-1)>>1;if(a[p].f<=v.f)break;a[i]=a[p];i=p;}
    a[i]=v;}
  pop(){const a=this.a,out=a[0],last=a.pop();if(!a.length)return out;
    let i=0;while(i*2+1<a.length){let c=i*2+1;
      if(c+1<a.length&&a[c+1].f<a[c].f)c++;
      if(last.f<=a[c].f)break;a[i]=a[c];i=c;}
    a[i]=last;return out;}
}
function joinRoute(a,b,connector){
  const front=[];let n=a;
  while(n.parent){front.push({...n.pose,cmd:n.edge,
    maneuver:'joystick primitive',dt:n.edge.dt});n=n.parent;}
  front.reverse();
  const back=[];n=b;
  while(n.parent){back.push({...n.parent.pose,cmd:n.edge,
    maneuver:'joystick primitive',dt:n.edge.dt});n=n.parent;}
  const steps=[...front,...(connector?.steps||[]),...back];
  // Only the ACTUAL destination is a terminal pose; intermediate
  // analytic-bridge endpoints are not little parking instructions.
  for(const step of steps)step.final=false;
  if(steps.length)steps.at(-1).final=true;
  return steps;
}

/** Experimental solver: both trees meet on (x,y,HEADING), never xy alone.
 * A short curvature-constrained connection heals the final search-bin
 * mismatch, so we don't silently teleport the robot between trees.
 */
export function bidirectionalMotionRoute(start,goal,limits,trees=[],opts={}){
  if(![start.x,start.y,start.heading,goal.x,goal.y,goal.heading].every(Number.isFinite))
    throw new Error('Nonfinite start/goal pose');
  if(!valid(start,trees)||!valid(goal,trees))return null;
  const maneuvers=arcPrimitives(limits).filter(p=>p.executable);
  const speed=Math.min(limits.vx_max*.90,.95);
  const yaw=Math.min(limits.yaw_abs*.78,1.25);
  const heuristic=(pose,other)=>Math.max(
    dist(pose,other)/Math.max(.1,speed),
    deltaAngle(pose.heading,other.heading)/Math.max(.1,yaw));
  const roots=[
    {pose:start,g:0,f:heuristic(start,goal),parent:null,edge:null,turn:0},
    {pose:goal,g:0,f:heuristic(goal,start),parent:null,edge:null,turn:0}
  ];
  const heaps=[new Heap(),new Heap()],costs=[new Map(),new Map()];
  const explored=[new Map(),new Map()];
  for(let i=0;i<2;i++){
    heaps[i].push(roots[i]);costs[i].set(key(roots[i].pose),0);
  }
  let best=null,expansions=0;
  const maxNodes=Math.min(Math.max(100,opts.maxNodes||6500),25000);
  while(heaps[0].length&&heaps[1].length&&expansions<maxNodes){
    const side=heaps[0].a[0].f<=heaps[1].a[0].f?0:1;
    const current=heaps[side].pop(),hash=key(current.pose);
    if(current.g>(costs[side].get(hash)??Infinity)+1e-8)continue;
    explored[side].set(hash,current);
    expansions++;
    const other=explored[1-side].get(hash);
    if(other && dist(current.pose,other.pose)<.34 &&
        deltaAngle(current.pose.heading,other.pose.heading)<.32){
      const f=side===0?current:other,b=side===0?other:current;
      const connector=planCurvatureRoute(f.pose,b.pose,limits,trees,{noDetour:true,
        radii:[.40,.60,.85]});
      // Only accept short physically continuous bridge, not a long loop
      // to mask a coarse node bin. The policy feedback handles residual
      // tracking errors along each true SE(2) segment.
      if(connector&&connector.meters<1.7){
        const score=f.g+b.g+connector.estimatedSeconds;
        if(!best||score<best.score)best={f,b,connector,score};
      }
    }
    if(best && current.f>best.score*1.12)break;
    for(const action of maneuvers){
      const pose=swept(current.pose,action,side===1,trees);
      if(!pose)continue;
      const h=key(pose);
      let cost=action.cost;
      if(current.edge){
        const prev=current.edge;
        if(Math.sign(prev.vx)!==Math.sign(action.vx))cost+=.13;
        if(Math.sign(prev.yaw)!==Math.sign(action.yaw)&&prev.yaw&&action.yaw)cost+=.05;
        // Penalise mode switches more than gradual yaw changes, but not
        // so heavily that short parking pivots become uneconomical.
        if(Math.abs(prev.vx-action.vx)>.42)cost+=.10;
        cost+=.06*Math.abs(prev.yaw-action.yaw)/Math.max(.1,yaw);
      }
      const g=current.g+cost;
      if(g>=(costs[side].get(h)??Infinity)-1e-8)continue;
      const child={pose,g,f:g+1.10*heuristic(pose,roots[1-side].pose),
        parent:current,edge:action,turn:action.yaw};
      costs[side].set(h,g);
      heaps[side].push(child);
    }
  }
  if(!best)return null;
  const steps=joinRoute(best.f,best.b,best.connector);
  return {start:{...start},steps,estimatedSeconds:best.score,solved:true,
          planner:'bidirectional-motion-lattice',
          expanded:expansions,
          meetPose:{...best.f.pose},
          connectionMetres:best.connector.meters,
          modes:['forward arcs','reverse arcs','in-place yaw','curvature bridge'],
          families:'9 yaw samples × {forward, reverse, pivot}, with no sideways cruise'};
}
