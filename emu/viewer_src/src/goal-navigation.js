// Browser-only pose navigation: time-weighted SE(2) lattice and closed-loop
// continuous line tracking. This plans *commands*, never neural weights. It is
// an estimated fastest route under a conservative velocity envelope,
// not proof of optimality or a learned capability/safety guarantee.
import { makeRng } from './rng.js';
import { buildRoutePath, trackRoutePath, parkAtGoal } from './path-follower.js';
import { PredictivePathController } from './predictive-navigation.js';
import { terrainRiskAt, sweptTerrainCost } from './navigation-terrain.js';
export const ARENA_RADIUS_M = 4;
export const GOAL_POSITION_TOLERANCE_M = .19;
export const GOAL_HEADING_TOLERANCE_RAD = .28;
// Extra tracking room beyond the tree's physical trunk footprint.
export const TREE_CLEARANCE_M = .62;
// Repeatable real-MuJoCo 5-pose course inside the park's gentler centre.
// This uses the CURRENT actor; no synthetic kinematics are scored as results.
export const NAV_BENCHMARK_TARGETS = Object.freeze([
  {x:1.25,y:.10,heading:0},
  {x:1.30,y:1.20,heading:Math.PI/2},
  {x:-.75,y:1.30,heading:Math.PI},
  {x:-1.20,y:-1.10,heading:-Math.PI/2},
  {x:.95,y:-1.20,heading:0},
]);
const GRID = .22, HEADINGS = 16, BOUND = 5.6;
const tau = 2*Math.PI;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const wrapAngle=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const angleError=(desired,current)=>wrapAngle(desired-current);
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const samePose=(a,b)=>Boolean(a&&b)&&distance(a,b)<1e-5&&
  Math.abs(angleError(a.heading,b.heading))<1e-5;
function treeCollision(x,y,trees=[]) {
  return trees.some(t=>Math.hypot(t.x-x,t.y-y)<TREE_CLEARANCE_M);
}
function inArena(p,trees,limit=BOUND,terrain=null){
  return Math.hypot(p.x,p.y)<limit && !treeCollision(p.x,p.y,trees) &&
    Number.isFinite(terrainRiskAt(terrain,p.x,p.y));
}
export function sampleGoal(rng=Math.random,trees=[],from={x:0,y:0},radius=ARENA_RADIUS_M,terrain=null){
  for(let i=0;i<180;i++){
    // Uniform area, not uniform radius.
    const r=Math.sqrt(rng())*radius,a=tau*rng();
    const point={x:r*Math.cos(a),y:r*Math.sin(a),heading:wrapAngle(tau*rng()-Math.PI)};
    if(distance(point,from)>.95 && inArena(point,trees,BOUND,terrain))
      return point;
  }
  // Deterministic fallback if obstacles fill the centre.
  return {x:0,y:0,heading:wrapAngle(tau*rng()-Math.PI)};
}
function commandSpeeds(limits){
  // Use the REAL checkpoint's published speed frontier, not the
  // arbitrary 0.48 m/s cap that made high-speed forward routes expensive.
  const f=clamp(Number(limits.vx_max)*.84,0,.95);
  const b=clamp(-Number(limits.vx_min)*.75,0,.45);
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
  add('forward',f,0,0,.40/f);
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
    add('diagonal-left',f*.80,s*.50,0,.40/Math.hypot(f*.80,s*.50));
    add('diagonal-right',f*.80,-s*.50,0,.40/Math.hypot(f*.80,s*.50));
    add('side-arc-left',0,s*.85,w*.67,clamp(.39/(w*.67),.4,1.1));
    add('side-arc-right',0,-s*.85,-w*.67,clamp(.39/(w*.67),.4,1.1));
  }
  // The current Pinnacle actor can pivot. Honour that capability.
  add('pivot-left',0,0,w*.85,clamp(.40/(w*.85),.3,1.0));
  add('pivot-right',0,0,-w*.85,clamp(.40/(w*.85),.3,1.0));
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
function segmentSafe(p,cmd,dt,trees,terrain=null){
  const count=Math.max(3,Math.ceil(dt/.20));
  for(let i=1;i<=count;i++){
    if(!inArena(integratePose(p,cmd,dt*i/count),trees,BOUND,terrain))return false;
  }
  return true;
}

/**
 * Refine exact-twist A* primitives into a usable continuous reference.
 * Do not smooth THROUGH reverse/pivot/lateral transitions or endpoints.
 * Every modified chord is checked against tree clearance and the arena.
 * This is intentionally a small bounded post-pass, NOT another planner.
 */
const motionKind=cmd=>Math.abs(cmd?.vx||0)>.045?(cmd.vx>0?'forward':'reverse'):
  Math.abs(cmd?.vy||0)>.035?'lateral':Math.abs(cmd?.yaw||0)>.035?'pivot':'terminal';
const chordSafe=(a,b,trees,extra=0,terrain=null)=>{
  const dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy;
  const n=Math.max(1,Math.ceil(Math.sqrt(d2)/.07));
  for(let j=0;j<=n;j++){
    const t=j/n,x=a.x+t*dx,y=a.y+t*dy;
    if(Math.hypot(x,y)>=BOUND-.03||
       !Number.isFinite(terrainRiskAt(terrain,x,y))||
       trees.some(tree=>Math.hypot(tree.x-x,tree.y-y)<TREE_CLEARANCE_M+extra))
      return false;
  }
  return true;
};
const safePolyline=(start,steps,trees,terrain=null)=>{
  let prev=start;
  for(const p of steps){
    if(!chordSafe(prev,p,trees,0,terrain))return false;
    prev=p;
  }
  return true;
};
export function refineRouteSteps(start,rawSteps,limits,trees=[],terrain=null,through=false){
  const terminal=rawSteps.at(-1)?.final?rawSteps.at(-1):null;
  const source=terminal?rawSteps.slice(0,-1):rawSteps;
  if(!source.length)return [...rawSteps];
  const dense=[];
  let from=start;
  for(const step of source){
    const angular=Math.abs(angleError(step.heading,from.heading));
    const length=distance(from,step);
    const slices=clamp(Math.max(1,Math.ceil(length/.12),
      Math.ceil(angular/.17)),1,18);
    for(let j=1;j<=slices;j++){
      const p=j===slices?{x:step.x,y:step.y,heading:step.heading}:
        integratePose(from,step.cmd,step.dt*j/slices);
      dense.push({...p,cmd:{...step.cmd},
        maneuver:step.maneuver,dt:step.dt/slices,refined:true});
    }
    from=step;
  }

  // Small elastic-band adjustment. Keep the original route as an anchor so
  // curvature can't be smoothed into a physically unachievable shortcut.
  const original=dense.map(p=>({...p}));
  const pointBefore=i=>i===0?start:dense[i-1];
  for(let pass=0;pass<12;pass++){
    for(let i=0;i<dense.length-1;i++){
      const kind=motionKind(dense[i].cmd);
      if(kind==='pivot'||kind==='terminal')continue;
      if(i===0||motionKind(dense[i-1].cmd)!==kind||
         motionKind(dense[i+1].cmd)!==kind)continue;
      const prev=pointBefore(i),next=dense[i+1],p=dense[i];
      const x=p.x+.26*(prev.x+next.x-2*p.x)+.08*(original[i].x-p.x);
      const y=p.y+.26*(prev.y+next.y-2*p.y)+.08*(original[i].y-p.y);
      // Prevent large changes to the original swept collision-free route.
      const dx=x-original[i].x,dy=y-original[i].y;
      const size=Math.hypot(dx,dy),ratio=size>.10?.10/size:1;
      const q={x:original[i].x+dx*ratio,y:original[i].y+dy*ratio};
      if(chordSafe(prev,q,trees,.045,terrain)&&chordSafe(q,next,trees,.045,terrain)){
        p.x=q.x;p.y=q.y;
      }
    }
  }
  // Source headings remain close to what the actor can actually execute;
  // adjust only short-interior tangents with bounded heading corrections.
  for(let i=1;i<dense.length-1;i++){
    const kind=motionKind(dense[i].cmd);
    if(kind!=='forward'&&kind!=='reverse')continue;
    if(motionKind(dense[i-1].cmd)!==kind||
       motionKind(dense[i+1].cmd)!==kind)continue;
    const prev=dense[i-1],next=dense[i+1];
    const bearing=Math.atan2(next.y-prev.y,next.x-prev.x);
    const wanted=wrapAngle(bearing+(kind==='reverse'?Math.PI:0));
    const correction=clamp(angleError(wanted,original[i].heading),-.28,.28);
    dense[i].heading=wrapAngle(original[i].heading+.55*correction);
  }

  // Cheap acceleration/braking profile (m/s^2): forward/backward passes
  // within each movement run. Stops at cusps and pivots, no calibration
  // step required when the user loads a different trained brain.
  const accelerate=.85,brake=1.0,yawMax=Math.max(.1,Number(limits.yaw_abs)||1);
  let begin=0;
  while(begin<dense.length){
    const kind=motionKind(dense[begin].cmd);
    let end=begin;
    while(end+1<dense.length&&motionKind(dense[end+1].cmd)===kind)end++;
    if(kind==='forward'||kind==='reverse'||kind==='lateral'){
      const speeds=[],spans=[];
      for(let i=begin;i<=end;i++){
        const previous=i===0?start:dense[i-1];
        const span=distance(previous,dense[i]);
        const base=Math.abs(kind==='lateral'?dense[i].cmd.vy:dense[i].cmd.vx);
        const radians=Math.abs(angleError(dense[i].heading,previous.heading));
        const curvature=radians/Math.max(.035,span);
        const cornerCap=curvature>.001?yawMax*.72/curvature:Infinity;
        spans.push(span);
        speeds.push(Math.max(.07,Math.min(base,cornerCap)));
      }
      // Keep a small but nonzero command; the actor retains dynamic gait.
      for(let i=0;i<speeds.length;i++){
        const before=i===0?(through?Math.min(.38,speeds[i]):.12):speeds[i-1];
        speeds[i]=Math.max(.07,Math.min(speeds[i],
          Math.sqrt(before*before+2*accelerate*spans[i])));
      }
      for(let i=speeds.length-1;i>=0;i--){
        const after=i===speeds.length-1?(through?Math.min(.40,speeds[i]):.12):speeds[i+1];
        speeds[i]=Math.max(.07,Math.min(speeds[i],
          Math.sqrt(after*after+2*brake*spans[i])));
      }
      for(let j=0;j<speeds.length;j++){
        const p=dense[begin+j];
        const cmd=p.cmd;
        const old=Math.abs(kind==='lateral'?cmd.vy:cmd.vx);
        const factor=old>1e-7?speeds[j]/old:1;
        p.cmd={...cmd,vx:cmd.vx*factor,vy:cmd.vy*factor,
          yaw:clamp(cmd.yaw*factor,-yawMax,yawMax),profiled:true};
        if(spans[j]>.0001)p.dt=spans[j]/Math.max(.07,speeds[j]);
      }
    }
    begin=end+1;
  }
  // Never publish a modified route that is less safe than the raw plan.
  // Even an original curved primitive is rechecked by dense chords.
  if(!safePolyline(start,dense,trees,terrain)||
     (terminal&&!chordSafe(dense.at(-1),terminal,trees,0,terrain)))return rawSteps;
  return terminal?[...dense,terminal]:dense;
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
// The lattice ends in a tolerance disk, not at the target. For transit goals,
// append an actual SE(2) command connecting its terminal pose to the marker.
// This is NOT the old zero-command "final parking correction" masquerading
// as a drivable line. The exact-twist primitive is collision-checked.
export function flythroughConnector(from,goal,limits,trees=[],terrain=null){
  const dx=goal.x-from.x,dy=goal.y-from.y;
  const distanceToGoal=Math.hypot(dx,dy);
  if(distanceToGoal<.025)return null;
  const c=Math.cos(from.heading),s=Math.sin(from.heading);
  const localX=c*dx+s*dy,localY=-s*dx+c*dy;
  const angle=angleError(goal.heading,from.heading);
  const sinc=Math.abs(angle)<1e-6?1:Math.sin(angle)/angle;
  const vers=Math.abs(angle)<1e-6?0:(1-Math.cos(angle))/angle;
  const denom=sinc*sinc+vers*vers;
  if(denom<.15)return null;
  // Inverse of the exact integrated body twist: required displacement at
  // unit time, including simultaneous yaw and strafe.
  const requiredX=(sinc*localX+vers*localY)/denom;
  const requiredY=(-vers*localX+sinc*localY)/denom;
  const f=Math.max(.01,Number(limits.vx_max)||0)*.66;
  const b=Math.max(.01,-Number(limits.vx_min)||0)*.65;
  const y=Math.max(.01,Number(limits.vy_abs)||0)*.75;
  const yaw=Math.max(.01,Number(limits.yaw_abs)||0)*.72;
  const dt=Math.max(.28,distanceToGoal/.43,
    requiredX>=0?requiredX/f:-requiredX/b,
    Math.abs(requiredY)/y,Math.abs(angle)/yaw);
  // Avoid claiming a long, implausibly slow terminal shuffle is a smooth
  // fly-through. Ordinary exact-pose parking remains the safe fallback.
  if(dt>1.9)return null;
  const command={vx:requiredX/dt,vy:requiredY/dt,yaw:angle/dt,pitch:0};
  if(!segmentSafe(from,command,dt,trees,terrain))return null;
  const predicted=integratePose(from,command,dt);
  if(distance(predicted,goal)>.005 ||
     Math.abs(angleError(goal.heading,predicted.heading))>.005)return null;
  return {x:goal.x,y:goal.y,heading:goal.heading,cmd:command,
    dt,maneuver:'flythrough',flythrough:true};
}

export function planPoseRoute(start,goal,limits,trees=[],options={}){
  if(![start.x,start.y,start.heading,goal.x,goal.y,goal.heading].every(Number.isFinite))
    throw new Error('Nonfinite navigation start/goal');
  const moves=primitives(limits),speed=commandSpeeds(limits);
  const terrain=options.terrain||null;
  if(!inArena(goal,trees,BOUND,terrain))return null;
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
    if(distance(p,goal)<.31&&Math.abs(angleError(goal.heading,p.heading))<.40
       &&chordSafe(p,goal,trees,0,terrain)){
      winner=p;break;
    }
    for(const cmd of moves){
      if(!segmentSafe(p,cmd,cmd.dt,trees,terrain))continue;
      const q=integratePose(p,cmd,cmd.dt),id=key(q);
      const slopeCost=sweptTerrainCost(terrain,p,q);
      if(!Number.isFinite(slopeCost))continue;
      const g=p.g+cmd.cost*slopeCost;
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
  const flythrough=solved&&options.passThrough
    ?flythroughConnector(last,goal,limits,trees,terrain):null;
  // If A* landed right on the destination, the existing last primitive
  // already crosses the marker. Don't add a bogus zero-length stop.
  const exactPass=Boolean(solved&&options.passThrough&&
    distance(last,goal)<.025&&
    Math.abs(angleError(goal.heading,last.heading))<.09);
  if(flythrough)steps.push(flythrough);
  else if(!exactPass)steps.push({x:goal.x,y:goal.y,heading:goal.heading,
              cmd:{vx:0,vy:0,yaw:0,pitch:0},
              maneuver:'final parking correction',dt:0,final:true});
  const refined=refineRouteSteps(start,steps,limits,trees,terrain,
    Boolean(flythrough||exactPass));
  // Commanded-time estimate is only a planning hint, not measured gait
  // duration. Reflect extra braking instead of advertising raw A* time.
  const nominalTime=refined.reduce((seconds,p)=>seconds+(p.dt||0),0);
  return {start:{...start},steps:refined,
    estimatedSeconds:Math.max(last.g,nominalTime)+heuristic(last,goal,speed),
    solved,expanded,flythrough:Boolean(flythrough||exactPass),
    planner:terrain?'terrain-aware pose lattice':'pose lattice',
    refined:refined!==steps,
    modes:[...new Set(steps.map(x=>x.maneuver))]};
}
export class GoalNavigator {
  constructor(seed=5017){
    this.rng=makeRng(seed);this.active=false;
    this.goal=null;this.route=null;this.path=null;this.index=0;
    this.runIndex=0;this.runProgress=0;
    this.crossTrack=0;this.crossings=0;this.crossSide=0;
    this.trackingTime=0;this.trackErrorIntegral=0;
    this.trackErrorSquaredIntegral=0;this.distanceTravelled=0;
    this.previousTravelPose=null;
    this.headingErrorIntegral=0;this.speedErrorIntegral=0;
    this.referenceSpeed=0;this.observedSpeed=0;
    this.modeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
    this.plannedModeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
    this.currentMode='parking';
    this.elapsed=0;this.arrivalHold=0;this.arrivals=0;
    this.replans=0;this.reason='Not started';this.replanDelay=0;
    this.searchCooldown=0;this.failures=0;
    this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
    this.requestPlan=null;this.pendingRequest=null;
    this.generation=0;this.plannerError=null;
    this.controllerMode='predictive';
    this.predictor=new PredictivePathController();
    this.prediction=[];this.predictionScore=null;this.terrain=null;this.latestPose=null;
    this.benchmark=null;
    this.following=null;this.hint=null;
    this.prefetchPending=null;this.prefetchRetry=0;this.prefetchError=null;
    this.lastMovementPose=null;this.motionStall=0;
  }
  // Tests can still exercise the pure synchronous planner. Browser MUST
  // install the real Worker transport, never a Promise around sync A*.
  useBackgroundPlanner(requestPlan){
    if(typeof requestPlan!=='function')throw new Error('Background planner required');
    this.requestPlan=requestPlan;
  }
  queuePlan(payload,consume) {
    if(this.pendingRequest)return false;
    const generation=this.generation;
    let promise;
    try {promise=Promise.resolve(this.requestPlan(payload));}
    catch(error){this.plannerError=String(error?.message||error);return false;}
    this.pendingRequest=promise;
    promise.then(result=>{
      if(generation!==this.generation||!this.active)return;
      consume(result);
    }).catch(error=>{
      if(generation===this.generation){
        this.plannerError=String(error?.message||error);
        this.reason='Planner error · walking safely';
        this.searchCooldown=2.0;
        if(payload.kind==='next'){this.goal=null;this.route=null;this.path=null;}
      }
    }).finally(()=>{
      if(generation===this.generation&&this.pendingRequest===promise)
        this.pendingRequest=null;
    });
    return true;
  }
  reset(){
    this.generation++;this.pendingRequest=null;this.plannerError=null;
    this.goal=null;this.route=null;this.path=null;this.index=0;
    this.runIndex=0;this.runProgress=0;
    this.crossTrack=0;this.crossings=0;this.crossSide=0;
    this.trackingTime=0;this.trackErrorIntegral=0;
    this.trackErrorSquaredIntegral=0;this.distanceTravelled=0;
    this.previousTravelPose=null;
    this.headingErrorIntegral=0;this.speedErrorIntegral=0;
    this.referenceSpeed=0;this.observedSpeed=0;
    this.modeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
    this.plannedModeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
    this.currentMode='parking';
    this.elapsed=0;this.arrivalHold=0;this.reason='Ready for a target';
    this.replanDelay=0;this.replans=0;this.searchCooldown=0;
    this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
    this.prediction=[];this.predictionScore=null;this.latestPose=null;
    this.predictor.reset();this.benchmark=null;
    this.following=null;this.hint=null;this.prefetchPending=null;
    this.prefetchRetry=0;this.prefetchError=null;
    this.lastMovementPose=null;this.motionStall=0;}
  startBenchmark(){
    const keepActive=this.active;
    this.reset();
    this.active=keepActive;
    this.benchmark={targets:NAV_BENCHMARK_TARGETS.map(g=>({...g})),
      index:0,results:[],complete:false,controller:this.controllerMode};
    this.reason='Benchmark · route 1 / '+NAV_BENCHMARK_TARGETS.length;
  }
  finishBenchmarkLeg(success,fallen=false){
    if(!this.benchmark||this.benchmark.complete)return;
    this.benchmark.results.push({
      goal:this.benchmark.index+1,success,
      elapsedSeconds:Number(this.elapsed.toFixed(2)),
      meanTrackErrorM:Number((this.trackingTime?
        this.trackErrorIntegral/this.trackingTime:0).toFixed(3)),
      rmsTrackErrorM:Number((this.trackingTime?Math.sqrt(
        this.trackErrorSquaredIntegral/this.trackingTime):0).toFixed(3)),
      distanceTravelledM:Number(this.distanceTravelled.toFixed(3)),
      fell:Boolean(fallen),
      meanHeadingErrorDeg:Number((this.trackingTime?
        this.headingErrorIntegral/this.trackingTime*180/Math.PI:0).toFixed(1)),
      crossings:this.crossings});
    this.benchmark.index++;
    if(this.benchmark.index>=this.benchmark.targets.length){
      this.benchmark.complete=true;
      this.reason='Benchmark finished · '+this.benchmark.results.filter(r=>r.success).length+
        '/'+this.benchmark.results.length+' successful';
    }
  }
  acceptRoute(pose,goal,route,{newMission=false,following=null}={}){
    this.goal={...goal};this.route=route;
    this.path=buildRoutePath(route,route.start||pose);
    this.plannedModeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
    for(const step of route.steps){
      const kind=motionKind(step.cmd);
      if(Object.hasOwn(this.plannedModeSeconds,kind))
        this.plannedModeSeconds[kind]+=Math.max(0,Number(step.dt)||0);
    }
    this.runIndex=0;this.runProgress=0;this.index=0;
    this.stagnant=0;this.bestProgress=-Infinity;this.lastIndex=-1;
    this.replanDelay=0;
    if(newMission){
      this.following=following;
      this.hint=following?.hint||null;
      this.lastMovementPose=null;this.motionStall=0;
      this.elapsed=0;this.arrivalHold=0;this.searchCooldown=0;
      this.crossTrack=0;this.crossings=0;this.crossSide=0;
      this.trackingTime=0;this.trackErrorIntegral=0;
    this.trackErrorSquaredIntegral=0;this.distanceTravelled=0;
    this.previousTravelPose=null;
      this.headingErrorIntegral=0;this.speedErrorIntegral=0;
      this.referenceSpeed=0;this.observedSpeed=0;
      this.modeSeconds={forward:0,reverse:0,lateral:0,pivot:0,parking:0};
      this.currentMode='parking';
    }
    this.reason=route.solved?'Following route':'Partial route · replanning';
  }
  setGoal(pose,goal,limits,trees=[]){
    const route=planPoseRoute(pose,goal,limits,trees,{terrain:this.terrain});
    if(!route)return false;
    this.acceptRoute(pose,goal,route,{newMission:true});
    return true;
  }
  nextGoal(pose,limits,trees=[]){
    if(this.benchmark){
      if(this.benchmark.complete)return false;
      const goal=this.benchmark.targets[this.benchmark.index];
      if(this.requestPlan){
        this.reason='Benchmark · planning goal '+(this.benchmark.index+1);
        return this.queuePlan({
          kind:'target',pose:{x:pose.x,y:pose.y,heading:pose.heading},
          goal,limits,trees,terrain:this.terrain
        },answer=>{
          const fresh=this.latestPose&&distance(this.latestPose,pose)<.30;
          if(answer?.route?.solved&&fresh){
            this.acceptRoute(pose,goal,answer.route,{newMission:true});
          }else{
            this.finishBenchmarkLeg(false);
            this.goal=null;this.route=null;this.path=null;
            this.searchCooldown=.5;
          }
        });
      }
      if(this.setGoal(pose,goal,limits,trees))return true;
      this.finishBenchmarkLeg(false);
      this.searchCooldown=.5;
      return false;
    }
    if(this.requestPlan){
      this.reason='Planning next target in background…';
      return this.queuePlan({
        kind:'next',pose:{x:pose.x,y:pose.y,heading:pose.heading},
        limits,trees,terrain:this.terrain
      },result=>{
        if(result?.goal&&result?.route?.solved){
          // Route already calculated off-thread; do not run A* again.
          this.acceptRoute(pose,result.goal,result.route,{
            newMission:true,following:result.following||null});
        }else{
          this.goal=null;this.route=null;this.path=null;
          this.searchCooldown=1.5;
          this.reason='No viable route yet · trying again soon';
        }
      });
    }
    for(let i=0;i<5;i++){
      const goal=sampleGoal(this.rng,trees,pose,ARENA_RADIUS_M,this.terrain);
      const planned=this.setGoal(pose,goal,limits,trees);
      if(planned&&this.route.solved)return true;
    }
    this.reason='No feasible full route; sampling again soon';
    this.goal=null;this.route=null;this.path=null;this.searchCooldown=1.5;
    return false;
  }
  // Prefetch from the PLANNED terminal pose while the actor is still
  // walking to it. No zero-command wait for the browser Worker at the marker.
  // This request is independent of replanning the live route.
  prefetchUpcoming(limits,trees=[]){
    if(!this.requestPlan||this.benchmark||!this.goal||
       this.following||this.prefetchPending||this.prefetchRetry>0)return false;
    const expectedGoal=this.goal,generation=this.generation;
    let request;
    try{
      request=Promise.resolve(this.requestPlan({
        kind:'prefetch',
        pose:{x:expectedGoal.x,y:expectedGoal.y,heading:expectedGoal.heading},
        hint:this.hint,limits,trees,terrain:this.terrain
      }));
    }catch(error){
      this.prefetchError=String(error?.message||error);
      this.prefetchRetry=1.5;
      return false;
    }
    this.prefetchPending=request;
    request.then(result=>{
      if(!this.active||generation!==this.generation||
         !samePose(this.goal,expectedGoal))return;
      if(result?.following?.route?.solved&&result.following.route.flythrough){
        this.following=result.following;
        this.prefetchError=null;
      }else{
        this.prefetchRetry=1.5;
        this.prefetchError='No continuous next route available';
      }
    }).catch(error=>{
      if(generation!==this.generation||!samePose(this.goal,expectedGoal))return;
      this.prefetchError=String(error?.message||error);
      this.prefetchRetry=1.5;
    }).finally(()=>{
      if(this.prefetchPending===request)this.prefetchPending=null;
    });
    return true;
  }
  tick(dt,pose,limits,trees=[],apply=()=>{}){
    if(!this.active)return;
    if(![pose.x,pose.y,pose.heading].every(Number.isFinite))return;
    const delta=clamp(dt,0,.1);
    this.prefetchRetry=Math.max(0,this.prefetchRetry-delta);
    this.latestPose={x:pose.x,y:pose.y,heading:pose.heading};
    if(this.goal&&this.previousTravelPose)
      this.distanceTravelled+=distance(pose,this.previousTravelPose);
    this.previousTravelPose={x:pose.x,y:pose.y};
    this.predictor.observe(pose,delta);
    // Standing while waiting avoids making an asynchronous plan stale.
    if(this.benchmark?.complete){
      apply({vx:0,vy:0,yaw:0,pitch:0});return;
    }
    if(!this.goal){
      if(this.pendingRequest){
        this.reason='Background route search';
        apply({vx:0,vy:0,yaw:0,pitch:0});return;
      }
      if(this.searchCooldown>0){
        this.searchCooldown=Math.max(0,this.searchCooldown-delta);
        apply({vx:0,vy:0,yaw:0,pitch:0});return;
      }
      if(!this.nextGoal(pose,limits,trees)||!this.goal){
        apply({vx:0,vy:0,yaw:0,pitch:0});return;
      }
    }
    if(!this.benchmark&&!this.following&&this.route?.flythrough)
      this.prefetchUpcoming(limits,trees);
    this.elapsed+=delta;
    const deadline=clamp(this.route.estimatedSeconds*4.5+16,30,105);
    if(this.elapsed>deadline){
      this.failures++;
      this.finishBenchmarkLeg(false);
      this.goal=null;this.route=null;this.path=null;
      this.reason=this.benchmark?.complete?this.reason:'Mission timed out · new goal';
      apply({vx:0,vy:0,yaw:0,pitch:0});
      this.nextGoal(pose,limits,trees);
      return;
    }
    let dist=distance(pose,this.goal);
    let aerr=Math.abs(angleError(this.goal.heading,pose.heading));
    const speed=Math.hypot(pose.vx||0,pose.vy||0);
    // A normal destination is a checkpoint on a continuous walk, not a
    // compulsory full stop. Exact-stop-and-face stays in benchmark mode.
    if(!this.benchmark&&this.route.flythrough&&this.following&&
       dist<.25&&aerr<.70){
      const next=this.following;
      this.arrivals++;
      this.acceptRoute(pose,next.goal,next.route,{newMission:true});
      // The new goal's route was prepared before the previous crossing.
      // Keep the gait's actual velocity and let the follower run this tick.
      this.hint=next.hint||null;
      this.prefetchUpcoming(limits,trees);
      dist=distance(pose,this.goal);
      aerr=Math.abs(angleError(this.goal.heading,pose.heading));
      this.reason='Flowing through waypoint · next target underway';
    }
    if((this.benchmark||!this.following||!this.route.flythrough)&&
       dist<GOAL_POSITION_TOLERANCE_M&&
       aerr<GOAL_HEADING_TOLERANCE_RAD&&speed<.23){
      this.arrivalHold+=delta;
      apply({vx:0,vy:0,yaw:0,pitch:0});
      if(this.arrivalHold>=.55){
        // Retire this mission before asking the Worker for a new one.
        this.arrivals++;
        this.finishBenchmarkLeg(true);
        this.goal=null;this.route=null;this.path=null;
        this.reason=this.benchmark?.complete?this.reason:
          'Goal reached · choosing next target';
        this.nextGoal(pose,limits,trees);
      }else this.reason='Verifying arrival';
      return;
    }
    this.arrivalHold=0;
    const path=this.path;
    const lastRun=path?.runs.length-1??-1;
    let result=null;
    if(this.runIndex<=lastRun){
      result=trackRoutePath(pose,path,this.runIndex,this.runProgress,limits);
      this.runProgress=Math.max(this.runProgress,result.progress);
      // Respect reverse/pivot boundaries; do not jump across a cusp.
      if(result.ready&&this.runIndex<lastRun){
        this.runIndex++;
        this.runProgress=0;
        result=trackRoutePath(pose,path,this.runIndex,0,limits);
      }else if(result.ready&&this.runIndex===lastRun){
        this.runIndex=lastRun+1;
      }
    }
    // Ease into final pose correction rather than switching controllers
    // abruptly at a fixed distance. Preserve pivots and reverse cusps.
    const finished=this.runIndex>lastRun||lastRun<0;
    const blending=!(this.route?.flythrough&&this.following)&&
      !finished&&this.runIndex===lastRun&&
      result?.mode!=='pivot'&&dist<.60&&result.remaining<.60;
    const blend=blending?clamp((.60-dist)/.40,0,1)*
      clamp((.60-result.remaining)/.42,0,1):0;
    const finalApproach=finished||blending;
    const cruise=result?.command||{vx:0,vy:0,yaw:0,pitch:0};
    const park=finished||blending?parkAtGoal(pose,this.goal,limits):null;
    let command=finished?park:!blending?cruise:
      Object.fromEntries(['vx','vy','yaw','pitch'].map(key=>[
        key,(1-blend)*(cruise[key]||0)+blend*(park[key]||0)]));
    this.prediction=[];this.predictionScore=null;
    if(this.controllerMode==='predictive'&&!finished&&!blending&&
       result?.mode!=='pivot'){
      const predicted=this.predictor.command(pose,path,this.runIndex,
        this.runProgress,limits,trees,this.terrain,command);
      command=predicted.command;
      this.prediction=predicted.prediction;
      this.predictionScore=predicted.cost;
    }
    this.predictor.issue(command);
    apply(command);
    this.reason=finalApproach?'Final pose correction'
      :result?.mode==='pivot'?'Following planned pivot'
      :dist>1.1?'Following path':'Approaching goal';
    const progress=this.runIndex*10000+this.runProgress;
    if(!this.lastMovementPose||
       distance(pose,this.lastMovementPose)>.09){
      this.lastMovementPose={x:pose.x,y:pose.y};this.motionStall=0;
    }else if(dist>.31&&result?.mode!=='pivot'&&
        Math.hypot(command.vx||0,command.vy||0)>.18)
      this.motionStall+=delta;
    else this.motionStall=Math.max(0,this.motionStall-.12);
    if(progress>this.bestProgress+.045){
      this.bestProgress=progress;this.stagnant=0;
    }else if(!finalApproach&&result?.mode!=='pivot')this.stagnant+=delta;
    this.index=this.runIndex;
    this.replanDelay+=delta;
    const signedCross=result?.mode==='pivot'?0:(result?.crossTrack||0);
    this.crossTrack=signedCross;
    const currentMode=finished?'parking':result?.mode||'parking';
    this.currentMode=currentMode;
    this.modeSeconds[currentMode]=(this.modeSeconds[currentMode]||0)+delta;
    this.trackingTime+=delta;
    this.trackErrorIntegral+=Math.abs(signedCross)*delta;
    this.trackErrorSquaredIntegral+=signedCross*signedCross*delta;
    this.headingErrorIntegral+=Math.abs(result?.headingError||0)*delta;
    this.speedErrorIntegral+=Math.abs(result?.speedError||0)*delta;
    this.referenceSpeed=result?.referenceSpeed||0;
    this.observedSpeed=result?.observedSpeed||0;
    // Hysteresis avoids treating each gait-induced wiggle as a crossing.
    const side=Math.abs(signedCross)>.045?Math.sign(signedCross):0;
    if(side&&this.crossSide&&side!==this.crossSide)this.crossings++;
    if(side)this.crossSide=side;
    const offRoute=Math.abs(signedCross);
    // Compare with the line, NOT distance to a far-ahead lookahead target.
    if(this.replanDelay>2.5&&!this.pendingRequest&&
       (offRoute>.67||!this.route.solved||
        (!finalApproach&&this.stagnant>3.2)||
        (dist>.34&&this.motionStall>2.4)||
        (finalApproach&&dist>.33&&this.stagnant>4.5))){
      this.replanDelay=0;
      if(this.requestPlan){
        const expectedGoal=this.goal;
        this.queuePlan({
          kind:'replan',pose:{x:pose.x,y:pose.y,heading:pose.heading},
          goal:expectedGoal,limits,trees,terrain:this.terrain,
          passThrough:!this.benchmark&&Boolean(this.route?.flythrough)
        },answer=>{
          // Reject stale background routes after significant motion.
          const fresh=this.latestPose&&distance(this.latestPose,pose)<.38&&
            Math.abs(angleError(this.latestPose.heading,pose.heading))<.45;
          if(answer?.route?.solved&&this.goal===expectedGoal&&fresh){
            this.acceptRoute(pose,expectedGoal,answer.route);
            this.replans++;
          }
        });
      }else{
        const route=planPoseRoute(pose,this.goal,limits,trees,
          {maxNodes:7500,terrain:this.terrain,
            passThrough:!this.benchmark&&Boolean(this.route?.flythrough)});
        if(route?.solved){
          this.acceptRoute(pose,this.goal,route);
          this.replans++;
        }
      }
    }
  }
  status(pose){
    return {goal:this.goal,route:this.route,
      distance:this.goal?distance(pose,this.goal):0,
      headingError:this.goal?Math.abs(angleError(this.goal.heading,pose.heading)):0,
      elapsed:this.elapsed,arrivals:this.arrivals,replans:this.replans,
      crossTrack:this.crossTrack,crossings:this.crossings,
      meanTrackError:this.trackingTime?this.trackErrorIntegral/this.trackingTime:0,
      rmsTrackError:this.trackingTime?
        Math.sqrt(this.trackErrorSquaredIntegral/this.trackingTime):0,
      distanceTravelled:this.distanceTravelled,
      meanHeadingError:this.trackingTime?this.headingErrorIntegral/this.trackingTime:0,
      meanSpeedError:this.trackingTime?this.speedErrorIntegral/this.trackingTime:0,
      referenceSpeed:this.referenceSpeed,observedSpeed:this.observedSpeed,
      modeSeconds:{...this.modeSeconds},
      plannedModeSeconds:{...this.plannedModeSeconds},currentMode:this.currentMode,
      failures:this.failures,planning:Boolean(this.pendingRequest),
      plannerError:this.plannerError,
      index:this.index,reason:this.reason,
      controllerMode:this.controllerMode,prediction:this.prediction,
      predictionScore:this.predictionScore,
      responseModel:this.predictor.model.report(),
      following:this.following||null,
      prefetching:Boolean(this.prefetchPending),
      prefetchError:this.prefetchError,
      motionStall:this.motionStall,
      predictionFallback:Boolean(this.predictor.fallback),
      flythrough:Boolean(this.route?.flythrough),
      benchmark:this.benchmark?{
        controller:this.benchmark.controller,index:this.benchmark.index,
        goals:this.benchmark.targets.length,
        complete:this.benchmark.complete,
        results:[...this.benchmark.results]}:null};
  }
}
