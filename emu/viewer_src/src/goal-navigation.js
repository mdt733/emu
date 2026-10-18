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
  // This dynamic biped was not taught to pivot while standing still.
  // Prefer forward/reverse/side arcs for heading changes.
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

/**
 * Select a target ~1.1-1.7 metres AHEAD along the planned polyline.
 * The old controller chased the next 22cm lattice node AND its heading,
 * which forced a slow sidestep to each quantised position. Pure pursuit
 * instead advances monotonically and treats intermediate headings as
 * planning context, NEVER as intermediate parking obligations.
 */
export function previewRoute(pose,steps,index=0,trees=[],speed=0){
  if(!steps?.length)throw new Error('Cannot follow empty navigation route');
  const last=steps.length-1;
  // Search only the local corridor; looping paths can cross themselves.
  let nearest=clamp(index,0,last),best=distance(pose,steps[nearest]);
  for(let i=nearest+1;i<=Math.min(last,index+12);i++){
    const d=distance(pose,steps[i]);
    if(d<best-.025){best=d;nearest=i;}
  }
  const lookahead=clamp(1.10+.65*Math.max(0,speed),1.10,1.70);
  let travelled=0,look=steps[nearest],targetIndex=nearest;
  let prev={x:pose.x,y:pose.y};
  for(let i=nearest;i<=last;i++){
    const q=steps[i],segment=distance(prev,q);
    if(travelled+segment>=lookahead && segment>.00001){
      const t=(lookahead-travelled)/segment;
      const candidate={...q,x:prev.x+(q.x-prev.x)*t,
                             y:prev.y+(q.y-prev.y)*t};
      if(directRouteClear(pose,candidate,trees)){
        look=candidate;targetIndex=i;
      }
      break;
    }
    if(!directRouteClear(pose,q,trees))break;
    travelled+=segment;prev=q;look=q;targetIndex=i;
  }
  // If a segment is directly blocked by a tree, don't shortcut the path;
  // return the last visible intermediate point instead.
  return {target:look,progress:nearest,targetIndex,
          lookahead,remaining:distance(pose,steps[last])};
}
function directRouteClear(from,to,trees){
  const length=distance(from,to);
  for(let i=1;i<=Math.max(1,Math.ceil(length/.10));i++){
    const t=i/Math.max(1,Math.ceil(length/.10));
    const x=from.x+(to.x-from.x)*t,y=from.y+(to.y-from.y)*t;
    if(treeCollision(x,y,trees))return false;
  }
  return true;
}

/**
 * Velocity controller for a DYNAMIC biped, not a holonomic parked robot.
 *
 * Far: drive quickly toward a far-ahead preview with one smooth yaw arc;
 *      no lateral corrections for intermediate lattice errors.
 * Near: release cruise velocity and steer toward the final pose.
 * Last metres: finish world-heading approach and use modest sideways
 *      trim if needed. All velocities remain within the learned envelope.
 */
export function followWaypoint(pose,waypoint,limits,{final=false,goal=null}={}){
  const actualGoal=goal||waypoint;
  const remaining=distance(pose,actualGoal);
  const dx=waypoint.x-pose.x,dy=waypoint.y-pose.y;
  const sight=Math.atan2(dy,dx);
  const eBearing=angleError(sight,pose.heading);
  const finalError=angleError(actualGoal.heading,pose.heading);
  const maxF=Math.max(0,Number(limits.vx_max)||0);
  const maxB=Math.max(0,-Number(limits.vx_min)||0);
  const maxS=Math.max(0,Number(limits.vy_abs)||0);
  const maxYaw=Math.max(0,Number(limits.yaw_abs)||0);
  const requested=waypoint.cmd||{vx:maxF,vy:0,yaw:0};
  const reverse= !final && Number(requested.vx)<-.07 &&
      remaining>.65 && Math.abs(eBearing)>1.65 && maxB>.08;
  const sightError=reverse?wrapAngle(eBearing-Math.PI*Math.sign(eBearing)):eBearing;
  // Heading alignment is gradual, not a requirement to stop at each node.
  const arrivalBlend=clamp((1.10-remaining)/.90,0,1);
  const turnError=(1-arrivalBlend*.76)*sightError+arrivalBlend*.76*finalError;
  const yaw=safe(1.45*turnError,-maxYaw*.82,maxYaw*.82);
  // Sprint when there is space and the turning radius permits it.
  const cruise=maxF*.90;
  const distanceThrottle=remaining>=1.5?1:
    remaining>=.75?(.52+.48*(remaining-.75)/.75):
    remaining>=.33?(.25+.27*(remaining-.33)/.42):
    .10;
  const curveThrottle=clamp(1-.65*Math.abs(sightError)/Math.PI,.28,1);
  let vx=cruise*distanceThrottle*curveThrottle;
  // This is a dynamic biped; maintain forward stepping in a broad turn.
  if(remaining>.55)vx=Math.max(vx,Math.min(.27,maxF*.30));
  if(reverse){
    vx=-Math.min(maxB*.76,Math.max(.12,maxB*.70*distanceThrottle));
  }
  // Sideways speed is precious: do NOT expend it following every 22cm
  // lattice point. Only trim residual error in the arrival neighbourhood.
  let vy=0;
  if(remaining<.64){
    const ex=actualGoal.x-pose.x,ey=actualGoal.y-pose.y;
    const lateral=-Math.sin(pose.heading)*ex+Math.cos(pose.heading)*ey;
    vy=clamp(lateral*.70,-maxS*.68,maxS*.68);
    // Do not compound full forward and lateral frontiers.
    const ellipse=Math.hypot(vx/Math.max(maxF*.90,.01),vy/Math.max(maxS*.70,.01));
    if(ellipse>1){vx/=ellipse;vy/=ellipse;}
  }
  if(final && remaining<.28 && Math.abs(finalError)>.30){
    // Give orientation correction room while still dynamically stepping.
    vx=Math.max(vx,Math.min(maxF*.16,.15));
  }
  return {vx:safe(vx,-maxB,maxF),vy:safe(vy,-maxS,maxS),
          yaw,pitch:0};
}

export class GoalNavigator {
  constructor(seed=5017){
    this.rng=makeRng(seed);this.active=false;
    this.goal=null;this.route=null;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.arrivals=0;
    this.replans=0;this.reason='Not started';this.replanDelay=0;
    this.searchCooldown=0;this.failures=0;this.warmup=0;
    this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
    this.requestPlan=null;this.pendingRequest=null;
    this.generation=0;this.plannerError=null;
    this.plannerMode='grid'; // UI selection; only changes background route search
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
        if(payload.kind==='next'){this.goal=null;this.route=null;}
      }
    }).finally(()=>{
      if(generation===this.generation&&this.pendingRequest===promise)
        this.pendingRequest=null;
    });
    return true;
  }
  reset(){
    this.generation++;this.pendingRequest=null;this.plannerError=null;
    this.goal=null;this.route=null;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.reason='Ready for a target';
    this.replanDelay=0;this.replans=0;this.searchCooldown=0;
    this.warmup=0;this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;}
  setGoal(pose,goal,limits,trees=[]){
    const route=planPoseRoute(pose,goal,limits,trees);
    if(!route)return false;
    this.goal={...goal};this.route=route;this.index=0;
    this.elapsed=0;this.arrivalHold=0;this.replanDelay=0;
    this.searchCooldown=0;this.stagnant=0;
    this.bestProgress=Infinity;this.lastIndex=-1;
    this.reason=route.solved?'Navigating':'Partial route · replanning';
    return true;
  }
  nextGoal(pose,limits,trees=[]){
    if(this.requestPlan){
      this.reason='Planning next target in background…';
      return this.queuePlan({
        kind:'next',pose:{x:pose.x,y:pose.y,heading:pose.heading},
        limits,trees,planner:this.plannerMode
      },result=>{
        if(result?.goal&&result?.route?.solved){
          // Route already calculated off-thread; do not run A* again.
          this.goal=result.goal;this.route=result.route;this.index=0;
          this.elapsed=0;this.replanDelay=0;this.arrivalHold=0;
          this.searchCooldown=0;this.stagnant=0;
          this.bestProgress=Infinity;this.lastIndex=-1;
          this.reason='Navigating';
        }else{
          this.goal=null;this.route=null;
          this.searchCooldown=1.5;
          this.reason='No viable route yet · trying again soon';
        }
      });
    }
    for(let i=0;i<5;i++){
      const goal=sampleGoal(this.rng,trees,pose);
      const planned=this.setGoal(pose,goal,limits,trees);
      if(planned&&this.route.solved)return true;
    }
    this.reason='No feasible full route; sampling again soon';
    this.goal=null;this.route=null;this.searchCooldown=1.5;
    return false;
  }
  tick(dt,pose,limits,trees=[],apply=()=>{}){
    if(!this.active)return;
    if(!Number.isFinite(pose.x)||!Number.isFinite(pose.y)||
       !Number.isFinite(pose.heading))return;
    const delta=clamp(dt,0,.1);
    // Start from dynamic gait, not a stationary pivot/standing command.
    if(this.warmup<1.2){
      this.warmup+=delta;
      this.reason='Walking warm-up before navigation';
      apply({vx:Math.min(.20,Math.max(.08,limits.vx_max*.38)),
             vy:0,yaw:0,pitch:0});
      return;
    }
    if(!this.goal){
      if(this.pendingRequest){
        this.reason='Background route search · still walking';
        apply({vx:.14,vy:0,yaw:0,pitch:0});
        return;
      }
      if(this.searchCooldown>0){
        this.searchCooldown=Math.max(0,this.searchCooldown-delta);
        apply({vx:.14,vy:0,yaw:0,pitch:0});return;
      }
      if(!this.nextGoal(pose,limits,trees)||!this.goal){
        // The Worker has been scheduled but hasn't returned a route yet.
        // Advance the policy with normal walking commands in the meantime.
        apply({vx:.14,vy:0,yaw:0,pitch:0});return;
      }
    }
    this.elapsed+=delta;
    const deadline=clamp(this.route.estimatedSeconds*4.5+16,30,105);
    if(this.elapsed>deadline){
      this.failures++;
      this.reason='Mission timed out · new goal';
      apply({vx:.14,vy:0,yaw:0,pitch:0});
      this.nextGoal(pose,limits,trees);
      return;
    }
    const dist=distance(pose,this.goal),aerr=Math.abs(angleError(this.goal.heading,pose.heading));
    const speed=Math.hypot(pose.vx||0,pose.vy||0);
    if(dist<GOAL_POSITION_TOLERANCE_M&&aerr<GOAL_HEADING_TOLERANCE_RAD&&speed<.23){
      // Dynamic arrival: NEVER leave the last turn/sideways command latched
      // while searching for another destination. Keep a gentle walking
      // request applied on EVERY simulator frame, even between missions.
      apply({vx:Math.min(.14,Math.max(.09,limits.vx_max*.25)),
             vy:0,yaw:0,pitch:0});
      this.arrivals++;
      this.reason='Pose reached while walking · next target';
      this.nextGoal(pose,limits,trees);
      return;
    }
    this.arrivalHold=0;
    const steps=this.route.steps;
    const preview=previewRoute(pose,steps,this.index,trees,
      Math.max(0,pose.vx||0));
    // Route progress is based on actual spatial progress, NOT reaching the
    // heading of every intermediate node. The planner's final heading is
    // still enforced at the actual destination.
    this.index=Math.max(this.index,preview.progress);
    const target=preview.target;
    const remaining=distance(pose,target);
    if(this.index!==this.lastIndex){
      this.lastIndex=this.index;this.bestProgress=remaining;this.stagnant=0;
    }else if(remaining<this.bestProgress-.08){
      this.bestProgress=remaining;this.stagnant=0;
    }else this.stagnant+=delta;
    const command=followWaypoint(pose,target,limits,{
      goal:this.goal,final:preview.remaining<.65
    });
    apply(command);
    const targetDistance=distance(pose,this.goal);
    this.reason=targetDistance>1.25?'Cruising toward goal'
      :targetDistance>.48?'Curved approach':'Final pose alignment';
    this.replanDelay+=delta;
    // Re-plan if significantly displaced from the planned next target.
    // Do not thrash the target from scratch every render frame.
    if(this.replanDelay>3.2 && (remaining>.75||!this.route.solved||this.stagnant>6.0)){
      this.replanDelay=0;
      if(this.requestPlan){
        const expectedGoal=this.goal;
        this.queuePlan({
          kind:'replan',pose:{x:pose.x,y:pose.y,heading:pose.heading},
          goal:expectedGoal,limits,trees,planner:this.plannerMode
        },result=>{
          if(result?.route?.solved && this.goal===expectedGoal){
            this.route=result.route;this.index=0;this.replans++;
            this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
          }
        });
      }else{
        const route=planPoseRoute(pose,this.goal,limits,trees,{maxNodes:7500});
        if(route){
          this.route=route;this.index=0;this.replans++;
          this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
        }
      }
    }
  }
  status(pose){
    return {goal:this.goal,route:this.route,
      distance:this.goal?distance(pose,this.goal):0,
      headingError:this.goal?Math.abs(angleError(this.goal.heading,pose.heading)):0,
      elapsed:this.elapsed,arrivals:this.arrivals,replans:this.replans,
      failures:this.failures,planning:Boolean(this.pendingRequest),
      plannerError:this.plannerError,
      index:this.index,reason:this.reason};
  }
}
