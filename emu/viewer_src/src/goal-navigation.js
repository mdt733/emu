// Browser-only pose navigation: time-weighted SE(2) lattice and closed-loop
// continuous line tracking. This plans *commands*, never neural weights. It is
// an estimated fastest route under a conservative velocity envelope,
// not proof of optimality or a learned capability/safety guarantee.
import { makeRng } from './rng.js';
import { buildRoutePath, trackRoutePath, parkAtGoal } from './path-follower.js';
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
  return {start:{...start},steps,estimatedSeconds:last.g+
    heuristic(last,goal,speed),solved,expanded,
    modes:[...new Set(steps.map(x=>x.maneuver))]};
}
export class GoalNavigator {
  constructor(seed=5017){
    this.rng=makeRng(seed);this.active=false;
    this.goal=null;this.route=null;this.path=null;this.index=0;
    this.runIndex=0;this.runProgress=0;
    this.crossTrack=0;this.crossings=0;this.crossSide=0;
    this.elapsed=0;this.arrivalHold=0;this.arrivals=0;
    this.replans=0;this.reason='Not started';this.replanDelay=0;
    this.searchCooldown=0;this.failures=0;
    this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;
    this.requestPlan=null;this.pendingRequest=null;
    this.generation=0;this.plannerError=null;
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
    this.elapsed=0;this.arrivalHold=0;this.reason='Ready for a target';
    this.replanDelay=0;this.replans=0;this.searchCooldown=0;
    this.stagnant=0;this.bestProgress=Infinity;this.lastIndex=-1;}
  acceptRoute(pose,goal,route,{newMission=false}={}){
    this.goal={...goal};this.route=route;
    this.path=buildRoutePath(route,route.start||pose);
    this.runIndex=0;this.runProgress=0;this.index=0;
    this.stagnant=0;this.bestProgress=-Infinity;this.lastIndex=-1;
    this.replanDelay=0;
    if(newMission){
      this.elapsed=0;this.arrivalHold=0;this.searchCooldown=0;
      this.crossTrack=0;this.crossings=0;this.crossSide=0;
    }
    this.reason=route.solved?'Following route':'Partial route · replanning';
  }
  setGoal(pose,goal,limits,trees=[]){
    const route=planPoseRoute(pose,goal,limits,trees);
    if(!route)return false;
    this.acceptRoute(pose,goal,route,{newMission:true});
    return true;
  }
  nextGoal(pose,limits,trees=[]){
    if(this.requestPlan){
      this.reason='Planning next target in background…';
      return this.queuePlan({
        kind:'next',pose:{x:pose.x,y:pose.y,heading:pose.heading},
        limits,trees
      },result=>{
        if(result?.goal&&result?.route?.solved){
          // Route already calculated off-thread; do not run A* again.
          this.acceptRoute(pose,result.goal,result.route,{newMission:true});
        }else{
          this.goal=null;this.route=null;this.path=null;
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
    this.goal=null;this.route=null;this.path=null;this.searchCooldown=1.5;
    return false;
  }
  tick(dt,pose,limits,trees=[],apply=()=>{}){
    if(!this.active)return;
    if(![pose.x,pose.y,pose.heading].every(Number.isFinite))return;
    const delta=clamp(dt,0,.1);
    // Standing while waiting avoids making an asynchronous plan stale.
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
    this.elapsed+=delta;
    const deadline=clamp(this.route.estimatedSeconds*4.5+16,30,105);
    if(this.elapsed>deadline){
      this.failures++;
      this.goal=null;this.route=null;this.path=null;
      this.reason='Mission timed out · new goal';
      apply({vx:0,vy:0,yaw:0,pitch:0});
      this.nextGoal(pose,limits,trees);
      return;
    }
    const dist=distance(pose,this.goal);
    const aerr=Math.abs(angleError(this.goal.heading,pose.heading));
    const speed=Math.hypot(pose.vx||0,pose.vy||0);
    if(dist<GOAL_POSITION_TOLERANCE_M&&
       aerr<GOAL_HEADING_TOLERANCE_RAD&&speed<.23){
      this.arrivalHold+=delta;
      apply({vx:0,vy:0,yaw:0,pitch:0});
      if(this.arrivalHold>=.55){
        // Retire this mission before asking the Worker for a new one.
        this.arrivals++;
        this.goal=null;this.route=null;this.path=null;
        this.reason='Goal reached · choosing next target';
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
    const finalApproach=dist<.42&&
      (this.runIndex>=lastRun||lastRun<0);
    const command=finalApproach||this.runIndex>lastRun
      ?parkAtGoal(pose,this.goal,limits)
      :result.command;
    apply(command);
    this.reason=finalApproach?'Final pose correction'
      :result?.mode==='pivot'?'Following planned pivot'
      :dist>1.1?'Following path':'Approaching goal';
    const progress=this.runIndex*10000+this.runProgress;
    if(progress>this.bestProgress+.045){
      this.bestProgress=progress;this.stagnant=0;
    }else if(!finalApproach)this.stagnant+=delta;
    this.index=this.runIndex;
    this.replanDelay+=delta;
    const signedCross=result?.mode==='pivot'?0:(result?.crossTrack||0);
    this.crossTrack=signedCross;
    // Hysteresis avoids treating each gait-induced wiggle as a crossing.
    const side=Math.abs(signedCross)>.045?Math.sign(signedCross):0;
    if(side&&this.crossSide&&side!==this.crossSide)this.crossings++;
    if(side)this.crossSide=side;
    const offRoute=Math.abs(signedCross);
    // Compare with the line, NOT distance to a far-ahead lookahead target.
    if(this.replanDelay>3.2&&!this.pendingRequest&&
       (offRoute>.80||!this.route.solved||(!finalApproach&&this.stagnant>6))){
      this.replanDelay=0;
      if(this.requestPlan){
        const expectedGoal=this.goal;
        this.queuePlan({
          kind:'replan',pose:{x:pose.x,y:pose.y,heading:pose.heading},
          goal:expectedGoal,limits,trees
        },answer=>{
          if(answer?.route?.solved&&this.goal===expectedGoal){
            this.acceptRoute(pose,expectedGoal,answer.route);
            this.replans++;
          }
        });
      }else{
        const route=planPoseRoute(pose,this.goal,limits,trees,{maxNodes:7500});
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
      failures:this.failures,planning:Boolean(this.pendingRequest),
      plannerError:this.plannerError,
      index:this.index,reason:this.reason};
  }
}
