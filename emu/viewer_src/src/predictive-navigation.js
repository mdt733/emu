// Receding-horizon, coupled vx/vy/yaw commands above EMU's unchanged PPO gait.
// Pure browser JS. Never changes the policy, joints, physics or sensor contract.
import {projectOnRun,referenceAt} from './path-follower.js';
import {terrainRiskAt} from './navigation-terrain.js';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const wrap=v=>Math.atan2(Math.sin(v),Math.cos(v));
const difference=(a,b)=>wrap(a-b);
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const zero=()=>({vx:0,vy:0,yaw:0,pitch:0});
const KEYS=['vx','vy','yaw'];
const DT=.20,HORIZON=5,WIDTH=5;
const limitYaw=l=>Math.max(.01,Number(l.yaw_abs)||0);

export function clampNavigationCommand(cmd,limits){
  return {vx:clamp(Number(cmd.vx)||0,Number(limits.vx_min)||0,Number(limits.vx_max)||0),
    vy:clamp(Number(cmd.vy)||0,-Math.max(0,Number(limits.vy_abs)||0),
      Math.max(0,Number(limits.vy_abs)||0)),
    yaw:clamp(Number(cmd.yaw)||0,-limitYaw(limits),limitYaw(limits)),pitch:0};
}

export function vectorFieldGuidance(pose,run,progress,limits){
  const here=projectOnRun(pose,run,progress);
  const s=Math.max(progress,here.s);
  const speed=Math.hypot(pose.vx||0,pose.vy||0);
  const lookahead=clamp(.16+.33*speed,.16,.42);
  const ref=referenceAt(run,s+lookahead);
  const current=referenceAt(run,s);
  const cmd=current.command;
  // The route's tangent follows actual progress, including reverse motion.
  const tx=Math.cos(here.tangent),ty=Math.sin(here.tangent);
  const profileSpeed=Math.hypot(cmd.vx||0,cmd.vy||0);
  const remain=Math.max(0,run.length-s);
  const forwardScale=clamp((remain+.12)/.42,.25,1);
  const motionSpeed=profileSpeed*forwardScale;
  const nX=-ty,nY=tx;
  const cross=here.crossTrack;
  const sideRate=Math.cos(here.tangent)*(
    Math.sin(pose.heading)*(pose.vx||0)+Math.cos(pose.heading)*(pose.vy||0))-
    Math.sin(here.tangent)*(
    Math.cos(pose.heading)*(pose.vx||0)-Math.sin(pose.heading)*(pose.vy||0));
  const correction=clamp(-1.05*cross-.32*sideRate,-.26,.26);
  const wx=tx*motionSpeed+nX*correction;
  const wy=ty*motionSpeed+nY*correction;
  const c=Math.cos(pose.heading),sn=Math.sin(pose.heading);
  const targetHeading=ref.pose?.heading??run.end.heading;
  const error=difference(targetHeading,pose.heading);
  const yaw=clamp((cmd.yaw||0)+1.65*error-.45*(pose.yawRate||0),
    -limitYaw(limits)*.84,limitYaw(limits)*.84);
  return {command:clampNavigationCommand({vx:c*wx+sn*wy,
      vy:-sn*wx+c*wy,yaw},limits),
    near:here,progress:s,headingError:error,
    targetSpeed:motionSpeed,reference:current.command};
}

export class OnlineGaitResponse {
  constructor(){this.reset();}
  reset(){
    this.gain={vx:1,vy:.75,yaw:.82};
    this.tau={vx:.32,vy:.40,yaw:.30};
    this.previous=null;this.issued=null;this.samples=0;
  }
  observe(pose,dt){
    if(!Number.isFinite(pose.x)||!Number.isFinite(pose.y))return;
    if(this.previous&&this.issued&&!pose.fallen&&dt>.035&&dt<.25){
      for(const key of KEYS){
        const prev=key==='yaw'?this.previous.yawRate:this.previous[key];
        const now=key==='yaw'?pose.yawRate:pose[key];
        const u=this.issued[key];
        if(![prev,now,u].every(Number.isFinite)||Math.abs(u)<(key==='yaw'?.11:.075))
          continue;
        const acceleration=(now-prev)/dt;
        if(!Number.isFinite(acceleration)||Math.abs(acceleration)>7)continue;
        const desired=this.gain[key]*u-prev;
        // Only identify lag while the command is actually being followed.
        if(desired*acceleration>.002&&Math.abs(desired)>.06){
          const t=clamp(desired/acceleration,.12,1.20);
          this.tau[key]=.985*this.tau[key]+.015*t;
        }
        // Near-steady observations estimate command tracking gain, not stride
        // oscillations or an accelerative transient.
        if(Math.abs(acceleration)<.32){
          const estimate=clamp(now/u,key==='yaw'?.35:.35,key==='yaw'?1.6:1.5);
          this.gain[key]=.99*this.gain[key]+.01*estimate;
        }
      }
      this.samples++;
    }
    this.previous={...pose};
  }
  rollout(state,command,dt=DT){
    const next={...state};
    for(const key of KEYS){
      const current=key==='yaw'?state.yawRate:state[key];
      const target=this.gain[key]*(command[key]||0);
      const a=1-Math.exp(-dt/this.tau[key]);
      if(key==='yaw')next.yawRate=current+a*(target-current);
      else next[key]=current+a*(target-current);
    }
    const heading=wrap(state.heading+.5*(state.yawRate+next.yawRate)*dt);
    const vx=(state.vx+next.vx)*.5,vy=(state.vy+next.vy)*.5;
    next.x=state.x+(Math.cos(heading)*vx-Math.sin(heading)*vy)*dt;
    next.y=state.y+(Math.sin(heading)*vx+Math.cos(heading)*vy)*dt;
    next.heading=wrap(state.heading+.5*(state.yawRate+next.yawRate)*dt);
    return next;
  }
  report(){return {samples:this.samples,gain:{...this.gain},tau:{...this.tau}};}
}

function collisionCost(pose,trees,terrain){
  if(Math.hypot(pose.x,pose.y)>5.45)return Infinity;
  let total=0;
  for(const tree of trees){
    const d=Math.hypot(pose.x-tree.x,pose.y-tree.y);
    if(d<.62)return Infinity;
    total+=.35*Math.pow(Math.max(0,.96-d),2)/(.34*.34);
  }
  const risk=terrainRiskAt(terrain,pose.x,pose.y);
  if(!Number.isFinite(risk))return Infinity;
  return total+2.5*risk*risk;
}

function commands(guide,legacy,previous,limits){
  const list=[],seen=new Set();
  const add=command=>{
    const c=clampNavigationCommand(command,limits);
    const id=KEYS.map(k=>Math.round(c[k]*100)).join(',');
    if(!seen.has(id)){seen.add(id);list.push(c);}
  };
  add(guide);add(legacy);add(previous);
  add(zero());
  for(const scale of [.65,1.12]){
    add({...guide,vx:guide.vx*scale,vy:guide.vy*scale});
  }
  const yawStep=Math.min(.30,limitYaw(limits)*.27);
  for(const sign of [-1,1]){
    add({...guide,yaw:guide.yaw+sign*yawStep});
    add({...guide,vy:guide.vy+sign*.075});
    add({...guide,vx:guide.vx*.80,yaw:guide.yaw+sign*yawStep});
  }
  return list;
}

function scoreStep(before,after,cmd,previous,run,progress,trees,terrain){
  const obstacle=collisionCost(after,trees,terrain);
  if(!Number.isFinite(obstacle))return {cost:Infinity};
  const midpoint=collisionCost({x:(before.x+after.x)*.5,
    y:(before.y+after.y)*.5},trees,terrain);
  if(!Number.isFinite(midpoint))return {cost:Infinity};
  const proj=projectOnRun(after,run,progress);
  const s=Math.max(progress,proj.s);
  const heading=referenceAt(run,s+.14).pose?.heading??run.end.heading;
  const headingError=difference(heading,after.heading);
  const desired=referenceAt(run,s).command;
  const speedError=Math.hypot(after.vx-(desired.vx||0),after.vy-(desired.vy||0));
  let jerk=0;
  for(const key of KEYS)jerk+=(cmd[key]-previous[key])**2;
  // Path error, future sideways momentum, and progress are more important
  // than exact actuator speed. The NN itself retains balance authority.
  const cost=DT*(6*proj.crossTrack**2+1.1*headingError**2+
    .45*speedError**2+.6*obstacle+.4*midpoint+.18*jerk+
    .06*(cmd.vx**2+cmd.vy**2+.4*cmd.yaw**2));
  return {cost:cost-1.45*Math.max(0,s-progress),
    progress:s,near:proj};
}

export class PredictivePathController {
  constructor(){this.model=new OnlineGaitResponse();this.reset();}
  reset(){this.model.reset();this.previousCommand=zero();this.prediction=[];this.cost=null;this.fallback=false;}
  observe(pose,dt){this.model.observe(pose,dt);}
  issue(command){this.previousCommand={...command};this.model.issued={...command};}
  command(pose,path,runIndex,progress,limits,trees=[],terrain=null,legacy=null){
    const run=path?.runs?.[runIndex];
    if(!run||run.mode==='pivot'||run.length<.001)
      return {command:legacy||zero(),prediction:[],cost:null};
    if(![pose.x,pose.y,pose.heading,pose.vx,pose.vy,pose.yawRate].every(Number.isFinite))
      return {command:zero(),prediction:[],cost:null,fallback:true};
    const base=legacy||zero();
    const init={...pose};
    let beam=[{pose:init,previous:this.previousCommand,
      first:null,points:[],score:0,progress}];
    for(let depth=0;depth<HORIZON;depth++){
      const next=[];
      for(const item of beam){
        const field=vectorFieldGuidance(item.pose,run,item.progress,limits);
        for(const cmd of commands(field.command,base,item.previous,limits)){
          const predicted=this.model.rollout(item.pose,cmd,DT);
          const evaluated=scoreStep(item.pose,predicted,cmd,item.previous,run,
            item.progress,trees,terrain);
          if(!Number.isFinite(evaluated.cost))continue;
          next.push({pose:predicted,previous:cmd,first:item.first||cmd,
            points:[...item.points,{x:predicted.x,y:predicted.y}],
            score:item.score+evaluated.cost,progress:evaluated.progress});
        }
      }
      if(!next.length)break;
      next.sort((a,b)=>a.score-b.score);
      beam=next.slice(0,WIDTH);
    }
    const winner=beam[0]?.first ? beam[0]:null;
    this.fallback=!winner;
    this.prediction=winner?.points||[];
    this.cost=winner?.score??null;
    // If every predicted branch violates clearance/terrain constraints,
    // command a neutral stop and let the normal replan watchdog take over.
    return {command:winner?.first||zero(),prediction:this.prediction,
      cost:this.cost,model:this.model.report(),fallback:this.fallback};
  }
}
