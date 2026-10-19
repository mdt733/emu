// Curvature-constrained planar pose planner for a dynamic walking biped.
// Inspired by Dubins CSC/CCC motion families and Nav2's kinematically
// feasible planner. No grid expansion or stationary spins. Candidate
// routes are scored by estimated time, NOT geometric length alone.
// The terminal pose includes desired heading. Trajectories use true arc
// integration; the trained actor only receives vx and yaw via the
// existing working viewer's setDemoCommands() pipeline.
const TAU=2*Math.PI, PI=Math.PI, CLEARANCE=.53, BOUND=5.6;
const mod=x=>(x%TAU+TAU)%TAU;
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const ok=p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.heading);
const safe=p=>Math.hypot(p.x,p.y)<BOUND;
const collision=(p,trees)=>trees.some(t=>Math.hypot(p.x-t.x,p.y-t.y)<CLEARANCE);
const sqrt=x=>x>=-1e-9?Math.sqrt(Math.max(0,x)):null;
const acos=x=>Math.abs(x)<=1+1e-9?Math.acos(clamp(x,-1,1)):null;
const move=(p,vx,yaw,dt)=>{
  const a=yaw*dt,dir=p.heading;
  const forward=Math.abs(yaw)<1e-9?vx*dt:vx*Math.sin(a)/yaw;
  const sideways=Math.abs(yaw)<1e-9?0:vx*(1-Math.cos(a))/yaw;
  return {x:p.x+Math.cos(dir)*forward-Math.sin(dir)*sideways,
          y:p.y+Math.sin(dir)*forward+Math.cos(dir)*sideways,
          heading:wrap(dir+a)};
};
function lengths(type,a,b,d){
  const sa=Math.sin(a),sb=Math.sin(b),ca=Math.cos(a),cb=Math.cos(b);
  const cab=Math.cos(a-b);
  let t,p,q,z,u;
  switch(type){
    case 'LSL':
      p=sqrt(2+d*d-2*cab+2*d*(sa-sb));if(p===null)return null;
      z=Math.atan2(cb-ca,d+sa-sb);
      return [mod(-a+z),p,mod(b-z)];
    case 'RSR':
      p=sqrt(2+d*d-2*cab+2*d*(sb-sa));if(p===null)return null;
      z=Math.atan2(ca-cb,d-sa+sb);
      return [mod(a-z),p,mod(-b+z)];
    case 'LSR':
      p=sqrt(-2+d*d+2*cab+2*d*(sa+sb));if(p===null)return null;
      z=Math.atan2(-ca-cb,d+sa+sb)-Math.atan2(-2,p);
      return [mod(-a+z),p,mod(-b+z)];
    case 'RSL':
      p=sqrt(-2+d*d+2*cab-2*d*(sa+sb));if(p===null)return null;
      z=Math.atan2(ca+cb,d-sa-sb)-Math.atan2(2,p);
      return [mod(a-z),p,mod(b-z)];
    case 'RLR':
      u=acos((6-d*d+2*cab+2*d*(sa-sb))/8);if(u===null)return null;
      p=mod(TAU-u);
      t=mod(a-Math.atan2(ca-cb,d-sa+sb)+p/2);
      q=mod(a-b-t+p);
      return [t,p,q];
    case 'LRL':
      u=acos((6-d*d+2*cab+2*d*(-sa+sb))/8);if(u===null)return null;
      p=mod(TAU-u);
      t=mod(-a-Math.atan2(ca-cb,d+sa-sb)+p/2);
      q=mod(b-a-t+p);
      return [t,p,q];
    default:throw new Error('Unknown Dubins type');
  }
}
const TYPES=['LSL','RSR','LSR','RSL','RLR','LRL'];
function generate(start,goal,limits,trees,radius,type,dims,reverse){
  const maxF=Math.min(Number(limits.vx_max)*.90,.95);
  const maxB=Math.min(Math.abs(Number(limits.vx_min))*.83,.48);
  const maxW=Math.min(Number(limits.yaw_abs)*.83,1.20);
  const cruise=reverse?maxB:maxF;
  if(!(cruise>.06&&maxW>.15))return null;
  const steps=[],segments=[];let p={...start},cost=reverse?.48:0,linear=0;
  let previousTurn=null;
  for(let j=0;j<3;j++){
    const mode=type[j],segLength=dims[j]*radius;
    if(segLength<1e-7)continue;
    const turn=mode==='L'?1:mode==='R'?-1:0;
    // Curves can be run quickly if their radius supports the yaw bound.
    const v=turn?Math.min(cruise,maxW*radius):cruise;
    const yaw=turn*v/radius;
    const vx=reverse?-v:v;
    const dt=segLength/v;
    const sub=Math.max(1,Math.ceil(segLength/.105));
    segments.push({mode,meters:segLength,seconds:dt,radius:turn?radius:Infinity,
                   vx,yaw});
    if(previousTurn!==null&&turn!==0&&previousTurn!==turn)cost+=.20;
    if(turn!==0)previousTurn=turn;
    cost+=dt*(turn?1.035:1);
    linear+=segLength;
    for(let i=0;i<sub;i++){
      p=move(p,vx,yaw,dt/sub);
      if(!safe(p)||collision(p,trees))return null;
      steps.push({...p,cmd:{vx,vy:0,yaw,pitch:0},
        curvature:turn/radius,dt:dt/sub,
        maneuver:reverse?'reverse-'+mode:mode});
    }
  }
  if(dist(p,goal)>.025||Math.abs(wrap(p.heading-goal.heading))>.03)return null;
  if(!steps.length)return null;
  // Use mathematically exact final pose; preserve the preceding segment's
  // curvature command instead of inserting a spurious stationary waypoint.
  steps[steps.length-1]={...steps.at(-1),x:goal.x,y:goal.y,heading:goal.heading,final:true};
  return {start:{...start},steps,segments,meters:linear,estimatedSeconds:cost,
          solved:true,expanded:0,planner:'continuous-curves',
          family:(reverse?'backwards ':'forward ')+type+' R='+radius.toFixed(2),
          modes:segments.map(x=>(reverse?'reverse ':'')+x.mode)};
}
function directCandidates(start,goal,limits,trees,radii,reverseAllowed=true){
  const dx=goal.x-start.x,dy=goal.y-start.y;
  const solutions=[];
  const directions=reverseAllowed?[false,true]:[false];
  for(const reverse of directions){
    const sh=wrap(start.heading+(reverse?PI:0));
    const gh=wrap(goal.heading+(reverse?PI:0));
    const theta=Math.atan2(dy,dx);
    const alpha=mod(sh-theta),beta=mod(gh-theta);
    for(const radius of radii){
      const d=Math.hypot(dx,dy)/radius;
      for(const type of TYPES){
        const values=lengths(type,alpha,beta,d);
        if(!values)continue;
        const trajectory=generate(start,goal,limits,trees,radius,type,values,reverse);
        if(trajectory)solutions.push(trajectory);
      }
    }
  }
  solutions.sort((a,b)=>a.estimatedSeconds-b.estimatedSeconds);
  return solutions;
}
// Only use a two-part, continuously curved detour when all direct
// forward/reverse CSC/CCC trajectories collide with static obstacles.
function detourCandidates(start,goal,limits,trees,radii){
  const targetTrees=trees.map(t=>{
    const dx=goal.x-start.x,dy=goal.y-start.y;
    const denom=dx*dx+dy*dy;
    const k=denom?clamp(((t.x-start.x)*dx+(t.y-start.y)*dy)/denom,0,1):0;
    const q={x:start.x+k*dx,y:start.y+k*dy};
    return {tree:t,gap:dist(t,q)};
  }).filter(x=>x.gap<1.7).sort((a,b)=>a.gap-b.gap).slice(0,3);
  let winner=null,examined=0;
  for(const {tree} of targetTrees){
    for(const outRadius of [.92,1.28]){
      for(let j=0;j<12;j++){
        const angle=TAU*j/12;
        const x=tree.x+outRadius*Math.cos(angle);
        const y=tree.y+outRadius*Math.sin(angle);
        for(const tangent of [-1,1]){
          const pivot={x,y,heading:wrap(angle+tangent*PI/2)};
          if(!safe(pivot)||collision(pivot,trees))continue;
          const first=directCandidates(start,pivot,limits,trees,radii,false).slice(0,2);
          if(!first.length)continue;
          const last=directCandidates(pivot,goal,limits,trees,radii,false).slice(0,2);
          examined++;
          for(const a of first)for(const b of last){
            const cost=a.estimatedSeconds+b.estimatedSeconds+.32;
            if(!winner||cost<winner.estimatedSeconds){
              const steps=[...a.steps,...b.steps];
              steps[steps.length-1]={...steps.at(-1),final:true};
              winner={start:{...start},steps,segments:[...a.segments,...b.segments],
                meters:a.meters+b.meters,estimatedSeconds:cost,
                family:'detour '+a.family+' + '+b.family,
                planner:'continuous-curves',solved:true,expanded:examined,
                modes:[...a.modes,...b.modes]};
            }
          }
        }
      }
    }
  }
  return winner;
}
export function planCurvatureRoute(start,goal,limits,trees=[],options={}){
  if(!ok(start)||!ok(goal))throw new Error('Nonfinite curvature start/goal');
  if(!(limits?.vx_max>.03&&limits?.yaw_abs>.1))
    throw new Error('Policy command envelope too small for curvature planning');
  if(!safe(goal)||collision(goal,trees))return null;
  if(dist(start,goal)<.02&&Math.abs(wrap(goal.heading-start.heading))<.04)
    return {start:{...start},steps:[{...goal,cmd:{vx:0,vy:0,yaw:0,pitch:0},final:true}],
      segments:[],meters:0,estimatedSeconds:0,solved:true,
      expanded:0,planner:'continuous-curves',family:'already at goal',modes:[]};
  // Different radii trade travel length against cornering speed.
  const radii=options.radii||[.52,.70,.95,1.25,1.75];
  const hits=directCandidates(start,goal,limits,trees,radii,options.reverse!==false);
  if(hits.length)return hits[0];
  return options.noDetour?null:detourCandidates(start,goal,limits,trees,[.70,1.15]);
}
