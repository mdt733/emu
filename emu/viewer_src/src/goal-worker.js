// Background pose-lattice planning. Normal exhibition selects a current goal
// AND the following goal so the actor can walk through markers instead of
// repeatedly stopping, facing an unrelated random bearing, and restarting.
// Benchmark/replan requests retain their stricter exact-pose semantics.
import {makeRng} from './rng.js';
import {sampleGoal,planPoseRoute} from './goal-navigation.js';
const random=makeRng(5017);
const bearing=(a,b)=>Math.atan2(b.y-a.y,b.x-a.x);
const distance=(a,b)=>Math.hypot(b.x-a.x,b.y-a.y);
function nearby(from,trees,terrain){
  let fallback=null;
  for(let i=0;i<18;i++){
    const point=sampleGoal(random,trees,from,undefined,terrain);
    fallback=point;
    const d=distance(from,point);
    // Moderately spaced points avoid repeated pivots and extremely long
    // cross-park trips. A* still decides whether the segment is navigable.
    if(d>1.25&&d<3.35)return point;
  }
  return fallback;
}
function planNext(pose,limits,trees,terrain){
  let fallback=null;
  for(let i=0;i<7;i++){
    const first=nearby(pose,trees,terrain);
    const second=nearby(first,trees,terrain);
    const third=nearby(second,trees,terrain);
    first.heading=bearing(first,second);
    second.heading=bearing(second,third);
    const route=planPoseRoute(pose,first,limits,trees,{
      maxNodes:6500,terrain,passThrough:true});
    if(!route?.solved||!route.flythrough)continue;
    const followup=planPoseRoute(first,second,limits,trees,{
      maxNodes:6500,terrain,passThrough:true});
    if(followup?.solved&&followup.flythrough)
      return {goal:first,route,following:{goal:second,route:followup,hint:third}};
    fallback={goal:first,route,following:null};
  }
  return fallback||{goal:null,route:null,following:null};
}
function prefetch(pose,goalHint,limits,trees,terrain){
  for(let i=0;i<7;i++){
    const goal=i===0&&goalHint?{...goalHint}:nearby(pose,trees,terrain);
    const hint=nearby(goal,trees,terrain);
    goal.heading=bearing(goal,hint);
    const route=planPoseRoute(pose,goal,limits,trees,{
      maxNodes:6500,terrain,passThrough:true});
    if(route?.solved&&route.flythrough)return {following:{goal,route,hint}};
  }
  return {following:null};
}
self.onmessage=event=>{
  const {id,kind,pose,limits,trees=[],goal,terrain=null,hint=null}=event.data||{};
  if(!Number.isSafeInteger(id)||
     !['next','replan','target','prefetch'].includes(kind))return;
  try{
    if(kind==='replan'||kind==='target'){
      self.postMessage({id,kind,goal,
        route:planPoseRoute(pose,goal,limits,trees,{maxNodes:6500,terrain,
          passThrough:kind==='replan'&&Boolean(event.data?.passThrough)})});
      return;
    }
    if(kind==='prefetch'){
      self.postMessage({id,kind,...prefetch(pose,hint,limits,trees,terrain)});
      return;
    }
    self.postMessage({id,kind,...planNext(pose,limits,trees,terrain)});
  }catch(error){
    self.postMessage({id,kind,error:String(error?.message||error)});
  }
};
