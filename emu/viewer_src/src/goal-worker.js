// One navigation mode: pose-lattice A* in a background Worker.
import {makeRng} from './rng.js';
import {sampleGoal,planPoseRoute,planCircuitLeg} from './goal-navigation.js';
const random=makeRng(5017);
self.onmessage=event=>{
  const {id,kind,pose,limits,trees=[],goal}=event.data||{};
  if(!Number.isSafeInteger(id)||!['next','replan','target'].includes(kind))return;
  try{
    if(kind==='replan'||kind==='target'){
      // Target mode is a caller-selected physical course waypoint; unlike
      // random exploration it must NOT be replaced with a random destination.
      self.postMessage({id,kind,goal,
        route:planCircuitLeg(pose,goal,limits,trees,{maxNodes:6500})});
      return;
    }
    let result=null;
    for(let i=0;i<5;i++){
      const candidate=sampleGoal(random,trees,pose);
      const route=planPoseRoute(pose,candidate,limits,trees,{maxNodes:6500});
      if(route?.solved){result={goal:candidate,route};break;}
    }
    self.postMessage({id,kind,...(result||{goal:null,route:null})});
  }catch(error){
    self.postMessage({id,kind,error:String(error?.message||error)});
  }
};
