// One navigation mode: pose-lattice A* in a background Worker.
import {makeRng} from './rng.js';
import {sampleGoal,planPoseRoute} from './goal-navigation.js';
const random=makeRng(5017);
self.onmessage=event=>{
  const {id,kind,pose,limits,trees=[],goal}=event.data||{};
  if(!Number.isSafeInteger(id)||!['next','replan'].includes(kind))return;
  try{
    if(kind==='replan'){
      self.postMessage({id,kind,goal,
        route:planPoseRoute(pose,goal,limits,trees,{maxNodes:6500})});
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
