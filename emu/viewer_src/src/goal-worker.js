// Dedicated background worker: route search NEVER blocks MuJoCo, neural
// inference or requestAnimationFrame on the browser main thread.
import {makeRng} from './live-demo.js';
import {sampleGoal, planPoseRoute} from './goal-navigation.js';

const random=makeRng(5017);
self.onmessage=(event)=>{
  const {id,kind,pose,limits,trees=[],goal}=event.data||{};
  if(!Number.isSafeInteger(id)||!['next','replan'].includes(kind))return;
  try {
    if(kind==='replan'){
      const route=planPoseRoute(pose,goal,limits,trees,{maxNodes:6500});
      self.postMessage({id,kind,goal,route});
      return;
    }
    let result=null;
    // All five expensive A* attempts occur inside this Worker, never in the
    // simulator animation callback. Unsatisfiable goals may return null.
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
