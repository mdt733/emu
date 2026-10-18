// Dedicated background worker: route search NEVER blocks MuJoCo, neural
// inference or requestAnimationFrame on the browser main thread.
import {makeRng} from './live-demo.js';
import {sampleGoal, planPoseRoute} from './goal-navigation.js';
import {planCurvatureRoute} from './curvature-planner.js';
import {bidirectionalMotionRoute} from './bidirectional-lattice.js';

const routeByMode=(pose,goal,limits,trees,planner)=>{
  switch(planner){
    case 'curves': return planCurvatureRoute(pose,goal,limits,trees);
    case 'arcs': return bidirectionalMotionRoute(pose,goal,limits,trees,
      {maxNodes:4200});
    case 'grid': return planPoseRoute(pose,goal,limits,trees,
      {maxNodes:6500});
    default: throw new Error('Unknown navigation planner mode: '+planner);
  }
};

const random=makeRng(5017);
self.onmessage=(event)=>{
  const {id,kind,pose,limits,trees=[],goal,planner='grid'}=event.data||{};
  if(!Number.isSafeInteger(id)||!['next','replan'].includes(kind))return;
  try {
    if(kind==='replan'){
      const route=routeByMode(pose,goal,limits,trees,planner);
      self.postMessage({id,kind,goal,route});
      return;
    }
    let result=null;
    // All five expensive A* attempts occur inside this Worker, never in the
    // simulator animation callback. Unsatisfiable goals may return null.
    for(let i=0;i<5;i++){
      const candidate=sampleGoal(random,trees,pose);
      const route=routeByMode(pose,candidate,limits,trees,planner);
      if(route?.solved){result={goal:candidate,route};break;}
    }
    self.postMessage({id,kind,...(result||{goal:null,route:null})});
  }catch(error){
    self.postMessage({id,kind,error:String(error?.message||error)});
  }
};
