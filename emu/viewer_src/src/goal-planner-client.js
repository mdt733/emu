// Background A* request broker. Never execute route search in the animation
// loop; errors/timeouts leave MuJoCo control frames uninterrupted.
export class GoalPlannerWorker {
  constructor({timeoutMs=16000,makeWorker}={}){
    this.timeoutMs=timeoutMs;
    this.makeWorker=makeWorker||(()=>new Worker(
      new URL('./goal-worker.js',import.meta.url),{type:'module'}));
    this.nextId=0;this.pending=new Map();
    this.disposed=false;this.open();
  }
  open(){
    this.worker=this.makeWorker();
    this.worker.onmessage=event=>{
      const data=event.data||{};
      const record=this.pending.get(data.id);
      if(!record)return;
      clearTimeout(record.timer);
      this.pending.delete(data.id);
      if(data.error)record.reject(new Error('Goal planner: '+data.error));
      else record.resolve(data);
    };
    this.worker.onerror=event=>{
      const message=event?.message||'Goal planner worker failed';
      this.restart(new Error(message));
    };
  }
  restart(error){
    if(this.disposed)return;
    for(const record of this.pending.values()){
      clearTimeout(record.timer);record.reject(error);
    }
    this.pending.clear();
    this.worker?.terminate();
    this.open();
  }
  request({kind,pose,limits,trees,goal}){
    if(this.disposed)return Promise.reject(new Error('Goal planner disposed'));
    if(!['next','replan','target'].includes(kind))
      return Promise.reject(new Error('Unknown route request'));
    const id=++this.nextId;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{
        if(!this.pending.has(id))return;
        this.restart(new Error('Goal planning exceeded '+this.timeoutMs+' ms'));
      },this.timeoutMs);
      this.pending.set(id,{resolve,reject,timer});
      try{this.worker.postMessage({id,kind,pose,limits,trees,goal});}
      catch(error){this.restart(error);}
    });
  }
  dispose(){
    this.disposed=true;
    for(const r of this.pending.values()){
      clearTimeout(r.timer);r.reject(new Error('Goal planner disposed'));
    }
    this.pending.clear();
    this.worker?.terminate();
  }
}
