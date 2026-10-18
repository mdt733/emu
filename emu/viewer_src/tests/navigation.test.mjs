import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ARENA_RADIUS_M, GOAL_POSITION_TOLERANCE_M, GOAL_HEADING_TOLERANCE_RAD,
  TREE_CLEARANCE_M, wrapAngle, angleError, sampleGoal,
  integratePose, planPoseRoute, followWaypoint, GoalNavigator
} from '../src/goal-navigation.js';
import {makeRng} from '../src/live-demo.js';

const limits={vx_min:-.3,vx_max:.65,vy_abs:.32,yaw_abs:1.8,pitch_abs_rad:Math.PI/12};
const pose={x:0,y:0,heading:0,vx:0,vy:0};

test('exact curved twist integration respects left/right and heading periodicity',()=>{
  let out=integratePose(pose,{vx:.4,vy:0,yaw:1},1);
  assert.ok(out.x>.30 && out.y>.15 && out.heading>0);
  out=integratePose(pose,{vx:0,vy:.25,yaw:0},.8);
  assert.ok(Math.abs(out.y-.2)<1e-10 && out.x===0);
  out=integratePose(pose,{vx:0,vy:0,yaw:-1},1);
  assert.ok(out.heading===-1 && Math.abs(out.x)<1e-12);
  assert.ok(Math.abs(angleError(-Math.PI+.01,Math.PI-.01)-.02)<1e-10);
  assert.ok(Math.abs(wrapAngle(4*Math.PI))<1e-12);
});

test('uniform area goal sampler stays within 4m and keeps clear of tree trunks',()=>{
  const random=makeRng(733);
  const trees=[{x:3.7,y:.1},{x:0,y:3.7},{x:-3.6,y:0}];
  for(let i=0;i<100;i++){
    const goal=sampleGoal(random,trees,pose);
    assert.ok(Math.hypot(goal.x,goal.y)<=ARENA_RADIUS_M);
    assert.ok(Math.hypot(goal.x,goal.y)>.95);
    for(const t of trees)
      assert.ok(Math.hypot(goal.x-t.x,goal.y-t.y)>=TREE_CLEARANCE_M);
    assert.ok(goal.heading>=-Math.PI-1e-9&&goal.heading<=Math.PI+1e-9);
  }
});

test('SE2 planner finds forward, sideways, turning and reverse pose solutions',()=>{
  for(const [goal,tag] of [
    [{x:1.1,y:0,heading:0},'forward'],
    [{x:0,y:.80,heading:0},'sideways'],
    [{x:0,y:0,heading:Math.PI/2},'turn'],
    [{x:-.80,y:0,heading:0},'reverse'],
  ]){
    const result=planPoseRoute(pose,goal,limits,[],{maxNodes:6500});
    assert.ok(result?.solved,tag+' route unavailable');
    assert.ok(result.estimatedSeconds>0,tag);
    assert.ok(result.steps.at(-1).final,tag);
    const path=result.steps;
    for(const p of path)
      assert.ok(Math.hypot(p.x,p.y)<5.6,tag+' left the arena');
  }
});

test('planner refuses tree collision and invalid target; never labels incomplete search solved',()=>{
  assert.equal(planPoseRoute(pose,{x:.5,y:0,heading:0},limits,
    [{x:.5,y:0}]),null);
  assert.throws(()=>planPoseRoute(pose,{x:NaN,y:0,heading:0},limits),/Nonfinite/);
  assert.throws(()=>planPoseRoute(pose,{x:1,y:0,heading:0},
    {...limits,vx_max:0}),/too small/);
});

test('navigator never commands beyond conservative policy limits or simultaneous full forward/strafe',()=>{
  const waypoint={x:1,y:1,heading:2,cmd:{vx:99,vy:99,yaw:99}};
  const c=followWaypoint(pose,waypoint,limits);
  assert.ok(c.vx<=limits.vx_max && c.vx>=limits.vx_min);
  assert.ok(Math.abs(c.vy)<=limits.vy_abs);
  assert.ok(Math.abs(c.yaw)<=limits.yaw_abs*.73+1e-8);
  assert.ok(Math.hypot(c.vx/(limits.vx_max*.78),c.vy/(limits.vy_abs*.8))<=1+1e-8);
  const final=followWaypoint(pose,{...waypoint,x:.05,y:0,heading:0},limits,{final:true});
  assert.ok(final.vx<c.vx && final.vx>=0);
});

test('closed loop driver reaches and parks a final pose in simplified kinematics',()=>{
  const nav=new GoalNavigator(800);nav.active=true;
  const goal={x:.9,y:0,heading:0};
  assert.ok(nav.setGoal(pose,goal,limits,[]));
  let simulated={...pose},arrived=false;
  for(let i=0;i<1400;i++){
    let command;
    nav.tick(.02,simulated,limits,[],c=>{command=c;});
    assert.ok(command);
    const next=integratePose(simulated,command,.02);
    simulated={...next,vx:command.vx,vy:command.vy};
    if(nav.arrivals>=1){arrived=true;break;}
  }
  assert.ok(arrived,'navigator did not park the robot and finish a goal');
  assert.ok(nav.elapsed>=0);
});

test('UI keeps manual and random-control modes alongside pose missions', async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const read=p=>readFileSync(fileURLToPath(new URL(p,import.meta.url)),'utf8');
  const html=read('../index.html'), main=read('../src/main.js'), sim=read('../src/sim.js');
  assert.match(html,/value="goals"/);
  assert.match(html,/value="commands" selected/);
  assert.match(html,/value="commands"/);
  assert.match(html,/id="navNext"/);
  assert.match(main,/navigator\.tick\(/);
  assert.match(main,/if\(goalDemoActive\(\)\) navigationUpdate\(dt\)/);
  assert.match(sim,/this\.navigationControl/);
  assert.match(sim,/this\.headingTarget=this\.actualHeading\(\)/);
  assert.match(sim,/this\.routeLine/);
  assert.match(sim,/this\.resetGeneration=/);
});


test('several longer target poses are planned inside the 4m field, with no tree entry',()=>{
  const start={x:0,y:0,heading:0};
  const goals=[{x:2.6,y:0,heading:0},{x:1.5,y:2.0,heading:Math.PI/2},
               {x:-2.0,y:-1.0,heading:-Math.PI/2}];
  for(const goal of goals){
    const route=planPoseRoute(start,goal,limits,[],{maxNodes:7000});
    assert.ok(route?.solved, 'goal not planned '+JSON.stringify(goal));
    assert.ok(route.estimatedSeconds>0);
    assert.ok(route.steps.at(-1).final);
  }
  const tree={x:.8,y:0};
  const goal={x:1.8,y:0,heading:0};
  const route=planPoseRoute(start,goal,limits,[tree],{maxNodes:7000});
  if(route){
    for(const step of route.steps)
      assert.ok(Math.hypot(step.x-tree.x,step.y-tree.y)>=TREE_CLEARANCE_M,
        'route entered collision keep-out');
  }
});

test('local .emupolicy launch exposes autonomous controls without a live server', async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const main=readFileSync(fileURLToPath(new URL('../src/main.js',import.meta.url)),'utf8');
  assert.match(main,/if \(livePanel\.hidden && !showcaseManifest\)/);
  assert.match(main,/demoActive = \$\('liveDemo'\)\.checked/);
  assert.match(main,/\$\('livePhase'\)\.textContent = 'Local policy'/);
});


test('browser planner uses a true message worker and returns asynchronously',async()=>{
  const {GoalPlannerWorker}=await import('../src/goal-planner-client.js');
  let fake;
  const broker=new GoalPlannerWorker({timeoutMs:2000,makeWorker:()=>{
    fake={posted:[],postMessage(x){this.posted.push(x);},terminate(){this.ended=true;}};
    return fake;
  }});
  const answer=broker.request({kind:'next',pose,limits,trees:[]});
  assert.equal(fake.posted.length,1);
  assert.equal(fake.posted[0].kind,'next');
  assert.equal(fake.posted[0].pose.x,0);
  let completed=false;answer.then(()=>{completed=true;});
  await Promise.resolve();
  assert.equal(completed,false,'search must not execute synchronously on render thread');
  fake.onmessage({data:{id:fake.posted[0].id,kind:'next',
    goal:{x:1,y:0,heading:0},route:{solved:true,steps:[],estimatedSeconds:2}}});
  const result=await answer;
  assert.equal(result.goal.x,1);
  assert.equal(completed,true);
  broker.dispose();
  assert.equal(fake.ended,true);
});

test('while route search is pending, every frame still commands the neural walker',async()=>{
  const nav=new GoalNavigator(92);nav.active=true;
  let resolvePlan,called=0;
  nav.useBackgroundPlanner(req=>{
    called++;
    assert.equal(req.kind,'next');
    return new Promise(resolve=>{resolvePlan=resolve;});
  });
  const commands=[];
  // Advance past 1.2 second dynamic warmup.
  for(let i=0;i<65;i++)nav.tick(.02,pose,limits,[],x=>commands.push(x));
  assert.equal(called,1);
  assert.ok(nav.pendingRequest);
  assert.equal(commands.length,65);
  assert.ok(commands.every(c=>c.vx>.08));
  for(let i=0;i<200;i++)nav.tick(.02,pose,limits,[],x=>commands.push(x));
  assert.equal(called,1,'only one in-flight worker plan');
  assert.equal(commands.length,265,'cannot freeze or skip frames during search');
  resolvePlan({goal:{x:1,y:0,heading:0},route:{
    solved:true,steps:[{x:1,y:0,heading:0,final:true,
      maneuver:'forward',cmd:{vx:.1,vy:0,yaw:0}}],
    estimatedSeconds:3,expanded:80}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(nav.goal && nav.route?.solved);
  const count=commands.length;
  nav.tick(.02,pose,limits,[],x=>commands.push(x));
  assert.equal(commands.length,count+1);
  assert.ok(Number.isFinite(commands.at(-1).vx));
});

test('navigation reset prevents stale worker result from changing a new mission',async()=>{
  const nav=new GoalNavigator(92);nav.active=true;
  let reply;
  nav.useBackgroundPlanner(()=>new Promise(resolve=>reply=resolve));
  nav.nextGoal(pose,limits,[]);
  assert.ok(nav.pendingRequest);
  nav.reset();
  reply({goal:{x:2,y:2,heading:1},route:{solved:true,steps:[],estimatedSeconds:10}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(nav.goal,null);
  assert.equal(nav.route,null);
  assert.equal(nav.pendingRequest,null);
});

test('a goal navigation exception cannot permanently stop MuJoCo animation frames',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const main=readFileSync(fileURLToPath(new URL('../src/main.js',import.meta.url)),'utf8');
  const sim=readFileSync(fileURLToPath(new URL('../src/sim.js',import.meta.url)),'utf8');
  const nav=readFileSync(fileURLToPath(new URL('../src/goal-navigation.js',import.meta.url)),'utf8');
  const worker=readFileSync(fileURLToPath(new URL('../src/goal-worker.js',import.meta.url)),'utf8');
  const frame=main.slice(main.indexOf('function frame(ms)'),main.indexOf('function stopDemo()'));
  assert.ok(frame.indexOf('requestAnimationFrame(frame)')<frame.indexOf('navigationUpdate(dt)'));
  assert.match(frame,/catch\(error\)\{[\s\S]*demoMode.value='commands'/);
  assert.match(main,/navigator.useBackgroundPlanner/);
  assert.match(main,/new GoalPlannerWorker\(\)/);
  assert.match(sim,/this\.controlFrames\+\+/);
  assert.match(nav,/if\(this\.pendingRequest\)/);
  assert.match(worker,/self\.postMessage\(/);
});
