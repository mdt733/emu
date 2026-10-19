import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ARENA_RADIUS_M, GOAL_POSITION_TOLERANCE_M, GOAL_HEADING_TOLERANCE_RAD,
  TREE_CLEARANCE_M, wrapAngle, angleError, sampleGoal,
  integratePose, planPoseRoute, followWaypoint, previewRoute, GoalNavigator
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
  assert.match(main,/if\(goalDemoActive\(\)\)\s*\{\s*try\s*\{\s*navigationUpdate\(dt\)/);
  assert.match(sim,/this\.navPresentationActive/);
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
  // The walker must remain in place while an asynchronous route is pending.
  for(let i=0;i<65;i++)nav.tick(.02,pose,limits,[],x=>commands.push(x));
  assert.equal(called,1);
  assert.ok(nav.pendingRequest);
  assert.equal(commands.length,65);
  assert.ok(commands.every(c=>c.vx===0&&c.vy===0&&c.yaw===0));
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


// Regression for the actual user-observed bug: both mission styles MUST enter
// exactly the same working actor input path, not a private navigation yaw path.
test('navigation and random commands use identical setDemoCommands and policy control',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const read=x=>readFileSync(fileURLToPath(new URL(x,import.meta.url)),'utf8');
  const main=read('../src/main.js');
  const sim=read('../src/sim.js');
  const controls=read('../src/controls.js');
  assert.match(main,/function applyAutonomousCommand\(command\)\s*\{\s*sim\.setDemoCommands\(command\);/);
  assert.match(main,/sim\.navigationTrees\(\),command=>\{plannedCommand=command;\}\)/);
  assert.match(main,/applyAutonomousCommand\(plannedCommand\)/);
  assert.match(main,/demo\.tick\(dt, limits\(\), applyAutonomousCommand/);
  assert.match(sim,/setNavigationPresentation\(active\)/);
  assert.doesNotMatch(sim,/setNavigationCommands\(/);
  assert.doesNotMatch(sim,/this\.navigationControl/);
  const control=sim.slice(sim.indexOf('  stepControl() {'),sim.indexOf('  flattenHistory() {'));
  assert.match(control,/nextYawCommand\(/);
  assert.match(control,/this\.policy\.infer\(this\.obs\)/);
  assert.doesNotMatch(control,/if\(this\.navPresentationActive\)/);
  assert.match(controls,/function nextYawCommand/);
  assert.doesNotMatch(main,/if\(sim\)sim\.paused=true/);
});


test('goal mission coalesces fast feedback into stable held velocity setpoints',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const src=readFileSync(fileURLToPath(new URL('../src/main.js',import.meta.url)),'utf8');
  const block=src.slice(src.indexOf('function navigationUpdate(dt)'),
                        src.indexOf('let liveRevision = null;'));
  assert.match(block,/navigator\.tick\(dt,pose,bundle\.control\.command_limits/);
  assert.match(block,/command=>\{plannedCommand=command;\}/);
  assert.match(block,/commandPeriod=\.10/);
  assert.match(block,/navCommandClock>=commandPeriod/);
  assert.match(block,/applyAutonomousCommand\(plannedCommand\)/);
  assert.match(src,/function applyAutonomousCommand\(command\)\s*\{\s*sim\.setDemoCommands\(command\)/);
  assert.doesNotMatch(block,/sim\.setNavigationCommands\(/);
});


test('actual actuator travel is tracked after each MuJoCo control step',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const sim=readFileSync(fileURLToPath(new URL('../src/sim.js',import.meta.url)),'utf8');
  const main=readFileSync(fileURLToPath(new URL('../src/main.js',import.meta.url)),'utf8');
  assert.match(sim,/this\.motorTravelRad\+=Math\.abs\(Number\(this\.data\.qpos/);
  assert.match(main,/sim\.motorTravelRad\.toFixed\(3\)/);
});


test('260M trained-speed corridor cruises fast without sideways tap-dancing',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const policy=JSON.parse(readFileSync(fileURLToPath(
    new URL('../../../showcase/champion.emupolicy',import.meta.url)),'utf8'));
  const trained=policy.control.command_limits;
  assert.ok(trained.vx_max>1.0,'confirm actual Pinnacle velocity envelope');
  const route={steps:[
    {x:.30,y:0,heading:.1,cmd:{vx:.7,vy:0,yaw:0}},
    {x:.65,y:0,heading:-.20,cmd:{vx:.7,vy:0,yaw:0}},
    {x:1.00,y:0,heading:.3,cmd:{vx:.7,vy:0,yaw:0}},
    {x:1.35,y:0,heading:-.5,cmd:{vx:.7,vy:0,yaw:0}},
    {x:1.70,y:0,heading:.2,cmd:{vx:.7,vy:0,yaw:0}},
    {x:2.25,y:0,heading:-.1,cmd:{vx:.7,vy:0,yaw:0}},
    {x:3.60,y:0,heading:0,cmd:{vx:0,vy:0,yaw:0},final:true}
  ]};
  const preview=previewRoute(pose,route.steps,0,[],.65);
  assert.ok(preview.target.x>1.1&&preview.target.x<1.7,
    'do not chase each tiny 22cm planner step');
  const fast=followWaypoint(pose,preview.target,trained,{
    goal:{x:3.6,y:0,heading:0}});
  assert.ok(fast.vx>.75,
    '1m/s-trained actor should cruise at >.75 m/s command, got '+fast.vx);
  assert.equal(fast.vy,0,'no lateral strafe for intermediate lattice nodes');
  assert.ok(Math.abs(fast.yaw)<.01);
});

test('turn at speed instead of strafing to a far sideways lattice point',()=>{
  const heading=0,start={x:0,y:0,heading};
  const goal={x:2.8,y:2.4,heading:1.3};
  const p=followWaypoint(start,{x:1.2,y:.85,heading:-2.5,
    cmd:{vx:.7,vy:.15,yaw:.5}},limits,{goal});
  assert.ok(p.vx>.20,'biped must keep stepping while making its turn');
  assert.equal(p.vy,0,'long-distance approach should not use sideways trim');
  assert.ok(p.yaw>.25,'heading towards route tangent should steer left');
  assert.ok(Math.abs(p.yaw)<=limits.yaw_abs*.82+.001);
});

test('only approach phase uses lateral trim and final heading; speeds stay in policy envelope',()=>{
  const target={x:.27,y:.18,heading:Math.PI/2,
                cmd:{vx:0,vy:0,yaw:0},final:true};
  const c=followWaypoint(pose,target,limits,{goal:target,final:true});
  assert.ok(c.vx<limits.vx_max*.3,'slow for the actual destination');
  assert.ok(c.vy>0,'lateral trim allowed in last ~60cm');
  assert.ok(c.yaw>0,'final heading matters once near destination');
  assert.ok(c.vx>=limits.vx_min&&c.vx<=limits.vx_max);
  assert.ok(Math.abs(c.vy)<=limits.vy_abs);
  assert.ok(Math.abs(c.yaw)<=limits.yaw_abs);
});

test('path preview advances past old nodes without requiring each node heading',()=>{
  const steps=[
    {x:.25,y:0,heading:Math.PI,cmd:{vx:.7,vy:0,yaw:0}},
    {x:.50,y:0,heading:-Math.PI/2,cmd:{vx:.7,vy:0,yaw:0}},
    {x:.85,y:0,heading:Math.PI/2,cmd:{vx:.7,vy:0,yaw:0}},
    {x:1.25,y:0,heading:0,cmd:{vx:.7,vy:0,yaw:0}},
    {x:2.25,y:0,heading:0,cmd:{vx:0,vy:0,yaw:0},final:true},
  ];
  const p=previewRoute({x:.68,y:.035,heading:0},steps,0,[],.6);
  assert.ok(p.progress>=1,'robot has already physically passed first lattice node');
  assert.ok(p.target.x>1.3,'should pursue distant route, not do local heading correction');
  const e=followWaypoint({x:.68,y:.035,heading:0},p.target,limits,
    {goal:{x:2.25,y:0,heading:0}});
  assert.ok(e.vx>.30 && Math.abs(e.vy)<1e-12);
});

test('preview cannot look through a tree trunk',()=>{
  const steps=[
    {x:.40,y:0,heading:0,cmd:{vx:.4,vy:0,yaw:0}},
    {x:.65,y:.8,heading:1,cmd:{vx:.4,vy:0,yaw:0}},
    {x:1.1,y:1.3,heading:0,cmd:{vx:.4,vy:0,yaw:0}},
    {x:2,y:1.3,heading:0,cmd:{vx:0,vy:0,yaw:0},final:true}
  ];
  const p=previewRoute(pose,steps,0,[{x:1,y:0}],.8);
  assert.ok(p.target.x<1.1||p.target.y>.2,
    'shortcut directly through a tree should not be selected');
});


test('all three navigation modes route through the same Worker and actor pipeline',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const read=p=>readFileSync(fileURLToPath(new URL(p,import.meta.url)),'utf8');
  const html=read('../index.html');
  const source=read('../src/main.js');
  const worker=read('../src/goal-worker.js');
  const client=read('../src/goal-planner-client.js');
  const nav=read('../src/goal-navigation.js');
  assert.match(html,/id="navPlanner"/);
  for(const id of ['grid','curves','arcs']){
    assert.match(html,new RegExp('value="'+id+'"'));
    assert.match(worker,new RegExp("case '"+id+"'"));
  }
  assert.match(source,/navigator\.plannerMode=navPlanner\.value/);
  assert.match(source,/navPlanner\.addEventListener\('change'/);
  assert.match(client,/this\.worker\.postMessage\(\{id,kind,pose,limits,trees,goal,planner\}\)/);
  assert.match(nav,/planner:this\.plannerMode/);
  assert.match(worker,/planCurvatureRoute\(/);
  assert.match(worker,/bidirectionalMotionRoute\(/);
  assert.match(source,/applyAutonomousCommand\(plannedCommand\)/);
  assert.doesNotMatch(source,/sim\.setNavigationCommands/);
});

test('planner request broker transmits selected arc mode with unchanged commands',async()=>{
  const {GoalPlannerWorker}=await import('../src/goal-planner-client.js');
  let fake;
  const broker=new GoalPlannerWorker({makeWorker:()=>{
    fake={postMessage(x){this.request=x;},terminate(){}};
    return fake;
  }});
  const next=broker.request({kind:'next',pose,limits,trees:[],planner:'arcs'});
  assert.equal(fake.request.planner,'arcs');
  assert.equal(fake.request.kind,'next');
  fake.onmessage({data:{id:fake.request.id,kind:'next',goal:null,route:null}});
  const value=await next;
  assert.equal(value.route,null);
  broker.dispose();
});


test('path-relative follower corrects the line with progressive slowdown',async()=>{
  const {buildRoutePath,trackRoutePath,projectOnRun}=
    await import('../src/path-follower.js');
  const cmd={vx:.7,vy:0,yaw:0};
  const route={start:{x:0,y:0,heading:0},steps:[
    {x:1,y:0,heading:0,cmd},{x:2,y:0,heading:0,cmd},
    {x:3,y:0,heading:0,cmd}
  ]};
  const path=buildRoutePath(route);
  const drive=y=>trackRoutePath({x:.5,y,heading:0,vx:.5,vy:0},
    path,0,0,limits);
  const on=drive(0),off=drive(.4),far=drive(.8);
  assert.ok(on.command.vx>off.command.vx&&off.command.vx>far.command.vx);
  assert.ok(off.command.yaw<0,'left of route should steer right');
  assert.ok(Math.abs(far.headingError)<=Math.PI/6+1e-8);
  assert.ok(Math.abs(projectOnRun({x:.7,y:.4},path.runs[0]).crossTrack-.4)<1e-9);
});

test('planned curvature, reversal and pivot survive path tracking',async()=>{
  const {buildRoutePath,trackRoutePath}=await import('../src/path-follower.js');
  const arc=buildRoutePath({start:{x:0,y:0,heading:0},steps:[
    {x:.4794,y:.1224,heading:.5,cmd:{vx:.5,vy:0,yaw:.5}}
  ]});
  assert.ok(trackRoutePath({x:0,y:0,heading:0,vx:.4},arc,0,0,limits).command.yaw>.1);
  const reverse=buildRoutePath({start:{x:0,y:0,heading:0},steps:[
    {x:-.8,y:0,heading:0,cmd:{vx:-.3,vy:0,yaw:0}}
  ]});
  assert.ok(trackRoutePath({x:-.1,y:0,heading:0,vx:-.2},reverse,0,0,limits).command.vx<0);
  const pivot=buildRoutePath({start:{x:0,y:0,heading:0},steps:[
    {x:0,y:0,heading:1,cmd:{vx:0,vy:0,yaw:.8}},
    {x:.5,y:.5,heading:1,cmd:{vx:.4,vy:0,yaw:0}}
  ]});
  const turn=trackRoutePath({x:0,y:0,heading:0},pivot,0,0,limits);
  assert.ok(turn.command.vx===0&&turn.command.yaw>0&&!turn.ready);
  assert.ok(trackRoutePath({x:0,y:0,heading:1},pivot,0,0,limits).ready);
});

test('final pose can rotate without any forced forward circling',async()=>{
  const {parkAtGoal}=await import('../src/path-follower.js');
  const cmd=parkAtGoal({x:0,y:0,heading:0,vx:0,vy:0},
    {x:0,y:0,heading:Math.PI/2},limits);
  assert.equal(cmd.vx,0);assert.equal(cmd.vy,0);assert.ok(cmd.yaw>0);
});

test('arrivals are held and counted exactly once while worker is pending',()=>{
  const nav=new GoalNavigator();nav.active=true;
  let calls=0;
  nav.useBackgroundPlanner(()=>{calls++;return new Promise(()=>{});});
  const p={x:0,y:0,heading:0,vx:0,vy:0};
  nav.goal={x:0,y:0,heading:0};
  nav.route={solved:true,estimatedSeconds:1,steps:[]};
  for(let i=0;i<20;i++)nav.tick(.02,p,limits,[],()=>{});
  assert.equal(nav.arrivals,0);
  for(let i=0;i<35;i++)nav.tick(.02,p,limits,[],()=>{});
  assert.equal(nav.arrivals,1);
  assert.equal(nav.goal,null);
  assert.equal(calls,1);
  for(let i=0;i<100;i++)nav.tick(.02,p,limits,[],()=>{});
  assert.equal(nav.arrivals,1);
});

test('perfect centreline progress never triggers the old lookahead replan bug',()=>{
  const nav=new GoalNavigator();nav.active=true;
  nav.useBackgroundPlanner(()=>{throw Error('unexpected replan');});
  const start={x:0,y:0,heading:0,vx:.6,vy:0};
  const goal={x:4,y:0,heading:0};
  const cmd={vx:.6,vy:0,yaw:0};
  nav.acceptRoute(start,goal,{start,solved:true,estimatedSeconds:7,steps:[
    {x:1,y:0,heading:0,cmd},{x:2,y:0,heading:0,cmd},
    {x:3,y:0,heading:0,cmd},{x:4,y:0,heading:0,cmd}
  ]},{newMission:true});
  for(let i=0;i<190;i++)
    nav.tick(.02,{x:i*.012,y:0,heading:0,vx:.6,vy:0},limits,[],()=>{});
  assert.equal(nav.replans,0);
  assert.equal(nav.plannerError,null);
});
