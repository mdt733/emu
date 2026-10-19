import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {
  ARENA_RADIUS_M,TREE_CLEARANCE_M,angleError,wrapAngle,
  sampleGoal,integratePose,planPoseRoute,GoalNavigator
} from '../src/goal-navigation.js';
import {GoalPlannerWorker} from '../src/goal-planner-client.js';
import {buildRoutePath,projectOnRun,trackRoutePath,parkAtGoal}
  from '../src/path-follower.js';
import {makeRng} from '../src/rng.js';
import {DemoHandoff,MANUAL_IDLE_SECONDS,commandStickPositions}
  from '../src/demo-handoff.js';
import {joystickToCommands} from '../src/controls.js';
import {VirtualStick} from '../src/joystick.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5,
              pitch_abs_rad:.08};
const origin={x:0,y:0,heading:0,vx:0,vy:0};
const read=p=>readFileSync(fileURLToPath(new URL(p,import.meta.url)),'utf8');

test('uniform pose goals remain within 4m and clear of solid trees',()=>{
  const rng=makeRng(733),trees=[{x:2,y:0},{x:0,y:2}];
  for(let i=0;i<100;i++){
    const p=sampleGoal(rng,trees,origin);
    assert.ok(Math.hypot(p.x,p.y)<=ARENA_RADIUS_M);
    assert.ok(Math.hypot(p.x,p.y)>.95);
    for(const t of trees)
      assert.ok(Math.hypot(p.x-t.x,p.y-t.y)>=TREE_CLEARANCE_M);
    assert.ok(Math.abs(p.heading)<=Math.PI);
  }
});

test('pose lattice plans forward, reverse, pivot and offset goals',()=>{
  for(const target of [
    {x:2,y:0,heading:0},{x:-.8,y:0,heading:0},
    {x:0,y:0,heading:Math.PI/2},{x:1.2,y:1.3,heading:1.1}
  ]){
    const route=planPoseRoute(origin,target,limits,[],{maxNodes:6500});
    assert.ok(route?.solved,'unplanned goal '+JSON.stringify(target));
    assert.ok(route.steps.length>1);
    assert.equal(route.steps.at(-1).final,true);
    assert.deepEqual(route.start,{...origin});
    assert.ok(route.estimatedSeconds>0);
  }
});

test('planned steps stay outside tree keep-outs and invalid goals are rejected',()=>{
  const obstacle={x:.8,y:0},goal={x:1.9,y:0,heading:0};
  const route=planPoseRoute(origin,goal,limits,[obstacle],{maxNodes:7000});
  if(route)for(const step of route.steps)
    assert.ok(Math.hypot(step.x-obstacle.x,step.y-obstacle.y)>=TREE_CLEARANCE_M);
  assert.equal(planPoseRoute(origin,{x:.8,y:0,heading:0},limits,[obstacle]),null);
  assert.throws(()=>planPoseRoute(origin,{x:NaN,y:0,heading:0},limits),/Nonfinite/);
  assert.ok(Math.abs(wrapAngle(4*Math.PI))<1e-12);
  assert.ok(Math.abs(angleError(-Math.PI+.01,Math.PI-.01)-.02)<1e-10);
});

test('analytic twist integration handles forward turns, lateral motion and pivot',()=>{
  const a=integratePose(origin,{vx:.4,vy:0,yaw:1},1);
  assert.ok(a.x>.3&&a.y>.15&&a.heading>.9);
  const b=integratePose(origin,{vx:0,vy:.25,yaw:0},.8);
  assert.ok(Math.abs(b.y-.2)<1e-9);
  const c=integratePose(origin,{vx:0,vy:0,yaw:-1},1);
  assert.equal(c.heading,-1);
});

test('path projection tracks the line, not isolated lattice points',()=>{
  const cmd={vx:.7,vy:0,yaw:0};
  const route=buildRoutePath({start:origin,steps:[
    {x:1,y:0,heading:0,cmd},{x:2,y:0,heading:0,cmd},
    {x:4,y:0,heading:0,cmd}
  ]});
  const near=projectOnRun({x:.75,y:.4},route.runs[0],0);
  assert.ok(Math.abs(near.s-.75)<1e-8);
  assert.ok(Math.abs(near.crossTrack-.4)<1e-8);
  const evaluate=y=>trackRoutePath({x:.5,y,heading:0,vx:.5,vy:0,
    yawRate:0},route,0,0,limits);
  const on=evaluate(0),off=evaluate(.4),far=evaluate(.8);
  assert.ok(on.command.vx>off.command.vx&&off.command.vx>far.command.vx);
  assert.ok(off.command.yaw<0);
  assert.ok(Math.abs(far.headingError)<=Math.PI/6+1e-8);
});

test('lateral velocity predicts drift before the robot crosses the line',()=>{
  const path=buildRoutePath({start:origin,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.8,vy:0,yaw:0}},
    {x:4,y:0,heading:0,cmd:{vx:.8,vy:0,yaw:0}}
  ]});
  const base={x:.5,y:0,heading:0,vx:.6,vy:0,yawRate:0};
  const aligned=trackRoutePath(base,path,0,0,limits);
  const sliding=trackRoutePath({...base,vy:.3},path,0,0,limits);
  assert.ok(sliding.command.yaw<aligned.command.yaw-.05);
  assert.ok(sliding.command.vx<aligned.command.vx);
  const leftSpin=trackRoutePath({...base,y:.25,yawRate:.8},path,0,0,limits);
  const rightSpin=trackRoutePath({...base,y:.25,yawRate:-.8},path,0,0,limits);
  assert.ok(rightSpin.command.yaw>leftSpin.command.yaw+.35);
});

test('route tracking retains real forward, reverse and pivot manoeuvre boundaries',()=>{
  const arc=buildRoutePath({start:origin,steps:[
    {x:.48,y:.12,heading:.5,cmd:{vx:.5,vy:0,yaw:.5}}
  ]});
  assert.ok(trackRoutePath(origin,arc,0,0,limits).command.yaw>.1);
  const reverse=buildRoutePath({start:origin,steps:[
    {x:-.8,y:0,heading:0,cmd:{vx:-.3,vy:0,yaw:0}}
  ]});
  assert.ok(trackRoutePath({x:-.1,y:0,heading:0,vx:-.2},reverse,0,0,limits).command.vx<0);
  const pivot=buildRoutePath({start:origin,steps:[
    {x:0,y:0,heading:1,cmd:{vx:0,vy:0,yaw:.8}},
    {x:.5,y:.5,heading:1,cmd:{vx:.4,vy:0,yaw:0}}
  ]});
  const first=trackRoutePath(origin,pivot,0,0,limits);
  assert.equal(first.command.vx,0);
  assert.ok(first.command.yaw>0&&!first.ready);
  assert.ok(trackRoutePath({...origin,heading:1},pivot,0,0,limits).ready);
});

test('final pose parks with a pivot rather than forced forward circling',()=>{
  const c=parkAtGoal(origin,{x:0,y:0,heading:Math.PI/2},limits);
  assert.equal(c.vx,0);assert.equal(c.vy,0);assert.ok(c.yaw>0);
});

test('navigation closed loop parks straight goal in analytic kinematics',()=>{
  const nav=new GoalNavigator(733);nav.active=true;
  assert.ok(nav.setGoal(origin,{x:.9,y:0,heading:0},limits,[]));
  let pose={...origin};
  for(let i=0;i<1400&&!nav.arrivals;i++){
    let cmd;
    nav.tick(.02,pose,limits,[],x=>cmd=x);
    assert.ok(cmd);
    const next=integratePose(pose,cmd,.02);
    pose={...next,vx:cmd.vx,vy:cmd.vy};
  }
  assert.equal(nav.arrivals,1);
  assert.equal(nav.failures,0);
});

test('arrival dwell never double-counts a completed goal while Worker is pending',()=>{
  const nav=new GoalNavigator(733);nav.active=true;
  let requests=0;
  nav.useBackgroundPlanner(()=>{requests++;return new Promise(()=>{});});
  nav.goal={x:0,y:0,heading:0};
  nav.route={solved:true,steps:[],estimatedSeconds:1};
  for(let i=0;i<20;i++)nav.tick(.02,origin,limits,[],()=>{});
  assert.equal(nav.arrivals,0);
  for(let i=0;i<130;i++)nav.tick(.02,origin,limits,[],()=>{});
  assert.equal(nav.arrivals,1);
  assert.equal(requests,1);
  assert.equal(nav.goal,null);
});

test('background planner uses only lattice requests and does not block the actor',async()=>{
  let fake;
  const broker=new GoalPlannerWorker({makeWorker:()=>{
    fake={posted:[],postMessage(x){this.posted.push(x);},terminate(){}};
    return fake;
  }});
  const promise=broker.request({kind:'next',pose:origin,limits,trees:[]});
  assert.equal(fake.posted.length,1);
  assert.equal(fake.posted[0].kind,'next');
  assert.equal('planner' in fake.posted[0],false);
  const message={id:fake.posted[0].id,kind:'next',
    goal:{x:1,y:0,heading:0},route:{solved:true}};
  fake.onmessage({data:message});
  assert.equal((await promise).goal.x,1);
  broker.dispose();
});

test('auto-stick position is the exact inverse of split manual commands',()=>{
  for(const command of [
    {vx:.6,vy:0,yaw:.4,pitch:.04},
    {vx:-.3,vy:.14,yaw:-.8,pitch:-.05},
    {vx:0,vy:0,yaw:0,pitch:0}
  ]){
    const sticks=commandStickPositions(command,limits);
    const back=joystickToCommands(sticks.left.x,sticks.left.y,
      sticks.right.x,sticks.right.y,limits);
    for(const key of ['vx','vy','yaw','pitch'])
      assert.ok(Math.abs(back[key]-command[key])<1e-8,key);
  }
});

test('visual-only joystick updates cannot be mistaken for operator inputs',()=>{
  const events=[];
  const nub={clientWidth:40,style:{}};
  const root={clientWidth:120,querySelector:()=>nub,addEventListener(){}};
  const stick=new VirtualStick(root,(x,y)=>events.push([x,y]));
  stick.setDisplay(.4,.6);
  assert.equal(events.length,0);
  stick.set(.2,.3);
  assert.equal(events.length,1);
});

test('manual stick or keyboard activity holds control; demo resumes after 30s idle',()=>{
  assert.equal(MANUAL_IDLE_SECONDS,30);
  const mode=new DemoHandoff();
  assert.equal(mode.autonomous,true);
  mode.takeControl(100);
  assert.equal(mode.autonomous,false);
  assert.equal(mode.shouldResume(129.99,false),false);
  assert.equal(mode.shouldResume(150,true),false,'held input cannot time out');
  mode.takeControl(140);
  assert.equal(mode.shouldResume(169.99,false),false);
  assert.equal(mode.shouldResume(170,false),true);
  mode.resume();assert.equal(mode.autonomous,true);
});

test('public viewer has one navigational demo and preserves manual commands',()=>{
  const html=read('../index.html'),main=read('../src/main.js');
  const worker=read('../src/goal-worker.js'),client=read('../src/goal-planner-client.js');
  assert.match(html,/id="liveDemo" type="checkbox" checked/);
  assert.match(html,/Pose lattice/);
  assert.doesNotMatch(html,/id="demoMode"|id="navPlanner"|Advanced commands/);
  assert.doesNotMatch(html,/Chase view follows behind/);
  assert.doesNotMatch(html,/heightCmd|rollCmd|stepCmd|phaseCmd/);
  assert.match(main,/let demoActive = true/);
  assert.match(main,/commandStickPositions\(navLastCommand,limits\(\)\)/);
  assert.match(main,/handoff.shouldResume\(now/);
  assert.match(main,/leftStick.setDisplay/);
  assert.match(main,/sim\.setDemoCommands\(command\)/);
  assert.match(main,/sim\?\.setJoystickTargets/);
  assert.match(worker,/planPoseRoute/);
  assert.doesNotMatch(worker,/planCurvatureRoute|bidirectionalMotionRoute/);
  assert.doesNotMatch(client,/planner='grid'/);
});
