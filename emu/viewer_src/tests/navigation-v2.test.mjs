import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {buildNavigationTerrain,terrainRiskAt,sweptTerrainCost} from '../src/navigation-terrain.js';
import {OnlineGaitResponse,PredictivePathController,vectorFieldGuidance,
  clampNavigationCommand} from '../src/predictive-navigation.js';
import {buildRoutePath,trackRoutePath} from '../src/path-follower.js';
import {planPoseRoute,integratePose,flythroughConnector,reachedRoamingWaypoint,
  forwardWaypointApproach,GoalNavigator,NAV_BENCHMARK_TARGETS} from '../src/goal-navigation.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5};
const pose={x:0,y:0,heading:0,vx:0,vy:0,yawRate:0};
const text=path=>readFileSync(fileURLToPath(new URL(path,import.meta.url)),'utf8');

test('4x4 depth laboratory lives only inside Settings and starts inactive',()=>{
  const html=text('../index.html'),css=text('../src/style.css');
  assert.ok(html.indexOf('id="settingsDrawer"')<html.indexOf('id="depthPanel"'));
  assert.ok(html.indexOf('id="depthPanel"')<html.indexOf('id="resultCard"'));
  assert.match(html,/<details class="drawer-group" id="depthDetails">/);
  assert.match(html,/<input id="depthOn" type="checkbox" \/>/);
  assert.match(css,/\.settings-drawer \.depth-panel \{\s*position:static/);
  assert.doesNotMatch(css,/\.depth-panel \{[\s\S]*?position:absolute/);
});

test('terrain grade cost is finite on gentle ground and blocks known severe slopes',()=>{
  const grade=(x,y)=>x<0?0:x*.32;
  const terrain=buildNavigationTerrain(grade,{cell:.20,cautionDegrees:4,avoidDegrees:12});
  assert.equal(terrainRiskAt(null,1,1),0);
  assert.equal(terrainRiskAt(terrain,-1,0),0);
  assert.equal(terrainRiskAt(terrain,1,0),Infinity);
  assert.equal(sweptTerrainCost(terrain,{x:-1,y:0},{x:1,y:0}),Infinity);
  const flat=buildNavigationTerrain(()=>0);
  assert.equal(terrainRiskAt(flat,1,0),0);
  assert.equal(sweptTerrainCost(flat,{x:0,y:0},{x:1,y:0}),1);
  assert.equal(planPoseRoute(pose,{x:1,y:0,heading:0},limits,[],{terrain}),null);
  assert.ok(planPoseRoute(pose,{x:1,y:0,heading:0},limits,[],{terrain:flat})?.solved);
});

test('vector-field guidance corrects signed off-track drift using vy',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.60,vy:0,yaw:0,profiled:true}},
    {x:2,y:0,heading:0,cmd:{vx:.60,vy:0,yaw:0,profiled:true}}]});
  const straight=path.runs[0];
  const on=vectorFieldGuidance({...pose,x:.3},straight,0,limits);
  const off=vectorFieldGuidance({...pose,x:.3,y:.25},straight,0,limits);
  assert.ok(Math.abs(on.command.vy)<1e-4);
  assert.ok(off.command.vy<-.02,'positive y deviation requests rightward correction');
  assert.ok(off.command.vx>0);
});

test('classic route follower preserves diagonal planner vx and vy',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:.18,heading:0,cmd:{vx:.6,vy:.11,yaw:0,profiled:true}},
    {x:2,y:.36,heading:0,cmd:{vx:.6,vy:.11,yaw:0,profiled:true}}]});
  const guided=trackRoutePath({...pose,x:.06,y:.01,vx:.2},
    path,0,0,limits);
  assert.ok(guided.command.vx>.15);
  assert.ok(guided.command.vy>.01,'sideways component should not be thrown away');
});

test('response model predicts inertia and learns safely bounded gains',()=>{
  const model=new OnlineGaitResponse();
  const first=model.rollout(pose,{vx:.7,vy:.1,yaw:.3},.2);
  assert.ok(first.vx>0&&first.vx<.7);
  assert.ok(first.yawRate>0&&first.yawRate<.3);
  assert.ok(first.x>0);
  let actual={...pose};
  for(let i=0;i<45;i++){
    model.issued={vx:.5,vy:.1,yaw:.3};
    model.observe(actual,.1);
    actual=model.rollout(actual,model.issued,.1);
  }
  assert.ok(model.report().samples>30);
  for(const key of ['vx','vy','yaw'])
    assert.ok(Number.isFinite(model.gain[key])&&model.tau[key]>.08);
});

test('predictive rollout returns bounded commands and a finite future trace',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.65,vy:0,yaw:0,profiled:true}},
    {x:2,y:0,heading:0,cmd:{vx:.65,vy:0,yaw:0,profiled:true}}]});
  const predictive=new PredictivePathController();
  predictive.observe(pose,.1);
  const baseline={vx:.5,vy:0,yaw:0,pitch:0};
  const out=predictive.command(pose,path,0,0,limits,[],null,baseline);
  assert.equal(out.prediction.length,5);
  assert.ok(Number.isFinite(out.cost));
  assert.deepEqual(clampNavigationCommand(out.command,limits),out.command);
  assert.ok(out.prediction.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
  const blank=predictive.command(pose,{runs:[]},0,0,limits,[],null,baseline);
  assert.deepEqual(blank.command,baseline);
});

test('global planner and predictive module stay above the neural actuator contract',()=>{
  const nav=text('../src/goal-navigation.js');
  const main=text('../src/main.js');
  const worker=text('../src/goal-worker.js');
  assert.match(main,/navControllerMode/);
  assert.match(main,/sim\.setHeadingHold\(false\)/);
  assert.match(worker,/terrain/);
  assert.match(nav,/PredictivePathController/);
  assert.match(main,/sim\.setNavigationPrediction/);
  assert.doesNotMatch(text('../src/predictive-navigation.js'),/data\.ctrl|policy\.infer|mj_step/);
});


test('five-goal physical benchmark has repeatable poses and finite summaries',()=>{
  assert.equal(NAV_BENCHMARK_TARGETS.length,5);
  assert.ok(NAV_BENCHMARK_TARGETS.every(g=>
    Math.hypot(g.x,g.y)<1.9&&Number.isFinite(g.heading)));
  const nav=new GoalNavigator(733);
  nav.active=true;
  nav.startBenchmark();
  assert.equal(nav.benchmark.index,0);
  assert.equal(nav.benchmark.controller,'predictive');
  for(let i=0;i<5;i++){
    nav.elapsed=i+1;
    nav.trackingTime=1;
    nav.trackErrorIntegral=.12*i;
    nav.finishBenchmarkLeg(i<4);
    assert.equal(nav.benchmark.results.at(-1).goal,i+1);
  }
  assert.equal(nav.benchmark.index,5);
  assert.equal(nav.benchmark.complete,true);
  assert.equal(nav.benchmark.results.filter(r=>r.success).length,4);
  let command;
  nav.tick(.1,pose,limits,[],c=>command=c);
  assert.deepEqual(command,{vx:0,vy:0,yaw:0,pitch:0});
  const ui=text('../src/main.js');
  assert.match(ui,/resetForNavigationBenchmark/);
  assert.match(ui,/navBenchmarkCopy/);
  assert.match(text('../src/goal-worker.js'),/kind==='target'/);
  assert.match(text('../src/sim.js'),/benchmark\?\(\)=>\.5:Math\.random/);
});


test('fly-through connector reaches the exact marker via bounded real commands',()=>{
  const from={x:0,y:0,heading:0};
  const goal={x:.22,y:.055,heading:.13};
  const seg=flythroughConnector(from,goal,limits);
  assert.ok(seg,'normal near-goal pose has an exact nonzero motion connector');
  const arrival=integratePose(from,seg.cmd,seg.dt);
  assert.ok(Math.hypot(arrival.x-goal.x,arrival.y-goal.y)<.002);
  assert.ok(Math.abs(arrival.heading-goal.heading)<.002);
  assert.ok(seg.cmd.vx>0);
  assert.ok(Math.abs(seg.cmd.vy)<=limits.vy_abs);
  assert.ok(Math.abs(seg.cmd.yaw)<=limits.yaw_abs);
  const entire=planPoseRoute(from,{x:1.20,y:0,heading:0},limits,[],
    {passThrough:true});
  assert.ok(entire?.solved&&entire.flythrough);
  const last=entire.steps.at(-1);
  assert.ok(Math.abs(last.x-1.20)<.001);
  assert.ok(last.cmd.vx>0,'route ends at the marker with a walk command, not a zero-speed drawing');
  const checkpoint=planPoseRoute(from,{x:1.20,y:0,heading:0},limits);
  assert.ok(checkpoint?.solved);
  assert.equal(checkpoint.steps.at(-1).cmd.vx,0,
    'benchmark/default route keeps precise final parking');
});

test('an already-planned following waypoint advances without stopping or resampling',()=>{
  const start={x:0,y:0,heading:0,vx:.3,vy:0,yawRate:0};
  const first={x:1,y:0,heading:0};
  const second={x:2,y:0,heading:0};
  const leg=(from,to)=>({
    start:from,steps:[{...to,cmd:{vx:.5,vy:0,yaw:0,pitch:0,profiled:true},
      dt:2,maneuver:'forward',flythrough:true}],
    estimatedSeconds:2,solved:true,flythrough:true
  });
  const nav=new GoalNavigator(733);
  nav.active=true;
  const future=leg(first,second);
  nav.acceptRoute(start,first,leg(start,first),
    {newMission:true,following:{goal:second,route:future}});
  let applied;
  nav.tick(.1,{...start,x:.85},limits,[],command=>{applied=command});
  assert.equal(nav.arrivals,1,'first waypoint crossed at normal walking speed');
  assert.equal(nav.goal.x,second.x,'second waypoint activates in same control tick');
  assert.ok(applied.vx>.05,'no zero-command arrival hold or precision parking');
  assert.equal(nav.arrivalHold,0);
});

test('a stalled physical route triggers recovery without waiting for extreme cross-track error',()=>{
  const nav=new GoalNavigator(733),start={...pose};
  nav.active=true;
  const goal={x:1.2,y:0,heading:0};
  const route=planPoseRoute(start,goal,limits);
  assert.ok(route?.solved);
  nav.acceptRoute(start,goal,route,{newMission:true});
  for(let i=0;i<75;i++)
    nav.tick(.1,{...start},limits,[],()=>{});
  assert.ok(nav.motionStall>0||nav.replans>0||nav.stagnant>0);
  assert.ok(nav.replans>0,'stall watchdog should replan from the actual pose');
});


test('a roaming waypoint is position-based even if the biped is moving and facing away',()=>{
  const goal={x:1,y:0,heading:0};
  const route={solved:true,flythrough:true,start:{x:0,y:0,heading:0},
    steps:[{x:1,y:0,heading:0,cmd:{vx:.5,vy:0,yaw:0}}]};
  assert.equal(reachedRoamingWaypoint(
    {x:.83,y:0,heading:Math.PI,vx:-.40,vy:0},goal,route),true);
  assert.equal(reachedRoamingWaypoint(
    {x:.40,y:.40,heading:0},goal,route),false);
  assert.equal(reachedRoamingWaypoint(
    {x:1.10,y:.12,heading:Math.PI},goal,route),true,
    'a fast walker may cross the target plane between control frames');
  assert.equal(reachedRoamingWaypoint(
    {x:1.06,y:.31,heading:0},goal,route),false,
    'do not silently claim a wide lateral miss');
});

test('a through-waypoint connector cannot reverse into the marker',()=>{
  const reverse=flythroughConnector(
    {x:0,y:0,heading:Math.PI},
    {x:.22,y:0,heading:Math.PI},limits);
  assert.equal(reverse,null);
  const direction=forwardWaypointApproach(
    {x:0,y:0,heading:Math.PI,vx:0,vy:0,yawRate:0},
    {x:.38,y:0,heading:0},limits);
  assert.ok(direction.vx>=0,'ordinary roaming never commands a reverse park');
  assert.ok(Math.abs(direction.yaw)>.1);
});

test('near-goal roaming arrival immediately continues even if the following leg is missing',()=>{
  const goal={x:1,y:0,heading:0};
  const nav=new GoalNavigator(734);
  nav.active=true;
  let requested=0;
  nav.useBackgroundPlanner(()=>{requested++;return new Promise(()=>{});});
  const pose={x:.81,y:0,heading:2.7,vx:.45,vy:0,yawRate:0};
  const route={
    start:{x:0,y:0,heading:0},solved:true,flythrough:true,
    estimatedSeconds:2,steps:[
      {x:1,y:0,heading:0,dt:1.5,
       cmd:{vx:.50,vy:0,yaw:0,pitch:0,profiled:true}}
    ]
  };
  nav.acceptRoute(pose,goal,route,{newMission:true});
  let command;
  nav.tick(.1,pose,limits,[],c=>command=c);
  assert.equal(nav.arrivals,1);
  assert.equal(nav.goal,null,'finished waypoint does not remain stuck in approaching goal');
  assert.equal(requested,1,'one background request for a new waypoint');
  assert.deepEqual(command,{vx:0,vy:0,yaw:0,pitch:0});
  for(let i=0;i<20;i++)nav.tick(.1,pose,limits,[],()=>{});
  assert.equal(nav.arrivals,1,'do not double count while planning');
});

test('the precise-pose benchmark retains its speed and heading hold',()=>{
  const nav=new GoalNavigator(733);
  nav.active=true;
  nav.startBenchmark();
  const start={x:0,y:0,heading:0,vx:0,vy:0,yawRate:0};
  const goal={x:1,y:0,heading:0};
  const route=planPoseRoute(start,goal,limits);
  assert.ok(route?.solved);
  nav.acceptRoute(start,goal,route,{newMission:true});
  nav.tick(.1,{...start,x:.90,heading:Math.PI,vx:.5},limits,[],()=>{});
  assert.equal(nav.arrivals,0,'strict orientation and speed remain enforced in benchmark');
});

test('a stalled near-goal approach does not wait for the full mission timeout',()=>{
  const nav=new GoalNavigator(734);
  nav.active=true;
  const start={x:0,y:0,heading:0,vx:0,vy:0,yawRate:0};
  const goal={x:.65,y:0,heading:0};
  const route=planPoseRoute(start,goal,limits);
  assert.ok(route?.solved);
  nav.acceptRoute(start,goal,route,{newMission:true});
  // Simulate an actor that turned to face away from the marker but cannot
  // translate. A pivot/terminal controller must not wait ~30-105 seconds.
  const stuck={x:.08,y:0,heading:Math.PI,vx:0,vy:0,yawRate:0};
  for(let i=0;i<105&&!nav.failures;i++)
    nav.tick(.1,stuck,limits,[],()=>{});
  assert.ok(nav.failures>=1,'after bounded recovery attempts, abandon missed target');
  assert.ok(nav.elapsed<30,'recovery precedes normal mission timeout');
});


test('a pivot reached at a realistic 30 cm drift can continue; a stuck pivot replans',()=>{
  const start={x:0,y:0,heading:0,vx:0,vy:0,yawRate:0};
  const goal={x:1.50,y:0,heading:0};
  const route={
    start,solved:true,estimatedSeconds:5,flythrough:false,
    steps:[
      {x:0,y:0,heading:Math.PI/2,dt:1.5,
        cmd:{vx:0,vy:0,yaw:.8,pitch:0},maneuver:'pivot-left'},
      {x:1.5,y:0,heading:0,dt:2.5,
        cmd:{vx:.5,vy:0,yaw:0,pitch:0},maneuver:'forward'}
    ]
  };
  const path=buildRoutePath(route);
  const close=trackRoutePath({...start,x:.30,heading:Math.PI/2},
    path,0,0,limits);
  assert.equal(close.ready,true,'the follower can absorb modest drift after a turn');
  const far=trackRoutePath({...start,x:.52,heading:Math.PI/2},
    path,0,0,limits);
  assert.equal(far.ready,false,'a large offset still needs recovery');
  const nav=new GoalNavigator(5036);
  nav.active=true;
  const requests=[];
  nav.useBackgroundPlanner(request=>{
    requests.push(request);return new Promise(()=>{});
  });
  nav.acceptRoute(start,goal,route,{newMission:true});
  const stopped={...start,x:.52,heading:Math.PI/2};
  for(let i=0;i<45;i++)
    nav.tick(.1,stopped,limits,[],()=>{});
  assert.ok(requests.some(req=>req.kind==='replan'),
    'finished-but-offset pivot must trigger async replanning');
  assert.ok(nav.pivotNoProgress>2);
});
