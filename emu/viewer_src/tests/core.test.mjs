import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {parsePolicy} from '../src/policy.js';
import {parseShowcaseManifest} from '../src/showcase.js';
import {withTerrain,generateTerrainPark,surfaceSlope} from '../src/terrain.js';
import {SensorObserver} from '../src/sensor.js';
import {joystickToCommands,nextYawCommand} from '../src/controls.js';
import {commandStickPositions,DemoHandoff} from '../src/demo-handoff.js';
import {VirtualStick} from '../src/joystick.js';
import {makeRng} from '../src/rng.js';
import {sampleGoal,planPoseRoute,refineRouteSteps,integratePose,GoalNavigator,TREE_CLEARANCE_M}
  from '../src/goal-navigation.js';
import {buildRoutePath,projectOnRun,trackRoutePath,referenceAt,parkAtGoal}
  from '../src/path-follower.js';
import {GoalPlannerWorker} from '../src/goal-planner-client.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5,pitch_abs_rad:.08};
const origin={x:0,y:0,heading:0,vx:0,vy:0};
const sourceText=path=>readFileSync(fileURLToPath(new URL(path,import.meta.url)),'utf8');

// 1. A public viewer must reject bad policy/manifest contracts.
test('policy loading preserves schema-2 sensor contracts and safe local assets',()=>{
  const c={
    schema:'sim2real_232',sensor_model_revision:'tripod_tof_flow_async_v1',
    history_len:4,action_dim:6,
    tripod_mounts_body_m:[[.035,-.05,-.025],[.035,.05,-.025],[-.04,0,-.025]],
    tripod_beams_body:[[0,0,-1],[0,0,-1],[0,0,-1]],
    tof_range_limits_m:[.035,.6],
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:.01,flow_dropout_prob:.02,gyro_bias_std_rad_s:.015,sensor_noise_scale:1.5}
  };
  const fields=Array.from({length:55},(_,i)=>'legacy_'+i).concat(
    ['ground_forward_grade_rad','ground_sideways_grade_rad','ground_visibility_confidence']);
  c.base_fields=fields;
  const bundle={format:'emu-policy',schema:2,
    policy:{action_dim:6,history_len:4,base_obs_dim:58,obs_dim:232,layers:[{},{},{}]},
    source:{checkpoint_schema:8},robot_xml:'<mujoco/>',
    observation:{base_fields:fields,sensor_contract:c},control:{}};
  assert.equal(parsePolicy(JSON.stringify(bundle)).policy.obs_dim,232);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,schema:1})));
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    observation:{...bundle.observation,
      sensor_contract:{...c,sensor_model_revision:'unknown'}}})),/Unsupported sensor\/observation model/);
  const manifest={schema:1,mode:'showcase',trained:true,
    policy:'policies/champion.emupolicy'};
  assert.equal(parseShowcaseManifest(manifest).trained,true);
  assert.throws(()=>parseShowcaseManifest({...manifest,policy:'../secret'}));
});

// 2. Terrain has real physics geometry; not merely decorative foliage.
test('terrain park creates a physical heightfield with solid tree trunks',()=>{
  const xml='<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>';
  const out=withTerrain(xml,'park',{maxDegrees:10});
  assert.match(out,/<geom name="viewer_park_ground" type="hfield"/);
  assert.match(out,/viewer_tree_trunk_0/);
  assert.doesNotMatch(out,/name="floor" type="plane"/);
  const park=generateTerrainPark({maxDegrees:10});
  assert.equal(park.heights.length,park.resolution**2);
  assert.equal(surfaceSlope('park',park,0,0,0).degrees,0);
  assert.ok(park.heights.some(x=>x>.05));
});

// 3. Realistic sensor timing is part of the trained actor interface.
test('delayed ToF/flow emulation retains reading time and dropout confidence',()=>{
  const c={schema:'sim2real_232',
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:0,flow_dropout_prob:0,gyro_bias_std_rad_s:0,sensor_noise_scale:0}};
  const p=v=>({torso:[0,0,.26],R:[1,0,0,0,1,0,0,0,1],
    vbody:[v,0,0],gyro:[0,0,0],heightAt:()=>0});
  const sensor=new SensorObserver(c);
  sensor.reset(p(0),false);
  sensor.advance(p(.15),false);
  assert.ok(Math.abs(sensor.read([0,0,-1]).velocity[0])<1e-8);
  sensor.advance(p(.15),false);
  assert.ok(sensor.read([0,0,-1]).velocity[0]>.14);
  assert.ok(sensor.read([0,0,-1]).groundConfidence>0);
  const lost=new SensorObserver({...c,settings:{...c.settings,
    tof_period_frames:1,flow_period_frames:1,tof_dropout_prob:1,flow_dropout_prob:1}});
  lost.reset(p(0),false);
  lost.advance(p(.1),true);lost.advance(p(.1),true);
  assert.equal(lost.read([0,0,-1]).velocityConfidence,0);
});

// 4. The operator's four taught commands and heading setpoint stay intact.
test('manual axes and heading hold obey the trained command envelope',()=>{
  assert.equal(joystickToCommands(1,0,0,0,limits).vy,-limits.vy_abs);
  assert.equal(joystickToCommands(0,-1,0,0,limits).vx,limits.vx_min);
  assert.equal(joystickToCommands(0,0,1,0,limits).yaw,-limits.yaw_abs);
  assert.equal(joystickToCommands(0,0,0,1,limits).pitch,-limits.pitch_abs_rad);
  let heading=0,target=0,request=0;
  for(let i=0;i<120;i++){
    const rate=i<25?.9:0;
    const next=nextYawCommand(request,rate,target,heading,.02,true,2.2,1.5);
    request=next.requestedYaw;target=next.headingTarget;
    heading+=next.appliedYaw*.02;
  }
  assert.equal(request,0);
  assert.ok(Math.abs(target-heading)<.04);
});

// 5. One proven planner with TWO demo destinations: random or the
// v5018 physical course circuit. Old discarded multi-planner and advanced
// actor-mutation controls must remain absent. Do not accidentally prohibit
// the new, deliberately supported demoMode dropdown.
test('viewer defaults to random lattice demo and selects an ordered challenge circuit without changing actor',()=>{
  const html=sourceText('../index.html');
  const main=sourceText('../src/main.js');
  const sim=sourceText('../src/sim.js');
  const worker=sourceText('../src/goal-worker.js');
  assert.match(html,/id="liveDemo" type="checkbox" checked/);
  assert.match(main,/let demoActive = true/);
  assert.match(html,/id="demoMode"/);
  assert.match(html,/<option value="random" selected>/);
  assert.match(html,/<option value="circuit">/);
  assert.match(main,/navigator\.setCircuitWaypoints\(circuit\)/);
  assert.match(worker,/kind==='replan'\|\|kind==='target'/);
  assert.doesNotMatch(html,/id="navPlanner"|Advanced commands|heightCmd|phaseCmd/);
  assert.doesNotMatch(sim,/setAdvanced\(|phaseOverride|viewerPitchLimitRad/);
  for(const f of ['this.command.height','this.command.roll',
                  'this.command.step_height','this.phase'])
    assert.ok(sim.includes(f),'missing trained actor observation '+f);
  assert.match(worker,/planPoseRoute/);
  assert.doesNotMatch(worker,/planCurvatureRoute|bidirectionalMotionRoute/);
});

// 6. Autonomous sticks are genuinely visual-only; the human can interrupt.
test('autonomous stick display mirrors commands without generating manual input',()=>{
  const c={vx:-.3,vy:.14,yaw:-.8,pitch:-.05};
  const sticks=commandStickPositions(c,limits);
  const back=joystickToCommands(sticks.left.x,sticks.left.y,
    sticks.right.x,sticks.right.y,limits);
  for(const k of ['vx','vy','yaw','pitch'])
    assert.ok(Math.abs(c[k]-back[k])<1e-9,k);
  let manualEvents=0;
  const root={clientWidth:120,querySelector:()=>({clientWidth:40,style:{}}),
    addEventListener(){}};
  const stick=new VirtualStick(root,()=>manualEvents++);
  stick.setDisplay(.6,.3);
  assert.equal(manualEvents,0);
  stick.set(.4,.2);
  assert.equal(manualEvents,1);
});

// 7. Manual input is never snatched away while a stick/key is held.
test('manual takeover returns to autonomous navigation only after 30 seconds idle',()=>{
  const h=new DemoHandoff();
  assert.equal(h.autonomous,true);
  h.takeControl(100);
  assert.equal(h.autonomous,false);
  assert.equal(h.shouldResume(129.99,false),false);
  assert.equal(h.shouldResume(145,true),false);
  h.takeControl(140);
  assert.equal(h.shouldResume(169.99,false),false);
  assert.equal(h.shouldResume(170,false),true);
  h.resume();
  assert.equal(h.autonomous,true);
});

// 8. The only global planner must handle fundamentally different poses.
test('pose lattice plans forward, reverse, pivot and offset goals',()=>{
  for(const goal of [
    {x:1.1,y:0,heading:0},{x:-.8,y:0,heading:0},
    {x:0,y:0,heading:Math.PI/2},{x:1.2,y:1.3,heading:1.1}
  ]){
    const route=planPoseRoute(origin,goal,limits,[],{maxNodes:6500});
    assert.ok(route?.solved,'failed '+JSON.stringify(goal));
    assert.ok(route.steps.at(-1).final);
    assert.ok(route.estimatedSeconds>0);
    assert.ok(route.steps.length>=2);
    if(goal.x>1 && goal.y===0){
      assert.ok(route.refined,'straight A* steps must be refined');
      assert.ok(route.steps.some(p=>p.cmd.profiled),'acceleration-aware speed profile');
    }
  }
});

// 9. Obstacle geometry must constrain the planner as well as the renderer.
test('lattice targets respect tree keep-out and reject unsafe goals',()=>{
  const trees=[{x:.8,y:0}];
  assert.equal(planPoseRoute(origin,{x:.8,y:0,heading:0},limits,trees),null);
  const rng=makeRng(733);
  for(let i=0;i<50;i++){
    const g=sampleGoal(rng,trees,origin);
    assert.ok(Math.hypot(g.x,g.y)<=4);
    assert.ok(Math.hypot(g.x-.8,g.y)>=TREE_CLEARANCE_M);
  }
  // Centreline path that forces a real obstacle detour. Check complete
  // interpolated route including the last parking segment, not nodes alone.
  const tree={x:1,y:0},route=planPoseRoute(origin,
    {x:2,y:0,heading:0},limits,[tree],{maxNodes:6500});
  assert.ok(route?.solved);
  let previous=origin;
  for(const step of route.steps){
    const length=Math.hypot(step.x-previous.x,step.y-previous.y);
    for(let j=0;j<=Math.max(1,Math.ceil(length/.025));j++){
      const n=Math.max(1,Math.ceil(length/.025)),t=j/n;
      const x=previous.x+(step.x-previous.x)*t;
      const y=previous.y+(step.y-previous.y)*t;
      assert.ok(Math.hypot(x-tree.x,y-tree.y)>=TREE_CLEARANCE_M-.001,
        'refining a route must never cut the obstacle clearance circle');
    }
    previous=step;
  }
});

// 10. The new feedback controller follows the route line, not waypoints.
test('path follower slows off-line, anticipates drift, and respects pivot/reverse',()=>{
  const cmd={vx:.7,vy:0,yaw:0};
  const path=buildRoutePath({start:origin,steps:[
    {x:1,y:0,heading:0,cmd},{x:2,y:0,heading:0,cmd},
    {x:4,y:0,heading:0,cmd}
  ]});
  assert.ok(Math.abs(projectOnRun({x:.6,y:.4},path.runs[0]).crossTrack-.4)<1e-8);
  const at=y=>trackRoutePath({x:.5,y,heading:0,vx:.5,vy:0,yawRate:0},
    path,0,0,limits);
  const on=at(0),off=at(.4),far=at(.8);
  assert.ok(on.command.vx>off.command.vx&&off.command.vx>far.command.vx);
  assert.ok(off.command.yaw<0);
  const drift=trackRoutePath({x:.5,y:0,heading:0,vx:.5,vy:.3,yawRate:0},
    path,0,0,limits);
  assert.ok(drift.command.yaw<on.command.yaw);
  const reverse=buildRoutePath({start:origin,steps:[
    {x:-.8,y:0,heading:0,cmd:{vx:-.3,vy:0,yaw:0}}
  ]});
  assert.ok(trackRoutePath({...origin,x:-.1,vx:-.2},
    reverse,0,0,limits).command.vx<0);
  assert.ok(parkAtGoal(origin,{x:0,y:0,heading:1},limits).yaw>0);
  // Both directions of the speed envelope are bounded by the planner's
  // simple acceleration/braking assumptions. No new test file needed.
  const refined=planPoseRoute(origin,{x:2,y:0,heading:0},limits,[]);
  const motion=refined.steps.filter(p=>p.cmd.profiled&&p.cmd.vx>0);
  assert.ok(motion.length>4);
  for(let i=1;i<motion.length;i++){
    const a=motion[i-1],b=motion[i],ds=Math.hypot(b.x-a.x,b.y-a.y);
    const dv2=b.cmd.vx**2-a.cmd.vx**2;
    assert.ok(dv2<=2*.85*ds+.055,'acceleration-limited reference');
    assert.ok(dv2>=-2*1.0*ds-.055,'braking-limited reference');
  }
  // The follower must consume the speed reference by *progress*, not
  // multiply the whole profiled trajectory by arbitrary cruising throttles.
  const profiled=buildRoutePath({start:origin,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.60,vy:0,yaw:0,profiled:true}},
    {x:2,y:0,heading:0,cmd:{vx:.20,vy:0,yaw:0,profiled:true}}
  ]});
  const reference=referenceAt(profiled.runs[0],1.5);
  assert.ok(Math.abs(reference.command.vx-.40)<1e-9);
  const measured=(vx,y=0)=>trackRoutePath({x:1.5,y,heading:0,vx,vy:0,
    yawRate:0},profiled,0,1.4,limits);
  const matched=measured(.40),slow=measured(.10),fast=measured(.65);
  assert.ok(Math.abs(matched.command.vx-.40)<.02,
    'on-line profiled velocity should pass through without double throttling');
  assert.ok(slow.command.vx>matched.command.vx+.04,
    'feedback should compensate for a gait slower than the planned speed');
  assert.ok(fast.command.vx<matched.command.vx-.04,
    'feedback should brake when measured gait is ahead of the reference');
  assert.ok(measured(.40,.5).command.vx<matched.command.vx,
    'off-route corrections still cap speed for safer interception');
});

// 11. A mission has one physical arrival, including settle/dwell time.
test('navigator reaches goals and does not double-count pending arrivals',()=>{
  const nav=new GoalNavigator(800);nav.active=true;
  assert.ok(nav.setGoal(origin,{x:.9,y:0,heading:0},limits,[]));
  let pose={...origin},seenTracking=false;
  for(let i=0;i<1400&&!nav.arrivals;i++){
    let c;nav.tick(.02,pose,limits,[],cmd=>c=cmd);
    assert.ok(c);
    pose={...integratePose(pose,c,.02),vx:c.vx,vy:c.vy};
    const tracked=nav.status(pose);
    if(tracked.modeSeconds.forward>.1){
      seenTracking=true;
      assert.ok(tracked.plannedModeSeconds.forward>0);
      for(const metric of ['meanTrackError','meanHeadingError','meanSpeedError'])
        assert.ok(Number.isFinite(tracked[metric])&&tracked[metric]>=0);
    }
  }
  assert.equal(nav.arrivals,1);
  assert.ok(seenTracking,'tracking telemetry must report live mission error');
  const held=new GoalNavigator(123);held.active=true;
  let requested=0;
  held.useBackgroundPlanner(()=>{requested++;return new Promise(()=>{});});
  held.goal={x:0,y:0,heading:0};
  held.route={solved:true,steps:[],estimatedSeconds:1};
  for(let i=0;i<20;i++)held.tick(.02,origin,limits,[],()=>{});
  assert.equal(held.arrivals,0);
  for(let i=0;i<100;i++)held.tick(.02,origin,limits,[],()=>{});
  assert.equal(held.arrivals,1);
  assert.equal(requested,1);
});

// 12. Planning stays asynchronous, and old multi-planner mode is gone.
test('background Worker handles route requests without blocking the viewer',async()=>{
  let fake;
  const broker=new GoalPlannerWorker({makeWorker:()=>{
    fake={posted:[],postMessage(x){this.posted.push(x);},terminate(){}};
    return fake;
  }});
  const result=broker.request({kind:'next',pose:origin,limits,trees:[]});
  assert.equal(fake.posted.length,1);
  assert.equal('planner' in fake.posted[0],false);
  fake.onmessage({data:{id:fake.posted[0].id,kind:'next',goal:origin,
    route:{steps:[],solved:true}}});
  assert.equal((await result).route.solved,true);
  broker.dispose();
});
