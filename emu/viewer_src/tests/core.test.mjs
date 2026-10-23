import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {parsePolicy} from '../src/policy.js';
import {parseShowcaseManifest} from '../src/showcase.js';
import {withTerrain,generateTerrainPark,surfaceSlope} from '../src/terrain.js';
import {SensorObserver} from '../src/sensor.js';
import * as depthLabSensor from '../src/sensor.js';
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
      sensor_contract:{...c,sensor_model_revision:'unknown'}}})),/Unsupported v5016 sensor/);
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

// 5. This is a single public product, not a menu of abandoned demos.
test('viewer starts in lattice demo without obsolete controls or actor changes',()=>{
  const html=sourceText('../index.html');
  const main=sourceText('../src/main.js');
  const sim=sourceText('../src/sim.js');
  const worker=sourceText('../src/goal-worker.js');
  assert.match(html,/id="liveDemo" type="checkbox" checked/);
  assert.match(main,/let demoActive = true/);
  assert.doesNotMatch(html,/id="demoMode"|id="navPlanner"|Advanced commands|heightCmd|phaseCmd/);
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


test('v5022 front sensor uses 4x4 raw range geometry and adjustable physical mount',()=>{
  const {depthProbeRays,readDepthProbe}=depthLabSensor;
  const torso=[0,0,.2584],identity=[1,0,0,0,1,0,0,0,1];
  const a=depthProbeRays(torso,identity,{forward:.09,tilt:50});
  const b=depthProbeRays(torso,identity,{forward:.15,tilt:65});
  assert.equal(a.directions.length,16);
  assert.ok(a.directions.every(zone=>zone.length===5));
  assert.ok(a.directions[0][0][2]>a.directions[15][0][2]);
  assert.ok(Math.abs(b.origin[0]-a.origin[0]-.06)<1e-9);
  assert.ok(b.directions[5][0][2]<a.directions[5][0][2]);
  // Synthetic plane forward model checks bearing, not a magic height grid.
  const plane=(origin,direction)=>direction[2]<-.001
    ? Math.max(0,-origin[2]/direction[2]):Infinity;
  const measured=readDepthProbe(a,plane);
  assert.equal(measured.ranges.length,16);
  assert.ok(measured.valid.some(Boolean));
  assert.ok(measured.valid.some((x,i)=>x&&measured.ranges[i]>0));
  const missing=readDepthProbe(a,()=>Infinity);
  assert.ok(missing.valid.every(x=>!x));
  assert.ok(missing.ranges.every(x=>x===0));
});

test('v5022 diagnostic obstacle courses alter REAL MuJoCo support geometry',()=>{
  const xml='<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05" /></worldbody></mujoco>';
  const rise=withTerrain(xml,'sensor',{obstacle:'step-up'});
  const down=withTerrain(xml,'sensor',{obstacle:'step-down'});
  const hill=withTerrain(xml,'sensor',{obstacle:'crest'});
  const log=withTerrain(xml,'sensor',{obstacle:'log'});
  assert.ok(rise.includes('depth_landing'));
  assert.ok(!rise.includes('name="floor"'));
  assert.ok(rise!==down);
  assert.ok(hill.includes('depth_rise')&&hill.includes('depth_exit'));
  assert.ok(log.includes('depth_log')&&log.includes('name="floor"'));
  assert.throws(()=>withTerrain(xml,'sensor',{obstacle:'unknown'}));
});


test('v5022 depth inspector compares 16/64/80 casts without averaging a step into a ramp',()=>{
  const {depthProbeRays,readDepthProbe}=depthLabSensor;
  const pose=depthProbeRays([0,0,.2584],[1,0,0,0,1,0,0,0,1]);
  for(const [subrays,expected] of [[1,16],[4,64],[5,80]]){
    let calls=0;
    const r=readDepthProbe(pose,()=>{calls++;return .25;},{subrays});
    assert.equal(calls,expected);
    assert.ok(r.valid.every(Boolean));
    assert.ok(r.ranges.every(x=>Math.abs(x-.25)<1e-8));
    assert.equal(r.selfOccluded.length,16);
    assert.equal(r.hitPoints.length,16);
  }
  assert.throws(()=>readDepthProbe(pose,()=>.2,{subrays:8}));
});

test('v5022 4-ray edge zones report mixed step returns and robot occlusion',()=>{
  const {depthProbeRays,readDepthProbe}=depthLabSensor;
  const pose=depthProbeRays([0,0,.2584],[1,0,0,0,1,0,0,0,1]);
  // Across each zone, two rays see a riser and two see ground.
  let calls=0;
  const mixed=readDepthProbe(pose,()=>{
    const j=calls++%4;
    return {distance:j<2?.18:.27,kind:j===0?'self':'terrain'};
  },{subrays:4});
  assert.ok(mixed.valid.every(Boolean));
  assert.ok(mixed.mixed.every(Boolean));
  assert.ok(mixed.selfOccluded.every(Boolean));
  // Lower median of .18/.18/.27/.27 is the measured .18 surface,
  // NOT the non-existent averaged .225 intermediate height.
  assert.ok(mixed.ranges.every(x=>Math.abs(x-.18)<1e-8));
  assert.ok(mixed.hitPoints.every(p=>p.every(Number.isFinite)));
});


test('v5022 slant range gradient on a single flat surface is not a step',()=>{
  const {depthProbeRays,readDepthProbe}=depthLabSensor;
  const pose=depthProbeRays([0,0,.2584],[1,0,0,0,1,0,0,0,1],{tilt:35});
  const flat=readDepthProbe(pose,(origin,ray)=>({
    distance:origin[2]/(-ray[2]),
    kind:'terrain',surface:'same_flat_ground',
  }),{subrays:4});
  assert.ok(flat.valid.some(Boolean));
  assert.ok(flat.mixed.every(v=>!v));
  assert.ok(flat.selfOccluded.every(v=>!v));
});


test('depth obstacle terrain leaves live sensor telemetry safe in every course',()=>{
  // Previously surfaceSlope('sensor') threw every frame; the sensor kept
  // scanning but the UI stopped updating outside the radial park.
  for(const mode of ['flat','ramp','park','sensor']){
    const park=mode==='park'?generateTerrainPark({features:false,trees:false}):null;
    const output=surfaceSlope(mode,park,0,0,0);
    assert.equal(typeof output.supported,'boolean');
    assert.ok(Number.isFinite(output.degrees));
  }
  assert.equal(surfaceSlope('sensor',null,.5,0,0).supported,false);
});

test('depth mount can sit at torso root and has fixed 4x4 zone geometry',()=>{
  const {depthProbeRays}=depthLabSensor;
  const R=[1,0,0,0,1,0,0,0,1];
  for(const forward of [0,.01,.045,.09,.18]){
    const p=depthProbeRays([0,0,.25],R,{forward,tilt:50});
    assert.equal(p.origin[0],forward);
    assert.equal(p.directions.length,16);
    assert.ok(p.directions.every(z=>z.length===5));
  }
  assert.throws(()=>depthProbeRays([0,0,.25],R,{forward:-.01,tilt:50}));
});

test('default park has real obstacle ring, clear spawn, and opt-out parity',()=>{
  const base='<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>';
  const park=generateTerrainPark({trees:false});
  assert.ok(park.flatRadius>=4);
  assert.equal(park.features.length,6);
  assert.ok(park.features.every(f=>Math.hypot(f.x,f.y)>2.5));
  assert.ok(park.features.every(f=>Math.hypot(f.x,f.y)<park.flatRadius));
  assert.ok(park.features.every(f=>f.geoms.includes('type=')));
  assert.ok(park.features.some(f=>f.id==='crest'&&f.geoms.includes('quat=')));
  assert.ok(park.features.some(f=>f.id==='step-up'));
  assert.ok(park.features.some(f=>f.id==='step-down'));
  assert.ok(park.features.some(f=>f.id==='log'));
  assert.ok(park.features.some(f=>f.id==='angled'));
  assert.ok(park.features.some(f=>f.id==='trough'));
  assert.equal(park.sampleHeight(0,0),0);
  const xml=withTerrain(base,'park',{park});
  assert.match(xml,/park_feature_step_up/);
  assert.match(xml,/park_feature_crest_rise/);
  assert.match(xml,/park_feature_crest_fall/);
  assert.match(xml,/park_feature_log/);
  assert.match(xml,/park_feature_trough_near/);
  const noFeatures=generateTerrainPark({trees:false,features:false});
  assert.equal(noFeatures.features.length,0);
  assert.equal(noFeatures.flatRadius,park.flatRadius);
  const cleanXml=withTerrain(base,'park',{park:noFeatures});
  assert.doesNotMatch(cleanXml,/park_feature_/);
  assert.match(cleanXml,/viewer_park_ground/);
  // Feature toggling must not mutate the physical hfield itself.
  assert.deepEqual(Array.from(park.heights),Array.from(noFeatures.heights));
});


test('square 45-degree optical footprint stays square at 90-degree downward tilt',()=>{
  const {depthProbeRays,depthGridRays}=depthLabSensor;
  const eye=[0,0,.26],R=[1,0,0,0,1,0,0,0,1];
  const footprint=depthGridRays(R,{tilt:90});
  assert.equal(footprint.length,25);
  const hit=footprint.map(ray=>{
    assert.ok(ray[2]<0);
    const t=-eye[2]/ray[2];
    return eye.map((p,k)=>p+t*ray[k]);
  });
  const half=.26*Math.tan(Math.PI/8);
  for(let row=0;row<5;row++)for(let col=0;col<5;col++){
    const p=hit[row*5+col];
    assert.ok(Math.abs(p[0]-(1-row/2)*half)<1e-10);
    assert.ok(Math.abs(p[1]-(col/2-1)*half)<1e-10);
    assert.ok(Math.abs(p[2])<1e-10);
  }
  // Corner cells form a true nondegenerate square, not a V/triangle.
  assert.ok(Math.abs(hit[0][0]-hit[4][0])<1e-10);
  assert.ok(Math.abs(hit[0][1]-hit[20][1])<1e-10);
  const zones=depthProbeRays(eye,R,{forward:0,tilt:90,height:0});
  assert.equal(zones.directions.length,16);
  for(let row=0;row<4;row++)for(let col=0;col<4;col++){
    const ray=zones.directions[row*4+col][0];
    const t=-eye[2]/ray[2],p=eye.map((x,k)=>x+t*ray[k]);
    assert.ok(Math.abs(p[0]-(.75-row*.5)*half)<1e-10);
    assert.ok(Math.abs(p[1]-(col*.5-.75)*half)<1e-10);
  }
});

test('sensor square FoV remains independent in both axes at all pitch settings',()=>{
  const {depthGridRays,depthProbeRays}=depthLabSensor;
  const R=[1,0,0,0,1,0,0,0,1];
  for(const tilt of [0,25,50,75,90]){
    const rays=depthGridRays(R,{tilt});
    assert.ok(rays.every(ray=>Math.abs(Math.hypot(...ray)-1)<1e-10));
    // Centre row left/right spread exists at 90 degrees too.
    assert.ok(Math.abs(rays[10][1]-rays[14][1])>.6);
    // Upper/lower spread is independent of horizontal image-plane axis.
    assert.ok(Math.hypot(...rays[2].map((x,i)=>x-rays[22][i]))>.6);
    const probe=depthProbeRays([0,0,.26],R,{tilt,forward:0});
    assert.equal(probe.directions.length,16);
    assert.ok(probe.directions.every(zone=>zone.length===5));
  }
  assert.throws(()=>depthGridRays(R,{tilt:91}));
});


test('0mm 70-degree provisional optical mount preserves side and front coverage',()=>{
  const {depthProbeRays}=depthLabSensor;
  const R=[1,0,0,0,1,0,0,0,1],torso=[0,0,.2584];
  const probe=depthProbeRays(torso,R);
  assert.ok(Math.abs(probe.origin[0])<1e-10);
  assert.ok(Math.abs(probe.origin[2]-.2184)<1e-10);
  const ground=(row,col)=>{
    const ray=probe.directions[row*4+col][0];
    const t=-probe.origin[2]/ray[2];
    return probe.origin.map((v,k)=>v+t*ray[k]);
  };
  const farL=ground(0,1),farR=ground(0,2);
  const nearL=ground(3,0),nearR=ground(3,3);
  assert.ok(farL[0]>.15 && farR[0]>.15);
  assert.ok(nearL[1]<-.05 && nearR[1]>.05);
  assert.ok(Math.abs(nearL[0]-nearR[0])<1e-10);
  assert.ok(farL[0]-nearL[0]>.14);
});


test('near outer zones of 0/70 mount distinguish left-to-right ground grade',()=>{
  const {depthProbeRays,readDepthProbe}=depthLabSensor;
  const pose=depthProbeRays([0,0,.2584],[1,0,0,0,1,0,0,0,1]);
  // Ground plane z=tan(10°)*worldY, not privileged future sample heights.
  const m=Math.tan(10*Math.PI/180);
  const scan=readDepthProbe(pose,(origin,dir)=>{
    const denom=dir[2]-m*dir[1];
    return denom<0?-origin[2]/denom:Infinity;
  },{subrays:4});
  assert.ok(scan.valid[12] && scan.valid[15]);
  assert.ok(scan.ranges[12]-scan.ranges[15]>.012);
  // The 4x4 grid represents slant range, not a precomputed ground angle.
  assert.equal(scan.ranges.length,16);
});
