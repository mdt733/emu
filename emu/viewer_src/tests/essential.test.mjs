import test from 'node:test';
import assert from 'node:assert/strict';
import { joystickToCommands, headingHoldRate, nextYawCommand } from '../src/controls.js';
import { parseShowcaseManifest, assertShowcasePolicyResponse } from '../src/showcase.js';
import { withGentleCourse, withTerrain, generateTerrainPark, localSlopeLimitDeg, surfaceSlope, GENTLE_SLOPE_DEGREES } from '../src/terrain.js';
import { sampleDemoCommand } from '../src/live-demo.js';
import { SensorObserver, inferTripod, rayHeight } from '../src/sensor.js';
import { parsePolicy } from '../src/policy.js';
const limits = {vx_min:-0.2,vx_max:0.6,vy_abs:0.22,yaw_abs:1.5,pitch_abs_rad:0.08};
const base = {schema:1,mode:'showcase',policy:'policies/champion.emupolicy',trained:true,autoplay:true,label:'EMU'};
const xml = '<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>';

test('manual joystick right/left are reversed from old vy mapping, yaw stays unchanged', () => {
  const right = joystickToCommands(1, 0, 0, 0, limits);
  const left = joystickToCommands(-1, 0, 0, 0, limits);
  assert.equal(right.vy, -0.22);
  assert.equal(left.vy, 0.22);
  assert.equal(Math.abs(right.yaw), 0);
  assert.equal(joystickToCommands(0, 0, 1, 0, limits).yaw, -1.5);
});

test('rejects path traversal, external URLs, absolute assets and malformed manifest', () => {
  for (const p of ['../secret', '/policies/champion.emupolicy', 'https://evil/anything', 'policies/../../secret', 'policies/foo.txt']) {
    assert.throws(() => parseShowcaseManifest({...base, policy:p}));
  }
  assert.throws(() => parseShowcaseManifest({...base, schema:2}));
});

test('Missing static .emupolicy SPA fallback yields actionable error', () => {
  const html = new Response('<!doctype html>', {
    status:200, headers:{'Content-Type':'text/html'},
  });
  assert.throws(() => assertShowcasePolicyResponse(html), /Bundled \.emupolicy file is missing/);
});

test('gentle terrain is real additional MuJoCo contact geometry', () => {
  const out = withGentleCourse(xml);
  assert.equal(GENTLE_SLOPE_DEGREES, 1);
  for (const name of ['viewer_ramp_up', 'viewer_ramp_top', 'viewer_ramp_down']) {
    assert.match(out, new RegExp(`name="${name}"`));
  }
  assert.match(out, /euler="0 -1 0"/);
  assert.match(out, /euler="0 1 0"/);
  assert.equal(xml.includes('viewer_ramp'), false);
  assert.throws(() => withGentleCourse(out), /already inserted/);
});

test('inverted pitch, blended diagonal steering, and side-only strafing',()=>{
  assert.equal(joystickToCommands(0,0,0,1,limits).pitch,-limits.pitch_abs_rad);
  const side=joystickToCommands(1,0,0,0,limits,'hybrid');
  assert.equal(side.vy,-limits.vy_abs);assert.equal(side.yaw,0);
  const diagonal=joystickToCommands(0.6,0.8,0,0,limits,'hybrid');
  assert.ok(diagonal.vx>0 && diagonal.yaw<0 && diagonal.vy<0);
  assert.ok(Math.abs(diagonal.vy)<Math.abs(side.vy));
});
test('heading-hold P responds to drift with saturating correction',()=>{
  assert.ok(headingHoldRate(0,0,0.1,0.75,1.5)<0);
  assert.ok(headingHoldRate(0,0,-0.1,0.75,1.5)>0);
  assert.equal(headingHoldRate(1.5,1.0,0,0.75,1.5),1.5);
});
test('continuous physical heightfield agrees with visual sample grid',()=>{
  const park=generateTerrainPark({maxDegrees:10});
  assert.equal(park.heights.length,park.resolution**2);
  assert.ok(park.heights.some(h=>h>0.05));
  assert.ok(park.trees.length>=10);
  assert.ok(park.trees.every(t=>Math.hypot(t.x,t.y)>3.4));
  assert.ok(park.trees.every(t=>Math.abs(park.sampleHeight(t.x,t.y)-t.z)<1e-7));
  const half=park.radius,cell=park.cell,n=park.resolution;
  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const x=-half+i*cell,y=-half+j*cell,k=j*n+i;
    if(Math.hypot(x,y)<=2)assert.equal(park.heights[k],0);
    if(i+1<n){
      const nx=x+cell;
      const maxAngle=Math.min(localSlopeLimitDeg(Math.hypot(x,y),10),localSlopeLimitDeg(Math.hypot(nx,y),10));
      const limit=Math.tan(maxAngle*Math.PI/180)*cell*0.56;
      assert.ok(Math.abs(park.heights[k]-park.heights[k+1])<limit+1e-5);
    }
  }
  assert.equal(localSlopeLimitDeg(2,10),1);
  assert.equal(localSlopeLimitDeg(3,10),2);
  assert.ok(localSlopeLimitDeg(12,10)>=10);
});
test('park XML uses exactly one joined heightfield and real tree collisions',()=>{
  const out=withTerrain(xml,'park',{maxDegrees:10});
  assert.match(out,/<hfield name="viewer_park_hfield"/);
  assert.match(out,/<geom name="viewer_park_ground" type="hfield"/);
  assert.match(out,/viewer_tree_trunk_0/);
  assert.doesNotMatch(out,/viewer_park_0" type="box"/);
  assert.doesNotMatch(out,/name="floor" type="plane"/);
  assert.equal(out,withTerrain(xml,'park',{maxDegrees:10}));
  assert.throws(()=>withTerrain(out,'park'),/already inserted/);
  const noTrees=withTerrain(xml,'park',{trees:false});
  assert.doesNotMatch(noTrees,/viewer_tree_trunk_0/);
});

test('yaw hold never feeds its applied correction back into the next request',()=>{
  let requestedYaw=0,headingTarget=0,realHeading=0;
  // Ask for a strong turn then RELEASE the stick. A simulated ideal yaw
  // response follows the applied command, so the P correction should vanish.
  for(let frame=0;frame<150;frame++){
    const targetYaw=frame<35?-limits.yaw_abs:0;
    const step=nextYawCommand(requestedYaw,targetYaw,headingTarget,realHeading,
                              0.02,true,0.75,limits.yaw_abs);
    requestedYaw=step.requestedYaw;
    headingTarget=step.headingTarget;
    realHeading+=step.appliedYaw*0.02;
    if(frame===35)assert.ok(requestedYaw<0);
    if(frame>60)assert.ok(Math.abs(requestedYaw)<0.000001);
    if(frame===149)assert.ok(Math.abs(step.appliedYaw)<0.015);
  }
  assert.equal(requestedYaw,0);
  const drift=nextYawCommand(0,0,0,0.1,0.02,true,0.75,1.5);
  assert.equal(drift.requestedYaw,0);
  assert.ok(drift.appliedYaw<0);
  const raw=nextYawCommand(0.2,0,0,0.1,0.02,false,0.75,1.5);
  assert.ok(Math.abs(raw.appliedYaw-raw.requestedYaw)<1e-12);
});
test('autonomous demonstration does not lock full ±1.5 yaw frontier',()=>{
  const out=sampleDemoCommand(limits,()=>0,[{vx:0.2,vy:0,yaw:-1.5}]);
  assert.equal(out.command.yaw,-0.5);
  assert.equal(joystickToCommands(0,0,1,0,limits).yaw,-1.5); // manual is still full
});
test('ground slope measures real generated terrain and robot-relative uphill arrow',()=>{
  const park=generateTerrainPark();
  const flat=surfaceSlope('park',park,0,0,0);
  assert.equal(flat.degrees,0);
  assert.equal(flat.flat,true);
  const off=surfaceSlope('park',park,park.radius+0.1,0,0);
  assert.equal(off.supported,false);
  // A known artificial eastward incline proves the angle and arrow mapping.
  const grade={radius:20,cell:0.2,sampleHeight:(x,y)=>0.1*x};
  const east=surfaceSlope('park',grade,0,0,0);
  assert.ok(Math.abs(east.degrees-Math.atan(0.1)*180/Math.PI)<1e-7);
  assert.ok(Math.abs(east.arrowDegrees)<1e-7); // forward is uphill
  const north=surfaceSlope('park',grade,0,0,Math.PI/2);
  assert.ok(Math.abs(north.arrowDegrees-90)<1e-7); // uphill points to robot's right
  const up=surfaceSlope('ramp',null,1.25,0,0);
  assert.ok(Math.abs(up.degrees-1)<1e-7);
  assert.ok(Math.abs(up.arrowDegrees)<1e-7);
  const down=surfaceSlope('ramp',null,2.5,0,0);
  assert.ok(Math.abs(down.degrees-1)<1e-7);
  assert.ok(Math.abs(Math.abs(down.arrowDegrees)-180)<1e-7);
  assert.equal(surfaceSlope('flat',null,0,0,0).degrees,0);
});


test('park grid is a separate overlay with 10 cm and 1 m lines enabled by default', async () => {
  const {readFileSync} = await import('node:fs');
  const {fileURLToPath} = await import('node:url');
  const path = fileURLToPath(new URL('../index.html',import.meta.url));
  const sim = readFileSync(fileURLToPath(new URL('../src/sim.js',import.meta.url)),'utf8');
  const appearance = readFileSync(fileURLToPath(new URL('../src/appearance.js',import.meta.url)),'utf8');
  const html = readFileSync(path,'utf8');
  assert.match(html, /id="gridToggle"[^>]+checked/);
  assert.match(appearance, /createParkGridTexture/);
  assert.match(appearance, /i<=10/);
  assert.match(sim, /parkGridOverlay\.visible=this\.showGrid/);
  assert.match(sim, /depthTest:true,polygonOffset:true/);
  assert.match(sim, /this\.parkGridOverlay\.material\.dispose\(\)/);
  assert.doesNotMatch(sim, /grassGridTexture/);
});


test('schema-8 browser sensor emulator reconstructs valid grade without privileged normal', () => {
  const ranges=[.235-.035*.06+.05*.035, .235-.035*.06-.05*.035, .235+.04*.06];
  const g=inferTripod(ranges,[0,0,-1]);
  assert.equal(g.valid,true);
  assert.ok(g.forward>0.03&&g.forward<0.09);
  assert.ok(g.side>0.01&&g.side<0.06);
  assert.ok(g.depth>.2&&g.depth<.4);
  assert.equal(inferTripod([0,.2,.2],[0,0,-1]).valid,false);
  assert.ok(Math.abs(rayHeight([0,0,.26],[0,0,-1],()=>0)-.26)<1e-5);
});

test('v5016 delayed ToF/flow readings preserve acquisition time and sensor confidence', () => {
  const contract = {
    schema:'sim2real_232',
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:0,flow_dropout_prob:0,gyro_bias_std_rad_s:0,sensor_noise_scale:0}
  };
  const pose = speed=>({
    torso:[0,0,.26],R:[1,0,0,0,1,0,0,0,1],
    vbody:[speed,0,0],gyro:[0,0,0],heightAt:()=>0
  });
  const o=new SensorObserver(contract);
  o.reset(pose(0),false);
  assert.equal(o.read([0,0,-1]).velocity[0],0);
  o.advance(pose(.15),false);
  assert.ok(Math.abs(o.read([0,0,-1]).velocity[0])<1e-8);
  o.advance(pose(.15),false);
  const read=o.read([0,0,-1]);
  assert.ok(read.velocity[0]>.14&&read.velocity[0]<.16);
  assert.ok(read.groundConfidence>0);
  const dropout=new SensorObserver({...contract,settings:{...contract.settings,
    tof_period_frames:1,flow_period_frames:1,tof_dropout_prob:1,flow_dropout_prob:1}});
  dropout.reset(pose(0),false);
  dropout.advance(pose(.1),true);
  dropout.advance(pose(.1),true);
  const bad=dropout.read([0,0,-1]);
  assert.equal(bad.velocityConfidence,0);
  assert.equal(bad.groundConfidence,0);
});

test('versioned browser parser accepts sensor 232 and keeps legacy schema independent', () => {
  const c = {
    schema:'sim2real_232',sensor_model_revision:'tripod_tof_flow_async_v1',
    history_len:4,action_dim:6,
    tripod_mounts_body_m:[[.035,-.05,-.025],[.035,.05,-.025],[-.04,0,-.025]],
    tripod_beams_body:[[0,0,-1],[0,0,-1],[0,0,-1]],
    tof_range_limits_m:[.035,.6],
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:.01,flow_dropout_prob:.02,gyro_bias_std_rad_s:.015,sensor_noise_scale:1.5},
  };
  const fields=Array.from({length:55},(_,i)=>'legacy_'+i).concat(
    ['ground_forward_grade_rad','ground_sideways_grade_rad','ground_visibility_confidence']);
  c.base_fields=fields;
  const policy={format:'emu-policy',schema:2,
    policy:{action_dim:6,history_len:4,base_obs_dim:58,obs_dim:232,layers:[{},{},{}]},
    source:{checkpoint_schema:8},robot_xml:xml,
    observation:{base_fields:fields,sensor_contract:c},control:{}};
  assert.equal(parsePolicy(JSON.stringify(policy)).policy.obs_dim,232);
  assert.throws(()=>parsePolicy(JSON.stringify({...policy,schema:1})));
  assert.throws(()=>parsePolicy(JSON.stringify({
    ...policy,observation:{...policy.observation,
      sensor_contract:{...c,sensor_model_revision:'unknown'}}})),/Unsupported v5016 sensor/);
});
