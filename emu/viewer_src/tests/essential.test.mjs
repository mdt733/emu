import test from 'node:test';
import {measuredAttitude,sampleEpisodeCadence,smoothFactor,chasePose,sampleShove} from '../src/sim-math.js';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parsePolicy} from '../src/policy.js';
import * as depthLabSensor from '../src/sensor.js';
import {joystickToCommands,nextYawCommand} from '../src/controls.js';
import {createSolidCarpet} from '../src/training-solid-carpet.js';
import {withTerrain,solidCarpetXml} from '../src/terrain.js';
import {trainingLimits} from '../src/training-carpet.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5,pitch_abs_rad:.08};
const measured=(step,more={})=>({source:'training',seed:735,step,
  foundationSteps:20_000_000,academyIntroSteps:60_000_000,
  academyStream:true,slopeDeg:12,transitionDeg:3,stepUpM:.03,stepDownM:.04,...more});
const data=step=>trainingLimits({training_step:step,
  latest_metrics:{step,frontier_vx_hi:'.28',frontier_vx_lo:'-.07',
    frontier_vy:'.12',frontier_yaw:'.7',frontier_push_n:'.65',
    resilience_difficulty:'.36',slope_max_deg:'.7',terrain_height_m:'.004',
    academy_height_up_m:'.003',academy_height_down_m:'.002',
    slope_exposure:'.8',terrain_exposure:'.5'},
  training_config:{terrain_carpet:true,academy_stream:true,seed:735,
    foundation_steps:20000000,academy_intro_steps:60000000,
    push_max_force_n:.7}});

test('array-only 360 terrain policy keeps legacy actors separate and derives ground from rays',()=>{
  const fields=Array.from({length:55},(_,i)=>'legacy_'+i).concat(
    ['ground_forward_grade_rad','ground_sideways_grade_rad','ground_visibility_confidence']);
  for(let i=0;i<16;i++)fields.push(`tof4x4_range_${Math.floor(i/4)}${i%4}_m`);
  for(let i=0;i<16;i++)fields.push(`tof4x4_valid_${Math.floor(i/4)}${i%4}`);
  const c={schema:'terrain_360',sensor_model_revision:'forward_tof4x4_array_plane_fit_no_tripod_2',
    base_fields:fields,history_len:4,action_dim:6,
    array_ground_estimator:'weighted_4x4_plane_residual_8mm_v1',
    array_ground_max_fit_residual_m:.008,array_ground_min_valid_cells:6,
    tof4x4_mount_body_m:[0,0,0],tof4x4_pitch_from_horizontal_deg:[40,60,80,100],
    tof4x4_yaw_deg:[-22.5,-7.5,7.5,22.5],tof4x4_range_limits_m:[.04,1.5],
    tof4x4_occlusion:'nearest ray-sphere foot hit masks terrain return',
    tof4x4_validity:'one bit per cell, delayed, zero on dropout',
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:0,flow_dropout_prob:0,gyro_bias_std_rad_s:0,sensor_noise_scale:0}};
  const bundle={format:'emu-policy',schema:3,source:{checkpoint_schema:9},
    robot_xml:'<mujoco/>',policy:{action_dim:6,history_len:4,base_obs_dim:90,
      obs_dim:360,layers:[{},{},{}]},observation:{base_fields:fields,sensor_contract:c},control:{}};
  assert.equal(parsePolicy(JSON.stringify(bundle)).policy.obs_dim,360);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,schema:2})),/schema-3/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    source:{checkpoint_schema:8}})),/schema-9/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    observation:{...bundle.observation,sensor_contract:{...c,
      sensor_model_revision:'tripod_tof_flow_async_v1'}}})),/array-only/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    observation:{...bundle.observation,sensor_contract:{...c,
      tripod_mounts_body_m:[[0,0,0]]}}})),/array-only/);
  const dirs=depthLabSensor.terrainActorRays([1,0,0,0,1,0,0,0,1]);
  const ranges=dirs.map(v=>-.25/v[2]);
  const fit=depthLabSensor.fitArrayGround(ranges,Array(16).fill(true),[0,0,-1]);
  assert.equal(fit.valid,true);
  assert.ok(Math.abs(fit.depth-.25)<1e-3,fit.depth);
  assert.ok(Math.abs(fit.forward)<1e-3);
  assert.ok(Math.abs(fit.side)<1e-3);
  const broken=ranges.slice();
  broken[0]-=.065;broken[1]-=.065;broken[2]-=.065;broken[3]-=.065;
  assert.equal(depthLabSensor.fitArrayGround(broken,Array(16).fill(true),[0,0,-1]).valid,false);
});


test('schema-9 4x4 rays intersect physical floor/box and foot occlusion',()=>{
  const {terrainActorRays,actorTerrainDistance,TerrainActorObserver}=depthLabSensor;
  const R=[1,0,0,0,1,0,0,0,1];
  const rays=terrainActorRays(R);
  assert.equal(rays.length,16);
  assert.ok(rays[0][0]>rays[12][0]);
  // Plane z=0 and raised box from x=.25 to .45, top z=.02.
  const model={ngeom:3,geom_type:[0,6,2],geom_bodyid:[0,0,1],
    geom_size:[0,0,0,.10,.2,.15,.018,0,0]};
  const data={
    geom_xpos:[0,0,0,.35,0,-.13,0,0,-1],
    geom_xmat:Array(3).fill(R).flat()
  };
  const origin=[0,0,.26],ray=rays[8];
  const ground=actorTerrainDistance(origin,ray,model,data,[]);
  assert.ok(ground>.04&&ground<1.5);
  const footHit=origin.map((v,i)=>v+ray[i]*.1);
  data.geom_xpos.splice(6,3,...footHit);
  assert.equal(actorTerrainDistance(origin,ray,model,data,[2]),0);
  const contract={schema:'terrain_360',
    settings:{tof_period_frames:2,tof_delay_frames:1,tof_dropout_prob:0,sensor_noise_scale:0}};
  const observer=new TerrainActorObserver(contract);
  const pose={R,torso:origin};
  observer.reset(pose,model,data,[2],false);
  assert.equal(observer.read().valid[8],false);
  // Delay holds last reading for a frame even when the occluding foot moves.
  data.geom_xpos.splice(6,3,0,0,-1);
  observer.advance(pose,model,data,[2],false);
  assert.equal(observer.read().valid[8],false);
  observer.advance(pose,model,data,[2],false);
  assert.equal(observer.read().valid[8],false);
  observer.advance(pose,model,data,[2],false);
  assert.equal(observer.read().valid[8],true);
});


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


test('published Pinnacle 260M is compatible with the lean 360-input viewer',()=>{
  const archived=readFileSync(new URL('../../../showcase/champion.emupolicy',
    import.meta.url),'utf8');
  const b=parsePolicy(archived);
  assert.equal(b.schema,3);
  assert.equal(b.policy.obs_dim,360);
  assert.equal(b.source.checkpoint_schema,9);
  assert.equal(b.observation.sensor_contract.sensor_model_revision,
    'forward_tof4x4_array_plane_fit_no_tripod_2');
});


test('browser MuJoCo sees nine literal contact boxes, no hfield or buried floor',()=>{
  const carpet=createSolidCarpet(measured(90_000_000));
  const raw='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="5 5 0.05"/></worldbody></mujoco>';
  const xml=withTerrain(raw,'carpet',{park:carpet});
  assert.equal((xml.match(/mocap="true"/g)||[]).length,9);
  assert.equal((xml.match(/viewer_carpet_solid_\d+" type="box"/g)||[]).length,9);
  assert.doesNotMatch(xml,/name="floor"/);
  assert.doesNotMatch(xml,/type="hfield"/);
  assert.doesNotMatch(xml,/viewer_park_hfield/);
  assert.equal((solidCarpetXml(carpet).match(/size="0.258 0.258 0.270"/g)||[]).length,9);
});


test('live true-prism origin explicitly warns about nine-box browser replay',()=>{
  const b=trainingLimits({
    training_step:60000000,
    latest_metrics:{step:60000000,slope_max_deg:.7,terrain_height_m:.004},
    training_config:{terrain_carpet:true,terrain_carpet_prisms:true,
      academy_stream:true,seed:735,foundation_steps:20000000}
  });
  assert.equal(b.trainingCarpet,true);
  assert.equal(b.trainingCarpetPrisms,true);
  assert.match(b.physicsMode,/18-prism MJX training/);
  const boxSource=data(60000000);
  assert.equal(boxSource.trainingCarpetPrisms,false);
});

test('simulation helper consolidation preserves independent physics/UI functions',()=>{
  const basis=[1,0,0, 0,1,0, 0,0,1];
  const feet=[0,0,0];
  const att=measuredAttitude(basis,[0,0,.2],feet,feet);
  assert.ok(Math.abs(att.chassisPitch)<1e-9);
  assert.equal(att.robotLean,0);
  assert.equal(sampleEpisodeCadence({vx:0,vy:0},{phase_rate_max_hz:2.6},()=>0),.85);
  assert.equal(smoothFactor(0,8),0);
  assert.ok(smoothFactor(.02,8)>0);
  const p=chasePose([0,0,.2],0);
  assert.equal(p.position.length,3);
  assert.equal(sampleShove({intensity:0,maxForceN:0}),null);
});
