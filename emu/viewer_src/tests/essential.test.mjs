import test from 'node:test';
import {measuredAttitude,sampleEpisodeCadence,smoothFactor,chasePose,sampleShove,clearanceAboveSupport} from '../src/sim-math.js';
import assert from 'node:assert/strict';
import {parsePolicy} from '../src/policy.js';
import * as depthLabSensor from '../src/sensor.js';
import {joystickToCommands,nextYawCommand} from '../src/controls.js';
import {terrainRoleEnvelope} from '../src/training-solid-carpet.js';
import {createSensorLabCarpet,sensorLabPrismVertices,sensorLabPrismXml,carpetSpawnSoleLift} from '../src/sensor-lab-carpet.js';
import {withTerrain} from '../src/terrain.js';
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

test('only the current physical 360-input policy contract is accepted',()=>{
  const fields=Array.from({length:55},(_,i)=>'base_'+i).concat(
    ['ground_forward_grade_rad','ground_sideways_grade_rad','ground_visibility_confidence']);
  for(let i=0;i<16;i++)fields.push(`tof4x4_range_${Math.floor(i/4)}${i%4}_m`);
  for(let i=0;i<16;i++)fields.push(`tof4x4_valid_${Math.floor(i/4)}${i%4}`);
  const identity=[1,0,0,0,1,0,0,0,1];
  const rays=depthLabSensor.depthProbeRays([0,0,0],identity,
    {tilt:70,forward:0,height:-.03,lateral:0,yaw:0}).directions.map(z=>z[0]);
  const c={
    schema:'terrain_360',
    sensor_model_revision:'vl53l8cx_array_plane_square45_below30_v3',
    base_fields:fields,history_len:4,action_dim:6,
    array_ground_estimator:'weighted_4x4_plane_residual_8mm_v1',
    array_ground_max_fit_residual_m:.008,array_ground_min_valid_cells:6,
    tof4x4_mount_body_m:[0,0,-.03],tof4x4_fov_horizontal_deg:45,
    tof4x4_fov_vertical_deg:45,tof4x4_tilt_down_deg:70,
    tof4x4_mount_yaw_deg:0,tof4x4_rays_body:rays,
    tof4x4_hardware_zone_ids:[3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12],
    tof4x4_range_limits_m:[.04,1.5],
    tof4x4_occlusion:'nearest ray-sphere foot hit masks terrain return',
    tof4x4_validity:'one bit per cell, delayed, zero on dropout',
    settings:{tof_period_frames:2,flow_period_frames:1,tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:0,flow_dropout_prob:0,gyro_bias_std_rad_s:0,sensor_noise_scale:0}};
  const bundle={format:'emu-policy',schema:3,source:{checkpoint_schema:9},
    robot_xml:'<mujoco/>',policy:{action_dim:6,history_len:4,base_obs_dim:90,
      obs_dim:360,layers:[{},{},{}]},observation:{base_fields:fields,sensor_contract:c},control:{}};
  assert.equal(parsePolicy(JSON.stringify(bundle)).policy.obs_dim,360);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,schema:2})),/current 360-input/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    source:{checkpoint_schema:8}})),/current 360-input/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    observation:{...bundle.observation,sensor_contract:{...c,
      sensor_model_revision:'forward_tof4x4_array_plane_fit_no_tripod_2'}}})),/current physical/);
  assert.throws(()=>parsePolicy(JSON.stringify({...bundle,
    observation:{...bundle.observation,sensor_contract:{...c,
      tof4x4_mount_body_m:[0,0,0]}}})),/Policy rays/);
  const lab={...bundle,sensor_lab:true,source:{checkpoint_schema:9,step:0,
      actor_sha256:'not-applicable-sensor-lab'},
    policy:{...bundle.policy,layers:[]}};
  assert.equal(parsePolicy(JSON.stringify(lab)).sensor_lab,true);
  assert.throws(()=>parsePolicy(JSON.stringify({...lab,
    policy:{...lab.policy,layers:[{},{},{}]}})),/cannot carry an actor/);

  const ranges=rays.map(v=>-.37/v[2]);
  const fit=depthLabSensor.fitArrayGround(ranges,Array(16).fill(true),[0,0,-1],rays,-.03);
  assert.equal(fit.valid,true);
  assert.ok(Math.abs(fit.depth-.4)<1e-3,fit.depth);
  assert.ok(Math.abs(fit.forward)<1e-3);
  assert.ok(Math.abs(fit.side)<1e-3);
  const broken=ranges.slice();
  broken[0]-=.065;broken[1]-=.065;broken[2]-=.065;broken[3]-=.065;
  assert.equal(depthLabSensor.fitArrayGround(
    broken,Array(16).fill(true),[0,0,-1],rays,-.03).valid,false);
});

test('physical ST 45-degree policy rays agree with inspector and reject wrong mount',()=>{
  const physical=depthLabSensor.depthProbeRays([0,0,0],[1,0,0,0,1,0,0,0,1],
    {forward:0,lateral:0,height:-.03,tilt:70,yaw:0});
  const centres=physical.directions.map(zone=>zone[0]);
  assert.equal(centres.length,16);
  assert.ok(centres.every(ray=>ray[0]>0&&ray[2]<0),'all physical zone centres aim ahead and down');
  const identity=[1,0,0,0,1,0,0,0,1];
  const actor=depthLabSensor.terrainActorRays(identity,centres);
  actor.forEach((dir,i)=>dir.forEach((val,j)=>
    assert.ok(Math.abs(val-centres[i][j])<1e-8)));
  // Lens reversal is physical, not a cosmetic UI flip. In ST UM3109
  // raw SPAD zone 0 (array bottom-left) observes scene top-right.
  const sceneToHardware=[3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12];
  assert.equal(sceneToHardware[3],0);
  assert.deepEqual(sceneToHardware.slice(12),[15,14,13,12]);
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
    sensor_model_revision:'vl53l8cx_array_plane_square45_below30_v3',
    tof4x4_mount_body_m:[0,0,-.03],
    tof4x4_rays_body:depthLabSensor.CURRENT_RAYS_BODY,
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


test('trained-policy Carpet uses exactly eight real 1m prism contacts',()=>{
  const envelope=terrainRoleEnvelope(measured(90_000_000),'step');
  const carpet=createSensorLabCarpet(envelope);
  const raw='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="5 5 .05"/></worldbody></mujoco>';
  const xml=withTerrain(raw,'carpet',{park:carpet});
  assert.equal((xml.match(/mocap="true"/g)||[]).length,8);
  assert.equal((xml.match(/sensor_lab_prism_\d+" type="mesh"/g)||[]).length,8);
  assert.doesNotMatch(xml,/name="floor"/);
  assert.doesNotMatch(xml,/viewer_carpet_solid_/);
  assert.equal(carpet.tiles.length,4);
  assert.equal(carpet.tileSizeM,1);
  assert.equal(carpet.physics,'eight-1m-prisms');

  // Real regression: original viewer started H40 at flat-floor qpos0
  // on a raised physical tile. Feet buried in the top collision surface
  // produced huge impulses and an immediate fall despite a working actor.
  const raisedSpawn=createSensorLabCarpet({slopeDeg:10,stepUpM:0,stepDownM:0});
  const geomPos=new Float64Array(9), geomSize=new Float64Array(9);
  geomPos.set([.5,.42,.018],3);   // right 18mm ball foot
  geomPos.set([.48,.58,.018],6);  // left ball foot slightly behind
  geomSize[3]=geomSize[6]=.018;
  const lift=carpetSpawnSoleLift(raisedSpawn,geomPos,geomSize,[1,2]);
  assert.ok(lift>.08,'a 10-degree tile needs a substantial positive spawn lift');
  for(const id of [1,2]){
    const i=id*3;
    const sole=geomPos[i+2]+lift-geomSize[i];
    const height=raisedSpawn.sampleHeight(geomPos[i],geomPos[i+1]);
    assert.ok(sole>=height-1e-9,'both soles must clear real terrain');
  }
  assert.ok(Math.abs(
    carpetSpawnSoleLift(createSensorLabCarpet({slopeDeg:0,stepUpM:0,stepDownM:0}),
      geomPos,geomSize,[1,2]))<1e-9,'flat fixture needs no lift');
  geomPos[3]=2.5;
  assert.throws(()=>carpetSpawnSoleLift(raisedSpawn,geomPos,geomSize,[1,2]),
    /valid prism support/,'no invisible floor outside the finite fixture');
});
test('15-degree slope preview is physically 15 degrees, not a 10-degree approximation',()=>{
  const slope=createSensorLabCarpet({slopeDeg:15,stepUpM:0,stepDownM:0});
  const delta=slope.sampleHeight(1.50,.50)-slope.sampleHeight(.50,.50);
  assert.ok(Math.abs(delta-Math.tan(Math.PI/12))<1e-6,delta);
  const left=slope.sampleHeight(1-1e-6,.50);
  const right=slope.sampleHeight(1+1e-6,.50);
  assert.ok(Math.abs(left-right)<1e-5,'no unrequested seam on a slope');
  const raised=createSensorLabCarpet({slopeDeg:0,stepUpM:.03,stepDownM:0});
  assert.ok(Math.abs(raised.sampleHeight(1+.001,.50)-
    raised.sampleHeight(1-.001,.50)-.03)<1e-7);
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
  assert.ok(Math.abs(clearanceAboveSupport(.42,.08)-.34)<1e-9);
  assert.equal(clearanceAboveSupport(.22,NaN),-Infinity,
    'carpet void must never masquerade as flat ground');
});

test('sensor lab is a fixed physical 2x2 grid: four 1m tiles, eight real triangular-prism meshes',()=>{
  const park=createSensorLabCarpet({slopeDeg:0,stepUpM:0,stepDownM:0});
  assert.equal(park.tiles.length,4);
  assert.equal(park.tileSizeM,1);
  assert.equal(park.sampleHeight(.5,.5),0);
  assert.ok(Number.isNaN(park.sampleHeight(2.5,.5)),'outside fixture is void');
  const prisms=sensorLabPrismVertices(park);
  assert.equal(prisms.length,8);
  assert.ok(prisms.every(p=>p.local.length===6));
  const xml=sensorLabPrismXml(park);
  assert.equal((xml.assets.match(/<mesh /g)||[]).length,8);
  assert.equal((xml.bodies.match(/type="mesh"/g)||[]).length,8);
  assert.equal((xml.bodies.match(/mocap="true"/g)||[]).length,8);
  const base='<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/></worldbody></mujoco>';
  const scene=withTerrain(base,'carpet',{park});
  assert.equal((scene.match(/type="mesh"/g)||[]).length,8);
  assert.doesNotMatch(scene,/name="floor"/);
  assert.doesNotMatch(scene,/viewer_carpet_solid_/);
});

test('physical h40 foot occlusion depends on mounting geometry, never a hardcoded zone mask',()=>{
  const R=[1,0,0,0,1,0,0,0,1];
  const pose=depthLabSensor.depthProbeRays([0,0,.4],R,
    {tilt:75,forward:0,height:.045});
  const feet=[[0,-.08,.018],[0,.08,.018]];
  const scan=depthLabSensor.readDepthProbe(pose,(origin,ray,max)=>{
    const ground=ray[2]<0?-origin[2]/ray[2]:Infinity;
    const foot=Math.min(...feet.map(centre=>
      depthLabSensor.footSphereHit(origin,ray,centre,.018)));
    if(foot<ground&&foot<=max)
      return {distance:foot,kind:'foot',surface:'foot'};
    return ground>0&&ground<=max
      ?{distance:ground,kind:'terrain',surface:'flat'}:Infinity;
  },{subrays:5});
  assert.deepEqual(scan.footOccluded.slice(12),[false,true,true,false]);
  assert.deepEqual(scan.footHitCounts.slice(12),[0,1,1,0]);
  assert.ok(scan.valid[12]&&scan.valid[15],
    'at the OLD high-mount position, corners see terrain');
  // The user's actual NEW physical mount is below the torso.
  // Do not keep reporting the former +45mm-high/75° success as though
  // it proved the real 70°/-30mm hardware arrangement.
  const chosen=depthLabSensor.depthProbeRays([0,0,.4],R,
    {tilt:70,forward:0,height:-.03,lateral:0,yaw:0});
  const actual=depthLabSensor.readDepthProbe(chosen,(origin,ray,max)=>{
    const terrain=ray[2]<0?-origin[2]/ray[2]:Infinity;
    const foot=Math.min(...feet.map(centre=>
      depthLabSensor.footSphereHit(origin,ray,centre,.018)));
    if(foot<terrain&&foot<=max)return {distance:foot,kind:'foot',surface:'foot'};
    return terrain>0&&terrain<=max
      ?{distance:terrain,kind:'terrain',surface:'flat'}:Infinity;
  },{subrays:5});
  assert.deepEqual(actual.footOccluded.slice(12),[true,false,false,true],
    'the chosen under-torso mount naturally grazes OUTER foot zones');
  assert.deepEqual(actual.footHitCounts.slice(12),[1,0,0,1]);
});
