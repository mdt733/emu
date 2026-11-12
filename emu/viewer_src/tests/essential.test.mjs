import test from 'node:test';
import {readFileSync} from 'node:fs';
import {measuredAttitude,sampleEpisodeCadence,smoothFactor,chasePose,sampleShove,clearanceAboveSupport} from '../src/sim-math.js';
import assert from 'node:assert/strict';
import {parsePolicy} from '../src/policy.js';
import * as depthLabSensor from '../src/sensor.js';
import {joystickToCommands,nextYawCommand} from '../src/controls.js';
import {terrainRoleEnvelope} from '../src/training-solid-carpet.js';
import {createSensorLabCarpet,carpetTileCorners,carpetLocalLimits,carpetLatitudeDifficulty,carpetHeightfieldPatch,carpetSpawnSoleLift,streamingCarpetXml,LAB_TILE_COUNT} from '../src/sensor-lab-carpet.js';
import {meadowTreeForTile} from '../src/growing-meadow.js';
import {withTerrain,terrainSensorHeightFallback} from '../src/terrain.js';
import {trainingLimits} from '../src/training-carpet.js';
import {SPECIALIST_SCENES,makeSpecialistScene,sceneRole,
  specialistTileCorners,supportedFirstLip} from '../src/specialist-scenes.js';

test('viewer exposes Meadow and separate physical specialists without retired worlds',()=>{
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  assert.match(html,/<select id="sceneMode"/);
  for(const [mode] of SPECIALIST_SCENES)
    assert.match(html,new RegExp('option value="'+mode+'"'));
  assert.match(html,/id="specialistScenePanel"/);
  assert.match(html,/id="sceneHeight"/);
  assert.match(html,/id="sceneDistance"/);
  assert.match(html,/id="sceneApproach"/);
  assert.match(html,/id="sceneCommandAxis"/);
  assert.match(html,/Automatic Meadow stroll/);
  assert.doesNotMatch(html,/id="worldField"|id="fieldPanel"|id="navNext"/);
  assert.match(sim,/this\.terrainMode\s*=\s*'carpet'/);
  assert.doesNotMatch(sim,/generateTerrainPark|createParkMarkers|setParkFeaturesEnabled/);
  assert.doesNotMatch(main,/GoalNavigator|GoalPlannerWorker|setWorld\('park'\)|navWorker/);
  assert.match(main,/function meadowStroll\(/);
  assert.match(main,/sim\.setTerrainMode\('carpet'\)/);
  assert.match(main,/let lastResetGeneration\s*=\s*-1/);
  assert.match(main,/function updatePhysicalObservation\(/);
  assert.match(main,/function selectObservationWorld\(/);
  assert.match(sim,/setTrainingScenario\(/);
});

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5,pitch_abs_rad:.08};
const measured=(step,more={})=>({source:'training',seed:735,step,
  foundationSteps:20_000_000,academyIntroSteps:60_000_000,
  academyStream:true,hillPeakDeg:12,transitionDeg:3,stepUpM:.03,stepDownM:.04,...more});
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


test('viewer Carpet uses 225 real heightfields instead of changing compiled collision meshes',()=>{
  const park=createSensorLabCarpet({seed:735,hillPeakDeg:0,stepUpM:0,stepDownM:0});
  assert.equal(park.tiles.length,225);
  assert.equal(park.count(),225);
  assert.equal(park.physics,'streaming-heightfield');
  assert.equal(park.sampleHeight(.5,.5),0);
  const xml=streamingCarpetXml(park);
  assert.equal((xml.assets.match(/<hfield /g)||[]).length,225);
  assert.doesNotMatch(xml.assets,/elevation=/,
    'nonzero XML elevations are re-normalized by MuJoCo compiler, changing real slopes');
  assert.equal((xml.bodies.match(/type="hfield"/g)||[]).length,225);
  assert.equal((xml.bodies.match(/mocap="true"/g)||[]).length,225);
  const base='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="5 5 .05"/></worldbody></mujoco>';
  const scene=withTerrain(base,'carpet',{park});
  assert.equal((scene.match(/type="hfield"/g)||[]).length,225);
  assert.doesNotMatch(scene,/name="floor"/);
  assert.doesNotMatch(scene,/sensor_lab_prism_/);
  assert.ok(park.tiles.every(t=>carpetHeightfieldPatch(t)
     .every(x=>x>=0&&x<=1)));
});

test('infinite world: deterministic north difficulty and unlimited east/west revisits',()=>{
  const config={seed:735,hillPeakDeg:15,stepUpM:.04,stepDownM:.03};
  const easy=carpetLocalLimits(-30,config),base=carpetLocalLimits(0,config),
        hard=carpetLocalLimits(60,config);
  assert.ok(easy.difficulty<1);
  assert.equal(base.difficulty,1,'the slider means the actual starting latitude');
  assert.ok(hard.difficulty>1);
  assert.ok(easy.slopeDeg<base.slopeDeg&&base.slopeDeg<hard.slopeDeg);
  assert.ok(easy.stepUpM<base.stepUpM&&base.stepUpM<hard.stepUpM);
  const meanGrade=latitude=>{
    let sum=0,count=0;
    // Natural hills include flats, saddles and ridges. Verify that the
    // NORTH difficulty trend holds statistically over a broad landscape,
    // not that every arbitrary 37m strip north is steeper than one south.
    for(let y=latitude-3;y<=latitude+3;y++)for(let x=-90;x<=90;x++){
      const [a,b,c]=carpetTileCorners(x,y,config);
      sum+=Math.hypot(b-a,c-a);count++;
    }
    return sum/count;
  };
  const south=meanGrade(-25),north=meanGrade(35);
  assert.ok(north>south*2.2,
    'actual physical landscape must steepen statistically north: '+
    north.toFixed(4)+' vs '+south.toFixed(4));
  assert.notDeepEqual(carpetTileCorners(5,35,config),
    carpetTileCorners(6,35,config),
    'east/west must vary terrain without changing the latitude envelope');
  assert.deepEqual(carpetLocalLimits(50,config),carpetLocalLimits(50,config));
  const park=createSensorLabCarpet(config);
  const origin=carpetTileCorners(0,0,config);
  const visit=(x,y)=>park.growAround(x+.5,y+.5,[
    {x:x+.44,y:y+.44,radius:.018},{x:x+.55,y:y+.55,radius:.018}]);
  for(let x=1;x<=80;x++)assert.equal(visit(x,0).length,15,
    'every eastward crossing reuses the exhausted western column');
  assert.deepEqual(park.at(80,0).corners,carpetTileCorners(80,0,config));
  for(let x=79;x>=0;x--)visit(x,0);
  assert.deepEqual(park.at(0,0).corners,origin,
    'revisiting an old coordinate produces exactly the old terrain');
  for(let y=1;y<=60;y++)visit(0,y);
  assert.deepEqual(park.at(0,60).corners,carpetTileCorners(0,60,config));
  for(let y=59;y>=-40;y--)visit(0,y);
  assert.deepEqual(park.at(0,-40).corners,carpetTileCorners(0,-40,config));
  assert.equal(park.tiles.length,LAB_TILE_COUNT,'no unbounded collider growth');
  assert.equal(park.count(),225,'fixed 15x15 active physical window');
  assert.ok(Number.isNaN(park.sampleHeight(10.5,10.5)),
    'unloaded distant terrain is void to physical sensors');
  assert.ok(Number.isFinite(park.previewHeight(10.5,10.5)),
    'visual fallaway may extrapolate but never supplies ToF ground');
  assert.equal(carpetLatitudeDifficulty(-100),.08);
  assert.equal(carpetLatitudeDifficulty(1000),1.8);
});

test('10-degree Meadow produces organic terrain with regular near-10-degree grades',()=>{
  const cfg={seed:735,hillPeakDeg:10,hillScaleM:9.5,
    stepUpM:0,stepDownM:0,stepFrequency:0,northGainPerM:0};
  const degrees=[],headings=[];
  // Sample actual *physical tile triangle* grades at +Y=0 latitude
  // across multiple seeds, x regions, and gently varying nearby y.
  for(let y=-7;y<=7;y++)for(let x=-40;x<=40;x++){
    const [a,b,c,d]=carpetTileCorners(x,y,cfg);
    const gx=b-a,gy=c-a;
    const angle=Math.atan(Math.hypot(gx,gy))*180/Math.PI;
    degrees.push(angle);
    if(angle>3)headings.push(Math.atan2(gy,gx));
    const diagonal=(b+c)*.5;
    const fromLower=a+(b-a)*.5+(c-a)*.5;
    const fromUpper=d+(c-d)*.5+(b-d)*.5;
    assert.ok(Math.abs(diagonal-fromLower)<1e-9 &&
              Math.abs(diagonal-fromUpper)<1e-9,
      'both physical triangles must meet exactly along their diagonal');
    const right=carpetTileCorners(x+1,y,cfg);
    assert.ok(Math.abs(b-right[0])<1e-9 &&
      Math.abs(d-right[2])<1e-9,'adjacent natural hills share exact edges');
  }
  degrees.sort((a,b)=>a-b);
  const pct=q=>degrees[Math.floor(q*(degrees.length-1))];
  assert.ok(pct(.9)>=8.5&&pct(.9)<=12,
    '10° means roughly 90th-percentile hill slope around 10°, got '+pct(.9));
  assert.ok(pct(.5)>=4&&pct(.5)<9,
    'natural hills must alternate gentle ground and challenges');
  assert.ok(pct(.25)<pct(.9)*.65,'not a uniform 10-degree plane');
  assert.ok(new Set(headings.map(a=>Math.round(a*3))).size>=8,
    'hills must slope in changing directions, not one constant grade');
  const flat={...cfg,hillPeakDeg:0};
  for(let i=-5;i<=5;i++)assert.deepEqual(
    carpetTileCorners(i,0,flat),[0,0,0,0],
    'zero hills and zero steps produce truly flat terrain');
  const north={...cfg,northGainPerM:.022};
  let down=0,up=0;
  for(let x=-20;x<=20;x++){
    const [a,b,c]=carpetTileCorners(x,-20,north),
      [d,e,f]=carpetTileCorners(x,20,north);
    down+=Math.hypot(b-a,c-a);
    up+=Math.hypot(e-d,f-d);
  }
  assert.ok(up>down*2,'north must materially increase actual hill angles');
});

test('streaming Carpet refuses to remove any tile supporting an H40 foot',()=>{
  const park=createSensorLabCarpet({seed:735,hillPeakDeg:10,stepUpM:.025,stepDownM:.03});
  const tile=park.at(-7,0),xy=[tile.i,tile.j],corners=tile.corners.slice();
  const blocked=park.growAround(1.5,.5,[
    {x:-6.52,y:.50,radius:.018},{x:1.54,y:.57,radius:.018}]);
  assert.equal(blocked.length,0,'rear support may never be teleported');
  assert.deepEqual([tile.i,tile.j],xy);
  assert.deepEqual(tile.corners,corners);
  const moved=park.growAround(1.5,.5,[
    {x:1.44,y:.50,radius:.018},{x:1.54,y:.57,radius:.018}]);
  assert.equal(moved.length,15);
  assert.equal(park.at(-7,0),null);
  assert.equal(park.at(8,0).slot,tile.slot,
    'exact same compiled slot has moved fifteen metres');
  assert.deepEqual(park.at(8,0).corners,carpetTileCorners(8,0,park.limits));
  assert.deepEqual(meadowTreeForTile({i:8,j:0},735),
    meadowTreeForTile({i:8,j:0},735),
    'tree positions do not depend on which collider slot hosts a tile');
  park.resetGrowth();
  assert.deepEqual(park.at(-7,0).corners,corners);
});

test('streaming ground is still visible to the frozen H40 4x4 ToF and optical flow',()=>{
  const carpet=createSensorLabCarpet({hillPeakDeg:0,stepUpM:0,stepDownM:0});
  const surface=terrainSensorHeightFallback('carpet',carpet);
  const meshModel={ngeom:1,geom_type:new Int32Array([7]),
    geom_bodyid:new Int32Array([0]),geom_size:new Float64Array(3)};
  const meshData={geom_xpos:new Float64Array(3),
    geom_xmat:new Float64Array(9)};
  const origin=[.5,.5,.4],down=[0,0,-1];
  assert.equal(depthLabSensor.actorTerrainDistance(
    origin,down,meshModel,meshData,[],null),0);
  assert.ok(Math.abs(depthLabSensor.actorTerrainDistance(
    origin,down,meshModel,meshData,[],surface)-.4)<1e-5);
  const contract={
    schema:'terrain_360',sensor_model_revision:'vl53l8cx_array_plane_square45_below30_v3',
    tof4x4_mount_body_m:[0,0,-.03],
    tof4x4_rays_body:depthLabSensor.CURRENT_RAYS_BODY,
    settings:{sensor_noise_scale:0},
  };
  const pose={R:[1,0,0,0,1,0,0,0,1],torso:origin,
    terrainHeightAt:surface,vbody:[.20,0,0],gyro:[0,0,0]};
  const tof=new depthLabSensor.TerrainActorObserver(contract)
    .sample(pose,meshModel,meshData,[],false);
  assert.ok(tof.valid.filter(Boolean).length>=6);
  const flow=new depthLabSensor.ArrayFlowObserver(contract)
    .synth(pose,meshModel,meshData,false);
  assert.ok(flow.flowValid&&Math.abs(flow.flow[0])>.1);
  assert.equal(depthLabSensor.rayHeight(
    [7.95,.5,.4],[.5,0,-Math.sqrt(.75)],surface),0,
    'after the active physical fringe, no made-up ground exists');
});

test('real MuJoCo browser WASM reuses an hfield collider without resetting H40',async()=>{
  const {default:loadMujoco}=await import('@mujoco/mujoco');
  const mj=await loadMujoco();
  const actor=JSON.parse(readFileSync(
    new URL('../../../showcase/champion.emupolicy',import.meta.url),'utf8'));
  const park=createSensorLabCarpet({hillPeakDeg:0,stepUpM:0,stepDownM:0});
  const model=mj.MjModel.from_xml_string(
    withTerrain(actor.robot_xml,'carpet',{park}));
  const data=new mj.MjData(model);
  try{
    assert.equal(Number(model.nhfield),225);
    assert.equal(Number(model.nmocap),225);
    assert.ok(model.hfield_data.length>=900);
    assert.ok(model.hfield_data.every(v=>v===0),
      'source XML must compile zero-filled hfields for exact metric runtime writes');
    for(const tile of park.tiles){
      const off=Number(model.hfield_adr[tile.slot]);
      const p=carpetHeightfieldPatch(tile);
      for(let i=0;i<4;i++)model.hfield_data[off+i]=p[i];
      const body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,
        'sensor_lab_tile_'+tile.slot);
      const mid=Number(model.body_mocapid[body]);
      data.mocap_pos[3*mid]=tile.x;
      data.mocap_pos[3*mid+1]=tile.y;
      data.mocap_pos[3*mid+2]=tile.base;
    }
    data.qpos[0]=data.qpos[1]=.5;
    data.qpos[2]-=.004;
    mj.mj_forward(model,data);
    assert.ok(data.ncon>=1,'the first 1m patch must support the robot');
    const body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,
      'sensor_lab_tile_0');
    const mocap=Number(model.body_mocapid[body]);
    const savedZ=data.mocap_pos[3*mocap+2];
    data.mocap_pos[3*mocap+2]=-30;
    mj.mj_forward(model,data);
    assert.equal(data.ncon,0,'evicted hfield must not be a ghost floor');
    const offset=Number(model.hfield_adr[0]);
    model.hfield_data[offset]=.02;
    model.hfield_data[offset+1]=.04;
    model.hfield_data[offset+2]=.02;
    model.hfield_data[offset+3]=.04;
    data.mocap_pos[3*mocap+2]=savedZ;
    mj.mj_forward(model,data);
    assert.ok(data.ncon>=1,'recycled in-WASM hfield must restore real contact');
  }finally{data.delete();model.delete();}
},{timeout:120000});

test('streaming Carpet places neutral H40 ball feet above actual terrain',()=>{
  const p=createSensorLabCarpet({hillPeakDeg:12,stepUpM:.02,stepDownM:0});
  const xyz=new Float64Array(9),size=new Float64Array(9);
  xyz.set([.5,.42,.018],3);xyz.set([.48,.58,.018],6);
  size[3]=size[6]=.018;
  const lift=carpetSpawnSoleLift(p,xyz,size,[1,2]);
  assert.ok(Number.isFinite(lift));
  for(const id of [1,2]){
    const i=3*id;
    const t=p.at(Math.floor(xyz[i]),Math.floor(xyz[i+1]));
    const slopeX=t.corners[1]-t.corners[0],slopeY=t.corners[2]-t.corners[0];
    const tangent=xyz[i+2]+lift-size[i]*Math.hypot(1,slopeX,slopeY);
    assert.ok(tangent>=p.sampleHeight(xyz[i],xyz[i+1])-1e-9);
  }
  xyz[3]=18.5;
  assert.throws(()=>carpetSpawnSoleLift(p,xyz,size,[1,2]),
    /valid prism support/);
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


test('specialist training carpet is REAL mutable signed terrain and 22cm first-lip spawn',()=>{
  const up=makeSpecialistScene('step-up',{stepCm:5,spawnCm:22,randomize:false});
  assert.equal(up.kind,'step');
  assert.equal(up.distanceM,.22);
  assert.equal(up.edgeX,1);
  assert.equal(up.lipM,.05);
  assert.equal(up.approachDeg,0);
  assert.equal(makeSpecialistScene('step-up',{
    stepCm:5,approachDeg:9,randomize:false}).approachDeg,9);
  assert.equal(sceneRole('step-up'),'step');
  assert.deepEqual(specialistTileCorners(0,0,up),[0,0,0,0]);
  assert.deepEqual(specialistTileCorners(1,0,up),[.05,.05,.05,.05]);
  const park=createSensorLabCarpet({
    seed:735,hillPeakDeg:0,stepUpM:0,stepDownM:0,scenario:up});
  assert.equal(park.physics,'streaming-heightfield');
  assert.equal(park.tiles.length,LAB_TILE_COUNT);
  assert.ok(Math.abs(park.sampleHeight(1.1,.5)-
    park.sampleHeight(.9,.5)-.05)<1e-9);
  assert.equal(park.sampleHeight(1.1,.5),.05);
  assert.ok(park.tiles.every(t=>carpetHeightfieldPatch(t)
    .every(h=>Number.isFinite(h)&&h>=0&&h<=1)));
  const down=makeSpecialistScene('step-down',{stepCm:2,randomize:false});
  park.configureScenario(down);
  park.resetGrowth();
  assert.equal(park.sampleHeight(.8,.5),0);
  assert.equal(park.sampleHeight(1.2,.5),-.02);
  assert.equal(park.at(1,0).base,-.02);
  const fixed=makeSpecialistScene('step-up',{stepCm:5,trial:3,randomize:false});
  assert.equal(fixed.distanceM,.22);
  const varied=Array.from({length:10},(_,trial)=>makeSpecialistScene('step-up',{
    stepCm:5,trial,seed:735,randomize:true}));
  assert.ok(new Set(varied.map(v=>v.distanceM.toFixed(4))).size>1);
  assert.ok(varied.every(v=>v.distanceM>=.15&&v.distanceM<=.30));
  assert.ok(varied.every(v=>v.lipM>=.0475&&v.lipM<=.0525));
  assert.ok(varied.every(v=>v.approachDeg>=-3&&v.approachDeg<=3));
  assert.ok(varied.every(v=>v.sideOffsetM>=-.05&&v.sideOffsetM<=.05));
  assert.ok(new Set(varied.map(v=>v.approachDeg.toFixed(3))).size>1);
  assert.throws(()=>makeSpecialistScene('invented'),/Unknown/);
});

test('cross slopes are physical, continuous and opposite in sign',()=>{
  const left=makeSpecialistScene('slope-left',{gradeDeg:10,randomize:false});
  const right=makeSpecialistScene('slope-right',{gradeDeg:10,randomize:false});
  const climb=makeSpecialistScene('slope-up',{gradeDeg:10,randomize:false});
  const descend=makeSpecialistScene('slope-down',{gradeDeg:10,randomize:false});
  assert.equal(left.axis,'y');
  assert.ok(specialistTileCorners(0,0,left)[2]>0);
  assert.ok(specialistTileCorners(0,0,right)[2]<0);
  assert.ok(specialistTileCorners(0,0,climb)[1]>0);
  assert.ok(specialistTileCorners(0,0,descend)[1]<0);
  assert.equal(specialistTileCorners(0,0,climb)[1],
    specialistTileCorners(1,0,climb)[0]);
});

test('first lip completion requires both real soles on the new physical surface',()=>{
  const scenario=makeSpecialistScene('step-down',{stepCm:2,randomize:false});
  const feet=[
    {x:1.12,soleZ:-.02,groundZ:-.02},
    {x:1.13,soleZ:-.02,groundZ:-.02}
  ];
  const args={scenario,bodyX:1.12,feet,fallen:false};
  assert.equal(supportedFirstLip(args),true);
  assert.equal(supportedFirstLip({...args,bodyX:.98}),false);
  assert.equal(supportedFirstLip({...args,fallen:true}),false);
  assert.equal(supportedFirstLip({...args,feet:[
    feet[0],{...feet[1],x:.98}]}),false);
  assert.equal(supportedFirstLip({...args,feet:[
    feet[0],{...feet[1],soleZ:.08}]}),false);
});
