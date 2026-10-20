// v5018 headless browser MuJoCo/WASM COLLISION smoke. No actor loaded.
// Real uneven grassy hfield, 80cm angled step bars, stepped platforms
// with tilted return ramps, and half-buried cylinder logs are genuine
// colliders. These static vertical ball probes do NOT certify six-servo
// crossing, motor forces, waypoint following, or real-browser visuals.
import loadMujoco from '@mujoco/mujoco';
import {fileURLToPath} from 'node:url';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {generateChallengeCourse,challengeProbe}
  from '../src/challenge-course.js';
import {withTerrain,generateTerrainPark} from '../src/terrain.js';

const wasmPath=fileURLToPath(
  new URL('../node_modules/@mujoco/mujoco/mujoco.wasm',import.meta.url));
const mj=await loadMujoco({
  locateFile:path=>path.endsWith('.wasm')?wasmPath:path,
});
const park=generateTerrainPark({
  maxDegrees:10,trees:true,challengeRing:true});
const hillLocation=[
  [7,0],[-7,0],[0,7],[0,-7],[6.4,3],[-6.4,3],[3,6.4],
].find(([x,y])=>park.sampleHeight(x,y)>.025 &&
  park.trees.every(t=>Math.hypot(x-t.x,y-t.y)>1.15));
if(!hillLocation)
  throw new Error('No clear natural-hill collision location');
const cases=[
  {tier:0,kind:'floor'},
  {tier:5,kind:'step'},
  {tier:10,kind:'step'},
  {tier:15,kind:'step'},
  {tier:20,kind:'step'},
  {tier:10,kind:'step_up',part:'top'},
  {tier:10,kind:'step_up',part:'ramp'},
  {tier:15,kind:'step_down',part:'top'},
  {tier:15,kind:'step_down',part:'ramp'},
  {tier:20,kind:'log',logs:true},
  {tier:10,kind:'hill'},
];
const report=[];
for(const fixture of cases){
  const course=generateChallengeCourse({
    tierMm:fixture.tier,logs:Boolean(fixture.logs)});
  const station=fixture.kind==='floor'?course.stations[0]
    :fixture.kind==='hill'?null
    :course.stations.find(s=>s.kind===fixture.kind);
  if(!station&&fixture.kind!=='hill')
    throw new Error('Missing physically collidable '+fixture.kind+' station');
  const probe=fixture.kind==='hill'
    ?{x:hillLocation[0],y:hillLocation[1],
      surfaceZ:park.sampleHeight(...hillLocation)}
    :fixture.kind==='floor'
      ?{x:station.x,y:station.y,
        surfaceZ:park.sampleHeight(station.x,station.y)}
      :challengeProbe(station,park.sampleHeight,fixture.part||'top');
  const groundHeight=park.sampleHeight(probe.x,probe.y);
  const baseZ=groundHeight+.16;
  const base=`<mujoco><option timestep=".005" gravity="0 0 -9.81" />
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .05" />
      <body name="test_sphere" pos="${probe.x} ${probe.y} ${baseZ}">
        <joint name="z_slide" type="slide" axis="0 0 1" />
        <geom name="ball" type="sphere" size=".018" mass=".01" />
      </body>
    </worldbody></mujoco>`;
  const xml=withTerrain(base,'course',{course,park});
  if(!xml.includes('viewer_park_hfield')||
     !xml.includes('viewer_park_ground')||
     !xml.includes('viewer_tree_trunk_0'))
    throw new Error('Combined physical challenge LOST ground/forest');
  let model=null,data=null;
  let result;
  try{
    model=mj.MjModel.from_xml_string(xml);
    if(!model)throw new Error('WASM rejected collidable mixed course XML');
    data=new mj.MjData(model);
    for(let i=0;i<300;i++)mj.mj_step(model,data);
    const observed=baseZ+Number(data.qpos[0]);
    const expected=probe.surfaceZ+.018;
    const error=Math.abs(observed-expected);
    const contactCount=Number(data.ncon??0);
    // The local park is intentionally UNEVEN; a heightfield triangle
    // plus a tiny angled/tilted collider may create a few mm discrepancy.
    // Report it independently. Never claim all errors are <1mm here.
    const tolerance=.008;
    result={
      ...fixture,station:station?.index??null,
      obstacle_height_mm:station?.heightMm??0,
      obstacle_yaw_offset_deg:station?.yawOffsetDeg??null,
      model_accepted:true,
      physical_park_height_m:groundHeight,
      retained_grass_hfield:true,
      retained_collision_trees:park.trees.length,
      sphere_centre_z_m:observed,
      nominal_surface_z_m:probe.surfaceZ,
      expected_sphere_centre_z_m:expected,
      absolute_height_disagreement_mm:error*1000,
      physical_contacts:contactCount,
      pass:Number.isFinite(observed)&&error<tolerance&&contactCount>0,
    };
  }catch(error){
    result={...fixture,station:station?.index??null,
      pass:false,error:String(error?.stack||error)};
  }finally{
    data?.delete?.();model?.delete?.();
  }
  report.push(result);
  console.log(JSON.stringify(result));
}
const filename=resolve(process.argv[2]||'v5018_wasm_challenge_results.json');
writeFileSync(filename,JSON.stringify({
  schema:2,study:'physical_random_hfield_mixed_angles_steps_ramps_half_buried_logs',
  mujoco_runtime:'@mujoco/mujoco 3.15.0',
  actor_or_checkpoint_loaded:false,
  actual_grass_hfield:true,physical_tree_trunks:park.trees.length,
  results:report,
  limitations:[
    'Constrained vertical sphere contact only; NOT trained six-servo crossing',
    'Nominal local height differs from geometric triangles on random hfield',
    'Static contact gap/height says nothing about healthy motor loading',
    'WASM 3.15.0 differs from Python/MJX MuJoCo 3.16.0',
    'Separate browser visual test required for waypoint-following and signage',
  ]
},null,2)+'\n');
console.log('Saved:',filename);
if(!report.every(r=>r.pass))process.exitCode=1;
