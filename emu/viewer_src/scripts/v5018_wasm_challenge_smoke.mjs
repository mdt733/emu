// v5018 browser MuJoCo/WASM physical obstacle-ring smoke.
// Collision is a real static MuJoCo worldbody box/capsule. A constrained
// 18mm sphere must land at the correct raised height; no actor is loaded.
// This does NOT validate the full six-servo actor, course completion or
// rendering and is intentionally separate from original CPU/MJX physics.
import loadMujoco from '@mujoco/mujoco';
import {fileURLToPath} from 'node:url';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {generateChallengeCourse} from '../src/challenge-course.js';
import {withTerrain} from '../src/terrain.js';

const wasmPath=fileURLToPath(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm',import.meta.url));
const mj=await loadMujoco({locateFile:path=>path.endsWith('.wasm')?wasmPath:path});
const report=[];
const cases=[
  {tier:0,kind:'floor'},
  {tier:5,kind:'step'},
  {tier:10,kind:'step'},
  {tier:15,kind:'step'},
  {tier:20,kind:'step'},
  {tier:10,kind:'log',logs:true},
];
for(const fixture of cases){
  const course=generateChallengeCourse({
    tierMm:fixture.tier,logs:Boolean(fixture.logs)});
  const station=fixture.kind==='floor'?course.stations[0]
    :fixture.kind==='log'?course.stations[3]
    :course.stations.find(s=>s.kind==='step'&&s.heightMm===fixture.tier);
  if(!station)throw new Error('Missing selected physical obstacle station');
  const x=station.x,y=station.y;
  const base=`<mujoco><option timestep=".005" gravity="0 0 -9.81" />
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .05" />
      <body name="test_sphere" pos="${x} ${y} .12">
        <joint name="z_slide" type="slide" axis="0 0 1" />
        <geom name="ball" type="sphere" size=".018" mass=".01" />
      </body>
    </worldbody></mujoco>`;
  const xml=withTerrain(base,'course',{course});
  let model=null,data=null;
  let result;
  try{
    model=mj.MjModel.from_xml_string(xml);
    if(!model)throw new Error('WASM did not compile collidable challenge XML');
    data=new mj.MjData(model);
    for(let i=0;i<260;i++)mj.mj_step(model,data);
    const observed=.12+Number(data.qpos[0]);
    const expected=.018+(fixture.kind==='floor'?0:
      fixture.kind==='log' ? .020 : fixture.tier/1000);
    const mismatch=Math.abs(observed-expected);
    result={...fixture,station:station.index,
      model_accepted:true,
      final_sphere_centre_z_m:observed,
      expected_sphere_centre_z_m:expected,
      absolute_height_disagreement_mm:mismatch*1000,
      physical_contacts:Number(data.ncon??0),
      pass:Number.isFinite(observed)&&mismatch<.004};
  }catch(error){
    result={...fixture,station:station.index,
      pass:false,error:String(error?.stack||error)};
  }finally{data?.delete?.();model?.delete?.();}
  report.push(result);
  console.log(JSON.stringify(result));
}
const filename=resolve(process.argv[2]||'v5018_wasm_challenge_results.json');
writeFileSync(filename,JSON.stringify({
  schema:1,study:'real_webassembly_step_and_log_collidable_ring',
  mujoco_runtime:'@mujoco/mujoco 3.15.0',
  actor_or_checkpoint_loaded:false,
  results:report,
  limitations:[
    'Vertical constrained sphere, not a dynamic six-servo crossing',
    'No measured foot-contact normal impulse, no CPU/MJX force parity',
    'Browser 3.15.0 differs from Python MuJoCo 3.16.0',
  ]
},null,2)+'\n');
console.log('Saved:',filename);
if(!report.every(r=>r.pass))process.exitCode=1;
