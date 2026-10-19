// P0B preliminary CPU MuJoCo 3.16 vs pinned browser MuJoCo/WASM 3.15
// Headless Emscripten build, same collision XML and 200 physical time steps.
// This is NOT a browser-rendered full-robot policy rollout.
import loadMujoco from '@mujoco/mujoco';
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';

const baselinePath=process.argv[2];
if(!baselinePath)throw new Error('Expected CPU/MJX result JSON path');
const baseline=JSON.parse(readFileSync(baselinePath,'utf8'));
const expectedRows=Array.isArray(baseline.cases)?baseline.cases:baseline;
const cases={
  flat:{terrain:'<geom name="ground" type="plane" size="2 2 .1"/>',x:0},
  step_up:{terrain:'<geom name="ground" type="plane" size="2 2 .1"/><geom name="obstacle" type="box" pos="0 0 .015" size=".15 .3 .015"/>',x:0},
  step_down:{terrain:'<geom name="lower" type="plane" pos="0 0 -.03" size="2 2 .1"/><geom name="upper" type="box" pos="-.22 0 -.015" size=".22 .4 .015"/>',x:.12},
  diagonal:{terrain:'<geom name="lower" type="plane" pos="0 0 -.12" size="2 2 .1"/><geom name="ramp" type="box" pos="0 0 -.005" euler="8 8 12" size=".25 .30 .02"/>',x:0},
  log:{terrain:'<geom name="ground" type="plane" size="2 2 .1"/><geom name="log" type="capsule" fromto="0 -.25 .018 0 .25 .018" size=".012"/>',x:0}
};
const wasmPath=fileURLToPath(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm',import.meta.url));
const mj=await loadMujoco({locateFile:filename=>filename.endsWith('.wasm')?wasmPath:filename});
const rows=[];
for(const [name,c] of Object.entries(cases)){
  const xml=`<mujoco><option timestep=".005" gravity="0 0 -9.81"/>
    <worldbody>${c.terrain}<body name="foot" pos="${c.x} 0 .08">
    <joint name="z_slide" type="slide" axis="0 0 1"/>
    <geom name="ball" type="sphere" size=".018" mass=".01"/>
    </body></worldbody></mujoco>`;
  const reference=expectedRows.find(r=>r.case===name);
  let model=null,data=null,row;
  try{
    model=mj.MjModel.from_xml_string(xml);
    if(!model)throw new Error('MjModel.from_xml_string returned no model');
    data=new mj.MjData(model);
    for(let i=0;i<200;i++)mj.mj_step(model,data);
    const z=.08+Number(data.qpos[0]);
    const cpuZ=Number(reference?.cpu_z);
    const discrepancyMm=1000*Math.abs(z-cpuZ);
    row={case:name,wasm_z_m:z,cpu_z_m:cpuZ,
         cpu_wasm_height_disagreement_mm:discrepancyMm,
         wasm_ncon:Number(data.ncon??-1),
         pass:Number.isFinite(z)&&Number.isFinite(cpuZ)&&discrepancyMm<5};
  }catch(err){
    row={case:name,pass:false,error:String(err.stack||err)};
  }finally{
    data?.delete?.();model?.delete?.();
  }
  rows.push(row);
  console.log(JSON.stringify(row));
}
const out=resolve('v5018_wasm_contact_gap.json');
writeFileSync(out,JSON.stringify({
  runtime:'@mujoco/mujoco 3.15.0 WASM via Node',
  comparison:'same vertical-slider sphere XML, 200 steps, CPU MuJoCo 3.16.0 result',
  rows,limitations:[
    'Headless WASM collision, not full browser plus Three.js or six-servo rollout',
    'Final resting height only; not contact-force or trajectory parity',
    'Differences confounded by MuJoCo 3.15 vs 3.16 version difference'
  ]
},null,2)+'\n');
console.log('Results:',out);
if(!rows.every(row=>row.pass))process.exitCode=1;
