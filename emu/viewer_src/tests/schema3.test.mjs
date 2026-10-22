import test from 'node:test';
import assert from 'node:assert/strict';

import { parsePolicy } from '../src/policy.js';
import { RollingTerrainDiagnostic, V_SAMPLES } from '../src/terrain-memory.js';
import { appendPreviewObservation } from '../src/preview-observation.js';
import { SensorObserver } from '../src/sensor.js';

const pos=[-5,0,5,10,20];
const terrainFields=['right','left'].flatMap(side =>
  pos.map(cm=>`terrain_height_${side}_${cm}cm_m`));
const confidenceFields=['right','left'].flatMap(side =>
  pos.map(cm=>`terrain_confidence_${side}_${cm}cm`));

function fixture(schema=3){
  const base=schema===3?78:schema===2?58:55;
  const fields=Array.from({length:55},(_,i)=>`legacy_${i}`);
  if(schema>=2)fields.push(
    'ground_forward_grade_rad','ground_sideways_grade_rad',
    'ground_visibility_confidence');
  if(schema===3)fields.push(...terrainFields,...confidenceFields);
  const sensor=schema===1?null:{
    schema:schema===3?'rolling_tof_312':'sim2real_232',
    sensor_model_revision:'tripod_tof_flow_async_v1',
    history_len:4,action_dim:6,base_fields:fields,
    tof_range_limits_m:[.035,.6],
    tripod_mounts_body_m:[[.035,-.05,-.025],[.035,.05,-.025],[-.04,0,-.025]],
    tripod_beams_body:[[0,0,-1],[0,0,-1],[0,0,-1]],
    settings:{
      tof_period_frames:2,flow_period_frames:1,
      tof_delay_frames:1,flow_delay_frames:1,
      tof_dropout_prob:.01,flow_dropout_prob:.02,
      gyro_bias_std_rad_s:.015,sensor_noise_scale:1
    }
  };
  if(schema===3)Object.assign(sensor,{
    rolling_cells_per_lane:51,rolling_spacing_m:.01,rolling_start_m:-.10,
    rolling_actor_indices:[5,10,15,20,30],
    rolling_actor_sample_x_m:[-.05,0,.05,.1,.2],
    rolling_model_revision:'rear_2d_apex_tof_v_tracks_51x2_v7_actor20cm',
    forward_model_revision:'two_far_rays_three_box_aware_tripod_and_camera_flow_depth_v6',
    forward_masked:false
  });
  return {format:'emu-policy',schema,
    policy:{action_dim:6,obs_dim:base*4,base_obs_dim:base,
      history_len:4,layers:[{},{},{}]},
    observation:{base_fields:fields,sensor_contract:sensor},
    source:{checkpoint_schema:schema===3?9:schema===2?8:7},
    control:{},robot_xml:'<mujoco />'};
}
const parse=p=>parsePolicy(JSON.stringify(p));

test('viewer parser accepts real 312x4 schema-3 contract and legacy 220/232',()=>{
  assert.equal(parse(fixture(3)).policy.obs_dim,312);
  assert.equal(parse(fixture(2)).policy.obs_dim,232);
  assert.equal(parse(fixture(1)).policy.obs_dim,220);
});

test('rejects schema-3 actors with fake 232 channels, wrong sensor layout or wrong checkpoint',()=>{
  for(const mutate of [
    p=>p.policy.obs_dim=232,
    p=>p.policy.base_obs_dim=58,
    p=>p.source.checkpoint_schema=8,
    p=>p.observation.base_fields[77]='terrain_confidence_left_15cm',
    p=>p.observation.sensor_contract.rolling_actor_indices[4]=25,
    p=>p.observation.sensor_contract.forward_model_revision='unknown',
    p=>p.observation.sensor_contract.schema='sim2real_232'
  ]){
    const p=fixture();mutate(p);
    assert.throws(()=>parse(p));
  }
});

test('schema-3 shares active 232 tripod / optical-flow sensor model',()=>{
  const obs=new SensorObserver(fixture(3).observation.sensor_contract);
  assert.equal(obs.settings.tof_period_frames,2);
});

test('V actor readout uses exact right/left -5/0/5/10/20cm sampling positions',()=>{
  const map=new RollingTerrainDiagnostic();
  assert.deepEqual(V_SAMPLES,[5,10,15,20,30]);
  for(let side=0;side<2;side++)for(let i=0;i<51;i++){
    map.heights[side][i]=side*100+i;
    map.confidence[side][i]=.5;
  }
  const r=map.actorReadout();
  assert.deepEqual(r.heights,[5,10,15,20,30,105,110,115,120,130]);
  assert.equal(r.confidence.length,10);
  const frame=new Float32Array(78);
  frame.fill(7,0,58);
  assert.equal(appendPreviewObservation(frame,58,map),78);
  assert.deepEqual(Array.from(frame.slice(58,68)),r.heights);
  assert.deepEqual(Array.from(frame.slice(68,78)),r.confidence);
  assert.equal(frame[57],7);
});

test('312 masked preview zeroes only its 20 sensor slots',()=>{
  const map=new RollingTerrainDiagnostic();
  map.heights[0][5]=.03;
  map.confidence[0][5]=.9;
  const frame=new Float32Array(78);frame.fill(2);
  appendPreviewObservation(frame,58,map,true);
  assert.equal(frame[57],2);
  assert.deepEqual(Array.from(frame.slice(58)),Array(20).fill(0));
  assert.throws(()=>appendPreviewObservation(new Float32Array(58),58,map));
  assert.throws(()=>appendPreviewObservation(new Float32Array(78),55,map));
});
