import {depthProbeRays} from './sensor.js';
const IDENTITY_ROTATION=[1,0,0,0,1,0,0,0,1];
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

export function decodeF32(blob) {
  if (!blob || blob.dtype !== 'float32-le' || !Array.isArray(blob.shape) || typeof blob.data_b64 !== 'string') {
    throw new Error('Invalid float32 blob');
  }
  const bin = atob(blob.data_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const expected = blob.shape.reduce((a, b) => a * b, 1);
  if (bytes.byteLength !== expected * 4) throw new Error(`Float blob byte size mismatch: got ${bytes.byteLength}, expected ${expected * 4}`);
  const src = new Float32Array(bytes.buffer);
  // Current .emupolicy float blobs are little-endian. Browsers we target
  // are little-endian; use DataView fallback if that ever stops being true.
  return new Float32Array(src);
}

export function parsePolicy(text) {
  // ONE active policy: 360 inputs, 6 outputs and the physical ST
  // 45° 4x4 sensor. Old schemas/ray layouts belong to Git history.
  const b=JSON.parse(text);
  const a=b?.policy,c=b?.observation?.sensor_contract;
  const lab=b?.sensor_lab===true;
  const f=b?.observation?.base_fields;
  if(b?.format!=='emu-policy'||b?.schema!==3||!a||!c||
     !b.control||typeof b.robot_xml!=='string'||
     a.action_dim!==6||a.history_len!==4||
     a.base_obs_dim!==90||a.obs_dim!==360||
     b.source?.checkpoint_schema!==9)
    throw Error('Only the current 360-input EMU robot is supported');
  if(!Array.isArray(a.layers)||(lab?a.layers.length!==0:a.layers.length!==3))
    throw Error(lab?'Sensor laboratory cannot carry an actor':'Expected three trained actor layers');
  if(!Array.isArray(f)||f.length!==90||
     JSON.stringify(f.slice(55,58))!==JSON.stringify([
       'ground_forward_grade_rad','ground_sideways_grade_rad',
       'ground_visibility_confidence'
     ])||
     !f.slice(58,74).every((name,i)=>name===
       `tof4x4_range_${Math.floor(i/4)}${i%4}_m`)||
     !f.slice(74,90).every((name,i)=>name===
       `tof4x4_valid_${Math.floor(i/4)}${i%4}`))
    throw Error('Current 360-input observation field order required');
  if(c.schema!=='terrain_360'||c.sensor_model_revision!=='vl53l8cx_array_plane_square45_below30_v3'||
     c.history_len!==4||c.action_dim!==6||
     JSON.stringify(c.base_fields)!==JSON.stringify(f)||
     c.array_ground_estimator!=='weighted_4x4_plane_residual_8mm_v1'||
     c.array_ground_max_fit_residual_m!==.008||
     c.array_ground_min_valid_cells!==6)
    throw Error('Only current physical 4x4 array contract is supported');
  const canonical=depthProbeRays([0,0,0],IDENTITY_ROTATION,
    {tilt:70,forward:0,height:-.03,lateral:0,yaw:0});
  const expected=canonical.directions.map(zone=>zone[0]);
  const actual=c.tof4x4_rays_body;
  if(!Array.isArray(actual)||actual.length!==16||
     actual.some((ray,i)=>!Array.isArray(ray)||ray.length!==3||
       ray.some((n,j)=>!Number.isFinite(n)||Math.abs(n-expected[i][j])>1e-6))||
     JSON.stringify(c.tof4x4_mount_body_m)!==JSON.stringify([0,0,-.03])||
     c.tof4x4_fov_horizontal_deg!==45||
     c.tof4x4_fov_vertical_deg!==45||
     c.tof4x4_tilt_down_deg!==70||
     c.tof4x4_mount_yaw_deg!==0||
     JSON.stringify(c.tof4x4_hardware_zone_ids)!==
       JSON.stringify([3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12])||
     JSON.stringify(c.tof4x4_range_limits_m)!==JSON.stringify([.04,1.5])||
     c.tof4x4_occlusion!=='nearest ray-sphere foot hit masks terrain return'||
     c.tof4x4_validity!=='one bit per cell, delayed, zero on dropout')
    throw Error('Policy rays, mounting or zone order differ from the h40 hardware');
  const settings=c.settings||{};
  for(const key of ['tof_period_frames','flow_period_frames',
    'tof_delay_frames','flow_delay_frames','tof_dropout_prob',
    'flow_dropout_prob','gyro_bias_std_rad_s','sensor_noise_scale']){
    if(!Number.isFinite(settings[key]))
      throw Error('Invalid sensor setting: '+key);
  }
  if(lab && (b.source?.step!==0||b.source?.actor_sha256!=='not-applicable-sensor-lab'))
    throw Error('Sensor laboratory must be an untrained fixture');
  return b;
}

class DenseLayer {
  constructor(spec) {
    this.w = decodeF32(spec.w);
    this.b = decodeF32(spec.b);
    this.inDim = spec.w.shape[0];
    this.outDim = spec.w.shape[1];
    if (this.w.length !== this.inDim * this.outDim || this.b.length !== this.outDim) throw new Error('Dense layer shape mismatch');
    this.out = new Float32Array(this.outDim);
  }

  forward(x, tanh = false) {
    const out = this.out;
    const w = this.w;
    const b = this.b;
    const ni = this.inDim;
    const no = this.outDim;
    for (let j = 0; j < no; j++) out[j] = b[j];
    for (let i = 0; i < ni; i++) {
      const xi = x[i];
      const row = i * no;
      for (let j = 0; j < no; j++) out[j] += xi * w[row + j];
    }
    if (tanh) for (let j = 0; j < no; j++) out[j] = Math.tanh(out[j]);
    return out;
  }
}

export class PolicyRunner {
  constructor(bundle) {
    this.bundle = bundle;
    this.obsDim = bundle.policy.obs_dim;
    this.actionDim = bundle.policy.action_dim;
    this.mean = decodeF32(bundle.observation.mean);
    this.std = decodeF32(bundle.observation.std);
    this.mask = decodeF32(bundle.observation.normalize_mask);
    this.layers = bundle.policy.layers.map(x => new DenseLayer(x));
    this.normalized = new Float32Array(this.obsDim);
    this.action = new Float32Array(this.actionDim);
    if (this.mean.length !== this.obsDim || this.std.length !== this.obsDim || this.mask.length !== this.obsDim) {
      throw new Error('Observation normalizer length mismatch');
    }
  }

  infer(obs) {
    if (obs.length !== this.obsDim) throw new Error(`Observation length ${obs.length} != ${this.obsDim}`);
    const z = this.normalized;
    for (let i = 0; i < this.obsDim; i++) {
      if (this.mask[i] > 0.5) z[i] = clamp((obs[i] - this.mean[i]) / Math.max(this.std[i], 1e-8), -10, 10);
      else z[i] = obs[i];
    }
    const h1 = this.layers[0].forward(z, true);
    const h2 = this.layers[1].forward(h1, true);
    const raw = this.layers[2].forward(h2, false);
    for (let i = 0; i < this.actionDim; i++) this.action[i] = clamp(raw[i], -1, 1);
    return this.action;
  }
}
