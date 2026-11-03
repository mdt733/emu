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
  // v5045 only loads the ACTUAL current 360-input array-only actor.
  // Old 220/232-input bundles remain recoverable from historical Git.
  const b=JSON.parse(text);
  if(b.format!=='emu-policy'||b.schema!==3)
    throw Error('Only schema-3 360-input policies are supported in this viewer');
  if(!b.policy||!b.observation||!b.control||typeof b.robot_xml!=='string')
    throw Error('Policy bundle is missing required sections');
  const a=b.policy,c=b.observation.sensor_contract;
  if(a.action_dim!==6||a.history_len!==4||a.base_obs_dim!==90||a.obs_dim!==360)
    throw Error('Viewer requires the six-servo 360-input four-frame actor contract');
  if(!Array.isArray(a.layers)||a.layers.length!==3)
    throw Error('Viewer requires a three-layer actor');
  const f=b.observation.base_fields;
  if(!Array.isArray(f)||f.length!==90||
     JSON.stringify(f.slice(55,58))!==JSON.stringify([
       'ground_forward_grade_rad','ground_sideways_grade_rad','ground_visibility_confidence'
     ]))
    throw Error('Invalid 360-input observation fields');
  if(b.source?.checkpoint_schema!==9||
     !c||c.schema!=='terrain_360'||
     c.sensor_model_revision!=='forward_tof4x4_array_plane_fit_no_tripod_2'||
     c.history_len!==4||c.action_dim!==6||
     JSON.stringify(c.base_fields)!==JSON.stringify(f)||
     c.tripod_mounts_body_m!==undefined||
     c.tripod_beams_body!==undefined||
     c.tof_range_limits_m!==undefined||
     c.array_ground_estimator!=='weighted_4x4_plane_residual_8mm_v1'||
     c.array_ground_max_fit_residual_m!==.008||
     c.array_ground_min_valid_cells!==6)
    throw Error('Invalid schema-9 array-only 360-input contract');
  const settings=c.settings||{};
  for(const key of ['tof_period_frames','flow_period_frames',
    'tof_delay_frames','flow_delay_frames','tof_dropout_prob',
    'flow_dropout_prob','gyro_bias_std_rad_s','sensor_noise_scale']){
    if(!Number.isFinite(settings[key]))
      throw Error('Invalid sensor setting: '+key);
  }
  if(JSON.stringify(c.tof4x4_mount_body_m)!==JSON.stringify([0,0,0])||
     JSON.stringify(c.tof4x4_pitch_from_horizontal_deg)!==JSON.stringify([40,60,80,100])||
     JSON.stringify(c.tof4x4_yaw_deg)!==JSON.stringify([-22.5,-7.5,7.5,22.5])||
     JSON.stringify(c.tof4x4_range_limits_m)!==JSON.stringify([.04,1.5])||
     c.tof4x4_occlusion!=='nearest ray-sphere foot hit masks terrain return'||
     c.tof4x4_validity!=='one bit per cell, delayed, zero on dropout'||
     !f.slice(58,74).every((name,i)=>name===
       `tof4x4_range_${Math.floor(i/4)}${i%4}_m`)||
     !f.slice(74,90).every((name,i)=>name===
       `tof4x4_valid_${Math.floor(i/4)}${i%4}`))
    throw Error('Invalid schema-9 4x4 physical range contract');
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
