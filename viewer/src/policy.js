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
  // .emupolicy schema 1 is little-endian. All mainstream browsers we target
  // are little-endian; use DataView fallback if that ever stops being true.
  return new Float32Array(src);
}

export function parsePolicy(text) {
  const p = JSON.parse(text);
  if (p.format !== 'emu-policy' || p.schema !== 1) throw new Error(`Unsupported policy format/schema: ${p.format}/${p.schema}`);
  if (!p.policy || !p.observation || !p.control || typeof p.robot_xml !== 'string') throw new Error('Policy bundle is missing required sections');
  if (p.policy.action_dim !== 6) throw new Error(`Expected 6 actions, got ${p.policy.action_dim}`);
  if (p.policy.history_len !== 4 || p.policy.base_obs_dim !== 55 || p.policy.obs_dim !== 220) {
    throw new Error(`Viewer v1 expects the v5013 55×4=220 observation contract; bundle says ${p.policy.base_obs_dim}×${p.policy.history_len}=${p.policy.obs_dim}`);
  }
  if (!Array.isArray(p.policy.layers) || p.policy.layers.length !== 3) throw new Error('Viewer v1 expects a 3-layer actor');
  if (!Array.isArray(p.observation.base_fields) || p.observation.base_fields.length !== 55) throw new Error('Invalid base observation field contract');
  return p;
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
