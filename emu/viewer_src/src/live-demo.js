// Deliberately independent of training. The live demo never supplies evidence
// to PPO or to held-out evaluation. Bounds are curriculum frontiers, *not*
// independently verified competence limits.
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function makeRng(seed=5014) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

export function sampleDemoCommand(limits, rng=Math.random, nodeCandidates=[]) {
  const n = (a,b) => a + (b-a)*rng();
  const vxHi = Math.max(0.02, Number(limits.vx_max ?? 0.1));
  const vxLo = Math.min(0, Number(limits.vx_min ?? -0.02));
  const vy = Math.max(0, Number(limits.vy_abs ?? 0.1));
  // An autonomous showcase should stroll, not continuously pirouette at
  // the full training frontier. Manual steering still has its full range.
  const yaw = Math.min(0.5, Math.max(0, Number(limits.yaw_abs ?? 0.3)));
  const pitch = Math.max(0, Number(limits.pitch_abs_rad ?? 0));
  if (nodeCandidates.length) {
    const picked = nodeCandidates[Math.floor(rng() * nodeCandidates.length)];
    return {
      command: {
        vx:clamp(Number(picked.vx), vxLo, vxHi),
        vy:clamp(Number(picked.vy), -vy, vy),
        yaw:clamp(Number(picked.yaw), -yaw, yaw),
        pitch:0,
      },
      kind:'field-evidenced node', seconds:n(5,10),
    };
  }
  const type = Math.floor(rng()*7);
  let cmd = {vx:0, vy:0, yaw:0, pitch:0};
  if (type === 0) cmd.vx = n(0.12,0.45)*vxHi;
  else if (type === 1) cmd.vx = n(0.45,0.72)*vxHi;
  else if (type === 2) cmd.vx = n(0.2,0.7)*vxLo;
  else if (type === 3) cmd.vy = n(-0.65,0.65)*vy;
  else if (type === 4) cmd.yaw = n(-0.7,0.7)*yaw;
  else if (type === 5) { cmd.vx=n(0.25,0.6)*vxHi; cmd.yaw=n(-0.5,0.5)*yaw; }
  else { cmd.vx=n(0.2,0.55)*vxHi; cmd.vy=n(-0.45,0.45)*vy; cmd.pitch=n(-0.4,0.4)*pitch; }
  // Never exceed the policy bundle's command limits; deliberately stay well
  // inside their physical bounds, since frontiers are not proof of success.
  cmd.vx = clamp(cmd.vx, vxLo, vxHi);
  cmd.vy = clamp(cmd.vy, -vy, vy);
  cmd.yaw = clamp(cmd.yaw, -yaw, yaw);
  cmd.pitch = clamp(cmd.pitch, -pitch, pitch);
  return { command:cmd, kind:['slow forward','forward','reverse','sidestep','turn','arc','combined'][type], seconds:n(5,10) };
}

export class DemoCommander {
  constructor(seed=5014) {
    this.rng = makeRng(seed);
    this.remaining = 0;
    this.kind = '—';
  }
  reset() { this.remaining = 0; this.kind = '—'; }
  tick(dt, limits, apply, nodeCandidates=[]) {
    this.remaining -= dt;
    if (this.remaining <= 0) {
      const next = sampleDemoCommand(limits, this.rng, nodeCandidates);
      this.remaining = next.seconds;
      this.kind = next.kind;
      apply(next.command);
    }
  }
}
