// Training draws cadence ONCE per episode from [0.85,1.20] Hz plus
// the birth-command speed gain. It does not retune cadence at every turn.
export function sampleEpisodeCadence({vx, vy}, control, rng=Math.random) {
  const base = 0.85 + 0.35 * rng();
  const ref = Math.max(0.05, Number(control.phase_speed_reference_mps ?? 0.6));
  const speed = Math.hypot(Number(vx) || 0, Number(vy) || 0);
  const speedFraction = Math.max(0, Math.min(1.5, speed / ref));
  return Math.min(Number(control.phase_rate_max_hz ?? 2.6),
                  base + Number(control.phase_speed_gain ?? 0.75) * speedFraction);
}
