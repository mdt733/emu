# JAX GPU allocation and execution audit — EMU v5045/v5046

This is a code-level budget, **not a kernel trace result**. The
2048-world 18-prism benchmark at commit `65bc3df2` reported
8.9096s per full 65,536-transition PPO update, and 8.3314s
for the separately compiled identical 32-frame policy+MJX+
sensor/reward rollout (93.5%, NOT additive time accounting).
It reported `peak_bytes_in_use=2242671616`,
`peak_pool_bytes=4292870144` and
`bytes_limit=6440353792` for that GPU process.

## What each update asks JAX to do

1. Carry 2,048 live per-world `mjx.Data` pytrees, task packets,
   4×90 sensor history, episode counters, 64×2 visited-tile
   history, two×256 terrain profiles, 25-frame reward history,
   per-world dynamic convex meshes and a shared actor/critic.
2. Run `lax.scan` over 32 control frames, with 6-channel
   Gaussian actor sampling, 4 sequential 5ms
   `mjx.step` calls per frame and separate real 17-ray
   (16 ToF plus downward flow) sensing. That's 262,144
   physics world-substeps per 2048×32 update.
3. **Unconditionally calculate** `_reset_with_keys` for all
   worlds every frame, BEFORE the per-world `done` mask.
   It calls `_make_batched_data`, samples tasks and the
   full deterministic carpet, builds dynamic prism meshes,
   runs `mjx.forward` and samples sensors. This work
   occurs even if zero of the 2,048 worlds ended.
   This is a verified source-level redundant-computation
   hypothesis, **not yet a measured GPU-percent attribution**.
4. `_advance_sensor_state` always performs `_sensor_sample`
   for all worlds before checking `tof_due` and `flow_due`.
   Standard ToF is due only every 2 frames. Midpoint
   recycling and true terrain contact analysis also run on
   every frame, as do hundreds of named scalar JAX metrics.
5. Store ~232 MiB of logical rollout outputs: 90MiB
   raw observations, 90MiB bootstrap observations,
   ~48MiB of ~192 scalar metrics, 3MiB actions and
   1.25MiB scalar PPO data. Save another 90MiB logical
   normalized observations for PPO (XLA may reuse storage).
6. Compute GAE, normalize advantages, run four PPO epochs
   of eight 8,192-transition minibatches. Model has
   318,221 trainable fp32 parameters: ~1.21MiB, Adam
   moments ~2.43MiB. The actor itself isn't a large
   VRAM allocation; intermediate 8192×256 activations
   are ~8MiB each.
7. Materialize returned telemetry, update the running
   normalizer and host curriculum, log and checkpoint
   periodically. The 18-prism run's own timers show that
   these host stages are negligible relative to the JIT.

## Why peak VRAM is not 'allocated every step'

JAX traces a static-shape graph; `lax.scan` carries state
rather than materializing 32 full MJX universes. However,
it saves rollout observations and telemetry for PPO.
Within MJX each world's contact and solver arrays are
padded to static model-derived capacities, independent
of how few contacts physically occur. The exact
`mjx.Data.contact`, `efc_*` and Jacobian dimensions
must be read from the compiled model; do not guess them
from the 8 or 18 top-level prism count.

Physical GPU allocation includes persistent arrays, XLA
temporaries/scratch, executable/code buffers, allocator
reuse, and possible aliasing. `peak_bytes_in_use` is
a maximum live allocator statistic, not bytes written
each frame. Compilation ~100+ seconds is a different
metric again. A pprof device heap snapshot is not a
within-kernel time series.

## One-run hardware audit (benchmark only)

On the eight-prism branch, from the repo with the
same activated GPU virtualenv:

```bash
python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 \
  --ppo --full-update --benchmark-updates 3 \
  --profile-rollout --gpu-audit-dir ~/emu_gpu_audit/8prism \
  --prisms
```

The new optional `--gpu-audit-dir` prints/stores:
- `gpu_allocation_audit.json`: every JAX array group's
  logical shapes, sizes, 16 largest leaves, real count
  and byte totals for `mjx.Data`, the padded contact buffer,
  model, optimizer, rollout and telemetry; the report includes
  `contact_capacity_per_world` and actual MuJoCo model counts.
- `xla_compiled_full_ppo_memory`: XLA's compiled
  argument, output, scratch/temp and alias budget. The
  lowered executable's compilation **may add another
  lengthy compile** after the timing benchmark; it
  cannot alter the measured full-update rate.
- `device_after_full_ppo.pprof`, plus a second profile
  with independently materialized rollout arrays:
  JAX device allocation snapshots.
- `xprof/`: GPU kernel timeline from ONE extra fully
  synchronized PPO update. Its overhead is excluded
  from measured throughput. Open the XProf TensorBoard
  profiler Memory Viewer and GPU trace to identify
  real kernel timing and peak live buffers.

Compare exactly the same command, after porting the
opt-in diagnostic, on the pinned 18-prism baseline.
The flag is diagnostics-only; the normal training,
actor, sensors, rewards and checkpoint behavior are
unmodified.

**Prioritize profiling:** (1) unconditional full reset
and MJX forward, (2) four contact/solver physics
substeps, (3) unconditional 17-ray sensor synthesis,
(4) per-frame report accumulation, (5) gradient steps.
Test optimizations as isolated physics-checked commits:
a faster result that changes contact semantics or silently
drops difficult tasks is a regression, not a win.
