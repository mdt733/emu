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


---

## Actual uploaded XProf results — 4 November 2026, eight prisms

**Input:** `8prism.tar.gz` supplied by the operator, generated by
`smoke_gpu_carpet.py --device cuda --worlds 2048 --frames 2 --ppo
--full-update --benchmark-updates 3 --profile-rollout --gpu-audit-dir
~/emu_gpu_audit/8prism --prisms`. Device: RTX 4060 Ti 8 GB,
JAX 0.12.0.1; four 75 cm squares, eight convex triangle prisms.
Profiler folder `xprof/plugins/profile/2026_11_04_19_17_25/`.
Analysis read the Chrome trace JSON plus the generated JSON inventory,
not just peak-memory counters. The uncompressed Chrome trace contained
**1,246,576** events, of which **950,683** were GPU activities.
No CUDA steady-state benchmark log was included with the archive.

### The execution time is now measured, not estimated

| Timing scope | Calls | Total | Share of control scan |
|---|---:|---:|---:|
| Full compiled `jit_train_update` | 1 | **4,751.5 ms** | — |
| Outer 32-frame scan (`while.1439`) | 1 | 4,734.6 ms | 100% |
| Four `mjx.step` loops per frame (`while.1372`) | 32 | **3,657.3 ms** | **77.25%** |
| Individual `mjx.step` iterations (`while.1354`) | 128 | 3,079.9 ms | nested in above |
| Separate post-physics loop (`while.1434`) | 32 | **812.3 ms** | **17.16%** |
| Other work inside outer scan | — | ~265.0 ms | ~5.60% |

The 128 lower-level step iterations are *within* the
3,657.3ms four-substep scope; do NOT add them to that total.
Similarly, CUDA graph launch CPU times can overlap device work.

**Interpretation of the 812ms post-physics loop:**
It occurs after the four physical steps in *every* control frame.
The GPU kernel signatures are almost identical to those in the
physical MJX loop: 160,078 GPU activities in the post-physics loop,
versus 742,508 inside the physics-step loop. The source calls
`_reset_with_keys` unconditionally, which in turn executes
`mjx.forward` for all 2,048 worlds before the `done` mask selects
any reset result. This is strong source+trace evidence that the
additional loop is the reset's **full-batch MJX forward solve**.
Exact function/HLO attribution still deserves a controlled ablation
or XLA source-location dump, but this is the leading candidate.

For perspective: the post-physics loop costs ~25.4ms per
20ms control frame; each of the 128 physics substep loop
iterations averages ~24.1ms. **We are effectively paying for
roughly one extra full MJX forward solve on top of the four
requested steps each control frame.** Eliminating all
unnecessary post-physics work would bound the possible speedup
to ~17% of the 4.735s scan. This is a mathematical maximum:
actual done worlds require genuine resets, and restructuring
can introduce overhead.

### What the GPU itself was running

| GPU kernel/event | Count | Accumulated device time |
|---|---:|---:|
| `input_reduce_fusion_2` | 52,450 | **921.4ms** |
| `loop_add_fusion_2` | 52,450 | 336.4ms |
| `gemm_fusion_dot_1233` | 1,049 | 269.8ms |
| `memcpy32_post` | 319,833 | 252.6ms |
| `input_multiply_reduce_fusion` | 1,049 | 201.2ms |
| `loop_multiply_fusion` | 52,450 | 149.0ms |

These counts include physical and post-physics solver work.
For example, `input_reduce_fusion_2` accounts for ~734.4ms
inside the four-step physics scope and ~186.9ms in the
post-physics loop. These are **XLA-generated HLO fusion names**,
not definitive MuJoCo source-function names: attributing them to
collision narrowphase, Jacobian assembly, solver reductions or
actuator integration specifically requires HLO line mapping.

There were **41,800 host `cuGraphLaunch (CudaGraph:119)`**
activities, and roughly **950,000 GPU activities** in 4.79s.
Measured union coverage of GPU event intervals was ~3.336s
within a 4.790s GPU event span (~69.7%). This indicates
potentially significant launch/synchronization/dispatch gaps.
It is **NOT** a hardware SM-utilisation measurement; CUPTI
does not prove that the other 30.3% of wall time was recoverable.

### Actual device memory: what is resident, padded and temporary

| Reported group | Logical footprint |
|---|---:|
| MJX model including all eight per-world mesh parameter arrays | 9.321MiB |
| Full batched `mjx.Data` | **74.883MiB** |
| Of which: MJX padded contact buffers (not extra memory) | 12.750MiB |
| Environment state excluding `mjx.Data` | 9.877MiB |
| Network weights, both MLPs | 1.214MiB |
| Adam/Optax state | 2.428MiB |
| Full standalone rollout outputs | **232.5MiB** |
| Of which: rollout metrics (not extra memory) | 48.0MiB |
| Of which: rollout `obs` and `bootstrap_obs` | 90 + 90MiB |
| XLA argument buffer budget | 91.22MiB |
| XLA output buffer budget | 320.43MiB |
| XLA temporary budget | **271.68MiB** |

MJX reports **51 statically padded contacts per world**,
**218 constraint rows per world**, and `efc_J` of shape
`[2048, 218, 14]`, about **23.84MiB alone**. There are
24 MuJoCo bodies, 22 geoms, 14 velocity DoF, six actuators
and 12 mocaps (eight real prisms and four non-contact
height sources). These are **true runtime shapes**.

Caution: tree totals overlap (`mjx_contact` is *inside*
`mjx_data`, and the metrics/obs are *inside* rollout outputs).
They must NOT be added to build a VRAM total.

The observed JAX allocator:
- Immediately following a full PPO update:
  `bytes_in_use=601,825,024`, `peak_bytes_in_use=1,223,067,392`.
- After the separately profiled update:
  `bytes_in_use=697,137,152`,
  `peak_bytes_in_use=1,226,784,512`.
- While additionally retaining standalone rollout data:
  `bytes_in_use=1,096,711,680`,
  `peak_bytes_in_use=1,600,172,800`.
- Max retained pool footprint in these captures:
  `pool_bytes=2,162,163,712`.
- The largest allocated buffer was 284,878,592 bytes,
  consistent with the XLA scratch allocation budget.

Comparisons with 18-prism peak 2,243,962,112 bytes
were made under different profiling/output-retention
conditions. They are **not** a controlled like-for-like
peak-memory A/B test.

### Comparing wall time with v5045

The older 18-prism run on the same RTX 4060 Ti was
**8.902s/full update**, 7,362 world-steps/s. The new
**single XProf-traced eight-prism full update is 4.7515s**,
equivalent to 13,793 world-steps/s (**1.87x** the
previous result). This is a compelling performance
improvement, but the old figure is a steady-state
unprofiled average, and the new one is a
profiler-instrumented single execution.

Capture the ordinary `steady state PPO PASS` line from
the v5046 run before calling it a matched throughput
measurement. An 18-prism XProf trace using the same
instrumentation would be necessary for **attribution**
of the speedup to contact/constraint workload rather
than other HLO shape or compilation differences.

### Optimisation order now justified by traces

1. **Measure a no-op reset-forward ablation in a dedicated
   profiling branch**, not production training. If skipped
   wholesale it violates the physical need to reset worlds,
   so this experiment is only a *timing lower bound*. The
   real fix must reset exactly those worlds whose `done`
   mask is true without solving a full 2,048-world
   `mjx.forward` each control frame.
2. Explore **sparse/compacted reset worlds**, and retain
   per-world deterministic task, terrain, collision contacts
   and sensor semantics. Do *not* assume `vmap(lax.cond)`
   avoids both branches: vmap ordinarily converts batched
   conditions to selects. A whole-batch
   `lax.cond(any(done),...)` also probably saves little,
   because there are often multiple terminations in 2048.
   Before implementation, quantify the actual number of
   resets per frame and the tail of that distribution to
   choose safe packed capacity and fallback.
3. The physical MJX loop still dominates (~77% of the
   control-frame scan). Investigate its **repeated tiny
   fused kernels, solver reductions and CUDA graph launch
   overhead** before assuming a large VRAM pool is the
   limiting resource. Obtain 18-prism same-instrumentation
   trace to compare contact slots, constraint rows,
   graph launches and inner solver iterations directly.
4. Conditional expensive ToF acquisition and reducing
   192 scalar telemetry streams remain possible, but the
   separate non-physics fraction of the measured 4.73s
   scan is only ~5.6%. Prioritise reset/solver work first.

**Source pin:** v5046 eight-prism branch commit
`aacf27efcabd86fe9ba7316da13b05a32f22e256`.
Training source and existing checkpoint lineages remain
unchanged by this report.

---

## Beyond collision reduction: targeted JAX/MJX speed opportunities (4 Nov 2026)

This is a **source-based experiment queue**, not a speedup claim. Profiling
on the 8-prism 2,048-world RTX 4060 Ti showed physics at 77.25% of the
32-frame control scan and a second solver-like post-physics loop at 17.16%.
Any percentage attributed to a proposed optimisation is unmeasured until an
unprofiled, source-locked, GPU-idle A/B.

| Priority | Candidate and exact code path | Expected payoff / risk | Acceptance |
|---|---|---|---|
| P0 | **Compact the entire episode birth**, not just `mjx.forward`: `environment.py:_step_impl -> _reset_with_keys` still constructs 2,048 worlds of RNG, task selection, prism generation and initial ToF even when only a handful of worlds terminate. Existing `_forward_reset_data` only compacts the solve. | Potentially high; substantial shape/seed complexity. Avoid per-world `vmap(cond)` because it generally lowers to selects. | Compare all reset fields for mixed/no/all/overflow masks; preserve episode RNG, model randomisation and exact timeout bootstrap. Select fixed compaction capacity from observed death-count distribution. |
| P1 | **Tune Newton solver and linesearch caps**: `robot.xml:<option>` has no explicit `iterations` or `ls_iterations`; benchmark-only `smoke_gpu_carpet.py --solver-iterations N --solver-ls-iterations N` creates a temporary XML, never alters production. | Potentially high; physics-sensitive. The MJX-JAX documentation explicitly recommends tuning these down. | Match baseline vs caps 4/4, 2/2 and, if stable, 1/1; compare solver errors, contact penetration, real 4cm step/sidewall and physical exam before any training default changes. |
| P2 | **Cap padded contact candidates**: `environment.py` builds a full MJX prism scene; `mjx.Data` previously padded 51 contacts and 218 constraints/world in the torso-contact variant. Investigate explicit MJX `max_contact_points`/`max_geom_pairs` and how much feet-only actually shrinks those shapes. | High if the padded solver still dominates; risky at seam double-support. | 2 feet × adjacent top/side faces, oblique crossings, diagonal corner and deepest drop must all receive real forces without truncation. Never conflate geometrically possible contacts with average active contacts. |
| P3 | **Analytic ToF ray–triangle intersection**: `terrain_carpet_mjx.py:ray_surface` does *three* ray/height fixed-point iterations across all four tiles and 17 rays. Each tile already has exactly two planes: intersect both analytically, enforce triangle barycentric membership, then check existing four physical sidewalls. | Moderate / low risk if a frozen numerical reference and occlusion tests pass. | Same 360-channel range/validity and foot self-occlusion, including near-parallel rays, 15cm walls and triangle diagonals. Benchmark separately from physics. |
| P4 | **Acquire delayed ToF only when due**: `environment.py:_advance_sensor_state` executes the full `_sensor_sample` every frame before masking stale rays; default ToF period is two frames. | Moderate; an ordinary vectorised `where` does not prevent sampling. | Pack due worlds into fixed-size cohorts, or exploit deterministic sensor-phase buckets without changing independent per-world timing. Keep gyro/flow freshness and delayed/dropout semantics exact. |
| P5 | **Eliminate duplicate PPO bootstrap observations**: `ppo.py:rollout_once` retains both `traj['obs']` and `traj['bootstrap_obs']`, roughly 90 MiB each for 2,048×32×360 fp32, then passes bootstrap through a second critic MLP. | Moderate memory win; throughput uncertain. | Use next-state values where valid, and separately compute *pre-reset* timeout bootstrap and the final transition. Preserve GAE masks exactly; benchmark against current batched GEMM, which may be faster than per-frame critic calls. |
| P6 | **Reduce 192-channel rollout telemetry**: `ppo.py:train_update` returns all per-frame metrics; recent audit measured 48 MiB logical metrics. Accumulate sufficient statistics for host reports and leave per-frame details only for fields whose timing is essential. | Moderate transfer/memory win, small physics-clock win. | Numeric parity for all curriculum promotions, per-sign lip evidence and fall/energy statistics. Don't silently change learning data, or assume the host is the bottleneck. |
| P7 | **Identify launch-heavy HLO reductions**: current trace has 52,450 `input_reduce_fusion_2` calls and ~41,800 CUDA graph launch events. Source locations might trace to solver, collision or repeated small reductions. Compare compiled HLO and source maps; inspect dense Jacobian choice, `eulerdamp` and CUDA graphs. | Potentially significant, but HLO names alone are NOT root-cause attribution. | Get source-mapped kernel traces and compare one variable per build, with exact same profiler settings and unprofiled steady timings. |
| P8 | **Lower physics substep count only as a deliberate model change**: `contracts.py` defines 4×5ms per 20ms control frame; `environment.py:_simulate_physics` calls MJX four times. A 2×10ms alternative could halve step calls but changes contact integration and spring stability. | Largest possible direct work reduction, also highest risk to trained dynamics. | New experimental lineage, revalidated servo damping, impact behavior, discrete steps, physical exams and fresh policy training. No checkpoint interchange. |
| P9 | **Compiler and engine alternatives**: A/B MuJoCo's documented `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` and evaluate MJX-Warp in a separate compatibility spike. | Flag is cheap; Warp may be faster for constraints but changes backend and must support per-world deforming convex prism meshes. | Do not assume Warp works just because plain MJX-Warp supports meshes; prove the current batched deformations and model randomisation first. |

**Recommended order:** First benchmark the already-built compact-forward and feet-only
switches to establish their separate actual benefit. Next sweep solver caps (P1)
because the experiment is now easy and benchmark-only. Then try reducing
contact candidates (P2), followed by fully sparse births (P0). Sensor/rollout
optimisations are worth doing, but source+trace say most time currently lives in
physics and the reset path. Save P8 and Warp for controlled experimental branches.

**Solver-only benchmark (no training changes):**

```bash
source ~/emu-graduation/.venv/bin/activate
# Run sequentially with GPU idle, on the SAME PR-35 SHA.
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --benchmark-updates 3 --prisms
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --benchmark-updates 3 --prisms \
  --solver-iterations 4 --solver-ls-iterations 4
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --benchmark-updates 3 --prisms \
  --solver-iterations 2 --solver-ls-iterations 2
```

Only use the steady-state timed updates after the compile/warm-up call;
a nominal solver speed win that breaks real contacts or increases falls is
not an improvement. Benchmark-only temporary XML is deleted at script exit.

---

## Owner GPU result: foot-only, four-iteration solver (4 Nov 2026)

Source: owner console, v5046 PR #35 GPU smoke with
`--device cuda --worlds 2048 --frames 2 --ppo --full-update
--benchmark-updates 3 --prisms --solver-iterations 4 --solver-ls-iterations 4`.
Commit code lineage before this documentation edit: `5da003c`.
RTX 4060 Ti 8GB, shared WSL virtual environment.

| Observed metric | Value |
|---|---:|
| First update including compilation | 77.604 s |
| Two post-compile updates, mean | **0.682 s / 65,536 world-steps** |
| Synchronized steady throughput | **96,150.4 world-steps/s** |
| Feet-only padded contacts per world | 17 |
| Feet-only constraint rows per world | 82 |
| Peak bytes in use | 1,088,438,272 |
| Episode deaths per 20 ms control frame | mean 65.62, p95 130, max 147 |
| Compact reset capacity | 128 worlds |
| Overflow fallback frames | 2 / 32 sampled frames |
| Contact/recycle/reset/small physical and full PPO gates | PASS |

**Interpretation:** The initial 96k result above was measured with solver caps 4/4. The subsequent matched default-solver A/B immediately below identifies a large solver-related speedup. The earlier 18-prism baseline remains an unmatched configuration and must not be used for attribution. Experimentally compare compact reset capacity 128 versus 192 after validating the faster solver's physical accuracy. Do not claim 96k steps/s for ordinary training without these checks.


### Matched same-checkout solver A/B (4 Nov 2026)

The operator ran both commands on the same v5046 checkout, GPU and smoke
configuration, changing **only** the benchmark XML solver caps (4/4 vs the
unmodified MuJoCo defaults). The 4/4 result was recorded above; the
subsequent default-solver run completed successfully.

| Metric | Defaults | 4/4 caps |
|---|---:|---:|
| Steady post-compile update (65,536 world-steps) | **2.031 s** | **0.682 s** |
| Synchronized steady world-steps/s | **32,267.0** | **96,150.4** |
| First update, includes compilation (not throughput) | 79.572 s | 77.604 s |
| Feet-only padded contact slots/world | 17 | 17 |
| Constraint rows/world | 82 | 82 |
| Allocator peak bytes in use | 1,088,438,272 | 1,088,438,272 |
| Mean reset births/control frame | 65.66 | 65.62 |
| Reset birth p95, max | 127, 143 | 130, 147 |
| 128-cap full-batch reset fallback | 1 / 32 frames | 2 / 32 frames |
| Physical geometry/reset/recycle/2-frame smoke, PPO update | PASS | PASS |

**Effect:** 96,150.4 / 32,267.0 = **2.98x measured steady
throughput**, or ~66.4% less measured time per update, under this exact
benchmark. Because the source and contact topology are the same, this is
strong evidence that reducing **solver and linesearch iteration limits
together** causes the gain. The two caps were changed together, so this
does NOT tell us which of them mattered most. Reset-count fluctuations
are small but not identical, and the steady-state figures average only
two iterations each, so repeat if exact precision matters.

**Physical quality is NOT certified:** the solver can be faster by
stopping before fully resolving contact impulses, penetration,
frictional support or actuator constraints. The smoke confirms only
finite trajectories and parameter movement; it does not certify
frozen-policy gait fidelity or stable multi-step traversals. Before
moving 4/4 to production:
1. Sweep (default, 8/8, 4/4, 2/2), then isolate solver vs linesearch
   caps if worthwhile. Record matched steady full-update timing.
2. Hold initial world/seed/action sequence fixed and compare torso
   height, foot penetration, contact impulses, support/sidewall contact,
   solver error and falling, at 0/4/15/40 cm signed steps.
3. Run longer frozen-policy physical exams for sustained locomotion,
   slopes and signed steps, with repeated sample seeds.
4. Only after passing the above consider a **new explicit training
   physics recipe** and full cold-start campaign. Never silently resume
   an older solver-physics checkpoint under changed integration.

The active training recipe and original `robot.xml` remain unchanged.
This result is benchmark-only and does not establish 96k steps/s in
long running `emu train` (logging, governor, evaluation and checkpoint
overheads differ).

### Four-hfield collision alternative (proposed, not implemented)

The four 75cm 2×2 per-world `HFIELD` assets ALREADY exist as
noncontact source data, providing exactly the same eight top-surface
triangles currently represented by eight custom convex prisms. The
runtime already passes batched `hfield_data` into the MJX model. A
**separate experimental collider mode** could make these four existing
hfield geoms physically active, exclude the prism geoms entirely, and
remove the per-step `PrismMeshBuilder.deform` path.

A continuous single hfield cannot represent an abrupt step without a
steep finite slope or duplicated horizontal coordinates. But **four
separate solid hfields can make literal boundary sidewalls**, including
signed risers, with no artificial 85-degree top ramp. The fourth hfield
asset dimension (base thickness) must be sized to accommodate the largest
allowed up/down offsets (current source hfields use 0.14m, while prism
solids extend 0.65m below their tops). Confirm that MJX-JAX permits
independently located / dynamically updated world-attached hfields with
exact collision on this installed MuJoCo version; do not infer from the
fact that the current noncontact hfield assets compile.

**High-risk parity cases:** 0cm seam (no wedge / phantom edge),
3mm-4cm and 15cm signed risers in both axes, 40cm high lip,
30cm ToF preview, sidewall blocking, two-feet cross-tile support, oblique
corner and protected tile recycling. For A/B, keep exactly the same
four 75cm footprints and per-world heights, solver iteration cap
4/4, 2048 worlds, 32 frames, 4 PPO epochs, 8 minibatches and
sequential synchronized steady timings. Compare padded contact and
constraint counts as well as FPS. Do not silently swap the existing
prism physics/checkpoint lineage while profiling.
