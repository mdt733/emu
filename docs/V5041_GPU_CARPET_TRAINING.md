# v5041 — first *real* 9-tile MJX GPU trainer

Experimental source: `feature/v5041-unified-terrain-carpet`, child of
v5040. **Do not merge to main** or resume v5040 optimizer state into this
different collision topology. Begin a new source-verified, cold training
run. The 260M champion and the old v5040 step streams remain available.

## Current experimental collider architecture (v5041)

Each JAX world has **one 3×3 deterministic rolling window** of 50cm
terrain cells. Each cell still generates **two source triangles**
(four vertex heights) from one seeded smooth gradient potential plus
independently bounded up/down step discontinuities. The *actual collision*
geometry is now **NINE TILTED SOLID BOX TOPS**, not heightfield skirts:

- A solid tile is a genuine rigid planar top and four real sides.
  Its plane is fitted to the same four corner samples (the two triangles
  are coplanar only when local curvature is zero).
- Non-contact 2×2 hfield samples remain in the batched JAX model for
  the deterministic **source terrain coordinates**, while the **4×4 ToF,
  optical-flow depth, support-height evidence, and STEP rewards now
  intersect the exact physical box geoms**. The original triangle
  source is retained for birth/generator accounting only.
- No 12-wall workaround is needed; all 9 physical boxes recycle using
  the existing safe-foot/one-cell window rule, with constant MJX topology.
- Exact independent UP/DOWN stepping evidence and the 64-tile
  teacher-only anti-rocking history are preserved. Six specialists still
  share the SAME 360-input PPO actor.
- Changing slope gradients are controlled by
  `--carpet-transition-cap-deg` (default 3 degrees) and mixed skill
  envelopes. A tile-box's top approximates the two triangle samples,
  with an explicit <8mm collision-to-sensor discrepancy test across the
  reference difficulty.
- Actual MJX foot-top contact must be upward-facing, near the **real box
  top**, and not a vertical side. No diagonal corner is paid as
  an imaginary single-lip transition.

**Reason for change:** MJX 3.17 gave a misleading upward contact
normal for a sphere striking a *heightfield skirt* from the side. An
isolated A/B physical test verified that a real solid box side gives
a horizontal contact and minimal unintended upward displacement.
The earlier nine-heightfield+12-wall approach could not eliminate the
spurious extra hfield contact. Rather than teach the PPO to exploit that
artifact, the source now has **NO active heightfield collision**.

**Limits not yet proven:** The tilted box is a planar approximation to
two possibly noncoplanar triangles, and its corners/edges can differ by
millimetres from the analytic source. This difference, high-step
clearances, 9-tile 3×3 recycling with actual biped feet, and the complete
360-input sensor pipeline must pass dedicated collision and PPO tests.
2048-world 8GB GPU memory/performance has **NOT** been verified; the
140M run is therefore NOT authorised yet.

## CPU source acceptance

These checks ran on the branch in CI:
```bash
python -m pytest -q tests/test_terrain_carpet.py
python -m pytest -q tests/test_carpet_physical_contacts.py
python -m pytest -q tests/test_carpet_solid_geometry.py
python -m pytest -q tests/test_carpet_sensor_geometry.py
python -m pytest -q tests/test_carpet_visit_rewards.py
python scripts/probe_carpet_hfield.py
python scripts/smoke_gpu_carpet.py --device cpu --worlds 4 --frames 2 --ppo
```
The smoke verifies actual MJX reset, nine collision bodies, dynamic
per-world hfield samples, sensor numerics, safe recycler, simulated
physics and an actual PPO gradient update. It writes no weights.

## Required GPU gate on the 8GB RTX 4060 Ti

Clone a completely separate checkout of this branch and activate the
**already working GPU/JAX 0.12 + MuJoCo 3.17 WSL environment**.
Do not inadvertently replace GPU-enabled JAX with CPU-only wheels.
Install this local package in that environment, then:

```bash
git status --short
git rev-parse HEAD
python -c "import jax; print(jax.__version__,jax.default_backend(),jax.devices())"
python scripts/smoke_gpu_carpet.py --device cuda --worlds 128 --frames 2 --ppo
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 --frames 2 --ppo --full-update
```

`--full-update` executes the production 32-frame rollout, four PPO
epochs and eight minibatches without saving a checkpoint. Both GPU
smokes explicitly reject a PPO update that leaves every policy/value
parameter unchanged, and print the measured maximum parameter delta.
The full gate also reports available JAX GPU peak-memory statistics.
The JSON result includes the synchronized first-update wall time and
world-step count. **First-update time includes XLA compilation; it is
not steady-state throughput**. Record the complete console output for
diagnosing OOM, compilation, or PPO failures.

If 2048 OOMs, rerun with 1024; NEVER report 1024 passing as a 2048
pass. Do not begin a 140M run before this gate.
Depending on CUDA kernel/XLA compilation, one first update can take
much longer than steady-state steps.

## Then a bounded lifecycle check (new external run directory)

The standard `emu train` bounded two-update smoke tests PPO plus
checkpoint creation, provenance and export, not walking competence.
Example at 2048 environments:

```bash
EMU_UNIFIED_ACCEPTANCE=bounded-ppo-lifecycle emu train \
  --run-dir "$HOME/robot_lab/emu_runs/v5041_carpet_smoke_s735" \
  --steps 131072 --envs 2048 --seed 735 --device cuda \
  --profile sim2real --terrain-actor --terrain-step-max-m .08 \
  --unified-capabilities --terrain-carpet --academy-stream \
  --terrain-lesson coherent-careful --growth-recipe main-mature \
  --foundation-mix main --force-program gentle
```

If the bounded lifecycle succeeds, and frozen physical terrain exams
show no sensor/contact mismatch, a *new* 140M from-scratch lineage
may be launched in another external directory:

```bash
EMU_UNIFIED_CONTINUOUS=cold-growth-approved emu train \
  --run-dir "$HOME/robot_lab/emu_runs/v5041_carpet_s735_140m" \
  --steps 140000000 --envs 2048 --seed 735 --device cuda \
  --profile sim2real --terrain-actor --terrain-step-max-m .08 \
  --unified-capabilities --terrain-carpet --academy-stream \
  --terrain-lesson coherent-careful --growth-recipe main-mature \
  --foundation-steps 20000000 --foundation-mix main \
  --fixed-eval-every-steps 10000000 --force-program gentle
```

Use **the same source SHA, learning arguments and run directory** for
resume; never claim correctness just because a small training smoke
passes. The physically relevant outcomes are stable fast walking,
supported uphill/downhill lip crossings, changing-slope control,
survival, effort and retained skills across all six roles.
