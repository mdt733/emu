# v5041 — first *real* 9-tile MJX GPU trainer

Experimental source: `feature/v5041-unified-terrain-carpet`, child of
v5040. **Do not merge to main** or resume v5040 optimizer state into this
different collision topology. Begin a new source-verified, cold training
run. The 260M champion and the old v5040 step streams remain available.

## What works in the trainer (not the browser lab)

`emu train --terrain-carpet` allocates **nine real MJX 2×2 heightfield
geoms** per world (18 actual collision triangles, 50cm tiles). They
remain the same fixed collision topology for every JAX JIT update. Each
world independently carries its 36 hfield samples, 9 movable body
positions, seeded 50cm tile coordinates, deterministic ±128-cell x/y
terrain profiles and a 3×3 rolling window.

The one shared 360-input PPO actor, six specialist assignments,
20M foundation, command tracking, recoveries, push curriculum and
Academy birth-share schedule remain. Every specialist trains on the
same physical generator; the teacher's envelope changes with role.
SLOPE inherits half of STEP's demonstrated height envelope and STEP
half of SLOPE's demonstrated maximum inclination, while the other
roles receive graded background terrain. Flat foundation uses zero
terrain exposure. Independent UP/DOWN Academy frontiers are preserved.
Only physically supported tile-to-tile transitions can earn the
new per-lip stepping rewards.

After a cell transition the old outer row/column is moved to the
new window **only if BOTH feet are still inside the new 3×3 footprint**.
It never recycles a plate that remains under a foot. World coordinates
and per-episode seeds make revisits deterministic. Only 9 tile bodies
exist regardless of the 14/24-second distance travelled.

Both 4×4 ToF slant-ray distances and optical-flow depth are computed
against the new tile vertex buffer, not the retired flat/slab model.
Teacher-only torso and foot support distances likewise read the
current real tile samples. Heightfield triangle ray intersections
are reconstructed analytically; tiny differences at triangle diagonals
or extreme vertical riser faces still require independent examination.

**Crucial limits:** A 2×2 hfield contains two top triangles and its
built-in thickness/skirt; it is not yet certified to have perfectly
vertical contact faces at every nonmatching tile boundary. The
no-rebuild, per-environment profile model was verified on four CPU
worlds; **2,048-world CUDA memory, speed and contact edge accuracy
are not verified by CPU CI**. The viewer's finite heightfield remains
a preview, not a frame-accurate replay of the MJX grid.

## CPU source acceptance

These checks ran on the branch in CI:
```bash
python -m pytest -q tests/test_terrain_carpet.py
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
epochs and eight minibatches without saving a checkpoint; monitor GPU
memory and throughput. If 2048 OOMs, rerun with 1024; NEVER report
1024 passing as a 2048 pass. Do not begin a 140M run before this gate.
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
