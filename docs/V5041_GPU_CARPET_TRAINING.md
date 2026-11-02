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
The first 2048-world CUDA PPO acceptance **PASSED** on commit
`c876240f935d1acc4346b229f84381286f314ace`, as documented below.
Sustained training throughput and later-source GPU performance are not yet
established; the 140M run therefore remains gated on the bounded lifecycle
and physical exams.

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

## Actual GPU acceptance result — 2 November 2026

User-reported WSL CUDA test on RTX 4060 Ti 8 GB, using source SHA
`c876240f935d1acc4346b229f84381286f314ace`:

- `128` worlds, 2 physics frames + real PPO: **PASS**, 2.61 mm
  sampled analytic/real collider surface difference.
- `2048` worlds, 2 physics frames plus **32-frame rollout, 4 PPO
  epochs, 8 minibatches**: **PASS**, 2.71 mm source/collider difference.
- Nine real tilted solid boxes, 360-input finiteness and tile recycling
  accepted; no error or OOM.
- JAX GPU reported `peak_bytes_in_use=1075855360`,
  `bytes_limit=6440353792`, `peak_pool_bytes=2149580800`.
  These numbers are JAX allocator diagnostics, **not** total physical
  system RAM use or a sustained-run measurement.
- Optional `warp`/ `mujoco_warp` import warnings were nonfatal. A
  CUDA timer accuracy warning did not stop PPO.

**Source caveat:** this success predates subsequent guard/test improvements:
PPO now explicitly requires a changed actor network, reports initial
compilation-inclusive timing, and tile recycling protects spherical
feet touching the actual tilted-box overhang. Do not describe the original
GPU test as verification of those later changes. Rerun the 128 and 2048
gates on the final source SHA after its CPU CI passes, then run the
bounded checkpoint/export/resume lifecycle using that same frozen SHA.

## Then an actual first/continue/verify lifecycle (new external run directory)

The existing three-stage `scripts/smoke_training_lifecycle.py` now has a
`--carpet` mode, explicitly selecting the **same nine-box v5041 physical
terrain** as production. Without `--carpet`, the old unified smoke still
uses the historical two-slab world and must NOT count as v5041 acceptance.

The three stages exercise a 2048-world one-update cold learner, save a
complete checkpoint, resume it into a fresh child run for one additional
update, verify unchanged parent bytes and advancing observation
normalization, and export/read back a 360-input schema-3 browser actor.
The script internally gates the run to no more than two PPO updates and
does not claim walking competence.

Run all stages from the exact clean Git SHA that passed the updated
128-world and 2048-world CUDA preflights:

```bash
ROOT="$HOME/robot_lab/emu_runs/v5041_carpet_lifecycle_s735"

python scripts/smoke_training_lifecycle.py first \
  --root "$ROOT" --unified --carpet --envs 2048 --seed 735

python scripts/smoke_training_lifecycle.py continue \
  --root "$ROOT" --unified --carpet --envs 2048 --seed 735

python scripts/smoke_training_lifecycle.py verify \
  --root "$ROOT" --unified --carpet --envs 2048 --seed 735
```

Choose a fresh `ROOT`: stage 1 refuses to overwrite one that already
exists. Do not change source SHA, seed, flags, or root between stages.
This script checks checkpoint, warm continuation and export; it is
stronger evidence than one uninterrupted `emu train --steps 131072`
invocation, which by itself does not prove warm resume.

## Follow-up engineering after first full GPU/lifecycle PASS

The user completed the **updated 128- and 2048-world CUDA acceptance** on
`728f0139b3f032dcc68770372730ccfb36494ac6`, including positive real
actor-weight changes (max 0.001431 at 2048), nine-box/18mm ball-foot
clearance and finite 360 inputs. The 2048 first update took **64.68s
including XLA compilation**; JAX reported peak 1,075,855,360 bytes in use.

The same SHA passed every stage of the true 2048-world carpet lifecycle:
first 65,536 steps, fresh parent-preserving warm continuation to 131,072,
normaliser count 65,536→131,072, completed-resume no-op and schema-3
actor export/readback. This IS an authentic source-pinned CUDA test,
but the exported actor has not learned to walk after two updates.

The one-update console printed an impossible ~2M FPS and zero reward
because the training wall clock started after the compilation+first PPO
update and the final rolling-reward list was empty. Source now accounts
for the first update and appends the last rollout's real reward. The
77.6% early fall rate is still an actual novice-physics measurement.

**Additional untested source improvements:** Frozen-policy trajectory
exams now report contact with the NINE actual solid tiles, not the
inactive Academy slabs, and only count a world's first uninterrupted
episode. Deployable actor exports identify the real `terrain_carpet`
collision topology. CPU trace regression, exam CLI, and optional
post-compile GPU throughput benchmarking have been added. These
improvements require **new CI and a freshly verified source SHA**;
do not retroactively attribute them to the already-passed `728f0139` run.

### Measure real sustained PPO throughput (bounded, no checkpoint)

After pulling a source SHA whose full carpet CI is green, run:

```bash
cd ~/emu-v5041-gpu-preflight
git pull --ff-only
git rev-parse HEAD
git status --short

python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo \
  --full-update --benchmark-updates 4
```

This trains only 4 × 65,536 temporary simulation steps, saving **no
weights**. It reports synchronized throughput from the THREE
post-compilation updates. It is not the performance of a full saved,
logged 140M campaign, but far more meaningful than first-update timing.

### Probe the actual frozen nine-box exam

For the existing two-update lifecycle checkpoint, first re-export the
same original actor to a **new** file so it carries the added physical
topology provenance. This does not alter or resume the checkpoint:

```bash
ROOT="$HOME/robot_lab/emu_runs/v5041_carpet_lifecycle_s735"

emu export-policy \
  "$ROOT/continued/checkpoints/final.pkl" \
  "$ROOT/tiny_carpet_exam_actor.emupolicy"

python scripts/exam_step_course.py \
  --policy "$ROOT/tiny_carpet_exam_actor.emupolicy" \
  --output "$ROOT/carpet_exam_untrained_s1701" \
  --carpet --worlds 8 --horizon 200 --seed 1701 \
  --heights-cm 1 --slope-deg 0 --speed .10 --device cuda
```

This is a read-only **exam-pipeline smoke**, not evidence of competence
from a virtually untrained actor. The output reports nominal frontier
vs actual supported lip heights, signed UP/DOWN attempts and crossings,
falls, and first-life forward progress; it cannot credit resets after
the first fall. Denominators of 0 are **inconclusive**, not successful.
No claim of independent external contact rescoring is made: the
crossings come from the existing physically contact-gated teacher
events. Do not call the 140M run ready until this works on actual GPU
and independent step/slope physical checks have been inspected.

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
