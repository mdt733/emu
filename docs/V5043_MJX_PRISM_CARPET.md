# v5043 — Real MJX-JAX triangular-prism carpet (experimental)

**Branch:** `feature/v5043-mjx-convex-prism-carpet` (based on viewer-only `feature/v5042-prism-terrain-viewer-lab`, itself based on the exposure repair). **Main, the historical nine-box learner, trained checkpoints and the original viewer have not been modified.** Prototype is OFF unless `--carpet-prisms` / `terrain_carpet_prisms=True` is explicitly requested.

## Design

- 50 cm per square; **18 independently addressable convex collision meshes** in the moving 3×3 tile window; each square has two physical triangular-prism tops.
- MJX-JAX backend keeps an immutable six-vertex/five-face topology for each mesh, but generates different vertex heights and collision face/edge normals **per world** from the 36 existing terrain samples. No native MuJoCo runtime mesh mutation or topology recompile.
- Both triangles of a square share their diagonal. Adjacent square edges share global vertex heights exactly, *except* for intentional signed changes in the square's own offset. True vertical faces give signed uphill/downhill step contacts. Actual edge normals must be tested (a visual approximation is not enough).
- Deterministic hash of global coordinates and per-episode/world seed supplies 2D random slope perturbations and separately signed X/Y step transitions. Grade magnitude remains close to the current specialist slope frontier; changes are constrained by `--carpet-transition-cap-deg`. Step profiles use per-direction up/down teacher limits and a gradually increasing challenge schedule. No per-tile Python RNG inside JIT.
- The previous nine 2×2 hfields remain **non-collidable** as terrain samples for analytic sensor/teacher geometry. The nine legacy **box geoms also remain non-collidable** in this opt-in mode. They stay untouched for normal `--terrain-carpet` jobs.
- Robot input remains `terrain_360`; 4×4 ToF, optical-flow depth, teacher support, foot-foothold confirmation and dynamic curriculum continue to operate. Prism sensor sides extend **65 cm below the varying upper edge**, matching the physical colliders. This is intentionally deep enough to keep the sidewall closed at diagonal corners when steep (up to 15°) slopes coincide with earned UP/DOWN step frontiers; it changes no mesh vertex count.
- No hard-coded polygon-shaped physical exam reward is sent to the actor. Contacts must have the correct normal and actual supported foot height.

## Current proof and limitations

**Real CI passed:** [GitHub Actions run 41579145641](https://github.com/mdt733/emu/actions/runs/41579145641), MuJoCo 3.17/MJX 3.17, JAX CPU. Eighteen compiled convex meshes, a **flat join without false horizontal wall contact at the tested location**, **4 cm upward riser with a real horizontal side-wall normal**, and **four complete six-servo EMU physics worlds** with finite 360-input observations through two frames. Earlier JAX source/deterministic tests passed. This is genuinely collision-enabled; not a visual render.

A subsequent stricter proof also checks every physical prism triangle corner against the corresponding analytic terrain height, in every tested world, and tests that adjacent edges differ only by their intended **constant** step offset. The latest CI result must pass before claiming this extra validation.

**Not yet proven:** 128-world and 2,048-world CUDA throughput/memory on the 8GB RTX 4060 Ti; sustained locomotion of a trained policy; contacts at every slope/diagonal/step-corner; stable 3×3 recycling across all boundary configurations. The numerical estimate of the transition bound is not a formal guarantee on every collision normal. Verify physical grade statistics before using the frontier for promotion. Do NOT launch 140M training on the strength of a 4-world CPU probe.

The older `scripts/exam_step_course.py --carpet` scorer is designed for nine boxes; do not use its independent box-geometry score as prism evidence. Viewer Prism Lab is a geometry preview only and does not yet use this new physics.

## WSL install / experiment (never replace an existing run)

```bash
# In a clean worktree or fresh clone; no GitHub authentication for public repo.
git clone --branch feature/v5043-mjx-convex-prism-carpet --single-branch \
  https://github.com/mdt733/emu.git ~/emu-prism-physics
cd ~/emu-prism-physics
git rev-parse HEAD
git status --short

# Activate the existing compatible venv or create one if this is a new PC.
python -m pip install -e '.[dev]'

# First on CPU (may compile for some time):
python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2

# Only after a passing CPU probe, on the RTX4060Ti:
python scripts/smoke_gpu_carpet.py --device cuda --worlds 128 \
  --frames 2 --ppo --prisms

# Then the actual 2048-world, full PPO-update and memory acceptance:
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --prisms
```

These are **bounded no-checkpoint acceptance tests**, not training.

The separate training CLI accepts `--terrain-carpet --carpet-prisms` together. Never silently resume pre-prism checkpoints in place: different physics must use a source-pinned, deliberate research fork into a **new** run directory and fresh source-provenance record. Preserve exact parent/child hashes when comparing against the historical 140M checkpoint.

## Physical acceptance still required before a big run

1. At each seed and tile coordinate compare collision vs sensor height to well below 1 mm; surface and step side normals; signed riser height 2–50 mm; seam intersections and diagonal transitions
2. Force recycling while both feet sit near the discarded row, then verify no support teleports or fake STEP reward
3. Measure 128 and 2048 worlds GPU compile, peak VRAM, steady steps/sec, PPO gradient shape and numerical finiteness
4. Read-only frozen-policy forward/reverse/cross-slope and up/down step exams; distinguish supported motion from simply surviving
5. Verify teacher attempt evidence persists long enough to promote the STEP specialists (the old transient end-of-episode evidence issue remains a separate open problem)

## Observed user RTX 4060 Ti GPU acceptance — 3 November 2026

User ran code at source SHA `7f4814aa2a9290e36432d9a86dcc67e47e203787` using
MuJoCo/MJX 3.17.0, JAX 0.12.0.1, CUDA and the 8 GB RTX 4060 Ti:

- `--device cuda --worlds 128 --frames 2 --ppo --prisms` **PASS**:
  18 actual dynamic prism top collision geoms, sensor/collider top difference
  **0.000 mm**, recycling safety and finite 360-input observations, successful
  actor gradient update (maximum actor parameter change **5.0008e-5**).
  Initial PPO update with compilation: **110.19 seconds**, 256 world-steps.
- `--device cuda --worlds 2048 --frames 2 --ppo --full-update --prisms`
  **PASS**: same contact, recycler, sensor and finiteness checks, one genuine
  **65,536-world-step** full PPO update with actor change **0.00149668**,
  initial update including compilation **128.19 seconds**. Peak GPU bytes
  in use **1,359,206,656 (1.36 GB decimal)**; peak memory pool
  **2,145,386,496 bytes (2.15 GB)**, allocation limit
  **6,440,353,792 bytes**. These numbers show large headroom on this machine.
- The original smoke script emitted an **incorrect final summary label**
  `"physics": "nine-solid-tilted-boxes"` despite successfully asserting the
  18 enabled prism geoms. This was a **reporting bug only**; fixed in a
  later branch commit, now with a label regression test.
- 128.19 s includes full JIT compilation and **must not be divided into
  65,536 to claim steady-state FPS**. The next bounded measurement is
  `--benchmark-updates 3` at 2048 worlds, yielding two synchronized
  post-compile PPO update timings. This saves no checkpoints.
- Independent frozen-policy walking and real multi-lip crossing exams
  remain required; one full PPO update does not establish learning quality.

## Run a trained robot on the real prisms — frozen read-only exam

The `scripts/exam_step_course.py` examiner now supports `--carpet --prisms`.
It loads an existing exported **360-input `.emupolicy`**, runs that
*unchanged actor* in 18 actual convex-prism MJX physics geoms, and reports
two independently measured results:

1. The training teacher's UP/DOWN credit and observed approach events, and
2. A **separate signed physical trajectory scorer**. It reads 18 real MJX
foot–prism contact pairs, upward-facing normals, actual 2-triangle top
heights in each live tile, a five-consecutive-frame support streak, and
first-lifetime trajectory (excluding post-fall resets). It never consumes
teacher crossing/reward flags to decide an independent crossing.

```bash
cd ~/emu-prism-physics
# Use a source-verified exported .emupolicy from the v5041 360-input run:
python scripts/exam_step_course.py \
  --policy /path/to/frozen_360_input.emupolicy \
  --output /tmp/emu_prism_exam_01 \
  --carpet --prisms --device cuda \
  --worlds 8 --horizon 1050 --speed .16 \
  --heights-cm .3 1 2 3 --slope-deg 10
```

The command creates new `academy_exam.json` and `academy_exam.md`.
The JSON explicitly identifies `eighteen_true_convex_triangular_prisms`,
records the actor's original source commit/weights hash, the examiner commit,
actual steps encountered, and disagreements between the two scorers. A
nine-box-trained actor evaluated on prisms is a **cross-geometry transfer
test**, not proof it learned on the new world. The 10° above is the **global
teacher request**: STEP specialist physical grades are scaled (typically
half), not a 10° dedicated SLOPE test. This exam does **not** alter a
checkpoint or launch training.

The separate `tests/test_prism_exam_scorer.py` contains signed-X/Y,
3 mm, vertical-side exclusion, first-fall censoring, and no-repeat tests.

## Not ready for an overnight run

A four-world JAX CPU contact/robot test verifies the **existence** of a
real collision implementation, not reasonable speed/memory at 2,048 GPU
worlds. The WSL 128/2,048-world `--prisms` PPO smoke above is still a
hard gate. If it fails (VRAM, compilation, invalid normals or nonfinite
gradient), leave this branch unmerged and send the full console output.
No trial training run should overwrite the 140M nine-box checkpoint.
