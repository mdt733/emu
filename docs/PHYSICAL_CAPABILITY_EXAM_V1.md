# EMU Physical Capability Examination v1

**Status: research / uncalibrated. Not a champion selection system.**

Research branch: `research/physical-capability-exam-v1`. Base source:
`78a261c22487871137c4c957a136080586842694`. Frozen
`showcase/champion.emupolicy` is the owner-approved 120M H40 actor.

This document is an **audit of the named code at that base commit**, not a
claim that these experiments have been executed or that Spring V1 has passed.

## Why the previous numbers were misleading

The earlier `scripts/v5053_matrix_exam.py` uses a **350-control-frame
default** (7 seconds at `CONTROL_DT=0.020`), with a permitted horizon of
80–700 frames (1.6–14 seconds). Its nominal `flat_control` explicitly commands
**+0.25 m/s**, not stillness. Its `stand` task injects a signed initial
velocity perturbation (up to 0.30 m/s) but has no bounded-position
requirement. In the matrix reducer `survived` means that the first physical
life lasted to the requested horizon; it is not a measure of commanded speed,
upright endurance, zero-command balance or travel direction. Independent
destination and traversal gates are more demanding but answer different
questions. A 7-second survival percentage can be very high for a robot that
falls regularly over 60 seconds.

The legacy matrix **does** intentionally stop credit at the first terminal,
retain world-level records, measure exact physical seam height, and record
source/actor/model fingerprints. These are valuable and should be retained.

The old `emu/evaluation.py` is **training retention**, not a physical
certification. Its four panels (`core`, `dynamic80`, `combo70`,
`resilience75`) default to the same 350 frames. It calls
`ppo.evaluate_summary`, which stores first fall but continues stepping the
auto-reset environment and averages tracking/support/posture across subsequent
lives. Its broad weighted score may therefore blend a failed initial
episode with later reset episodes. This may be useful for retention but cannot
answer first-life endurance. The panels also randomise commanded conditions
rather than fixing a zero-command stillness test.

`emu/prism_exam.py` performs physical contact inspection of procedurally
generated true prism terrain; its default horizon is 300 frames (6 seconds),
with an allowed maximum of 650 (13 seconds). It masks everything after the
first terminal for independent crossings, and gives per-world gait, contact
and energy diagnostics. Actual encountered terrain severity is measured,
not assumed to equal the teacher's requested frontier. It is a useful
research tool, not a long-term generalist exam.

The visual Infinite Meadow uses its own browser CPU terrain/heightfield
workflow. It is **qualitative** until per-frame observation, motor and
collision parity with the exact MJX scene is verified. The new visual world
must not be replaced or silently treated as the GPU ground truth.

## Exam v1 — first delivered slice: flat balance, locomotion, transitions

Entry point: `python -m emu.physical_capability_exam`.

The exam uses the **existing** `make_trainer()["evaluate_trace"]`, which
runs the immutable actor's deterministic mean action on the actual MJX
environment. It reuses the actor sensor/normalizer contract, physical model
and servo target/actuation pipeline. No policy optimiser update, PPO loss or
curriculum promotion occurs. `main`, viewer, and training code are unchanged.

The examiner lengthens `max_episode_steps` **beyond its longest test** and
disables Academy success terminals. Initial command and per-frame command
channels are deliberately applied through the established trace evaluator,
rather than modifying the reward engine.

| Scenario | Duration | Operator instruction |
|---|---:|---|
| Zero-command standing | 60 s | vx=0, vy=0, yaw=0 |
| Forward | 30 s | vx=+0.25 m/s |
| Reverse | 30 s | vx=-0.20 m/s |
| Left/right lateral | 30 s each | vy=±0.15 m/s |
| Left/right yaw | 30 s each | yaw=±0.50 rad/s |
| Combined | 30 s | vx=+0.20, vy=+0.10, yaw=+0.30 |
| Stop/reverse | 30 s | 10s +0.25, 10s zero, 10s -0.20 m/s |

Each world gets its own deterministic reset key. **The first terminal ends
its measured life forever**; later automatic resets inside the inherited
environment are discarded by the host scorer, even when the underlying
physics kernel keeps running. First physical post-step positions and headings
are used; commanded trajectories integrate requested yaw, not actual yaw,
to prevent an actor from steering off course without penalty. Sensor and
command corrections internal to the policy are not substituted for the
operator's desired trajectory.

The standing pass requires **all** these provisional gates for the full
60 seconds: no terminal/fall, maximum birth-relative XY radius ≤0.20 m,
maximum birth-relative heading error ≤0.35 rad, 95th-percentile torso tilt
≤20°, measured foot-contact coverage ≥75%, and bad-contact coverage ≤5%.
Corrective stepping is allowed; wandering away is not. These tolerances are
not yet calibrated against both reference actors.

For each trial we record survival, first-life duration, time to first fall,
failure class, maximum/final XY drift, heading drift, tilt, contact coverage,
bad contacts, achieved vx/vy/yaw, operator-frame velocity tracking errors,
desired-path tracking error, and positive servo shaft work. The stop interval
has its own speed measure so forward/reverse velocities cannot average out
and masquerade as a successful stop.

**No aggregate weighted championship score is assigned.** Summary rows
retain the denominator of the number of evaluated worlds. A policy does not
pass as a generalist merely because a stair or speed specialist is strong.

### Execute on the source-matched H40 workstation

From this research branch, with WSL/JAX/MJX installed:

```bash
export PYTHONPATH=$PWD
export EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Cheap contract/unit tests, no GPU or actor required:
python -m unittest tests.test_physical_capability_exam -v

# Small compilation/loader diagnostic; NOT a full qualification:
python -m emu.physical_capability_exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/exams/h40-v1-stand-diagnostic.zip \
  --worlds 2 --device cuda --scenario stand_zero_60s

# Frozen initial flat capability campaign:
python -m emu.physical_capability_exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/exams/h40-physical-v1-flat-s1701.zip \
  --worlds 8 --seed 1701 --device cuda
```

**Do not** reuse an output ZIP path. Every run records source-file SHA-256,
actor weights and artifact identity, robot XML, compiled solver settings,
sensor contract, sampling seed and exam settings. Test only actor/physics
source combinations accepted by the existing strict actor loader.

The ZIP contains `manifest.json`, `summary.json`, `summary.md`,
`worlds.csv` and `capabilities.csv`. There are no invented results or
replays: **replay export is not yet implemented**. Keep console logs and
the exact ZIP along with the source commit.

### Important initial limitations

- The implementation uses the retained full-trajectory MJX trace, at
  2–32 worlds, to make precise first-life analysis feasible on 8 GB VRAM.
  A future streaming on-device accumulator is needed for larger samples
  and 5–15-minute endurance without giant host trace transfers.
- Contact coverage records actual foot/terrain contact, **not** the centre
  of pressure, load distribution or dynamic stability margin. The current
  torso tilt metric is based on the MJX gravity vector.
- No fresh GPU rollout was possible through the GitHub-only connection.
  No 120M/140M calibration results have yet been generated in this branch.
- No confidence interval or cross-training-seed generality is inferred from
  multiple world resets of one trained actor.
- Internal source simulator auto-reset still executes after a terminal but
  **cannot affect the measured first life**. A future kernel should freeze
  dead worlds to reduce computation and enable stronger parity comparisons.

## Next work, in strict order

1. **Calibrate the baseline.** Run the 120M actor with v1. Inspect world rows
   and at least three success/failure trajectories. Align measured stability
   with the existing viewer, or identify a concrete source of disagreement.
2. **Spring V1 source/authentication.** The 140M Spring V1 actor/checkpoint
   is **not** compatible with current `main`'s `h40` model. Recover its
   original model from training source
   `22266821bdbfc82a529ef708e5171c82c6c9f29c`; run with its own
   physical/motor target contract. Port the v1 scoring contract with
   exactly documented differences; compare *capability outcomes* between
   source-authenticated environments, not pretend equal mechanical physics.
3. **Disturbances.** Add exact birth velocity/tilt and timed push impulses,
   with recovery time, maximum excursion, contact safety and first-life
   survival. Keep a no-perturbation standing gate mandatory.
4. **Terrain.** Add source-matched signed +X lip/grade levels, explicit
   first destination top contact for each foot, transfer, continuation
   and full multi-step course completion. Distinguish up/down and
   include trajectory/lateral drift. Verify the physical geometry.
5. **Endurance.** Add 5–15 minute continuous, mixed-command, mixed-terrain
   first-life exams; stream compact JAX stats and selected replay frames
   rather than storing unbounded [time, world, contact] arrays.
6. **Cross-engine comparison.** Replay fixed first-state pose, commands,
   terrain prisms, servo target/actuation histories and sensor packets in
   MuJoCo CPU/browser and MJX GPU. Compare per-frame positions, velocities,
   impulses, ToF returns, command delivery, joint angles and first terminal.
   Browser Meadow aesthetics are kept separate from authoritative parity.
7. **Benchmark acceptance.** Require stable first-life results across several
   seeds, reproducible fresh ZIPs, explicit contact/fall definition review,
   and calibration to both reference actors before using any pass gate to
   accept a replacement policy.

## Archive policy

Historical matrix/prism/teacher exams remain evidence under their **original
source commit, physics model, actor, trial configuration and date**. Do not
combine them into a v1 leaderboard, modify their original score meaning, or
overwrite their ZIPs. The old source code is retained temporarily for
forensics, not reused as a current champion scoreboard.
