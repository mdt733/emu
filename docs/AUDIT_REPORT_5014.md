# EMU v5014: functional release review (2026-10-13)

> **Dated record — RC1 source audit (13 October 2026).** Findings and tests below describe that release candidate, not every later Playground change. For current observed status see [../RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md).

## Review basis and limits

Inspected the complete uploaded v5014 release, v5014 domain-dev pass, v5013 tournament metrics, and browser viewer sources. This was a *functional source review*, not solely an assertion that tests passed. Checked training phase boundaries, checkpoint/restoration contracts, physical-domain sampling, 55 × 4 observation/6-action export, browser policy inference and WebAssembly serving, live feed and autonomous controls, and repository rollout. The tournament does not prove a nodes-only teacher wins across seeds; this release uses low-dose full graph.

**Release state: locally audited release candidate; machine acceptance still required.** This environment lacks MuJoCo/MJX, CUDA, npm registry DNS, and the user's real training checkpoints. Therefore neither a full training run nor the actual browser physics loop is certified.

## Concrete defects found and fixed

1. **BLOCKER — training cadence expression:** `EmuMJXEnv._sample_task` passed *three* arguments to `jax.numpy.minimum`, accidentally passing the 0.14-radian pitch constant between the cadence ceiling and the real cadence. This throws during real sampling/JIT and would have prevented `emu forge` from training. Replaced with `command_phase_rate(...)` and added a real-JAX JIT regression for cadence bounds and monotonic response to speed.
2. **HIGH — unsafe cross-run continuation:** `emu forge` previously treated any schema-valid phase checkpoint at the right step as reusable, even when seed, environment count, teaching profile, or physical-domain radii differed. The launcher now verifies every existing phase's saved `training_config.json` against the intended contract before checkpoint promotion or resume; resume device and file path may differ.
3. **MODERATE — viewer domain-sensor mismatch:** the browser default omitted the synthetic measurement noise used throughout training. Default is now training-like noise and scales to 1.5×; the toggle still allows idealized/no-noise comparisons. Note that sinusoidal noise in a single browser world is still not the full stochastic MJX distribution.
4. **MODERATE — gait cadence parity:** viewer previously recalculated phase rate continuously from joystick commands and used the current curriculum frontier as a speed reference. Training samples base cadence per episode and keeps it fixed across command transitions using the task configuration speed reference. Viewer now samples/holds birth cadence, and exports the reference speed explicitly.
5. **MODERATE — CSS hidden attribute:** added an explicit global `[hidden]` display guard so overlays, live panel, results, and policy selectors cannot defeat `hidden` due to application CSS specificity.
6. **HIGH — final/snapshot non-finite writes:** regular `latest.pkl` already rejected non-finite policy state, but the rotating and final checkpoints did not. They now use the same guard. Failed snapshot writes cannot evict earlier known-good snapshots; failed final writes cannot replace a usable `latest.pkl` or promote a bad final.
7. **OPERATIONAL — uncontrolled checkout replacement:** introduced a source-only Git installer that refuses a dirty checkout, creates a reversible backup, switches to a new review branch, updates only specific source folders, preserves all other `results` and the separate `emu_runs`, checks the diff, and deliberately does not commit/tag automatically.

## Functional agreements checked

- Full-graph field influence is 0.10 after foundation; training-derived field samples are not held-out evaluation data.
- Foundation/mature phases target 20/60/140/260/400M absolute environment steps, with an identical optimizer/policy/normalizer/curriculum/field checkpoint contract. The environment's simulator state is intentionally freshly initialized on resume; it is not an exact frame rewind.
- JAX/MJX hidden-domain samples affect underlying physics only; geometry-topology randomization and heightfield roughness are **not** integrated in the default forge path.
- The browser actor contract matches the Python model: same 55-feature ordering, four-frame history, normalized-channel mask, two tanh hidden layers, six clipped mean outputs, command delay and six servo-target units.
- The live watcher serves *complete, validated* schema-7 checkpoints, retains last actor if a newly read checkpoint is invalid, and does not write to the trainer.
- The viewer's autonomous commands use self-assessed training-field nodes when present; otherwise they conservatively sample command frontiers. A field node is not an independently certified safe transition.
- Manual left/right/yaw mappings reflect the preceding user's requested inversion, but camera-relative sign must be checked in a real browser. 

## Validation actually executed here

- Python: `PYTHONPATH=. pytest -q` => **92 passed, 1 skipped, 5 subtests**.
- JAX: executable/JIT cadence regression ran on CPU; seed/domain incompatibility regression passed.
- JavaScript: syntax checks and Python-export-to-JS 220-to-6 contract test (from the original release and rechecked below).
- Migration: new installation script tested on a deliberately separate throwaway, clean Git repository, and an old source file was removed while current files and compact v5013 tournament results arrived.
- npm: **not build-verified**. Registry lookup failed with `EAI_AGAIN` (DNS), so there is no browser runtime acceptance in this environment. `package.json` still uses a floating `@mujoco/mujoco` version until a local successful `npm install` generates a lockfile. Commit `viewer/package-lock.json` after a successful build and re-test before tagging.

## Gates before committing or launching 400M

1. In WSL, install source in a fresh review Git branch using `scripts/install_reviewed_5014.sh`; do not replace `.git`, `.venv`, or `../emu_runs`.
2. In the existing correct CUDA/JAX Python environment: `python -m pip install -e .`, `pytest -q`, `emu doctor --domain-smoke`. This must compile/reset/step an MJX world, not just import JAX.
3. `emu viewer-build`, `emu viewer --live`; make sure WebAssembly loads, the fog overlay vanishes, movement signs correspond to the robot, policy updates occur and randomized commands actually drive a moving robot.
4. `emu forge --plan` to check expected targets; optionally run `emu forge --target 20000000` to validate a foundation before committing to the full 400M. The completed 20M phase will be reused automatically when the full run resumes.
5. Review `git diff` and status. Then stage/commit the audited source, add the locally generated `viewer/package-lock.json`, and tag v5014.0 only after GPU+browser acceptance.

## Do not silently change

- Never copy the v5013 checkpoint into an incompatible robot-model/schema state or into a different-phase run root.
- Never use `git reset --hard`, `git clean -fd`, or overwrite/delete prior `emu_runs` to install this release.
- An apparent 400M completion is a training budget, not an optimality certificate; fixed held-out scores must establish improvement.
