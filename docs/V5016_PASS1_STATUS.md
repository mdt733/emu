# EMU v5016 — Pass 1 implementation handover

**Branch:** `feature/v5016-directional` based on `refactor/tiny-emu`. Working `main`, public 60M champion, historic training roots, PPO source and training recipe are untouched.

## What this pass actually implements

| File | New single responsibility | Used by training yet? |
|---|---|---|
| `emu/directional.py` | Pure world-frame requested-course examiner, tilted-gravity uphill basis, yaw errors over full and 1s/3s windows, bounded optional heading P step | **No** — intended as independent evaluation oracle |
| `emu/ground_plane.py` | Three body-fixed distance beams + gravity -> valid ground plane normal, slope angle and local uphill direction | **No** — read-only prototype for sensor validation |
| `emu/observation_versions.py` | Explicit v5015 220/6 and proposed v5016 244/6 actor contracts; old mapping retains exact ordering | **No** — candidate schema only, not falsely claimed deployable |
| `tests/test_v5016_control.py` | Synthetic perfect and cheating trajectories, yaw control, turning, reverse/sidestep, resets, slope direction and versioned contract | CPU-only |
| `tests/test_v5016_ground.py` | Flat and ±0.5/1/2° inclines, independent robot leaning, defective/missing ranges, geometry degeneracy | CPU-only |
| `docs/EMU_V5016_PLAN.md` | Plan revised to permit new actor inputs while demanding complete versioned training/export/hardware integration | Documentation |

### Strict boundaries

- **No reward coefficients changed**, no new actor produced, and no old actor weights or checkpoints transformed.
- Current 220-input `contracts.py` and existing PPO/browser runtime remain unchanged until the complete new 244-input path can be verified. **This is not treating 220 as a design ceiling**; it avoids accidentally replaying the 220-input policy as though it understood six new sensors.
- Gravity-tilt simulator worlds represent **constant planar inclines only**; slope angle/orientation and signed uphill progress are computed from world coordinates, not chassis-relative forward speed or simulator Z.
- `examine_course` scores an **uninterrupted episode**. It rejects reset-contaminated traces and distinguishes `course_pass` (command following) from `walking_pass` (also requires independently verified completed strides). There is **no certified stride counter yet**; never pass foot-lift proxy as `completed_strides`.
- `heading_hold_step` is a bounded rate-request P controller, **not direct servo actuation**. It tracks integrated operator yaw commands and reanchors on loss/reacquisition of estimated yaw. No drift-free absolute yaw is assumed.
- Three perfect rays always determine a plane, even when one hit is wrong. Geometry validity is **not** robust sensor confidence. Four or more readings and outlier rejection are subsequent hardware choices.

## Test and acceptance commands

From a clean worktree containing this branch, the CPU-only pass is:

```bash
python -m pytest -q tests/test_v5016_control.py tests/test_v5016_ground.py
python -m pytest -q tests
```

**CI acceptance must be inspected separately**; CPU unit tests cannot establish MuJoCo runtime, actual sensor fidelity, robotic walking competence or hardware yaw drift.

## Next passes before GPU training

1. Connect the world-fixed examiner to *actual* `EmuMJXEnv` pre-reset trajectories without modifying PPO. Include level, ±0.25°, ±0.5°, ±1°, 1.5°, 2° exams, controller on/off with matched seeds, and preserve non-slope commands.
2. Add a proper contact/touchdown stride detector using physical position, alternating foot transitions, displacement and no fall. Verify against negative controls before allowing `walking_pass=True`.
3. Compare the four existing 80M actors plus protected 60M policy using their **own** legacy runtime; no weight conversion, same held-out directional evaluation.
4. Wire the new v5016 244-input observation deliberately: simulated/hardware sensor signal, jitter/failure flag, normalisation, policy architecture/config, checkpoint schema, policy export and browser decoder. Train a **fresh versioned actor** if these new inputs add competence.
5. Bound the heading controller inside actual operator-command ownership, compare on/off before making it the training default. Test gyro integration bias/noise and corrections that exceed yaw competence.
6. Run bounded MJX/GPU smoke and new logs in the shareable `results.md` evidence bundle. Only after that try matched small-slope teaching strategies; don't immediately queue another 260M-step tournament.

## Release criterion

**The first milestone is honest directed progress on 1°**, even in the presence of plausible turning incentives, preserving flat-ground survival and command obedience. Avoid claiming any policy passes before real controlled examinations run.
