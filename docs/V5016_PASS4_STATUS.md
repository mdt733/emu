# EMU v5016 — Pass 4: command transitions, gyro realism, and shareable GPU campaign

**Status:** implementation candidate on `feature/v5016-directional`; CI and saved-policy GPU acceptance must be checked on this exact commit. Neither `main`, the 60M champion, the original four-way training, nor any checkpoint is changed.

## Actual files and responsibility

| File | Change |
|---|---|
| `emu/command_exams.py` | Deterministic, time-indexed *operator* commands: steady, turn-stop left/right, sidestep left/right, reverse and stop-resume |
| `emu/ppo.py` | Exam-only, real MJX `evaluate_trace` accepts scheduled commands, finite gyro bias, seeded white gyro noise and 0–5 control-frame delay. **Critically aligns internal `cmd_target`, `cmd_ramp_from`, task, current actor observation and issued yaw each frame.** No PPO update, reward or training-path change. |
| `emu/directional_exam.py` | Per-policy ON/OFF comparison now supports schedule and gyro realism; logs independent issued-vs-operator errors; validates same protocol in each pair |
| `emu/v5016_batch.py` | Four stage profiles; PLAN by default, explicit `--run`; subprocess isolation for each case, fail-closed resume on source/checkpoint/parameter mismatch, automatic `results.md`/`results.csv`/`results_data.json`/`results_bundle.zip` |
| `emu/cli.py` | Exposes `emu directional-batch` and single-case schedule/gyro options |
| `tests/test_v5016_schedule_batch.py` | CPU protocol generation, invalid inputs, no-write plan, bundle/no-checkpoint rules and matching-checkpoint resume checks |
| `tests/test_gait_exam5016.py` | Existing real MJX CPU smoke extended to **mid-episode change of yaw direction and zero yaw**, bias, random gyro noise, delayed gyro reading, and post-step environment command parity with PPO request |

Why the internal ramp matters: `environment._step_impl` derives its next command from `state.cmd_target`, `state.cmd_ramp_from` and `state.cmd_ramp_frac`; changing just `state.task` or the actor observation is not sufficient. The examiner now forces one coherent issued command through all of these values during read-only evaluation.

The controller grades against the **original operator command**. Command schedules are external, fixed by timestep and are not derived from the robot's current direction; turning downhill cannot redefine uphill as success. Sensor bias and noise only affect the controller's gyro integration; they are not a replacement for the sensor-normalizer training pipeline. Sensor latency is represented as an integer queue of 20ms frames. Simulated body gyro is not world yaw. Future work still needs a calibrated real-IMU model, contact impulse/stance slip and geometric ramp transitions.

## Suggested afternoon GPU plan

This is a *test-only* campaign, **not 20M/60M new training**. Keep the training checkout pinned. Make a new dedicated WSL worktree for v5016:

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-v5016-afternoon origin/feature/v5016-directional
cd ~/emu-v5016-afternoon
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
git status --short

RUNROOT="$HOME/robot_lab/emu_runs/tiny_fourway_001"
OUT="$HOME/robot_lab/emu_exams/v5016_afternoon_001"
```

The following `emu directional-batch` commands print a plan and do **not** touch disk/GPU unless `--run` is added.

### Phase 1 — actual saved-policy smoke: 2 paired cases

```bash
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile smoke
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile smoke --run --device cuda
```

Two baseline cases: flat steady at 2 worlds × 40 frames (shape/compiler check), then flat turn–stop–hold at 2 worlds × 160 frames. **Stop** on incorrect operator requests, nonfinite gyro, post-step task disagreement, or empty contact evidence. Run failure must not trigger training.

### Phase 2 — four-way equal-task pilot: 20 paired cases

```bash
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile pilot
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile pilot --run --device cuda
```

Each of baseline/shadow/slope/pitch15 gets exactly the same five tests: flat turn-stop left, flat turn-stop right, directed 0.25° uphill, directed 0.5° uphill and 0.5° cross-slope. **Every case is a matched heading-control OFF/ON pair**, 8 worlds × 350 frames per condition. No automatic winner; survival, commanded-course progress, 1s/3s yaw error, gyro drift and contact-stride estimates remain separate.

### Phase 3 — sensor stress: 4 paired cases

```bash
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile sensor --run --device cuda
```

Compare baseline and slope on flat turn-stop and 0.25° uphill. Simulated gyro-Z: **0.015 rad/s constant bias**, **0.02 rad/s independent Gaussian per-control-frame noise** and **2 frames (40 ms) latency**. These are *engineering probe values*, not measurements of your actual IMU.

### Phase 4 — optional 1° stretch: 4 paired cases

```bash
emu directional-batch --root "$RUNROOT" --output "$OUT" --profile stretch --run --device cuda
```

Only after earlier stages show meaningful progress; all four policies, 1° uphill at 8 worlds × 350 frames, OFF/ON. No 10° experiments. A single 1° success is not a robust walking qualification.

Each profile writes `$OUT/<profile>/results.md`, `results_data.json`, `results.csv`, `results_bundle.zip` and per-world `off/`/`on/` evidence. **Send the four ZIP files** for review. Cases run sequentially (not one process per GPU) and can resume from complete files. If a case is partially written, the runner refuses to treat it as finished; inspect its error before choosing a new output root.

## Gates before authorising training or enabling heading P by default

- Exact CI head: full Python / real MJX CPU contracts, JS browser package, clean after build.
- Real GPU on saved 80M checkpoints, not a fresh zero policy; 40/160 frame smoke first.
- ON/OFF initial physics, original operator schedule and checkpoint SHA must match, and environment **post-step** commands must match the issued command except on reset frames.
- Command transitions must not destroy level-ground behaviour. Good yaw with worse survival is a failure.
- Sensor stress must expose effect of bias/noise/delay without passing privileged simulator compass yaw to PPO.
- Contact pair events are geometric proxies; no claim of load-bearing walking until force/impulse and slip are validated.
- v5016 candidate **244-input actor is still an explicit proposal**; this pass doesn't convert old weights or alter active training observations.
