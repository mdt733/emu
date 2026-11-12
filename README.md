# EMU — learning to walk, then learning to keep walking

**One 40 cm dynamically unstable biped · six chassis-mounted 12V servos ·
one 360-input/6-output learned policy · real MuJoCo/MJX physics.**

EMU learns balance, locomotion, steering, rough terrain and recovery through
PPO. The gait is learned, not scripted. The operator supplies `vx`, `vy`
and yaw; a useful robot must follow them even when an obstacle appears.

[Browser playground](https://mdt733.github.io/emu/) ·
[Architecture](docs/ARCHITECTURE.md) ·
[Install & run](docs/OPERATIONS.md) ·
[Validation & promotion](docs/VALIDATION.md) ·
[Roadmap](https://github.com/mdt733/emu/issues/62)

## What is actually current?

| Area | Status |
|---|---|
| **Mainline model** | H40; 40 cm neutral height, six chassis-mounted Feetech C018 servos, passive knee support and lightweight carbon limbs; all-up mass ~506 g **estimated** |
| **Motors** | Rated output hard-capped at **±0.980665 N·m**; real motor/gear/battery/thermal behavior remains unvalidated |
| **Sensors** | 4×4 downward-tilted ToF plus IMU, optical flow, servo telemetry and command history; 360 inputs across four frames |
| **Training** | One actor + critic; 2,048 parallel GPU worlds; four rolling 1 m tiles and **eight real convex MJX prisms** |
| **Committed champion** | Original **H40 120M** policy, `showcase/champion.emupolicy`, seed 735. It remains the baseline until an explicit verified promotion |
| **Active research candidate** | **Experiment C:** fresh 0→500M training target, signed two-lip stairs and sustained post-lip `vx`. Not yet a proven result or the mainline model |
| **Viewer** | Free-roaming Infinite Meadow plus physical browser specialist drills; CPU MuJoCo heightfields are not the GPU prism examiner |

The 120M actor is packaged for inference; its complete checkpoint,
optimizer and original run logs live outside Git. Earlier B and current
C policies/runs do **not** replace the mainline champion by existing
on a branch. The trained source SHA and complete physical contract
matter more than a run label.

## Quick start (WSL2 Ubuntu)

```bash
git clone https://github.com/mdt733/emu.git ~/emu
cd ~/emu
bash scripts/setup_wsl.sh --install-node --cuda13 --check
source .venv/bin/activate

# Browser view of the checked-in champion (not a training run):
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy --no-open --port 8765
```

Open http://127.0.0.1:8765/ . The top-bar **WORLD** selector lets you
choose the generalized Infinite Meadow or specialist Commands, four
Slopes, Academy UP/DOWN stairs, Resilience, Transitions and Control.
Inspect the 4×4 ToF and actor inference while watching a frozen policy.
The browser uses real CPU collisions but **not the exact MJX prisms**.

For a current GPU physics smoke, use `bash scripts/check.sh --full`,
`emu doctor --require-gpu` and the existing
`scripts/smoke_gpu_carpet.py` acceptance tool.

## Experiment C: two steps are not enough unless EMU carries on

The cold Experiment C course places **two signed physical lips one
metre apart** and grades success only after bilateral transfer over
both, another **50 cm** of supported forward movement, and at least
**0.4 seconds** of reasonable commanded-`vx` tracking.
The first lip earns introductory coaching, not completion.
Near-step velocity relief disappears as the up/down teachers
graduate; final behavior should be ordinary `vx` tracking, not a
position-goal trick. These are *training rules*, not proven capabilities.

The separately pinned **500M cold launch**, same-lineage resume, source
identity and run output locations are in [Operations](docs/OPERATIONS.md).
Draft [PR #84](https://github.com/mdt733/emu/pull/84) contains the
training implementation. Do **not** update the running source checkout
to this housekeeping branch.

## Source structure (one owner per concept)

```text
emu/cli.py -> run.py + training_config.py
     -> training.py
         -> curriculum.py (teacher only)
         -> environment.py (robot, prism contacts, sensor/evidence)
         -> rewards.py (teaching score)
         -> ppo.py (one actor + critic)
     -> checkpoint_io.py -> results.py / independent prism_exam.py
     -> policy_export.py -> viewer_server.py + viewer_src/
```

Only the **deployable noisy sensor history and commands** enter the
actor. True physical contacts, task goals and curriculum proof stay
on the training/evaluation side. The exact 360-channel contract,
physics topology, resume state and output provenance are checked.

## What must happen before a new mainline?

An independent, matched genuine-prism exam must show improvement over
the current champion without losing basic walking, lateral/turning,
slopes, resilience, motor safety or useful efficiency. A viewer's
nose-dive (or a teacher's claimed success) cannot decide this alone.
See [Validation](docs/VALIDATION.md) and
[issue #62](https://github.com/mdt733/emu/issues/62).

Git retains old experiments and discussions; this working tree
intentionally keeps **only current code paths and three critical docs**.
GitHub Pages publication is a **separate manually approved action**;
merging source does not publish a new robot.
