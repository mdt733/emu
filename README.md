# EMU — teaching a biped to walk, then to handle the world

**Six actuators. Two feet. One brain. Many complementary lessons.**

EMU is a reinforcement-learning project for a dynamically unstable, six-actuator biped. A single PPO policy learns locomotion through parallel **MuJoCo/MJX** physics simulations rather than a hand-coded gait. The longer-term goal is not simply to walk faster or survive one obstacle: it is to walk **reliably, accurately, gracefully and efficiently** across commands, disturbances and real terrain.

**This is the `feature/v5023-capability-growth` research branch.** It builds on the successful foundation from `main` but introduces a new sensor contract and unified capability curriculum. The approved public robot remains separate:

**[Try the existing EMU Playground](https://mdt733.github.io/emu/)** · [Learning design](docs/UNIFIED_CAPABILITY_GROWTH.md) · [Capability dimensions](docs/CAPABILITY_DIMENSIONS.md) · [Active handover](HANDOVER.md) · [Install & acceptance](docs/INSTALL.md) · [Viewer guide](docs/PLAYGROUND.md)

## The learning idea

> Start with the walking lessons that worked on `main`. Keep one policy. Give different simulated worlds a different **teaching emphasis**, measure the outcomes of *completed episodes*, then grow only the capabilities the robot has actually earned.

### Phase 1: protect the first 20 million steps

From a **cold start**, the 360-input policy receives the same *kind* of gentle standing and walking curriculum that made `main` effective:

- Preserve the original eight gait rewards, 32-frame rollouts, four PPO epochs, eight minibatches, learning rate `2e-4`, seed `733` and normally `2048` parallel worlds.
- Keep the original first **70 PPO updates steady-only**, conservative early movement commands, fall-sensitive retreats, and main-style disturbance burden.
- Use the new **4×4 forward ground-sensing array from step zero**; never splice a 220/232-input checkpoint into the different 360-input network.
- Keep the new step and slope surfaces physically dormant in this stage. **20M is a teaching-distribution boundary, not proof of mastered walking and not a required campaign stop.**

The settings and early teaching distribution are protected in code. **Equivalent walking performance has not yet been demonstrated** by a freshly trained unified policy.

### Phase 2: six teachers, one shared actor

After 20M, specialist activation blends linearly over 10M further steps (0% to 100% inside the allocated birth cohorts). That blend is only lesson frequency, not a maturity gate or another skill's permission. Each world still encounters gentler versions of the other challenges; its focus determines what it probes closer to or just beyond the currently demonstrated limit.

| Teacher focus | Target share | What it deliberately challenges |
| --- | ---: | --- |
| **Commands** | 20% | Forward/reverse `vx`, lateral `vy`, yaw, and combinations |
| **Resilience** | 20% | Pushes, disturbances, supported recovery |
| **Slopes** | 20% | Uphill, downhill, left and right cross-slope |
| **Steps** | 20% | Genuine step-ups/downs, height and rotated edges |
| **Transitions / posture** | 10% | Starting, stopping, reversing, acceleration, chassis attitude |
| **Control / grace** | 10% | Accurate supported motion, smoothness, slip, mechanical effort |

There are **not six neural networks**. All worlds update the **same PPO policy**. Teacher identity and exact simulator terrain truth are not privileged actor inputs.

A complete training loop is:

```text
Sample worlds from current capability limits
    → act with the same shared policy
    → finish episodes and record what was really attempted
    → measure survival, command tracking, support and contact
    → advance / hold / retreat individual skill frontiers
    → nudge each physical target independently
    → sample the next set of experiences
```

**Command requests use an L1 diamond** in the `vx/vy` plane (an octahedron including yaw), not a square that allows maximum forward and maximum sideways speed simultaneously. Sampling deliberately includes the interior, boundary and modest probes just beyond it. Success at one mixed-direction command is **not** evidence that every pure-axis maximum can expand.

**Slopes and steps are real collision geometry.** Two oriented solid slabs produce actual incline pitch/roll, rises/drops and angled step lips. Supported far-side **foot contact** matters; moving the robot's root past an edge is not a successful crossing.

**Efficiency never means standing still.** The control teacher values walking and tracking first, then reduced needless motion, slip and simulated servo shaft work. Shaft joules are **not** electrical battery energy.

For the full decision rules, observation contract and non-regression checklist, read [the unified learning contract](docs/UNIFIED_CAPABILITY_GROWTH.md).

## What works today — and what does not

| Area | Current state |
| --- | --- |
| Public 232-input Pinnacle demonstration | **Protected and available** in the Playground |
| New 360-input actor, 4×4 array and mixed physical terrain | **Implemented**; small CPU physics/gradient smoke available |
| Single-policy six-focus curriculum and local moving targets | **Implemented**; behavioural learning still needs long-run evidence |
| Frozen `capability-exam` for commands, real slopes/steps, recovery and control | **Implemented**; not yet proven on a successfully trained new policy |
| Source-pinned checkpoint/resume/export acceptance | **Bounded test path available**; real unified CUDA lifecycle still needs verification |
| One cold-start continuous 360 learner | **Source-guarded mode implemented**, with same-source resume/target extension; long CUDA walking run still needs real GPU acceptance and evidence |

Follow [CI on GitHub Actions](https://github.com/mdt733/emu/actions) for the *current commit*. Green CI or two PPO updates demonstrate engineering health, **not** a better walking robot.

## Quick start — fresh Ubuntu/WSL2 checkout

Requires **Python 3.13+ with `venv`**, Git and, for CUDA experiments, a working NVIDIA GPU/WSL driver. The setup script installs project dependencies, CUDA-enabled JAX and Node for the viewer; it **does not start training**.

```bash
# On a fresh WSL home directory; clone into a known location.
cd ~
git clone --single-branch --branch feature/v5023-capability-growth \
  https://github.com/mdt733/emu.git "$HOME/emu"
cd "$HOME/emu"

bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate

python --version
git branch --show-current
git status --short      # clean checkout required for source-pinned runs
emu doctor --no-require-gpu --no-mjx-smoke
```

If you already have a checkout, **use its real directory rather than cloning over it**. If setup reports missing Python 3.13/venv, fix that prerequisite before continuing. For CPU-only setup, omit `--cuda13`. All following commands assume the virtual environment is active and you're in the repository root.

### Inspect the unified training plan — safe, read-only

```bash
emu train --run-dir "$HOME/robot_lab/emu_runs/unified_candidate" \
  --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
  --unified-capabilities --seed 733 --envs 2048 \
  --steps 20000000 --device cuda --plan
```

**`--plan` does not train or create a run.** The continuous live mode is an explicit `EMU_UNIFIED_CONTINUOUS=cold-growth-approved` opt-in for a clean 2048-world CUDA learner, and should only be used after exact-SHA CI and the owner's GPU acceptance. It does not require a fixed 20M, 40M or 60M checkpoint. The legacy `--profile foundation` command is a **different, 220-input actor**, not a substitute for this unified cold start.

### Engineering acceptance — short, explicit tests

A small CPU test exercises physical terrain, all six teaching focuses and one real PPO gradient:

```bash
python scripts/smoke_unified_focus.py --device cpu --envs 12
```

On your CUDA machine, this attempts **one production-shaped 2,048-world PPO update** (32-frame rollout, four epochs, eight minibatches). It does not create a training run or checkpoint:

```bash
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
```

Only after that succeeds, the bounded **two-update** lifecycle can test checkpoint, source-matched resume, observation normalisation and export. Use a **new** run root outside the checkout:

```bash
mkdir -p "$HOME/robot_lab/emu_runs"
ROOT="$HOME/robot_lab/emu_runs/unified_accept_$(git rev-parse --short HEAD)_8"

python scripts/smoke_training_lifecycle.py first --root "$ROOT" --unified --envs 8 &&
python scripts/smoke_training_lifecycle.py continue --root "$ROOT" --unified --envs 8 &&
python scripts/smoke_training_lifecycle.py verify --root "$ROOT" --unified --envs 8
```

These acceptance runs remain limited to two PPO updates, **not enough to learn walking**. If the small lifecycle passes, the next gate uses a *fresh* root at 2,048 worlds. See [INSTALL.md](docs/INSTALL.md) for full copy-paste cold launch, interrupted resume and same-run target extension commands.

### Frozen assessment and browser viewer

Once a 360-input `.emupolicy` exists, you can run an independent, **read-only** quick exam of its actual physics. A two-update acceptance export is suitable only for testing the examiner's plumbing, **not** demonstrating locomotion:

```bash
emu capability-exam --policy "$ROOT/tiny_smoke_actor.emupolicy" \
  --output "$ROOT/quick_physics" --device cpu --worlds 4 --quick

emu viewer showcase/champion.emupolicy --no-open
# Browser: http://127.0.0.1:8765/
```

The complete unified exam measures actual command tracking, all four slope directions, paired supported step crossings, mixed incline/step geometry, disturbances, transitions, slip, action smoothness and simulated shaft work. The public champion uses an **older observation contract**, so use `terrain-exam` (not `capability-exam`) for its legacy terrain diagnostics. The local viewer has no authentication; do not expose it to the internet.

## Code map

```text
emu/robot.xml                    The physical six-actuator robot
emu/environment.py               MJX worlds, sensors, real contacts, episode facts
emu/rewards.py                   Eight fundamental gait payments
emu/ppo.py + emu/training.py      One actor, PPO, rolling updates, checkpointed run
emu/curriculum.py                Evidence windows, difficulty and retreat governor
emu/capability_focus.py          Six teacher emphases and diamond command probes
emu/run.py + emu/training_config.py  One source-pinned configuration and run entry
emu/evaluation.py + emu/results.py  Frozen physical exams and learning reports
emu/viewer_server.py + emu/viewer_src/  Local browser simulator and policy replay
```

The current CLI entry point is `emu`; use `emu --help` and `emu train --help` for available options. Older `ladder`, `lab5015`, `campaign5016` and multi-arm trial machinery were retired from this branch; the old implementations remain recoverable from Git history.

## Working agreements

- **Protect `main`, the approved 260M Pinnacle `.emupolicy` and the public showcase.** This branch is research; promotion requires explicit approval and separate frozen-policy evidence.
- **Keep one continuous policy lineage.** Runs, snapshots and reports live outside the Git checkout. Never resume a checkpoint under a different source commit or observation/sensor contract.
- **Improve competence, not just training reward.** A candidate that gains terrain performance but forgets basic supported walking has not surpassed the reference.
- **Prefer simpler existing modules over new campaign frameworks.** Current tasks and preserved acceptance gates live in [HANDOVER.md](HANDOVER.md); the long-term design lives in [UNIFIED_CAPABILITY_GROWTH.md](docs/UNIFIED_CAPABILITY_GROWTH.md).

This branch is aiming for **one cold-startable walker whose demonstrated capabilities grow together**—not six separate champions, and not another pile of one-off experiments.
