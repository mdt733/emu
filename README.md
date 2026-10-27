# EMU — one biped, one brain, an expanding set of skills

**Six actuators · two feet · one PPO policy · physical MuJoCo/MJX learning**

EMU is a reinforcement-learning biped project. A single actor learns to walk
from scratch, then trains on commanded forward/lateral/yaw motion, recovery,
inclines, discontinuous steps, transitions and smoother, less wasteful motion.
The objective is **supported, accurate locomotion**, not just surviving a slope,
moving downhill, accumulating reward, or raising a curriculum target.

[Public EMU Playground](https://mdt733.github.io/emu/) ·
[Installation and commands](docs/INSTALL.md) ·
[Completed v5024 experiment](docs/V5024_RUN_PLAN.md) ·
[50M + 12×20M exploration, refinement and 400M graduation](docs/GRADUATION_CAMPAIGNS.md) ·
[Learning design](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Physical capability inventory](docs/CAPABILITY_DIMENSIONS.md) ·
[Release readiness](docs/RELEASE_READINESS.md) ·
[All documentation](docs/README.md) ·
[Engineering handover](HANDOVER.md)

## Which generation is this?

This is the **`feature/v5024-release-polish`** engineering branch, forked
from `feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`.

- **Candidate system:** 360-input, delayed 4×4 forward-ground sensor;
  one cold-start learner and one six-emphasis adaptive teacher.
- **Frozen training source:** the separate v5024 branch and exact commit
  above, used by the **completed** 3 × 60M GPU campaign. **Do not pull polish commits
  into its running or resumable checkout.**
- **Published reference:** the older 232-input, 260M Pinnacle in the public
  Playground. It is a *temporary reference and rollback policy*, not the
  intended future system. Replacing it requires independent physical exams,
  a reviewed merge and separately authorised public release.

Do not infer that this branch has already outperformed Pinnacle from passing
CI or increasing slope difficulty. Training results and promotion remain open.

## How it works

```text
        2,048 batched MuJoCo/MJX worlds
                       |
       single 360-observation, 6-action PPO actor
                       |
             completed physical episodes
                       |
    command tracking • support • recovery • steps
      slope stability versus tracking • energy
                       |
          six focused lesson allocations
                       |
        local difficulty targets: easy / near / hard
                       |
                  next lessons
```

All emphases teach the **same** actor. A world can emphasise commands,
resilience, slopes, steps, transitions/posture or control/efficiency. The
teacher's focus label and exact simulator terrain are *not* privileged inputs.
The policy sees delayed sensor observations instead.

A physical-practice target is **where to teach**, not a certification of
competence. Completed near-target successes may nudge it up; persistent easier
failures may nudge it down; absent evidence and failed hard probes ordinarily
hold it. The held-out examiner, not the training curriculum, determines what
the actor can actually do.

**Cold start:** main-style baseline gait learning is preserved, including
the first 70 steady-only PPO updates. The foundation can be 10M or 20M world
steps; the v5024 comparison uses 10M. Specialist activation then blends in
over another 10M steps. These are teaching-allocation boundaries, never
mandatory checkpoint stops or proof of maturity.

**Physical terrain:** real contact geometry creates inclined support and
upper/lower step surfaces. Supported far-side foot contact distinguishes a
completed step from a root that merely passed an edge. An upright robot
sliding downhill is stable, but **has not passed commanded tracking**.

**Efficiency:** smoothness and simulated shaft work are diagnostics and
teaching signals subordinate to useful, supported and accurately tracked
movement. Simulated shaft joules are not measured battery energy.

## Quick start — WSL2 / Ubuntu

Requires Python 3.13 with `venv`, Git and Node. NVIDIA drivers and
CUDA-capable JAX are required only for GPU training. Work on a **separate
checkout** from any running experiment:

```bash
git clone --single-branch --branch feature/v5024-release-polish \
  https://github.com/mdt733/emu.git "$HOME/emu-polish"
cd "$HOME/emu-polish"
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
bash scripts/check.sh
```

CPU-only installation: omit `--cuda13`. To check the new experimental
system without training:

```bash
python scripts/v5024_campaign.py           # plan only; 180M across 3 arms
python scripts/smoke_unified_focus.py --device cpu --envs 12
```

On an appropriate GPU, the standalone 2,048-world CUDA PPO preflight is:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
```

These probes do not establish walking quality. To run the **existing frozen
v5024 campaign**, use the original `feature/v5024-terrain-control`
checkout and its documented source-pinned launcher, **not this branch**.
See [INSTALL.md](docs/INSTALL.md) for exact commands, safe resume and
independent physical examinations.

## Evaluating policies and viewing the robot

```bash
# 360-input physical terrain matrix: 1/2/3/5° inclines and up/down steps
emu capability-exam --terrain-matrix \
  --policy /path/to/policy360.emupolicy \
  --output "$HOME/robot_lab/emu_runs/my_terrain_exam" \
  --device cpu --worlds 8 --horizon 350

# Local browser simulator and protected published-reference policy
emu viewer showcase/champion.emupolicy --no-open
# Visit http://127.0.0.1:8765/
```

The independent examiner separates **supported moving stability** from
**command tracking**, including signed downhill/lateral drift, yaw errors and
supported step crossings. The local viewer can vary slope direction/grade and
step height. It is not authenticated; do not publish its server port.

## Source map

| Area | Purpose |
| --- | --- |
| `emu/robot.xml` | Robot bodies, joints, actuators and contact geometry |
| `emu/environment.py` + `emu/contracts.py` | Batched physics, delayed observations, physical episode facts |
| `emu/rewards.py` | Original gait rewards |
| `emu/ppo.py` + `emu/training.py` | Shared actor, PPO update and training lifecycle |
| `emu/curriculum.py` + `emu/capability_focus.py` | Locally advancing/retreating targets and lesson allocation |
| `emu/run.py` + `emu/training_config.py` | CLI training config, source-pinned resume/extension |
| `emu/evaluation.py` + `emu/results.py` | Frozen physical exams and reports |
| `emu/viewer_server.py` + `emu/viewer_src/` | Browser MuJoCo viewer and policy replay |
| `scripts/v5024_campaign.py` | 3-arm comparison (opt-in `--run`) |
| `tests/` + `emu/viewer_src/tests/` | Maintained contracts, physics and browser checks |

Older 220/232-input compatibility code is **scheduled for removal** when a
360-input replacement is approved and the public release migrates. It remains
temporarily so the reference policy and exporter can be compared honestly;
it is not a second endorsed learner. The v5023 tournament is preserved as a
historical comparison, not a current default launcher.

## Release agreement

Only develop on feature/polish branches. Keep the training SHA, v5023 archives,
Pinnacle and public Pages untouched during evaluation. Verify frozen physical
performance and cold-start/resume/export before proposing a reviewed PR to
`main`; merge and public deployment both require explicit owner approval.
[Promotion and legacy-removal checklist](docs/RELEASE_READINESS.md).
