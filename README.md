# EMU · Learning to walk

**Six actuators. Two feet. One learning policy.** EMU trains a dynamically unstable biped in parallel MuJoCo/MJX worlds without a scripted walking gait.

[Learning design](docs/UNIFIED_CAPABILITY_GROWTH.md) · [Current handover](HANDOVER.md) · [Installation](docs/INSTALL.md) · [Playground](https://mdt733.github.io/emu/) · [Viewer guide](docs/PLAYGROUND.md)

## The architecture

```text
robot.xml + environment.py    Robot, real contacts, terrain, realistic sensors
rewards.py                    Original eight gait payments; bounded motor effort
ppo.py + training.py          ONE shared actor/optimizer and checkpointed updates
curriculum.py                 Main-style steady gait + evidence-driven frontiers
capability_focus.py           Six teacher-only experience emphases
run.py + training_config.py   One source-pinned run, cold start, safe resume
results.py + evaluation.py    Learning evidence and independent physical exams
viewer_server.py / viewer_src Browser inference, policy replay and debugging
```

**The first 20M steps matter.** Preserve `main`'s gentle 70-update steady-only bootstrap, original PPO/rewards, source-pinned checkpoints, fall-rate retreat and shared growth budget. The feature-branch design progressively introduces six complementary emphases (commands 20%, resilience 20%, slopes 20%, steps 20%, transitions/posture 10%, control/grace 10%) when basic gait competence is earned—**not six separate brains** and not six maximum challenges from birth.

Each completed episode provides direction- and severity-aware physical evidence. Qualified competence may slowly expand a frontier; sparse evidence holds it, genuine repeated difficulty retreats it. Command mixtures use an L1 diamond/octahedral envelope rather than a rectangle. Inclines and steps use **real solid collision slabs**, not decorative slopes. The new policy has **360 inputs from a delayed 4×4 ToF array and other realistic sensors**, plus six joint targets.

**Honest status:** the unified learner remains deliberately **pre-launch**; current CPU physics/smoke and CI cannot substitute for fresh 2,048-world CUDA PPO/checkpoint/viewer acceptance or a credible held-out mixed-terrain exam. The public `showcase/champion.emupolicy` is still the protected **v5016 260M Pinnacle, 232-input** actor. Never silently replace it.

## Quick start (WSL/Linux)

```bash
git clone https://github.com/mdt733/emu.git
cd emu
git switch feature/v5023-capability-growth
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --v5015-smoke --domain-smoke
```

Preview the **proven foundation** with no training output:

```bash
emu train --run-dir "$HOME/robot_lab/emu_runs/foundation20" \
  --profile foundation --steps 20000000 --seed 733 --envs 2048 --device cuda --plan
```

The corresponding command without `--plan` launches a **legacy 220-input foundation**. It is a working comparison, **not** a 360-input unified training run. For the unified path (design inspection only, still launch-guarded):

```bash
emu train --run-dir "$HOME/robot_lab/emu_runs/unified_candidate" \
  --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
  --unified-capabilities --steps 20000000 --seed 733 --envs 2048 --plan
```

Keep training roots **outside** the repository and never resume checkpoints using a different source commit or sensor contract. Read [INSTALL](docs/INSTALL.md) and the [current handover](HANDOVER.md) before attempting GPU acceptance.

## Project boundaries

Development stays on the feature branch. The approved `main`, historical 20M/260M checkpoints, public champion, and release workflow are protected. Older experiment ladders, Pinnacle campaign launchers and raw research remain available from Git history; they are **not current training commands**. The new objective is **one clean foundation followed by one coherent, evidence-driven capability curriculum**—not another collection of campaign variants.
