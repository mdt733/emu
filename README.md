# EMU — a six-servo biped learning to walk (140M development baseline)

EMU is a **40 cm, dynamically unstable H40 robot** trained with one JAX/MuJoCo MJX PPO policy. Its balance, walking, steering, slope negotiation and stair strategies are learned rather than scripted. This repository is the *reproducible simulation and software* for H40; it is **not yet a complete CAD/wiring/validated hardware-build kit**.

**Development champion: cold H40 Spring V1 at 140,050,432 steps (seed 735).** We promote the *reproducible source and curriculum* behind it; the full original optimiser, frozen stage and exported actor remain outside Git in `~/emu-data/runs/h40-holistic-cold-s735/stages/140m/`. The checked-in `showcase/champion.emupolicy` is still the previous approved 120M public demonstration.

Independent physical exams over evaluation seeds 1701/1711/1729 (one training seed) measured **57.4% flat control**, **50.5% uphill 44–60 mm**, **19.1% uphill 64–80 mm**, and **71.6% downhill 44–80 mm**. These are simulated first-life physical exams, not Academy reward or real-hardware proof. Reliable multi-step courses and repeatability across random training seeds remain unsolved.

**Three supported workflows:** reproduce a fresh 140M actor, examine it independently on real simulated contacts, and watch a verified actor in a browser.

## Quick start · Windows 11 / WSL2 Ubuntu

Requires Python 3.13+, a WSL-compatible NVIDIA Windows driver for CUDA training, and Node.js for the browser. See [installation](docs/INSTALL.md) for clean-machine steps and caveats.

```bash
git clone https://github.com/mdt733/emu.git
cd emu
bash scripts/setup_wsl.sh --cuda13 --install-node --check
```

### 1. Train from random weights

```bash
bash scripts/run_h40_holistic_cold_wsl.sh plan  # print the exact recipe, no training
bash scripts/run_h40_holistic_cold_wsl.sh all   # cold run to 140M + exams
```

The maintained program uses **one actor and one optimiser** through 0–20M walking foundation; 20–40M withdrawal of gait-specific shaping; 40–120M outcome-based command, terrain and resilience training; 120–140M signed stair approach/commitment. **It stops at 140M before the unsuccessful historical transfer stage trains.** Uphill/downhill difficulties are independently governed. Checkpoints at 20/40/120/140M are immutable. New runs default to `~/emu-data/runs/h40-main-140m-s735/`, avoiding the original verified experiment directory.

**A successful seed-735 actor has been physically evaluated.** That establishes a reference policy, not that fresh training seeds will reach the same skill. Reproduce from new seeds and compare held-out examinations; reward and Academy frontier are not proof.

### 2. Examine frozen abilities

At the end of a full cold run, the launcher checks **2,000 true-prism signed stairs and 2,000 matched flat controls** at 120M and 140M, with no PPO updates, and creates `holistic_cold_results_bundle.zip`. The maintained standalone public examination command is `emu prism-exam --help`; source-fixed research matrix fixture logic is in `scripts/v5053_matrix_exam.py` and should not be used to silently substitute physics or a different actor.

Compare verified first-foot support, both feet on the new level, upright/aligned traversals, survival, command tracking, slopes and resilience; don't promote on reward alone. [Physical-quality criteria](docs/PHYSICAL_QUALITY.md).

### 3. Watch the robot

```bash
source .venv/bin/activate
emu viewer-build
emu viewer showcase/champion.emupolicy --no-open --port 8765
```

Open http://127.0.0.1:8765/ in your Windows browser. For the untrained sensor/geometry laboratory: `bash scripts/start_sensor_lab_wsl.sh`. For a live experiment: `emu viewer --live --root ~/emu-data/runs/h40-holistic-cold-s735 --no-open`. Browser Meadow and training MJX prisms use different colliders, so the viewer is qualitative—not an independent stair certification.

The checked-in `showcase/champion.emupolicy` remains the **previous approved 120M demonstration**; the new *development* champion's local actor is `~/emu-data/runs/h40-holistic-cold-s735/stages/140m/actor.emupolicy`. For that policy, use `EMU_PHYSICAL_PROFILE=h40-spring-v1 emu viewer ~/emu-data/runs/h40-holistic-cold-s735/stages/140m/actor.emupolicy --no-open`. No merge updates the public demonstration without separate approval.

## What lives where

| Owner | Files |
| --- | --- |
| Robot, six rated actuators and sensors | `emu/robot.xml`, `physical_variants.py`, `contracts.py`, `tof_geometry.py` |
| True eight-prism MJX physics | `emu/environment.py`, `terrain_carpet_mjx.py`, `carpet_prism_mjx.py` |
| One PPO actor, outcomes and curriculum | `emu/ppo.py`, `training.py`, `rewards.py`, `curriculum.py`, `capability_focus.py` |
| Safe resume, export, read-only evaluation | `emu/checkpoint_io.py`, `policy_export.py`, `evaluation.py`, `prism_exam.py` |
| Single-browser viewer and live dashboard | `emu/viewer_src/`, `viewer_server.py`, `console_dashboard.py` |
| Installation, one cold launcher and tests | `scripts/setup_wsl.sh`, `scripts/run_h40_holistic_cold_wsl.sh`, `tests/` |

Full architecture: [Core skeleton](docs/CORE_SKELETON.md) · [Installation](docs/INSTALL.md) · [Quality and exams](docs/PHYSICAL_QUALITY.md) · [Documentation index](docs/README.md) · [Current handover](handover.md).

**Safety boundary:** simulated motors, spring forces, mass, sensor occlusion and fast feet-only contact approximations are not yet certified against measured hardware. The original experiment sequence and failed hypotheses are preserved in [research findings](docs/RESEARCH_FINDINGS.md) and Git history, not live experiment wrappers.
