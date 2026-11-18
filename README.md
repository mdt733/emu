# EMU — a little robot learning to walk

**One dynamically unstable 40 cm biped · six central servos · one learned policy**

EMU uses PPO reinforcement learning in MuJoCo/MJX to discover balancing, walking, steering and disturbance recovery. Its gait is **learned**, not scripted.

[Browser playground](https://mdt733.github.io/emu/) · [Current handover](handover.md) · [Core skeleton](docs/CORE_SKELETON.md) · [Installation](docs/INSTALL.md) · [Forward roadmap](https://github.com/mdt733/emu/issues/62)

## Current robot: H40 120M

The current source model is the **40 cm central-servo H40**: six 12 V Feetech ST-3215-C018 servos in a roughly 460 g central chassis (about **506 g estimated** including lightweight carbon limbs, feet and carriers), remote knee actuation, passive knee/shin springs and **±0.980665 N·m capped** actuator output. Masses, transmission losses, thermal performance and carbon components are **not yet hardware-validated**.

One 360-input / six-output policy observes a four-frame history of noisy IMU, optical flow, proprioception, command and a forward **4×4 VL53L8CX** approximation. The 45°×45° sensor points 70° down from a mount 30 mm below the torso centre. Training uses **four 1 m tiles / eight true triangular prisms** with recycling.

The owner-selected H40 actor was trained **from scratch to 120,061,952 world steps** (seed **735**), with these immutable identities:

| Identity | Value |
|---|---|
| Trained source commit | `301d18b7b1cd1b165e6578fc53f9f8cda94fb680` |
| Run | `h40-1m-tof70-120m-s735` |
| Actor | `showcase/champion.emupolicy` |
| Actor SHA-256 | `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673` |
| Exported policy SHA-256 | `178c201af661d3172d701a7d1d8a7711f9b2106e1f7058bc4902aa9c64a37c7a` |
| Sensor contract SHA-256 | `d8f559641d240f6eb4bd54a3dc7e96f414994682ebc1b478209df202cc903995` |

**Capability status:** The owner reports a stable, fast gait and successful slopes up to about 15° in viewer use. That is **not** an independent frozen MJX exam. **Reliable step ascent remains unsolved.** Infinite Meadow's streaming CPU heightfields are a qualitative playground, **not** the GPU trainer's eight convex-prism physics or a step-certification exam. Never use this policy to imply tested hardware feasibility.

The exact training `final.pkl`, its normalizer/optimizer state, logs and GPU run directory stay **outside Git**, frozen in `~/emu-data/runs/h40-1m-tof70-120m-s735/` on the training PC. The checked-in `.emupolicy` contains only the deployable actor and its model/sensor contract.

## Try the trained robot

From a checkout with its dependencies installed:

```bash
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy --no-open --port 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). **Infinite Meadow is the only viewer world and the public showcase's default.** A seeded 15×15 m physical CPU heightfield window streams indefinitely through natural hills; +Y north gradually increases hill and step difficulty, and returning to old coordinates reproduces the same ground and walk-through trees. It features the original locally bundled Meadow panorama. The public trained-policy showcase begins with an optional **scripted eastbound stroll** (not learned navigation), and joystick/WASD takes manual control. No GPU training environment is needed for playback.

For the **untrained geometry/ToF calibration lab**:

```bash
bash scripts/start_sensor_lab_wsl.sh
```

This uses the current generated H40 model but deliberately contains **no neural policy**. It can be useful for checking ray occlusion, mounting and contact before buying hardware.

## Train, inspect, export

```text
physical robot + 4×4 ToF + four-tile MJX world
                    ↓
              2048 GPU worlds
                    ↓
         shared 360→256→256→6 PPO
                    ↓
         checkpoints + independent exams
                    ↓
         export-policy → browser viewer
```

Training starts from scratch; historical models and incompatible sensor contracts are not silently migrated. Use `emu train --help`, `emu prism-exam --help`, and [the core skeleton](docs/CORE_SKELETON.md) for the supported interfaces. Keep training and policy evaluation pinned to the exact source SHA, generated robot XML and observation contract.

**Release is separate from source promotion.** GitHub Pages deployment is manual and requires `showcase/release_approval.json` matching the exact actor and the original checkpoint SHA-256; `python scripts/verify_showcase_release.py` must pass before the public build can deploy. A merge alone does not update GitHub Pages.

## Architecture

| Purpose | Owner |
|---|---|
| Robot XML + physical variants | `emu/robot.xml`, `emu/physical_variants.py` |
| Observations + sensor contract | `emu/contracts.py`, `emu/tof_geometry.py` |
| Physics + four-tile terrain | `emu/environment.py`, `emu/carpet_prism_mjx.py` |
| PPO, curriculum and rewards | `emu/ppo.py`, `emu/curriculum.py`, `emu/rewards.py` |
| Source-pinned checkpoint + export | `emu/checkpoint_io.py`, `emu/policy_export.py` |
| Exam + browser playback | `emu/prism_exam.py`, `emu/viewer_server.py`, `emu/viewer_src/` |

The active code supports **one 360-input physical actor**, not old 220/232-input policies or retired sensor geometries. Git is the archive: do not keep duplicate experiments or compatibility scaffolding in the current source.

## Historical research and measured limitations

[Retained research findings](docs/RESEARCH_FINDINGS.md) distills the measured v5032–v5058 experiments, GPU performance, stair scoring pitfalls, gait-incentive hypotheses and rated-torque correction. The original complete reports and old run commands remain at the [pre-cleanup Git snapshot](https://github.com/mdt733/emu/tree/ded2a111c9ab50c304e1b1a70736a84b32449887/docs), **not** in current `main`.

Old checkpoints belong to their original source/model hashes; never warm-start the new physical configuration using an incompatible checkpoint.

[Documentation index](docs/README.md) · [Maintenance principles](docs/MAINTENANCE.md) · [Current handover](handover.md).
