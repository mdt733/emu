# EMU — a little robot learning to walk

**One dynamically unstable biped · Six servos · One neural policy**

EMU learns to balance, walk, change direction, recover from
disturbances, and cross slopes and steps. Its footsteps are **not
scripted**. A single neural policy learns six actuator commands
through reinforcement learning (PPO) in thousands of parallel
MuJoCo/MJX physics worlds.

[**Watch the public robot**](https://mdt733.github.io/emu/) ·
[How the code works](docs/ARCHITECTURE.md) ·
[Train the next Pinnacle](docs/PINNACLE_CANDIDATE.md) ·
[Fresh install](docs/INSTALL.md) ·
[Current handover](HANDOVER.md)

## The idea in one minute

```text
Teacher samples a command and terrain challenge
                   |
                   v
MuJoCo/MJX simulates a six-servo robot in real contact geometry
                   |
       noisy IMU / optical flow / 4×4 ToF + joint measurements
                   |
                   v
             ONE neural policy
                   |
                   v
            six motor commands
                   |
                   v
Real motion, contact, balance, energy and falls are measured
              /                      \
       teaching reward          next lesson choice
              \                      /
                  PPO improves
                  the same policy
                       |
            checkpoint, independent exam,
                  browser showcase
```

Six **teaching focuses** practise commands, slopes, steps,
disturbance recovery, command transitions and economical
control. These are six groups of training worlds, **not six
networks**. Each also encounters easier combinations of the
other challenges.

The actor has **360 inputs** in the new research lineage:
90 deployable observation fields (including a 4×4 ToF depth
array and validity bits) across four recent frames. The
teacher can inspect exact terrain and contact to choose
lessons, but that hidden simulator truth **is not fed into
the neural controller**.

## What is current?

| | Published Pinnacle | Next candidate |
|---|---|---|
| Code | `main`, protected | `release/v5032-pinnacle-candidate` |
| Robot input | 232 channels | 360 channels |
| Training | Previously completed | **260M cold-start campaign prepared** |
| Status | Public trained showcase | **Not trained or promoted yet** |

The most recent completed v5031 experiments improved actual
2–2.5 cm upstep crossings and downhill control, but still
failed real 3 cm uphill crossings and did not prove sustained
1.0 m/s walking. The next candidate combines those lessons,
the proven 20M main-style foundation, and 25% fast-flat
gait practice in **one fresh training lineage**.

The ambition is a robot that is fast, controllable, resilient,
efficient and capable on real terrain—not one that merely
scores a larger training reward. Read the
[measured graduation criteria](docs/RELEASE_READINESS.md)
before comparing it with the current published robot.

## Find what you need in the code

| Question | Go here |
|---|---|
| What is the robot made of? | `emu/robot.xml` |
| What do its sensors and 360 inputs mean? | `emu/contracts.py` |
| How does gravity, terrain, contact and sensor noise work? | `emu/environment.py` |
| How does PPO know what is good? | `emu/rewards.py` |
| Who chooses the next lesson and difficulty? | `emu/curriculum.py`, `emu/capability_focus.py` |
| Where is measured training competence stored? | `emu/capability_field.py` |
| Where is the neural actor, critic and gradient update? | `emu/ppo.py` |
| Who runs the loop and protects checkpoints? | `emu/training.py`, `emu/run.py`, `emu/checkpoint_io.py` |
| How do independent physical exams work? | `emu/physical_exam.py`, `emu/evaluation.py` |
| How are results saved? | `emu/results.py` |
| How do I run ONE clean flagship experiment? | `scripts/train_pinnacle.py` |
| How does live browser playback work? | `emu/viewer_server.py`, `emu/viewer_src/` |
| How are the new policy and old main compared? | `scripts/compare_policies.py` |
| How is the public release approved? | `scripts/verify_showcase_release.py` |

**[Complete architecture, boundaries and data flow](docs/ARCHITECTURE.md)**
explains how these pieces connect and which files to edit.

## Three useful commands

Install Python 3.13, JAX CUDA, MuJoCo and test tools using
[the new-machine guide](docs/INSTALL.md). In a clean checkout
of the candidate branch:

```bash
# Print the source-pinned 260M training recipe without starting it
python scripts/train_pinnacle.py

# Once CI and GPU availability are verified: full CPU/CUDA
# preflight, followed by training in the foreground
python -u scripts/train_pinnacle.py --run

# Inspect a saved policy in an independent physical exam
emu capability-exam --policy PATH/TO/robot.emupolicy \
  --output PATH/TO/exam --device cpu --terrain-matrix

# Follow the latest saved checkpoint in a separate browser simulator
emu viewer --live \
  --root ~/robot_lab/emu_runs/v5032_pinnacle_s761_v2 \
  --no-open --port 8765
```

The launcher prints the exact Git SHA, seed and settings and
**refuses incompatible resume state**. Don't edit/pull an
active training checkout. Full flags, retries and acceptance
checks are in [PINNACLE_CANDIDATE](docs/PINNACLE_CANDIDATE.md).

## Checkpoints, tests, and the public bird

`.pkl` is a complete **resumable training state**: actor,
optimizer, observation normalizer, curriculum, metadata and
step. `.emupolicy` is the exported **deployable actor** with
its sensor contract and source identity—no PPO optimizer.
Training results live outside Git under
`~/robot_lab/emu_runs/` and need separate backups.

The browser viewer has live-checkpoint watching and a
graduated eight-challenge park. It is an interactive
diagnostic, **not** proof that an independent physical exam
passed. Exams assess achieved commands, supported crossing,
survival, steering and energy rather than requested difficulty
or reward alone.

The published `main` champion stays unchanged until a
specific trained candidate passes common physical exams,
browser and sensor checks, source/asset provenance,
explicit owner approval and the protected release workflow.
This branch and its [draft PR](https://github.com/mdt733/emu/pull/16)
prepare the replacement but **do not publish one**.

**Deferred after the candidate:** genuine supported 3 cm-plus
step-ups, curved/organic hills, and better terrain-aware
traversability. The plan is recorded in [HANDOVER](HANDOVER.md);
historical experimental code is available at its recorded Git SHAs,
not mixed into the one current launcher.

[Documentation index](docs/README.md) ·
[Project hygiene](docs/MAINTENANCE.md).
