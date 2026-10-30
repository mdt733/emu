# EMU — learning to walk, recover and negotiate real terrain

**Six servos · one dynamically unstable biped · one PPO policy · measurable physical progress**

![EMU learning architecture](docs/images/emu-architecture.svg)

[Public EMU playground](https://mdt733.github.io/emu/) ·
[**Next Pinnacle candidate**](docs/PINNACLE_CANDIDATE.md) ·
[**Fresh install / GPU recovery**](docs/INSTALL.md) ·
[Engineering handover](HANDOVER.md) ·
[Learning architecture](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Release rules](docs/RELEASE_READINESS.md)

## Current status — 31 October 2026

The published `main` repository and GitHub Pages champion remain **protected**.
The old public actor has 232 observation inputs; it is not silently
replaced by experiments.

The development successor is a **360-input** neural locomotion policy with
an integrated forward 4×4 terrain ToF array, optical flow and inertial/
proprioceptive measurements. It learns to command six force-limited
servos without hand-programmed footstep animation. Its 360 inputs and
six outputs are versioned and verified by the native simulator and
browser exporter.

The new clean flagship branch is
**`release/v5032-pinnacle-candidate`**. It has a **single
source-pinned 260M-step cold-start launcher**, not a stack of
cross-revision experiment forks. Source preparation does **not** mean
that this candidate has completed training or replaced `main`.

### Last proven checkpoint findings

The completed v5031 B100/C100 A/B used a common B80 checkpoint
and 100M training steps. Against identical real MJX slabs:

| Frozen physical test | B100 control | C100 verified support |
|---|---:|---:|
| Supported uphill 2.0 cm | 7/8 | **8/8** |
| Supported uphill 2.5 cm | 2/8 | **4/8** |
| Supported uphill 3.0 cm | 0/8 | 0/8 |
| Supported downhill 3.5 cm | 1/8 | **7/8** |
| 12° downhill mission | 1/16 | **13/16** |
| Flat 0.8 m/s command — 700-frame survival | 9/16 | **12/16** |
| Flat 1.0 m/s command — 700-frame survival | 0/16 | 0/16 |

Broad fixed-evaluation scores were nearly tied
(B=0.77593, C=0.77503), and C had higher rollout fall rate/shaft
work. C's supported-terrain advantage is evidence worth exploring,
**not** unconditional proof that it is the best released model.

The new flagship starts from **random weights** and combines the
proven 20M main-style foundation, six independent but shared-policy
teaching emphases, 25% real fast-flat gait rehearsals, a blended
25-frame speed tracking reward and the verified-support ascent
teaching. Its longer training outcome remains to be measured.

## What EMU learns

| Curriculum focus | Actual evidence of competence |
|---|---|
| **Commands** | Forward/reverse/sideways/yaw motion matches requests |
| **Slopes** | Uphill, downhill and camber with supported, controlled progress |
| **Steps** | Real crossing to the far-side surface with foot contact, separately UP and DOWN |
| **Resilience** | Continued locomotion and recovery under physical disturbance |
| **Transitions** | Changing command targets without instability or time-wasting |
| **Control / efficiency** | Smooth useful movement with measured shaft work and slip |

The trainer uses genuine colliding support geometry, not privileged
terrain readings supplied to the actor. The teacher may inspect
simulator collision contacts to judge success; the neural input
comes from its independently simulated sensors.

**Never confuse difficulty with mastery.** A 15° sampled hill,
a 3-cm curriculum setting or a high requested velocity is not a
successful physical mission. Independent exams measure survived
distance, measured velocity, drift, support contacts, cross-frontier
retention and energy proxies.

## The only active training recipe

```bash
# In a clean checkout of release/v5032-pinnacle-candidate,
# with its installed Python 3.13 CUDA+dev environment:
python scripts/train_pinnacle.py          # print plan; no files/GPU
python -u scripts/train_pinnacle.py --run # CPU+CUDA preflight, then cold 260M
# After interruption, SAME exact Git SHA and matching preflight stamp:
python -u scripts/train_pinnacle.py --run --skip-preflight
```

The launcher writes into
`~/robot_lab/emu_runs/v5032_pinnacle_s761/`, outside Git,
and eventually exports a **candidate** `.emupolicy`, a
SHA-256 provenance report and independent speed/step/slope exams.
It **never** promotes or publishes a new champion on its own.

Use [the complete candidate runbook](docs/PINNACLE_CANDIDATE.md)
rather than improvising checkpoint resumes. Fresh machines should
start with [INSTALL](docs/INSTALL.md). Recorded source SHA and
unmodified checkpoints are required for continuation.

## Live browser and obstacle playground

The browser viewer has full six-servo MuJoCo inference,
live-checkpoint follow, commands/joystick, 4×4 ToF overlays,
an autonomous pose-lattice navigator and an expanded graduated
obstacle park. The park's central optical-flow/heightfield parity
and crest contact seam have been corrected.

The park provides **eight real-contact challenges** with Gentle,
Balanced and Stretch heights tailored to the 100M research frontier.
The 3D signposts are non-colliding visual guides, not policy inputs.
These explorations are **not** interchangeable with held-out MJX
physical exams.

```bash
python -m emu.cli viewer-build
python -m emu.cli viewer --live \
  --root ~/robot_lab/emu_runs/v5032_pinnacle_s761 \
  --no-open --port 8769
```

For public deployment, the previously reviewed
`showcase/champion.emupolicy` remains unchanged. A future candidate
needs explicit owner approval, common physical comparisons, signed
source/policy/checkpoint provenance, browser validation and the
protected GitHub Pages release workflow before replacing it.

## Code map

```text
emu/
  environment.py        MJX physics, 4x4 ToF, optical flow, real support evidence
  rewards.py            one actor's training payments
  curriculum.py         adaptive six-focus physical-growth lessons
  ppo.py                shared actor, value network, optimizer
  run.py, training.py   guarded checkpoint-provenanced trainer
  checkpoint_io.py      strict source/sensor/optimizer identity and exporter
  physical_exam.py      read-only independent directional and terrain exams
  viewer_server.py      validated browser policy/live checkpoint serving
  viewer_src/           browser MuJoCo and graduated playground
scripts/
  train_pinnacle.py     ONE active cold-start 260M training campaign
  campaign_support.py  foreground logging, results, ETA, source evidence
  smoke_unified_focus.py
  smoke_training_lifecycle.py
  prepare_showcase.py, verify_showcase_release.py
docs/
  PINNACLE_CANDIDATE.md  current training and release runbook
  INSTALL.md             new Windows/WSL/CUDA machine
  RELEASE_READINESS.md   measured release gates + rollback
  VIEWER_GRADUATED_PARK.md
  history/               archived experiments; historical exact SHAs
```

## Deferred work

Do not interrupt this flagship candidate to implement another
step-up reward or organic hill curriculum. Once it is measured,
priorities are: **supported 3+ cm step-up**, physical
flat→incline→flat and irregular-ground transitions, robust
sensor-guided traversability/refusal, and replicated gait/
efficiency improvement. These remain documented research goals,
not hidden modifications to the current run.

[More documentation](docs/README.md) · [Physical capability dimensions](docs/CAPABILITY_DIMENSIONS.md).
