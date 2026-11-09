# EMU — a little robot learning to walk

**Six servos · One dynamically unstable biped · One learned policy**

EMU learns to balance, walk, turn, survive disturbances and negotiate
real slopes and steps. Its footsteps are **not scripted**: the robot
learns six servo commands using PPO reinforcement learning in
thousands of parallel MuJoCo/MJX physics worlds.

[**Open the EMU playground**](https://mdt733.github.io/emu/) ·
[**Run-to-viewer core skeleton**](docs/CORE_SKELETON.md) ·
[**200M released policy / measured limitations**](docs/V5048_SHARED_PARENT_RESULTS_20261105.md) ·
[**Original 260M Pinnacle baseline**](docs/V5032_PINNACLE_RELEASE.md) ·
[**Fresh installation**](docs/INSTALL.md) ·
[**Current handover**](handover.md)

## The idea in one minute

```text
Teacher selects a command and a physical challenge
                       |
                       v
       MuJoCo / MJX simulates body, terrain,
             six servos and ground contact
                       |
                       v
     Noisy IMU + optical flow + forward 4×4 ToF
              + proprioception + command
                       |
                       v
            ONE neural-network policy
                       |
                       v
                Six servo targets
                       |
                       v
     Measured motion, support, efficiency, falls
                       |
                +------+------+
                |             |
          PPO reward      Next lesson
                |             |
                +------+------+
                       |
                 New shared weights
                       |
              checkpoint → real exam
                       |
                    viewer
```

Six *teaching focuses* practise commands, slopes, supported steps,
resilience, command transitions and economical control. They are
different groups of **training environments**, not six networks.

The current policy observes **360 inputs**: 90 deployable sensor
and command fields across four history frames, including 4×4
forward ToF range/validity signals. The teacher can inspect
precise simulator contacts to decide whether a skill succeeded,
but those privileged facts are **not actor inputs**.

## Cold-start learning curve — don't mistake age for competence

EMU starts with **random policy weights**; it is a dynamically unstable
six-servo biped, not a pre-programmed standing or walking controller.
The following is the owner's **historical, approximate observation**,
not a source-pinned, controlled learning-curve measurement:

| Total world steps | What is reasonable to expect |
|---|---|
| **~0–5M** | Discovery period: generally falls; even standing on flat ground is not yet reliably learned. |
| **~5M** | First signs of learning to **stand on boring, flat ground**; not proof of reliable balance, gait or terrain skills. |
| **~20M** | The v5046 source-pinned 20M trial now shows reliable **low-speed** held-out survival and ~0.094 m/s achieved mean forward speed; this is not yet champion-equivalent or verified step-climbing. Earlier generations learned more slowly. |
| **~40M** | Early, **infant-like gait development** rather than mature controlled walking. |
| **Later** | Meaningful speed, directional control, slopes and supported step crossing must be verified on **frozen physical exams** at the same source, sensor contract and seed. |

Timing varies with seed, architecture and curriculum. **No global
mastery gate** should block all six specialist populations until
another specialty succeeds. Conversely, do not call a 131k-step
smoke policy broken because it falls: those tests validate physics,
PPO, memory and checkpoints, **not learned locomotion**.
See the [foundation / six-specialist curriculum](docs/UNIFIED_CAPABILITY_GROWTH.md)
and [active engineering handover](handover.md).

## Historical reference — v5032 Pinnacle 260M

<small>First released 360-input generation · 31 October 2026</small>

| Attribute | Released baseline |
|---|---|
| Policy | **v5032 Pinnacle, 260,046,848 world steps** |
| Source used for training | `9b11bbe596aea7201d29a6beb2d192922f8e2e9a` |
| Inputs / outputs | 360 observation inputs / six servo targets |
| Run | 2,048 parallel GPU worlds, seed 761, cold start |
| Published actor | `showcase/champion.emupolicy` |
| Reproducibility | Export identity, sensor schema and checkpoint hash in `showcase/release_approval.json` |
| Previous public champion | 232-input policy at Git commit `2ec9e9db0a816c658131311129ef89ec7a595a12` |

It **physically crosses** 2 cm upsteps in 8/8 worlds, but
only 3/8 pass the **full movement-and-heading mission**.
At 2.5 cm the supported crossing rate is 4/8 (mission 3/8);
at 3 cm, supported ascent is still 0/8. It survives 12° downhill in all 16 frozen
worlds but passes the full *heading-controlled mission* in
only 4/16. At a requested 0.8 m/s, 12/16 survive 14 seconds,
but none pass the strict full heading/path mission.

That's a useful, measured new baseline—not a declaration
that the research is finished or that the new model surpasses
the old one on **every** test.

[**Full measured release card, known limitations and rollback**](docs/V5032_PINNACLE_RELEASE.md) ·
[**Actual crossing vs controlled mission baseline**](docs/PHYSICAL_BASELINE_260M.md) ·
[**Prioritised future investigations**](https://github.com/mdt733/emu/issues/17)

## Where to find the code

| Question | File owner |
|---|---|
| What's the robot physically made of? | `emu/robot.xml` |
| What does it see? | `emu/contracts.py` |
| How do terrain, springs, physics and sensor noise work? | `emu/environment.py`, `emu/domain_randomization.py` |
| What does PPO pay for? | `emu/rewards.py` |
| Who selects the next lesson/difficulty? | `emu/curriculum.py`, `emu/capability_focus.py` |
| Who monitors PPO health and recovery? | `emu/training_health.py` |
| What keeps physical competence evidence? | `emu/capability_field.py` |
| Where is the actor, critic and update? | `emu/ppo.py`, `emu/training.py` |
| Who validates checkpoints and source lineage? | `emu/run.py`, `emu/checkpoint_io.py` |
| Where are install diagnostics? | `emu/doctor.py` |
| Who runs independent physical exams? | `emu/prism_exam.py` (raw physical prism contacts), `emu/evaluation.py` (training's fixed held-out panels) |
| Who serves the trained bird and browser MuJoCo? | `emu/viewer_server.py`, `emu/viewer_src/` |
| What makes a public policy release safe to publish? | `scripts/verify_showcase_release.py` |
| Where are the old experiments and next questions? | Pinned Git history, [issue #17](https://github.com/mdt733/emu/issues/17) |

[**The five-stage core skeleton and cleanup plan**](docs/CORE_SKELETON.md)
is the current entry point from source-pinned `emu train` through
procedural MJX physics, PPO, physical exams, export and local viewer.
The [original architecture in Git](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/ARCHITECTURE.md) remains
available for preserved legacy interfaces.

## Three useful things to run

After following [INSTALL](docs/INSTALL.md) and activating the
checkout's Python environment:

```bash
# Check the checked-in public model, sensor contract and exact approval.
python scripts/verify_showcase_release.py

# Build and view the released actor in the local browser.
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy \
  --no-open --port 8765

# The shipped champion uses historical training physics. Its original
# physical exams live in the pinned v5044/pre-cleanup Git history;
# never substitute modern prism contacts and call it a matched baseline.
```

The viewer offers live-checkpoint following and eight graduated,
real-collision park obstacles. The park is an interactive
diagnostic, **not** a held-out MJX physical certification.

### Want to train a new robot?

The completed v5032 launcher is preserved at its original source SHA
`9b11bbe596aea7201d29a6beb2d192922f8e2e9a`, together with its
[historical runbook in Git](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/PINNACLE_CANDIDATE.md). It no longer lives
in active `scripts/`. New source-pinned robots begin with `emu train`.
To reproduce the exact released training source, use its original
frozen Git SHA and independently backed-up full `final.pkl`
checkpoint. Don't rerun a 260M campaign or transplant an
optimizer merely because a branch exists.

A `.pkl` holds optimizer, normalizer and curriculum state;
an `.emupolicy` is an inference-only actor with a sensor
contract. Full checkpoints are not in Git; keep external backups.

## Current development — v5048 200M software release

The **v5048 GPU experiment has completed 550M unique steps**,
pinned to original training SHA
`088a29be47f624be9e2c322c504ec98049b3f9bf` on
[research PR #38](https://github.com/mdt733/emu/pull/38).
It trained one shared 50M foundation, eight matched 50M forks and
continued the automatic temporal-speed finalist to **200,015,872
steps**, with **148 independent GPU prism-exam panels**.
[Measured v5048 findings](docs/V5048_SHARED_PARENT_RESULTS_20261105.md).

Early dedicated step training increased distinct worlds with actual
≥2cm *supported uphill contact* on nominal 3cm procedural terrain:
**2/32** without Academy versus **19/32** early-lips or **21/32**
fine-frontier. But full actual 3cm ascent and sustained controlled
post-step travel are **not yet demonstrated**. The 200M finalist
achieved **0.595m/s** at a requested 0.70m/s, with **31/32**
fast-survival worlds, but retained ~12cm zero-command drift
over ~10 seconds and no gain in ≥2cm uphill incidence after 100M.

The new runtime has four recycled **75cm** squares, each with two
actual convex triangular-prism colliders (**eight** in all),
six specialist teaching cohorts and one 360-input neural policy.
A v5046 34-arm solver investigation measured steady throughput
of **31,974 world-steps/s** at reference solver settings versus
**119,662 world-steps/s** with experimental Newton/linesearch 2/4
on an RTX 4060 Ti. The 2/4 physics fidelity still requires
contact/penetration and CUDA reset-parity assessment.
[Measured engineering results](docs/V5046_OVERNIGHT_RESULTS_20261105.md).

The cleaned engineering architecture entered `main` through
[PR #39](https://github.com/mdt733/emu/pull/39). The owner also
tested the exported v5048 temporal-speed 200M policy in the viewer
and approved it as a **software-playground milestone** through
[PR #40](https://github.com/mdt733/emu/pull/40).
That decision does not claim matched-v5032 speed/steering mastery,
a full actual 3cm ascent, quiet standing or validated hardware
physics. The previous 260M released policy is retained in Git
history for rollback. The final training `final.pkl` was not in
the evidence archive: the release approval explicitly records
this and pins the actual exported actor, training source, sensor
and 148-panel examination archive instead.
[Release provenance and limitations](docs/RELEASE_READINESS.md).

[Documentation index](docs/README.md) ·
[Maintenance principles](docs/MAINTENANCE.md) ·
[Research backlog](https://github.com/mdt733/emu/issues/17).

## CPU-only 40 cm sensor calibration (not a trained gait)

The h40 sensor laboratory is a **local browser/MuJoCo-WASM** preview; it does
not require a CUDA GPU, a Python ML installation, a training checkpoint, or a
separate exported actor. It loads the actual rated-torque h40 robot geometry
from `robot.xml` via `physical_variants.py`, with **no neural inference**.

```bash
python -m emu.cli viewer-build
python -m emu.cli viewer --sensor-lab --port 8765
```

Run from the checked-out repository, with Node/npm available. On a clean CPU
machine use the [sensor-audit maintenance branch](https://github.com/mdt733/emu/tree/maintenance/v5058-clean-skeleton-sensor-audit);
`python` only needs the standard library for this mode. Visit
`http://127.0.0.1:8765/` if your browser does not open automatically.

The lab's rear chassis-to-hip bracket geometry is shown as three red rails
that cover the original **19 mm** visual gap introduced by the h40
centre-of-mass compensation. They have **zero added mass and no contacts**;
they illustrate required mechanical mounting rather than silently changing
the baseline physics. Real brackets must eventually have measured mass
and inertia.

The lab opens a **static 2×2 arrangement of four 1 m tiles**, with eight
*real convex triangular-prism* CPU collision meshes, and a 40 cm neutral-pose
robot. The scene starts paused for stable measurement; **Run physics** enables
untrained position-servo motion. Open the Sensor inspector to adjust the
VL53L8CX 45°×45° optical preview's pitch, yaw, forward/lateral and vertical
mounting position. **Frame feet** starts at 75° down / 45 mm above torso
centre / 0 mm forward, aiming for the bottom row to show
`ground / FOOT / FOOT / ground`. Use 5 rays per 4×4 zone, watch the live
foot-blocked cells and copy the raw scan/configuration back for calibration.

This is a geometry/occlusion inspection instrument, **not** an MJX terrain
recycler, a corrected training sensor contract or a proof of walking.
Training continues on the original 75 cm v5058 source until explicitly
revised and independently retrained.
