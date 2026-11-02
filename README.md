# EMU — a little robot learning to walk

**Six servos · One dynamically unstable biped · One learned policy**

EMU learns to balance, walk, turn, survive disturbances and negotiate
real slopes and steps. Its footsteps are **not scripted**: the robot
learns six servo commands using PPO reinforcement learning in
thousands of parallel MuJoCo/MJX physics worlds.

[**Open the EMU playground**](https://mdt733.github.io/emu/) ·
[**How the code works**](docs/ARCHITECTURE.md) ·
[**260M Pinnacle measurements**](docs/V5032_PINNACLE_RELEASE.md) ·
[**Fresh installation**](docs/INSTALL.md) ·
[**Current handover**](HANDOVER.md)

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
| **~20M** | Still **barely standing** in historical early-foundation observations. The 20M curriculum foundation cutoff is a **recipe boundary**, not a pass certificate for walking. |
| **~40M** | Early, **infant-like gait development** rather than mature controlled walking. |
| **Later** | Meaningful speed, directional control, slopes and supported step crossing must be verified on **frozen physical exams** at the same source, sensor contract and seed. |

Timing varies with seed, architecture and curriculum. **No global
mastery gate** should block all six specialist populations until
another specialty succeeds. Conversely, do not call a 131k-step
smoke policy broken because it falls: those tests validate physics,
PPO, memory and checkpoints, **not learned locomotion**.
See the [foundation / six-specialist curriculum](docs/UNIFIED_CAPABILITY_GROWTH.md)
and [active engineering handover](HANDOVER.md).

## Current baseline — v5032 Pinnacle 260M

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
| Who runs independent physical exams? | `emu/physical_exam.py`, `emu/evaluation.py` |
| Who serves the trained bird and browser MuJoCo? | `emu/viewer_server.py`, `emu/viewer_src/` |
| What makes a public policy release safe to publish? | `scripts/verify_showcase_release.py` |
| Where are the old experiments and next questions? | `docs/history/`, [issue #17](https://github.com/mdt733/emu/issues/17) |

[**The end-to-end architecture map**](docs/ARCHITECTURE.md)
explains each boundary: actor vs teacher, teaching vs independent
exam, saved learner vs exported actor, and local viewer vs Pages.

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

# Independent frozen physics challenge, no PPO training.
python -m emu.cli capability-exam \
  --policy showcase/champion.emupolicy \
  --output ~/robot_lab/emu_exams/pinnacle_260m \
  --device cpu --quick
```

The viewer offers live-checkpoint following and eight graduated,
real-collision park obstacles. The park is an interactive
diagnostic, **not** a held-out MJX physical certification.

### Want to train a new robot?

The completed v5032 launcher is preserved at its original source SHA
`9b11bbe596aea7201d29a6beb2d192922f8e2e9a`, together with its
[historical runbook](docs/PINNACLE_CANDIDATE.md). It no longer lives
in active `scripts/`. New source-pinned robots begin with `emu train`.
To reproduce the exact released training source, use its original
frozen Git SHA and independently backed-up full `final.pkl`
checkpoint. Don't rerun a 260M campaign or transplant an
optimizer merely because a branch exists.

A `.pkl` holds optimizer, normalizer and curriculum state;
an `.emupolicy` is an inference-only actor with a sensor
contract. Full checkpoints are not in Git; keep external backups.

## What's next?

The experimental child branch
[`feature/v5041-unified-terrain-carpet`](https://github.com/mdt733/emu/tree/feature/v5041-unified-terrain-carpet)
adds **Training Carpet** to the existing 3D robot viewer without replacing
the original Terrain Park. The viewer uses its own finite browser MuJoCo
hfield scene; the **GPU trainer has a different real collision implementation**:
a recycled 3×3 ring of nine tilted solid-box tiles. Both show the same
terrain concepts, but they are **not geometrically identical**.
See [viewer runbook](docs/V5041_TRAINING_CARPET_VIEWER.md) and
[real GPU trainer runbook](docs/V5041_GPU_CARPET_TRAINING.md).


The active research branch is
[`feature/v5040-multi-lip-academy-world`](https://github.com/mdt733/emu/tree/feature/v5040-multi-lip-academy-world):
the STEP specialist now uses a three-platform streaming course with
endlessly random UP/DOWN steps, independent difficulty frontiers and
physically supported progress rewards.
It uses the same actor and retains the general curriculum, but has not
yet been GPU-profiled or trained. See the [v5040 guide](docs/V5040_REAL_MULTILIP_ACADEMY.md).
The released 260M policy on `main` is unchanged.

Our [research backlog](https://github.com/mdt733/emu/issues/17)
starts with better downhill/path steering measurements, real
2.5–3 cm support transfer, torso crouch and stored spring
energy, fast controllable gait and meaningful J/m. Organic
terrain comes later. Future work gets one clear hypothesis
and a comparable physical exam before another full training.

[Documentation index](docs/README.md) ·
[Maintenance principles](docs/MAINTENANCE.md).

### v5041 GPU trainer experiment (new source, not released)

The opt-in `emu train --terrain-carpet` path uses **nine real tilted
solid boxes** with vertical contact sides and a safely recycled 3×3
window in each MJX world. It retains source-sampled two-triangle tile
heights for terrain generation, but the actual feet touch box geometry.
All six teaching specialties share one PPO actor with 360-input
4×4 ToF/IMU/optical-flow observations; a 64-tile teacher visit memory
prevents repeated crossing reward from rocking backward and forward.

**GPU engineering passed on 3 November 2026:** 2,048 real CUDA worlds,
full 32-frame / four-epoch PPO updates, sustained **16,818 world
steps/s** (three synchronized post-compile updates), JAX peak
allocation **1.40 GB**, and a checkpoint/resume/export lifecycle.
A frozen first-life nine-box exam also ran on CUDA: raw-contact
independent scorer **0 UP / 1 DOWN** and teacher **0 UP / 1 DOWN**
on eight worlds; **all eight fell** on the untrained 131k-step actor.
These results validate engineering and exam plumbing, **not learned
walking or step mastery**. The historical cold-start learning curve
is documented above; a fresh long run must demonstrate it anew.
See [GPU trainer runbook](docs/V5041_GPU_CARPET_TRAINING.md).
