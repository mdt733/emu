# EMU — a little robot learning to walk

**Six servos · One dynamically unstable biped · One learned policy**

EMU learns to balance, walk, turn, survive disturbances and negotiate
real slopes and steps. Its footsteps are **not scripted**: the robot
learns six servo commands using PPO reinforcement learning in
thousands of parallel MuJoCo/MJX physics worlds.

[**Open the EMU playground**](https://mdt733.github.io/emu/) ·
[**Run-to-viewer core skeleton**](docs/CORE_SKELETON.md) ·
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
| **~20M** | The v5046 source-pinned 20M trial now shows reliable **low-speed** held-out survival and ~0.094 m/s achieved mean forward speed; this is not yet champion-equivalent or verified step-climbing. Earlier generations learned more slowly. |
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

## Current development — engineering promotion preparation

The current GPU experiment is **v5048**, pinned to original source
`088a29be47f624be9e2c322c504ec98049b3f9bf` on
[draft PR #38](https://github.com/mdt733/emu/pull/38).
It is training **one 50M-step foundation**, then eight matched
**50M-step continuations of the SAME checkpoint**, selecting the
best physically evaluated policy for **another 100M steps**.
The proposed 550M total is a research plan, **not a completed
200M gait or proven stair-climbing result**.

The new runtime has four recycled **75cm** squares, each with two
actual convex triangular-prism colliders (**eight** in all),
six specialist teaching cohorts and one 360-input neural policy.
A v5046 34-arm solver investigation measured steady throughput
of **31,974 world-steps/s** at reference solver settings versus
**119,662 world-steps/s** with experimental Newton/linesearch 2/4
on an RTX 4060 Ti. The 2/4 physics fidelity still requires
contact/penetration and CUDA reset-parity assessment.
[Measured engineering results](docs/V5046_OVERNIGHT_RESULTS_20261105.md).

The separate
[future-main release-prep branch](https://github.com/mdt733/emu/tree/release/v5049-main-promotion-prep)
retains the production CLI and tests while removing disposable
v5045/46/48 experiment launchers, stale historical runbooks and
duplicate workflow code. **It does not modify the running v5048
source**, and its cleanup is not an actor training continuation.

We plan **two explicit approvals**: upgrade the engineering
architecture after fresh CI/CUDA verification, and *separately*
consider replacing the published 260M champion after independent
cross-seed physical gait exams and browser proof. Until both
have been evaluated, `main` and the approved public actor
remain unchanged. [Promotion gates and rollback](docs/RELEASE_READINESS.md).

[Documentation index](docs/README.md) ·
[Maintenance principles](docs/MAINTENANCE.md) ·
[Research backlog](https://github.com/mdt733/emu/issues/17).
