# EMU handover — one known-good robot, one improving school

**Updated 23 November 2026.** This is the short entry point for resuming the
project. The task checklist stays in [issue #62](https://github.com/mdt733/emu/issues/62).
Detailed measurements, caveats and historical decisions belong in
[research findings](docs/RESEARCH_FINDINGS.md) and Git history, not a
second roadmap here.

## The two active source identities

| Lane | Purpose | Identity |
| --- | --- | --- |
| `main` | Known-good original H40, protected browser policy and approval | Original training commit `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`; source-pinned 120M seed735 actor `showcase/champion.emupolicy` |
| `research/h40-unified-school-cold-v1` | **One** active improvement branch; teach robust locomotion and stairs from birth | [Draft PR #103](https://github.com/mdt733/emu/pull/103); explicit Spring-only research physics and survival-aware Academy teacher |

**Do not create a new branch for every lesson idea.** Maintain this one
improvement branch against `main` until we have evidence worth promoting.
The old Spring-only, Balanced Stairs and Safe Ascent PRs are **completed
diagnostics**, preserved in Git, not active competing schools. Do not
replace the checked-in champion or publish to Pages without independent
qualification and owner approval.

**North star:** a coherent, repeatable **teacher** that produces the
strongest possible six-servo H40 policy **from random initial weights**,
across multiple independent training seeds, without hand-changing lessons
every 20M steps. Warm starts remain a **supported generic research tool**;
our present cold-study chooses not to use one.

## Actual system

- **Robot:** about 40 cm; six chassis-mounted 12 V C018 servo actuators,
  passive shins/knees and 36 mm ball feet. `h40` is the immutable
  mainline physical profile. The opt-in `h40-spring-only` research
  profile uses a calibrated 150 N/m passive knee rocker; hardware
  design and physical manufacture are **unproven**. Never conflate
  the two models.
- **Observation/action:** one 360-input four-frame IMU, optical flow,
  4×4 ToF, proprioception and commanded-movement contract → 256 →
  256 → **six motor position actions**. No privileged teacher-only
  position in the deployed actor.
- **World:** MuJoCo MJX rolling 2×2 metre-scale carpet with eight real
  signed convex prisms, randomized disturbances and sensor error.
  Actual contacts, not a fake box/height proxy, teach the robot. The
  browser's Infinite Meadow has *different* CPU/WASM geometry and is
  a demonstration and sensor inspection tool, **not** the stair exam.
- **Learning:** one PPO actor/critic/optimizer; one governor decides
  physical difficulty from completed experience. One
  `birth_lesson_weights` decision combines six focus-cohort
  allocation, retention, athletic rehearsal and Academy births.
  `CapabilityField` records command/recovery/transition experience;
  its current `shadow` setting **does not choose lessons**. Do not
  add another scheduler or neural teacher.
- **Academy:** supported first-lip crossing requires independent
  confirmed upper-tile support of **both feet**. Uphill graduation
  requires that crossing **and survival in the same completed life**.
  Uphill/downhill frontiers advance separately; keep ordinary
  command, flat, slope and recovery rehearsal. Strict global-XY
  parking is **not** required for dynamic competence. Ordinary
  **nominal** step cap 40 mm, Academy **nominal** aspiration 55 mm.
  The desired 8 cm uphill step is a stretch exam, **not yet taught
  intentionally at 8 cm**.
- **Evidence/outputs:** full optimizer, critic, normalizer, teacher,
  health, capability field, rollout metrics and provenance live in
  `~/emu-data/runs/` **outside Git**. Read-only physical examiners
  independently check signed 4–80 mm true crossings, first-life
  survival, commands, slope, balance, recovery and post-lip progress.
  A `.emupolicy` export cannot resume PPO.

See [core skeleton](docs/CORE_SKELETON.md) for module ownership,
[physical quality](docs/PHYSICAL_QUALITY.md) for acceptance,
[maintenance](docs/MAINTENANCE.md) for supported tests and
[release readiness](docs/RELEASE_READINESS.md) for the manual
showcase/Pages barrier.

## What's actually measured

- The original `main` H40 seed735 120M actor is protected:
  actor SHA256 `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673`.
  Original actor and training checkpoint are not interchangeable;
  the original complete `final.pkl` remains on the operator's PC.
  Viewer gait/slope impressions are qualitative, not hardware proof.
- Independent **evaluation seeds 1701 and 2701** of the warm KS150
  150M learner reproduced improved **4–40 mm uphill traversals** and
  dramatically better tall downhill survival, **but worse 44–56 mm
  uphill survival**. This is a reason to test the safety-aware teacher,
  **not** proof that it works cold. Exact tables and actor/physics
  fingerprints are in [research findings](docs/RESEARCH_FINDINGS.md).
- **Current user-local sanity trial:** separate `0→120M` cold school,
  seed735, from source commit
  `1fa832fc7cd9241e2772b44ad434613c6080adfa`, using
  `EMU_SCHOOL_STEPS=120000000`. Its run directory is
  `~/emu-data/runs/h40-school-cold-120m-s735/`. The owner reported
  it appeared healthy while running; **completion and qualification
  results have not yet been supplied**. It is a did-we-break-training
  check, not a teacher championship.
- **Branch continues to evolve during that trial.** Current
  `research/h40-unified-school-cold-v1` head is intentionally newer
  than the running student's source. Do **not** pull the moving head
  into a checkout used to resume the original trial. The trainer
  verifies source fingerprints. Assess the result using its own
  source-pinned config and checkpoint.
- The intended longer school experiment is **0→180M** with the same
  predefined teacher; new full cold seeds must use one frozen code/
  configuration/budget. Final physical exams use preregistered
  evaluation seed **4701** (1701/2701 were design-time evidence).
  An interesting single learner is not proof of teacher quality.

## Operator commands and next work

```bash
# Only in a SEPARATE checkout with no active/resumable GPU learner:
cd ~/emu
git fetch origin
git switch research/h40-unified-school-cold-v1
git pull --ff-only
source .venv/bin/activate

# Isolated from-birth 120M sanity budget:
export EMU_SCHOOL_SEED=735
export EMU_SCHOOL_STEPS=120000000
bash scripts/check.sh --full
bash scripts/run_h40_school_wsl.sh plan
bash scripts/run_h40_school_wsl.sh preflight
bash scripts/run_h40_school_wsl.sh gpu-smoke
# Only if all checks pass, and with the intended source SHA frozen:
bash scripts/run_h40_school_wsl.sh all
```

For a **new** `180M` run, unset `EMU_SCHOOL_STEPS`; its separate
default directory is `~/emu-data/runs/h40-school-cold-s735/`.
An interrupted student must resume at its original Git commit with
the exact seed/recipe, **never** at an arbitrarily newer branch head.

**Next priorities:** finish/inspect the 120M execution sanity run;
validate full-episode teacher signals (a two-frame CUDA smoke cannot
do that); remove duplicated or proven-unused scaffolding **in this
same research branch**; test candidate teacher improvements as
well-designed controlled ablations without changing a running
student; then repeat complete cold training across multiple seeds
against an equal-budget cold baseline. Maintain hardware skepticism.
No auto-promotion.

**Research principle:** one student, one teacher for its whole life;
one improving research branch; physical crossing **plus** survival and
retained walking matter more than reward or a pretty viewer clip.
