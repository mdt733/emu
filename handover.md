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
- **Finished user-local 0→120M sanity trial:** seed735, source
  `1fa832fc7cd9241e2772b44ad434613c6080adfa`, **120,061,952**
  steps / 2,048 worlds, no external parent, no recorded resume.
  Operator supplied `school-evidence.zip` with clean 120M run and
  three independent read-only physical exams. Full checkpoints remain
  outside Git at
  `~/emu-data/runs/h40-school-cold-120m-s735/`.
  **Pipeline succeeded; teacher still struggles uphill.** Fixed
  350-frame core/dynamic/combined survival each 100%; 30/60s deep
  trials survive 8/8 across scenarios, but command missions
  (especially lateral/reverse/stop) frequently fail.
  In the signed 4–80mm test (50 trials/height/direction, seed4701):
  controlled traversal **UP 39/500 at 4–40mm and 0/500 at 44–80mm**;
  **DOWN 138/500 at 4–40mm and 71/500 at 44–80mm**. Uphill
  Academy frontier reached just 3.5mm three times then retreated to
  **3mm**; downhill reached **55mm**. Full source/evidence audit in
  [research findings](docs/RESEARCH_FINDINGS.md). This is **a
  successful execution shakeout, not a proven good teacher**.
- **Revised 120M cold student (not promoted):** seed **736**, source
  `4048e8450c36882501cbb624f4ca656e11da9419`, reached
  **120,061,952** steps and completed held-out physical exams at seed
  **5701**. Signed five-frame traversals: **uphill 64/500** at
  4–40 mm and **0/500** at 44–80 mm; **downhill 132/500** and
  **84/500** respectively. The uphill Academy frontier ended
  **3.5 mm**, downhill **55 mm**. Short pulse command scores improved
  in some tasks, but long missions and station keeping remain weak.
  Different training/evaluation seeds and teacher sources mean this
  is not controlled proof of improvement, nor an actor qualified to
  replace the original H40 on `main`. Results and caveats live in
  [research findings](docs/RESEARCH_FINDINGS.md).
- **Source isolation:** the finished seed-735 run is pinned to
  `1fa832fc7cd9241e2772b44ad434613c6080adfa`; revised seed-736
  is pinned to `4048e8450c36882501cbb624f4ca656e11da9419`.
  Neither can be resumed from a newer cleanup commit. Trainer
  source fingerprints protect checkpoint continuity.
- The intended longer school experiment is **0→180M** with a
  predeclared, unchanged teacher; new full cold seeds require one
  frozen code/configuration/budget. Exam seed **5701** was first held
  out for the revised seed-736 run and is now **inspected evidence**:
  use a new preregistered unseen exam seed for future revisions.
  A single successful learner is not proof of repeatable teacher quality.

## Operator commands and next work

```bash
# In the research checkout: reproducible baseline checks always select
# original h40 inside check.sh (no caller-shell profile mutation).
cd ~/emu
bash scripts/check.sh --full

# Run only against the appropriate frozen source identity:
# seed-735: 1fa832fc7cd9241e2772b44ad434613c6080adfa
# seed-736: 4048e8450c36882501cbb624f4ca656e11da9419
# Do not resume completed training from the moving branch head.
```

**Next priorities:** preserve both complete checkpoint/export identities
and the external evidence bundle; perform a matched authenticated
original-H40-versus-Spring-only task comparison for *selection*, not
causal teacher attribution. The revised teacher's downstream-descents
correction is still a **hypothesis**: small uphill traversals increased
in one revised cold student, but the uphill Academy frontier remains
near 3 mm and no 44–80 mm uphill traversal was achieved.
For a future cold replicate, pre-register an unused training seed,
one frozen source/config/budget and an unseen qualification seed.
No actor promotion or Pages release without separate owner approval.

**Research principle:** one student, one teacher for its whole life;
one improving research branch; physical crossing **plus** survival and
retained walking matter more than reward or a pretty viewer clip.
