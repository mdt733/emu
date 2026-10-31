# EMU — unified capability learning contract

**Mission:** one robot, one neural policy, one PPO optimiser, many complementary
physical lessons. The goal is not the highest reward or one-dimensional speed.
It is useful, accurate, supported, recoverable, mechanically reasonable
locomotion across commands, slopes, steps, disturbances and transitions.

**Physical dimension inventory:** [CAPABILITY_DIMENSIONS.md](CAPABILITY_DIMENSIONS.md)  
**Current engineering state:** [HANDOVER.md](../HANDOVER.md)  
**Complete cold run, resume and extension commands:** [INSTALL.md](INSTALL.md)

## Design in one diagram

```text
Real MJX robot + real contacts + delayed 4×4 sensor
                     |
                ONE actor/PPO
                     |
          ONE source-pinned checkpoint
                     ^
                     |
     Local physical episode evidence ledger
          |          |          |
      commands    terrain    recovery/...     [many skill families]
          |          |          |
     tiny target nudges from local evidence
                     |
         sampler: easy / near / hard lessons
                     |
            Back to real MJX worlds

Frozen independent physics examiner: measures competence, never
silently grants curriculum promotions or rewrites trained policy.
```

No separate policy per skill, no privileged ground truth in the actor
observation, no new timer every time someone thinks of another terrain.

> **Current implementation (31 October 2026):** The
> [v5032 Pinnacle candidate](PINNACLE_CANDIDATE.md) is one
> 360-input, 260M-step cold-start run with a 20M foundation.
> The old 220/232-input actor contracts remain for the protected
> public baseline and independent comparisons until explicitly
> approved for retirement.

## The autonomous target rule

Each capability has a **training target**, not a mastery level. Its teacher
samples easier practice, around-target episodes and occasional more difficult
probes, attributes completed physical outcomes to real difficulty and
direction, and periodically takes one small step:

- **Near-target success:** increase difficulty a little.
- **Hard exploration failure:** hold; keep learning, do not panic.
- **Sustained genuinely easier failure:** ease the target downward.
- **No adequate evidence:** hold and collect more, rather than inventing
  competence.

Small steps are the low-pass filter. Individual dimensions must not wait
for unrelated skills or an unrelated number of PPO updates. A temporary
decline is ordinary infant learning, not automatically a collapse. There is
separate emergency protection for invalid PPO updates, gross deterioration
and broken/invalid checkpoints.

Do not freeze difficult problems out: maintain easy/near/stretch practice
even when the current target retreats. Never use reward alone as a
certificate of physical competence; measure achieved signed velocity,
contact, support, falls, crossing completion and mechanical effort.

## Why one engine / two profiles, not "90% command + 10% grace"

Historical source reference: unchanged `main` at commit
`2ec9e9db0a816c658131311129ef89ec7a595a12`. The first 70 PPO updates
were *steady-only*, but after that `main` allocated ordinary transition
and resilience regimes based on their measured fall rates. Its four base
difficulty scalars continued adapting. Hence the mathematically correct
foundation profile is **main-style ordinary regime sampling**, not an
invented 90/10 command/grace specialist allocation. The latter would
change the lesson distribution and introduce motor-cost shaping before
we have any parity evidence.

The **foundation profile** keeps the same eight baseline reward weights,
task envelope and PPO settings. Its `main` recipe disables all new specialist
births/frontier decisions and terrain; the optional `curious` experiment
activates a small (~5% of all worlds) easy grace cohort after the original
70-update steady bootstrap. Both use the SAME learner and actor.
The **growth profile** uses the same robot and PPO, gradually activates
focused worlds and their local physical targets. An independent main
checkpoint/eval reference is still needed for *behavioural*, not numeric,
parity because the new actor observes a 360-channel terrain-aware sensor
rather than main's historical actor contract.

Three seeds on a paired frozen exam are a good *offline promotion gate*
before retiring the original training route, not a runtime dependency:
a particular learner never has to wait for three different seed runs
or require a checkpoint-era permit before it can keep learning.

## Cold foundation and the six emphases

A new policy needs to discover standing/stepping from nothing. Retain the
original main-style simple gait learning and its proven physical rewards.
During the configured **10M or 20M foundation world steps**, specialist
emphases are inactive. During the **following 10M**, focused lessons linearly
blend from 0 to **full activation inside their allocated birth cohorts**.
Historical studies tried both 10M and 20M. The **current Pinnacle
candidate uses 20M**. Thus a 20% command allocation eventually
means about 20% active command-specialist worlds, not 7%. Every active world still practises non-focus challenges at about half
their current target severity (control/grace at about one-quarter).
This is teaching emphasis, **not** a declaration of capability.

From then on the six emphases (command, resilience, slope, step, transition,
control/efficiency) share ONE network and learn concurrently. Typical cohort
allocation is 20/20/20/20/10/10 percent, with the remainder of each cohort
retaining simpler gait learning during the activation ramp and via easier
cross-training in every specialist world after the ramp.
Specialist limits are driven by their *own* completed physical evidence.

The original first 70 PPO updates of simple gait remain a useful reference
inside the foundational controller. No code may require inheriting an exact
20M or 40M checkpoint from a named source SHA; those were experiments,
not milestones or an irreversible graduation process.

## Roles of the code

- `emu/environment.py`: single real robot, task sampler, two true solid
  incline/step collision slabs, real supported crossing and MJX telemetry.
  It also owns delayed array sensor observations and per-episode metrics.
- `emu/capability_focus.py`: compact fixed role allocation and smooth
  post-foundation exposure fraction. A role label is **teacher-only**.
- `emu/results.py`: translate actual completed episode measurements into
  signed direction/severity outcome counters, with UNKNOWN rather than
  fabricated success when no examples exist.
- `emu/curriculum.py`: original main-style gait regime governor, plus
  separate small target controllers. The redundant post-decision global
  growth-budget arbiter has been removed.
- `emu/training.py`, `emu/checkpoint_io.py`, `emu/run.py`: one source,
  one PPO optimiser, checkpointed teacher/health/normaliser/field, validated
  same-source recovery and restart, and strict fresh-run launch discipline.
- `emu/evaluation.py` and independent capability exams: frozen comparable
  physical exams, not another hidden curriculum governor.

Detailed measured-versus-proposed frontiers, including the fact that
**four measured slope directions still share one angle** and **step
obliquity of ±45° is not yet a trained frontier**, live in
[CAPABILITY_DIMENSIONS.md](CAPABILITY_DIMENSIONS.md). Do not exaggerate them
as independent working skills.

## Non-regression contract — what must never be traded away

Preserve all of the following unless a deliberate, matched experiment
shows a better alternative:

1. A fresh random-seed policy can learn from zero on a new computer with
   no hand-carried "champion" checkpoint or fixed source-handoff ritual.
2. The stable main-style standing/stepping foundation, eight locomotion
   rewards, two springed shin links, six force-limited actuators and modest
   starting command range are not casually replaced with new reward soup.
3. One policy, one PPO learner and one honest actor-input sensor contract:
   the 360-input array-only model, plausible delay, no ideal collision
   normal, specialist index or desired obstacle answer leaked into inputs.
4. Physical contacts must be real MuJoCo/MJX contacts. A step means a
   higher or lower supported surface, not a short cosmetic bump, and a
   slope is a rotated surface rather than merely rotated gravity.
5. Completed, severity-attributed evidence is required before declaring
   a harder frontier. Sign matters, up/down matters, and movement must
   *actually* occur. Failed hard probes need not trigger retreat.
6. Reward, training fall percentage, simulator shaft work and electrical
   battery energy are not interchangeable measurements.
7. All skills learn without global mastery gates; the exposure fade at 20M
   schedules teaching *frequency* only. Protect real PPO safety rather than
   demanding infant perfection.
8. Full PPO, optimiser, normalisation, curriculum, health and capability-field
   state survives interruption and a same-source `--resume auto`. Never
   mix checkpoint source revisions or silently overwrite old training roots.
9. Keep a compact, understandable codebase. Delete retired launch modes,
   contradictory doctrines and redundant counters after replacement;
   do not reduce line count by hiding work in another maze.
10. `main`, approved Pinnacle and public showcase remain protected.
    Promotion requires explicit owner permission and matched held-out proof.

## Measuring progress without stalling learning

Use independent, matched frozen-policy exams and actual episode evidence
to inspect forward/reverse, signed lateral/yaw, mixed commands, physical
pushes/recovery, supported slope and step crossings, motor work and control
smoothness. New hard lessons may temporarily weaken an immature 20–40M
policy; measure trends and starvation rather than enforcing a premature
all-green qualification window. The live teacher keeps learning while
the examiner provides honest diagnostics to humans.

For the 8GB RTX 4060 Ti, do NOT create another 2048-world fixed evaluation
allocation alongside the learner: cap frozen panels to 32 worlds. Confirm
2048-world CUDA PPO shape in a separate smoke before a lengthy campaign.
CPU CI alone does not certify GPU performance or actual learned walking.

## Operational approval

The preferred production path is **one clean, source-pinned cold run**
with `EMU_UNIFIED_CONTINUOUS=cold-growth-approved`. Choose a cumulative
target (currently 260M), train through the gradual specialist blend, and use
`--resume auto` under the exact same source/physical config after
interruption. A completed run can extend its *total step target* from
its validated final checkpoint on the same source, not start a new
"phase" or handoff. See [INSTALL.md](INSTALL.md) for full commands.

Before starting on the owner's GPU, confirm exact-SHA GitHub CI, 12-world
CPU physical smoke, 2048-world CUDA PPO/memory smoke and a clean source
checkout. Neither these checks nor the ability to invoke a launch mode
constitutes evidence of real locomotion or permission to publish a policy.

Historical campaigns and source-specific 20M→40M/40M→60M experiments
remain accessible in Git history. They are *not* the current design.
