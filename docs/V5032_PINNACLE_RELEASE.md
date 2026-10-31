# EMU Pinnacle 260M — first 360-input baseline

**Release decision: 31 October 2026.** The owner elected to make the
completed v5032 260M actor the new maintained `main` baseline. It is
a **measured milestone**, not a claim of perfect all-terrain mastery.

## Exactly which robot?

| Field | Identity |
|---|---|
| Robot | Six actuated servos, two passive shin-flex springs, one PPO neural policy |
| Observation | `terrain_360`, 90 sensor/command fields × 4 history frames |
| Forward sensor | 4×4 ToF; IMU/optical-flow sensing |
| Action | Six servo targets |
| Training steps | **260,046,848** cumulative environment steps |
| Environment count / seed | 2,048 / 761 |
| Original frozen training source | `9b11bbe596aea7201d29a6beb2d192922f8e2e9a` |
| Exported policy SHA-256 | `ef929c0220c38e5519391320e5f0629a200e1ef37a5808086f6383cfb63f7ec8` |
| Actor weights SHA-256 | `e268dfd500a4c2a63cee1b064d44f510db7488715f3f4eff32b4ac0e985e45f9` |
| Checkpoint SHA-256 (reported) | `238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b` |
| Release assets | `showcase/champion.emupolicy` and `showcase/release_approval.json` |

The standalone 886,002-byte exported policy attached to the
project conversation matches the SHA-256 recorded in the original
`overnight_results(9).zip`'s `candidate_provenance.json`.
The full `final.pkl` checkpoint is stored on the operator's
WSL GPU machine, not committed to Git. Its reported SHA-256
has not been recomputed directly from the checkpoint bytes here.
Back up that full checkpoint independently.

**Source identity matters:** the public main branch will contain
release/housekeeping commits made *after* the actor's training.
That does not retroactively change its trained source or weights;
the original frozen training source is the commit above.

## Independent physical results

Held-out CPU MJX exams used actual inclined two-slab collision
geometry, seed1701, same 16 physical worlds per case, with no
reward or curriculum feedback. These are the first released
baseline measurements, **not deployment guarantees**.

| Frozen terrain case (0.15 m/s, 7 s) | Success |
|---|---:|
| Step-UP 1.0 cm | 8/8 |
| Step-UP 2.0 cm | **8/8** |
| Step-UP 2.5 cm | 4/8 |
| Step-UP 3.0 cm | 0/8 |
| Step-UP 3.5 cm | 0/8 |
| Step-DOWN 3.0 cm | 8/8 |
| Step-DOWN 3.5 cm | 1/8 |
| Uphill 12° | 15/16 missions |
| Cross-slope 12° (both bearings) | 16/16 each |
| Downhill 12° | 16/16 survived, **4/16 heading-controlled missions** |
| Downhill 14.9° | 16/16 survived, 2/16 missions |

| Flat speed request (14 s) | Survived | Complete mission |
|---|---:|---:|
| 0.3 m/s | 16/16 | 16/16 |
| 0.4 m/s | 16/16 | 11/16 |
| 0.5 m/s | 16/16 | 0/16 |
| 0.6 m/s | 16/16 | 0/16 |
| 0.8 m/s | 12/16 | 0/16 |
| 1.0 m/s | 9/16 | 0/16 |

**Important reading of these results:** the step figures above report
**verified physical supported crossings**, NOT automatic full
heading-controlled missions. At 2 cm UP, crossing was 8/8 but full
mission only **3/8**. At 2.5 cm UP, crossing was 4/8 and mission 3/8.
At 3 cm DOWN, crossing was 8/8 but mission only 4/8.
See [the exact frozen physical capability analysis](PHYSICAL_BASELINE_260M.md)
for separated support, movement, heading, survival and tested-range AUC.

The frozen training evaluation reached **0.8296** at 260M
(best sampled ~0.8352 at 180M); these aggregate scores do not
replace detailed physical tests.

**Interpretation:** good survival and real progress on slopes
and steps, but persistent fast-walk yaw bias, weak steep-downhill
directional authority and an unsolved 3cm supported uphill step.
At 12° downhill, the robot generally did not fall or freeze:
its heading control failed the strict mission gate. Normal
stride oscillation must be distinguished from real path drift.

## Known training recipe discrepancy

The v5032 campaign *manifest* requested
`verified_ascent_reallocation=3.0`, but the actually saved
`TrainingConfig` contained **0.0**: the original
`emu/run.py` accepted that CLI flag without forwarding it.
The released weights are genuine and valuable, but **did not**
receive that particular ascent treatment. Record this honestly
and fix the end-to-end setting test before any future training.
Do **not** rename this policy to a verified-ascent result.

## The clean project baseline

`main` now owns the current robot/source architecture and the
approved portable actor. A normal clone supports the viewer,
policy validation and independent exams without needing
someone's private full PPO checkpoint.

The public web viewer has optical-flow heightfield parity,
a continuous ramp crest and eight graduated physical
playground challenges. The viewer is a diagnostic: it does not
prove the same measured MJX physical result as the frozen exams.

**Retained historical rollback:** old public 232-input champion,
including its policy, release approval and source, is preserved
at `2ec9e9db0a816c658131311129ef89ec7a595a12`.
Revert the public files and deployment with explicit approval
if needed; don't rewrite the original model history.

## What happens next?

The research backlog is [GitHub issue #17](https://github.com/mdt733/emu/issues/17).
Investigate the physical examiner first (especially heading/path
bias and crouch/spring traces), then 2.5–3cm actual support,
downhill braking/steering, fast gait, active recovery and efficiency.
Organic hills are deliberately deferred.

The previous `research/v5033-24-arm-science` branch is
**paused and not a validated training recipe**. It contains
exploratory changes; none of its new rewards, sensor/teacher
variants or candidate weights are included in this release.
Do not launch 24 science arms just because the branch exists.

Train a future replacement only when a small set of research
questions has produced reproducible physical evidence. Keep the
current 260M policy as the comparison baseline.

[Architecture](ARCHITECTURE.md) ·
[Handover](../HANDOVER.md) ·
[Source-pinned historical runbook](PINNACLE_CANDIDATE.md) ·
[Public release validator](../scripts/verify_showcase_release.py).
