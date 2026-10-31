# EMU — active engineering handover

**31 October 2026 · New baseline: v5032 Pinnacle 260M**

The owner elected to promote the completed v5032 360-input
Pinnacle as the new `main` model, **with its known weaknesses
documented** rather than starting another experimental branch.
The 24-arm v5033 science work is **paused** and not released.

## What is the robot?

One six-servo dynamically unstable MJX biped, 360-input
4×4 forward ToF + IMU/optical-flow / proprioception observations,
one shared PPO actor. No programmed walk cycles. Real signed
slope and sharp step contacts, six independently growing
teacher focus groups. The actor does not receive privileged
teacher terrain truth. See [ARCHITECTURE](docs/ARCHITECTURE.md).

## The frozen trained baseline

| Fact | Value |
|---|---|
| Completed world steps | **260,046,848** |
| Training source SHA | `9b11bbe596aea7201d29a6beb2d192922f8e2e9a` |
| Source branch at training | `release/v5032-pinnacle-candidate` |
| Environments and seed | 2048, 761 |
| Observation / action | 360 / 6 |
| Policy artifact SHA-256 | `ef929c0220c38e5519391320e5f0629a200e1ef37a5808086f6383cfb63f7ec8` |
| Checkpoint SHA-256 | `238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b` (report; full file is on WSL) |
| Local run directory | `~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/` |
| Published policy location | `showcase/champion.emupolicy` |

`main` may have later documentation/release commits; the
actor's **training source stays frozen at its original SHA**.
Do not relabel those weights as a different training recipe.
Keep the whole `final.pkl` separately backed up outside Git.

**Important recipe discrepancy:** the launch manifest requested
verified-ascent reward reallocation 3.0, but the actual saved
`TrainingConfig` had **0.0** because the trainer's CLI
forwarding was missing. The released actor did not learn that
treatment. Recheck **resolved training config**, not CLI printouts,
before any future PPO campaign.

## What does the measured robot do?

Independent 16-world CPU MJX physical exams on real collision
surfaces (seed1701, 7s terrain, 14s speed):

- **Step-up:** 1–2cm 8/8; 2.5cm 4/8; 3cm 0/8.
- **Step-down:** 3cm 8/8; 3.5cm 1/8.
- **Slopes:** uphill12° 15/16, both cross-slopes12° 16/16;
  downhill12° survival 16/16 but heading-controlled 4/16.
- **Speed:** at requested 0.6m/s, 16/16 survived but 0/16
  passed full heading-controlled mission; at 0.8m/s survival
  was 12/16 and strict mission 0/16.
- **Frozen overall score:** 0.8296 at 260M.

Do not turn these into a claim of 1.0 m/s or 3cm mastery.
In particular, downhill 12° at 0.15m/s was NOT a sprint or
balance failure: mean signed yaw drift disqualified the
heading-control mission.

See [full release measurements](docs/V5032_PINNACLE_RELEASE.md)
and the [research backlog](https://github.com/mdt733/emu/issues/17).

## Publication and rollback

The intended release changes the *code branch and the portable
policy together*, using the exact policy SHA and approved
`showcase/release_approval.json`. The frontend is validated
by `scripts/verify_showcase_release.py` and GitHub Actions.
The GitHub Pages workflow uses explicit manual release
dispatch; merging source alone does not deploy it.

**Previous main rollback:** `2ec9e9db0a816c658131311129ef89ec7a595a12`
(the old 232-input champion, approval file, viewer and code).
Retain this immutable Git SHA, not a casually renamed ZIP.

The original long handover was preserved as
[Handover v5032 before release](docs/history/HANDOVER_V5032_PRE_RELEASE.md).
The new README and [architecture map](docs/ARCHITECTURE.md)
are the entry points for somebody new to the project.

## What to work on next (not yet)

**Read-only diagnostics first.** In order:

1. Separate downhill signed yaw-rate bias, alternating gait
   jitter and **world-frame heading/path drift**; use a 5-speed
   × 4-grade frozen test before changing rewards.
2. Trace real supported 2.5–3cm step-up crossings: knee/hip
   extension, upper-contact duration, lower-foot stance,
   torso dip/rebound and spring energy *release*.
3. Investigate the persistent height reward after foundation.
   A 5cm pre-step crouch is heavily discouraged even though
   springs can physically load; test a temporary near-lip
   allowance only in a future **single controlled experiment**.
4. Study 1mm step-frontier adjustment versus historical
   +18%/-18% jumps, with more completed evidence per direction.
5. Diagnose high-speed heading/turn authority, damping and
   meaningful shaft J/m without rewarding stationary behaviour.
6. Only afterwards revisit organic hills, traversability
   refusal and a future from-scratch Pinnacle.

The complete categorised backlog is
[GitHub issue #17](https://github.com/mdt733/emu/issues/17).
It includes negative tests and acceptance criteria. No new
PPO science arms have been launched in this promotion.

## Guardrails

- `main` is the new **reference**, not a scratchpad for training
  variations. Work on one explicitly justified change at a time.
- The abandoned `research/v5033-24-arm-science` branch
  contains **unvalidated experimental code**. Do not execute
  its 24-arm launcher or mistake its changes for released EMU.
- Keep the old 232 policy recoverable from rollback SHA.
- Published actor `.emupolicy` is inference-only; full
  checkpoint `final.pkl` has optimizer/teacher/normaliser state.
- Use common independent exams with actual supported step
  crossing and achieved velocity, not curriculum headlines.

[README](README.md) ·
[Release measurements](docs/V5032_PINNACLE_RELEASE.md) ·
[Architecture](docs/ARCHITECTURE.md) ·
[Documentation index](docs/README.md).
