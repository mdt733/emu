# EMU engineering rules — keep one understandable system

This is how to change EMU **without recreating hundreds of
experiment branches and incompatible launch procedures**.

## The boundaries

- **Physics and sensors:** `emu/robot.xml`, `emu/environment.py`,
  `emu/contracts.py`. Real support geometry, simulated sensor packets,
  contact and disturbance evidence.
- **Teaching:** `emu/rewards.py` chooses PPO payment;
  `emu/curriculum.py` and `emu/capability_focus.py` choose *which
  worlds and difficulty to practise*. One shared actor, not
  specialist sub-policies.
- **Learning:** `emu/ppo.py` updates policy and value network;
  `emu/training.py` runs the lifecycle.
- **Memory and recovery:** `emu/checkpoint_io.py`, `emu/run.py`
  validate source, config, complete learner state and exact resume.
- **Independent evidence:** `emu/evaluation.py`,
  `emu/physical_exam.py` and `emu/capability_cloud.py`.
  Their test results never adjust teacher decisions.
- **Viewing and publishing:** `emu/viewer_server.py` and
  `emu/viewer_src/` run a *separate* browser simulation.
  `scripts/verify_showcase_release.py` and the protected Pages
  workflow control publication.

The full newcomer explanation is [ARCHITECTURE](ARCHITECTURE.md).

## One maintained campaign

`scripts/train_pinnacle.py` is the **only current flagship
training launcher**. It defines the 260M cold-start recipe,
CPU and 2048-world CUDA preflight, independent exams and
candidate export. `scripts/campaign_support.py` supplies generic
logging/ETA and does not invent another curriculum.

Historical launchers are preserved by their **Git commits**
in the [research archive](history/README.md), not carried
alongside the active launcher. The old v5030-only comparison
script is now `scripts/compare_policies.py`: read-only physical
main-versus-candidate evaluation with no obsolete branch lock. `main` continues to host the
protected public 232-input champion. The 360-input successor
is not published merely because a training command finishes.

## Rules before changing functionality

1. **Choose the owner, not a wrapper.** If the change is a
   lesson, edit curriculum; if an input or contact changes,
   edit physics/contracts; if a payment changes, edit rewards.
   Do not add a second parallel training engine.
2. **State the data boundary.** Which values are actor inputs,
   teacher-only truth, PPO reward, host-side competence evidence,
   independent tests or browser-only navigation?
3. **Avoid changing learning semantics during cleanup.**
   Rename docs and fix actual bugs on a separate source-pinned
   revision. Any reward, sensor or physics change needs its
   own new cold-start lineage or explicitly verified research fork.
4. **Test what matters.** Reward math, real MJX contacts,
   360-input schema, live-checkpoint parsing, same-SHA resume,
   actual CPU/GPU PPO update, browser viewing and independent
   physical exams. Don't keep tests solely for an obsolete
   launcher name, but don't discard meaningful contracts.
5. **Protect the proof.** A clean SHA, manifest, venv/JAX
   versions, checkpoint hashes, full learner state, source
   and report must accompany any trained result. Git alone
   does **not** back up policy weights.
6. **Require physical graduation.** Measured forward and
   lateral/yaw tracking, slope/step-UP and DOWN, recovery,
   J/m and survival on matched tasks. A target severity or
   a high reward is not a passed physical mission.
7. **Keep the public champion recoverable.** No automatic
   `main` replacement, showcase asset overwrite, approval
   file update or Pages deployment without an explicit review
   and a tested rollback path.

## Documentation discipline

| Change | Update |
|---|---|
| Operator command, branch, new candidate | [HANDOVER](../HANDOVER.md), [runbook](PINNACLE_CANDIDATE.md) |
| Software prerequisites/installer | [INSTALL](INSTALL.md) |
| Module ownership, actor/teacher boundary | [ARCHITECTURE](ARCHITECTURE.md) |
| Public behavior or project goal | [README](../README.md) |
| Graduation/asset validation | [RELEASE READINESS](RELEASE_READINESS.md) |
| Completed experiment, old launch command | [history](history/README.md) |

Don't make a new permanent Markdown file for every tweak; put
completed studies in `docs/history/` or keep the original
SHA-linked record. Keep the README short enough for a newcomer
to understand the project before learning the engineering
history. The [doc index](README.md) is the single navigation map.

## Current source migration after the stopped v5032 launch

The first v5032 launch was rejected **before PPO training**:
the flagship launcher set `EMU_UNIFIED_CONTINUATION`,
but the trainer required
`EMU_UNIFIED_CONTINUOUS=cold-growth-approved`.
The repaired launcher has a direct regression test.
The old SHA's manifest and preflight logs remain historical;
use the new `v5032_pinnacle_s761_v2` results root for the
new source SHA, never patch old provenance by hand.

This migration is intentionally a **launcher fix and documentation
cleanup**. No reward weights, observation channels, physics, or
public policy are changed.
