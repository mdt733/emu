# EMU — active engineering handover

**Updated: 31 October 2026.**
**Active candidate source:** `release/v5032-pinnacle-candidate`.
**Public incumbent:** `main` + `showcase/champion.emupolicy`, frozen,
232-input v5016 Pinnacle (~260M checkpoint). This is still the
published baseline. No replacement or Pages publication has occurred.

## Current mission

Stop the long chain of experiments and produce ONE clean, reproducible,
source-pinned **360-input** flagship model that can become the next
Pinnacle **only after objective graduation**.

- Six-servo dynamically unstable gait, not a static-standing cheat.
- 20M main-style cold nursery; unified six-specialist growth to 260M.
- True 360-input ToF/optical-flow/IMU sensor model, real
  colliding slopes and sharp step-up/down slabs.
- Fast measured forward walking, lateral/yaw authority and
  controllable signed heading bias. Normal rhythmic yaw oscillation
  during fast strides is not automatically bad.
- Retain moderate slopes, step down, actual upstep support,
  resilience, efficiency and low falls as speed grows.
- Do not cap capability growth for visual neatness.
- Organic terrain/extra specialised step-up intervention is
  **DEFERRED** until this flagship candidate has completed and
  been measured.

## Training branch and the one launch command

**Only active launcher:** `scripts/train_pinnacle.py`.
**Case:** `pinnacle_cold260`, seed761, 2048 GPU worlds, 260M cumulative.
**Root:** `~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/`.

Pinned recipe: 20M foundation / main mix / balanced / main-mature,
coherent-careful real steps, gentle pushes, 25% athletic flat
rehearsal, 25-frame forward reward blend 0.5, verified upper-support
reward reallocation 3.0. 360-input policy; 8-cm geometry ceiling
(not a training target). Every 20M steps is fixed evaluated.
New source branch SHA must be frozen and printed in terminal
**immediately before starting**; it must not be updated mid-run.

```bash
# On the dedicated, clean source checkout
cd "$HOME/emu-pinnacle"
source .venv/bin/activate
git branch --show-current
git rev-parse HEAD
git status --short
python scripts/train_pinnacle.py                 # PLAN only
python -u scripts/train_pinnacle.py --run        # CPU+2048 CUDA gate; train
# If interrupted, only after verifying SAME SHA/manifest/preflight:
python -u scripts/train_pinnacle.py --run --skip-preflight
```

A brand-new PC must first follow [INSTALL](docs/INSTALL.md).
Run in foreground, no tmux; GNU Screen for remote disconnects if
desired. No other GPU campaigns at the same time. The launcher
refuses dirty source, foreign checkpoint lineage, changed manifest
and incomplete CUDA acceptance.

Run output includes checkpoint/optimizer evidence, frozen exams,
report ZIP and `pinnacle_candidate_260M.emupolicy`. No weights are
auto-committed to Git or published.

## Source-gate fix after first attempted launch

The first v5032 260M cold-start attempt stopped **before any PPO
training**: the launcher passed the CPU and full 2048 CUDA smoke
but set `EMU_UNIFIED_CONTINUATION` instead of the trainer's required
`EMU_UNIFIED_CONTINUOUS=cold-growth-approved`.

The launcher is now aligned with `emu/training.py`, with a
CPU-only regression test in `tests/test_pinnacle_launch.py`.
Because the first attempt wrote a manifest/preflight stamp pinned
to its older source SHA, the repaired campaign uses the **new
`v5032_pinnacle_s761_v2` directory**. Do not delete or silently
rewrite the old run metadata; do not resume training from an
unvalidated partial state. Use `git pull --ff-only` only in the
stopped v5032 checkout, then verify the new exact SHA before
launching a new run.

## What we learned before the cold flagship

**v5030:** A 25-frame/half-second forward reward blended into the
normal command reward improved held-out broad-score performance.

**v5031:** Both A/B arms resumed from the identical v5030 B80
checkpoint at original SHA
`a7be11b587f0912e642e100b22ec6d617f548f5b`.
Their source was
`2ffa1abce1b171dc38f8d864c8f43cbf3ea4b773`,
and both completed 100,007,936 steps (seed747).
Only the verified-ascent reward reallocation changed.

| Frozen physical challenge | B control | C verified-support |
|---|---:|---:|
| 2 cm supported UP | 7/8 | **8/8** |
| 2.5 cm supported UP | 2/8 | **4/8** |
| 3 cm supported UP | 0/8 | 0/8 |
| 3.5 cm DOWN | 1/8 | **7/8** |
| 12° downhill mission | 1/16 | **13/16** |
| 0.8 m/s 700-frame survival | 9/16 | **12/16** |
| 1.0 m/s 700-frame survival | 0/16 | 0/16 |

Their fixed broad scores were nearly identical (B=0.77593,
C=0.77503), and C had worse training falls and shaft-work
proxy. A single seed is not release-ready. C is the chosen
flagship **experiment recipe**, not a proven new champion.
Measured 3-cm UP crossing is STILL unresolved; 0.8m/s survival
is not 1m/s mastery.

The two existing attached v5031 ZIP reports and historical
frozen Git SHAs are the evidence. Do not modify completed
runs or checkpoints.

## Showcase status and safe publication

Current research viewer has the park's optical-flow heightfield
fix, continuous crest support and eight graduated real-contact
challenges (Gentle/Balanced/Stretch). This is an interactive
diagnostic; it is **not** an independent MJX physical exam.

After 260M completion, compare the trained exported candidate to
both v5031 C100 and **protected main** on common physical
commands/ground/support protocols. Grade actual sustained forward
speed including 0.6/0.8/1.0, yaw/path bias, lateral/backward,
supported step-UP and DOWN 2/2.5/3/3.5 cm, four-bearing
slopes, resilience, falls, and J/m. Repeat key claims across
seeds/worlds. Never certify curriculum labels as physical success.

Only after explicit review/owner approval do we update
`showcase/champion.emupolicy`, generate/validate
`showcase/release_approval.json`, commit the release and
publish via the existing protected GitHub Pages workflow.
Preserve old main SHA + policy bytes for instant rollback.
No automatic `main` merge or deployment from candidate training.

Read [graduation gates](docs/RELEASE_READINESS.md) and
[the exact candidate runbook](docs/PINNACLE_CANDIDATE.md).

## Housekeeping performed in this candidate source tree

- The README, installation procedure, maintenance/runbook,
  release checklist and handover now point to the flagship.
- One active source-pinned launcher replaces four historical
  A/B and recovery launchers. Original Git branches and
  commits preserve their source code and all old experiments.
- Source-level tests, real physics exams and release validators
  are retained. No history rewritten, local checkpoints deleted,
  or champion replaced.
- Canonical viewer uses the fixed park optical flow and
  corrected crest seam, with graded obstacles.
- The existing `scripts/campaign_support.py` still supplies
  streaming foreground logs, ETA, and source-pinned bundle
  generation.

## NEXT when the trained candidate returns

1. Verify `candidate_provenance.json` hash, checkpoint step,
   360-input actor contract and frozen-exam protocol.
2. Compare against v5031 C100 and main with **the same
   actual contact geometry and request schedules**.
3. Assess speed vs supported terrain vs efficiency and falls.
   If the candidate fails, preserve it; repair under a NEW
   source-pinned experiment rather than manipulating `main`.
4. If it graduates, prepare the reviewed 360-input
   `showcase/champion.emupolicy` plus approval and rollback.
5. Only AFTER flagship release consider follow-up lessons:
   (a) real supported 3-cm step-UP instead of lip-stalling,
   (b) organic terrain transitions/curvature, (c) sensor-aware
   traversability and safe refusal, (d) multi-seed/real-hardware
   calibration and physical efficiency.

## Read these, not archived launch instructions

[Current runbook](docs/PINNACLE_CANDIDATE.md) ·
[Fresh-machine install](docs/INSTALL.md) ·
[Physical release evidence](docs/RELEASE_READINESS.md) ·
[Viewer graded park](docs/VIEWER_GRADUATED_PARK.md) ·
[Code map](docs/ARCHITECTURE.md) ·
[Learning architecture](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Complete v5031 handover preserved](docs/history/HANDOVER_V5031_COMPLETE.md) ·
[Historical experiments](docs/history/README.md).

**Guardrail:** freeze `main` until the independent evidence and
explicit publication approval both exist.
