# EMU handover — preparing the next `main` milestone

**5 November 2026.** This checkout is the **future-main
release-preparation branch**, not the original completed GPU training
source. The v5048 research bundle has been reviewed; no neural actor
has been promoted to the public `main` release.

## What is approved, and what remains under research?

| Layer | Status |
|---|---|
| Public baseline | v5032 **260,046,848**-step 360-input Pinnacle; immutable `showcase/champion.emupolicy` and `showcase/release_approval.json` on `main` |
| Existing public `main` commit (audit reference) | `4e3e1f45b98945c9729aab4d5c68d9ee122a6876` |
| Completed GPU experiment | v5048 **550M unique steps** in 50M shared parent → 8 matched 100M actors → 200M temporal_speed finalist; 148 real frozen exam panels, [measured findings](docs/V5048_SHARED_PARENT_RESULTS_20261105.md) |
| Training-source identity | `088a29be47f624be9e2c322c504ec98049b3f9bf`, seed 735; same-parent checkpoint SHA256 `8047f4bd4d75894c627e8689e0c645a6bde23496caf6a807f408fa8c88a4085f` |
| Release-prep code | `release/v5049-main-promotion-prep`: remove temporary experiment machinery, keep the one public CLI and the permanent exam/provenance contracts |
| Main code merge | **Not yet approved; `main` untouched** |
| New public policy | **200,015,872-step** temporal_speed export exists, source/actor hash recorded; actual ≥3cm climbing, stillness, heading-controlled mission and matched main parity **not demonstrated**. Actor unchanged |
| GitHub Pages deployment | Independent protected/manual approval; not triggered by this cleanup |

The completed v5048 training and all `~/emu-data/runs/`
artefacts belong to **the original training SHA**, not this later
cleanup commit. Do not reinterpret or attempt to resume the finished
model on the release-prep branch. The historical
[v5048 experiment source](https://github.com/mdt733/emu/tree/experiment/v5048-shared-parent-pinnacle)
retains its original temporary launcher for source-pinned reproduction.

This release-prep branch deliberately removes that launcher,
the old 140M and 20M launchers, and the v5045-only duplicate CI.
Their measured findings and source-pinned historical implementations
remain accessible in Git; the maintained runtime does not depend on
any of them.

## What the present live robot-training system contains

- **One** six-servo dynamically unstable robot, **one** PPO actor/critic,
  six specialist teaching cohorts, and a **360-input** deployable
  policy based on IMU, optical flow, 4×4 ToF, proprioception and
  command history.
- **Four 75cm** recyclable terrain tiles, each with two real convex
  triangular-prism collision solids — **eight colliders** total.
  One generator and one actual MJX contact world across all cohorts.
- Same-source checkpoint and optimiser recovery, strict sensor
  semantics, source SHA/actor export verification and independent
  raw-supported foot-contact step examination.
- Standalone CPU terrain-physics acceptance, full 2048-world GPU PPO
  smoke, Python and browser contracts, and deliberately protected
  public deployment.
- The source owners and files are described in the slim
  [CORE_SKELETON](docs/CORE_SKELETON.md). Future runs use
  [one clean WSL installation](docs/INSTALL.md), one checkout,
  one `.venv`, `~/emu-data/runs/` for permanent run artifacts,
  and `~/emu-data/scratch/` for throwaway study output.

## Measured engineering gains — do not confuse with stair capability

The completed 34-arm v5046 solver investigation, two seeds on the
RTX 4060 Ti, measured about **31,974** steady full-PPO world-steps/s
using the reference solver, and **119,662** with experimental 2/4.
The one 20.054M continuous 2/4 cold-start reached ~**0.094m/s**
achieved mean forward speed, final training rollout fall 2.1%,
and +0.530 held-out broad score, but **zero supported signed lips**
in its independent procedures. [Full measured results and caveats](docs/V5046_OVERNIGHT_RESULTS_20261105.md).

Solver 2/4 is still a **research physics choice**: foot penetration,
virtual torso/leg strikes and a measured CUDA packed/full reset
`.qacc` disagreement have not been cleared for hardware fidelity.
The frozen policy examiner had a **9×4 retired-tile reshape** that
prevented a v5047 GPU exam; v5048 fixed it to the physical four-tile
75cm geometry before launching the replacement tournament.

## The 550M experiment's scientific decision

[The independently audited results](docs/V5048_SHARED_PARENT_RESULTS_20261105.md)
show real five-frame **supported signed lip crossings**, with stronger
harder-step outcomes from the early Academy and finer 1mm frontier.
On the 3cm-*requested* panel (actual lips ~1.8–2.9cm), worlds
with at least one actual ≥2cm supported uphill transition increased
from **2/32 without Academy** to **19/32 early Academy** or **21/32
precise frontier**. None demonstrated full actual 3cm ascent.

The auto-selected 200M temporal-speed actor had **16/32** worlds
with ≥2cm uphill support (unchanged from its 100M predecessor),
**31/32** survival at requested 0.7m/s and **0.595m/s** achieved.
Its fast yaw bias improved but it still drifted ~**12cm** and
made rapid footfalls under zero command. Do not claim it learned
stillness, verified full 3cm steps or equivalent main missions.
The tournament's step count contribution saturated at 12 total
crossings for every branch; this composite rank did **not**
select a uniquely best stair climber. Record this as experiment
design feedback before another GPU campaign.

## The next release decision (two independent gates)

**Gate A: approve better engineering for `main`.** The 550M
study is complete and measured. Inspect its physical implications,
then inspect the complete
new-source versus `main` diff. Re-run Python, browser, actual
four-world CPU MJX contacts and a **new same-SHA 2048-world CUDA
PPO/checkpoint/export/frozen-exam** acceptance after the cleanup.
Verify the published 260M policy and release metadata are still
byte-identical. Only then ask for an explicit code merge decision.
This can proceed even if the new learned gait is merely comparable.

**Gate B: replace the public Pinnacle policy.** A 200M frozen
candidate has been exported from the original v5048 training SHA,
but publication **remains on hold** until we obtain
independently measured supported step ascent/descent with post-lip
progress, zero-/low-speed stability, achieved faster strides, heading
and lateral control, slopes, recovery, energy proxy and two or more
held-out seeds. Compare *equivalent missions* against the 260M
baseline, not absolute scores from different terrain engines.
Record policy/checkpoint/source hashes and prove the browser can
replay the exact 360-input sensor contract. The owner must separately
approve a new public policy and Pages deployment.

[Complete release gates and current evidence intake](docs/RELEASE_READINESS.md) ·
[260M independent physical baseline](docs/PHYSICAL_BASELINE_260M.md) ·
[Old champion release card](docs/V5032_PINNACLE_RELEASE.md) ·
[Open gait/step hypotheses](https://github.com/mdt733/emu/issues/17).

## Project hygiene contract

1. No merge to `main`, actor swap or Pages release without explicit
   permission.
2. The active research source and `final.pkl` are never rewritten by
   cleanup; no checkpoint is magically reproducible from Git.
3. **Experiments are temporary. Findings are permanent. Working
   architecture earns its place.** Put conclusions in the issue and
   measured results; delete completed one-off launchers.
4. Keep the stable CLI, physical tests and truly needed browser assets;
   don't delete a meaningful regression guard just to reduce test count.
5. Before code promotion, verify the new runtime installs cleanly from
   scratch, remains explainable from one skeleton document and handles
   errors without old paths or alternate trainer branches.
