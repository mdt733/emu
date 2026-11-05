# EMU — measured release gates for future successors

**Baseline status (31 October 2026): owner-approved v5032 260M,
360-input Pinnacle is the new `main` actor.** It has known limitations
explicitly recorded in the [release card](V5032_PINNACLE_RELEASE.md).
The previous v5016 232-input champion is retained at
`2ec9e9db0a816c658131311129ef89ec7a595a12`.

This checklist defines the bar for **future** successor policies;
release of v5032 as a reference milestone is not a claim that all
future capability targets have already been met.

The target is not just higher rewards, harder curriculum targets, or
a new neural input shape. It is a **genuinely better six-servo robot**:
measured fast dynamic locomotion approaching the incumbent's ~1m/s
**requested** forward-speed envelope, plus supported terrain crossing,
resilience and efficient control.

## v5048 200M browser milestone — approved with explicit limitations

On 5 November 2026 the operator inspected the exported
`v5048-temporal_speed-s735` 200,015,872-step actor in the local
browser and explicitly approved it for `main` and the public
playground. The eight-prism engineering was merged independently
as [PR #39](https://github.com/mdt733/emu/pull/39), producing
`6bbcdd1`, before the actor release on
[PR #40](https://github.com/mdt733/emu/pull/40).

The approved **inference-only** artifact is schema 3, 360 inputs,
six actuator targets, source training SHA `088a29b`, actor SHA256
`29219e34e149159df1d050281903c79752d665fa13dcafe303bafcc45bfbdce1`,
Git blob `020bd17ab0efa84d97b68c7a4f43b759097bdfbb`
and exported file SHA256
`36508ecca685cb7f113f62502541594f8f9da345dda3025dbb4c9e7fac6668c6`.
The exact independently submitted 148-panel research evidence
bundle has SHA256
`7071c8a8742cf3e5dd053e0d8affb9ecef10f0e9f3a8cde87f376adcdea7a541`.

**Provenance exception (explicit, not fabricated):** the original
200M `final.pkl` bytes and checkpoint SHA256 were **not** included
in the operator's archive. Approval schema 2 therefore records
`checkpoint_sha256: null` and `checkpoint_archive_status:
not_provided`, strictly verifies the source/actor/sensor/export
and examination archive hashes, and documents an
`owner-reviewed-inference-only` approval. Existing schema-1
checkpoint-archived approvals remain unchanged and strict.
This is a software-demo release, **not** a guarantee that the
training optimiser can be resumed or identical hardware behaviour
reproduced from the saved inference artifact.

**Still unproven:** a full measured 3.0cm stair ascent, sustained
post-lip headed progress, quiet zero-command stance, 0.7m/s
tracking accuracy at fast speed, direct equivalent-mission parity
against the 260M reference, and CUDA compact/full `.qacc`
solver2/4 physical fidelity. These remain *scientific and
hardware-readiness backlog items*, not reasons to mislabel the
owner-approved browser milestone.

The previous v5032 approved policy and release files are preserved
at main Git SHA `4e3e1f45b98945c9729aab4d5c68d9ee122a6876`
for straightforward rollback. The Pages workflow pins the new
approved actor and viewer Git blobs and will refuse mismatched
unreviewed deployment requests.

## Historical staged promotion checklist — preserve the champion

The **source upgrade** and **new policy release** are two independent
review/approval decisions. The v5048 50M parent + eight 50M
forks + 100M winner extension **completed 550M unique steps and
148 real independent GPU exam panels** on source SHA
`088a29be47f624be9e2c322c504ec98049b3f9bf`.
A **200,015,872-step** candidate export exists, but it has NOT
demonstrated controlled full-3cm climbing or v5032 main parity.
[Full measured findings](V5048_SHARED_PARENT_RESULTS_20261105.md).

### Gate A — merge the improved ENGINEERING without replacing the actor

- [x] Analyse the complete 550M shared-parent tournament **on its
  original source SHA**; record 148 independent panels, actor hashes,
  actual lip heights and score saturation in the [permanent result card](V5048_SHARED_PARENT_RESULTS_20261105.md). The future-main
  cleanup source is a **different** version.
- [ ] Review the complete `main` → new-source diff: eight exact
  collision prisms across four 75cm tiles, GPU/MJX contact changes,
  sensor contract, PPO curriculum, checks, browser, CLI, install and
  checkpoint resume.
- [ ] Remove one-off tournament scripts and obsolete CI; preserve
  hypothesis, measured numbers and original research commit SHA in
  permanent results/issue #17. The sole production entry point is
  `emu train`, not a copied campaign-specific wrapper.
- [ ] Pass the final candidate SHA's full Python, browser, 4-world
  MJX CPU contact/recycle probe, actual 2048-world CUDA PPO,
  checkpoint/save/restore/actor export and frozen 4-tile contact exam.
  Do not infer a CUDA result from CPU-only GitHub Actions.
- [ ] Explicitly review 2/4 versus 4/4 solver stability, foot
  penetration and previously observed CUDA compact/full
  `.qacc` reset mismatch; retain rollback if physics is suspect.
- [ ] Verify **byte-identical** public
  `showcase/champion.emupolicy`,
  `showcase/release_approval.json`, and
  `showcase/deployment_request.json` against approved
  v5032. Current approved Git blobs are respectively
  `45f189367d51ad56dd57726b67e13be08b80bfae`,
  `3cf944808935bfc3c2167b9e0b60aa4614f91739`, and
  `fc70cc8fa5ab2522b2e41dc4636a5ac9fdddc191`.
  The release-prep audit found these **identical on old main and
  the research source** before making any proposed merge.
- [ ] Get explicit owner approval for a code-only merge to `main`.
  No workflow should publish GitHub Pages merely because code merged.

A green Gate A upgrades the **trainer and architecture**, not the
public actor. It can be worth doing even if the v5048 contender never
beats the approved Pinnacle. Existing champion weights stay
compatible with their own recorded source/sensors: do not silently
interpret them as trained on the new terrain.

### Gate B — qualify and deliberately publish a NEW Pinnacle actor

- [x] Receive original v5048 `results_bundle.zip` with **148
  `prism_exam.json` panels**, training config, source run manifests
  and actual **200,015,872-step** `.emupolicy` export.
  This archive does **not** contain the trained `final.pkl`
  checkpoint or a matched v5032 physical exam. Detailed post-step
  heading/path traces remain a separate outstanding requirement.
- [ ] Verify its full **training-source provenance**, including
  original `088a29be...` SHA, seed 735, age 200,015,872,
  original checkpoint hash (not in uploaded bundle), sensor
  timing/normalizer and 360-input interpretation. The exported
  `.emupolicy` file SHA256 has been checked independently:
  `36508ecca685cb7f113f62502541594f8f9da345dda3025dbb4c9e7fac6668c6`.
  The cleanup commit is NOT the actor's training SHA.
- [ ] Compare against the 260M released baseline using matched
  task **intent, physical difficulty, command schedule, durations,
  episode counting and mission criteria**, rather than naïvely
  equating a legacy two-slab course with procedural four-tile
  profiles. Report simulator/solver differences honestly.
- [ ] Report independently **supported** signed UP and DOWN step
  crossings (1, 2, 2.5, 3cm), **continued commanded progress
  after support**, survival and heading mission separately.
  One lucky fall onto an upper step is not a climb.
- [ ] Report stationary drift/true zero command, requested/achieved
  slow velocity, reversing and yaw response, forward 0.3–1.0m/s
  speed, stride length versus cadence, turns, 12° slope/camber,
  perturbation recovery, and useful mechanical J/m. At least
  two held-out seeds; report both successes and eligible attempts.
- [ ] Compare against the released main reference specifically:
  supported 2cm UP 8/8 (but controlled mission 3/8),
  3cm UP 0/8, 3cm DOWN 8/8 (controlled mission 4/8),
  0.3m/s flat command 16/16 full mission, and 12° downhill
  survival 16/16 but heading mission 4/16. These are **different
  historical course conditions**, not an automatic leaderboard.
- [ ] Confirm the candidate can be **exported, loaded and driven**
  correctly by the current browser; do not treat a valid JSON
  policy alone as evidence of a realistic gait or exact terrain
  replay.
- [ ] Get a second explicit owner approval to replace the champion
  `.emupolicy` and its approved release metadata; run
  `scripts/verify_showcase_release.py` and the complete browser
  build/asset verification *after* the staged swap.
  The GitHub Pages workflow remains a separately reviewed/manual
  deployment. Keep the current v5032 Git commit as rollback.

**Stop conditions:** no signed supported crossings, poor or unsafe
heading/path control, unstable 2/4 physical contact, source-provenance
mismatch, inability to export/replay sensor-correct actor, or CI/GPU
regressions. These mean **hold model promotion**, not that the
engineering upgrade has no value.

### Promotion evidence intake (pending measurements)

| Field | Known or still required |
|---|---|
| Approved incumbent | v5032 260,046,848 steps; exported policy SHA256 `ef929c0220c38e5519391320e5f0629a200e1ef37a5808086f6383cfb63f7ec8` |
| Incumbent original training SHA | `9b11bbe596aea7201d29a6beb2d192922f8e2e9a` |
| v5048 source pinned for currently running tournament | `088a29be47f624be9e2c322c504ec98049b3f9bf` |
| v5048 candidate policy and step | **Temporal-speed 200,015,872 steps**, exported actor file SHA256 `36508ecca685cb7f113f62502541594f8f9da345dda3025dbb4c9e7fac6668c6` |
| Cross-seed supported steps and heading-controlled missions | Five-frame foot-supported actual ≥2cm UP **16/32 worlds**, ≥2.5cm UP **5/32** on nominal-3cm procedural challenge; **no full 3cm** claim; post-step heading mission **not tested** |
| Exact source / exported actor / checkpoint SHA256 | Source and exported actor SHA256 verified; original complete final.pkl SHA256 **not in bundle** |
| Gate A code merge | **Not approved or merged** |
| Gate B public actor change | **Not approved or merged** |
| GitHub Pages publication | **Not requested** |

## 1. Training provenance and code hygiene

- [ ] Fresh training or provably compatible source-pinned lineage;
  log exact source SHA, seed, JAX/MuJoCo versions and checkpoint hashes
- [ ] Hold completed training checkpoints immutable; **never** resume
  the running source under a later cleanup branch
- [ ] One documented source-pinned launcher and no surviving duplicate
  campaign-support framework; historical studies live at immutable Git SHAs
- [ ] No unexplained duplicated rewards, alternate actor schemas or
  undocumented privileged observation fields
- [ ] No stale README, dead links, superseded "current" run instructions
- [ ] Automated Python checks, full 2048-world CUDA smoke, checkpoint
  resume/export and browser build tested on the actual candidate revision
- [ ] Current code and documentation pass peer review; no necessary
  capability or release test was deleted merely to reduce code size

## 2. Independent physical competence

Use **common physical tasks** to compare today's 360-input `main`
with any future challenger. For historical 232-input comparisons,
retain each actor's original sensor contract and use identical
body, ground, disturbances, command schedule, horizons and scoring.

- [ ] Signed forward tracking at 0.15, 0.3, 0.6, ~1.0 m/s; record
  actual mean velocity, steady-state error, falls and recovery. A
  1m/s teacher request is **not** a 1m/s pass.
- [ ] Reverse, lateral and yaw including direction changes and braking,
  without hidden turning-back shortcuts
- [ ] Four-bearing slopes (uphill, downhill, two cambers), record
  physical stability **and** commanded motion/heading separately
- [ ] Steps: support-verified **UP** and **DOWN** at 1, 2, 2.5, 3,
  3.5cm and onward; no mixed up/down headline scores
- [ ] Realistic combinations of incline + sharp ledge, orientation
  variation and terrain perturbations; no false geometric shortcuts
- [ ] Resilience conditioned on known shove, direction, command
  and remaining support (not merely training resistance target)
- [ ] Effort measured as meaningful shaft-work proxy per useful
  metre; do not call it electrical consumption
- [ ] Complete per-world JSON and time-resolved traces for difficult
  events, more than one seed and more than eight worlds for claims
- [ ] Retain already passed easier tasks when hard frontiers expand;
  certify beyond the specific held-out curriculum fixtures

## 3. Navigational safety and deployability

- [ ] Delayed/noisy 4×4 ToF observations provide measurable lead time
  to detect a high step or missing landing surface
- [ ] The dynamically unstable policy can **actually slow, hold,
  backstep and recover** under supervisor commands
- [ ] Prove a deliberate safe refusal/replan is not a mere stalled
  policy, and measure both false refusals and hazardous overcommitment
- [ ] No supervisor has access to secret terrain geometry at runtime;
  no hard-coded servo climbing animation or alternate actor
- [ ] Verify exact actor/export observation contract in browser replay,
  including all sensor timings and units

## 4. Release and rollback

- [ ] Share comprehensible README, install and safety docs, concise
  handover and historical links; avoid active v5024/v5025/26/27
  launcher clutter
- [ ] Leave the current approved 360-input `showcase/champion.emupolicy`
  unchanged until a *future* candidate has owner-reviewed evidence
- [ ] Explicit owner approval for the new champion and for merging to
  `main`; this checklist does not authorize either
- [ ] Record release source and checkpoint fingerprints, deterministic
  export, viewer smoke, and rollback to the prior 260M baseline
- [ ] Update the public Pages deployment only with a reviewed,
  independently tested actor + frontend combination

## Current candidate, release workflow and retained historical reference

The completed [v5032 runbook](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/PINNACLE_CANDIDATE.md) is historical
source-pinned reproducibility documentation. The [v5032 release card](V5032_PINNACLE_RELEASE.md)
is the current physical baseline, and
[issue #17](https://github.com/mdt733/emu/issues/17) holds proposed
future diagnostics. The v5031 study remains historical evidence,
not a second active training system.

Before publication, the owner must review the exact exported
actor and checkpoint SHA256, 360-input sensor contract, measured
flat/slope/step commands, energy/falls, browser playback and
rollback plan. The only approved release format is checked by
`scripts/verify_showcase_release.py`, and the GitHub Pages
workflow remains explicitly protected. Never infer approval
from a successful training exit, higher reward or merged commit.

[Maintenance](MAINTENANCE.md) ·
[Historical research](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/history/README.md) ·
[Public playground](PLAYGROUND.md) ·
[Previous candidate handover](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/history/HANDOVER_V5031_COMPLETE.md).
