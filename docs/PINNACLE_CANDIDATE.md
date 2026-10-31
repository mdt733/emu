# EMU — next Pinnacle candidate: cold start, release and rollback

**Status (31 October 2026): source preparation, NOT trained and NOT approved.**
**Branch:** `release/v5032-pinnacle-candidate`.
**Published public champion:** preserved on `main` with its existing
`showcase/champion.emupolicy` and GitHub Pages release checks. Do not
overwrite either while conducting candidate experiments.

## What is actually supported by experiment results

The completed v5031 read-only physical A/B used the same v5030 B80
ancestor and seed747 with different ascent reward. Both reached
100,007,936 steps at SHA `2ffa1abce1b171dc38f8d864c8f43cbf3ea4b773`.

| Frozen physical case | B100 control | C100 verified-support |
|---|---:|---:|
| Supported step up 2.0 cm | 7/8 | 8/8 |
| Supported step up 2.5 cm | 2/8 | 4/8 |
| Supported step up 3.0 cm | 0/8 | 0/8 |
| Supported step down 3.5 cm | 1/8 | 7/8 |
| Downhill 12° mission | 1/16 | 13/16 |
| Flat request 0.8 m/s, survived 700 frames | 9/16 | 12/16 |
| Flat request 1.0 m/s, survived 700 frames | 0/16 | 0/16 |

Frozen broad scores B=0.77593 and C=0.77503 (near tied).
C was better in several *supported terrain* cases, but had more
training falls (18.5% vs 13.5%), higher reported mechanical shaft
J/m (104.8 vs 97.6) and a worse final rollout vx/cmd average.
The latter is NOT a matched speed test; the independent ladder
showed C remained competitive at 0.6–0.8 m/s. One seed does not
prove the reward is universally better.

The next candidate explicitly adopts the measured C treatment,
with 0.5 temporal forward-speed blend, a 3-point reallocation
toward verified upper-foot support, 25% flat athletic rehearsal,
real inclined/up/down slabs, and 20M main-style foundation.
This combination is **new as a cold-start 260M training run**.
Its success is a hypothesis, not an already trained policy.

## Goals and deliberately deferred work

The single learned 360-input policy controls six servos. We seek:
- Dependable dynamic gait, 0.6 m/s sustained and progress toward
  sustained 0.8–1.0 m/s forward walking, without runaway yaw bias.
- Real supported uphill and downhill motion, side slopes, honest
  step-up and step-down crossings, purposeful foot transfers.
- Lower falls, resilience and smoother/economical shaft J/m.
- High quality *across* capabilities, not a single inflated reward.
- The same trained actor should work in browser and frozen MJX exams.

**Deferred:** organic hills, irregular-ground training, and additional
specialist step-up reward experiments. Preserve the findings and
next-work checklist, but do not change physics/rewards during
this source-pinned flagship run. The viewer's graduated obstacles
are a demonstration and diagnostic world, **not** a replacement for
a matched independent physical exam.

## One launcher, one source, one run directory

Source branch: `release/v5032-pinnacle-candidate`. Freeze the exact
40-character commit at launch; do not pull/edit it afterward.
Use `~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/` as the output directory,
**outside the Git clone**, with candidate checkpoint under
`pinnacle_cold260/checkpoints/final.pkl`. Do not copy an old trained
checkpoint into the run directory: this campaign MUST be a cold start.

From a NEW WSL checkout after completing [installation](INSTALL.md)
and with no other 2048-GPU training active:

```bash
git clone --single-branch --branch release/v5032-pinnacle-candidate \
  https://github.com/mdt733/emu.git "$HOME/emu-pinnacle"
cd "$HOME/emu-pinnacle"
# On a fresh machine or new venv, install the exact GPU + dev dependencies:
bash scripts/setup_wsl.sh --cuda13 --install-node --check
source .venv/bin/activate
git branch --show-current
git rev-parse HEAD
git status --short
python scripts/train_pinnacle.py
```

Plan-only outputs the full command without writing a checkpoint. On the existing GPU PC, you may alternatively activate the
already-proven `~/emu-graduation/.venv` instead of installing a second
CUDA environment. The source-import validation prevents silently
using another checkout's EMU implementation. A new machine should
follow the full install above before running training.

After CI is green and the working tree is clean:

```bash
cd "$HOME/emu-pinnacle"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u scripts/train_pinnacle.py --run
```

The launcher refuses a changed branch, dirty checkout, foreign run
directory, mismatched manifest, wrong Python import, wrong source
checkpoint and unvalidated 360 export. It executes reward tests, a
24-world CPU PPO step, and a **full 2048-world CUDA PPO update** before
training. Initial training is not resumed from v5031 or `main`.

A properly interrupted run is resumed **on precisely the same SHA**
using:

```bash
python -u scripts/train_pinnacle.py --run --skip-preflight
```

The skip is only accepted when the SHA and hashed experiment manifest
match the prior CPU/CUDA preflight stamp. Never delete stale
manifests to force incompatible resumes. Observe progress directly
in the foreground console; no tmux is required.

## Recovery after the first stopped v5032 preflight

The first launch under source `3f0703f2c4c43644924f5cb30d1c60308d4c87fd`
passed the CUDA smoke but did not begin training. The actual
error was an incorrectly named environment variable in the
flagship launcher, not an observed policy failure:

```text
launcher: EMU_UNIFIED_CONTINUATION=pinnacle-coldstart-approved
trainer:  EMU_UNIFIED_CONTINUOUS=cold-growth-approved   (required)
```

The launcher now uses the trainer's real gate; the
`tests/test_pinnacle_launch.py` check runs the actual
`unified_launch_mode` decision without a GPU.
**Never reuse the old manifest at the new source SHA.** The
repaired source's default results directory is
`~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/`. The earlier
`v5032_pinnacle_s761/` folder is retained as preflight
history. The new candidate remains a **true cold start**:
no old checkpoints are transplanted.

## Milestones and independent quality gates

The run has **260M cumulative environment steps**, with fixed-eval
snapshots at 20M intervals and saved recovery checkpoints.
Inspect milestones around 20/60/100/160/200/260M; milestone
labels are for operator review, not a second trainer or a licence
to cherry-pick a score.

After successful 260M completion the launcher:
1. Validates the full final checkpoint and exports a portable
   360-input `pinnacle_candidate_260M.emupolicy` outside Git.
2. Writes `candidate_provenance.json` with checkpoint SHA256,
   exported actor identity, run identity and exact training SHA.
3. Performs a read-only 16-world speed ladder (`0.1–1.0 m/s`)
   and real slope/step matrix with separate UP and DOWN success.
4. Writes a shareable `overnight_results.zip` *without*
   copying optimizer/checkpoint weights into a public artifact.

**Required promotion evidence**, beyond a good average fixed score:
- Comparable controlled physical performance against both the
  best v5031 C100 policy AND the old 232-input public Pinnacle.
- Correct actual velocity tracking, survival and signed heading
  bias at 0.6, 0.8 and 1.0 m/s (NO success from brief speed spikes).
- Four-bearing 5–15° slope missions, supported upper contact
  on steps 2, 2.5, 3, 3.5 cm, with step DOWN distinct.
- No degraded flat commands, combined lateral/yaw, recoverability,
  fall fraction or mechanical J/m disguised by terrain improvements.
- A repeat or independent seed for the final large capability claim;
  a single seeded 260M result is *a candidate*, not conclusive proof.

Compare the completed candidate against the existing champion
with the **maintained read-only command**:

```bash
python scripts/compare_policies.py            # plan only; requires exported candidate
python scripts/compare_policies.py --run      # common frozen physical fixtures, reports
```

The command validates the genuine 232-input `main` and 360-input
candidate actor contracts, uses identical ground and request schedules,
and never changes either policy. For historical C100 comparisons pass
`--candidate PATH/TO/C100.emupolicy --output-root PATH/TO/new_exam`.
Do not invent 232↔360 observation conversions.

## Showcase preview and future release

The new browser viewer contains the repaired optical-flow handling
and crest seams and the graduated 8-obstacle park. Test the trained
candidate **locally** before contemplating GitHub Pages:

```bash
cd "$HOME/emu-pinnacle"
source .venv/bin/activate
python -m emu.cli viewer-build
python -m emu.cli viewer \
  "$HOME/robot_lab/emu_runs/v5032_pinnacle_s761_v2/pinnacle_candidate_260M.emupolicy" \
  --no-open --port 8769
```

Or use `--live --root ~/robot_lab/emu_runs/v5032_pinnacle_s761_v2`
to watch checkpoints during training. A second SSH/Tailscale
terminal can forward `8769` to localhost without exposing weights.

**Before formal publication**:
- Preserve the original `showcase/champion.emupolicy` bytes and
  record the current main commit as a rollback tag/ref.
- Operator explicitly approves a specific frozen checkpoint,
  a specific policy SHA256, the physical evidence and the viewer.
- Commit only the reviewed 360-input exported policy as the
  replacement `showcase/champion.emupolicy`, plus a valid
  `showcase/release_approval.json` generated with
  `scripts/verify_showcase_release.py`'s established
  `approval_for_bundle` contract and exact checkpoint hash.
- Verify the policy schema-3 360-input sensor/actor/controller
  contract, build and test static assets, validate GitHub Pages'
  approval gate, and only then merge the release to `main`.
- A merge alone must **not** automatically publish the live
  GitHub Pages demo: the workflow is protected and deployment
  still requires its reviewed manual/release-request path.
- If a post-release regression appears, restore the saved
  public actor/reference and redeploy through the same
  approval workflow; do not rewrite training history.

Release is never automatic from `train_pinnacle.py`. An exported
candidate on disk is **not** a published or approved Pinnacle.

## Housekeeping: historical branches, no experiments left live

Active current path: `scripts/train_pinnacle.py`.
Keep `scripts/campaign_support.py`, `scripts/smoke_unified_focus.py`,
`scripts/smoke_training_lifecycle.py`, frozen physical-exam
implementation, policy exporter and public release validators.

The following old launchers are removed from the release-candidate
branch but recoverable from original source SHAs:

| Historical launcher | Exact source commit |
|---|---|
| v5029 speed restoration | `8288dc0b02c2b97f9a915cd14c0b65958e6b77c4` |
| v5030 temporal reward A/B | `a7be11b587f0912e642e100b22ec6d617f548f5b` |
| v5030 report recovery | `4e8b4628e8986a9082f2b698483819f794f02d1c` |
| v5031 verified ascent A/B | `2ffa1abce1b171dc38f8d864c8f43cbf3ea4b773` |

Old `docs/V5028_*`, `docs/V5030_*`, `docs/V5031_*` and
`docs/history/` remain marked historical evidence, not active launchers.
We are not deleting Git branches, rewriting history or deleting
local training results/checkpoints; those are valuable evidence.

## Next research AFTER flagship grading (deferred)

1. Step-UP 3 cm+ with real upper-surface support and continued travel;
   isolate why feet stall at the lip. Make this a controlled experiment,
   not a random additional reward in the flagship.
2. True multi-surface organic crests, valleys and changing camber with
   contact-verified physical exams and terrain-sensor parity.
3. Longer-term sensor-conditioned traversability and safe refusal;
   avoid treating slow stalled behaviour as deliberate planning.
4. Multi-seed replication and eventual sim-to-real torque/thermal
   calibration, with explicit hardware safety margins.

[Release gates](RELEASE_READINESS.md) ·
[Training architecture](UNIFIED_CAPABILITY_GROWTH.md) ·
[Historical evidence](history/README.md).
