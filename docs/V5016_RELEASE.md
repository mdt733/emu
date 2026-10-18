# v5016 release handover — Pinnacle → main → public demo

This is a **two-stage release**. Preparing v5016 code while GPU training
continues is safe; replacing the public model is intentionally blocked until
the actual campaign-selected actor and its evidence can be reviewed.

## Already prepared on feature/v5016-sim2real

- Complete 232-input sim2real training, schema-8 checkpoint isolation,
  versioned browser export and read-only live checkpoint watcher
- Local and static viewer with 10 cm grid over grass, third-person chase view,
  compass-heading target P controller, and optional ±30°/±45° experimental
  manual pitch requests
- Four-arm Pinnacle tournament and resume-aware best/final promotion
- Pages packaging for **either** the currently pinned 220-input legacy
  schema-1 policy or a **reviewed and pinned** 232-input schema-2 policy
- CI on every feature-branch push and PR against main
- Manual-only GitHub Pages deployment, not automatic on merge

The 120M historical showcase remains byte-for-byte protected by the old
Git-blob/provenance gate while there is no
`showcase/release_approval.json`. No future trained policy is included in
this release-preparation commit.

## When GPU campaign has stopped and results are final

**Use a separate checkout** (for example `~/emu-viewer-5016`) on
`feature/v5016-sim2real`. **Do not update the original training checkout
mid-campaign**; Pinnacle pins its source commit.

```bash
cd ~/emu-viewer-5016
git pull --ff-only
source .venv/bin/activate
export JAX_PLATFORMS=cpu

# Read-only evidence preflight; no commits, no trainer, no GPU.
python scripts/stage_v5016_showcase.py

# Only after reviewing selected checkpoint and reports:
python scripts/stage_v5016_showcase.py --stage
```

If the campaign exits early because its soft GPU-hour budget is exhausted,
`campaign.json` may still have `finished: false`. It is **not an
automatic success**. After verifying the campaign process is stopped and
the scored champion is satisfactory, explicitly add
`--accept-budget-stop` to both commands. The stage command still requires
the named checkpoint to match its SHA-256 and to have an eligible scored
entry in the campaign's held-out scoreboards.

The stage command selects **`champion.json`**, not a guessed latest or
largest-step checkpoint. It verifies the exact `.pkl`, the campaign's
source commit and sensor contract, and the exported neural actor's 232-input
browser schema. It then writes exactly two tracked files:

- `showcase/champion.emupolicy`: browser-only neural policy and normalized
  observation contract; no optimizer state or pickle
- `showcase/release_approval.json`: exact asset SHA-256 and Git blob,
  checkpoint SHA-256, trained steps, source commit, actor hash and sensor
  fingerprint

Neither script commits, pushes, merges or deploys. If the campaign has
not finished, or no eligible checkpoint exists, nothing is staged.

## Review and promotion

1. Inspect `results.md`, `scoreboards/*.json`, champion path/hash,
   and exported policy metadata; do not substitute viewer appearance for
   independent held-out tests. Check motor load and failures, not just reward.
2. Run `bash scripts/check.sh`, or CI via a feature-branch push.
3. Preview the **static GitHub Pages build** using the actual staged
   `showcase/champion.emupolicy`:
   ```bash
   python scripts/verify_showcase_release.py
   python scripts/prepare_showcase.py --require-champion
   npm ci --prefix emu/viewer_src --no-audit --no-fund
   npm test --prefix emu/viewer_src
   npm run build --prefix emu/viewer_src
   python scripts/verify_showcase_dist.py
   cd emu/viewer_src && npm run preview -- --host 127.0.0.1
   ```
4. On the feature branch only, review and commit the two staged files.
   Push the **feature branch**. Keep PR #8 as the review boundary.
5. After explicit approval, merge PR #8 into `main`. Verify its checks,
   that the pinned policy exists and that `main` points to the merged
   revision. Nothing auto-deploys.
6. Explicitly run **Actions → EMU Playground (GitHub Pages) → Run workflow**
   on **main**. The release workflow refuses a missing/mismatched actor.
7. Check the deployed `/emu/showcase.json`, deployed
   `/emu/policies/champion.emupolicy`, and browser behavior on desktop
   and mobile. Confirm **schema 2, 232 inputs, 6 actions**, correct
   campaign step, valid wasm MIME, joystick direction and chase/heading/pitch
   controls. If Pages fails, the previously published site remains the
   reference; redeploy only after diagnosing the failure.

No direct writes to main or Pages are part of release preparation.
Source weights remain in the local campaign's `.pkl` files; the Git
showcase contains **only the small inference actor**, not training state.

## What changes in the product

v5016 is a major actor-observation version change, not a renamed v5015:
the neural policy consumes ToF-derived local floor grade, optical-flow
velocity and gyro with sample timing/delay/dropout. The public viewer
replays these observations approximately in browser MuJoCo/WASM.
That is a research demonstration, **not a verified physical deployment
or a measured real-servo battery efficiency claim**.

If there are outstanding reviewer/CI issues at release time, retain
draft PR #8 and the old pinned public brain until resolved.
