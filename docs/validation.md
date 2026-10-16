# Validation and release gates

EMU has **three different questions**: does the code run, does the learning loop behave correctly, and can the resulting policy actually walk? Passing one does not answer the others.

## 0. Clean WSL bootstrap and branch identity

Follow the [Windows → Ubuntu → Python 3.13 installation sequence](INSTALL.md#from-blank-windows-11-to-a-ready-ubuntu-shell). During pre-merge acceptance explicitly switch to `refactor/emu-readable` **before** installation; `main` is not yet the test candidate. Record `wsl --list --verbose` in PowerShell and `python3.13 --version`, `node --version`, `git rev-parse HEAD` in the Ubuntu checkout.

The first user-visible milestone is the **existing trained 60M Playground running locally**, not a zero-weight fixture. Follow [the exact static-preview steps](PLAYGROUND.md#preview-the-exact-static-site-locally), then test a joystick command in the real browser. Only after that verify the GPU learning pipeline.

## 1. Every change: source and contract checks

Run from the repository root with the editable Python environment active:

```bash
python -m compileall -q emu tests scripts
python -m pytest -q tests
npm ci --prefix emu/viewer_src --no-audit --no-fund
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
git diff --check
```

Python tests check checkpoint schema and publication, learning-state resume contracts, the 220-input/six-action interface, ladder ancestry, matched screening and experiment validation. The `test_exported_actor_matches_browser_inference` in `tests/test_delivery.py` uses a deterministic NumPy actor oracle and the **actual Node.js/browser PolicyRunner**, checking float32 packing, masked normalisation, layer orientation and output values.

The browser CI job additionally packages and verifies the pinned 60M public policy and MuJoCo WASM. A successful web build does **not** establish visual correctness.

## 2. Physics, trainer or observation changes: GPU smoke

On a supported NVIDIA/JAX CUDA system:

```bash
emu doctor --v5015-smoke --domain-smoke
```

Confirm the backend is `gpu`, MJX can reset/step, domain variation stays finite, and the doctor executes a **real PPO optimiser update** plus a held-out evaluation summary. If this fails, do not launch a long run.

A new checkpoint/run-loop change also needs a bounded fresh-root experiment: validate the saved checkpoint, deliberately stop, warm-resume it, perform a new update and compare the saved meta, normaliser, optimiser and evaluator artifacts. Warm resume restores *learning state* but starts new simulator episodes; it is **not** byte-for-byte trajectory replay.

### Documented two-update checkpoint/resume acceptance (GPU only)

The `verify` stage now **checks both stage directories' source Git commit**
against the CURRENT clean checkout before loading any checkpoint. A different
commit is an error even if the policy shapes still match. Return to the exact
original source; never rewrite the checkpoint identity or reuse an active root.

The basic doctor checks only one in-memory PPO update. The **separate training-lifecycle
smoke** below checks on-disk checkpoint creation, a new process restoring a prior
policy/optimizer/normalizer/governor/field, an independent fixed examination, the
already-complete no-update path, and a browser-policy export.

This is a **strictly bounded acceptance test, not the 20M training ladder**:
8 parallel environments × 32 rollout frames = **256 environment steps per
PPO update**. Stage 1 trains one update to step 256; stage 2 starts from its
unchanged `final.pkl` and performs one more update to step 512 in a **new**
folder. Both have forced final held-out evaluations, so those GPU exams are
additional work beyond the two PPO updates. No policy trained for only two
updates is expected to walk.

From a clean checkout, create a unique scratch root **outside the Git tree**:

```bash
cd ~/emu
source .venv/bin/activate
git status --short
git rev-parse HEAD
python scripts/smoke_training_lifecycle.py first --root ~/robot_lab/emu_runs/readable_H1_new_01
python scripts/smoke_training_lifecycle.py continue --root ~/robot_lab/emu_runs/readable_H1_new_01
python scripts/smoke_training_lifecycle.py verify --root ~/robot_lab/emu_runs/readable_H1_new_01
```

A successful viewer build should not change tracked files. In an earlier draft,
a generated viewer copy and the canonical README disagreed, causing the checkout to become dirty on build.
The build regenerated the mirror and made Git dirty, correctly blocking training.
The refactor removes the duplicate tree entirely; CI checks the sole canonical source and verifies that building does not modify tracked files.
If the provenance gate fails, run git status --short and git diff -- <path>.
Restore only generated files you have inspected; never discard personal edits.

**Run these commands one at a time and inspect errors before advancing.** The
helper refuses a non-empty/old root at stage 1 or an existing continuation at
stage 2. It never deletes files, changes the public champion or edits a source
file. Each training stage writes its full console log and `training_config.json`
under the scratch root. If an error occurs, retain the outputs for diagnosis;
do not wipe/reuse the root blindly.

Expected outcome: `first/checkpoints/final.pkl` at step 256,
`continued/checkpoints/final.pkl` at step 512, a finite best policy and
`fixed_eval.csv` for each stage, restored `resumed_from_step=256`, increased
normaliser count, an unchanged parent checkpoint, and a validated
`tiny_smoke_actor.emupolicy`. The third command additionally confirms that
resuming an already-completed step 512 is a no-op.

**Do not merge just on this result.** This proves two updates and their
persistence plumbing, not end-to-end 20M/260M draft four-way experiment selection, survival,
command following, slope ability or long-horizon walking.

## 3. Candidate policies: measure actual locomotion

A promising training reward is only a hypothesis. Collect deterministic fixed exams, a common matched screen, independent command tracking, stepping/support metrics, and any applicable changed-physics stress test. Report seeds, cohorts, absolute checkpoint steps, source commit and evaluator config. The current v5015 promotion gates require survival/posture/slope evidence but **do not yet establish a calibrated minimum for stepping and command-following**; treat a winner as provisional.

Do not call `robust40` a changed-hardware test: it varies **task demands**. Gravity-equivalent slopes are a separate physics variation; they do not prove performance on the browser's actual collision ramp.

Long-term height degradation previously appeared at ~100–200M **per lineage**. The new draft ladder's four primary branches reach 80M each; its optional 120M finalist can probe later deterioration, but cannot validate survival to 200M.

## Public Playground gate

The public demonstration is a deliberately separate manual deployment. Before running the Pages workflow, confirm: the intended 60M or separately approved champion policy matches its source identity; `showcase.json` reports the correct trained status and policy URL; MuJoCo WASM is loaded; joystick signs and camera/ramp switches work in a real browser; and the published site still loads after deployment.

The local `emu viewer --live` server is loopback-only and unauthenticated. Never use it as a public deployment.

## Record what was actually observed

For each significant release or training experiment, record `git rev-parse HEAD`, clean/dirty status, package versions, exact test commands, actual results, checkpoint step, evaluator settings and remaining failures. Use [project status](STATUS.md) for the current evidence summary; dated historical observations remain in [history](history.md) and Git history.

**Do not promote or publish a checkpoint solely because its tests pass or its viewer motion looks impressive.**
