# Validation and release gates

EMU has **three different questions**: does the code run, does the learning loop behave correctly, and can the resulting policy actually walk? Passing one does not answer the others.

## 0. Clean WSL bootstrap and branch identity

Follow the [Windows → Ubuntu → Python 3.13 installation sequence](INSTALL.md#from-blank-windows-11-to-a-ready-ubuntu-shell). During pre-merge acceptance explicitly switch to `refactor/ladder-human-first` **before** installation; `main` is not yet the test candidate. Record `wsl --list --verbose` in PowerShell and `python3.13 --version`, `node --version`, `git rev-parse HEAD` in the Ubuntu checkout.

The first user-visible milestone is the **existing trained 60M Playground running locally**, not a zero-weight fixture. Follow [the exact static-preview steps](PLAYGROUND.md#preview-the-exact-static-site-locally), then test a joystick command in the real browser. Only after that verify the GPU learning pipeline.

## 1. Every change: source and contract checks

Run from the repository root with the editable Python environment active:

```bash
python -m compileall -q emu tests scripts
python -m pytest -q
npm ci --prefix viewer --no-audit --no-fund
npm test --prefix viewer
npm run build --prefix viewer
git diff --check
```

Python tests check checkpoint schema and publication, learning-state resume contracts, the 220-input/six-action interface, ladder ancestry, matched screening and experiment validation. The `test_policy_inference_parity.py` suite uses a deterministic NumPy actor oracle and the **actual Node.js/browser PolicyRunner**, checking float32 packing, masked normalisation, layer orientation and output values.

The browser CI job additionally packages and verifies the pinned 60M public policy and MuJoCo WASM. A successful web build does **not** establish visual correctness.

## 2. Physics, trainer or observation changes: GPU smoke

On a supported NVIDIA/JAX CUDA system:

```bash
emu doctor --v5015-smoke --domain-smoke
```

Confirm the backend is `gpu`, MJX can reset/step, domain variation stays finite, and the doctor executes a **real PPO optimiser update** plus a held-out evaluation summary. If this fails, do not launch a long run.

A new checkpoint/run-loop change also needs a bounded fresh-root experiment: validate the saved checkpoint, deliberately stop, warm-resume it, perform a new update and compare the saved meta, normaliser, optimiser and evaluator artifacts. Warm resume restores *learning state* but starts new simulator episodes; it is **not** byte-for-byte trajectory replay.

## 3. Candidate policies: measure actual locomotion

A promising training reward is only a hypothesis. Collect deterministic fixed exams, a common matched screen, independent command tracking, stepping/support metrics, and any applicable changed-physics stress test. Report seeds, cohorts, absolute checkpoint steps, source commit and evaluator config. The current v5015 promotion gates require survival/posture/slope evidence but **do not yet establish a calibrated minimum for stepping and command-following**; treat a winner as provisional.

Do not call `robust40` a changed-hardware test: it varies **task demands**. Gravity-equivalent slopes are a separate physics variation; they do not prove performance on the browser's actual collision ramp.

Long-term height degradation previously appeared at ~100–200M **per lineage**. A ladder whose longest branch reaches 46M cannot verify that failure is solved.

## Public Playground gate

The public demonstration is a deliberately separate manual deployment. Before running the Pages workflow, confirm: the intended 60M or separately approved champion policy matches its source identity; `showcase.json` reports the correct trained status and policy URL; MuJoCo WASM is loaded; joystick signs and camera/ramp switches work in a real browser; and the published site still loads after deployment.

The local `emu viewer --live` server is loopback-only and unauthenticated. Never use it as a public deployment.

## Record what was actually observed

For each significant release or training experiment, record `git rev-parse HEAD`, clean/dirty status, package versions, exact test commands, actual results, checkpoint step, evaluator settings and remaining failures. Use [project status](STATUS.md) for the current evidence summary; dated historical observations remain in [history](history.md) and Git history.

**Do not promote or publish a checkpoint solely because its tests pass or its viewer motion looks impressive.**
