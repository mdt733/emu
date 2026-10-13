# EMU — current project handover (v5015 WIP)

**Last reconciled:** 13 October 2026, using the audited v5014 sources, Playground v2/v3 changes, joystick `vy` correction, and the operator's WSL terminal/browser feedback. **This document supersedes `docs/HANDOVER_5014.md`.** It is a concise continuation guide, not a second design specification.

## v5015 status (14 October 2026)

- **Separate post-merge cleanup work:** `refactor/ladder-human-first` is NOT merged. It extracts the currently tested 20/26/34/46M hypotheses and screening target into `emu/ladder_recipe.py`, adds the friendly `emu ladder` command alias and a thin `run_ladder.sh` wrapper, and reduces obsolete docs. No PPO, physics or training-config behaviour change. The user reports active training; do not pull this into its checkout. The historical `forge` runner has now been detached from the active ladder by extracting its shared safety/config helpers into `emu/phase_config.py` (old imports are re-exported for compatibility); it remains callable for reproduction.

- **Baseline:** Keep `main` at `fabba945` and `showcase/champion.emupolicy` unchanged until v5015 is reviewed and tested. This is the successfully deployed 60,030,976-step policy.
- **Development:** [draft PR #1](https://github.com/mdt733/emu/pull/1) on branch `develop/playground-attitude-terrain`. Optional physical 1° ramp, three camera views, separated lean/pitch telemetry, accessible README. Local MuJoCo WASM/ramp tests remain necessary.
- **Terminated 400M run:** user stopped it due to gradual crouching. Historical data may remain under `~/robot_lab/emu_runs/`; do not erase it.
- **Diagnostic first:** v5014 logged absolute height but not the commanded-height error. `emu posture-report` is a read-only standard-library scan of existing `metrics.csv` files. New rollouts also export `height_error_m`. Hypothesis: weak height incentive can favour crouching; not proven without actual results.
- **v5015 training code implemented (not yet GPU-validated):** opt-in `emu forge5015 --plan` has separate `~/robot_lab/emu_runs/v5015_production/` run root and 8/15/25/35/45° *requested* pitch ceilings at the 20/60/140/260/400M milestones. Chassis pitch/roll are compared to their commands in posture reward; robot lean derives from foot-midpoint geometry and is used in structural competence. Uncommanded crouch lowers reward; held-out summary reports pitch MAE/height shortfall and v5015 checkpoint selection penalises both. TRANSITION tasks can change pitch mid-episode. All retain 220 actor inputs and six outputs. Existing `emu forge` and its v5014 defaults remain separate. CPU contract tests passed; **no GPU smoke or new trained policy yet**. Further work: coordinate frame at extreme pitch, real hill transitions, reward tuning and full browser verification.
- **v5015 branch lab (current):** `emu lab5015 --stage plan` proposes one 20M foundation, four forks at 26M (baseline, shadow/no teacher, T=.12 slope-equivalent, and 15° pitch), two manually promoted 34M finalists, and one manually promoted 46M continuation; 72M environment steps of training total. All offspring resume full validated schema-7 training state. `--stage score --round round1/round2/round3` uses the same 15° held-out evaluation for all arms, records checkpoint SHA-256, includes height shortfall and pitch tracking, and DOES NOT auto-promote. Results under `~/robot_lab/emu_runs/v5015_lab1` outside Git. **No lab experiment has been started yet.** Run `emu doctor --v5015-smoke` before GPU training; see [docs/tournament.md](docs/tournament.md#v5015-short-experiment-lab).
- **Unattended lab added:** `bash scripts/run_lab5015.sh` validates the current import path, runs the GPU doctor/MJX/PPO smoke, and executes `emu lab5015 --stage auto`: 20M foundation, four 26M forks, matched held-out score, automatic top-two eligible 34M continuation, score, one eligible 46M continuation, score. No eligibility bypass; insufficient qualified candidates stop the lab safely. `--stop-after round1` permits manual inspection. **The user observed crouching around 100–200M steps in one lineage**, so even a successful 46M early champion must undergo later 100–200M long-horizon checks. No GPU training is confirmed to have started.
- **Versioning:** v5015 is an *experimental milestone*, not a completed training release. v5014 forge code remains intact for forensic reproduction.

## Mission and invariants

Build a small, dynamically unstable **six-servo biped** that actually catches steps, walks, steers and recovers. Avoid rewarding static survival alone. Use deployable sensors: **55 named features × 4 frames = 220 actor inputs; six actuator targets**. Train with JAX/MJX + PPO. Keep capability evidence semantic, command/recovery units physical, held-out exams quarantined, resume strict, and learning state intact across phases. The v5014 trainer still offers **one long production lineage**, not another tournament, but the operator stopped the 400M run. v5015 training is not yet released.

## Current implementation

- **Python/production:** `emu forge` (default seed 733, 2048 parallel environments, 400M absolute environment steps) with boundaries at **20/60/140/260/400M**. The foundation precedes mature training with a low-dose **10% full graph capability teacher**. Hidden R/T/H domain radii widen by phase. Do not infer that the entire 400M has finished from this document.
- **Physics randomisation:** batched gravity-tilt equivalent, friction, mass, actuator gain/torque/zero and joint mechanical spread are in the production fast path. `emu/morphology.py` and `emu/terrain.py` contain separate generator work; **full morphology-bank and true heightfield terrain are not integrated into default `forge`**.
- **Safety of continuation:** schema-7 snapshots use atomic replacement and finite-policy guards; phase configs are checked against existing training settings before resuming. Resume preserves learning state, **not the exact MJX episode frame**. Never change seed/env-count/teacher/domain settings inside the same run root.
- **Local browser viewer:** `emu viewer --live --no-open` serves completed, validated checkpoints from `~/robot_lab/emu_runs/v5014_production/` on **127.0.0.1:8765**. CPU-only watcher, browser-local MuJoCo WASM and actor inference, roughly 12-second polling, clean reset when actor changes. Demonstration samples supported command nodes when available, otherwise conservative ranges. It is **not a held-out exam**.
- **Public Playground:** the separate Vite static preview uses **port 4173**, `viewer/public/showcase.json` and a bundled `.emupolicy`; GitHub Pages requires no Python server. The untrained fixture must remain prominently labelled **UNTRAINED PREVIEW** until a held-out-selected champion is actually exported. The live viewer uses a real Python API; the static preview does not. A JSON MIME/fallback bug was fixed in v3.
- **Visuals/controls:** right-side settings drawer keeps joysticks clear. Measured 10 cm grid, 1 m major lines, axis marks, subtle trail/sky/lighting and toggleable soft shadows. Latest manual correction: **joystick right maps to *negative* physical `vy`**, while turn-right maps to negative `yaw`; autonomous commands remain in canonical coordinates. Both the simulated command and UI readout use `viewer/src/controls.js` (and its packaging mirror).

## Evidence, not aspirations

**Observed on the operator's WSL machine:** editable `emu-biped==5014.0.0` installed, 93 Python tests + five subtests passed on the audited checkout; JAX 0.11.2 saw CUDA; MuJoCo 3.15.0 and MJX domain reset/step passed at `(8, 220)`; Node/npm and Vite viewer build succeeded, with `.wasm` served; local live viewer loaded training demos; the operator subsequently reported the redesigned viewer looked much better and live checkpoints were working. There have been later add-on tests, but do not treat an older test count as a final run-wide acceptance certificate.

**Not yet established from the conversation:** that the full 400M run is complete; the winning checkpoint's held-out scores; physical-hardware transfer; deployed public GitHub Pages URL; exact current Git branch, remote, commit/tag and whether `main` was pushed. Do not invent these. The latest `vy` fix was delivered as a patch and accepted in principle; recheck the viewer if the user reports another sign issue. The static Playground may be published in untrained-preview form before the policy exists.

## Reproducible working commands

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
cat emu/VERSION
emu --help
python -m pytest -q
emu doctor --domain-smoke
emu forge --plan
```

For **trainer** (terminal A, only when intentionally starting/resuming a long job):

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu forge
```

For **local live viewer** (terminal B; trainer is independent):

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu viewer --live --no-open
# Open http://127.0.0.1:8765/ in Chrome or Edge
```

For **static-site preview**, not live watching:

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
python scripts/prepare_showcase.py
emu viewer-build
python scripts/verify_showcase_dist.py
cd viewer && npm run preview -- --host 127.0.0.1
# Open http://127.0.0.1:4173/
```

Both Python and Node dependencies are pinned separately: Python in `pyproject.toml` (version ranges), browser resolution in `viewer/package-lock.json` (must be committed). `.nvmrc` selects Node 24. See [docs/INSTALL.md](docs/INSTALL.md).

## Git and release state — observed snapshot, verify before changing

**Last operator-confirmed topology (13 October 2026, before promotion):** original checkout `~/robot_lab/emu` was on local `main` at **`7c84439` (v5012)**, matching `origin/main`. Local `v5014` pointed to **`501e293`** (production + Playground); `origin/v5014` and existing annotated **`v5014.0`** tag pointed to the earlier **`ed5699d`** commit. Historical `v5013` is retained. The original checkout also contained **untracked `emu/` and `viewer/` directories**; these are not safe to delete blindly. A separate Git worktree `~/.emu-v5014-release-review` on `release/v5014-main` was created to reconcile docs and prepare promotion. **This is a dated observation, not a claim about the current remote state.**

Before making claims or changes, run `git status --short`, `git branch -vv`, `git remote -v`, `git log --oneline --decorate -8`, and `git tag -l 'v5014*'` in the appropriate worktree. The intended official branch is `main`; promote by ordinary fast-forward only after review. GitHub Pages should trigger from `main`; confirm both `.github/workflows/emu-playground.yml` and GitHub Settings → Pages.

**Safe pending workflow:** Complete doc reconciliation in the separate release worktree, inspect staged content, commit there, fetch and verify that `origin/main` is an ancestor, then normal-push `HEAD:main` (no force). Do not change the original checkout while its trainer might still be using it. Do not retarget the existing `v5014.0` tag without deliberately planning a new release tag.

Never use `git reset --hard`, `git clean -fd`, or force pushes as a shortcut. `../emu_runs/` is outside the source tree. Preserve `results/v5013_tournament/` as compact experiment evidence. Keep generated `.venv`, `node_modules`, `dist`, `webviewer_dist`, raw `.pkl` and training runs out of Git. Publish `showcase/champion.emupolicy` only after independent evaluation and policy-format review.

## What happened scientifically

The v5013 tournament compared 10/25/40/65% graph teacher doses, node/edge/recovery mechanisms and schedules. On seed 733 at ~120M, **10% full graph** produced broad 0.810 and hard recovery 0.920; **nodes-only** produced broad 0.803, hard recovery 0.909, but greater headroom (0.824 vs 0.734). Seed-1742 confirmation was mixed. This supports restrained full graph as a **pragmatic choice, not a conclusive causal winner**. The complete original experiment is at `results/v5013_tournament/`; `docs/history.md` preserves past failure modes.

## Recommended continuation

1. Confirm which source/lockfiles/docs are actually committed and whether `main` and the annotated release tag have been pushed; **do not assume a remote exists**.
2. Keep the current 400M lineage running if it is already active; avoid altering its checkpoint schema/robot contract mid-run.
3. Validate the latest viewer + static preview after any doc/asset changes; ensure mode distinction, sideward/yaw controls and `showcase.json` build check work.
4. After training: run fixed held-out exams, choose best policy rather than blindly the last snapshot; `emu export-policy CHECKPOINT showcase/champion.emupolicy`; preview + validate then publish Pages from the configured branch.

**Where to look:** [docs/README.md](docs/README.md) for the complete documentation map; [docs/v5014-production.md](docs/v5014-production.md) for training mechanics; [docs/PLAYGROUND.md](docs/PLAYGROUND.md) for deployment; [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) for dated acceptance evidence; [docs/AUDIT_REPORT_5014.md](docs/AUDIT_REPORT_5014.md) for the original release-code audit.
