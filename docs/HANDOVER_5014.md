# EMU v5014 continuation handover — 2026-10-13

This file is here because the previous conversation hit its context limit while consolidating v5014. Use this instead of reconstructing assumptions from a zip filename.

## Project objective

One continuously trained compact six-servo dynamically unstable biped. Dynamic stepping/catching and general locomotion, NOT static standing. Train from scratch in JAX/MJX, deploy actor-only inference using the existing 55 × four-frame = 220 observation contract and 6 commands. Strong preference for simplicity, strict checkpoints, read-only exams, physics and units, and direct honest diagnostic evidence.

## Supplied baseline files

- `emu_v5013_tournament(1).zip`: canonical v5013 experiment source.
- `tournament_summary(1).csv`, `tournament_results(1).md`, `tournament_decisions(1).json`: completed user-supplied tournament results, now archived under `results/v5013_tournament/`.
- `emu_browser_viewer_v0_2_2(1).zip`: original separate browser viewer.
- `emu_v5014_domain_dev1(1).zip`: exploratory domain randomisation layer.
- `emu_v5014_consolidation_pass1(1).zip`: existing production integration with 400M-step lineage and viewer.

## What is now present in the release

1. v5014 production lineage `emu forge` with 20/60/140/260/400M absolute update targets, bounded 10% full-graph teacher, and batched hidden domain variation. Foundation/phase resumes retain learning state.
2. Portable `.emupolicy` export, browser-local MuJoCo/WASM+Three renderer+Float32 actor.
3. `emu viewer --live [--root PATH]`: loopback-only read-only checkpoint watcher, `/api/live/status` and `/api/live/policy`, 12-second browser polling, validated CPU-only snapshot export, actor hot-swapping, clean episode resets. The viewer process does NOT run on CUDA or feed training.
4. Random autonomous commands every 5–10 seconds. When possible it samples candidate nodes directly from the training capability field (posterior >=0.70, at least three effective event samples). Otherwise it uses a conservative fraction of curriculum frontiers. These are NOT held-out competence certificates, and transitions between nodes may fail.
5. Lateral/yaw UI mappings reflect previous operator's backwards-control feedback: joystick right currently maps to positive `vy`, positive joystick right maps to negative `yaw`. Physical orientation still requires in-browser human acceptance testing.
6. Checkpoint writes now use temporary file + atomic replace to protect training restarts and live readers. Resume selection prefers most recently committed latest/final.
7. v5013 results archived verbatim for reproducibility. Main README reflects production path instead of obsolete v5013 default tournament.
8. Retained Git-ready source tree. The original user's `.git` history is not in the uploads; do not fabricate it. Copy into existing checkout and commit there after review.

## v5013 evidence for 10% teaching

At ~120M on seed 733: 10% full graph broad ~0.810, hard recovery ~0.920, growth ~0.734; nodes-only ~0.803/~0.909/~0.824; shadow ~0.797/~0.904/~0.705. Seed-1742 confirmation changed the ordering; the mechanism comparison is mixed/tied, not proof of a nodes-only winner. Higher doses did not earn their extra influence in the screens. The production selection is a practical conservative working choice.

## Verified / not verified

- **Passed**: 92 Python tests, 1 skipped (5 subtests passed). JavaScript syntax checks. Native `.emupolicy` serialization round-trip from a synthetic valid checkpoint and browser `PolicyRunner` 220→6 inference. HTTP loopback endpoint/WASM MIME unit tests. Demo-command bounds in Node. Source/package browser mirrors synchronised at archive creation.
- **Not verified here**: complete CUDA 400M training, actual 5014 checkpoint performance, npm/Vite build, and a real browser WebAssembly/MuJoCo visual acceptance test. npm installation timed out in this sandbox. No actual trained `.pkl` was supplied; 5013 assets contain code and result summaries but not policy checkpoints.
- **Not integrated by design**: true randomized 3-D floor contact geometry/morphology banks. Fast-path randomized physical parameters are active; low-frequency heightfield and morphology bank are still separate development work and need throughput validation.

## Next useful actions on the user's WSL machine

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
python -m pip install -e .
pytest -q
emu doctor --domain-smoke
emu viewer-build
emu viewer --live
# separate terminal / after previous process runs:
emu forge --plan
emu forge
```

Run `viewer --live` before/while `forge` to observe completed checkpoints as they appear. Verify screen-space lateral and yaw mapping by driving the viewer manually, test drop/load overlay and WASM MIME, then let autonomous random-command demo operate. Review final held-out evidence before claiming a successful generalist. If continuing code work, prioritize a real browser acceptance pass and an explicit training health/throughput review; avoid adding many new knobs or speculative ablations to the production path.

## Audited follow-up

See `docs/AUDIT_REPORT_5014.md` for the subsequent functional source review, repaired JAX phase-rate call, safer phase compatibility, additional checkpoint finiteness protection, more accurate browser birth cadence/noise, Git installation script, and outstanding real hardware/GPU/browser acceptance gates. The final audited source supersedes the initial 87-test release.
