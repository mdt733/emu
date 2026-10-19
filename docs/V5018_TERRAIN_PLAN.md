# v5018 Terrain Awareness — living branch checklist

## Status — 20 October 2026

Branch: `feature/v5018-terrain-awareness`. P0A **partial**, P0B pending. No measured terrain contact results, training success or GPU FPS yet. Never touch main/showcase without explicit approval.

### Fresh WSL acceptance evidence (user-run)
- [x] Fresh installation reached `python -m pytest -q`: **73 passed, 5 failed, 24 deprecation warnings, 428.56 seconds**.
- [x] All five failures were outdated 120M schema-1/220-input tests calling `showcase/champion.emupolicy` which currently contains the protected **260,112,384-step, checkpoint schema-8, export schema-2, 232-input** champion. This was verified against actual JSON metadata, not guessed.
- [x] Showcase XML embedded in the policy byte-for-byte matches current `emu/robot.xml`.
- [~] Implemented schema-1/2 exam-reader validation, independent reconstruction of the 232 sensor contract and runtime examiner matching, plus updated current-champion tests and package installer. **Not yet rerun on user's WSL; no regression PASS claimed**.
- [!] Because the old installer ran `pytest` before `emu viewer-build`, nonzero exit prevented viewer/npm checks, doctor and the v5018 terrain probe from running. The adjusted installer builds/tests viewer first and allows explicit `--skip-tests` for diagnostic-only use.
- [ ] Rerun updated Python tests and npm viewer tests in WSL; verify legacy 220 load remains intact.
- [ ] Confirm CUDA-13 JAX plugin/device, driver, doctor, Node/npm, viewer build, GPU VRAM; user's pasted output proves only Python dependency installation and partial test suite.
- [ ] Capture results from `scripts/v5018_terrain_spike.py`.

### P0A user-run results — initial minimal spike
- [x] CPU MuJoCo 3.16.0 / MJX-JAX first fixture run produced five final ball-foot contact cases without reported exceptions (uploaded `v5018_p0a_spike_results.json`).
- [x] Final centre-height deltas absolute max **0.00057742 mm** (0.57742 µm). This is final resting geometry agreement **only**; no trajectory/impulse/contact-force transfer proof and not evidence of GPU backend usage without device metadata.
- [x] Raised step: support on `obstacle`, final CPU centre z 0.0476328 m; elevated about 30 mm above flat. Log: support on `log`, final CPU centre z 0.0476328 m.
- [!] Step-down fixture landed on `upper` shelf, not `lower`, so it did **not** demonstrate descending contact.
- [!] Diagonal fixture landed on `ground`, not `ramp`, so it did **not** demonstrate graded contact.
- [~] Replaced P0A script with **unexecuted v2**: vertical-slider sphere contact (prevents ramp rollout masquerading as collision failure), geometrically unambiguous lower landing/tilted slab, actual contact geom labels and CPU force magnitude; CPU static `mj_ray` vs JAX-oriented-box/plane primitive intersection including risers and occlusion, marked unsupported capsule ray; optional 2-world same-topology vmap. Writes `v5018_p0a_spike_v2_results.json`. ALL v2 validation remains pending local execution. This is not full free-body gait or train-loop benchmark.
- [ ] Execute `python scripts/v5018_terrain_spike.py --batch` after `git pull --ff-only`; inspect every `pass` and any exception, the CPU and JAX geom hits, contact force and 2-world batch results.
- [ ] Separate CPU trajectories/swing-foot impacts and moving-foot collision after stationary clearance proof. Resolve log/capsule JAX sensor-ray intersection against CPU truth before declaring sensing feasible.

### P0A physics and sensing
- [~] `scripts/v5018_terrain_spike.py` v2 fixes lower/diagonal fixture ambiguity and adds primitive ray/occlusion and optional two-world evidence collection. **Committed, not yet executed; not an end-to-end training benchmark**.
- [ ] Actual CPU/MJX contact, support force, step geometry, riser/toe strike, ray occlusion and geometry-pair agreement.
- [ ] Actual two-world heterogeneous batch, then 256/512/1024/2048 VRAM/FPS scale trials where possible on RTX 4060 Ti (8GB).
- [ ] Identify viable exact ray/collision representation; hfield collision support does not imply hfield raycast support.
- [ ] P0B clearance envelope, sensor footprint, MJX/CPU/WASM gap, and installed/runtime version pin.
- [ ] P1–P7 blocked pending P0 evidence.

### Engineering decisions
1. Preserve 220/232 contracts and every existing released policy file. The current champion is schema 2; tests and examiner must acknowledge it rather than alter its file or coerce it to 220.
2. For schema-2 examiner loading, validate sensor field order, normalizer shape/mask, source hashes and recompute the locally authoritative sensor-contract fingerprint; keep legacy schema-1 decoding.
3. Keep `terrain_312` provisional. No contact, sensor coverage or learning success is proved by the initial script alone.
4. `@mujoco/mujoco` resolves to **3.15.0** in npm lock; the browser package declares a mutable `latest` selector and will require pinning in P0B.

### Reproduce on the user's fresh WSL

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/setup_wsl.sh --cuda13 --install-node
python scripts/v5018_terrain_spike.py --batch
```

If full suite fails again, keep the output. To finish dependency/viewer installation independently use `bash scripts/setup_wsl.sh --cuda13 --install-node --skip-tests` (clearly **not a test pass**), then run targeted tests with `python -m pytest -q tests/test_direction.py::test_protected_export_reader_refuses_contract_and_actor_tampering tests/test_physics.py::test_protected_sensor_champion_full_directional_exam_cpu`. Send GPU diagnostics, log and `v5018_p0a_spike_v2_results.json`.

### Next
Resolve any newly observed schema-2 exam failure without weakening hash/shape checks. Then build same-topology two-world MJX collidable fixtures and sensor-ray occlusion probe. Update only this checklist after each pass.
