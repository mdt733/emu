# v5014 audited release validation (2026-10-13)

## Passed locally

- `PYTHONPATH=. pytest -q`: **92 passed, 1 skipped, 5 subtests passed**.
- `node --check` for `viewer/src/main.js`, `sim.js`, and `live-demo.js`.
- 10,000 deterministic command-generator samples stayed within exported physical command limits, including seven maneuver families.
- Synthetic schema-7 checkpoint saved and successfully exported into browser `.emupolicy` format, including one training-field-evidenced demonstration node.
- JavaScript `parsePolicy` + `PolicyRunner` loaded that export and correctly computed a 6-action vector from 220 observations.
- Unit tests cover atomic checkpoint replacement on interrupted pickle dump, read-only viewer watcher revision cache/corruption fallback, source phase selection, training-field node export, HTTP API config and WASM Content-Type.

## Not yet passed in this environment

- Browser bundle npm installation/build: dependency installation timed out (packages not installed in this container); no physical MuJoCo/WASM visualization claim is made here.
- CUDA+MJX training execution: full GPU prerequisites and user training checkpoints are absent. Run `emu doctor --domain-smoke` and an actual multi-stage run on the training PC.
- Behavioural comparison of lateral/yaw directions: the UI mapping follows the user's feedback, but should be checked by eye in the running viewer.

## Post-install acceptance checklist

- Run `emu viewer-build` and open `emu viewer --live` from Chrome or Edge with no policy: no blank overlay, status says waiting for validated checkpoint.
- Run a known real `.emupolicy`, verify MuJoCo loads `.wasm` with correct MIME, robot appears, sim works, and drop overlay disappears.
- Test sideways left/right and turn left/right against operator expectation, then keyboard WASD and Q/E.
- Confirm autonomous demo changes direction/cadence after ~5–10 seconds, restarts after falls, and hands control back to manual joystick immediately.
- Confirm a newly completed checkpoint updates displayed step and actor hash without modifying trainer throughput and without opening the GPU from the viewer process.
- Run fixed held-out evaluation on completed 400M policy and compare against v5013 shadow/graph baseline; do not treat live-browser demos as exam scores.
- Offline Python wheel build (`pip wheel . --no-deps --no-build-isolation`) succeeded and contained the mirrored browser assets, `emu/VERSION`, and `robot.xml`. This verifies packaging structure, not the npm runtime build.

## Functional audit additions (post original release)

- **Fixed a launch-blocking JAX bug:** phase-rate sampling erroneously called `jnp.minimum` with three positional arguments, accidentally inserting the pitch-command limit. A CPU/JIT executable regression now catches this specific failure.
- **Checkpoint finiteness:** final and rotating snapshots now use the same host-side finite-policy guard as `latest.pkl`; rejected snapshots do not rotate away usable checkpoints.
- **Hardened lineage safety:** completed/partial phases cannot silently resume under different seeds, environment counts, capability-teacher schedules, or domain-randomisation radii; expected `training_config.json` is validated before any checkpoint reuse.
- **Viewer sensor fidelity:** the default now enables the normal training 1.5× synthetic noise, rather than offering a no-noise nominal view as the default. This is still only a single-browser-world approximation, not a quantitative parity certificate.
- **Browser gait cadence:** model display now samples episode cadence once at birth, matching training’s birth-speed dependence and its fixed cadence through command transitions; a standalone JavaScript range test is included.
- **Browser hidden panels:** explicitly respect the `[hidden]` attribute across overlays/panels.
- **Still outstanding:** live GPU training, browser WebAssembly runtime and npm lockfile reproducibility need user-machine acceptance before calling v5014 fully proven.

## Owner's WSL acceptance evidence (2026-10-13)

The operator confirmed the installed `v5014` on the intended GPU system:

- `93 passed, 5 subtests passed` on the existing audited source, before this portability patch.
- `emu doctor --domain-smoke` completed with JAX 0.11.2 GPU (`CudaDevice(id=0)`), MuJoCo 3.15.0, reset/step observation `(8,220)`, and randomized terrain slope up to 2.64 degrees.
- `emu viewer-build` succeeded under Node/npm with Vite 7.3.7 and created the 10.37 MB MuJoCo WASM asset; npm reported no audited vulnerabilities.
- The Python viewer served index, CSS, JavaScript, initial policy and `.wasm` with HTTP 200. The unrelated favicon returned 404.

The Vite `module` externalization and chunk-size notices did not stop the build. **These logs do not by themselves verify WebGL rendering, manual control directions, live hot-swap or final GPU training.**

The reproducibility/cleanup patch adds `npm ci` lockfile enforcement, fresh-machine setup and workspace archive tooling, plus two unit regressions; rerun the full suite and browser acceptance after applying it. The real Node dependency lock comes from the successful operator build and must be committed with the source. Python dependency versions remain bounded by `pyproject.toml` and GPU/JAX platform selection, rather than being falsely described as bit-for-bit locked across machines.
