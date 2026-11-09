# EMU Playground — H40 local viewer and separately gated public site

[Public playground](https://mdt733.github.io/emu/) · [H40 handover](../handover.md) · [Release gate](RELEASE_READINESS.md)

The **H40 120M** actor is now checked into GitHub `main` as `showcase/champion.emupolicy`. Its source is a 40 cm, six-servo physical model with a 360-input learned policy and 45°×45° VL53L8CX approximation. **Do not assume the public Pages URL is updated:** Pages is manual. The checkpoint-backed `showcase/release_approval.json` has now been supplied, but a successful CI verification and explicit workflow dispatch are required before declaring the site updated.

## What the local browser can show

The local MuJoCo/WASM viewer runs frozen inference, not GPU PPO. The same deployed 360-input actor receives commands from the manual sticks and demo navigation through one command path.

- **Field/Park:** a larger walking demonstration, long-run gait/slope exploration and adjustable park challenges. This is not the MJX trainer's 2×2 rolling terrain. Autonomous route planning may choose paths around obstacles; choose manual control for a direct terrain challenge.
- **Training Carpet:** a **static** 2×2 m fixture of four 1 m tiles and eight real triangular-prism CPU colliders. Slope and signed-step settings are configurable, but it is not an endless recycler, an exact stochastic training replay or an independent physics exam. Reset before the robot leaves the fixture.
- **Sensor inspection:** a 4×4 visualization for checking the 70° downward, 30 mm below-torso mount and foot occlusion. A centre ray per zone is still a sensor approximation.
- **Separate untrained lab:** physical H40 geometry and ray calibration with no neural policy. Its motion must not be mislabelled learned locomotion.

Viewer success on one CPU scene is **not** proof of autonomous step traversal or real hardware feasibility.

## Start the actual trained H40

From an updated checkout of `main` on WSL/Ubuntu, with Node/npm installed:

```bash
cd ~/emu
bash scripts/start_sensor_lab_wsl.sh --policy showcase/champion.emupolicy
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

The separate calibration-only viewer uses the same script **without** `--policy`:

```bash
bash scripts/start_sensor_lab_wsl.sh
```

You may instead run `python -m emu.cli viewer-build` followed by
`python -m emu.cli viewer showcase/champion.emupolicy --no-open --port 8765`.

For live checkpoint inspection, use `emu viewer --live --root RUN_DIR --no-open` from the source-compatible checkout. Never resume or mutate the frozen H40 training run merely to see it in the viewer.

## Publishing

`scripts/verify_showcase_release.py` checks the exact deployed actor, physical 360-input sensor contract, weight shapes, Git blob, original checkpoint SHA-256 and approval metadata. The manual [Pages workflow](../.github/workflows/emu-playground.yml) refuses an unapproved actor; source merges do **not** deploy.

The old v5032 graduated park specification is preserved in the [historical research ledger](RESEARCH_FINDINGS.md), which links to its original Git version. Its measured successes cannot be ascribed to H40.

For current training-vs-viewer differences, source identity and outstanding H40 tests, see the [handover](../handover.md).
