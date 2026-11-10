# EMU Playground — H40 local viewer and separately gated public site

[Public playground](https://mdt733.github.io/emu/) · [H40 handover](../handover.md) · [Release gate](RELEASE_READINESS.md)

The **H40 120M** actor is now checked into GitHub `main` as `showcase/champion.emupolicy`. Its source is a 40 cm, six-servo physical model with a 360-input learned policy and 45°×45° VL53L8CX approximation. **Do not assume the public Pages URL is updated:** Pages is manual. The checkpoint-backed `showcase/release_approval.json` has now been supplied, but a successful CI verification and explicit workflow dispatch are required before declaring the site updated.

## What the local browser can show

The local MuJoCo/WASM viewer runs frozen inference, not GPU PPO. The same deployed 360-input actor receives commands from the manual sticks and demo navigation through one command path.

- **Field/Park:** a larger walking demonstration, long-run gait/slope exploration and adjustable park challenges. This is not the MJX trainer's 2×2 rolling terrain. Autonomous route planning may choose paths around obstacles; choose manual control for a direct terrain challenge.
- **Infinite Meadow Carpet:** endless world coordinates with a **moving 15×15 m window of 225 real MuJoCo CPU heightfield tiles** (each 1 m wide, two top triangles). Unused distant tiles recycle only after leaving the H40's physical reach, and their 2×2 height samples are safely updated in the compiled model. Previously visited terrain **regenerates identically** from its world coordinates and seed; the whole traveled route is *not* held in memory. The physical window has real support and a drop-off at its edge, while an additional **2.5 m visual-only falling apron** fades into the fog. +Y is **north**: difficulty (random slopes, signed steps and optional shove exposure) rises gradually toward a capped frontier, falls to easy terrain to the south, and stays at roughly the same envelope east–west. A warm procedural skybox, camera-following distant mountains, flowers and sparse walk-through trees are decorative, not collision geometry. The browser heightfield contact topology deliberately differs from the MJX trainer's **eight convex prisms**, so this is a qualitative playground, not a matched independent step exam.
**Unified Infinite Meadow difficulty:** Choose **Gentle**, **Rolling (default)**, **Rugged (10°)** or **Extreme**, or edit the sliders directly. The open viewer controls, not the training-specialist inspector, define the physical terrain: **Natural hill peaks at the starting latitude**, hill size (metres), maximum signed step-up/step-down heights, step-tile frequency, and northward difficulty gain (% per metre). When set to 10°, roughly the steepest tenth of sampled 1 m grades approach 10°; most hills are gentler and their directions change organically over large, continuous ridges and valleys. The target rises northward and falls southward. Changing the geological settings restarts the viewer robot to prevent shifting a slope under an occupied foot. Shoves are opt-in and independently adjustable. Imported training frontier values remain *read-only exercise/diagnostic context*, never an invisible override of physical geology.

**Natural hills (no sinusoidal terrain):** The former rolling terrain was built from three sine waves and made visually repetitive corrugations. The Infinite Meadow now samples seeded **2D gradient noise**, warped by independent low-frequency noise fields, with several non-aligned scales of regional hills, gently carved ridge/valley structure, and subdued short-wave surface detail. There is **no repeating waveform or fixed ridge direction**. The noise is indexed by global world coordinates, so streaming a tile away and returning reproduces exactly the same terrain. The `hillScaleM` slider changes the size of landforms while a measured 1m physical-slope percentile calibrates `hillPeakDeg` rather than applying a constant incline. The implementation is inexpensive and *erosion-inspired*, **not** a full hydraulic erosion simulation; each tile is still represented by two physical triangles, not a high-resolution landscape mesh. Testing a few seeds and long walks in the browser remains important for judging visual realism.

**Original Meadow panorama:** The exact 2,491,081-byte original `skybox.png` from [Meadow source commit 6a56516](https://github.com/mdt733/meadow/blob/6a56516386883c820a8b5ae283101c7dd636728f/skybox.png) has been copied *byte-for-byte* into EMU at `emu/viewer_src/src/assets/meadow-skybox.png` (Git blob `9eaed55fcc9a9f86846336808d5ecae56b43097c`). The Vite viewer bundles it locally, so the Infinite Meadow skybox works even offline or after the original Meadow project is retired. If image loading fails, the scene falls back to Meadow's procedural blue/peach sky. Mountain ridges, trees and flowers remain visual-only.

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
