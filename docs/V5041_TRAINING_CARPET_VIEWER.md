# v5041 — Training Carpet **inside the real EMU viewer**

Research branch: `feature/v5041-unified-terrain-carpet`, based on the
v5040 STEP streaming branch. The frozen champion on `main` is unchanged.

## What is running and what is not

**Running now:** an additional `Training Carpet · live difficulty` terrain
mode in EMU's existing 3D MuJoCo/WebGL policy viewer, with its original
robot, 360-input sensor observer, camera views, grass, solid trees,
joysticks, heading hold and diagnostics.

The Terrain Park remains an **unchanged separate selectable mode**, with
all its trees and eight feature obstacles. Training Carpet uses the same
underlying real browser MuJoCo heightfield and grass renderer but replaces
the fixed radial terrain with a seeded 50cm-tile-based surface. A highlighted
moving 3×3 patch displays 18 visual triangles around the robot. The
browser builds a **finite 22m-wide physical hfield** at 12.5cm sampling;
the gold 3×3 guide follows the robot, but the browser does not continually
shift MJX colliders. Sharp ledges are rasterized and therefore approximated
by short steep triangles. Do not mistake this for the final vertically
walled, GPU-native rolling 3×3 polygon world.

**Research only:** `emu/terrain_carpet.py` holds the proposed single
specialist-to-terrain contract, and `scripts/probe_carpet_hfield.py`
confirms that small per-world heightfield samples can be changed in MJX
without recompiling XML. Neither substitutes for an actual 2048-env GPU
PPO training smoke. The v5040 training engine remains the real active
trainer until the heightfield+vertical-riser and live recycling collision
path passes a dedicated physical exam and GPU test.

## Live difficulty in the existing viewer

The viewer's **Settings → Terrain → Training Carpet** mode adds:
- Specialist selection: standing / command / resilience / slope / step /
  transition / control.
- An importer for the raw `metrics.csv` from any source-pinned EMU run,
  or `results_data.json` / `terrain snapshot.json`.
- A live read-only JSON endpoint `/api/live/terrain`, available when
  starting `emu viewer --live --root YOUR_RUN`; it refreshes every 5s
  while Training Carpet is visible.
- Measured slope, **separate UP and DOWN** lip frontiers and source step.
  Slope-angle-transition capability is an assumed preview value when
  historic results do not report it; the UI labels it as assumed.
- Per-specialist projected difficulty envelope; `slope` inherits half
  the step frontier while `step` inherits half the slope frontier.
- An explicit **Apply terrain to robot viewer** button that reloads only
  the browser's physical terrain. Live telemetry updates do NOT reset EMU
  during a good walk. A policy swap preserves the chosen role and limits.

### From Windows/WSL, in a separate, clean clone

```bash
git clone --branch feature/v5041-unified-terrain-carpet --single-branch \
  https://github.com/mdt733/emu.git ~/emu-carpet
cd ~/emu-carpet
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
emu viewer-build
emu viewer --live --root /home/mdt733/robot_lab/emu_runs/YOUR_RUN --no-open
```

Open `http://127.0.0.1:8765/` in a browser, load the latest
`.emupolicy` (or use the auto-watched validated checkpoint), open
**Settings → Terrain → Training Carpet**. Choose a specialist, then
**Refresh training values**, then **Apply terrain**. This live viewer
server remains bound to localhost; do not expose policy data directly
to the public internet.

Without a live run, select the new terrain mode then import the
training `metrics.csv` or `results_data.json` through Settings.

### Tests

```bash
python scripts/probe_carpet_hfield.py
python -m pytest -q tests/test_terrain_carpet.py
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
```

The MJX proof is a tiny CPU batch; it is NOT evidence the 9-tile rolling
collision topology fits an RTX 4060 Ti 8GB. Training and curriculum data
are read-only in this release, so the user can give visual feedback
without risking a long run's optimizer/checkpoints.
