# v5042 Prism Terrain Lab — viewer-only geometry prototype

**Purpose:** Let humans inspect the geometric joins between piecewise-planar triangular-prism terrain tiles before proposing MJX training changes. This does **not** change training, PPO, reward calculation, MJX contact geometry, existing nine-box viewer physics or checkpoints.

## Open the interactive lab

From WSL on a machine with Node.js/npm:

```bash
git clone https://github.com/mdt733/emu.git ~/emu-prism-viewer
cd ~/emu-prism-viewer
git switch --track origin/feature/v5042-prism-terrain-viewer-lab
cd emu/viewer_src
npm ci --no-audit --no-fund
npm run dev -- --host 0.0.0.0
```

Open **http://localhost:5173/prism-lab.html** in your browser (use the port printed by Vite if 5173 is occupied). No policy file, Python server, GPU or MuJoCo installation is needed for this page. Or open the regular EMU viewer and click **△ Prism Lab**. Changes to slider inputs rebuild *only* this visual model.

## How to inspect

- Set **Steps: None**, **Base slope: 15°**, **Transition: 10°**, and try Rolling / Crest / Saddle with different diagonals. There should be no vertical ledge anywhere between neighbouring 50 cm tiles.
- Switch to **Step UP**, **Step DOWN**, **Up + crossing down step**, then **Staircase**, and inspect the deliberate orange vertical risers and their corners.
- Set **3 × 3** to inspect the proposed 18-prism active window; try 5 × 5 or 7 × 7 to see longer transitions.
- Toggle **Triangle / tile seams**, **Full prism outlines**, **Transparent prism bodies**, and the **36 mm foot reference**. Raise **Height exaggeration** if the millimetre-scale edges are hard to see; it only scales the presentation.
- Use **Copy settings link** to share a reproducible configuration and **Save screenshot** for visual feedback.

## Geometry contract

- Four top corners per 50 cm tile; diagonal cuts each square into two planar triangles.
- Every top triangle is extruded to a common lower plane, creating a closed convex triangular-prism *geometric* solid (six points).
- Underlying slope/transition height is sampled deterministically from global XY coordinates. Adjacent continuous tiles reference identical height coordinates, independent of tile order and diagonal pattern.
- Deliberate step patterns apply discrete, tile-local offsets. Only edges with differing offsets have signed elevations on opposite sides and are capped by explicit vertical riser quads.
- Seam diagnostics compare both actual edge endpoints and report unexpected seam discontinuities separately from intended lip height.
- The diagonal split creates changes in surface normal (creases), **not** infinitely differentiable terrain. Truly curved support requires finer mesh resolution.

## Evidence / limitations

`node --test tests/prism-terrain.test.mjs` validates 3/5/7/9-tile counts, continuous edge equality across shapes and diagonals, intentional risers, barycentric height sampling and deterministic seeds.

This lab is **Three.js visualization only**, not proof that 18 runtime-updatable convex prism collision meshes are supported in MJX. MuJoCo may use convex-hull collision for meshes; neighbour internal faces, step-corner contacts, edge tolerances, and GPU compile/performance still require separate investigation. There is **no NN or foot-contact solver running in this lab**. Existing browser robot simulator, MJX training and PR #30 remain untouched.
