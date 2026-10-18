# EMU Playground

**[Open the public browser demo](https://mdt733.github.io/emu/)**. The repository's exported champion is the **120M Pitch15 policy** (120,061,952 steps, 220 inputs → 6 actions). A previous published Pages build may still be live until someone intentionally dispatches the manual release workflow.

The browser runs a frozen neural network and MuJoCo WebAssembly physics, not PPO. Experiment with joysticks, view modes, lighting and the optional 1° physical ramp or radial terrain park; visual behaviour alone is not proof of real servo safety or matched MJX contact dynamics. `emu/viewer_src/` is the only editable browser source.

## Local view of checkpoints

```bash
source .venv/bin/activate
emu viewer --live --root "$HOME/robot_lab/emu_runs/my-study" --no-open
```

Open `http://127.0.0.1:8765/`. The local watcher selects completed validated snapshots, swaps the browser policy without teaching the trainer and serves a read-only unauthenticated API. **Do not expose this service publicly.** To view a fixed bundle: `emu viewer /path/to/brain.emupolicy --no-open`.

## Exact static preview

```bash
python scripts/prepare_showcase.py --require-champion
npm ci --prefix emu/viewer_src --no-audit --no-fund
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
python scripts/verify_showcase_dist.py
cd emu/viewer_src
npm run preview -- --host 127.0.0.1
```

Open the Vite preview URL (normally `http://127.0.0.1:4173/`), verify joystick directions (including inverted pitch), selectable heading hold, split/hybrid steering, both physical terrain modes, and that `showcase.json` reports step 120061952. The static site must bundle `dist/policies/champion.emupolicy` and a `.wasm` asset; a zero-action fixture does not count.

## Publish (manual on purpose)

After merging a reviewed code + viewer + real actor update into `main`, open GitHub **Actions → EMU Playground (GitHub Pages) → Run workflow → main**. The workflow verifies the exact Git blob SHA pinned in `.github/workflows/emu-playground.yml`, actor provenance, Node browser tests, WASM and policy files. Check the deployment job, then reload the public URL and inspect the step/build commit in the UI. **Merging Git code does not publish Pages.**

Do not commit a training `.pkl` or alter the protected `showcase/champion.emupolicy` without deliberately updating the verified pin in the SAME reviewed change.

## Viewer driving and terrain

By default, the left stick commands forward/sideways motion, and the right
stick commands yaw and chassis pitch. Pitch is inverted: pushing the right
stick up requests negative model pitch (the nose-up direction in this
model's convention). Select **Hybrid** for diagonal steering: forward plus
left/right diagonals mix turning and strafing, while purely sideways means
strafe. The right stick always supports separate turning and pitch.

**Heading hold** is an optional browser-only P controller (default on,
Kp=0.75). It integrates the requested turn rate into a target heading and
adds a bounded correction for observed torso yaw drift. It neither edits
the neural policy nor guarantees accurate heading on unfamiliar terrain.

Terrain choices are **flat**, **1° ramp**, and **radial terrain park**.
The park starts with a flat 2 m radius; the permitted incline grows
approximately 1 degree per metre beyond that (1° at the 2 m boundary,
2° at 3 m), capped by the chosen maximum (default 10°, adjustable 1–20°).
The default park extends about 11 metres each way so the outer region can
reach the 10° *allowed* gradient; actual generated slopes may be gentler.
The park is now a **single triangulated heightfield**: actual MuJoCo
contact triangles are rendered as a matching continuous grassy surface,
with a procedurally generated grass texture and daylight sky. A sparse
deterministic grove of trees is **enabled by default**: trunks are static
MuJoCo collision cylinders, while low-poly foliage is visual-only.
The radial park is the viewer's default environment. You can select flat
or the original ramp and turn trees on or off in Settings. Trees stay well
outside the 2 m flat spawn zone, and the landscape is
generated locally with no network image downloads.

Changing terrain, tree collisions or maximum slope rebuilds the browser
physics world and resets the on-screen robot. The training process and
saved .pkl remain untouched. Rendering continuity is better than tiled
boxes, but heightfield resolution and browser physics still need real
walking inspection, especially near small ball-foot contacts.
This is a viewer playground, not training terrain or independent walking proof.

## Heading and ground-slope instruments

The telemetry distinguishes **yaw requested** (joystick or demonstration) from
**yaw applied** (including bounded heading-hold P correction), and shows the
robot's measured yaw rate. The heading controller does not feed its correction
back into the next turn request. The automatic public demo uses gentler
maximum yaw rates than manual control, so turning at the full frontier is not
the default demonstration behaviour.

**Ground under feet** reports an approximate local incline in degrees,
derived from the actual generated heightfield beneath the midpoint of the two
feet. The green arrow points **uphill relative to the robot's own direction**:
up means forward, left means uphill to its left, right means uphill to its
right. Flat terrain reads `0.0° · level`; walking beyond the sampled terrain
reports `No terrain underfoot`. This is a terrain probe, not a measurement of
body pitch or proof that both feet have contact.


## Third-person chase camera

Cycle the **View** button: Orbit → Chase → POV → Level POV.
**Chase** follows behind and above EMU in the robot's *actual* heading,
with a slightly forward-looking aim. The camera smoothly handles turns
including the ±180° boundary, maintains a level horizon independently of
chassis roll/pitch, and keeps above the ground as the robot traverses the
terrain park. It snaps into position on reset or policy hot-swap instead
of slowly flying across the scene; returning to Orbit restores its earlier
orbit target and camera position. The ordinary Follow toggle applies to
Orbit; Chase always follows by definition.

The camera runs only in the browser renderer; it does not alter commands,
sensor observations, controls, physics, trained weights or campaign files.
