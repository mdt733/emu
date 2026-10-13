# EMU Browser Viewer

**[Try the public Playground](https://mdt733.github.io/emu/)** · [Project introduction](../README.md)

This is the **only editable browser source**. The package-side `emu/viewer_src/` directory is a generated mirror updated by `emu viewer-build`; keep it synchronised before committing. JavaScript runs exported actor inference and MuJoCo WASM locally.

**Cameras:** View: orbit → View: POV → View: level POV. Raw POV is attached to the chassis and sees its real shake; level POV stabilises only the *image*. **Telemetry:** Robot lean describes a feet-midpoint-to-chassis-centre line; chassis pitch measures torso/IMU rotation. [Definitions](../docs/architecture.md#robot-lean-vs-chassis-pitch).

**Experimental 1° browser ramp:** Settings enables real MuJoCo fixed-box collision geometry (up-ramp, level landing, down-ramp), resets the simulation and leaves the GPU training code and actor weights untouched. Ramp transitions aren't certified abilities of the current champion.

---

# EMU browser viewer — local and public modes

The **canonical web source** is in `viewer/`. `emu/viewer_src/` is a generated packaging mirror for Python installations; run `emu viewer-build` after editing canonical files. Browser physics uses MuJoCo WASM, rendering uses Three.js, and inference runs the same schema-1 220→256→256→6 actor in JavaScript. No Python/JAX inference takes place inside the web page.

## Choose the right viewer

| Mode | Launch | Policy feed |
|---|---|---|
| Local policy | `emu viewer path/to/model.emupolicy --no-open` then `http://127.0.0.1:8765/` | One selected local bundle; can also drag-drop |
| **Local live training** | `emu viewer --live --no-open` then `http://127.0.0.1:8765/` | Read-only validated checkpoints; browser polls about every 12 seconds |
| **Public/static preview** | `python scripts/prepare_showcase.py`; `emu viewer-build`; `cd viewer && npm run preview -- --host 127.0.0.1`; open `http://127.0.0.1:4173/` | Static `showcase.json` and bundled `.emupolicy`, **not live training** |

The local watcher uses CPU for checkpoint validation and never feeds browser telemetry into the PPO learner. Compatible actor weights hot-swap with a fresh simulator episode. Invalid/incomplete writes do not replace the last validated policy. It is loopback-only and unauthenticated: **never port-forward it**.

The public static Playground requires no viewer Python API, no GPU training server, and no local WSL machine once deployed. It must show **UNTRAINED PREVIEW** until a real held-out-selected champion is published. The Vite dev/preview server can reply with HTML for nonexistent paths: the viewer now checks response MIME before parsing JSON. See the repository-root `docs/PLAYGROUND.md` for publication instructions.

## Install/build

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
emu viewer-build
```

Node 24 is selected by `.nvmrc`. The viewer uses `viewer/package.json` **plus committed `viewer/package-lock.json`**, and `emu viewer-build` uses `npm ci`; it is not a Python `requirements.txt` dependency. For development, `cd viewer && npm run dev` is separate from the checkpoint-watching Python server.

## Interaction and visual cues

- **Left stick / W-S:** forward and reverse `vx`; **left/right / A-D:** lateral motion. In the current **manual joystick mapping**, screen-right maps to **negative canonical `vy`**. This sign is deliberate for the viewer camera/robot convention, not a reversal of the actor's physical command definition.
- **Right stick / Q-E:** turning `yaw` (right maps to negative yaw). Right stick up/down or `R/F`: pitch command.
- `Space`: pause/resume; `Backspace`: reset. Manual joystick input disengages autonomous demonstration.
- **Settings drawer:** follow camera, auto-reset, training-like sensor noise (enabled by default), advanced commands, live/demo controls and grid/trail/shadow toggles. The drawer does not sit over the joysticks when closed.
- **World cues:** fine 0.10 m square grid, stronger metre marks, fixed X/Y reference, bounded movement trail, gentle sky/fog, warm shadow light with smaller 1024² shadow map. These do not affect physics and can be disabled for slower mobile GPUs.

In live mode the autonomous command sampler uses training-field-evidenced nodes when available (otherwise conservative limits) and changes commands periodically. It is **an interesting visual demonstration, not an independent held-out score**. A zero-action `demo_zero.emupolicy` fixture only confirms rendering; it cannot walk.

## Exported-policy contract

`.emupolicy` schema 1 contains model XML, named observation order, actor layers, normalizer, control limits, provenance and optional compact training-field candidates. It excludes critic, optimiser, PPO runtime and held-out examination feedback. Consult [`POLICY_FORMAT.md`](POLICY_FORMAT.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
