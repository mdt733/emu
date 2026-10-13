# EMU v5014 browser viewer

This is the **canonical browser source** for EMU, not a separate training project. Install EMU itself and use `emu viewer` or `emu viewer --live`; see the repository-root [README](../README.md).

## Two modes

- **Local model viewer** — load or drag `.emupolicy` JSON bundles. Physics executes via official MuJoCo WebAssembly, rendering via Three.js, and the 220→256→256→6 actor via Float32 JavaScript inference.
- **Live training demonstration** — `emu viewer --live` serves the newest validated, fully written checkpoint from a production lineage. The browser polls every ~12 seconds, hot-swaps compatible actor/normalizer weights, begins a fresh episode, and cycles through training-field-evidenced random command nodes every 5–10 seconds. Until the field has enough evidence, the sampler uses conservative curriculum bounds. It is **not a held-out exam** and it never teaches PPO.

The watcher runs in a separate loopback-only CPU process. The active GPU trainer is unaffected by the viewer. Do not port-forward this unauthenticated development service.

## Build

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
emu viewer-build
emu viewer --live
```

For front-end development you can run `npm install && npm run dev` inside this folder instead, then drag/drop exported policy files. Vite without the Python server has no live-checkpoint API.

No JS dependencies are included in the source archive: `emu viewer-build` fetches them using npm. `.wasm` is loaded using an explicit Vite asset URL and served with `application/wasm` from the Python server.

## Controls

- Left joystick or `W/S`: forward/reverse (`vx`). Left joystick left/right or `A/D`: lateral (`vy`).
- Right joystick left/right or `Q/E`: turning (`yaw`). Right stick up/down or `R/F`: chassis-pitch command.
- `Space`: pause/resume. `Backspace`: reset.
- Toggle **Automatically load newer policies** or **Demonstration** independently. Manual joystick movement disables the demonstration.
- The optional training-like sensor noise is visualisation only; the deterministic actor default remains active.

The lateral/yaw sign mapping was corrected in v5014 from operator feedback; check it with the real browser simulation. The first live-policy snapshot may take longer to load because strict checkpoint validation runs before export.

## Deployment contract

`.emupolicy` schema 1 includes the robot XML, six-actuator target contract, actor layers, normalization, provenance and optional compact field-evidenced node candidates. It does **not** include critic/optimizer, PPO runtime, raw capability map or held-out exam feedback. The exported command frontiers are trainer-derived boundaries rather than certified physics capability. Browser actors can be driven from other devices only if you separately design a suitably authenticated service; the built-in `emu viewer` binds to 127.0.0.1.

See [`POLICY_FORMAT.md`](POLICY_FORMAT.md) for the JSON contract and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependencies.
