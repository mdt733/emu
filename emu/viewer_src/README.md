# EMU browser viewer

**[Public Playground](https://mdt733.github.io/emu/)** · [Project](../../README.md) · [Static publishing](../../docs/PLAYGROUND.md) · [Policy format](POLICY_FORMAT.md)

This directory is the **only canonical editable frontend**: browser MuJoCo WebAssembly physics, Three.js rendering and deterministic JavaScript actor inference. It is included in the Python package. No generated second source tree is tracked.

## The three viewer modes

| Mode | Command | Policy source |
|---|---|---|
| Local static policy | `emu viewer path/to/actor.emupolicy --no-open` | One user-selected export |
| Live local watcher | `emu viewer --live --root ~/robot_lab/emu_runs/v5015_lab1 --no-open` | Validated completed/checkpointed policies, read-only |
| Public/static preview | `python scripts/prepare_showcase.py --require-champion && npm run build --prefix emu/viewer_src`, then `npm run preview` in `emu/viewer_src/` | Tracked public 60M champion |

The first two serve `http://127.0.0.1:8765/` on loopback only. The **live watcher is unauthenticated**; never port-forward or deploy it publicly. The browser refreshes validated policy revisions without mutating trainer state. The static Pages demo requires no private Python endpoint or CUDA device.

`demo_zero.emupolicy` is a **zero-weight rendering fixture**, not the trained public champion.

## Controls and presentation

- Left joystick: forward/back and lateral movement. In the current camera convention, screen-right maps to negative canonical `vy`.
- Right joystick: yaw (screen-right is negative yaw) and commanded chassis pitch.
- View button: orbit → raw chassis POV → image-levelled POV. Levelling is **purely visual**.
- Settings: demo commands, sensor noise, shadows/grid/trail, and an experimental **1° physical collision ramp**. The ramp resets the browser world; it does not affect the actor or GPU training.
- Robot lean means feet-midpoint-to-chassis-centre geometry; chassis pitch means torso rotation relative to gravity. They are not the same variable.

A browser robot that appears to walk is an illustration, not an independent physical-capability score. The published policy is not certified to negotiate physical ramp transitions.

## Development

Requires Node 24 and the committed `emu/viewer_src/package-lock.json`:

```bash
npm ci --prefix emu/viewer_src --no-audit --no-fund
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
```

Run the local Vite server using `cd emu/viewer_src && npm run dev`. Build the package-owned browser with `emu viewer-build`. Keep `emu/viewer_src/dist/` and `node_modules/` out of Git.

The Python test `tests/test_policy_inference_parity.py` constructs a deterministic actor bundle and compares the **actual browser PolicyRunner** with a NumPy reference on several frozen observations. It checks inference and float32 serialisation, **not** equivalence of MJX and WASM sensor/physics pipelines.

[Policy format](POLICY_FORMAT.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Playground release](../../docs/PLAYGROUND.md)
