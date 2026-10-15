# EMU v5014 — dated historical verification ledger

> Archived snapshot from 13 October 2026. The currently validated code and open limitations are in [Status](../STATUS.md). The following findings refer to the older v5014 release, not the current training ladder or public Pages state.

**Purpose:** record what was *observed*, on which environment, and what remains unproven. This is not a setup guide; see [docs/validation.md](docs/validation.md) for the test procedure. Do not turn a prior test count into a claim about later modified code.

## Operator's WSL workstation — 13 October 2026

| Gate | Observed result | Scope |
|---|---|---|
| Python package | `emu-biped==5014.0.0` editable install; v5014 ten-command CLI | Installed checkout |
| Python suite | **93 passed; five subtests passed**, after audited RC1 install | Before later Playground/visual/joystick patches |
| GPU JAX / MJX | JAX 0.11.2, CUDA device 0, MuJoCo 3.15.0, reset/step **PASS**, domain smoke **PASS**, observation `(8, 220)` | Real WSL NVIDIA GPU, 8-world smoke |
| Viewer build | Node/npm installed; Vite 7.3.7 built JS/CSS and ~10.37 MB MuJoCo WASM | Real WSL, before later front-end add-ons |
| Python viewer server | Index, JS, CSS, initial policy, `.wasm` returned HTTP 200 | Does not alone prove graphical execution |
| Browser / live | Operator later confirmed the polished browser looked better and **live training demo loading worked** after switching back from static preview to `:8765` | Functional visual report; not a held-out evaluation |
| Visuals | 10 cm grid, right-side settings drawer and lighting upgrade reported pleasing | Operator qualitative acceptance |
| Lateral joystick | `vy` correction delivered using shared `controls.js`; operator later reported satisfaction | Specific sign retest was not pasted into chat |

Expected non-blocking notices: missing optional Warp import while MJX passed; CUDA timer accuracy warning; Vite externalised `module`; chunk size warning; favicon 404 before v3 corrected it. None of these logs establish GPU training throughput.

## Maintainer source audit (before the user's machine tests)

The original RC1 audit executed **92 Python tests, one skip, five subtests**, JAX/JIT cadence regression and synthetic policy Python-export→JavaScript 220→6 inference. It found and repaired: invalid three-argument `jnp.minimum` at environment birth, unsafe cross-phase checkpoint reuse, non-finite snapshot promotion, mismatch in viewer birth cadence/noise and hidden CSS panels. Details in [docs/AUDIT_REPORT_5014.md](docs/AUDIT_REPORT_5014.md).

Subsequent Playground development reported additional targeted JS/Python tests, but **no final whole-repository test transcript** from the user's checkout after the last changes is available. Re-run [the checklist](docs/validation.md) after this documentation update.

## Not yet established

- Full uninterrupted/resumed **400M** lineage completion and actual last-step checkpoint; throughput, collapse/recovery history and held-out outcome.
- Held-out selection of a champion; physical transfer to hardware.
- Public GitHub Pages publication and deployment URL; `main` merge/push/tag status.
- Browser parity with all hidden MJX physical variations (the browser is a nominal visual simulator, not a robust-domain exam).

If a later test fails, add the actual command, revision and failure here. If a gate passes, record its machine and date rather than rewriting prior evidence.
