# EMU's 20 essential checks

Exactly **20 automated tests** are kept: **16 Python** and **4 browser**.
There is no hidden “full” suite, quick-test manifest or pass-numbered test collection.

Run them with one command:

```bash
bash scripts/check.sh
```

The files answer five practical questions:

| Question | Checks | Location |
|---|---:|---|
| Are rewards, configuration and training state handled correctly? | 5 | `tests/test_learning.py` |
| Are servo physics and the real MJX evaluation path sane? | 3 | `tests/test_physics.py` |
| Does the directional examiner reject cheating, and does slope sensing distinguish chassis lean from terrain? | 5 | `tests/test_direction.py` |
| Are the branch ranking, results ZIP and exported browser actor reliable? | 3 | `tests/test_delivery.py` |
| Does the browser obey joystick commands, reject unsafe policy paths, report missing assets and create real ramp geometry? | 4 | `emu/viewer_src/tests/essential.test.mjs` |

The selected tests are actual behavioural checks retained from the earlier suite—not placeholder tests that only check imports. We deliberately **deleted the other assertions**, rather than hiding them behind a different command. Python integration includes a small real MJX/JAX rollout, and the browser parity test executes Node inference on the same actor input as a NumPy reference.

**20 passing tests do not prove EMU can walk.** The separate practical gates are:

- **GPU lifecycle:** short new-root PPO update, checkpoint save, resume and re-evaluation using `scripts/smoke_training_lifecycle.py`; do not run against an active training directory.
- **Real walking:** `emu directional-batch --profile smoke` then `pilot` on saved policies, checking actual course progress, yaw tracking, survival, and contact-based stride candidates.
- **Release:** build the browser and validate policy/WASM packaging; CI does this independently of the four browser behaviour tests.

If one of the 20 no longer protects a real failure, replace or delete it—don't grow a second test taxonomy. The objective is understandable production code and honest physical evidence, not an assertion count.
