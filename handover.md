# EMU handover — H40 120M mainline

Updated 2026-11-09. **This is the one active handover.** Git history preserves all retired experiments and runbooks; the canonical source maintains just one physical robot, 360-input actor, terrain generator, and modern browser.

## Status and lineage

- **Chosen successor:** H40, 400 mm nominal root height, six central C018 actuators, new VL53L8CX sensor contract, four 1 m terrain tiles.
- **Complete cold-start run:** `h40-1m-tof70-120m-s735`, seed 735, **120,061,952** world-steps, trained at exact commit `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`. Source, final checkpoint, metrics, and intermediate checkpoints in `~/emu-h40-120m` and `~/emu-data/runs/h40-1m-tof70-120m-s735/` must remain frozen.
- **Proven policy export:** `showcase/champion.emupolicy`, uploaded owner-supplied bytes SHA-256 `178c201af661d3172d701a7d1d8a7711f9b2106e1f7058bc4902aa9c64a37c7a`, Git blob `eef79b1010f2a9561d0b72dd7aae3e0fcef512ae`; 360→256→256→6 network; actor SHA-256 `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673`.
- **Research code:** [PR #52](https://github.com/mdt733/emu/pull/52) at `f7e3fc851c2942b5f2785aaf068882c6fd7d32c5`; training physics used for the 120M policy was frozen at the earlier exact source commit. Subsequent code includes CPU/browser fixture improvements and must not be represented as having retrained the policy.
- **Promotion completed:** H40 source and the byte-identical 120M export entered `main` via [PR #53](https://github.com/mdt733/emu/pull/53), commit `1b796144b128b326be1ac54f6a289ad87857cde6`; its 24 Python tests, CPU MJX checks and browser CI passed. Public GitHub Pages deployment is **still separate** and requires exact actor approval, retained checkpoint SHA-256, and `python scripts/verify_showcase_release.py`. A source merge must not deploy the site.

### What is demonstrated

The operator reports a good fast gait and roughly **15° slopes** while inspecting the exported trained actor. This is a qualitative owner observation, not a controlled held-out MJX examination. **Up/down step traversal remains unreliable**, including the baseline's historical limitations; no successful 3–4 cm stair capability can be claimed. No robot has been physically built or electrically/thermally validated.

The original `final.pkl` is **not in this repository or in the chat upload**. It is not interchangeable with the 863 KiB inference-only `.emupolicy`. The checkpoint checksum was measured on the original training PC as `632c09788613e0e5fdc077637efd0a4bf4fc1be7ae0c07dc3662087c6c12970b` and recorded in strict `showcase/release_approval.json`. The checkpoint bytes remain private/off GitHub; do not loosen the release gate.

```bash
sha256sum ~/emu-data/runs/h40-1m-tof70-120m-s735/checkpoints/final.pkl
```

## The one H40 physical design

`emu/robot.xml` + `emu/physical_variants.py` generate the model, used identically by the local sensor laboratory. No separate painted browser-only robot or fake structural braces.

- Six **12 V Feetech ST-3215-C018** motors are inside the **460 g estimated chassis** (330 g of motors included), centered 5 mm forward of hip axes. Remote knee transmissions are represented by idealized joint-space actuators. No knee-mounted servo mass.
- Two original small hip carriers, 36 mm-diameter ball feet, one 2×2 mm square carbon thigh rod and one split 2 mm round carbon shin per leg. Total estimated mass **~506 g**. Batteries and mass distribution are estimates; real BOM, backlash, transmission friction, electronics and cooling are unverified.
- Passive compression-knee support, shin-flex hinges and hip-roll centering remain. Position actuator `kv=0.624` is **inside** each ±0.980665 N·m `forcerange`; it is not unlimited passive joint braking. Motor torque-speed/thermal curve and 0.08 N·m joint friction are not characterized.

## The one sensor and world

- VL53L8CX 4×4 approximation: 45°×45° FoV, 70° down, 0 mm forward, 30 mm below torso, no lateral/yaw offset. Sixteen centre-of-zone rays, delayed/noisy confidence and nearest foot-sphere occlusion.
- ST receiving lens inverts both axes. Nominal scene-to-raw zone order: `3,2,1,0 | 7,6,5,4 | 11,10,9,8 | 15,14,13,12`; real PCB orientation and which bottom zones feet occlude still need hardware checks.
- Canonical sensor SHA-256: `d8f559641d240f6eb4bd54a3dc7e96f414994682ebc1b478209df202cc903995`. Older sensor/checkpoint schemas are rejected.
- MJX trainer uses rolling 2×2 four **1 m** tiles, eight true convex triangular prisms, with recycler support safety and signed step goals. CPU browser **Training Carpet** uses a static eight-prism fixture, not an infinite/stochastic terrain replay. **Field** mode is a separate longer gait/slope diagnostic.

## Next work: no branch explosion

1. **Complete:** H40 promotion PR #53 passed Python/physics and browser CI; source and trained actor identities remained unchanged. Continue to run these checks on future maintenance changes.
2. **Checksum and release approval supplied:** the original `final.pkl` SHA-256 is recorded in strict `showcase/release_approval.json`. Verify it against the policy and the final compiled site in CI before triggering the **manual** Pages workflow; the old public site stays untouched until a successful deployment.
3. Keep the H40 baseline stable. Future step experiments start from a new feature branch **after** promotion; do not rewrite its source, checkpoint or weights.
4. For stairs, use a **source-pinned independent MJX exam** with successful actual up/down lip crossing, controlled post-step movement, and command/heading survival. Browser footage is not a valid pass condition.

## Run it

```bash
# CPU browser with the trained export (not the untrained sensor-lab flag)
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy --no-open --port 8765

# Separate geometry and ray diagnostic: untrained
bash scripts/start_sensor_lab_wsl.sh
```

The local trained actor may also be loaded directly from `~/emu-data/actors/emu-h40-120m.emupolicy`. Do not invoke `--sensor-lab` to view neural inference. The old champion, older compatible interfaces and research data stay available through Git history, not active code.
