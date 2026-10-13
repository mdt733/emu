# Release validation checklist

This is the **procedure**. For tests actually run, dates and results, see [../RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md). For the current experiment see [tournament.md](tournament.md); v5014's 400M recipe is archived in [v5014-production.md](v5014-production.md).

## Source and contract checks

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
python -m pytest -q
node --test viewer/tests/*.test.mjs
git diff --check
```

Check actor 55×4=220 observations, six controls, finite weights, schema-7 checkpoints, exact learning-state resume, phase-configuration compatibility, field-event semantics, held-out quarantine and safe snapshot replacement. Python tests are **not** a substitute for a real MJX compile.

## Target machine / actual viewer

```bash
emu doctor --domain-smoke
emu ladder --stage plan
emu viewer-build
python scripts/prepare_showcase.py
emu viewer-build
python scripts/verify_showcase_dist.py
```

Run `emu viewer --live --no-open` at `127.0.0.1:8765` and confirm it receives completed checkpoints, updates the revision without modifying the trainer, and actually renders a robot. Run `npm run preview -- --host 127.0.0.1` inside `viewer/`, open `127.0.0.1:4173` and confirm it autoloads the **untrained** static policy (not the live feed). Verify MIME-safe fallback, settings drawer, 10 cm grid, trail, shadows, reset/overlay, joystick `vy`/yaw signs, and no console errors. Run an actual MJX rollout/training/evaluation smoke appropriate to the GPU before changing production settings.

## Evidence and release gates

- Do **not** equate passing unit tests, WASM HTTP 200, browser visual activity or training-derived command nodes with verified locomotion quality.
- Record versions (`python`, `node`, `npm`, MuJoCo, JAX, CUDA), code revision (`git rev-parse HEAD` plus dirty status), observed test results and training checkpoint step.
- Choose the final public policy with **held-out evaluation**; never by elapsed steps or live browser appearance alone.
- Review `git diff --cached --name-only`, `git diff --cached --check`, remote, branch and tags before any merge or push. Reject raw `.pkl`, private configs, `node_modules/`, `.venv/`, generated `dist/`.
- Validate the GitHub Pages build artefact (`index.html`, manifest, bundled `.emupolicy`, MuJoCo WASM) before deployment. A public preview must say it is untrained until a true champion exists.

## CLI inventory

`emu --help` is authoritative. v5014 currently exposes `train`, `compare`, `tournament`, `view`, `evaluate`, `doctor`, `forge`, `export-policy`, `viewer-build` and `viewer`. The first set retains historical/experimental entry points; the intended production path is `forge` plus `viewer --live`.
