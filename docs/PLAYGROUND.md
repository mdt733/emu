# EMU Playground

**[Open the public browser demo](https://mdt733.github.io/emu/)**. The repository's exported champion is the **120M Pitch15 policy** (120,061,952 steps, 220 inputs → 6 actions). A previous published Pages build may still be live until someone intentionally dispatches the manual release workflow.

The browser runs a frozen neural network and MuJoCo WebAssembly physics, not PPO. Experiment with joysticks, view modes, lighting and the 1° physical ramp; visual behaviour alone is not proof of real servo safety or matched MJX contact dynamics. `emu/viewer_src/` is the only editable browser source.

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

Open the Vite preview URL (normally `http://127.0.0.1:4173/`), verify joystick directions, terrain and that `showcase.json` reports step 120061952. The static site must bundle `dist/policies/champion.emupolicy` and a `.wasm` asset; a zero-action fixture does not count.

## Publish (manual on purpose)

After merging a reviewed code + viewer + real actor update into `main`, open GitHub **Actions → EMU Playground (GitHub Pages) → Run workflow → main**. The workflow verifies the exact Git blob SHA pinned in `.github/workflows/emu-playground.yml`, actor provenance, Node browser tests, WASM and policy files. Check the deployment job, then reload the public URL and inspect the step/build commit in the UI. **Merging Git code does not publish Pages.**

Do not commit a training `.pkl` or alter the protected `showcase/champion.emupolicy` without deliberately updating the verified pin in the SAME reviewed change.
