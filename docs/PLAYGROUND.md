# EMU Playground — public browser demo

**[Open the EMU Playground](https://mdt733.github.io/emu/)**

EMU's public MuJoCo/WASM viewer runs the real six-servo policy locally
in the browser. The approved `showcase/champion.emupolicy` is now the
**260,046,848-step v5032 Pinnacle actor** (360 observation inputs;
six actions), pinned by `showcase/release_approval.json`.
The actual public URL may still serve the previous build until the
manually approved GitHub Pages workflow finishes deployment.

## What the demo does

On loading, the viewer starts **pose-lattice autonomous navigation**.
A Web Worker chooses a reachable target pose and plans a route; a
feedback controller follows the continuous line, anticipates sideways
drift, slows as tracking error grows, and performs the final pose correction.
The trained actor, its observation contract, and MuJoCo physics remain
unchanged.

The left stick controls forward/reverse and sideways velocity. The right
stick controls yaw rate and chassis pitch. While autonomous navigation is
active, the stick knobs show the **requested** control inputs (not measured
motion), without pretending a human touched them.

Touch either stick or use WASD / Q-E / R-F to take manual control.
Navigation resumes from the current pose after **30 seconds of no human
input**; holding a key or stick prevents the handoff. The settings drawer
holds optional terrain, camera, grid, heading-assist, policy, and evaluation
controls. There is one autonomous demo—not multiple experimental planners.

## Local viewer

For a CPU-only WSL viewer (no CUDA/JAX required):

```bash
curl -fsSL https://raw.githubusercontent.com/mdt733/emu/main/scripts/install_viewer_wsl.sh \
  -o /tmp/emu-viewer-install.sh
bash /tmp/emu-viewer-install.sh --destination "$HOME/emu-viewer-cpu-clean"
```

Open `http://127.0.0.1:8767/`. The installer now checks out `main` by default. Supply
`--branch feature/your-experiment` to explicitly test another branch. It installs viewer dependencies and runs exactly
**twelve** maintained browser contract tests.

For live training inspection, `emu viewer --live --root RUN_DIR --no-open`
continues serving validated checkpoints independently of the public demo.

### Graduated obstacle park

On the **`main`** branch the browser
has eight real-collision challenges, numbered 1–8 in the world,
with Settings → Challenge difficulty:
**Gentle**, **Balanced** (default, mostly 1–2.5 cm), and **Stretch**
(up to 5 cm). The upgraded park retains the optical-flow fix on
heightfields and the fully supported crest top. Read the
[graded course guide](VIEWER_GRADUATED_PARK.md) for exact obstacles,
training-versus-browser limitations and Tailscale launch commands.

The optical-flow/crest repairs and graduated park are now part
of the released baseline. They do not alter the immutable 260M
actor weights or count as proof of an independent MJX exam.

## The current 360-input release

The portable v5032 schema-3 actor and its exact sensor contract,
trained-source SHA, actor hash and checkpoint hash are approved by
`showcase/release_approval.json`. The previous 232-input actor remains
recoverable from Git commit
`2ec9e9db0a816c658131311129ef89ec7a595a12`.

The 360-input release is a measured baseline, not a perfect robot:
read [real supported terrain and heading results](V5032_PINNACLE_RELEASE.md)
and [deferred research](https://github.com/mdt733/emu/issues/17).
The old zero-action fallback is not used by the public builder.

## Publishing safeguards

The approved champion is **not automatically replaced** when Python training
or browser code changes. The GitHub Pages workflow
`.github/workflows/emu-playground.yml` supports manual dispatch and a
specific owner-approved `showcase/deployment_request.json` update on
`main`. Ordinary main merges without that request do not publish.

The workflow verifies the exact champion release approval and provenance,
runs the twelve browser checks, builds static WASM/JS, validates bundled
assets, and deploys GitHub Pages. Check the workflow deployment status
before considering the new public site live. The training checkpoint
(.pkl) is never shipped to browsers.

Release approval and public deployment remain enforced by
`showcase/release_approval.json` and the existing GitHub Pages workflow.
Historical navigation/release design notes remain in Git history. Neither a
successful browser build nor a visually stable walk proves servo safety on hardware.
