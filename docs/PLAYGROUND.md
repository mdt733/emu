# EMU Playground — public browser demo

**[Open the EMU Playground](https://mdt733.github.io/emu/)**

EMU's public MuJoCo/WASM viewer runs the real six-servo policy locally
in the browser. The approved `showcase/champion.emupolicy` is the
**260,112,384-step v5016 Pinnacle actor** (232 observations; six actions),
pinned by `showcase/release_approval.json`. The site can serve a previous
build until the approved Pages workflow has completed.

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

### 360-input research obstacle park

On the **`release/v5032-pinnacle-candidate`** branch the browser
has eight real-collision challenges, numbered 1–8 in the world,
with Settings → Challenge difficulty:
**Gentle**, **Balanced** (default, mostly 1–2.5 cm), and **Stretch**
(up to 5 cm). The upgraded park retains the optical-flow fix on
heightfields and the fully supported crest top. Read the
[graded course guide](VIEWER_GRADUATED_PARK.md) for exact obstacles,
training-versus-browser limitations and Tailscale launch commands.

The public `main` playground remains unchanged. This research
viewer is included with the upcoming candidate source; don't copy
viewer files into an already running source-pinned checkout.

## Planned 360-input replacement

The release validator in this candidate tree can independently accept a
schema-9 360-input, array-only 4×4 terrain-aware actor *if* an explicit
release approval exactly matches its sensor contract, checkpoint/source
provenance and policy Git blob. The currently approved 232-input policy is
unchanged; merely validating a future format never publishes it.
The old zero-action fallback is no longer used by the public builder.


The published 232-input Pinnacle is a **protected historical benchmark**,
not a permanently required second architecture. Once the 360-input unified
learner wins replicated frozen physical evaluations, is verified end-to-end
in the browser and the owner approves, the public release can replace its
policy. Only after that migration should the old 220/232 runtime paths and
their tests be removed from the primary codebase. Git history / a tagged
release should preserve the old policy for rollback and research.
See [release readiness](RELEASE_READINESS.md).

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
