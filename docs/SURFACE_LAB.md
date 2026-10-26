# Surface laboratory — test a LIVE v5023 policy without stopping training

This branch, `feature/v5023-surface-lab`, was cut from the exact
**d6f3622** cold campaign revision. **Only viewer files and documentation
have changed.** Do NOT `git pull` the source-pinned training checkout
while any training subprocess is active.

The lab is a manual physical comparison, **not** a frozen MJX proficiency
exam. It uses **actual browser MuJoCo contacts** and the exported
360-input actor with its saved normalizer and physical 4×4 sensor model.

## One-time WSL viewer setup (use a SECOND terminal)

```bash
(
set -euo pipefail
VIEW="$HOME/emu-surface-lab"
TRAIN="$HOME/emu-capability-growth"
RUN="$HOME/robot_lab/emu_runs/overnight_teaching_v5023_audited/growth_balanced_s733"

test -f "$TRAIN/.venv/bin/activate" || {
  echo "Expected the running checkout's Python environment at $TRAIN/.venv"; exit 1;
}
test -d "$RUN/checkpoints" || {
  echo "No training checkpoints in $RUN yet"; exit 1;
}
if ! git -C "$VIEW" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -e "$VIEW" ]; then
    echo "STOP: $VIEW exists but is not a Git checkout"; exit 1;
  fi
  git clone --single-branch --branch feature/v5023-surface-lab \
    https://github.com/mdt733/emu.git "$VIEW"
fi
cd "$VIEW"
test -z "$(git status --porcelain)" || {
  echo "STOP: viewer checkout contains uncommitted changes"; exit 1;
}
git switch feature/v5023-surface-lab
git pull --ff-only
source "$TRAIN/.venv/bin/activate"
python -c 'import emu; print("Viewer code:",emu.__file__)'
command -v npm >/dev/null || {
  echo "Node/npm needed in WSL. See scripts/install_viewer_wsl.sh"; exit 1;
}
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m emu.cli viewer \
  --live --root "$RUN" --port 8766 --no-open
)
```

Leave this second-terminal process open, then browse to

**http://127.0.0.1:8766/**

The read-only watcher loads the newest **validated** available
`latest.pkl`, `final.pkl` or rotating snapshot. It never alters,
deletes, writes or resumes the trainer. Loading newer policies may briefly
use some CPU and RAM but not CUDA; the browser simulates locally using
MuJoCo/WASM.

You can also load any previously exported `.emupolicy` via the viewer's
Load button, or manually export one (from THIS viewer checkout):

```bash
JAX_PLATFORMS=cpu python -m emu.cli export-policy \
  "$RUN/checkpoints/latest.pkl" \
  "$HOME/emu_surface_75m.emupolicy"
```

The training process must remain on its **original Git commit** for resume.

## How to use the diagnostics

Open the **FRONT DEPTH · 4×4** panel, choose a ground option under
**v5023 surface laboratory**. The continuous uphill/downhill/left/right
camber cases each use one rotated, flat-top box replacing the flat
floor, with a real geometric top surface and no coincident floor contact.
Use *Slope angle* from **0.5 to 12 degrees**, usually test 1°, 2°, 3°,
5° then 8°; reset and drive the same modest forward command at each
angle. The underfoot slope readout updates for these four cases.

Choose *Custom step up* or *Custom step down* and adjust *Step height*
from **0.1 to 6 cm**. These are discontinuous support surfaces, not
a visual image of a step. Try 0.3, 0.5, 1 and 2 cm.

Controls remain the same two joysticks. Start with `vx≈0.04 m/s`,
zero strafe/yaw, then compare heading assist ON/OFF and training-style
sensor noise ON/OFF. Reset between trials. Browser trajectory,
fall behaviour and requested/actual velocity are diagnostic observations,
not physical-episode certification under the slope curriculum.

## Why we're testing this

Current training slope advancement needs >=8 completed near-target
episodes in **all 4 directions**, each >=65% successful, to raise its
ONE angle by 0.08°. An easy-direction batch below 30% can retreat the
shared angle and clear windows. This can create a slow/stalled frontier
even when the policy really does handle more incline.

If the walker looks comfortable on 3°–5° continuous grades here but
the teacher remains below 1°, inspect frozen MJX direction-conditioned
evidence next and consider changing the NEXT research revision's
evidence/advancement rule. Never patch the in-flight d6f3622 campaign.
