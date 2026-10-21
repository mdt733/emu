# EMU v5020 — independent LIVE 312-input browser spectator

**Do not modify the running** `~/emu-v5020` **checkout.**
The 4096-world GPU campaign has its own source-locked 380M lineage.
The browser spectator is a separate, read-only checkout, with its own
assets, and uses the original Python virtualenv without reinstalling it.

## Start alongside active training

In a **second WSL terminal**, not the one running the campaign:

```bash
git clone --single-branch --branch feature/v5020-312-live-viewer \
  https://github.com/mdt733/emu.git ~/emu-viewer-312
cd ~/emu-viewer-312
bash scripts/run_v5020_live_viewer.sh \
  ~/robot_lab/emu_runs/v5020_lab_4096
```

Open [http://localhost:8765/](http://localhost:8765/) on Windows.

The first startup builds the standalone browser assets from the locked npm
dependencies in the **viewer-only** checkout (requires existing Node 24).
Later startups reuse them. This process does **not** import JAX for training,
alter the GPU run, or upload any actor.

The original `~/emu-v5020/.venv/bin/python` is reused for read-only
checkpoint export, so no second `pip install` is required. Override
`EMU_V5020_PYTHON` if the training venv lives elsewhere, or
`EMU_V5020_VIEWER_PORT` if 8765 is in use.

## What is genuinely live?

`emu viewer --live` polls completed atomic checkpoints in all campaign
branches, selects the newest by modification time, validates it and exports
schema-9 browser-native actor weights and observation normalization. It
can display the active foundation now, and later branch runs as they
publish newer checkpoints.

The browser locally runs **one representative robot in MuJoCo WASM**,
not all 4096 MJX training worlds. Each new accepted checkpoint hot-swaps
its full 312-input neural actor, resets the demo to avoid mixing runs,
and shows the policy step, branch and revision. The 312 channels are
58 historic delayed tripod ToF/gyro/optical-flow measurements plus
10 sparse right/left V terrain heights and 10 corresponding confidences,
with four frames of actor history. The V memory updates independently
of whether its optional diagnostic panel is open. Closing the panel or
toggling its freeze control never freezes the **neural** inputs.

There are still limitations. The viewer reconstructs the raw ToF/flow/V
sensor readings locally in JavaScript, *approximately* following the
physical model, not by replaying actual GPU trajectories. Scene terrain
and course shapes are selectable viewer demonstrations, not exact
snapshots of every per-world MJX obstacle that a policy trained on.
Browser success and position traces are demonstrations, **not independent
physical performance grades**. For that use the 20mm true-contact held-out
MJX exams and `results.md` in the research campaign.

The browser's default autonomous waypoint mode can command a newborn
policy more aggressively than it has learned; switch autonomous off
for a stationary test, or use the joysticks. A checkpoint can be loaded
only after the trainer publishes it; until then the viewer shows a
waiting state. Viewer process is independent: Ctrl+C there does not
cancel GPU training.

This sidecar branch is experimental and intentionally does **not** merge
to `feature/v5020-coldstart-terrain`, `main`, or the protected public
showcase. Review in the browser before promoting.
