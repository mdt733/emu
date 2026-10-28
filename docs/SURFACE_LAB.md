# EMU physical surface laboratory — 15° viewer frontier

This is a **local browser MuJoCo physical diagnostic**, not a training
curriculum or an independent MJX capability certificate.

The radial park already accepted a 1–20° *maximum difficulty setting*
but did not guarantee an exact incline anywhere. The dedicated Surface
Lab previously stopped at **12°**. On
`feature/v5027-viewer-frontier` the same real rotated support boxes now
allow **0.5–20°** in each of four directions, with quick buttons for
5°, 10°, and 15°. The step lab remains 0.1–6.0 cm (quick 2/3 cm steps).
No modifications to actor inputs, weights, training terrain, PPO,
teacher, or protected `main`.

## Run the local viewer with the actual v5027 policy

The completed GPU campaign in `~/emu-v5027` remains frozen at source
`5dec60c`. Use a separate clean viewer checkout from this branch.
A verified exported A/B/C policy is already in the campaign root. If
not, first verify its existence or export it from the **old source**.

```bash
DEV="$HOME/emu-v5027-viewer"
git clone --single-branch --branch feature/v5027-viewer-frontier \
    https://github.com/mdt733/emu.git "$DEV"
cd "$DEV"
source "$HOME/emu-graduation/.venv/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN="$HOME/robot_lab/emu_runs/v5027_coherent_retention_s735_r2"
POLICY="$RUN/physical_policies/B_coherent_direct.emupolicy"
test -f "$POLICY"
# Auto-builds the local Vite/WASM browser bundle using npm, if missing.
python -m emu.cli viewer "$POLICY" --port 8767 --no-open
```

Open <http://127.0.0.1:8767/>. In **FRONT DEPTH · 4×4** select
Ground: `Uphill · continuous plane` or `Downhill`, `Left camber`,
`Right camber`, then choose 5°, 10°, 15° with quick buttons or set
any half-degree between 0.5–20° using the slider.

The Quick challenge button automatically switches into the physical
Surface Lab while preserving a selected downhill/camber or step-down
direction. Moving the sliders also rebuilds real contact geometry
when the lab is active. Rebuilding resets the robot to its initial
position; it does not edit or retrain the policy.

To test 3 cm steps, choose Step up / Step down and 3 cm; keep a modest
forward command around 0.15m/s initially, then try different speeds.
The actual robot-relative uphill arrow is available in telemetry.
Disable autonomous navigation for manual tests and try heading assist
both ON/OFF. Use camera follow/chase or raw view as preferred.

**Interpretation:** `slope≤15°` and `step≤3.1cm` in overnight logs are
training frontiers. It is entirely possible to remain standing on a
15° incline while drifting downhill instead of following commands.
The viewer is a great way to *see* the behavior. For reproducible
passing scores, use the [read-only frozen frontier examiner](V5027_OVERNIGHT_FINDINGS.md#new-read-only-high-frontier-physical-examiner-results-audit-branch).

The public Pages site still uses the approved 232-input Pinnacle and
is **not** auto-deployed by this feature. No main merge or champion
promotion is implied.
