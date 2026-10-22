# EMU v5021 — Considered Walker (first runnable cold-start pass)

**Branch:** `feature/v5021-considered-walker`; never merge to `main`
without explicit approval.

This is a REAL cold start. No .pkl parent from v5020, v5015, or a
pinnacle. Fresh PPO model, optimizer, observation normalizer, terrain
rating, and curriculum at step zero. Entire 312-input actor sensor model
is active from the first environment step.

## Run on WSL 2 + RTX 4060 Ti

Keep previous source checkout and its policies untouched.

```bash
cd ~/emu-v5020
git fetch origin feature/v5021-considered-walker
git worktree add -b feature/v5021-considered-walker \
  ~/emu-v5021 origin/feature/v5021-considered-walker

cd ~/emu-v5021
bash scripts/run_v5021_considered.sh --plan
bash scripts/run_v5021_considered.sh --run
```

The launcher uses the existing `~/emu-v5020/.venv` when no local venv
exists. It uses `PYTHONPATH` to load the NEW checkout's `emu` source.
2048 envs, seed733, cumulative 10M -> 25M -> 50M -> 100M. Re-running
`--run` on an interrupted stage validates and resumes the same-source
checkpoint. For an intentionally shorter run,
`bash scripts/run_v5021_considered.sh --run --stop-after s10`.
Source changes *after a campaign starts* invalidate subsequent resume.
Only ever run from a clean committed `feature/v5021-considered-walker`
worktree.

Watch the training in a **separate detached viewer worktree**. Never
rebuild tracked browser assets inside the source-pinned trainer checkout:

~~~bash
git -C ~/emu-v5021 worktree add --detach ~/emu-v5021-viewer HEAD
cd ~/emu-v5021-viewer
source ~/emu-v5020/.venv/bin/activate
PYTHONPATH="$PWD" python -m emu.cli viewer-build
ROOT=$(ls -dt "$HOME"/robot_lab/emu_runs/v5021_considered_* | head -n1)
PYTHONPATH="$PWD" python -m emu.cli viewer --live --root "$ROOT" --port 8766
~~~

Then open http://127.0.0.1:8766/ . The viewer follows SAVED checkpoints,
not the GPU training frames.
## Implemented now

- PPO train uses 312-input V-shaped sensor memory; older schema1/2 remain.
- `--profile considered` is an explicit nonlegacy configuration.
- Adaptive physical step UP with varied X/Y/yaw, a zero-mm rated birth
  and evidence-based rated promotion/retreat.
- Geometry validation interval currently 0–150mm (fixed existing 150mm
  MJX kinematic platform); 6mm is NOT an endpoint. Beyond 150mm needs
  correctly rescaled collidable geometry and a tested new contract.
- Active evidence-driven speed, reverse, lateral, yaw, and acceleration
  governor. Command exploration safety bound is 12m/s in the forward
  dimension, not a trained speed target; old 0.40/1.20m/s lesson caps gone.
  This high numerical limit DOES NOT imply the physical robot can reach it.
- Considered motor payment weights overload quadratically and charges
  shaft power, network target jitter, and contact-weighted downward foot
  speed; actual height tracking and command following still pay more for
  real forward motion. The 0.638Nm continuous versus 1.913Nm stall torque
  is retained in physical telemetry. These penalties are physical PROXIES,
  NOT verified battery J/m, current or impact force measurements.
- Flat-ground rehearsal continues across task regimes.
- Fixed-eval checkpoints and per-stage evidence report/ZIP are generated.
- Existing historical rewards are identical when `considered_reward=False`.
- New test suite runs before PPO. **CUDA runtime is not yet validated on
  user's WSL; the preflight must succeed before an overnight run starts.**

## Remaining research work (NOT implemented by this launcher)

- Actual **live** per-world slopes/ramps simultaneously with step-up
  platforms. Existing v5020 terrain coupling rules still reject them.
- Sensor-dependence/causal test with the exact same trained actor's
  terrain memory preserved versus intervened upon in matched conditions.
  Unmasked 312-inputs proves connection, NOT useful sensor awareness.
- True per-motor electrical/thermal model, contact impulse-based shock,
  and phase cadence controlled directly by the policy.
- Expansion above 150mm through tested dynamically rescaled platform
  geometry and corresponding checkpoint-contract migration.
- Independent held-out gait score including achievement at several speeds,
  jerk/vertical impact, mechanical work J/net forward metre, step
  completion, sensor-on/off delta, and retention.
- Thorough GPU acceptance on the user's exact WSL/CUDA environment.

Before promoting any checkpoint to 'best', check BOTH observed command
tracking and energy/torque, not reward alone. Keep v5020/v5015 baseline
policies immutable as external comparisons.
