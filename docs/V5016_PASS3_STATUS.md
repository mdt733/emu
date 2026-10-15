# EMU v5016 — Pass 3: matched heading-hold evaluation

**Code status:** evaluation-only optional heading P loop, real MJX rollout traces,
ON/OFF matched comparison and tests. **Not trained as a v5016 policy**.
**Protected:** `main`, 60M champion, all 20M/80M checkpoints, existing trainer
reward/optimizer, browser showcase and policy observation contract.

## Files actually changed

| File | Responsibility |
|---|---|
| `emu/directional.py` | One shared NumPy/JAX `heading_rate_request` law, reused by the existing pure Python controller and new JAX evaluator |
| `emu/ppo.py` | `evaluate_trace(...,heading_kp)` integrates **body gyro-Z relative yaw** and the intended heading from original operator yaw commands. It corrects only the task yaw rate and *current latest observation history frame* before calling the trained actor. Does not change `train_update` or `evaluate_summary`. |
| `emu/gait_exam.py` | Separately records issued-vs-operator yaw request, gyro relative-heading error and correction magnitude. World-fixed course scoring still uses original operator commands, not corrected policy request |
| `emu/directional_exam.py` | `--compare-heading` runs OFF and ON with same initial seeds/checkpoint/commands, writes separate full per-world evidence and a human-readable delta scorecard in a new output directory |
| `emu/cli.py` | Forwards `--compare-heading --heading-kp` without changing old `emu` commands |
| `tests/test_gait_exam5016.py` | Tiny *actual* 2-world MJX smoke for both controller OFF and ON with identical operator commands, plus separate trace-shape/finite checks |
| `tests/test_heading_pair5016.py` | Pure sign/saturation/wrap tests, same-checkpoint/same-protocol pairing, missing-data and overwrite safety |

**Not implemented:** gyro sensor noise/bias/delay injection, adaptive gains,
turn-stop schedules, controller deployment parity tests, physical ramps,
actual 244-input PPO actors, motor-load promotion gates, or new rewards.

## Why controller-on doesn't cheat the grade

All trajectories are evaluated against the **original operator command**.
The controller may request a different yaw rate from the trained PPO actor, but
it cannot rewrite what the user asked or the world-frame course oracle.
The **explicit operator triplet** `(vx,vy,yaw_rate)` is supplied separately
to the JAX evaluator for each matched exam (not inferred from potentially stale
sampled task values). The same explicit commands also populate the task packet
used for the **initial physics reset**, preventing unequal starting yaw velocity.
The regression suite compares both the operator commands and the initial root
velocity vectors across ON/OFF. At every control step the environment task and the last
observation history frame receive the same operator vx/vy and either the operator
yaw-rate (OFF) or controller-corrected yaw-rate (ON), *before* the policy forward
pass. Previous frames remain real history. This first evaluator intentionally uses **constant STEADY commands**,
not command transitions. Rollout input and checkpoint/network shapes remain
the original 220→6; the independent v5016-candidate 244-input contract stays
declared but not runtime-ready until later passes.

The heading controller integrates body-frame gyro-Z with 20ms control ticks.
It is a **relative gyro-based stabiliser**, not drift-free world compass or
accurate true world yaw under arbitrary torso roll. Simulator world yaw appears
ONLY in independent examination metrics. Same initial seeds and episode
conditions make the comparison a controlled on/off intervention.

No new scalar reward and no teacher world heading are added.

## Run on WSL GPU (after CPU CI/real MJX smoke have passed)

Work in the separate `feature/v5016-directional` checkout; do NOT check out
a new commit inside your existing PPO-run worktree.

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-v5016-heading origin/feature/v5016-directional
cd ~/emu-v5016-heading
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate

RUNROOT="$HOME/robot_lab/emu_runs/tiny_fourway_001"
OUT="$HOME/robot_lab/emu_exams/v5016_heading_001"

# First: tiny real-GPU compile and paired stepping
emu directional-exam --run-dir "$RUNROOT/round1/baseline" \
  --output "$OUT/baseline_flat_smoke" --angle 0 \
  --worlds 2 --horizon 40 --settle-frames 5 \
  --compare-heading --heading-kp 0.75 --device cuda

# Then: meaningful small incline and 350-frame uninterrupted course
emu directional-exam --run-dir "$RUNROOT/round1/baseline" \
  --output "$OUT/baseline_up025" --angle 0.25 --direction uphill \
  --worlds 8 --horizon 350 --compare-heading \
  --heading-kp 0.75 --device cuda
```

Report files:
- `$OUT/baseline_up025/heading_comparison.md`: aggregate OFF / ON / ON-minus-OFF, no automatic promotion;
- `heading_comparison.json`: full independent comparison;
- `off/directional_exam.{md,json}` and `on/directional_exam.{md,json}`: individual original-command outcomes, survival, progress, heading error, contact-stride pairs and controller corrections.

Use different output folders for each new slope, direction or policy. If
successful, repeat at **0°, ±0.25°, ±0.5°, ±1°**, using `uphill`, `downhill`,
`cross_left`, `cross_right`; on downhill, the same forward command correctly
means *downhill* travel. The angle here rotates gravity on a flat floor;
it does NOT create a physical sloped mesh. All evaluation results remain
read-only and detached from automatic training selection.

## Acceptance and known limits

1. Pure heading law: shortest-error sign, rate clamping, request passthrough.
2. Real two-world MJX on/off JIT run: same requested `command` arrays, corrected
   yaw bounded to policy's trained envelope, no nonfinite trajectories.
3. Paired report: same checkpoint and tested conditions, no skipped/missing
   evidence silently converted into a good score.
4. Original policy shape (220 inputs, six outputs) and known main training
   workflow unchanged, plus full Python/browser CI.
5. **Still to prove on your GPU:** end-to-end baseline/slope/pitch15/shadow
   differences, heading-hold usefulness, 1° directional competence and
   enough genuine alternating load-bearing contact cycles.

A heading reduction that damages course progress or survival is **not**
a win. A small surviving subset with low heading error is not proof of
an overall better controller. Compare raw per-world results, then widen
seeds and examine realistic IMU bias/latency before considering deployment.
