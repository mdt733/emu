# EMU v5058 — rated C018 motor force is a hard physical limit

## Why this is a new physical experiment

The earlier v5057 40cm 200M run had MuJoCo position-actuator
`forcerange="-2.941995 2.941995"`: the datasheet **stall** output,
not its **rated** output. The reported 57.7% over-rated actuator
exposure was physically allowed by that wrong ceiling. The old
200M performance is **NOT evidence for a rated-torque-feasible robot**.

Every six-axis 12V C018 motor now has **explicit** MuJoCo/MJX
`forcelimited="true"` and `forcerange="-0.980665 0.980665"`
(10 kgf.cm = 0.980665 N.m), including all eight morphological builds.
The generic hardware domain randomizer clamps its per-world
`actuator_forcerange` to the SAME nominal rated bound, rather than
allowing a +15% torque variation to defeat the hardware ceiling.
The environment refuses to initialize if any actuator disables the
limit or advertises an out-of-rated limit.

This is an **actual force ceiling in the simulation**, not just a
reward penalty, scaled action, logging warning or after-the-fact
network output clip. Individual passive tendon/hip springs and
gravity are separate from **motor** output and may create joint loads
greater than a single motor's rating. Those loads are not delivered
by the actuator.

The speed-dependent electrical torque-current envelope, servo
temperature, thermal duty, real load-to-speed curve, bus voltage sag,
and current draw remain uncalibrated. In particular the existing
0.624 N.m.s/rad passive joint damping is an unverified speed proxy
that creates braking torque even when the motor is unpowered.
Do not claim hardware realism just because the actuator hard ceiling
now holds. The old 200M model and this model are intentionally
**different physics**.

## Acceptance

Existing core physical test (test count unchanged) checks **all eight**
compiled MuJoCo models, both positive and negative maximum servo
commands, actuator generalized forces, and a batched MJX domain model
at maximum randomization radius. The CUDA PPO smoke and independent
frozen matrix are still operator-side acceptance; CI is not GPU evidence.

## WSL: clean independent 40cm rated-torque cold start

```bash
cd ~/emu &&
git fetch origin \
  '+refs/heads/research/v5058-c018-rated-torque-hardcap:refs/remotes/origin/research/v5058-c018-rated-torque-hardcap' &&
git worktree add --detach ~/emu-v5058-rated \
  refs/remotes/origin/research/v5058-c018-rated-torque-hardcap &&
cd ~/emu-v5058-rated &&
test -f emu/physical_variants.py &&
echo "SOURCE: $(git rev-parse HEAD)"
```

```bash
(
  set -e
  cd ~/emu-v5058-rated
  source ~/emu/.venv/bin/activate
  export PYTHONPATH="$PWD"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export EMU_PHYSICAL_PROFILE=h40

  # Tests use real MuJoCo actuator output and maximum control range.
  python -m pytest -q tests/test_core_physics.py
  python -m emu.cli doctor --no-require-gpu --no-mjx-smoke

  # Real 2,048-world GPU and full PPO update before any 20M investment.
  python scripts/smoke_gpu_carpet.py \
    --device cuda --worlds 2048 --frames 2 \
    --ppo --full-update --benchmark-updates 3 \
    --prisms --solver-iterations 2 --solver-ls-iterations 4

  # ONE independent fresh 20M cold start; never reuse the old source.
  python scripts/v5057_physical_tournament.py \
    --root "$HOME/emu-data/runs/v5058-rated-40cm" \
    --profiles h40 --seeds 735 --execute
)
```

The original `v5057-h40-athletic-s735-200m` and h40 20M parent remain
immutable **historical, over-rated-torque** experiments. Their
checkpoints cannot be silently resumed in v5058: model SHA and source
SHA checks reject mismatches. A future deliberate transfer learning
experiment would need explicitly designed, **separately validated**
weights-only transplantation. The recommended default is the new
rated-torque cold start to establish a defensible baseline.

An acceptable rated run must show
`servo_over_rated_frac == 0` and
`servo_peak_torque_limit_frac <= 1` (up to floating-point epsilon)
across the rollout, not merely at a single checkpoint.
