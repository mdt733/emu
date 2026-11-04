# v5046 Overnight Solver Experiment — 4 November 2026

This is an **experiment**, not a new training recipe or a replacement for
main. It leaves all checkpoints and `emu/robot.xml` unchanged. It exercises
the exact v5046 four-tile, eight-prism, six-servo, 360-input PPO path with one
CUDA process at a time.

## Launch — one foreground command

Run from the local PR #35 checkout after a clean fast-forward:

```bash
cd ~/emu-v5046-four-tiles
git status --short
git pull --ff-only
source ~/emu-graduation/.venv/bin/activate

PYTHONPATH="$PWD" python -u scripts/overnight_solver_campaign.py \
  --preset overnight \
  --output ~/emu_gpu_audit/v5046_solver_overnight \
  2>&1 | tee ~/emu_v5046_overnight_console.log
```

**No tmux, no cron, no automatic merge, no checkpoints.** Leave this foreground
shell open and keep the GPU otherwise idle. The campaign uses the interpreter
from the activated virtualenv; it verifies the current source SHA and rejects
dirty tracked source or execution on main. A failure cannot silently change
a normal production training configuration.

## Experiment matrix

Two paired seeds: 735 and 761. **17 configurations per seed = 34 GPU arms**,
each in a fresh Python process, using 2,048 environments and 8 total
full 32-frame PPO updates (first compiles; 7 measured). Each subprocess exits
before the next begins to avoid 8GB VRAM contention and compiler memory
accumulation. Nominal 34 × 8 × 65,536 = **17,825,792 world-steps** across the
full set, including compile/warmup updates.

Baseline and 4/4 are repeated at the end of each seed to detect timing
drift. Arms:

| Group | Variants |
|---|---|
| Reference | unmodified solver, solver 4/4, solver 8/8 |
| Iteration vs linesearch | 2/2, 4/8, 8/4, 4/default, default/4, 2/4, 4/2, 1/1 |
| Reset cost | 4/4 with reset-forward batch 192, 256 or 0 (full-width fallback) |
| Contact topology cost | 4/4 with torso collider restored |
| Drift controls | repeat unmodified default, repeat 4/4 |

All solver variants create benchmark-only temporary XML assets. A solver
iteration count and linesearch count set together **do not establish which
one caused a speed difference**; their isolated runs investigate that.
All variants are compared at the same seed and same compiled topology except
the explicit torso-contact arm.

## What is checked

1. Before the sweep: **CPU** four-world eight-prism geometry/contact/40cm
   sidewall, recycling and masked reset physical acceptance (same backend
   as the passing GitHub CI). The optional detailed four-world gate gave an
   unexplained assertion on the operator's GPU; this is explicitly not
   treated as GPU parity evidence.
2. Each GPU arm: **CUDA** terrain geometry, real 2,048-world reset and
   recycling, two-frame physical step, 360-channel observations and a
   finite real actor update. A CPU preflight pass never bypasses these.
3. Each arm's last 32-frame PPO rollout: after timing, save mean fall, support,
   bad-contact, slip, virtual-strike, command/actual speed, energy and step
   indicators, plus padding/constraint sizes and reset-mask occupancy.
4. Each arm's synchronized **steady full PPO updates**, excluding compilation.
   Report actual solver limits from the compiled MuJoCo model.

**This is NOT a trained-gait validation.** A newly initialised policy's contact
and fall statistics are smoke/physics signals, not proof that a policy can
walk the terrain. Only a later frozen-policy controlled physical exam should
allow moving a faster solver into `robot.xml` or a real training campaign.

## Outputs and resumption

Files in `~/emu_gpu_audit/v5046_solver_overnight/`:

- `state.json` — atomic after each arm; includes exact Git SHA and plan.
- `logs/s735_solver4_ls4.log`, etc. — raw full captured output.
- `results.csv` — one row per attempted arm, physical diagnostics included.
- `SUMMARY.md` — continuously refreshed ranking, same-seed speed ratios.
- `preflight.log` — physical preflight output.
- `overnight_results.zip` — automatically packages all completed results and logs for a single upload after the last arm.

A later rerun of the **same command in the same checkout** skips successful
arms and resumes failures/missing arms. If only the preflight ran and failed
before any GPU arm, a new source SHA can safely restart it, archiving the
failed preflight log and recording both SHAs. Once even one GPU arm has
run, source SHA, seed selection, number of updates and Python executable must
match to resume. Choose a new `--output` directory for a new campaign.

If the preflight fails, it does not launch the expensive GPU sweep.
Three consecutive failed GPU arms stop the run rather than accumulating
hours of repeated tracebacks.

## Smaller preflight / sanity options

```bash
# Safe even without CUDA: show the exact commands, no GPU access.
python scripts/overnight_solver_campaign.py --preset quick --dry-run

# Optional quick 3-arm GPU check (new output folder).
python scripts/overnight_solver_campaign.py --preset quick \
  --output ~/emu_gpu_audit/v5046_solver_quick
```

## Decision after results

First reject any arm with nonfinite outputs, incomplete contact or physical
smoke failure, or major fall/contact/slip regression relative to matched
seed controls. Rank remaining candidate solver settings by repeatable
throughput, prefer 8/8 or 4/4 only after held-out frozen-policy exams
confirm impact dynamics, wall contact, sliding and signed step mastery.
Compare 192 versus 128 reset capacity independently. The four-hfield
alternative is **not in this campaign**: it first needs correct moving
hfield contact physics and identical 0/4/15/40cm geometry proofs, rather
than accidentally mixing a new collision engine into the solver sweep.
