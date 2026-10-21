# EMU v5020 — one-command fresh foundation + five-way research campaign

**Review branch**: `feature/v5020-full-campaign`. Nothing is merged into
`main`, the old 4096-world `v5020_lab_4096`, the source-pinned
`~/emu-v5020` checkout, or the public showcase.

This is a thin wrapper around the **existing** schema-9 PPO training engine,
validated shared foundation phase, and the newer five-arm winner-refork lab.
No new learning algorithm is hidden in the wrapper.

## Actual study, from zero weights

1. **Foundation**: new 312-input cold start to 20M steps (PPO batch boundary
   may finish slightly beyond exactly 20,000,000). Gentle physical contact
   slopes, transitions, walking and recovery. This is a new root, not the
   previous 20,054,016 checkpoint.
2. **walk40**: general gait, turns, recovery, efficiency and a level-ground
   continuously earned step-height specialist, each initialized from the
   **same exact** foundation. Each contender gets a 2M collapse probe before
   receiving the remaining training toward absolute 40M steps.
3. **walk60**: all five fork the single eligible 40M winner, test and
   competitively select again.
4. **walk80**: repeat the same five-way generation, publish the winner and
   evidence. This is the normal automatic stopping point.
5. **terrain100**: an optional extra generation, gated on the 80M champion's
   verified walking competence. Earned step frontier may grow up to the
   existing 150mm physical safety ceiling, *not* a target claim.

Every generation uses matched frozen general locomotion, command, recovery,
contact-slope and **physically collidable** 5/10/20/40/80/150mm step
examinations. It promotes only a qualifying winner; it never makes a
failing policy look successful. The parent is a baseline too.

An isolated three-world JAX/MJX examination is run *before* starting the
expensive first foundation. It must compile and pass. It is a plumbing test,
not proof of gait quality. The check lives in its own subprocess so JAX
doesn't retain compiled preflight state while the 4096-world trainer starts.

## Clean installation and plan (WSL)

Use a **new checkout** to protect the old source-pinned training experiment:

```bash
git clone --single-branch --branch feature/v5020-full-campaign \
  https://github.com/mdt733/emu.git ~/emu-v5020-full
cd ~/emu-v5020-full
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate

git status --short           # must be empty
emu campaign-full5020        # plan only, zero training and no run-dir writes
emu campaign-full5020 --preflight
```

Then start the **entire** campaign, no manual branching:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
emu campaign-full5020 --run \
  --envs 4096 --screen-envs 32 --screen-horizon 300 \
  --seed 733 --stop-after walk80 \
  --root ~/robot_lab/emu_runs/v5020_full_4096
```

The exact same command resumes an interrupted campaign, revalidates source
identity and prior checkpoints, and continues. Preflight is repeated at each
launch for safety; after a pass from the SAME committed source, add
`--skip-preflight` if desired. Do **not** change `--envs`, seed, active
Git revision, source branch, or root midway. Keep JAX on CUDA, only one
GPU trainer process, and do not update the checkout while it runs.

Stop after `walk40` for an intentional mid-study review, or use
`--stop-after terrain100` to include the fourth, harder generation
**only if the 80M incumbent qualifies**. Changing `--stop-after`
doesn't alter an already computed phase's identity.

## Results / read-only status

```bash
emu campaign-full5020 --status \
  --root ~/robot_lab/emu_runs/v5020_full_4096

cat ~/robot_lab/emu_runs/v5020_full_4096/results.md
cat ~/robot_lab/emu_runs/v5020_full_4096/refork/results.md
```

Output structure:

```text
v5020_full_4096/
  lab_context.json
  foundation20/
    checkpoints/final.pkl      # preserved complete parent
    metrics.csv / fixed_eval.csv / console.log
  results.md / results_data.json / results_bundle.zip
  refork/
    refork_context.json
    walk40/
      general,turns,recovery,efficiency,steps/
        probe/                # optional 2M qualified learner
        finish/               # training to absolute 40M
      scoreboard.json / scoreboard.md / selection.json
    walk60/...
    walk80/...
    terrain100/...           # optional
    results.md / results_data.json / results_bundle.zip
```

Refork reports are regenerated after every generation or a no-winner
stoppage, and once more when the wrapper finishes. Reports are
**read-only and nonfatal**: they never control PPO learning or alter
selection decisions. The ZIP includes existing metrics, configs,
scorecards, provenance and probe verdicts but **not** huge neural
checkpoints or console logs. The originally successful 20M run remains
in its old directory and is neither deleted nor overwritten.

The newer `emu campaign-refork` is retained for experiments which
*intentionally* reuse an already validated 12–22M foundation;
the historical four-arm `emu campaign5020` is also retained unchanged.

## Simultaneous viewer

The separate `feature/v5020-312-live-viewer` spectator checkout can
follow `~/robot_lab/emu_runs/v5020_full_4096` and hot-swap its newest
completed schema-9 checkpoint. The CPU/WASM preview is one robot,
not a replay of all 4096 training worlds. Don't modify the active
training checkout or publish unfinished brains to the public site.

## Acceptance / source restrictions

Tests cover source lock, null foundation-parent reports, legacy
checkpoint mismatch, stage transition/selection, stable schema-9 actor,
terrain JIT, and read-only report packing. GitHub CI is the gate for
this integration branch; the full ~320M aggregate training workload
is **not** reproduced in CI or this repository.

No direct commits or merges to `main` were performed.
