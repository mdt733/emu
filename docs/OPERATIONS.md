# EMU operations — install, train, inspect, recover

Use this guide from **WSL2 Ubuntu** with code and data on Linux ext4.
Commands run in the foreground. Git source checkouts live under `~/emu*`;
large checkpoints and measured results stay under `~/emu-data/`.

## Install and basic checks

```bash
git clone https://github.com/mdt733/emu.git ~/emu
cd ~/emu
bash scripts/setup_wsl.sh --install-node --cuda13 --check
source .venv/bin/activate
bash scripts/check.sh --full
```

This installer uses Python ≥3.13, Node 24 and a local `.venv`.
It does not start training, reset Git or overwrite old runs. The dedicated
CPU-only viewer is `bash scripts/setup_wsl.sh --viewer-only --install-node`.

Hardware/GPU acceptance is a separate gate:

```bash
source ~/emu/.venv/bin/activate
cd ~/emu
emu doctor --require-gpu
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --prisms
```

These are validation operations, not a license to change an existing
source-pinned training run. A 20M cold gait smoke and matched exams are
necessary before considering major policy changes for promotion.

## Experiment C — cold 500M campaign

The new C campaign (unpromoted, draft [PR #84](https://github.com/mdt733/emu/pull/84))
**starts from random network weights, not 120M/180M/420M**. The
original tested training revision is `69c9d6a959147d5f02ebf2a024a28de00362a6b4`.
It is the experiment's own immutable source; this maintenance branch
is NOT an interchangeable resume source.

```bash
cd ~/emu
git fetch origin experiment/h40-cold-dual-lip-continuation
git worktree add --detach ~/emu-cold-duallip \
  69c9d6a959147d5f02ebf2a024a28de00362a6b4
cd ~/emu-cold-duallip
source ~/emu/.venv/bin/activate
bash scripts/run_h40_cold_duallip_500m_wsl.sh
```

Same-run recovery, **only inside that same pinned worktree**:

```bash
cd ~/emu-cold-duallip
source ~/emu/.venv/bin/activate
bash scripts/run_h40_cold_duallip_500m_wsl.sh --resume
```

The launcher refuses a nonempty cold target, and accepts `--resume auto`
only for an explicit restart. The source and checkpoint/config identity
are validated before PPO begins. Original run path:
`~/emu-data/runs/h40-cold-duallip-500m-s735`.

The 500M target, seed 735 and 2048 environments are **requested steps**,
not an observed result. Foundation starts with 20M gait steps; stairs
unlock after a further competency gate and stage schedule. Each completed
C Academy course requires two signed lips and controlled continuation;
none of those training events is by itself an independent frozen exam.

## Viewer and sensor lab

Build and view the current checked-in H40 champion:

```bash
cd ~/emu
source .venv/bin/activate
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy --port 8765 --no-open
```

Open http://127.0.0.1:8765/ . Select **WORLD** to switch between
unrestricted Infinite Meadow and physical Commands/Slopes/Academy
UP/DOWN/Resilience/Transitions/Control drills. Inspector displays raw 4×4
depth data and inference heartbeat. `--live --root PATH` selects
the newest validated checkpoint from a specific run for observing training.
The laboratory **without a policy** is `python -m emu.cli viewer
--sensor-lab --no-open`; it demonstrates geometry, not learned behavior.

The viewer is CPU/WASM MuJoCo on 225 heightfields. It does not certify
the eight-prism MJX trainer's contact or step skills.

## Examine and export

```bash
# Read-only report from one saved run:
python -m emu.cli report --root ~/emu-data/runs/h40-cold-duallip-500m-s735

# Read-only genuine-prism frozen actor exam; use a NEW output directory:
python -m emu.cli prism-exam \
  --policy ~/emu-data/actors/h40-paired-420m.emupolicy \
  --output ~/emu-data/exams/b-up-smoke-new \
  --device cuda --worlds 32 --mode step --lip-cm 5

# Export only after a checkpoint is validated:
python -m emu.cli export-policy \
  ~/emu-data/runs/h40-cold-duallip-500m-s735/checkpoints/final.pkl \
  ~/emu-data/actors/h40-cold-duallip-500m.emupolicy
```

For a larger **ten-capability GPU matrix** (command, slope and signed
stair bins), use the one current retained frozen examiner instead of the
retired versioned script:

```bash
python -m scripts.capability_matrix_exam \
  --policy ~/emu-data/actors/h40-paired-420m.emupolicy \
  --output ~/emu-data/exams/b-capability-matrix-new \
  --device cuda --worlds 2048 --horizon 350 --seed 1701
```

This examiner is read-only; moving it to a role-based module name
changes its recorded **exam-source fingerprint** and is not equivalent
to old version-pinned results without source verification.

Neither `prism-exam` nor the browser changes policy weights. Do not
reuse a completed exam output directory. The exact `final.pkl` exists
only **after** the campaign reaches its final target; interim viewing
may use the source-pinned `--live` option.

## Safety and promotion rules

1. Keep `main` champion, `showcase/champion.emupolicy`, the original
   private `final.pkl`, and its Pages release approval unchanged until
   an explicit, evidenced promotion decision.
2. Never edit, reset, change SHA, merge over or clean inside a currently
   running source-pinned worktree; create an independent maintenance worktree.
3. An exported actor must match the physical H40 XML, sensor contract,
   observation history, and source lineage used to train it. A copy of an
   `.emupolicy` is not a complete recovery checkpoint.
4. Keep one learner and one physics owner. Tests are at most **24 core
   Python cases**, plus real CPU/MJX prism probes, GPU PPO acceptance
   and browser checks; do not accumulate separate test harnesses.
5. Public GitHub Pages is **manual**. The workflow
   `.github/workflows/emu-playground.yml` requires an explicit,
   checkpoint-backed `showcase/release_approval.json`; merging a PR
   alone does not publish a new policy.
6. Experimental sources and full old investigations are recoverable from
   Git and PRs. Keep only current operations in the working tree.

See [Architecture](ARCHITECTURE.md) and [Validation](VALIDATION.md).
