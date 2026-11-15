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

## Current experiment — 50mm foot tutor and recovered-chassis escrow

This source is a **research continuation of draft PR #88**, not an update to
the current H40 champion. The 50mm uphill Academy entry was deliberate:
small steps often reward unchanged walking, not obstacle negotiation.
The new tutor rewards bounded actual-prism foot lift and clearance,
then confirmed support, but holds the chassis-progress reward until both
feet independently supported the first upper tile and commanded walking
recovered for 20 consecutive control frames with safe pitch angular speed.
Only the recoverable body position can release escrow, never a falling lunge.
All tutor calculations use privileged teacher-side physics, not actor inputs.
The original 20mm DOWN school remains intact as an internal control.

Use an **independent, clean Git worktree** at the tested research commit.
Do not change source or target config while a run is in progress.

~~~bash
cd ~/emu
git fetch origin research/h40-stair-tutor-escrow
git worktree add --detach ~/emu-stair-tutor origin/research/h40-stair-tutor-escrow
cd ~/emu-stair-tutor
source ~/emu/.venv/bin/activate
# Expected parent: ~/emu-data/runs/h40-d-coach-lab-s735/foundation-60m/checkpoints/final.pkl
bash scripts/run_h40_stair_tutor_wsl.sh preflight
bash scripts/run_h40_stair_tutor_wsl.sh smoke
# The smoke includes 128-world CUDA PPO AND a frozen 50mm real-prism
# Academy run with the escrow reward code enabled.
# Shared 60M→80M walking bridge (20M trained ONCE); no duplicate warmup.
bash scripts/run_h40_stair_tutor_wsl.sh bridge
# Two matched 80M→100M, 20M tutor forks from identical checkpoint.
bash scripts/run_h40_stair_tutor_wsl.sh both
# Frozen-policy two-seed true-prism exams:
bash scripts/run_h40_stair_tutor_wsl.sh exam
bash scripts/run_h40_stair_tutor_wsl.sh report
~~~

No billion-step campaign is authorised by this experiment. **Stop** if
uphill Academy exposure is negligible (automated legacy-arm gate) or if the escrow arm fails to improve
physical pre-root upper-foot contact and controlled recovery. Decide whether
to extend from 100M to 120M only with independent physical data, and repeat
with a different training seed before any causal claim. Running this script
without an explicit mode prints help and trains nothing. Running preflight
never launches PPO. The launch checks parent SHA-256 and source identity;
same-source extensions require the run's own validated checkpoint.

Independent exams include 50mm UP, mixed UP/DOWN and 20mm DOWN programmes
at seeds 1701 and 2701 for both frozen actors, PLUS an exact-height 50mm
UP/DOWN exam paired by per-world index against a zero-height seam control.
These use eight authentic MJX prisms, not a viewer or teacher success flag.
Upload the two-arm tutor report plus exams for review before another budget.

**Historical studies:** The 500M C launcher from source
69c9d6a959147d5f02ebf2a024a28de00362a6b4 and the completed 16-arm D
launcher from draft PR #88 belong in Git history. They are not maintained
as active launch scripts. Use their original source for historic reruns;
never silently resume a campaign under the tutor source.

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
