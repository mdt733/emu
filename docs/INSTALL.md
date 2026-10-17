# Install & operate EMU

EMU has a Python JAX/MJX learning environment and a separate JavaScript MuJoCo/WebAssembly viewer. **GPU training requires a working NVIDIA driver exposed to WSL**; browser viewing does not.

## Fresh Windows / WSL2

From an administrator PowerShell on Windows 11:

```powershell
wsl --list --verbose
wsl --install -d Ubuntu-24.04
wsl --update
```

Do not reinstall or unregister an existing suitable Ubuntu environment. In Ubuntu, check Python 3.13+; Ubuntu 24.04's stock Python may be 3.12. One choice is the third-party deadsnakes PPA (evaluate its security/update policy before using it):

```bash
sudo apt update
sudo apt install -y git curl ca-certificates software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv
python3.13 --version
```

Do not replace `/usr/bin/python3`. Install the Windows NVIDIA driver, **not the Linux graphics driver inside WSL**. Check `nvidia-smi`. See [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Clone, install and verify

```bash
git clone https://github.com/mdt733/emu.git "$HOME/emu"
cd "$HOME/emu"
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --v5015-smoke --domain-smoke
bash scripts/check.sh
emu ladder --stage plan
```

For CPU-only inspection use `bash scripts/setup_wsl.sh --install-node` and `emu doctor --no-require-gpu --no-mjx-smoke`. The installer builds the viewer, not a long run.

**The six-servo policy shape is fixed:** 220 observations → 6 actions. Training checkpoint schema 7 preserves weights, optimiser, normaliser and curriculum. Do not resume across a changed source revision or reuse run roots with different environment counts.



### v5016 read-only physical reality exam (do this BEFORE expensive training)

The protected Pitch15 public policy is a legitimate **frozen actor** and now
works as an examiner input without the historical training PKL. Use a new,
isolated output directory outside Git; this creates reports only:

```bash
cd "$HOME/emu-v5016"
source .venv/bin/activate

OUT="$HOME/robot_lab/emu_exams/pitch15_steady_$(date +%Y%m%d_%H%M%S)"
emu directional-exam --policy showcase/champion.emupolicy \
  --output "$OUT" --device cuda --worlds 4 --horizon 350 \
  --seed 1701 --scenario steady --speed 0.10 --angle 0

# Bounded independent stop/resume and turning checks, different output roots.
for SCENE in rest_start speed_step brake_stop stop_resume reverse turn_stop_left turn_stop_right; do
  OUT="$HOME/robot_lab/emu_exams/pitch15_${SCENE}_$(date +%Y%m%d_%H%M%S)"
  emu directional-exam --policy showcase/champion.emupolicy \
    --output "$OUT" --device cuda --worlds 4 --horizon 350 \
    --seed 1701 --scenario "$SCENE" --speed 0.10 --angle 0
done
```

If the CUDA examiner fails due to VRAM, collect the error and consider
`--worlds 2`; do not silently substitute a CPU pass for GPU acceptance.
Exam markdown reports signed progress, alternating touchdown candidates,
per-servo torque-duty evidence, shaft work per signed metre and maneuver
latency. Inspect the corresponding raw JSON before judging any policy.
These tests never update the 120M trained policy, the showcase or a
running WSL experiment. They do **not** establish electrical consumption,
winding temperature, calibrated ground reaction forces or safe hardware use.

## GPU acceptance before a large campaign

The following is a deliberately small but complete **two-update** fresh-checkpoint test. Use a never-used root outside Git:

```bash
ROOT="$HOME/robot_lab/emu_runs/acceptance_$(date +%Y%m%d_%H%M%S)"
python scripts/smoke_training_lifecycle.py first --root "$ROOT"
python scripts/smoke_training_lifecycle.py continue --root "$ROOT"
python scripts/smoke_training_lifecycle.py verify --root "$ROOT"
```

This proves update, save, warm resume, independent fixed evaluation and actor export, **not walking ability**. Use `nvidia-smi` to watch VRAM. Earlier 8GB RTX 4060 Ti runs succeeded at 2048 worlds, but this is not a promise for other machines.

## Train and inspect

```bash
RUN="$HOME/robot_lab/emu_runs/study_$(date +%Y%m%d_%H%M%S)"
emu ladder --stage foundation --root "$RUN" --seed 733 --envs 2048 --device cuda
emu ladder --stage auto --root "$RUN" --seed 733 --envs 2048 --device cuda --stop-after round1
emu report --root "$RUN"
emu viewer --live --root "$RUN" --no-open
```

The 20M foundation is shared; four arms reach approximately 80M each. Two eligible finalists can reach 100M; a winner can reach 120M. See `emu ladder --stage plan` for the live recipe. Only completed validated checkpoints feed the viewer. Open `http://127.0.0.1:8765/`. The read-only viewer server is **unauthenticated and local-only**; never forward its port publicly.

For one-off deterministic browser policy export:

```bash
emu export-policy /path/to/checkpoints/final.pkl /tmp/my_robot.emupolicy
```

## One continuous run (no experimental branches)

For one 120M-step trajectory, use `emu train` rather than `emu ladder`.
The mature profile starts from scratch (so this does **not** reproduce the
historical 20M-foundation/forked Pitch15 experiment). `--pitch-deg 15` uses
that study's promising pitch command range; 1024 worlds is a more conservative
starting point for an 8 GB GPU, though it still needs real GPU validation.
Use a unique run directory *outside* Git and do not change its source revision
or options during training.

```bash
RUN="$HOME/robot_lab/emu_runs/single120_pitch15"
emu train --run-dir "$RUN" --steps 120000000 --envs 1024 --seed 733 --device cuda --profile mature --pitch-deg 15
```

In a second WSL terminal using the same checkout and environment:

```bash
emu viewer-build
emu viewer --live --root "$HOME/robot_lab/emu_runs/single120_pitch15" --no-open
```

Open http://127.0.0.1:8765/ in Windows. Viewer is read-only,
runs inference on the browser CPU and loads completed checkpoints as they
arrive; initially it may say 'Waiting for first checkpoint'. If interrupted,
repeat the identical training command with `--resume auto`. A source
revision change requires a new run root, not reuse of the existing one.

## Tests and troubleshooting

`bash scripts/check.sh` runs the intentionally small Python and browser contract suite plus CLI/CPU-doctor checks. The GitHub browser CI job separately builds the static frontend and verifies the bundled policy/WASM assets. Passing tests is **not** physical gait proof. Verify signed command tracking, footfall pairs, dynamic slopes, servo torque/power and simulator parity independently.

If an install fails, do not delete old experiments. Verify `git status --short`, `git rev-parse HEAD`, `python -c "import emu; print(emu.__file__)"`, `nvidia-smi`, and the failing run's `console.log`. Optional `warp` import warnings don't necessarily mean MJX failed. If a checkpoint is corrupt, preserve the original and use the validated automatic recovery paths. Any source change requires a new campaign identity.

## Public release

The only deployable policy in Git is `showcase/champion.emupolicy`. The manual GitHub Pages workflow checks its pinned Git blob hash; training `.pkl` files stay local. [PLAYGROUND.md](PLAYGROUND.md) describes build and publishing.
