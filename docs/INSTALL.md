# EMU installation and operation

> **v5018 feature-branch instructions (20 October 2026).** This section is
> current for `feature/v5018-terrain-awareness`; earlier v5016 experimental
> commands below remain as historical examples, **not** directions to run
> v5018 terrain training. See [v5018 handover](V5018_HANDOVER.md) and the
> [one living status/checklist](V5018_TERRAIN_PLAN.md).

## A genuinely blank Ubuntu WSL2 (one copy/paste block)

EMU has **two runtimes**: Python 3.13+ + JAX CUDA + MuJoCo/MJX for
physics/learning, and Node.js 24 + npm + pinned MuJoCo/WASM + Three.js/Vite
for the viewer. The Python virtual environment and nvm live in the normal
WSL home directory, not in Windows system Python.

Windows: first verify WSL2 and a recent NVIDIA **Windows** driver; run
`wsl --update` in PowerShell if needed. Do **not** install Linux NVIDIA
display drivers inside WSL. [NVIDIA's WSL instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Open Ubuntu WSL as the normal user, **not sudo/root**, and paste:

```bash
set -e
sudo apt-get update
sudo apt-get install -y git curl ca-certificates

# Dedicated branch clone. Never clone main and then run a v5018 campaign.
cd "$HOME"
git clone --branch feature/v5018-terrain-awareness --single-branch \
  https://github.com/mdt733/emu.git emu-v5018
cd "$HOME/emu-v5018"

# Ubuntu packages, user-local uv/Python 3.13, .venv and pip,
# nvm/Node 24/npm, JAX CUDA 13, editable EMU, locked viewer + tests.
bash scripts/bootstrap_wsl_v5018.sh

# Verify the active shell. setup sources nvm internally; source it again
# in this login shell when you need npm yourself.
source .venv/bin/activate
export NVM_DIR="$HOME/.nvm"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then source "$NVM_DIR/nvm.sh"; fi
python --version
python -c 'import jax,mujoco; from mujoco import mjx; print(jax.__version__,mujoco.__version__,jax.devices())'
node --version
npm --version
nvidia-smi
git branch --show-current
git rev-parse HEAD
```

The bootstrapper installs Ubuntu packages via `sudo apt-get`, installs
`uv` from its official installer if needed, obtains Python 3.13
**without replacing Ubuntu's `/usr/bin/python3`**, creates/reuses
`~/emu-v5018/.venv`, bootstraps `pip` even when uv's venv lacks it,
then invokes `scripts/setup_wsl.sh --cuda13 --install-node`.
That existing setup installs `jax[cuda13]` and `.[dev]`, installs
Node 24 via pinned nvm, builds the viewer using `npm ci` and the
committed lockfile, runs npm tests, the Python pytest suite, CLI plan and
GPU/domain doctor. The GPU doctor checks the environment; the separate
campaign below includes a small real PPO-update smoke. Full setup does
**not** start a long training run.

A `curl | sh` installer is convenient but executes upstream install
code: inspect it or pin/verify a trusted version in controlled environments.
The exact CUDA/JAX wheels may evolve; record the versions shown by your
local run. Tested on the user's WSL: Python 3.13.16, JAX 0.12.0,
MuJoCo 3.16.0, `cuda:0`. The browser dependency is pinned separately
to **MuJoCo/WASM 3.15.0**; the engine gap is an open P0B measurement,
not evidence of equivalence.

### CPU-only mode and partial-install recovery

Use `bash scripts/bootstrap_wsl_v5018.sh --cpu` if a GPU is unavailable;
this cannot establish GPU feasibility. `--skip-tests` explicitly skips
the Python regression suite **only**; it is *not* a successful regression
or full P0 gate. It can finish the npm installation while diagnosing test
failures. The first clean-WSL Python suite was **73 passed / 5 failed**
because tests assumed an obsolete 220-input champion; the branch has
compatibility patches, **not yet reverified with full pytest**.

If `python` is missing, the OS default may expose only `python3`:
activate `source .venv/bin/activate`. If Python is 3.12 or an old/forbidden
`.venv` exists, do **not** overwrite another project's environment:
inspect `ls -ld .venv` and use the dedicated v5018 clone and Python
3.13 bootstrap. If `node` or `npm` disappear in a *new terminal*:

```bash
cd ~/emu-v5018
source .venv/bin/activate
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"
nvm use "$(cat .nvmrc)"
node --version && npm --version
```

For GPU diagnostics run `nvidia-smi`, then
`python -c 'import jax; print(jax.devices())'`. A host-driver/WSL
GPU issue should be repaired on the Windows NVIDIA-driver side, not
by installing another Linux driver. Optional `warp` import warnings
do **not** by themselves mean MJX-JAX failed. If npm fails, preserve
the complete error and check `node --version`, `npm --version`,
`emu/viewer_src/package-lock.json`, and `npm --prefix emu/viewer_src ci`.
Avoid changing the lockfile during a running test campaign.

### Existing clone, tests and viewer

**If the big P0 batch is currently running, leave its checkout and
dependencies untouched until it finishes and its report archive is saved.**
To update after it completes:

```bash
cd ~/emu-v5018
git status --short
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/setup_wsl.sh --cuda13 --install-node
# or, for a controlled quick check:
bash scripts/check.sh
```

Local viewer: `emu viewer-build` then `emu viewer --help`
(for available arguments); the local read-only server uses
`http://127.0.0.1:8765/` and is **not remotely authenticated**.

### The v5018 P0 experimental campaign and evidence

```bash
cd ~/emu-v5018
source .venv/bin/activate
bash scripts/run_v5018_p0_campaign.sh
```

This runs full Python regression; npm install, test and build;
GPU doctor plus a bounded PPO smoke; stationary real collision and
plane/box/capsule rays; headless MuJoCo/WASM CPU comparison;
moving toe strikes and non-strike control; per-world MJX geom pose
variation; two-lane forward scanner coverage; kinematic clearance;
and isolated 256→512→1024→2048-world GPU environment stepping
(flat / unloaded obstacle / real foot-contact stress). It is a
**feasibility campaign, not long PPO terrain learning**.

Each independent stage logs PASS/FAIL and continues; larger GPU sizes
are skipped after a smaller failed size. Outputs include
`stages.tsv`, per-stage logs, JSON files, and a `.tar.gz` outside Git
at `~/robot_lab/emu_experiments/v5018_p0_YYYYMMDD_HHMMSS.tar.gz`.
Save/upload the archive even if the script exits 1; **do not rerun
the whole campaign automatically if a single stage fails**. Inspect
`environment.txt` (branch, commit, GPU) to identify the exact source
revision tested. A PASS means only the stated probe passed, not that
the biped learned obstacle climbing.

This feature branch uses **two frozen policy interfaces** (legacy
55×4=220 and sim2real 58×4=232) with six actions. The current
protected showcase champion is the 260,112,384-step **232-input**
export (checkpoint schema 8). A new terrain observation interface is
**not yet frozen**; `terrain_312` is only a hypothesis pending
P2 scanner/bin experiments. Never modify a champion or resume old
training checkpoints into a changed source/schema.

---

## Earlier general EMU / v5016 install and operation guidance (historical)

# Historical general installation notes (prior to v5018)

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

**Historical Pitch15 interface only:** 220 observations → 6 actions, checkpoint schema 7. The protected current showcase is 232-input/checkpoint schema 8; see the v5018 instructions above. Do not resume across a changed source revision or reuse run roots with different environment counts.



### v5016 spring and real-contact slope checks

The independent physical exam now accepts a genuinely inclined infinite
MuJoCo collision plane (the original XML stays unchanged, and gravity stays
vertical). This is **not** the earlier tilted-gravity approximation, and
the printed report explicitly identifies which kind of slope was examined.

```bash
cd "$HOME/emu-v5016"
source .venv/bin/activate
OUT="$HOME/robot_lab/emu_exams/v5016_real_slope_$(date +%Y%m%d_%H%M%S)"
emu directional-exam --policy showcase/champion.emupolicy \
  --output "$OUT" --device cuda --worlds 4 --horizon 350 \
  --seed 1701 --scenario steady --speed .10 \
  --angle 2 --direction uphill --contact-slope --compare-heading
```

For a full planned protected-policy course campaign, the
`--profile physical` batch now has **13 scenarios**, including real 2°
uphill/downhill/cross slope cases. Run the planned command without `--run`
first. These experiments never alter the protected actor or train PPO.

The physical JSON records each passive shin's spring-energy charging/release
and whether its release coincided with uninterrupted stance contact. Stance
release is only an energy-timing observation—not energy delivered to the
ground, an efficiency percentage, or proof of spring-assisted propulsion.
`robot.xml` still lacks shin/torso sentinel contacts, and the actual
incline is a single infinite plane rather than rocks, steps or varying
per-world terrain.

### v5016 read-only physical reality exam (do this BEFORE expensive training)

The protected Pitch15 public policy is a legitimate **frozen actor** and now
works as an examiner input without the historical training PKL. Use a new,
isolated output directory outside Git; this creates reports only.

For a matched 10-course exam (heading correction OFF/ON, seed-controlled),
first print the planned jobs, then opt in explicitly with `--run`:

```bash
cd "$HOME/emu-v5016"
source .venv/bin/activate
BASE="$HOME/robot_lab/emu_exams/pitch15_$(date +%Y%m%d_%H%M%S)"
emu directional-batch --policy showcase/champion.emupolicy \
  --profile physical --device cuda --worlds 4 --seed 1701 \
  --output "$BASE"
# When the plan looks right, append --run to the command above.
# Repeat with a DIFFERENT BASE for held-out seeds 1702 and 1703.
```

The executable source is the approved exported policy, never any training root.
Individual output directories are source- and checkpoint-hash checked; partial
or stale results fail closed instead of being silently overwritten.
For a single case, use:

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

## v5016 common pitch and opt-in motor teaching — future trials

`emu train --profile sim2real` keeps the mature PPO, v5015 posture model and
220-input/six-action actor, but sets the same ±30° commanded chassis pitch
limit for every new physical-learning experiment. This does not change or
retrain the protected Pitch15 policy. The historical `emu ladder` still
describes the old Pitch15 comparison; **do not use that ladder as the v5016
efficiency/terrain tournament**.

Two pure plans for future matched experiments, with unique new run roots:

```bash
cd "$HOME/emu-v5016"
source .venv/bin/activate
BASE="$HOME/robot_lab/emu_runs/v5016_study_$(date +%Y%m%d_%H%M%S)"
emu train --run-dir "$BASE/baseline" --profile sim2real \
  --steps 12000000 --envs 512 --device cuda --seed 733 --plan
emu train --run-dir "$BASE/motor_preview" --profile sim2real \
  --steps 12000000 --envs 512 --device cuda --seed 733 \
  --motor-guidance-weight 0.06 --plan
```

These are **plan-only previews**, not an instruction to launch independent
cold starts or to compare unlike training ancestries. The actual study should
use one short shared foundation, fork matched training arms from exactly the
same checkpoint, retain source lineage and exam identical held-out courses.
Only run after GPU acceptance and interpretation of the protected champion
exams. The motor term is an uncalibrated training proxy for overload and
mechanical shaft demand; spring return and electrical consumption are *not*
inferred. Treat ±30° as an aspirational teaching envelope to verify in the
presence of walking, not evidence that every physical angle is attainable.

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

### v5016 optional real-contact slope training (planning only)

The new sim2real training mode can use one shared, genuinely inclined contact
floor per run, with vertical gravity. This is a conservative FIRST STEP,
**not** per-world random terrain. Historical runs retain zero incline.

```bash
OUT="$HOME/robot_lab/emu_runs/v5016_grade_$(date +%Y%m%d_%H%M%S)"
emu train --profile sim2real --run-dir "$OUT/flat" --steps 12000000 \
    --envs 512 --device cuda --seed 733 --plan
emu train --profile sim2real --run-dir "$OUT/grade2" --steps 12000000 \
    --envs 512 --device cuda --seed 733 --contact-slope-deg 2 --plan
```

These commands do not launch training. Physical grading requires matched
frozen-policy exams on both flat and real slopes and a separate GPU acceptance.
Never combine this mode with the older tilted-gravity `domain_terrain_radius`.


### v5016 per-world contact slope pilot — plan only

The independently randomized physical terrain preview is now a single
flag on the existing `sim2real` training command. **No training starts
when `--plan` is present.** It is not the historical tilted-gravity
domain study, and not a heightfield or loose/uneven ground.

```bash
RUN="$HOME/robot_lab/emu_runs/v5016_contact_pilot_$(date +%Y%m%d_%H%M%S)"
emu train --run-dir "$RUN" --profile sim2real \
  --contact-terrain-radius 0.4 --steps 2048 --envs 4 \
  --device cpu --seed 733 --plan
```

Radius 1 corresponds to **up to 5° of actual inclination** in any azimuth,
with world-specific friction variation and unchanged vertical gravity.
Worlds have different fixed planes sampled once at environment creation.
Do not combine `--contact-terrain-radius` with `--contact-slope-deg`.
Separate CPU and bounded GPU memory/performance acceptance is needed before
trying 512/2048 worlds or comparing teaching methods. This is a
simulation capability, not validated servo or on-robot estimator fidelity.
