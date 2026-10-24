> **RETIRED — DO NOT RUN.** Owner replaced the flat/slope-free A/B campaign with a single shared-policy, all-capabilities specialist-world curriculum. See [UNIFIED_CAPABILITY_GROWTH.md](UNIFIED_CAPABILITY_GROWTH.md). The old launcher now refuses to start.\n\n# EMU v5023 — overnight G6 prep, operator runbook

**Feature branch ONLY.** This runbook is for experimental training; NEVER promotes
a policy into `main`, public playground, or existing `champion.emupolicy`.
The ten-hour campaign is configured to run one 2,048-world GPU job at a time.

## Which source is which

- **Control**: untouched original **`411b841de08ebfbb65c29b357c8ee0d39d577dd1`** checkout at `~/emu-v5023-window`. Retains existing 34M final checkpoint `~/robot_lab/emu_runs/v5023_window_fresh733/growth34/checkpoints/final.pkl`.
- **New A/B source**: clean detached checkout of **THIS feature branch's reviewed final commit** at `~/emu-v5023-overnight`, distinct from the old checkout. Has the new saved setting `--terrain-training-arm {off,speed,foothold}`.
- Both use `terrain_360`, same 256×256×6 feedforward actor, same physical 4×4 forward ToF, identical flat 20M physics foundation and robot XML. Arm B adds only one bounded, once-per-episode physical upper-foot touchdown payment to Arm A's matched speed curriculum.

## Preflight, when the current branch CI and CUDA doctor are green

~~~bash
cd ~
git clone --branch feature/v5023-capability-growth --single-branch \
  https://github.com/mdt733/emu.git emu-v5023-overnight
cd ~/emu-v5023-overnight
# Freeze to the reviewed commit; do NOT git pull this checkout during training.
git checkout --detach "$(git rev-parse HEAD)"
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --require-gpu
python3 scripts/smoke_v5023_arms.py --device cuda --envs 4
emu viewer-build
# Does NOT launch or write model checkpoints:
python3 scripts/overnight_v5023.py --dry-run
~~~

The experiment script refuses dirty worktrees, missing venv executables, wrong
original control source SHA and absent original 34M checkpoint. It writes
`campaign_identity.json` before training, including old/new Git SHAs and
the SHA256 of the old 34M parent checkpoint. Every restart verifies this identity.

Do **not** upgrade the old `~/emu-v5023-window` source. Older source 411b841
must be used for the unchanged-control 34M→60M→110M run, and only the *new*
source may run `speed`/`foothold`.

## Live viewer: separate WSL terminal, entirely read-only

~~~bash
cd ~/emu-v5023-overnight
source .venv/bin/activate
ROOT="$HOME/robot_lab/emu_runs/v5023_overnight733"
emu viewer --live --root "$ROOT" --port 8765
~~~

Open `http://127.0.0.1:8765/` on the Windows PC. The live viewer polls validated
`latest.pkl`/`final.pkl` checkpoints every ~12 seconds. It chooses the most
recent fully readable checkpoint across separate `control60`, `foundation20`,
`A60`, `A100`, `B60`, `B100`, etc., without reading the GPU's live tensors or
running MJX in the viewer. Latest `terrain_360` browser policy now uses delayed
**16 real slant-range rays with valid bits**, not the old 232-input parser.
Viewer's **1cm and 2cm depth-lab steps** are also physical collision meshes.

**Interpretation caveat:** this is a locally run browser physics demonstration
driven by the latest policy. It is *not* a video of a specific batched MJX world.
The browser uses a simple geometric raycast approximation to MJX's terrain
sensor and has independently selectable courses. It cannot show actual training
episode populations, phase seeds or exact contact history. Frozen CPU exams are
the authoritative physical comparison.

If viewer says it is waiting for checkpoints, start the campaign below. If
a checkpoint is being atomically replaced, the watcher retains the last valid
actor. The viewer binds to loopback by design, not a LAN-facing address.

## Actual overnight launch: in a DIFFERENT terminal

~~~bash
cd ~/emu-v5023-overnight
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 scripts/overnight_v5023.py --hours 10
~~~

`python3` runner launches the installed `.venv/bin/emu` executable separately
from the OLD control checkout and NEW experiment checkout. It doesn't mutate
any source. It handles fresh individual target steps, or `--resume auto`
inside a partially completed SAME-run directory when restarting.

Run order:
1. Existing 411b841 34M control →60M→110M.
2. New source identical flat 0→20M foundation.
3. Arm A (matched speed × step, no extra touchdown reward) 20→60→100M.
4. Arm B (same plus supported first foothold) 20→60→100M.
5. Each control/A/B policy receives frozen CPU exam at three command speeds
   (`0.15/0.25/0.30m/s`) × three independent terrain seeds
   (`1701/1702/2417`), with genuine platform contact, survival, work and
   separate UP/DOWN counts.
6. A and/or B earn further 160/260/380M blocks only if they *beat the unchanged
   control* in upward crossings while preserving reasonable fall rate, flat
   progress and shaft-work per metre. A qualified runner-up can grow to 180M.

The wall-budget guard predicts time per block conservatively, keeps a 45-minute
reserve before a new stage, and never starts two GPU trainers together. It does
**not** kill a running block at exactly ten hours: checkpoint consistency and
preserving existing experiments take precedence over a hard deadline. It may
finish early if no candidate earns credible further compute, or if time is
insufficient for another whole stage.

Saved outputs live in:
`~/robot_lab/emu_runs/v5023_overnight733/{campaign_identity.json,control60,control110,foundation20,A60,A100,B60,B100,exam_policies,exams,ab_exam_scores.json,...}`

If the GPU process is interrupted, rerun the exact SAME command in the SAME
checkout to resume the first incomplete stage, provided it has a validated
`latest.pkl`. Do not change the code SHA, seed, env count, observation schema
or arm. If the control/experiment code changes, start an entirely new campaign.

## Decision gates and risk notes

- Real uphill vs downhill supported crossings, falls, and actual sampled step
  heights are separate. A 1cm course does not mean the physical step was 1cm.
- Faster requested speed is **not** the real speed at the lip. High-speed
  collisions can produce both spurious progress and more falls.
- Core quality and mechanical shaft-work budget matter; neither PPO reward
  nor a single speed's crossing count alone grants promotion.
- A 0.95cm step can partly roll over an 18mm spherical foot. Demand >1.8cm
  physically witnessed ascents before claiming deliberate high foot clearance.
- Existing champion, public demo, main branch, old control code and prior
  run directories are all protected.
- CI is necessary, but only an actual **CUDA doctor + small G6 360-input
  training smoke on the operator's 4060 Ti** can validate device memory,
  new JAX tracing and wall FPS. Do not begin a full overnight run if that
  smoke cannot finish cleanly.

**Current status:** launcher and two arms have been committed; this document
does not represent a claim that the 10-hour CUDA run has executed successfully.
