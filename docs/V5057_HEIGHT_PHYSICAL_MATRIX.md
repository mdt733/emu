# EMU v5057 — taller bodies, physically matched springs, paired 20M cold starts

**Purpose:** choose a mechanically plausible geometry that learns stable walking
*from zero* in 20M world steps, BEFORE spending 80–200M on stair learning.
Parent v5056 C018/12V progressive knee-spring **mechanics** remain
unchanged at `base26`. v5057 deliberately re-aims the ToF array
for *all* candidates (including base26), so the sensor calibration is
not identical to the archived v5056 source. No merges into `main`,
no inherited checkpoint, no new policy or separate terrain engine.

## How tall? Ground-to-hip at neutral, not just straight-leg reach

The existing two ~170mm links produce ~258mm hip height at their neutral
45°+45° crouch, with 18mm-radius ball feet. A **400mm neutral hip** needs
~270mm thigh and ~270mm shin (540mm straight-leg span). That's a major
1.59× growth in leg length, NOT a 40cm version of the existing short legs.

All options below keep the original **300g torso mass**, two ±80mm hip
mount offsets, ball feet, 360 sensor channels, six C018 12V servo
outputs, servo gear=1:1, spring attachment levers (30mm), and
contact topology. To preserve neutral fore-aft balance, we **move the
torso payload centre forward** as longer folded rods shift their mass
rearward: +26mm (base), +36.5mm (30cm), +48.3mm (34cm), **+68.8mm
(40cm)** relative to the hip. This is a real packaging requirement,
not free anti-gravity. Verify battery, wiring, chassis width and actual
motor mount distribution before committing to a taller physical build. The **same rigid
4×4 ToF array is tilted downward** according to neutral hip height,
so its shallowest beam hits level ground ~25cm ahead. These actual
per-profile extrinsics travel in the exported sensor contract.
Longer thigh/shin tubes gain mass in proportion to length, while their
own inertia is recomputed by real MuJoCo geometry. The passive carbon
shin flex's rotational stiffness decreases as EI/L for the longer tube. Torso mass stays fixed, but its neutral position above the feet is
rebalanced as described. This is a first-order carbon-tube mass model,
NOT a CAD bill of materials. Larger limbs also produce more swing inertia and demand more
hip/knee torque; a longer leg is not automatically better at a step.

| Profile | Hip neutral | Thigh & shin | Total mass approx | Knee spring k | Hip roll k |
|---|---:|---:|---:|---:|---:|
| base26 | 25.84cm | 17.00cm | 480g | 750 N/m | 2 Nm/rad |
| h30 | 30cm | 19.94cm | 504g | ~924 N/m | 2 Nm/rad |
| h34 | 34cm | 22.74cm | 527g | ~1,104 N/m | 2 Nm/rad |
| h40 | 40cm | 27.02cm | 562g | ~1,400 N/m | 2 Nm/rad |
| h34_k75 | 34cm | 22.74cm | 527g | ~828 N/m | 2 Nm/rad |
| h34_k125 | 34cm | 22.74cm | 527g | ~1,380 N/m | 2 Nm/rad |
| h34_roll1 | 34cm | 22.74cm | 527g | ~1,104 N/m | **1** Nm/rad |
| h40_k75 | 40cm | 27.02cm | 562g | ~1,047 N/m | 2 Nm/rad |

**Selection premise:** 30–34cm is the practical leading range.
40cm is worth testing as an upper-bound experiment, not promising it
will be feasible. Height alone cannot remove the 36mm-diameter spherical
foot's contact/rolling challenge at a 3–4cm riser.

### Knee spring design per body

At 90° neutral knee flexion the two 30mm strut anchor arms have a
42.4mm separation; full extension gives 60mm unloaded separation;
deep -55° gives ~18mm and 42mm spring compression. To preserve
near-weight-balanced *neutral* posture, stiffness is scaled by **total
robot mass × leg length** relative to the base26 body. Spring percentage
75%, 100%, 125% is then a separate physical treatment.

The 40cm 100% support variant can exceed **1.6Nm** extension torque
at the deepest crouch, which the swing leg must fight to fold. 2.94Nm
is a **stall** torque, and 0.98Nm is the manufacturer's nominal rated
torque. Hence `h40_k75` explicitly tests whether partial support is
better than an oversized spring. This is NOT a buildable cartridge
design yet; the high-k 21mm-stroke rocker needs mechanical/CAD and
heat/current review before hardware production.

### Why gearbox ratios are held at 1:1

The C018 is rated ~4.72 rad/s no-load. The current code has a
position-actuator peak torque clamp plus viscous damping, NOT a calibrated
speed-dependent motor torque/thermal model. Assigning 1.5:1 reduction by
simply increasing the torque limit would give false results; a genuine
gear variant would have to change servo-length/target conversion,
reflected inertia and load-limited speed. Hold all external gear ratios
at 1:1 here to isolate the morphology questions.

v5057 adds **read-only** nominal motor speed warning metrics:
`servo_speed_over_noload_frac`, `servo_peak_speed_ratio` and
`servo_linear_envelope_warning_frac`. These do NOT enforce a
mechanical/electrical velocity limit, and forced/back-driven motion can
exceed the no-load rating. Reject seemingly great gaits with huge
unphysical speed demands; do not claim simulated success as hardware
proof. ToF is 0.04–1.5m. The existing 40° far ray would
look ~48cm ahead from a 40cm hip—past the four-tile window's
guaranteed 37.5cm nearest support. We now **rigidly re-aim all
four pitch rows together** (preserving their 20° spacing) to keep
the shallowest ray's flat-ground footprint ~25cm ahead at neutral.
Extreme pitch and downhills can still create voids; inspect real ToF
validity before choosing a finalist.

## Physics/CI safety: no robot cloning

`emu/robot.xml` remains the **single master skeleton**. One small
`emu/physical_variants.py` generates a deterministic MJCF model per
profile, in a content-addressed **cache outside Git**. The selected
profile is one environment variable, `EMU_PHYSICAL_PROFILE`, set
**before starting Python**.

`contracts.ROBOT_XML` resolves exactly the selected model everywhere:
MJX training, fixed evaluation, doctor, browser policy exporter and
independent prism/matrix exams. `NEUTRAL_ROOT_Z` and exported viewer
neutral height track the selected body. TrainingConfig explicitly saves
`physical_profile`; the existing checkpoint IO pins the exact XML
SHA256 and refuses wrong-profile resumes. The same Git source commit
is shared by all variants; the **combination of source+profile+XML hash**
is physical source identity.

The **existing** core physics test (no extra test file) compiles every
variant with real MuJoCo, checks six actuators, two compression tendons,
full robot mass, neutral foot elevation, symmetric passive knee torque
and unmodified motor ceilings. Default `base26` still validates the
original v5056 spring sweep and eight-prism physical contact.

## Run from WSL: independent 8×2 = 16 cold starts

Start with a fresh *clean* worktree; never edit `robot.xml` between
runs. The launcher runs **one** 2048-world GPU learner at a time in the
foreground. It exits immediately on a failed run and preserves existing
checkpoints; rerunning the same command validates and resumes safely.
All output is outside the repo.

```bash
# Explicitly fetch this branch: the default fetch refspec may NOT fetch
# experimental refs, even when the commit exists on GitHub.
cd ~/emu &&
git fetch origin \
  '+refs/heads/research/v5057-height-physics-matrix:refs/remotes/origin/research/v5057-height-physics-matrix' &&
git worktree add --detach ~/emu-v5057-height \
  refs/remotes/origin/research/v5057-height-physics-matrix &&
cd ~/emu-v5057-height &&
test -f emu/physical_variants.py &&
echo "V5057 SOURCE: $(git rev-parse HEAD)"

# STOP on any failed command above. Never run a smoke from ~/emu instead.
cd ~/emu-v5057-height || exit 1
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Mandatory source-specific model preflight: genuine h40 XML must be
# distinct from the 26cm source, even before importing MJX.
EMU_PHYSICAL_PROFILE=h40 python -c '
from emu.contracts import ROBOT_XML
from emu.physical_variants import profile_numbers
m=profile_numbers("h40")
print("MODEL:", ROBOT_XML)
print("NEUTRAL HIP:",m["hip_height_m"], "MASS:",m["mass_kg"])
assert m["hip_height_m"] == .40
assert "h40" in ROBOT_XML.name
'

# Core physical checks before borrowing the GPU:
python -m pytest -q tests
EMU_PHYSICAL_PROFILE=h40 python -m emu.cli doctor --no-require-gpu --no-mjx-smoke

# Source-pinned full GPU acceptance on the largest morphology:
EMU_PHYSICAL_PROFILE=h40 python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 \
  --ppo --full-update --benchmark-updates 3 \
  --prisms --solver-iterations 2 --solver-ls-iterations 4

# Read-only, prints all 16 commands before any training:
python scripts/v5057_physical_tournament.py

# First paired block: same seed 735, all eight bodies, 20M each:
python scripts/v5057_physical_tournament.py \
  --seeds 735 --execute

# Second paired block: new independent cold seed for every body:
python scripts/v5057_physical_tournament.py \
  --seeds 761 --execute

# Compare ALL 16 runs, no GPU:
python scripts/v5057_physical_tournament.py --report
```

Expected outputs:

```text
~/emu-data/runs/v5057-physical-matrix/
  experiment.json
  h34-s735/{training_config.json,run_meta.json,metrics.csv,
             fixed_eval.csv,results.md,checkpoints/final.pkl,...}
  h34-s761/{...}
  ...
  physical_matrix.csv
  physical_matrix.md
```

Each run uses exactly **20,000,000 target world steps**, 2048 envs,
same source SHA, C018 model, seed-matched learning economics, same
full-sized PPO and **Academy ramp only after 20M**. This is a
foundational stability screen, not a staircase teaching tournament.
The same run's final frozen evaluation is invoked at the 20M
boundary. Training reward is logged but is NOT a selection score.

### How to select winners without fooling ourselves

- Stage 0: all model variants pass static physical geometry and
  symmetric spring checks, MJX 2048 GPU smoke and finite loss.
- Stage 1: inspect walking **by 5M, 10M and 20M** for both seeds.
  Target stable balance by 20M, not merely the lowest fall rate.
- Stage 2: compare **physical episode length**, fall fraction,
  actual vs requested walking speed, rated torque exposure,
  motor speed-limit warnings, roll saturation and independent
  fixed-eval score. Pair seeds for each identical physical
  model. Reward improvements alone are insufficient.
- Stage 3: choose 2–3 candidates with strong *paired* generalist
  performance, run an independent seed 907 per finalist and frozen
  flat/step physical matrix at signed 1–5cm heights. Check *both*
  upper-foot contacts and sustained post-lip forward progress.
  Training is still 20M in this comparison; do NOT compare a 20M
  tall body against an old 180M short body and call it a win.
- Stage 4: graduate the mechanically best source-pinned build to
  **20→80→140M** continued Academy only after it shows real
  2–3cm ascent without degenerating flat gait, yaw/lane control
  or hip-roll authority. Continue via same-profile
  `--resume auto` and unchanged source/checkpoint model hash,
  not a foreign-policy transplant.

To export one selected body's actor, always select its matching
profile **before** `emu` starts:

```bash
export EMU_PHYSICAL_PROFILE=h34
ROOT="$HOME/emu-data/runs/v5057-physical-matrix/h34-s735"
python -m emu.cli export-policy \
  "$ROOT/checkpoints/final.pkl" \
  "$ROOT/h34-s735.emupolicy"
python scripts/v5053_matrix_exam.py \
  --policy "$ROOT/h34-s735.emupolicy" \
  --output "$ROOT/independent-stairs" \
  --mode stairs --worlds 2048 --horizon 350 \
  --seed 1701 --levels 20 --trials-per-level 100
```

The independent matrix is **not run automatically for every 20M
candidate** because a barely-walking foundation is not a fair stair
claim and 16 GPU exam compilations would waste time. Do it for the
top 2–3 finalists under identical controlled conditions.

## Gating and next challenges

What we learn: achievable mass/height/spring trade-off for cold walking
under identical training. What we **do not** learn from this tournament:
verified real servo speed/thermal capacity, full spring hardware
packaging, robust stair learning, high-speed biped hardware stability,
whether the hip-roll action saturation is a torque problem, or whether
the 360 actor can reliably classify 1–3mm ToF steps.

Keep the losing variant evidence in the recorded `physical_matrix.md`
and Git history, then delete this one-off campaign launcher when the
result is integrated into the maintained robot skeleton. Don't add
eight alternative training architectures or a new wall of tests.
