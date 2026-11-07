# v5056 — 12V C018 + progressive knee compression struts, then honest stair learning

**Status:** unvalidated physical-model research branch, NOT a hardware-build approval,
checkpoint continuation, promoted policy or GPU proof. Parent is v5055 credit audit.
`main`, the v5054 182M actor/checkpoint and the released showcase are untouched.

## Physical model: one shared robot and one PPO

The old motor profile mixed a 6V speed with 7.4V C001 stall torque.
All **six** position actuators now represent FEETECH ST-3215-C018 at **12V**:
stall 30 kgf·cm = **2.941995 N·m**, manufacturer rated 10 kgf·cm =
**0.980665 N·m**, no-load 0.222s/60° (4.717 rad/s), 55g/servo and
stall **2.7A**. Torque limit is peak-stall, NOT continuous usable torque.
The existing **0.624 N·m·s/rad** joint damping is a provisional speed
proxy (peak / no-load speed); it damps even when unpowered and is not a
faithful electrical torque-speed or thermal model. `kp=10`, armature
`0.012` and Coulomb friction `0.08` are also uncalibrated inherited
values. Do not infer real temperature, battery current, mechanical power
or endurance from simulation metrics.

Knee springs are two **spatial compression paths** with physical site
coordinates **30 mm from each knee pivot**, one on thigh and one on shin.
Each has **750 N/m** equivalent stiffness and **60 mm rest length**.
Actual site separation is ~42.4 mm at neutral q=0°, ~18 mm at deep
q=-55° and 60 mm at straight q=+90°. The passive MuJoCo spring is
restoring under *compression*, producing approximately
**0.280 N·m** knee extension at neutral, **0.900 N·m** in deep crouch
and **0** at straight. This is a **strut**, not a tension string.
Two knees at neutral can nominally support ~0.48kg in simplified
geometry; at deep crouch a knee provides an initial upward push
exceeding the full robot weight. It is NOT capable of unpowered
single-leg rising all the way to neutral because the force drops as
it extends. The actuator still controls real swing and lift.

Removed old `hip_pitch` and knee torsion-spring moments; retained 2
N·m/rad hip-roll centering and compliant shin flex for now. These
strong hip-roll springs remain an **independent suspect** in the observed
left action saturation: 19.4%→67.2%. New actuator torque does not increase
the hip's command angle range and cannot alone cure action clipping.

For a real prototype a **2:1 rocker** driving a **3.0 N/mm** guided
cartridge with about **21mm working stroke** is a starting geometry,
not a finished parts specification. Packaging, coil bind, buckling,
load path, hysteresis, link interference, tolerance and fatigue require
CAD and a physical measurement. Source XML's effective 750N/m strut
is the target torque curve, not yet a validated assembly.

Manufacturer references:
- https://ftservo.github.io/ftservo-wiki/en/products/models/st-3215-c018/main/
- https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf

**Electrical:** C018 is a 12V servo. Never put 12V on an existing C001,
and don't assume the same cabling/regulator can carry the new peak.
6 × 2.7A = **16.2A theoretical simultaneous stall draw**. Design for
transients, fuse/wire ratings, overheating and servo protection; verify
the actual supply and connector current limits.

## Physics gates: prove before training

The **existing** core physics test contains real MuJoCo `mj_forward`
passive-force measurements at knee q = -55°, -40°, 0°, 45°, 90°.
It must check sign, left/right symmetry, force curve and zero torque
at straight, plus all six C018 peak ranges. The **existing** four-world
MJX stair test checks that both struts actually store energy and
produce passive force. No added test file or duplicate trainer.

On the Windows/WSL GPU, from a fresh v5056 worktree and the common
`~/emu/.venv`:

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-v5056-springs \
  origin/research/v5056-spring-c018-stairs
cd ~/emu-v5056-springs
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m pytest -q tests
python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2
python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 128 --frames 4 --ppo --prisms \
  --solver-iterations 2 --solver-ls-iterations 4
python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo --full-update \
  --benchmark-updates 3 --profile-rollout --prisms \
  --solver-iterations 2 --solver-ls-iterations 4
```

Gate: CPU/MJX spring sign and static torque curve pass, real eight
prisms still support the foot, no nonfinite contact/rollout, GPU
PPO updates improve parameters, and the new two spatial tendons don't
destroy GPU throughput/memory. Measure sustained throughput against
the **same** solver on the previous source, not against an unrelated
34-arm benchmark. The earlier solver 2/4 vs reference fidelity caveat
still applies—repeat a small physical crossing comparison at reference
iterations before claiming hardware fidelity.

## Academy repair from v5055: no blind promotion

The old 1.5mm **requested** minimum was actually an unclassifiable
first lip (0% recognised, even without a fall). v5056's minimum frontier
is **3mm**, and the first `-1→0` or `0→1` seam samples close to the
requested frontier *after physical jitter*, rather than starving the
`0.78..1.05` promotion band. Later terrain seams retain their
original varied challenge. Existing core test samples 4096 deterministic
seeds and checks both possible spawn seams: heights >1.5mm and a useful
near-frontier majority. This is a **teacher repair**, separate from
the physical motor/spring change in Git history. It does not mean a
3mm lip is visible to a noisy ToF; paired fixed-pose sensing remains
a needed exam.

Do not use the 182M checkpoint as `--resume` into this hardware model:
the physical XML and learning recipe changed. Source-pinned **cold
starts only**. The previous v5055 8M diagnostic is a separate
old-physics experiment, and its normalized PPO completion-credit
result should be reviewed before deciding whether to modify rewards.

## Training: graduation gates, not an automatic 200M money pit

### Stage A — verify new physics and start 20M cold foundation

Use one source pin, two explicitly separate seeds (735 initially,
761 only after the first passes), 2048 environments. The usual first
20M establish low-speed reliable balance and confirm that stepping
mechanics did not break standing. **Do not** run a long Academy
campaign on a robot that cannot get back up.

```bash
cd ~/emu-v5056-springs
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
ROOT="$HOME/emu-data/runs/v5056-strut-c018-s735"
test ! -e "$ROOT" || { echo "Run exists; refusing overwrite"; exit 1; }

ARGS=(
  --run-dir "$ROOT" --steps 20000000
  --envs 2048 --seed 735 --device cuda --profile sim2real
  --terrain-actor --unified-capabilities
  --terrain-carpet --carpet-prisms --academy-stream
  --terrain-step-max-m .08 --terrain-lesson coherent-careful
  --growth-recipe main-mature --foundation-steps 20000000
  --fixed-eval-every-steps 20000000
  --solver-iterations 2 --solver-ls-iterations 4
  --academy-intro-steps 20000000
  --academy-start-steps 40000000 --academy-ramp-steps 15000000
  --academy-intro-share .20 --academy-max-share .60
  --athletic-rehearsal-share .15 --retention-share .15
  --temporal-vx-reward-share .25
  --science-step-approach-m .22 --science-step-goal-weight .30
  --science-step-finish-on-transfer
)
python -m emu.cli train "${ARGS[@]}"
```

Stage A graduation: stable gait by ~20M, physical motor load not
permanently at stall, knee strut compression and stored energy vary
as expected, two feet make bona fide contact and source-pinned
held-out survival is reasonable. If not, fix dynamics before PPO.

### Stage B — continue SAME source & run to 80M

Only if Stage A passes, continue the **same run and config** with
`--steps 80000000` (absolute world-step target), using EMU's
same-source checkpoint/resume contract. Do not invent an external
cross-source fork or discard optimizer/curriculum state. Review
`emu train --help` and the run's `training_config.json` before
resuming; the full argument recipe other than steps must match.

At 40M the Academy expands; at 55M it can use up to 60% of worlds.
Retain 15% useful walking so the stair nursery cannot erase all
generalist control. This is one shared network, not specialist actors.

**Stop and investigate before 80M** if:
- hip-roll command clipping diverges left/right, *especially outside
  Academy*, while held-out lane tracking deteriorates;
- Academy real first-lip attempts / two-foot support stay near zero
  despite actual generated lips;
- enough real completed climbs have **predominantly negative
  normalised PPO advantages** (use v5055 cumulative event-weighted
  audit; zero-event mean is unknown);
- physical knee/hip motors spend much of the time at peak stalls;
- supported ascending contact improves but controlled after-lip
  traversal and lane/yaw score continue to regress.

### Stage C — independent physical exam, then 80→140M or forks

Export the 80M actor to a separate `.emupolicy`, then run held-out
**same-seed, same-world-count** flat, uphill and balanced/downhill
panels with the maintained matrix examiner:

```bash
ACTOR="$ROOT/actors/v5056_80m.emupolicy"
mkdir -p "$ROOT/actors"
python -m emu.cli export-policy "$ROOT/checkpoints/final.pkl" "$ACTOR"
python scripts/v5053_matrix_exam.py --policy "$ACTOR" \
  --output "$ROOT/exams/flat" --mode flat \
  --worlds 2048 --horizon 350 --seed 1701 --trials-per-level 2000
python scripts/v5053_matrix_exam.py --policy "$ACTOR" \
  --output "$ROOT/exams/stairs" --mode stairs \
  --worlds 2048 --horizon 350 --seed 1701 \
  --levels 20 --trials-per-level 100
python scripts/v5053_matrix_exam.py --policy "$ACTOR" \
  --output "$ROOT/exams/dense" --mode stairs \
  --worlds 2048 --horizon 350 --seed 1701 \
  --levels 40 --trials-per-level 50
```

The `--levels 40` height scan samples 1.25mm increments (1.25..50mm),
including the interesting **28–44mm foot-diameter transition**.
The published matrix script has no start-distance randomisation,
ToF ablation or dedicated downhill mode: these are proposed next
fixed-policy tests, **not** secretly claimed as implemented.

Measure: per-height first upper-top foot, independent second foot,
5-frame controlled traversal, exact side/lane drift and heading,
strict parking and motor/spring effort. Do NOT equate step contact
with completing a route. Compare the 0-height seam first: if flat
drift equals stair drift, fix deployable yaw/odometry feedback before
paying more stair-specific reward.

Advance 80→140M only if 2cm and 3cm supported *controlled* ascent is
improving without destroying generalist gait. If there is a clear
winner from matched seeds, **then** fork controlled treatment arms,
not eight random modifications to PPO. A narrow A/B can isolate:
- motor/spring benefit (same teaching recipe, old hardware source);
- 3mm close-to-frontier Academy delivery (v5056 teacher repair);
- completion bonus/absorbing-terminal fix ONLY if v5055 shows
  negative normalized credit;
- yaw-hold/perception control ONLY if matched flat/ToF tests justify.

No training reward redesign, actor inputs, six-policy architecture,
additional simulators or broad test scaffolding in this branch.
Once a losing treatment is falsified, delete its code and leave the
scientific results in docs/Git history. Keep <=24 core tests.

## Hardware/learning conclusions not yet established

- Actual C018 continuous torque, heat and current under EMU gait.
- Physical spring package with the specified moment arm and travel.
- MJX-vs-CPU passive spring/contact force parity across knee angles
  (the tests must establish this).
- Whether hip roll is position- or torque-limited in walking.
- Whether ToF reliably resolves a 2–3mm first lip at approach speed.
- Whether the end-of-course bonus has positive policy gradient credit.
- Any actual improvement beyond 3cm or 4cm stairs.
