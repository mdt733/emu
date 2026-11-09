# EMU handover — ONE 40 cm successor candidate (2026-11-08)

## Decision: stabilize the 40 cm robot before any other hardware research

User priority: **one trustworthy 40 cm, six-servo C018 12V EMU capable of
replacing main**, then revisit carbon-flex, feet and further step-specialist
work. No more stacked research PRs until one successor is proven.

**Active stabilization base:** `research/v5058-c018-rated-torque-hardcap`,
PR #49, which INHERITS the proven h40 geometry from v5057 but enforces
C018 rated output torque. No additional branch created to implement
this decision. **Main stays frozen at v5050 champion** until replacement
passes common held-out exams.

https://github.com/mdt733/emu/pull/49
https://github.com/mdt733/emu/pull/48

## Why v5058, not v5059/v5060

- v5057 h40, seed 735, 200M athletic run: 0.258m/s average forward
  vs 0.314m/s command; final training fall rate 16%; an independent
  350-frame exam survived 200/200 fast trials, 199/200 uphill and
  200/200 downhill up to 12 degrees. This is genuine simulated
  capability, but **NOT rated-torque hardware-feasible**, since
  57.7% of actuated samples exceeded rated torque and MuJoCo
  allowed 2.941995N.m PEAK STALL for unlimited time.
- v5058 **same original tall morphology and 36mm-diameter feet**,
  same 2.517 N.m/rad midpoint shin pitch spring, same scaled
  compressive knee-support tendon, six C018 servos at 12V.
  Change vs v5057: hard 0.980665N.m rated output force,
  and randomized motor limits cannot exceed that. Existing
  uncalibrated passive damping remains at 0.624N.m.s/rad:
  this is a sim-to-real LIMITATION and must be reported,
  not silently treated as realistic motor resistance.
- v5059 tried to make a realistic carbon-leg structure by CHANGING
  MANY physical assumptions simultaneously: light 478g body,
  10mm ball foot, 0.0633N.m/rad shin spring in two axes,
  0.0006N.m.s/rad shin damping, low powered-joint damping,
  new tensile knee cable, and inertia placement.
  Its 20M seed 735 cold start FAILED: 100% falls at 5/10/20M,
  26.5 frame mean episodes at 20M; measured vx ~0.638m/s
  vs requested ~0.043m/s. This cannot isolate shin stiffness.
- v5060 PR #51 contains analysis-only tooling and reports
  for that failed carbon body. Keep as an archive, not
  as the parent of the main replacement.

**Important:** 40cm geometry and rated torque HAVE NOT YET been
demonstrated together to learn a stable 20M cold-start gait.
Do NOT claim v5058 inherits the v5057 200M performance: actuator
physics differs. Do not resume a v5057 checkpoint silently under
the v5058 model hash.

## One-source acceptance ladder — do not skip cheap gates

1. **Source and CPU check.** In an isolated v5058 worktree:
   `EMU_PHYSICAL_PROFILE=h40` and `PYTHONPATH=$PWD`;
   execute `python -m pytest -q tests/test_core_physics.py`.
   Check six actuator force limits and left/right mirrored
   knee torque. The special generated h40 robot XML is cached
   by immutable source-profile SHA outside Git.
2. **2048 GPU PPO smoke.** Run existing
   `scripts/smoke_gpu_carpet.py --device cuda --worlds 2048
   --frames 2 --ppo --full-update --benchmark-updates 3
   --prisms --solver-iterations 2 --solver-ls-iterations 4`.
   Require `rated-torque MJX PASS`, recycler/physics pass,
   actual PPO updates, finite state. This is NOT a gait test.
3. **Early diagnostic only; 20M is our standard minimum judgement.**
   Start ONE new, totally independent rated h40 seed 735 run,
   observe growth around 5M, but normally complete the full
   **20M**, which is cheap enough on our GPU to judge a new gait.
   Interrupt early only for proven physical divergence/invalid state;
   a novice still falling at 5M is not enough evidence to abandon it.
   For a failed 20M, investigate four-world neutral stance/torque,
   passive support, roll balance, fixed feet and fall reasons.
4. **20M cold start, same physical source.** A successful
   seed 735 20M foundation must show standing and sustained,
   real command tracking, not just apparent rapid fall velocity.
   Report `fall_rate`, `ep_len_mean`, `vx/cmd`,
   `servo_over_rated_frac=0`, `servo_peak_torque_limit_frac<=1`,
   `servo_speed_over_noload_frac`, spring/passive damping
   energy. Keep exact actor/model/source hashes.
5. **Grow same candidate** to 50–100M only if the earlier
   gates pass. Match held-out physical trials against current
   main actor: sustained standing, realistic forward/lateral
   speeds, ±10–12deg slopes, yaw, shoves, symmetric hip-roll,
   and honest rated-force output. Stairs not a promotion gate
   right now.
6. **Promote only with evidence.** Main currently has a
   trained v5050 actor and viewer; do not replace either
   with an untrained robot. When one rated h40 actor matches
   or beats the main actor on meaningful frozen exams,
   prepare a SINGLE clean review/promotion PR that removes
   the accumulated superseded stair/research scaffolding.
   Do NOT merge the entire cumulative v5053→v5058 chain
   unchecked; it includes unrelated experimental training
   changes. Follow the user's ~24 core-test discipline.

## One maintained research branch during stabilization

- **Use v5058 / PR #49** as the stabilization source.
- Leave v5057 / PR #48 as the overpowered positive reference.
- Leave v5059 / PR #50 and v5060 / PR #51 archived/unmerged.
  Their first carbon mechanics attempt is research evidence, not
  part of the immediate promotion candidate.
- Leave v5053–v5056 PRs unmerged; independent exams and
  useful diagnosis may be selectively retained when building
  the final clean promotion PR.
- No `v5061` branch just to change another parameter.
  Any necessary v5058 bugfix should be justified by a
  failing cheap check and use an existing test.

## WSL operator recipe (initial validation; NO training automatically)

```bash
cd ~/emu
git fetch origin
if [ ! -e "$HOME/emu-v5058-rated/.git" ]; then
  git worktree add --detach "$HOME/emu-v5058-rated" \
    origin/research/v5058-c018-rated-torque-hardcap
fi
cd ~/emu-v5058-rated
test -z "$(git status --porcelain)" || { echo "Dirty checkout: stop"; exit 1; }
git switch --detach origin/research/v5058-c018-rated-torque-hardcap
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD" EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -q tests/test_core_physics.py
python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo --full-update \
  --benchmark-updates 3 --prisms \
  --solver-iterations 2 --solver-ls-iterations 4
```

After cheap gates pass, launch an independent source-pinned
h40 rated 20M cold start with
`python scripts/v5057_physical_tournament.py
 --root "$HOME/emu-data/runs/v5058-h40-rated-cold"
 --profiles h40 --seeds 735 --execute`.
This recipe's built-in 20M goal cannot itself stop at 2–5M;
the operator should monitor early CSV/console and manually
interrupt if the robot does not begin standing.
It may resume safely with its built-in checkpoint rules.
Never reuse the failed carbon or overpowered v5057 run folder.

## Future: genuine rod-based hardware changes

Only **after** a stable h40 rated candidate is established:
recover or remeasure carbon shin flex; real 2mm rod and
10mm ball-foot design; accurate 40mm shin-tail *tension*
spring lever and upper-leg double-rod linkage; actual
servo current/thermal/load-speed and passive backdrive;
calibrated Hall sensor. Make one physical change at a
time on the established 40cm baseline and retain its
independent exam results.

## Hygiene

This is the current source-of-truth handover for the
40cm main-replacement effort. The v5059/v5060 handovers
are **historical context for the failed carbon run**,
not competing active roadmaps.

## Sensor/terrain candidate (NOT applied to v5058 20M baseline)

The intended ST VL53L8CX in 4×4 mode has a **45° horizontal × 45° vertical** detection field (65° diagonal); the current 360-input MJX ray contract uses **45° columns × 60° rows**. At 40 cm two lower actor rows look backwards. The browser now overlays its exact actor rays (magenta) and independent adjustable 45° square physical ray preview (green). Adjust mounting aim visually, then make any corrected actor/sensor contract a **new source-identified** cold start.

Similarly, four **1 m** tiles in a 2×2 window may give the taller robot more forward visibility than the frozen four **75 cm** tiles. It is a hypothesis, **not** a silent edit to the active rated-torque run: validate recycler coverage, genuine eight-prism mesh normals/collisions, independent signed-step exam, ToF visibility and GPU PPO first. Do not mix 1 m terrain and FOV changes into the source of the first rated-torque 20M baseline.

## 2026-11-09 — newer physical model, separate from the frozen baseline

Hardware correction: all **six 55g C018 motors are chassis-mounted**.
The previous v5058 physics distributed ~140g into thick rods and shifted
the 300g torso forward at h40 to make COM neutral, leaving hip mounts
visually disconnected. The newer model in draft PR #52 uses a single
460g central torso at x=+5mm, physically embracing hip pivots; two 2mm
carbon thigh rods and one 2mm split carbon shin per leg; and original
small hip-carrier and ball-foot masses. Calculated all-up ~508g at 40cm,
pending actual weighing and remote-knee-transmission validation. It
preserves old spring constants and rated torque but is a **new robot
model/COM/inertia identity**, incompatible with the frozen old-h40
checkpoints. The sensor preview defaults 70° down, 0mm forward, -30mm
below torso, zero sideways/yaw; PPO's old sensor contract is unchanged.
