# EMU v5016 — Pass 2: real-physics directional examination

**Branch:** `feature/v5016-directional`, stacked above the *draft* tinyEmu refactor.
**Acceptance status:** code implemented; real user-GPU acceptance required before claiming a competent slope walker.
**Protected:** `main`, original 60M showcase actor, 20M shared foundation, four 80M checkpoints, policy observation contracts and production Pages.

## What changed — actual code, not another roadmap

| Path | Responsibility | What it does NOT do |
|---|---|---|
| `emu/ppo.py` | Exposes a **read-only `evaluate_trace`** alongside the accepted `evaluate_summary`: JAX-scanned pre-step physics state, actual issued commands, floor/foot contact pairs, first termination flags | Does not touch `train_update`, rewards, actor weights, normalizer, curriculum, checkpoint state or legacy `evaluate_summary` |
| `emu/gait_exam.py` | Converts uninterrupted episode prefixes into signed world-course and candidate stride verdicts; counts alternating floor-foot touchdown pairs with airborne clearance, foot relocation and body displacement | Not a learned reward. Does **not** use foot lift as proof of real locomotion or confuse a reset teleport with movement |
| `emu/directional_exam.py` | Explicit legacy-checkpoint read-only command: exact 0–5° tilted-gravity tests on shared plane, deterministic command probes, output SHA/provenance and per-world MD/JSON | No checkpoint writes or edits, no automatic campaign winner promotion; **no ramp geometry or heading P yet** |
| `emu/cli.py` | Public `emu directional-exam` dispatch with bounded inputs and required separate output | Does not enable the experimental 244-input actor |
| `tests/test_v5016_exam.py` | Pure reset/footfall/numeric tests and small bounded real-MJX CPU trace smoke (when optional physics dependencies available) | Not proof of the user's real GPU or finished gait |

### Why episodes are truncated before auto-reset

Existing `env.step` automatically replaces a failed world with a newborn one. The new `evaluate_trace` records **world XY, physical heading, two feet, actual floor–foot contact geometry pairs and command targets BEFORE each `env.step`**, followed by its done/fell flags. On the host, first termination truncates the episode; no post-reset position can count as uphill progress. A timeout before the requested horizon is also an incomplete course test, not a pass.

### Why contact-based strides are still estimates

The new count needs a contact-pair event, >=3 airborne frames, >=4 mm clearance above the flat plane, >=12 mm foot relocation, >=25 mm torso displacement, then an alternating other-foot touchdown with body translation between. These are conservative **contact-geometry estimates**; MuJoCo pair distance within 2 mm doesn't prove nonzero normal reaction force, and neither a fake foot-lift score nor positive center-of-mass velocity alone qualifies. Keep `course_pass` and `walking_pass` separate. A later pass should confirm contact impulse and stance/slip robustness on actual rollouts.

### Run after branch CPU CI passes — read-only from separate checkout

Use the checkout created for v5016; **do not change the source checkout already running PPO**. The same new source can examine older 220-input policies because the examination kernel still uses the unchanged v5015 actor path. There is deliberately no silent 220→244 conversion.

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-v5016-exam origin/feature/v5016-directional
cd ~/emu-v5016-exam
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate

RUNROOT="$HOME/robot_lab/emu_runs/tiny_fourway_001"
EXAMS="$HOME/robot_lab/emu_exams/v5016_001"

# Minimal real MJX compile/trace check before any longer examination:
emu directional-exam --run-dir "$RUNROOT/round1/baseline" \
  --output "$EXAMS/baseline_flat" --angle 0 --worlds 2 --horizon 40 --device cuda

# If above succeeds, run **small slopes** with a distinct output per case:
emu directional-exam --run-dir "$RUNROOT/round1/baseline" \
  --output "$EXAMS/baseline_up025" --angle .25 --direction uphill \
  --worlds 8 --horizon 350 --device cuda
emu directional-exam --run-dir "$RUNROOT/round1/baseline" \
  --output "$EXAMS/baseline_down025" --angle .25 --direction downhill \
  --worlds 8 --horizon 350 --device cuda
```

Each exam writes `directional_exam.md` and `directional_exam.json` to its **own new output directory**; it refuses to overwrite an existing nonempty destination. Extend to 0.5°, then 1°, and other arms **only after inspecting initial results**. The published 60M `.emupolicy` is browser-only; this Python examiner requires a validated schema-7 training checkpoint with optimizer/policy tensors.

**Current exam constraints:**
- 2–64 worlds, 30–700 frames, same exact gravity vector for all worlds; commanded forward speed defaults to 0.1 m/s.
- All exams in this first pass issue steady operator commands; heading P (ON/OFF) and terrain sensor observation remain **future integration**. Turning anti-cheat is independently scored in the existing pure examiner.
- Model gravity is rotated; geometrical floor is flat. These are infinite constant-incline equivalent tests, **not** ramp approaches, abrupt terrain transitions or a directly measured incline.
- Spawn state has its legacy velocity already blended toward command; the first 25 frames are excluded from course qualification.
- Simulator ground heading uses its true geometry only in the **evaluation oracle**; it is not secretly fed to the old deployable actor.

## Test gates and remaining work

1. CPU unit suite and browser CI for this exact branch commit; inspect actual test skips/failures, not inherited baseline labels.
2. Run smallest real WSL MJX trace. Check that floor-contact arrays are populated sensibly, full-horizon survival/termination codes, world movement/sign, and no reset teleports.
3. Compare baseline vs shadow/slope/pitch15 at matched 0°, 0.25°, 0.5°, 1° uphill/downhill/cross-gradient. Add a **batch report generator** so results can join `results.md`, with seed/checkpoint SHA verification.
4. Strengthen gait proof with support normal forces/stance slip and trace snapshots. Benchmark bad controls against true physics, not just synthetic tracks.
5. Compare optional heading P on/off after real command connection; instrument 1s/3s yaw errors.
6. Only then train the proposed **244-input v5016 actor** with ground and heading measurements properly wired through env→PPO→checkpoints→export→viewer. This pass does not claim it exists yet.

This is a **measurement instrument**, not a new training version release. No 80M+ continuation is justified until the examiner demonstrates what currently fails.
