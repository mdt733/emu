# EMU v5051 — real PPO gait and step experiment (60M shared parent → 120M forks)

**This is a training experiment.** Unlike the frozen-policy 44-panel diagnostic suite, `scripts/v5051_train_campaign.py` launches the existing full PPO trainer and creates **15 different learnable policies**. Each arm receives 60 million *additional* world-steps, not merely more examinations.

All 15 arms load the exact same source-pinned **60M `final.pkl`** with actor, critic, optimizer, observation normaliser, curriculum, learning-health and capability-field state. Fresh physical episode state/RNG is the normal restart contract. No branch replaces `main`. No rollout, critic or actuator dimensions change.

## Actual GPU budget

| Block | Distinct PPO policies | Newly trained world-steps |
|---|---:|---:|
| Cold shared foundation (0 → 60M) | 1 | 60M |
| Unchanged 60 → 120M continuation | 1 | 60M |
| 12 individually targeted treatment continuations | 12 | 720M |
| Two interaction/bundle continuations | 2 | 120M |
| **Total** | **16** | **960M** |

These are **world-steps across 2048 batched worlds**, not time steps for one robot. All arm targets are absolute steps (`--steps 120000000`) and the runner rejects a parent checkpoint that is not exactly at the 60M update boundary.

### Frozen common training conditions

- Same **360 history inputs**, **six outputs**, PPO hyperparameters, real 8-prism MJX world, underlying servo/spring XML, source SHA, fixed 2048 CUDA worlds and training seed 735.
- 20M gentle foundation then capability growth, `main-mature` growth recipe, Newton/LS **2/4**, coherent-careful terrain lessons, athletic rehearsal 15%, dynamic skill retention 15%, temporal velocity reward share 25%.
- Academy starts introducing lessons at **60M**, grows from 10% at 75M toward 50% by 90M in baseline/single-treatment arms. Academy height progress remains *evidence-driven*, never increased simply because more time elapsed.
- Existing true-contact supported-crossing rewards remain in every treatment. New positive rewards are opt-in; all default knobs preserve released reward economics.

## Explicit registered treatment arms

| Arm | Independent 60–120M training treatment |
|---|---|
| `00_control` | Unchanged continuation from same 60M parent |
| `01_gate_zero` | Remove activity gate only at near-zero vx/vy/yaw requests |
| `02_gate_off` | Remove stepping-activity gate entirely |
| `03_phase_slow` | Phase advances at 25% speed near zero command; ramps to normal by 0.1 m/s |
| `04_phase_pause` | Phase stops at zero command; runs normally at commanded gait speeds |
| `05_phase_reward_off` | No phase-synchronisation component in organic lift reward |
| `06_stillness` | Reserve 15% true zero-command births, with zero-command gate exemption |
| `07_crouch` | Add 3cm safe crouch margin in STEP-role posture/locomotion *rewards* only |
| `08_supported_transfer` | Real foothold + supported crossing + 5-frame bilateral top support, then **50cm** more heading-controlled progress |
| `09_sensor_teacher` | Additional sparse real-foot placement reward only when 4x4 ToF has a valid close-range return |
| `10_frontier_1mm` | Fixed 1mm ordinary step growth, .5mm retreat, 12 qualifying attempts. **Academy remains 0.5mm**, so this tests the ordinary frontier only |
| `11_academy_heavy` | Bring 65% Academy birth allocation in sooner, retaining non-Academy specialists |
| `12_long_stride` | Bounded successful forward-progress-per-lift reward, with command/support/slip gates. This is a proxy, not a physically counted stride |
| `13_freedom_combo` | Near-zero gate relaxation + slower phase + no phase-sync pay + stillness births + crouch margin |
| `14_step_combo` | Verified foothold, bilateral upper support **and 50cm follow-through** + ToF-conditioned teacher + denser Academy + precise ordinary frontier |

The follow-through reward is **one-time per confirmed lip**; it cannot pay for tippy-taps in place, backtracking or repeatedly touching the same foot. A positive training reward still is not a held-out physics examination.\n\nThe combination arms are **exploratory**, not causal attributions. No phase-input deletion is attempted from this parent because removing phase dimensions would violate the 360-input checkpoint and require a separate cold-start matched control/treatment pair. The phase clock itself is changed only through the environment's training and frozen-exam configuration; exported deployment/viewer parity must be checked before release.

The ToF-conditioned `09_sensor_teacher` arm is a **reward-side teacher hypothesis**, not an implementation of asymmetric-critic PPO or a new hardware sensor. It only pays when an authentic foothold is seen *and* the delayed actor-visible 4x4 sensor supplies a valid close return. Success would warrant more detailed sensor causal studies.

## One Windows/WSL machine — real foreground launcher

Run after this branch's source checks pass and when the GPU is available. Keep the same checkout SHA once the 60M parent exists; a new source commit requires a new cold source lineage or explicit validated cross-source fork.

```bash
cd ~/emu
git fetch origin
git switch research/v5051-gait-autopsy
git pull --ff-only origin research/v5051-gait-autopsy
source .venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

git status --short  # must be EMPTY
git rev-parse HEAD

# CPU contracts + true PPO argument parsing for all 15 arms.
python -m pytest -q tests/test_v5051_training_science.py

# Prints 60M parent and 15 actual PPO arm commands. Trains nothing.
python -m scripts.v5051_train_campaign --stage plan

# Step A: 60M real cold-start PPO; saves final.pkl.
python -m scripts.v5051_train_campaign --stage foundation

# Step B: 15 genuine independent 60M PPO continuations, sequentially.
python -m scripts.v5051_train_campaign --stage arms

# Step C: independent 16-world frozen physical exams per trained actor.
python -m scripts.v5051_train_campaign --stage exams

# Step D: tabulated study findings, not a saturated scalar rank.
python -m scripts.v5051_train_campaign --stage report
```

For a single foreground 960M campaign *with exams at the end*:

```bash
python -m scripts.v5051_train_campaign --stage all
```

For staged selection, `--only 00_control,01_gate_zero,08_supported_transfer` trains or examines only these three arms, always using the same immutable parent. Re-running a completed stage verifies the final checkpoint and skips redundant updates; an interrupted training arm resumes its own `latest.pkl`, not the shared parent. A corrupted or mismatched study contract, dirty checkout, different source SHA or missing complete 60M checkpoint **halts safely**.

Results are under `~/emu-data/runs/v5051-science-s735/`:

```text
science_contract.json
parent_identity.json         # verified full SHA256, source and age
foundation/
    training_config.json
    run_meta.json
    checkpoints/final.pkl
arms/
    00_control/
        training_config.json
        run_meta.json
        metrics.csv
        checkpoints/final.pkl
    01_gate_zero/ ...
exports/
    00_control.emupolicy ...
exams/
    00_control/<held-out panel>/prism_exam.json ...
SCIENCE_RESULTS.md
SCIENCE_RESULTS.json
```

## Promotion criteria — genuine locomotion, not a reward-tournament winner

The independent exams use held-out procedural prism terrain, and report the **signed measured contact height** rather than treating a nominal 3cm frontier as an actual 3cm wall. The permanent post-lip autopsy separately checks simultaneous bilateral actual top contact and 50cm extra heading-controlled progress. Results also compare zero-command footfalls/drift, 0.70m/s achieved forward speed and first-life survival.

A candidate must improve physically supported uphill difficulty and *completed post-lip mission rate* while retaining a controllable fast gait, commanded heading, resilience, and sane mechanical drive work. An arm with impressive training return or a single upper-foot touch does not automatically win. The 15-way stage is **one PPO seed per arm**: retest finalists and the control with 2–3 independent PPO seeds for reliable evidence, not just two held-out *evaluation* seeds.

**Limits:** This tournament does not implement a cold-start phase-deletion pair, exact-height *frozen-policy* wall missions, true support-wrench/CoM load-transfer proof, or privileged-critic teacher/student. Its mechanical fixed-height CPU riser screens remain a separate diagnostic. An absence of completed missions after 120M is a real negative result requiring redesigned teaching; it should not be relabelled progress.
