# EMU v5051 — historical 960M PPO campaign / capability standard v2

> **Archived experiment, not an active launcher.** The v5051 runner has been
> removed from the current `scripts/` directory. The historical commands below
> are for reproducing the original study **only after checking out its pinned
> source**: [v5051 campaign runner](https://github.com/mdt733/emu/blob/6a3c9cc971936d12f95822a6561cc244abf896e7/scripts/v5051_train_campaign.py).
> New work uses `emu train`, not an old tournament wrapper.

**REAL PPO training, not 44 frozen diagnostic panels.** A clean 0→60M
cold foundation is saved as a complete checkpoint (actor, critic, optimizer,
normalizer and curriculum). Fifteen arms independently resume precisely that
parent and train to **120M** each. Total **60 + 15×60 = 960M world-steps**,
with standard 2048 CUDA MJX worlds and six physical servos. Nothing in this
branch changes released `main`.

## Why the capability exam was rebuilt before restarting

The previous fixed evaluator had **ten overlapping locomotion panels and five
different 32-world JAX environments**. Compilation often dominated the
training interruption. The source now has **FOUR fixed retention panels on
TWO worlds**, with exactly the same 350-frame summary procedure and only
one recurring exam at the configured 60M interval. The four are:

| Guard | Question and real outcome |
|---|---|
| `core` | Does basic supported walking still function? Survival, velocity errors, posture and foot support |
| `dynamic80` | Can EMU track changing commands? Survival and velocity/yaw error |
| `combo70` | Can it steer laterally while moving? Simultaneous command errors and survival in combo task world |
| `resilience75` | Does it survive and recover from disturbances? Survival and support/attitude measures |

These checks **do not test actual stairs**. Their weighted score chooses
`best.pkl` as a *general locomotion retention* convenience, not an
autonomous stair-champion ranking. The final complete `final.pkl` is retained
independently. No hidden slope or step success is imputed from flat-world
survival.

After PPO training, every actor undergoes the **same SIX frozen physical
scenarios**, defined once in `PHYSICAL_PANELS` in the campaign runner:

| Scenario | Command | Physical question |
|---|---|---|
| `standing` | vx=0 | Real footfall rate, support, 10–13s drift and survival |
| `slow_walk` | vx=0.10m/s | Slow gait and non-prescribed cadence |
| `fast_walk` | vx=0.70m/s | Fast gait, stride and survival retention |
| `slope` | vx=0.12m/s, 8° requested sampling envelope | Actual encountered grade, forward travel and survival |
| `step_3cm` | vx=0.10m/s, 3cm nominal lip envelope | Measured signed foot-supported rise; complete both-feet plus 50cm controlled progress |
| `step_4cm` | vx=0.10m/s, 4cm nominal lip envelope | Harder contact exposure, exact same real mission rules |

**Never confuse a requested 3/4cm generator envelope with an actual 3/4cm
wall!** The report counts measured signed heights, upper-foot support, and
controlled post-contact missions separately. A two-foot contact is not a
measured support wrench and step collision does not prove toe-over-riser
clearance. Exact-height actor wall certification is not yet included.

### Exam time and scientific tradeoff

- Old physical screen: **4 conditions × 2 held-out seeds × 16 worlds =
  128 evaluator worlds per arm** (8 costly independent exam executions).
- New initial screen: **6 conditions × 1 held-out seed × 8 worlds =
  48 evaluator worlds per arm** (6 exam executions).
- All 15 trained arms: **90 frozen exam calls** versus 120 previously,
  using 62.5% fewer exam world-rollouts; fixed recurrent exam compilation
  is reduced from five world setups to two.
- `--exam-seeds 1701,1702` adds a second matched held-out seed to
  **selected finalists and the unchanged control**, without retraining.
  That is 12 calls per selected arm including previously completed ones.
  The study's exam world count and scenario definitions are locked once
  exams begin, so comparisons cannot silently switch protocols.
- This rapid screen is **one PPO training seed** and **one held-out
  physics seed**, not a confidence interval or reliable champion result.
  Repeat finalists with independent PPO seeds before promotion.

### Training arms (each 60→120M)

| Arm | Intervention |
|---|---|
| `00_control` | Unchanged continuation |
| `01_gate_zero` | Relax stepping-activity gate only near zero request |
| `02_gate_off` | Remove activity gate everywhere |
| `03_phase_slow` | Clock 25% near stationary command |
| `04_phase_pause` | Clock fully paused at zero command |
| `05_phase_reward_off` | Remove phase synchronisation reward component |
| `06_stillness` | 15% real zero-command episode births |
| `07_crouch` | Add 3cm STEP-only bounded crouch reward allowance |
| `08_supported_transfer` | Reward real foothold, supported crossing, bilateral top support and 50cm heading-controlled follow-through |
| `09_sensor_teacher` | Extra authentic foothold bonus conditioned on valid nearby ToF return |
| `10_frontier_1mm` | 1mm ordinary frontier expansion and .5mm retreat (Academy retains existing .5mm increment) |
| `11_academy_heavy` | Increase Academy exposure to 65% |
| `12_long_stride` | Small reward for useful distance per lift with tracking/slip gates |
| `13_freedom_combo` | Gate + slower phase + phase shaping off + stillness + crouch |
| `14_step_combo` | Verified step/progress + sensing reward + Academy exposure + precise frontier |

The interaction arms cannot identify causality without independent single
factor results. The 360-input actor/6-output network and 4×4 ToF sensory
meaning remain unchanged. This is not teacher-student privileged-critic PPO.

## Reproduction (historical source only)

The 960M campaign is complete; its one-off launch commands and obsolete
virtualenv paths have been removed from current documentation. The original
[study runner](https://github.com/mdt733/emu/blob/6a3c9cc971936d12f95822a6561cc244abf896e7/scripts/v5051_train_campaign.py)
remains available at its immutable Git commit for forensic reproduction.
Recorded data live at `~/emu-data/runs/v5051-capability-v2-s735/`,
including exam protocols and independent result artifacts.

**Release rule:** sustained real step capability is evaluated independently
of PPO reward and broad survival retention. No arm earns promotion solely
from surviving a nominal 3cm procedural panel, touching a higher tile, or
improving the four-panel retention score. Keep the compact physical
examiner; remove this one-off tournament runner after the study's artifacts
are securely archived.
