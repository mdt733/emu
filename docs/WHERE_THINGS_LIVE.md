# WHERE THINGS LIVE — early tinyEmu refactor branch

This is a map of **current branch code**, not a claim that the Pass 9 runtime was transplanted. Source is descended from accepted main commit `40c5d0ce543ebbbc3690591695afddc18def8232`. The separate Pass 9 archive is a reference, not a checkpoint-compatible replacement.

## When you want to change an objective

1. `emu/robot.xml` defines physical joints, actuator limits, passive springs, and body geometry. The servo model is an approximation: there is no calibrated electrical current, battery wattage or winding-temperature measurement.
2. `emu/environment.py::_metrics` reads MuJoCo/JAX physics and task commands. Important caveat: `support` is principally a foot-height proxy, and `step_evidence` is foot lifting, not a verified stride.
3. `emu/scoring.py::training_reward` computes tracking, posture, contact, anti-slip, smoothness and joint-position safety *shaping factors*, while `weighted_training_terms` owns all eight **positive reward payments**. `scoring.apply_early_fall_penalty` owns the independent early-fall cost; `environment.py::_step_impl` merely invokes these functions. Do not confuse payment names with observed electrical motor effort.
4. `emu/ppo.py::rollout_once` hands `out.reward` to PPO in the trajectory's `reward` field; `compute_gae` then builds advantages. PPO does not independently read spring torque or heating.
5. `emu/environment.py::_step_impl` exposes `reward_weighted_*` telemetry; `emu/rollout_metrics.py::summarize_rollout` reduces per-frame values, including a negative `reward_weighted_fall_penalty`; `emu/training.py::make_metrics_row` writes the rollout columns. The columns use **mean reward contributions per observed frame**, not individual episode totals.
6. `emu/curriculum.py::RegimeGovernor.update` uses survival/tracking and posture signals to move task difficulty. **There is no thermal-efficiency curriculum dial.**
7. `emu/evaluation.py` owns fixed held-out panels and a survival-weighted broad checkpoint score. `emu/screening.py` runs common held-out screening; `emu/branch_lab.py::score_row` and `promotion_choices` decide branch eligibility/ranking. Current stepping/support are reported but are **not hard promotion requirements**, and motor energy is not a promotion input.
8. `tests/test_learning.py` checks original payment arithmetic, fall-penalty equivalence, both posture profiles, JAX compilation and the reducer. The actual environment `_reward` method has been removed. Whole-MJX training equivalence is still a separate GPU acceptance gate.

## What the current objective actually says

| Payment | Weight | Input | Caveat |
|---|---:|---|---|
| alive | 0.58 | not terminated | Survival alone does not prove walking |
| locomotion | 1.55 | tracking × posture × contact × locomotion maturity | Command tracking and spawned initial velocity can fool simple motion metrics |
| posture | 0.68 | height/attitude/cone × step-activity gate | Height and cone are proxies |
| contact | 0.50 | feet-only proxy × step-activity gate | Not ground reaction force |
| anti_slip | 0.18 | near-ground foot velocity proxy | Depends on foot-position differencing |
| clearance | 0.16 | foot clearance/phase shaping | A lift does not demonstrate a completed step |
| smoothness | 0.12 | action-delta norm | **Not** joules or electrical power |
| joint_safety | 0.17 | position-limit and scissor pressure | **Not** sustained motor overload |

A true early fall incurs a separate negative penalty defined in `scoring.apply_early_fall_penalty` and triggered by `environment._step_impl`. **No calibrated torque, mechanical work, servo heat or spring-energy objective currently exists.** Read-only measurement is the next prerequisite; only then consider shaping rewards.

## Read-only actuator and spring diagnostics (new)

`environment.py::_metrics` also reads *actual modeled* MJX actuator forces and joint velocities. Its `servo_effort_metrics` helper gives five **non-reward, non-actor, non-curriculum** indicators, averaged across rollout frames in `rollout_metrics.py`:

| Logged column | Definition | What it does NOT mean |
|---|---|---|
| `servo_shaft_abs_power_w` | Sum of `abs(actuator_force × joint_qvel)` over six joints | Not DC battery wattage or total motor losses |
| `servo_shaft_drive_power_w` | Sum of positive shaft power | Not an electrical power estimate |
| `servo_shaft_brake_power_w` | Sum of magnitude of negative shaft power | **Not** recovered regenerative energy |
| `servo_over_rated_frac` | Fraction of the six modeled torque magnitudes over 0.638 N·m | Not an overheating diagnosis; continuous rating is a rough reference |
| `servo_peak_torque_limit_frac` | Maximum torque demand relative to **each world's actual model** force ceiling | Not a thermal limit or cycle duty rating |

Note the stationary-stall counterexample: `torque × speed == 0` even when the motor is heavily loaded and consuming electrical power. **Do not use this mechanical-power proxy alone as an efficiency reward.** Damping, Coulomb friction and electrical drive losses are not captured by `actuator_force × joint_qvel`.

The same `_metrics` path computes `spring_potential_energy_j` from the six driven hinge springs and two passive shin-flex springs as `½ × stiffness × (q − qpos_spring)²`; `shin_spring_energy_j` isolates the latter two. The reference positions and stiffnesses come from the *loaded MuJoCo model*, not hand-copied tuning constants. This is stored **potential energy**, not delivered spring work or guaranteed assistance.

Both quantities use model truth for evaluation and diagnostics, not deployable sensor inputs. `tests/test_physics.py` covers known numerical cases, reference offsets, high-torque stationary joints, JAX compilation and CSV reduction. Full real-MJX runtime, domain-variation and WSL GPU acceptance are still required.

## Run one experiment (new in this branch)

`emu train --run-dir ~/robot_lab/emu_runs/tiny_trial --steps 256 --envs 8 --profile foundation --plan` prints the exact configuration **without creating files or starting PPO**. Remove `--plan` only on a clean committed source checkout with a unique run directory and an available GPU. Training itself still occurs in `emu/training.py` using the existing `run_support.run_training`; this wrapper is **not** a second trainer.

Use `--resume auto` to continue a partial checkpoint in the **same run directory with the same source commit and full experiment config**. For a new run descended from a compatible complete checkpoint, choose a **new** directory and pass `--resume /absolute/path/to/final.pkl`; parent and child must share the source commit, and the parent cannot already be beyond the requested target.

`--profile foundation` uses the accepted v5015 foundation physics/curriculum defaults (shadow field mode); `--profile mature` uses the accepted mature graph profile. Neither removes the historical field/teacher implementation yet. Targets are absolute environment steps, rounded up to one complete PPO update. `--plan` is a configuration print, **not** a GPU readiness check.

A first startup that fails **before** CUDA/training initialises may leave only `training_config.json` and `console.log`. Retrying the identical command in that directory is allowed; any unexpected files, missing checkpoint metadata, dirty source or different run configuration still fail closed. Do not delete unknown checkpoint files to make the guard pass.

The existing `emu ladder` remains unchanged while we test the simpler one-run interface. Branch GPU acceptance and browser loading remain mandatory before merging.

## Contract and checkpoints

The deployable actor has 55 ordered measurements × four frames = **220 inputs**, and **six motor actions**. `emu/contracts.py`, `emu/ppo.py`, `emu/policy_export.py` and `viewer/src/policy.js` must agree. The candidate Pass 9 checkpoint schema 9 is **not directly compatible** with the present main-derived schema 7; use new run roots, never edit/reinterpret old training snapshots. The public `showcase/champion.emupolicy` stays untouched.

## Acceptance status

- Earlier main acceptance on real WSL/GPU is historical evidence for **the parent commit only**.
- This branch changes reward assembly/telemetry, so **new main-branch GPU smoke, fresh-root train/resume, and viewer checks are required** before merge.
- Numerical equivalence of eight *positive payments* does not alone establish full physics/reward equivalence; exercise falls/timeouts and held-out scoring too.

## Shareable experiment reporting

`emu report --root "$RUNROOT"` writes `results.md`,
`results_data.json` and `results_bundle.zip`. The single ZIP contains
all branch metrics, complete held-out CSVs/JSON, configuration, provenance and
rankings. No raw checkpoint binaries are bundled. Reports can be regenerated
offline from a new reporting-only checkout using `python3 emu/results.py --root ...`.
Do not change an in-progress trainer's pinned source revision just to run
the reporter. Scores and motor telemetry are labelled by evidence type; no
measured shaft power should be mistaken for electrical battery draw.

## v5016 directional experiments (draft, not PPO training)

The new independent **course examiner** is in `emu/directional.py`; `emu/gait_exam.py` uses physical trajectory/contact evidence without changing learning rewards. `emu/command_exams.py` declares external operator-command schedules, `emu/directional_exam.py` runs read-only saved-policy exams, and `emu/v5016_batch.py` assembles matched OFF/ON comparisons and results bundles. `emu/ground_plane.py` is a read-only candidate for 3-range/IMU slope estimation; `emu/observation_versions.py` declares the **untrained** 244-input candidate. Keep these draft pieces distinct from the production 220-input actor.

For developers: **`bash scripts/check.sh`** covers 15 representative contracts; **`bash scripts/check.sh all`** checks the entire Python suite and browser. The complete list is intentionally kept for CI; it is not the normal editing workflow. See [the testing guide](TESTING.md).
