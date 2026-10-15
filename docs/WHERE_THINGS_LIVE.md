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
8. `tests/test_reward_ledger.py` checks original payment arithmetic, fall-penalty equivalence, both posture profiles, JAX compilation and the reducer. The actual environment `_reward` method has been removed. Whole-MJX training equivalence is still a separate GPU acceptance gate.

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
