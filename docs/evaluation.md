# Evaluation: evidence, not a reward graph

EMU uses *separate* mechanisms to train a policy, decide what tasks to practise, and judge whether the resulting actor can actually move. Those answers must not be conflated.

## Whose policy is being measured?

Each fixed-evaluation call receives a `PolicySnapshot`: the **post-update actor**, its associated observation mean/variance and its absolute environment step. That is the same actor/normaliser state published to the checkpoint at that step. Rollout diagnostics intentionally describe the **pre-update actor that collected experience**, so a rollout reward and a fixed-evaluation score at the same printed step may describe different policy versions.

`best.pkl` is promoted by the **current actor's individual `eval_broad_score`**. The trailing three-evaluation average (`eval_broad_tail3`) is graph/reporting telemetry only; it does not select a different actor. Legacy best-checkpoint metadata may carry an old rolling `best_eval_score`, so the loader prefers the saved `instant_eval_score` when present.

## Fixed evaluation

`emu/evaluation.py` defines deterministic, read-only command panels: core, dynamic, combo, resilience and growth. It measures survival, command errors, posture/height and stepping/support diagnostics.

The broad score weights survival and penalises tracking/posture error. This is a **longitudinal regression ruler**, not a scientifically proven scalar measure of useful walking. In particular, `robust40` changes command difficulty, *not* the physical robot's mass/friction or contact geometry. Do not describe it as held-out hardware robustness.

## Ladder screens and promotion

`emu/screening.py` evaluates finished checkpoints against the **same** task cohorts, robot assumptions, pitch command limits and a separate gentle gravity-equivalent slope setting. Screening uses saved checkpoint policy weights and observation normalisers; it does not teach PPO or the capability field.

`emu/branch_lab.py` requires real, finite screen results and applies minimum hard-survival and maximum pitch/height/slope-height constraints. The scorecard also reports motion error, stepping evidence and support. **Those last quantities are currently descriptive**; no calibrated minimum stepping threshold has been accepted. Consequently an auto-promoted arm is a *provisional experimental finalist*, not independently certified to walk well.

Scores are attached to each checkpoint's SHA-256 identity. A changed checkpoint, evaluation horizon or cohort count invalidates cached screens and promotion records.

## Capability field and independent testing

The in-training capability field maps nodes, transitions and recovery evidence to propose **practice**, inside curriculum limits. Its observations are sampling-biased by the tasks it helps select; they cannot replace independent held-out testing. The held-out capability cloud checks calibration and recovery on independent commands, without feeding its answers into the field. The more expensive atlas probes physical-unit limits after a milestone.

The public browser viewer is *not* any of these examiners. It runs a nominal MuJoCo/WebAssembly simulation and can show compelling motion without establishing task reliability.

## What remains to prove

- Survival must be accompanied by **actual commanded stepping and movement**. Calibrate motion/step eligibility thresholds from real matched-screen distributions, rather than selecting arbitrary values that could reject all early learners.
- Physical robustness requires changed-physics, contact-geometry and multi-seed tests. Gravity-equivalent slope perturbations do not prove walking on physical ramps.
- The actor's Python↔browser deterministic inference has a numerical parity test, but MJX versus WASM **observation and physics parity** is a separate open task.
- Long-horizon crouching requires evaluations beyond 100M *per policy*, not just a 72M aggregate experiment budget.

[Training lifecycle](training.md) · [Experiment ladder](tournament.md) · [Validation gates](validation.md)
