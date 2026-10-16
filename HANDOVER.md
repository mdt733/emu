# EMU — Living handover

**Read this first if a chat reaches its message limit.** Update this same file after every meaningful pass. Do not create pass-specific handover files.

## The goal

Turn EMU into a robot-learning system that reads like a book. The scientific goal is to discover good ways to *teach* a six-servo dynamic biped to walk—not just to optimise one robot's reward. Complicated maths is fine; obscure ownership and duplicated mechanisms are not.

**Canonical references:** [Architecture](docs/architecture.md), [One simplification checklist](docs/SIMPLIFICATION.md), [20 tests](docs/TESTING.md).

## Seven layers

1. **Universe**: robot.xml + emu/environment.py — MuJoCo physical model, reset, stepping, measured observations and falls.
2. **Lessons and curriculum**: emu/tasks.py generates per-world commands/posture/transition timing; emu/curriculum.py adjusts allowed practice based on training evidence.
3. **Rewards**: emu/rewards.py judges each control step. Scalar teaching reward and component ledger are not proof of walking.
4. **Learner**: emu/ppo.py owns networks, experience rollouts and gradient updates.
5. **Training**: emu/training.py connects learner, worlds, curriculum, diagnostics, recovery and checkpoints.
6. **Meta-experiments**: emu/ladder_recipe.py + emu/branch_lab.py compare a shared ~20M foundation and matched teaching branches.
7. **Understanding**: emu/rollout_metrics.py, emu/evaluation.py, emu/results.py produce independent evidence; emu/policy_export.py and emu/viewer_src/ inspect a frozen brain.

## Branches, safety and provenance

- Active cleanup: **refactor/emu-readable**; [draft PR #5](https://github.com/mdt733/emu/pull/5). Started from v5016 experimental commit 18b566d.
- v5016 directional and heading-hold experiments remain on **feature/v5016-directional**, [draft PR #4](https://github.com/mdt733/emu/pull/4). Do not switch a running experiment's checkout.
- Protected main baseline: **40c5d0ce543ebbbc3690591695afddc18def8232**. Do not merge or alter original training roots without consent.
- Protected public champion: showcase/champion.emupolicy, the 60M model. Four original 80M branch checkpoints (baseline, shadow, slope, pitch15) and their 20M common foundation remain untouched.
- Existing trained actors are **220 inputs = 55 measurements x four frames, six outputs**. Proposed 244-input observation model is NOT yet trained, exported or viewer-compatible. Old schema-7 checkpoints must not be silently migrated.
- Source hashes are part of reproducibility. Test source changes on a fresh, isolated GPU run root; do not resume old long PPO lineages from a changed checkout.
- User wants 20 meaningful tests, simple names, no test explosion, infrequent grouped commits, and no pass-N status documents.

## Completed work

- **Before Pass A:** trimmed 226+ checks down to exactly **16 Python + 4 JavaScript** in five files; baseline CI passed at 18b566d.
- **Pass A:** canonical seven-layer architecture and a single checklist; renamed training reward module scoring.py to rewards.py without adjusting reward calculations. Source commit d2abc8b passed [CI](https://github.com/mdt733/emu/actions/runs/38834067868) (16 Python, four JS, viewer build). Docs-only ledger updated at a173582.
- **Pass B1 (verified):** moved lesson generation out of environment.py into **emu/tasks.py**. The unchanged sampling methods live in a stateless TaskSampler inherited by EmuMJXEnv; TASK_FIELDS/T and related task constants also moved. Environment remains owner of resets, physical stepping, observations and measured events. The sampler uses precisely the same JAX context and RNG streams; no second copy of lesson state.
- **Pass B1 CI:** [passed](https://github.com/mdt733/emu/actions/runs/38835597285) at commit **3102062**: 16 Python and four browser tests, production viewer build and showcase packaging verified. Python logged 24 existing nonfatal MuJoCo deprecation warnings. This is CPU CI, NOT a fresh CUDA learning acceptance.
- **Pass C (CI verified, commit `39651e4`):** Reward owner now puts the overall `training_reward` calculation first, followed by three coherent ingredients and the unchanged weighted ledger. `REWARD_ONLY_METRICS` and `PAYMENT_NAMES` belong to `rewards.py`, consumed by the environment and host-side summaries. `environment._metrics` explicitly distinguishes physical facts, servo/spring telemetry and reward-only proxies. Existing `test_training_reward_and_terminal_payment_jit_compile_in_jax` was strengthened (without increasing test count) to prove that adding nominal servo power/spring energy telemetry does not change PPO payment. **Actual servo power remains diagnostic-only; smoothness and joint safety are the energy-adjacent teaching terms.** [CI 38837791165](https://github.com/mdt733/emu/actions/runs/38837791165): **16 Python, 4 browser passed; viewer built**; 24 existing nonfatal MuJoCo warnings. Fresh CUDA learning/exam not yet run.
- **Pass B2 (software CI verified at `df82901`):** `_step_impl` now composes physical frame → next lesson → previous-command reward → next observation/reset, instead of mixing all stages in one ~284-line method; `_simulate_physics` retains actual motor/push/substep/nonfinite handling. `tasks._advance_commands` owns command-ramp timing and samples. `_reset_with_keys` is now a ~40-line reset orchestrator using `_birth_physics` for initial pose/velocity. Two named JAX-compatible return records clarify data without introducing duplicate mutable state. **20-test limit unchanged.** [CI run 38836739692](https://github.com/mdt733/emu/actions/runs/38836739692): **16 Python + 4 browser passed**, viewer built; 24 existing nonfatal MuJoCo warnings. **Fresh CUDA lifecycle still not run.**

## Physical path and invariants

1. EmuMJXEnv loads robot.xml and builds batched JAX/MJX model variations once per run.
2. Reset samples an independent birth task (tasks.py), then places each robot in a dynamic initial pose/velocity, initialises target-ramp state and history observations.
3. Step delivers the PREVIOUS actuator target, applies disturbance forces and integrates MuJoCo for the configured substeps; it rolls back non-finite physics states and terminates them.
4. Optional TRANSITION lessons sample a new target and advance the command ramp.
5. Physical metrics are evaluated and rewards.py scores **the command the actor saw before it acted**; the advanced command is for the NEXT observation. This distinction must never be lost.
6. Failed/expired episodes auto-reset; pre-reset state is preserved separately for PPO bootstrap and independent examiner logic.
7. PPO updates the learner. Curriculum updates only the next practice distribution; separate held-out exams must never secretly teach.

Task columns (regime, difficulty, capability-field provenance) are teacher-only and must never leak into the deployable actor observation. Foot-height contact and foot lift are not verified load-bearing walking. Terrain training uses tilted-gravity infinite-plane approximation, not geometric ramp transitions.

## What to do next (the next 'keep going')

**Pass D:** Make `training.py` read top-to-bottom without adding new control layers. Follow resume/setup → learner rollout/update → teacher and health feedback → periodic held-out examinations → durable reporting and final checkpoint. Do not change the saved checkpoint contract. **Before merging**, the B/C changes still require fresh isolated CUDA PPO update/save/resume and a frozen-policy exam.

After B2 follow Passes C–H in [SIMPLIFICATION.md](docs/SIMPLIFICATION.md): rewards and honest telemetry, readable training lifecycle, curriculum/capability, meta-experiments, reports/viewer, then isolated GPU acceptance.

### Working rules

- Before deleting a mechanism, find its actual call path, runtime effect, and owner. Prefer removing duplication over adding adapters.
- Keep all changes to the readable refactor branch in batched commits; preserve main, champion and active runs.
- Run **bash scripts/check.sh** and check commit-specific CI (16 Python + 4 browser, packaged viewer). The tiny real-MJX smoke is included; it is NOT GPU learning validation.
- Real GPU acceptance will need short PPO update/save/resume/export in a NEW run root and honest signed-direction slope/course exams against frozen policy brains.
- After each pass, update THIS handover and SIMPLIFICATION.md; update architecture.md only when dataflow/ownership actually changes.

## Paste into a new chat

> Continue from branch refactor/emu-readable, draft PR #5. First read root HANDOVER.md, docs/architecture.md and docs/SIMPLIFICATION.md from branch HEAD, then check CI. Preserve main, champion, active training runs, 220-input legacy actor and only 20 automated tests. Implement the next meaningful pass (currently D), simplify real code rather than only docs, run checks, update the SAME handover and checklist, commit a small batch, and report exactly what changed and what remains unverified.
