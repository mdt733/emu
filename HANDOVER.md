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

- **Pass D1 (software CI verified: [38842290106](https://github.com/mdt733/emu/actions/runs/38842290106)):** Moved resume/fallback/early-completion/recovery seeding into the existing `run_support.py` and rollout provenance labelling into `rollout_metrics.py`. Original training.py went from 993 to ~905 lines without changing physical loops, exam timing, rewards or checkpoint schemas. Extended an existing resume test rather than adding new tests. Next is D2 held-out exam and checkpoint/log orchestration.

- **Pass D2 (software CI verified: [38842871586](https://github.com/mdt733/emu/actions/runs/38842871586)):** extracted the fixed-policy exam/write logic into `evaluation.record_fixed_exam` and capability-cloud panel/calibration logic into `capability_cloud.record_capability_cloud_exam`. `training.py` keeps the original exam timing and snapshot selection, while exam owners write held-out evidence and guard teaching-field immutability. The checkpoint metadata and 20-test limit are unchanged. Main training file is now 787 lines. D3 next: logging and periodic checkpoint publication.

- **Pass D3 (software CI verified: [38843252714](https://github.com/mdt733/emu/actions/runs/38843252714)):** metric row construction, field-influence annotations and periodic/final report/field-artifact writing are now owned by `artifacts.py`; report schemas and the final checkpoint-before-field-publication order stay unchanged. The delivery contract test was strengthened, with 20 tests retained. `training.py` is now 750 lines. Reporting takes its learning metadata as explicit values rather than importing training configuration. D4: recovery and remaining PPO orchestration; fresh isolated CUDA still outstanding.

- **Pass D4 (software CI verified: [38844333684](https://github.com/mdt733/emu/actions/runs/38844333684)):** full health-triggered recovery now lives with checkpoint resume/handoff in `run_support.try_restore_recovery`. It rehydrates policy, optimizer, governor, health, teaching field, observation normalizer and LR, then resets all physical worlds with the same `fold_in(key0, absolute_step)` rule. The trainer explicitly rebinds returned state and clears the competence window only on success; failed restores retain the collapse streak. Extended the existing resume test to exercise full rewind/reset with a real saved checkpoint. **No new tests, changed reward, key ordering or checkpoint schema.** D5 next: clarify PPO/update and checkpoint save lifecycle. GPU acceptance still required.

- **Pass D5 (software CI verified: [38844623465](https://github.com/mdt733/emu/actions/runs/38844623465)):** moved the KL-adaptive PPO LR scaling decision into `TrainingHealth.adjust_learning_rate` beside its KL-stress gate, with the exact old deadband/clamp and nonfinite hold. The training loop now explicitly requests the NEXT LR from training evidence, while frozen examinations stay independent. Removed an unused `field` argument from the checkpoint recovery helper. Strengthened the existing config test; the test count remains 20. Next work: investigate the remaining checkpoint/rollout ownership vs Pass E curriculum complexity; GPU lifecycle still outstanding.

- **Pass E1 (software CI verified: [38844844745](https://github.com/mdt733/emu/actions/runs/38844844745)):** `RegimeGovernor.update` now tells the lesson-control story without embedding the ~106-line physical frontier machinery: `_advance_physical_frontiers` retains the original settled-tracking axis/combination/push/sensor/transition logic, EMA/hysteresis and safety limits verbatim in the SAME owner file. An existing test exercises frontier expansion from real training evidence and confirms disabled adaptive frontier stays off. 20 tests remain. No attempt to delete unproven complexity or to reward servo power. Next E2: audit capability-field evidence → teaching packet → task sampling for duplicate/unexecuted controls.

## Latest verification and unfinished work

- E2 `d2f349c`: [CI 38845714642](https://github.com/mdt733/emu/actions/runs/38845714642), **16 Python + 4 browser and viewer build passed**. Birth and transition sampling share one JAX command envelope. This did NOT change random split arity, sampling probabilities, task fields or PPO rewards.
- E3 `3391ba6`: [CI 38845909394](https://github.com/mdt733/emu/actions/runs/38845909394), **16 Python + 4 browser and viewer build passed**. Capability-field live event trackers now have one initializer; learned checkpoint evidence stays unchanged.
- **Still unverified:** fresh-root CUDA PPO learning/update/save/resume, actor export, physical signed-direction and gentle-slope frozen-policy walking examinations. Existing CPU CI cannot establish this.
- **Next:** E4 controlled evidence/decay and `field_taught` clean-evidence isolation audit. Do not remove node/edge/recovery features simply because off/shadow mode sets their teaching fraction to zero. After E, proceed through F experiment planning, G reports/viewer and H isolated acceptance.

## E2 evidence-to-teaching trace (verified from executable call paths)

The next pass should follow this chain BEFORE deleting machinery:
`training.py` runs `field.update_from_rollout(rollout_metrics, step, exclude_field_taught=...)`; only TRAINING rollouts update this field. The live field is converted into `teaching_packet` node/edge anchors with a scheduled influence. `tasks._sample_task` samples node anchors for episode births, while `tasks._sample_cmd_target` samples node or source-local edge anchors for TRANSITION command targets (edge has priority). The chosen commands remain bounded by the governor's current physical envelope. `field_taught` marks field-selected experience, and the optional clean-evidence setting excludes that experience from the field's own future evidence. Held-out fixed/cloud exams cannot update the field. `field_teach_recovery` separately gates the node-proposed recovery severity and stays bounded by governor resilience. No node/edge/recovery path is proven dead merely because its influence is temporarily zero (off/shadow or pre-ramp).

E2 should audit whether the capabilities and teacher controls *earn their complexity*, not silently remove a call path that is temporarily inactive. Preserve JAX RNG split counts, teacher-only task columns, exact live policy provenance, and the 20-test limit.

- **Pass E2 (software CI verified: [38845714642](https://github.com/mdt733/emu/actions/runs/38845714642)), commit `d2f349c`:** traced the actual teaching path through `CapabilityField.update_from_rollout` → scheduled teaching packet → `TaskSampler._sample_task` node births and `_sample_cmd_target` source-local node/edge transition anchors. The duplicated physical command envelope calculations in these two task entry points now share one deterministic `_command_envelopes` in `tasks.py` (one full + one reduced physical envelope). This preserves the original JAX arithmetic, transition jump, field-command clipping, and every RNG split/key. Strengthened the existing real-MJX sampler test to check both adaptive/non-adaptive envelopes and node-selected birth provenance, without growing the 20 tests. **No weight, reward, policy, checkpoint or 220-input actor changes.** E3 next: evaluate further field complexity and tighten evidence isolation where independently verifiable; CUDA acceptance still due.\n\n- **Pass E3 (software CI verified: [38845909394](https://github.com/mdt733/emu/actions/runs/38845909394)), commit `3391ba6`:** `capability_field.py` previously created its 19 per-world node/recovery/edge event arrays twice (zero-world initialisation and later world-count setup). A single `_reset_runtime(n_envs)` now owns both layouts. `_ensure_runtime` resets only when cohort size changes; a new process still begins with zero in-progress events, while learned count arrays and schema remain intact. Strengthened the existing checkpoint-resume test: same-sized runtime retains in-flight windows; resizing clears them; changing the runtime never changes the serialized field fingerprint; resumed checkpoints begin with empty windows. No policy, task, reward, RNG, decay or teacher-score changes. **20-test limit preserved.** Next E4: check field/evidence weighting and healthy age-based decay against small JAX host fixtures, not remove genuine learning mechanisms.\n\n- **Pass E4 (source committed; software CI pending):** made the common event-window interruption explicit in `CapabilityField._cancel_event_windows`. The node/recovery/edge trackers now use that one method for new episodes and for field-selected frames in clean-evidence mode; a node switching command cells still resets only its own node window (not unrelated recovery/edge state). Strengthened the *existing checkpoint/evidence test*, without adding tests, to verify that clean sustained experience produces evidence, tagging a frame breaks a half-finished window, non-isolated field-taught experience can still be counted, and a precise half-life halves accumulated evidence. No change to Bayesian competence estimators, node/edge/recovery ranking, rewards, JAX sampling keys, checkpoint schemas or actor dimensions. E5 next: inspect capability-field event grammar for additional duplication only if meaningful; otherwise move toward F experiments.\n\n## Physical path and invariants

1. EmuMJXEnv loads robot.xml and builds batched JAX/MJX model variations once per run.
2. Reset samples an independent birth task (tasks.py), then places each robot in a dynamic initial pose/velocity, initialises target-ramp state and history observations.
3. Step delivers the PREVIOUS actuator target, applies disturbance forces and integrates MuJoCo for the configured substeps; it rolls back non-finite physics states and terminates them.
4. Optional TRANSITION lessons sample a new target and advance the command ramp.
5. Physical metrics are evaluated and rewards.py scores **the command the actor saw before it acted**; the advanced command is for the NEXT observation. This distinction must never be lost.
6. Failed/expired episodes auto-reset; pre-reset state is preserved separately for PPO bootstrap and independent examiner logic.
7. PPO updates the learner. Curriculum updates only the next practice distribution; separate held-out exams must never secretly teach.

Task columns (regime, difficulty, capability-field provenance) are teacher-only and must never leak into the deployable actor observation. Foot-height contact and foot lift are not verified load-bearing walking. Terrain training uses tilted-gravity infinite-plane approximation, not geometric ramp transitions.

## What to do next (the next 'keep going')

**Pass F1:** With the core E mechanisms preserved and evidence boundaries verified, simplify the experiment plan and branch/orchestration story: shared foundation → independently taught branches → equal exams → promotion. Revisit E only for proven duplication. Follow resume/setup → learner rollout/update → teacher and health feedback → periodic held-out examinations → durable reporting and final checkpoint. Do not change the saved checkpoint contract. **Before merging**, the B/C changes still require fresh isolated CUDA PPO update/save/resume and a frozen-policy exam.

After B2 follow Passes C–H in [SIMPLIFICATION.md](docs/SIMPLIFICATION.md): rewards and honest telemetry, readable training lifecycle, curriculum/capability, meta-experiments, reports/viewer, then isolated GPU acceptance.

### Working rules

- Before deleting a mechanism, find its actual call path, runtime effect, and owner. Prefer removing duplication over adding adapters.
- Keep all changes to the readable refactor branch in batched commits; preserve main, champion and active runs.
- Run **bash scripts/check.sh** and check commit-specific CI (16 Python + 4 browser, packaged viewer). The tiny real-MJX smoke is included; it is NOT GPU learning validation.
- Real GPU acceptance will need short PPO update/save/resume/export in a NEW run root and honest signed-direction slope/course exams against frozen policy brains.
- After each pass, update THIS handover and SIMPLIFICATION.md; update architecture.md only when dataflow/ownership actually changes.

## Paste into a new chat

> Continue from branch refactor/emu-readable, draft PR #5. First read root HANDOVER.md, docs/architecture.md and docs/SIMPLIFICATION.md from branch HEAD, then check CI. Preserve main, champion, active training runs, 220-input legacy actor and only 20 automated tests. Implement the next meaningful pass (currently F1), simplify real code rather than only docs, run checks, update the SAME handover and checklist, commit a small batch, and report exactly what changed and what remains unverified.
