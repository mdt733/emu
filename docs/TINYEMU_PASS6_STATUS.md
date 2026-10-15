# tinyEmu Pass 6 — Review-gated 20M→four 80M experiment campaign

**Date:** 15 October 2026. **Status:** source/CPU contract pass; **new-branch WSL GPU execution still pending**. Draft [PR #3](https://github.com/mdt733/emu/pull/3) remains unmerged. The known-working `main` remains at `40c5d0c`; the unverified Pass 9 ZIP remains reference material, not replacement code.

## Code actually changed

- `emu/ladder_recipe.py`: replace old 20M→26M→34M→46M targets with **20M foundation→80M four forks→optional 100M two→optional 120M one**. The first four each add 60M **from the exact same** checkpoint.
- `emu/branch_lab.py`: update the real autopilot plan, incremental GPU budget, review points and messages. **Reuses** the accepted lineage runner, source-commit/parent hashes, checkpoint resolver, held-out `screening.evaluate_runs`, scorecard and eligibility gates. No second PPO/ladder orchestration engine introduced.
- `scripts/smoke_training_lifecycle.py`: require all seven new servo and spring telemetry metrics in the actual first/continued GPU `metrics.csv`; reject missing, nonfinite, physically impossible fraction or shin energy larger than total spring potential. This validation runs **only when the real WSL operator invokes the 256→512 smoke**, not on GitHub CI.
- `tests/test_ladder.py`, `tests/test_training_contracts.py`: pin the absolute per-policy targets, 260M nominal four-way total and 320M optional full funnel; retain parent-selection/provenance, no-GPU plan and fail-closed selection contracts.
- `tests/test_smoke_training_lifecycle.py`: enforce that missing or NaN physical telemetry fails the bounded acceptance before large runs.
- `docs/TINYEMU_GPU_CAMPAIGN.md`: single operational runbook: independent worktree/venv, real GPU doctor and 0→256→512 resume/exam/export, separate 20M foundation, review, four sequential +60M arms, common held-out scorecard, optional finalists.
- `docs/tournament.md`, `README.md`, `docs/STATUS.md`, `docs/validation.md`, `docs/evaluation.md`: remove stale active 72M/46M guidance or clearly label the previous accepted-main recipe as historical. The tournament page points to the one operational runbook rather than repeating commands.

## Exact intent and resource budget

Four branches: `baseline` 10% graph-teacher reference; `shadow` teacher disabled; `slope` terrain-equivalent domain randomisation; `pitch15` 15° requested chassis pitch. These are **one-dimensional training hypotheses** already supported by `TrainingConfig` and are not new reward adjustments.

Nominal total through first common evaluation: **20M + 4×60M = 260M environment steps**; optional 2×20M + 1×20M makes **320M aggregate**. Targets are absolute per-policy steps, rounded to complete `envs × 32` PPO update boundaries. First-round auto mode **stops before finalists** with `--stop-after round1`, so human review is a required planning milestone.

## Retained / removed

Retained unchanged: actor 55×4=220 inputs/six outputs, PPO optimiser, scalar reward, domain sampling, existing evaluation and source/checkpoint hashing, champion, Pages manual deployment and all earlier run roots. No runtime training modules were removed in this pass: the improvement is an **experiment recipe using existing orchestration** instead of building another layer.

Historical 72M-target *current instruction* is retired from the active pages, not from Git history or old experimental results.

## Actual tests (not guesses)

- First new ladder CI [#113](https://github.com/mdt733/emu/actions/runs/38686054858) found two stale assertions: one hard-coded 26/34/46M training contract, one synthetic CSV fixture with literal `\\n` escapes. Both corrected.
- [Current PR CI](https://github.com/mdt733/emu/pull/3/checks) is authoritative for the final source revision. No GPU test can be claimed simply because these contracts pass.
- Previously [#106](https://github.com/mdt733/emu/actions/runs/38685041786) validated scoring/servo/spring CPU-JAX contracts and static browser build (167 Python passed plus 5 subtests, 18 JavaScript passed).

## Mandatory manual WSL gate before actual 260M experiment

1. Follow [the exact isolated runbook](TINYEMU_GPU_CAMPAIGN.md) at a pinned, clean Git SHA. New `.venv` belongs exclusively to the new worktree.
2. Pass `emu doctor --v5015-smoke --domain-smoke`, then the 8-world, 256→512 environment-step PPO/checkpoint/warm-resume/fixed-exam/no-op/export acceptance *including real actuator and spring telemetry*. Reject any run with missing/nonfinite columns.
3. Start **only** the shared 20M foundation; evaluate height drift, command tracking and obvious pathological stepping before forking.
4. After human review, run the four 20M→80M forks sequentially, stop at common matched held-out screening, and review the scorecard plus mechanical telemetry. Do not promote by training reward or appearance alone.
5. The current `step_evidence` is foot lifting, **not a certified completed stride**. Current training-rollout power averages vary across each arm's curriculum and are **not comparable held-out efficiency scores**. Calibrate genuine stride/effort examiner before reward or promotion changes.
6. Confirm with later independent seeds / 100–200M per-lineage long-horizon evaluation before claiming robust hardware-ready walking.

**Never** use these commands on the accepted working checkout or an old output root, and do not merge or deploy the draft branch without fresh real-GPU evidence.
