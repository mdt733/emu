# Testing EMU without drowning in tests

Most old one-test modules have been folded into their owning test files. This is intentional: one obvious place per subsystem is easier to review than a collection of single-purpose filenames. Test assertions stay intact.

There are two separate questions: **did my code change break a contract?** and **can the trained robot genuinely walk?** Python assertions alone cannot answer the second.

## Three useful levels

| When | Command | What it means |
|---|---|---|
| During ordinary editing | `bash scripts/check.sh` | **15 selected contracts** covering course anti-cheats, checkpoint safety, reward arithmetic, sensor estimation, tournament selection and result export. No new GPU training. |
| When changing a physics evaluator | `bash scripts/check.sh physics` | Small real MJX CPU compilation/rollout of command transitions and simulated gyro measurements. Not proof of walking. |
| Before merging or trusting a release | `bash scripts/check.sh all` | Every Python regression plus JavaScript viewer tests. CI additionally builds browser assets, validates champion/WASM and checks generated-file cleanliness. |

`bash scripts/check.sh full` runs only the complete Python suite, and `bash scripts/check.sh browser` only JavaScript. The [quick selection](../tests/quick_cases.txt) is intentionally short and explicit—**not an attempt to turn the full test suite into another clever framework**.

The complete suite currently reports about 226 Python cases and 18 browser cases; the top-level Python test files have been reduced from 34 to **25**. Many are inexpensive parameter variations; the important quantity isn't the number, but whether they uniquely protect actual behaviour. Keep the full suite in CI while simplifying duplicates case-by-case with evidence. **Do not delete valuable checks merely to reach an arbitrary numeric target.**

## Where to look for tests

| Area | Most relevant tests |
|---|---|
| Reward, locomotion and servo diagnostics | `test_reward_ledger.py`, `test_motion_tracking.py`, `test_servo_effort.py` |
| PPO, checkpoints, resume and source identity | `test_checkpoint_resume.py`, `test_single_run.py`, `test_smoke_training_lifecycle.py`, `test_training_contracts.py` |
| Experiment ladders, results and promotion | `test_ladder.py`, `test_results.py`, `test_capability_field.py` |
| v5016 directional commands and heading controller | `test_v5016_control.py` |
| v5016 downward ranging and true ground slope | `test_v5016_ground.py` |
| v5016 physical traces, stride evidence and paired GPU campaign | `test_v5016_exam.py` |
| Viewer compatibility and protected public champion | `test_policy_inference_parity.py`, `test_showcase_product.py`, `test_live_viewer.py` |

### Important limitations

- `quick` is for developer feedback, **not** a release gate.
- CPU MJX proves integration and deterministic invariants, **not trained gait quality**.
- Real policy comparisons require `emu directional-batch` on your GPU, with completed saved policies and untouched training roots; see [v5016 Pass 4 instructions](V5016_PASS4_STATUS.md).
- Never run a GPU/training acceptance exercise against `main` just because a refactor branch's unit tests passed.
