# Pass 1: source comparison (15 October 2026)

## Provenance

Current GitHub `main` HEAD is `40c5d0ce543ebbbc3690591695afddc18def8232` (PR #2 merged). Candidate is the user-supplied `tinyEmu_Pass9_Validation_Candidate(2).zip`, descended from the *older* `7cafca983eccee92e3334c3031a4424b7d8223e3` source. These are not two versions of one sequential patch. **Do not blanket-copy the candidate.** The stable baseline's trained 60M policy has Git blob `4743d8d3cd4706edce04809fe1b61df014e57d5e`, identical in the ZIP.

Candidate's actual `emu/*.py` files: 18 modules, **5,961** physical lines, 325,291 bytes. Original audit's old basis: 31 modules/9,198 lines; current main file inventory has **30 operational Python modules plus `__init__.py`**, approximately 9,198 lines. Thus the candidate's 35% reduction is against the current-size class, but code-size savings **do not establish feature retention**.

## Runtime file-by-file treatment

`M` means retained module with changed behaviour or implementation; `=` exactly matching Git blob between candidate and current main; `-` absent from Pass 9; `+` candidate-only. This audit is a migration/ownership decision, not numerical equivalence evidence.

| Main module | Candidate | Treatment | Why |
|---|---|---|---|
| `artifacts.py` | M | keep reduced helper | Same run/CSV responsibilities; review formatting instead of duplicating writer |
| `atlas.py` | - | archive/rebuild selectively | Physical capability probing lost; use independent walking examiner later |
| `branch_lab.py` | - | rebuild small `ladder.py` | Multi-arm ancestor, matched screens, selection and integrity protection lost |
| `capability_cloud.py` | - | retire hot path; retain history | Sparse recovery exams/calibration removed |
| `capability_field.py` | - | retire hot path; change learning semantics | Node/edge graph teaching and evidence map removed |
| `checkpoint_io.py` | M | **do not swap**; explicit lineage | main schema 7 with field state; candidate schema 9 without; incompatibility |
| `checkpoints.py` | - | merge safe discovery | Candidate consolidates checkpoint ownership |
| `cli.py` | M | stage simple `emu train` | Candidate drops ladder and atlas CLI; preserve options until replacement tested |
| `contracts.py` | M | keep 220/6 invariant | internal packet was simplified; actor dimensions and order must stay fixed |
| `curriculum.py` | = | keep and explain | Large controller (578 lines) still complex; refactor on-policy inputs later |
| `domain_randomization.py` | = | keep | Model parameter variability not removed |
| `environment.py` | M | incremental, oracle-tested | Candidate removes teacher dependence/internal task packet; no GPU physical parity proof |
| `evaluation.py` | M | keep held-out | Candidate 11-field fixed exams; keep benchmark protocol stable |
| `field_viewer.py` | - | retire alongside graph field | Specialist capability visualisation lost |
| `health.py` | = | keep | Training health and recovery retained |
| `ladder_recipe.py` | - | rebuild readable recipe | Current automatic 20M→26M→34M→46M branches gone |
| `live_viewer.py` | - | consolidate in viewer server | Single-run viewer should retain validated checkpoint hot-swap |
| `paths.py` | = | keep | Paths and version ownership |
| `phase_config.py` | - | consolidate safely | Phase configuration must stay ancestor/source safe |
| `policy_export.py` | M | staged migration | Candidate removes tournament/demo-node metadata, keeps schema-1 browser export and atomic write |
| `posture_report.py` | - | optional read-only tool | Historical crouch/height inspection no longer bundled |
| `ppo.py` | M | retain math; test optimizer | Candidate shrinks diagnostics, cannot assume old optimiser semantics perfectly preserved |
| `rollout_metrics.py` | M | reduce carefully | Candidate keeps on-policy governor summaries, fewer field/graph dimensions |
| `run_support.py` | - | merge safe lifecycle | Current source validation, protected lineage and completion flow must survive |
| `runtime.py` | = | keep | Device selection and profiling |
| `screening.py` | - | restore matched held-out exams | Separate winner evaluation needed, not simple training score |
| `tracking_contract.py` | - | consolidate into contracts | Avoid duplicate thresholds, preserve reference semantics |
| `training.py` | replaced by `train.py` | candidate idea worth keeping | 975-line main trainer vs 777-line candidate; candidate removes cloud/field control and experiments |
| `training_config.py` | M | keep typed config | Candidate removes some experiment/teacher settings; explicit knobs important |
| `viewer_server.py` | M | single-source packaging | Candidate gains live-watcher ownership; validated file swaps must still work |
| `train.py` | + | port only after GPU acceptance | Simpler one-run entrypoint; no compatibility with schema-7 checkpoint |
| `scoring.py` | not in Pass 9 | **new branch work** | Make the reward payment ledger directly navigable, before moving metric calculations |

## Other files and migration hazards

- **Model:** `emu/robot.xml` is byte-identical between ZIP and main (Git blob `6bbe55ea44a75a7383920fa78ef49824256f5e4d`). The six force-limited actuators, springs, and geometry survive.
- **Champion:** protected `showcase/champion.emupolicy` exactly matches; do not modify, regenerate, or auto-deploy.
- **Browser physics:** `emu/viewer_src/src/sim.js` exactly matches (Git blob `4c6900643d95455ee4f582f397b19cd876fc8765`). Other browser policy/HTML/style/packaging files **differ**; candidate's stronger fail-closed parser is useful but needs cross-runtime and actual browser gates.
- **Mirrored viewer:** main tracks `viewer/` and `emu/viewer_src/` matching trees. Candidate keeps the package copy and drops top-level mirror. Benefit: eliminates dirty-after-build hazard, provided Pages build and packaging sources are verified.
- **Install:** candidate `scripts/setup_wsl.sh` lacks the fresh-nvm `source "$NVM_DIR/nvm.sh" --no-use` fix already on main. Do not port this regression.
- **Acceptance:** main has `scripts/smoke_training_lifecycle.py` with accepted clean-process GPU 0→256→512, exam, no-op, export. Candidate lacks it; `scripts/validate_pass9.sh` covers a different bounded sequence. Combine gates rather than replacing proven lifecycle evidence with tests that skipped MuJoCo.
- **Tests:** main has branch/provenance/installer, policy-parity, live-viewer, scorecard and clean-install tests; candidate drops some while adding new architecture, train-command and Pass 9 tests. Each test removal must be mapped to explicitly retired behaviour, not quietly lost.
- **Historical evidence:** current `results/` and `docs/archive/` are not in the candidate; do not delete learning history or experiment results to achieve smaller file counts.
- **Workflow:** `.github/workflows/emu-playground.yml` differs from candidate. Preserve manually dispatched Pages and explicit champion selection. CI checks likewise need review instead of overwrite.

## Technical risks and owners

1. **Checkpoint schema 7→9** is not a clean resume: model weights may export to the viewer, but optimiser, governor and field state do not carry over losslessly. New lineage unless a dedicated migrator is proven.
2. Removing the graph teacher changes the *training distribution*, even when PPO gradients, reward coefficients and XML are unchanged.
3. `step_evidence` rewards lifted feet, not completed alternating touchdown and displacement; `support` is a height-based proxy, not actual normal force. Survival-weighted `best.pkl` and branch promotion cannot certify gait.
4. Reward smoothness and position-limit pressure do **not** represent actuator power/thermal stress. Add read-only torque, speed, work and spring energy observables before proposing motor-efficiency rewards.
5. Evaluation summary metrics may span automatic resets. Build uninterrupted episode/stride metrics and bad-controller controls before changing promotion thresholds.

## Current isolated branch work, not Pass 9 migration

Branch `refactor/tiny-emu` is derived **directly from accepted main**, retaining its schema-7 lifecycle, ladder, viewer and champion. New `emu/scoring.py` owns the *existing eight positive weighted payments*; `emu/environment.py` invokes the ledger and emits named contributions; `emu/rollout_metrics.py` surfaces them to CSV; `tests/test_reward_ledger.py` locks their original arithmetic; `docs/WHERE_THINGS_LIVE.md` traces source→reward→PPO→log→evaluation→curriculum→promotion.

The candidate copy's Python suite and Node suite ran successfully (105 pass/2 skip and 20 pass); isolated ledger prototype tests passed (107 pass/2 skip). Candidate reward arithmetic was bitwise equal to its old implementation for two profiles × 1,024 synthetic NumPy input states. **These are not tests of the new GitHub branch on GPU.** Next gate is branch Python/JS CI, real WSL/MJX doctor, fresh-root 0→256→512, resume/no-op, held-out exam, export and interactive trained browser.

## Next steps

1. Keep main and 60M champion untouched; leave prior training roots untouched.
2. Run branch CI, inspect new reward-column drift, document exact branch runtime/test results.
3. Implement opt-in `emu train` without deleting the accepted ladder until end-to-end evidence exists.
4. Move raw physical metrics into a narrow readable packet; preserve oracles and the 220/6 contract.
5. Build read-only walking and per-servo efficiency scorecards. Do not modify PPO reward or promotion criteria until measurements are trustworthy.
