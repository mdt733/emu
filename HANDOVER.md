# EMU engineering handover — lean prism v5045

**4 November 2026.** The earlier v5044 140M experiment was manually
stopped; its source commit and checkpoints remain historical. New work
lives only in draft PR #34 on `feature/v5045-core-skeleton-audit`.
`main` and the published 260M champion are unchanged.

## One current model

- Six-servo dynamically unstable EMU; **one** 360-input 4×4-ToF
  array-only actor, critic and PPO optimizer, 2,048-world GPU target.
- **One** 3×3 infinite logical carpet: 18 physically colliding
  triangular prisms, nine noncolliding 2×2 vertex samples, no old
  two-slab/nine-box/three-platform collision bodies.
- First 20M: cold ordinary walking. 20–30M: gradual six-specialist
  exposure. Afterward the governor updates physical command, slope,
  up/down step, recovery and transition frontiers from completed
  episode evidence, not the calendar.
- STEP Academy is another birth cohort in that same carpet and PPO;
  the CONTROL role practices useful low-cost walking, not zero-motion
  energy farming. Teacher geometry never becomes policy input.

## Cleanup completed

- Over **23k deleted lines** vs frozen v5044, the historic tests reduced
  to 18 Python and six browser maintained contracts. Historic science
  stays in old Git SHAs rather than runtime compatibility branches.
- `environment.py` is prism-only; zero extra slab/box bodies.
  `ppo.py` no longer emits old slab and box exam traces.
- Obsolete physical exam, physical-quality, old commands and duplicate
  prism scorer retired. `emu prism-exam` runs one frozen
  true-prism actor and scores raw supported signed up/down crossings,
  actual terrain exposure, velocity, survival and shaft work.
- `policy_export.py`, `viewer_server.py` and
  `capability_viewer.py` separate actor, local HTTP and field UI.

## Current release/launch gates

1. GitHub 24-test + browser-build + four-world MJX physical probe **must
   pass on the exact final source SHA**.
2. On the RTX 4060 Ti, repeat the matched **2,048-world, 3-update
   full-PPO benchmark**. Old prism reference: 7,492 world-steps/s
   after JIT. First measured v5045 GPU benchmark: **7,536.1
world-steps/s** on two steady 2,048-world updates (+0.6%; not a
statistically convincing speedup), peak 1,818,866,944 device bytes.
Next required: exact-source 8-world checkpoint/resume/export and
frozen-actor true-prism exam smoke.
3. Same-source CUDA two-update `first/continue/verify` checkpoint,
   normalizer, optimizer and exported actor acceptance.
4. Only then start a **new empty** 140M run root. Never resume old
   v5044 state with v5045 code. Frozen-policy real-prism examination
   later validates walking, not the mere presence of PPO gradients.

**[New v5045 runbook](docs/V5045_LEAN_PRISM_RUNBOOK.md)** has exact
install, check, benchmark, lifecycle and launch commands.
**[Core source skeleton and live 102-file ownership/retirement checklist](docs/CORE_SKELETON.md#live-file-ownership-and-retirement-checklist)**
reconciles every surviving module with its job and identifies open gates. The historical full handover remains
[at the pinned v5044 commit](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/HANDOVER.md).

## 4 November — make the physical control frame understandable

- `EnvState` was **59 fields, now 52**. The retired single-slab
  approach counters and ascent peaks never changed in real-prism training;
  they no longer occupy an MJX state. Reset populates fields by name and
  per-frame updates use `state._replace(...)` with named live changes,
  rather than counting 59 positional constructor arguments.
- The only supported `params_array` always packs the full terrain/six-role
  teacher packet. Frozen exams and the trainer both construct the real
  18-prism model; `evaluation._world` rejects other geometry before JIT.
- The environment has no alternate rotated-ground plane, tripod/220-input
  sensor contract, or dead standalone finite-step platform dimensions.
  Robot and hardware model randomization are preserved. Exact 4×4 ToF
  acquisition, servo delivery, physical prism contacts and supported
  terrain events remain the active simulation.
- Removed the inactive v5031 single-lip careful-approach/crouch reward
  and unused offline counterfactual ascent diagnostics from `rewards.py`.
  The actual verified-prism crossing payment and the optional 25-frame
  forward tracking blend remain. Retired config fields were removed from
  `training.py` and `training_config.py`; this is a **new source-locked
  checkpoint lineage** and old branch checkpoints cannot be resumed.
- The 438-line `_step_impl` is now about 224 lines. Its reporting dictionary
  and physical slope/STEP episode grading moved into `_step_report`,
  which reads the pre-auto-reset episode accumulator and the same physical
  contact facts. A byte-for-byte comparison confirms the 8,032-character
  reporting dictionary and reward invocation were preserved.
- The existing four-world prism probe now requires finite, world-shaped
  rewards, episodes, signed-step, slope and stance evidence. Two more
  never-set Academy flags were removed; state construction now requires
  all 52 fields, without optional None defaults.
- Remaining large concepts: `_step_report` still collects substantial
  telemetry, `curriculum.py` still has a long
  evidence governor, and `results.py` still records legacy metric names.
  Consolidate only with fresh physical equivalence proof; do not change
  reward economics or teacher thresholds as part of cosmetic refactoring.
- On any latest commit, require core Python/JS and 4-world MJX CI green,
  then same-SHA CUDA first/continue/verify plus exported frozen-actor
  prism examination **before** starting a fresh 140M run.

## 4 November — tree-to-skeleton reconciliation

- [Live file ownership and retirement checklist](docs/CORE_SKELETON.md#live-file-ownership-and-retirement-checklist)
  compares the complete `c4e3c739` tree to historical v5044:
  **102 retained tracked paths**, **76 removed**, **12 newly created**,
  net **64 fewer files**. Every one of the **25 surviving Python engine
  modules** has an explicit owner; 77 support files are classified.
  Six retired Python engines, eight scripts, 21 Python test files,
  11 browser files, three workflows and 27 historical documents are
  accounted for. This is an inventory, not an assertion that every
  function has had a deep physical equivalence review.
- `scripts/audit_core_dependencies.py` now fails the usual fast CI if
  an unknown Python engine module appears, a required skeleton owner
  disappears or a retired engine module returns. `scripts/check.sh`
  also checks the current docs links, without adding new test files.
- The maintained install guide now checks out the correct v5045 branch
  and runs `probe_carpet_prism_mjx.py` and `smoke_gpu_carpet.py`, not
  the deleted v5031 tests or `smoke_unified_focus.py`. Maintenance
  points to `prism_exam.py`, not the removed `physical_exam.py`.
- Removed the phantom `--verified-ascent-reallocation` switch and the
  unsupported 232-input/rotated-plane CLI flags; removed the matching
  dead guard branches from `run.py`. The single 360-input prism source
  and legitimate training experiment controls remain. Historic programs
  still live in their own Git commits and are not source-compatible.
- The latest edited source must pass fresh Python/browser and real-prism
  CI, then source-matched CUDA lifecycle and frozen-policy examination.
  **Do not infer new GPU performance from line or file counts.**

## 4 November — further checkpoint and PPO cleanup

- The governor accepts only the exact current checkpoint evidence layout:
  no v5040 pooled step-success migrations, no fabricated missing Academy
  records and no command/slope windows silently filled from defaults.
  The existing core-run test now rejects incomplete state and checks
  near/easy signed STEP graduation and retreat.
- Deleted the non-unified STEP governor and the already-disabled
  generic command/combo growth loop. The active six-specialist controller
  remains the only command-frontier owner; physical up/down verification
  and current growth/retreat thresholds are retained.
- Removed three orphaned PPO exam kernels. Current frozen evaluation still
  uses `evaluate_summary`; the actual prism policy examiner still uses
  `evaluate_trace`. No active optimizer, actor or critic logic changed.
- Removed the retired rotated-plane/tilted-gravity console slope reporting
  fallback. The learner reports its real prism slope and exposure only.
- All changes are on draft PR #34. The final SHA requires complete CI,
  the owner's exact-source CUDA PPO/checkpoint/export acceptance and a
  frozen actor exam before beginning a new 140M lineage.

## 4 November — file-count pruning

- Whole-tree inventory reduced from 102 to **91 tracked files**,
  keeping all 25 backend owner modules and the three core Python tests.
- Removed seven browser files net: folded the seeded navigation RNG into
  navigation; auto/manual stick handoff into controls; preview sliders
  and geometry signature into training inspector; grouped viewer-only
  attitude, cadence, chase-camera and shove maths in `sim-math.js`.
  The existing browser suite also checks the combined helper imports.
- The redundant viewer-specific `.gitignore` is gone because root ignores
  all its patterns. The three research-era v5041/v5043/v5044 documents
  now have pinned source-history links from the current documentation
  index and root README; active v5045 procedures remain local.
- No network, PPO, reward, GPU terrain or released-champion behaviour was
  removed. The latest head still requires CI and same-SHA GPU/physics
  acceptance; file-count savings are not a performance claim.

- The separate `check_current_docs.py` has been consolidated into the
  core static dependency audit. `scripts/check.sh` and GitHub CI still
  fail on broken local Markdown links, with one fewer script to maintain.
  Live inventory: **90 tracked files**; relative to pinned v5044,
  **89 retired paths and 13 added paths**.

## 4 November — stop maintaining duplicate checkpoint/release entry points

- Removed uncalled historical `checkpoint_io` selectors and separate
  staged-foundation helpers; kept rotating current-schema snapshots,
  validated resume and source-pinned rollback. Removed duplicate CSV read
  alias in favour of `read_csv_rows`.
- Folded the 52-line `verify_showcase_dist.py` into
  `verify_showcase_release.py --verify-dist`. Both the protected Pages
  publishing workflow and the main integration workflow use the shared
  release check. Branch CI now actually stages, builds and validates
  the approved actor + WASM assets on the branch before calling browser
  delivery green.
- Current complete inventory: **89 tracked files**, **90 retired paths
  and 13 newly created paths** relative to source-pinned v5044. Core
  physics, six-specialist teaching and `main` champion remain unchanged.

## 4 November — remove inactive fake terrain physics

- Dropped 99 lines from `domain_randomization.py` and corresponding
  expired constructor/config parameters. Rotated-plane inclines and
  tilted-gravity approximations could not be selected in the
  v5045 prism environment anyway.
- Preserved optional randomized mass, gravity magnitude, joints and
  actuator variation with unchanged robot/hardware RNG-key assignments.
  The real 18-prism collision mesh is still the terrain source.
- Enhanced exact-head CI with `emu doctor --domain-smoke` to actually
  run optional manufacturing-noise MJX alongside the independent real
  prism contact probe. Until green, this branch stays research-only.
