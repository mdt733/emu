# EMU core skeleton — run to robot walking in the viewer

**Audit branch:** `feature/v5045-core-skeleton-audit`, 4 November 2026.
**Current new-run candidate:** this v5045 feature branch; the previous
source-pinned v5044 large run was stopped. Its saved checkpoints belong to
that old source SHA and must never resume on v5045. `main` and the released
browser champion remain untouched.

## Live file ownership and retirement checklist

**Live inventory:** 4 November 2026; pinned [v5044 source tree](https://github.com/mdt733/emu/tree/a2f59bb31bec049204b60a3eb86500b100db039f)
compared with the current draft PR #34 tree. The first complete
[census at `c4e3c739`](https://github.com/mdt733/emu/tree/c4e3c739938bf3e5cc6e2d51bef42b9776c47cee)
had 102 files. Further browser-helper consolidation, historical-doc
retirement and one redundant ignore file leave **90 tracked files:
89 paths retired and 13 new paths** relative to v5044. That is 76
fewer tracked files, not 89 fewer *features*. Historical research stays
accessible at the pinned old SHA. The 886 KB
`showcase/champion.emupolicy` is a binary/encoded actor artifact,
not readable source. Verify its hash and manifest with
`scripts/verify_showcase_release.py`.

**Status key:** checked = present and assigned an owner / retired from
the current tree. A checked box is **not** a claim of completed
new-policy training, GPU speed improvement, or complete source audit.

### 1. One owner for each live Python module — 25/25

| Status | Skeleton responsibility | Actual Python files (all paths relative to `emu/`) |
|---|---|---|
| [x] | Command and configuration (4) | `__init__.py`, `cli.py`, `run.py`, `training_config.py` |
| [x] | Physical world, sensors and terrain (6) | `contracts.py`, `environment.py`, `terrain_carpet_mjx.py`, `carpet_prism_mjx.py`, `domain_randomization.py`, `directional.py` |
| [x] | One learner and six teaching specialties (7) | `training.py`, `ppo.py`, `rewards.py`, `curriculum.py`, `capability_focus.py`, `capability_field.py`, `training_health.py` |
| [x] | Complete state, reports, independent exams (4) | `checkpoint_io.py`, `results.py`, `evaluation.py`, `prism_exam.py` |
| [x] | Export and browser interfaces (3) | `policy_export.py`, `viewer_server.py`, `capability_viewer.py` |
| [x] | Installation and environment diagnostics (1) | `doctor.py` |

The robot's six-servo physical definition lives separately in
`emu/robot.xml`. None of the six specialist roles owns a separate
actor, PPO optimizer or terrain generator. The 360-input policy is
assembled from real sensor and command data, not privileged teacher
geometry. The browser's nine-box *illustration* is not one of the
eighteen real MJX collision prisms.

### 2. Every remaining supporting file group — 65/65

| Status | Current folder or files | Count | Why retained |
|---|---|---:|---|
| [x] | `emu/viewer_src/src/`: 17 JavaScript modules + `style.css` | 18 | Local MuJoCo/WASM viewer: actor, sensors, physics, controls, navigation and representative terrain; **not** a GPU trainer |
| [x] | `emu/viewer_src/` build files, notices, one browser test | 6 | Reproducible `npm ci`, build and seven browser contracts |
| [x] | `scripts/`: launch (1), physical/GPU/lifecycle probes (3), release/static checks (5), WSL setup (2) | 11 | One production campaign; explicit acceptance and guarded deployment |
| [x] | `tests/test_core_{run,physics,delivery}.py` | 3 | 18 focused Python contracts; no archived test matrix |
| [x] | `.github/workflows/`: `core-skeleton-proof.yml`, `emu-checks.yml`, `emu-playground.yml` | 3 | Branch physics checks, protected main checks and approved publishing |
| [x] | `docs/`: 13 Markdown files + one architecture diagram | 14 | Current skeleton/runbook/installation plus clearly dated physical baselines |
| [x] | `showcase/`: `champion.emupolicy`, `deployment_request.json`, `release_approval.json` | 3 | Protected released 260M actor and approved release identity |
| [x] | Root metadata, `emu/VERSION` and `emu/robot.xml` | 7 | `README.md`, `HANDOVER.md`, `pyproject.toml`, `.gitignore`, `.nvmrc`, version, robot model |
| | **Total, including 25 engine modules** | **90** | No unassigned tracked category at this census |

### 3. What was actually cleared away — 89/89

| Status | Retired paths compared with v5044 | Count | Replacement or reason |
|---|---|---:|---|
| [x] | Engine: `capability_cloud.py`, `physical_exam.py`, `physical_quality.py`, `step_academy_world.py`, `terrain_carpet.py`, `terrain_spike.py` | 6 | One prism physics engine, current governor, `prism_exam.py` |
| [x] | Scripts: eight retired experiments plus redundant `check_current_docs.py` (merged into the core static audit) | 9 | Current runbook, three explicit smoke/lifecycle/physics scripts and immutable old research |
| [x] | Viewer: old Prism Lab (4), seven obsolete tests and eight merged micro-helper files | 19 | Integrated viewer and `essential.test.mjs` |
| [x] | Old Python tests, including `conftest.py` and 20 research fixture/test files | 21 | Three maintained core test files + real MJX contact probe |
| [x] | Superseded box/hfield/prism workflow YAML files | 3 | One complete branch proof workflow |
| [x] | Historical/duplicated `docs/` pages and `docs/history/` files | 30 | Current docs reference historical commits instead of carrying experiments as active code |
| [x] | Redundant `emu/viewer_src/.gitignore` (root patterns cover all three entries) | 1 | Root `.gitignore` owns viewer build exclusions |
| | **Total removed paths** | **89** | Contents can be recovered from source-pinned v5044 |

**Not deleted:** `training-carpet.js` and
`training-solid-carpet.js` are still required by the *browser*
as a deliberately approximate terrain preview. `directional.py`
remains part of the live velocity/sensor contract. The released actor,
checkpoint provenance, genuine signed terrain crossing tests, all
18 prism contacts, and all six lesson roles stay.

### 4. Verification and remaining engineering work

- [x] One 360-input, six-output actor; 18 physically colliding triangular
  prisms; 9 noncolliding source-sample tiles; one shared PPO.
- [x] No old slab, nine-box GPU collision, legacy physical examiner or
  obsolete Prism Lab in the active tree.
- [x] `EnvState` requires 52 explicitly named fields; control-frame
  reporting is isolated in `_step_report`; source-level report
  dictionary and reward invocation compared unchanged.
- [x] Core static import inventory, current local Markdown links,
  18 Python contracts, 7 browser contracts and source-locked champion
  checks. Last confirmed fully green branch SHA before this continuation:
  `ab9a52a6` ([CI](https://github.com/mdt733/emu/actions/runs/41740744834)).
  Subsequent cleanup commits require their own complete rerun.
- [x] Actual 4-world MJX prism contact, 360-input reset/step,
  safe recycle and essential telemetry probe at that last green SHA.
- [ ] **Repeat all CI checks on the final new source SHA.** A green
  earlier commit does not green-light subsequent cleanup commits.
- [ ] Same-SHA RTX 4060 Ti 2,048-world, 3-update full-PPO throughput
  comparison. The earlier 7,536.1 world-steps/s measurement does
  not benchmark the latest state or reporting changes.
- [ ] Same-SHA 8-world CUDA `first → continue → verify` checkpoint,
  normalizer and optimizer lifecycle; independently export its actor.
- [ ] Same-SHA **frozen exported actor** on real MJX prisms,
  proving the examiner consumes genuine contact and sensor evidence.
- [ ] Review `environment.py` episode event/report coupling,
  `curriculum.py` evidence/state complexity, and `results.py`
  redundant host-side outputs. Change no numerical thresholds
  without a measured equivalence check.
- [ ] Only after those physical gates: a fresh, empty 140M cold-start
  directory. **Never** resume an earlier source lineage or move the
  protected main actor before owner-reviewed capability evidence.

**How to repeat the census:** GitHub tree/compare against the pinned
SHAs above, plus `python scripts/audit_core_dependencies.py` (now also validating
local Markdown links). Import reachability is a
*review aid*, not proof that a module is safe to remove. The released
actor is validated by `scripts/verify_showcase_release.py`; GPU and
checkpoint behavior require the real acceptance runs.

## The five-stage minimum product

```text
   user + CLI command
       emu/cli.py -> emu/run.py  [validate source SHA, run config, resume]
                          |
                          v
       emu/training.py            [one life-cycle and PPO orchestration]
           |                 |
           v                 v
   REAL WORLD / OBS       ONE LEARNER
   robot.xml              ppo.py          [actor, critic, PPO]
   contracts.py           rewards.py      [teaching signals only]
   environment.py         curriculum.py   [evidence-based difficulty]
   terrain_carpet_mjx.py  capability_focus.py  [6 episode cohorts]
   carpet_prism_mjx.py    training_health.py   [PPO recovery]
   domain_randomization.py
           |                 |
           +--------+--------+
                    v
    checkpoint_io.py + results.py
    [weights, optimiser, obs normalizer, governor/health, seed,
     source SHA, metrics, energy, snapshots, final.pkl]
                    |
              +-----+-----+
              |           |
              v           v
          FROZEN EXAM   VIEWER EXPORT
          evaluation.py  policy_export.py -> .emupolicy
           prism_exam.py                         v
                                       browser policy.js
                                       browser sensor.js
                                       browser sim.js [MuJoCo/WASM]
                                       browser main.js [controls/camera]
```

The rectangle above is a **logical ownership map**, not a claim that
modules have been physically moved yet. Code coverage and checkpoint
compatibility take priority over folder aesthetics.

## The actual user journey and artifacts

| Stage | Entry point | Must preserve | Output |
|---|---|---|---|
| Plan/start | `emu train` / `scripts/run_prism_140m.py` | immutable 140M recipe, 2,048 worlds, clean exact source, no overwrite | `training_config.json`, `run_meta.json` |
| Simulate | `EmuMJXEnv.reset/step` | six actuators, 360 observation meaning, true prism contacts, real ToF and support, 3×3 safe tile recycling | per-world observations, actions and contact facts |
| Learn | `emu/training.py` + `make_trainer` | one actor/optimizer, six teacher cohorts, evidence-driven promotions, gradients and health | new weights, rollout/role metrics |
| Preserve/grade | `checkpoint_io.py`, `evaluation.py`, `results.py` | optimizer, curriculum, health, normalizer and source hashes; fixed physical exams on **the same prism contact mode** | `latest.pkl`, snapshots, `final.pkl`, `metrics.csv`, `fixed_eval.csv`, `results.md` |
| Replay | `emu export-policy`, `emu viewer` | policy array/normalizer fidelity, source/collider provenance and read-only checkpoint access | `.emupolicy` and browser visualization |

The browser is currently a **second implementation**: it runs real
MuJoCo **WebAssembly** physics with the frozen network locally, but its
terrain and random episode state are **representative**, not the exact
batch of JAX/MJX prism-worlds from training. Viewers should explicitly
display that distinction. A genuine live full-fidelity prism viewer
would require streaming simulator states/geometry or porting the exact
deterministic 18-prism terrain generation to the viewer and proving
replay parity. That is a future feature, not something a cleanup can
pretend already exists.

## One shared evidence-driven training loop

This is the intended operational interpretation of the v5044/v5045 code,
not six simulators or six competing networks:

1. **Host curriculum** (`training.py` + `curriculum.py`) reads the global
   step count and previously completed episode evidence. Through the 20M
   foundation, ordinary gait tasks take priority; the following 10M gradually
   activates specialist attention. This *time schedule* governs who teaches,
   not which physical frontiers have been earned.
2. **New episode birth** (`capability_focus.py` +
   `environment._focus_for_birth`) samples a teacher role from the six
   configured probabilities and an independent specialist-activation draw.
   An athletic retention rehearsal may override the difficult lesson. Neither
   role ID nor exact terrain parameters are privileged actor observations.
3. **One procedural environment** (`terrain_carpet_mjx.py` +
   `carpet_prism_mjx.py`) uses the sampled role and independently earned
   slope/up/down limits to generate the same kind of 3x3 true-prism contact
   window for every robot. CONTROL gets mild difficulty; STEP and SLOPE
   concentrate their own frontier while retaining easier exposure to others.
   The carpet recycles around the moving robot, not at arbitrary episode time.
4. **One PPO** consumes 360 deployable observation values and the same six
   servo action channels regardless of teacher role. Rewards use measured
   physics; role weighting only changes lesson focus, never the network.
5. **Aggregate evidence** (`results.py` and `training.py`) flows back to
   the saved governor after each PPO rollout. Its command sectors, physical
   slope directions, supported up/down lip windows and regime recovery
   decisions have distinct evidence thresholds, promotion and retreat rules.
   The next births sample the updated limits. KL/collapse monitoring can veto
   expansion, not counterfeit physical achievement.

**Clarification:** this is not yet a single differentiable N-dimensional
“capability map” driving every challenge. The checkpointed governor has
several complementary limit/window controllers, alongside the optional
command-neighbourhood `CapabilityField` teaching packet. Mechanical shaft
work in J/m is measured, and CONTROL has a small effort cost, but no single
adaptive efficiency frontier graduates like up/down steps. Such a controller
would require its own measured physical competence evidence, not a guessed
schedule or energy-only success threshold.

## Lean runtime: keep

| Mandatory module | Why it exists |
|---|---|
| `contracts.py`, `robot.xml` | physical body, observations/actions, sensor interface and checkpoint meaning |
| `cli.py`, `run.py`, `training_config.py` | CLI, immutable run recipe, provenance and resume approval |
| `training.py`, `ppo.py` | JAX rollout, optimizer, evaluation cadence, checkpoint scheduling |
| `environment.py`, `domain_randomization.py` | actual MJX state, actuators, contacts, sensors, noise and environment resets |
| `terrain_carpet_mjx.py`, `carpet_prism_mjx.py` | shared procedural generator, exact convex prism collision meshes, recycler |
| `rewards.py`, `curriculum.py`, `capability_focus.py`, `training_health.py` | payment, six specialists, evidence-based challenge and health |
| `checkpoint_io.py`, `results.py` | durable complete learning state, diagnostics and offline reporting |
| `policy_export.py`, `viewer_server.py`, `capability_viewer.py`, `viewer_src/src/{policy,sensor,sim,main}.js` | verified actor export, local HTTP physics replay, separate capability atlas and user controls |

**Retain but isolate as optional interfaces:** `evaluation.py` is the
trainer's read-only held-out evaluator; `prism_exam.py` independently
scores frozen 360-input actor traces from true prism contacts.
`capability_field.py` continues the active rollout-driven map; the atlas
viewer, showcase publisher, physics probes and diagnostics stay separate.

## Proven obsolete in the active path (removed from this branch)

- `emu/terrain_spike.py`: old standalone finite-step spike world.
- `emu/terrain_carpet.py`: pre-MJX deterministic specification, not
  consumed by actual prism contacts or curriculum. Its obsolete unit
  tests are replaced by tests of `terrain_carpet_mjx.py`.
- `scripts/smoke_step_course.py`, `scripts/probe_carpet_hfield.py`:
  old slab and hfield spikes. The current probe validates 18 real
  six-vertex prism meshes, physical walls, actor reset and stepping.
- `emu/viewer_src/src/training-carpet.js` is **retained** as the browser's
  metrics CSV parser and representative role/frontier adapter. Its
  `training-solid-carpet.js` companion still builds illustrative nine-box
  MuJoCo terrain, **not** the GPU trainer's 18 moving convex prisms.
  The old *MJX training* nine-box collision path is what was removed.
- `scripts/smoke_unified_focus.py` and old box/hfield probes: removed;
  the active single-policy lifecycle and prism physics probes replace
  them. Their old source remains in Git history.
- The viewer now accepts only schema-3 **360-input array-only** actors.
  Old 220/232 tripod-based policy readers and sensor simulators were
  deleted. The published v5032 260M champion was inspected and
  confirmed to use the retained exact array-only contract.
- The obsolete slab exam, directional CLI, legacy physical-quality scorecard,
  finite-platform examiner and duplicate standalone scorer are deleted.
  `emu prism-exam` uses one true-prism trace and independent raw contacts.

## Legacy, not automatically dead

- The v5045 `EmuMJXEnv` constructor now **rejects** old 220/232
  actors, slab scenes and non-prism terrain. Old experiments are recoverable
  only from their historical source commits, not runnable against the new
  constructor. Do not claim backward-compatible physical exam execution.
- **Prism-only collision scene no longer creates the nine old box
  bodies at all.** Tile recycling uses a conservative test against
  each *actual* half-metre prism square in XY, expanded by each foot's
  spherical radius and a 4mm safety margin, and only moves abandoned
  slots once both feet clear. Nine inert 2×2 hfields remain as **source
  vertex samples** for the analytic height, sensors and mesh deformation;
  they are not collision surfaces. Old nine-box modes live only in Git history.
- The Academy reuses the shared prism geometry; old three-platform
  lesson bodies are no longer compiled in this branch.
- `directional.py` remains used by live sensor estimation, locomotion
  control and PPO; deleting it wholesale would break the 360 actor.
- Historical experiment documents and old handovers are now **removed
  from the working tree**; the complete original contents remain in
  immutable Git commits. Only the current runbook and physics proof
  stay next to the source.

**Crucial:** disabled modes behind an ordinary Python
`if self.terrain_carpet_prisms` static flag may disappear from the JIT
compiled graph. A smaller Python source file does NOT by itself mean a
faster RTX 4060 Ti runtime. Delete based on dependency/coverage proof;
optimize based on synchronized GPU measurements.

## The staged trim map

| Priority | Work | Proof before removal |
|---|---|---|
| P0 | Preserve killed v5044 source/checkpoints; keep a short current handover | original handover and all links preserved; no Python behaviour change |
| P1 | Draw/import-audit actual runtime dependencies; label legacy/dev-only/public showcase paths | import graph, CLI dispatch, tests and viewer imports |
| P2 | Extract physics groups from `environment.py` into focused modules **without changing numerical meaning** | exact old-vs-new contacts, ToF, reward, PPO and checkpoint shape tests |
| P3 | Retire obsolete box/slab paths **from the prism-only runtime** after replacing recycler guard with prism-true clearance | 4cm and 40cm walls, diagonal contacts, first-life support, safe 3×3 recycling |
| P4 | Consolidate CLI run/export/viewer boundaries; split UI/WebGL physics from policy/schema | fixed exported actor byte/field parity, browser sensor parity, champion showcase regression |
| P5 | Archive tested-unused modules and redundant old docs; delete only proven unreachable code | affected tests green, release/legacy path comparison and documented rollback |

## Physical examiner migration

Historical `emu evaluate`, `emu directional-exam`,
`emu capability-exam`, `physical_exam.py` and `physical_quality.py`
have been retired with their incompatible two-slab assumptions.
`emu prism-exam` is the new read-only frozen 360-actor route; it
runs on real MJX prisms, independently verifies signed supported
new-tile crossings and reports measured slope, tracking, survival and
shaft work. **Requested severity is not guaranteed exposure.**
`evaluation.evaluate_fixed` continues to grade training snapshots
on real prisms. The new frozen-policy examiner still requires an
end-to-end same-source GPU acceptance run before release.

## Performance programme (no invented speedup claims)

The matched source-pinned benchmark (RTX 4060 Ti, 2,048 worlds,
65,536 steps per PPO update) was **17,774** nine-box and **7,492**
true-prism world-steps/s, 1.37 vs 1.90 GB peak. The resulting
**2.37×** time cost is a measurement of all PPO activity, not a
profile assigning the whole loss to mesh collision.

Measure and keep separate:
1. **JAX compile once** (start+first update): PRISMS 123.7s;
   BOXES 66.1s. Do not report this as steady-state FPS.
2. **Physics step / collision**: batched substeps and mesh deformation.
3. **PPO update**: four-epoch gradient and observation normalization.
4. **Host overhead**: metric copy/summaries, checkpoint serialization.
5. **Eval and viewer**: separate small held-out worlds and browser
   rendering; never combine their wall time with the GPU rollout.

v5045 replaces **18 prism mocap scatters and nine source-height
tile scatters** at birth/recycle with one batched scatter per group.
The source-tile updates are JIT-tested against their original scalar
versions, as are the physical prism mocap updates. The second
experiment removes nine inactive legacy-box bodies from prism physics,
avoids their redundant pose and rotation calculations, and uses the
real vertical prism footprint for safe recycling. Unit tests compare
old scalar versus JIT batched updates and exercise foot-contact/clear
boundaries. The second experiment changes the collision model's inert
body count; **do not resume a v5044 checkpoint with v5045 code**.
This is a *compilation/operation-count hypothesis*, not a measured
training speed improvement. Before considering a merge or a future
training run, repeat the identical `--benchmark-updates 3` test and
real multi-world physical probes. Use a **new experiment directory**
for changed source; never load v5045 into an old pinned v5044 run.

## The deliberately small test suite

**Exactly 24 maintained test functions**, split by responsibility:

| Test file | Functions | What it protects |
|---|---:|---|
| `tests/test_core_run.py` | 6 | 140M prism CLI, restart recipe, 360-input checkpoint integrity, real-prism evaluation |
| `tests/test_core_physics.py` | 11 | 18 real prism triangles, actual 3° grade, signed 3mm steps, recycling, ToF, independent supported-lip evidence |
| `tests/test_core_delivery.py` | 1 | Numeric Python/JavaScript agreement on a real schema-3 360-input actor |
| `emu/viewer_src/tests/essential.test.mjs` | 6 | Actor schema, raw 4×4 rays/occlusion, joystick/yaw hold, shipped champion, solid browser contacts, prism preview warning |

**Every other historical test and the 232-input archived-policy fixture has
been deleted from the current working tree**, not hidden or skipped.
To reproduce old research tests, check out the original v5044 source SHA.

Run the complete fast maintained suite:

```bash
python -m pytest -q tests
npm test --prefix emu/viewer_src
```

The following are **operational acceptance checks, not more unit-test
files**. They prove actual MJX dynamics or GPU PPO/checkpoint lifecycle
which a simple Python assertion cannot substitute:

```bash
python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --benchmark-updates 3 --prisms
```

All maintained physics, GPU and diagnostic probes now explicitly use the
**v5015 attitude profile and 8° pitch limit** of the pinned 140M recipe.
Older stand-alone GPU throughput numbers used the retired v5014 default;
treat those as historical measurements, **not** a matched performance
comparison for the v5015 launch. Repeat the full-PPO benchmark under the
same profile and source SHA before making any speedup claim.

Before starting a new 140M training lineage, also execute the
`smoke_training_lifecycle.py first / continue / verify` GPU checks on
the **same source SHA**. No unit test proves the policy learned a step;
only independent frozen-policy physical exams do.

We have deliberately **not** added a test that tests the test-count.
The list above is a simple maintenance budget, not another framework.

## Permanent boundary rules

1. Exactly **one** PPO actor/critic and six target actuators, regardless
   of six curriculum teacher roles.
2. Teacher privileged geometry and physical exam truth **never**
   become additional policy observation channels.
3. All terrain modalities agree on collision shape, top height,
   sensor raycasts and supported step scoring.
4. Existing checkpoint/source hashes and 360-observation contracts
   are versioned invariants. No silent schema conversions.
5. Examinations are read-only and physically independent of rewards.
6. Browser representative ground is clearly identified; no phantom
   "identical to MJX" claim.
7. Historical branches and source-pinned research stay reproducible.

This document is the **lean current entry point**. The earlier broader
[original architecture](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/ARCHITECTURE.md) and
[original engineering handover](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/HANDOVER.md)
preserve the historical detail.

## Viewer module separation (v5045)

`policy_export.py` now owns frozen checkpoint-to-actor normalizer/weights
encoding and portable actor validation. `viewer_server.py` owns the
loopback HTTP server, read-only checkpoint watcher and browser serving.
`capability_viewer.py` owns the separate field-atlas web app. They were
originally 1,032 lines combined in one file; each can now be tested,
modified or deleted separately. The CLI and physical exams call their
owning module directly. Actor export and viewer state never feed PPO.

## Retired standalone browser Prism Lab

The separate v5042 experimental Prism Lab (`prism-lab.html`, its own
procedural geometry script and CSS) was a **visual-only sandbox** and was
not the current MJX collision world or the robot-following browser scene.
It has been removed along with its two main-viewer links and extra Vite
entry point. The real integrated viewer's Field and Training Carpet
modes remain, with their collision-preview caveat. The exact old Prism
Lab lives in the preserved v5044 Git history.

## v5045 — runtime-only slab scoring removal

The prism-only `environment.py` no longer carries dead floor/upper-slab
step-height evidence or the averaged nine-box slope fallback. On-prism
events use signed, supported shared-edge crossings; slope evidence is
sampled from the physical triangle containing the root. The PPO frozen
trajectory packer also drops its unreachable two-slab return path,
while the GPU smoke test checks the static floor directly rather than
accidentally indexing `geom_contype[-1]`. The frozen v5044 trainer remains
on its original SHA.

## v5045 — direct prism-only MJX model construction

The true-prism branch now constructs physics **directly from robot.xml**:
`add_carpet_geoms(robot.xml, prisms=True)`. The original static floor
is retained by name for diagnostics but filtered from contact. It no longer
first constructs two **64-metre historical unified slabs** merely to
bury their mocap bodies 30 metres below every new prism episode.
Current prism births do not touch any historical slab IDs. The 18 convex
prisms and nine noncontact vertex-sample hfields remain; saved v5044
checkpoints must not be resumed with this new model topology.

## Follow-up: one physical control frame, not 59 anonymous positions

As of the 4 November v5045 cleanup, `environment.py` has one legal
constructor contract: 360 history inputs, one 4×4 array-only sensor,
one 3×3 18-prism collider carpet and six sampled lesson roles. It is
not a four-mode general simulator. A failed geometry/schema contract
is raised on construction, before MJX compilation.

The per-world `EnvState` now names its 52 surviving data fields at
reset; `_step_impl` only replaces fields genuinely advanced on that
frame. The old terrain-cross streak, two ascent high-water counters,
verified-height peak and one-slab careful-approach frame counter were
never meaningfully updated by the true-prism path, and were removed.
The active `terrain_crossed` flag and independently verified crossing
and episode evidence **were preserved**.

The `rewards.py` equation now has no unreachable 120-frame single-lip
underspeed/crouch discount or dormant offline v5031 ascent-credit
utilities. Experimental temporal velocity tracking is still supported
when explicitly enabled. `evaluation._world` refuses any non-prism
fixed-eval world; the browser's nine-box representative preview remains
honestly labelled as a visual approximation, not physical proof.

**Still not a license to train:** a successful 4-world physical probe
checks resets, prism geometry, contacts and recycling, not the complete
same-SHA 8-world CUDA checkpoint/resume/export lifecycle or the actual
exported-policy frozen prism exam. Keep PR #34 draft until those pass.

## Follow-up: isolate reported evidence from the control frame

The one physical control frame in `EmuMJXEnv._step_impl` now has a
straight-line JAX path: apply motor forces and physics; advance the
next teacher command; score the previous operator command; update
episode accumulators; acquire the next 360-input history; recycle and
conditionally reset; publish physical reports.

The detailed `slope_episode_*`, `academy_*` and
`terrain_episode_*` metrics are built in `_step_report` from the
pre-auto-reset `stepped` state, the original task and the physical
prism heights. No additional network or world exists. The report
dictionary was compared byte for byte with its pre-extraction
implementation; the `training_reward` call was also unchanged.

`EnvState` has 52 compulsory fields, all named at reset and validated
against the per-frame `_replace` names. Two inert Academy
clearance/transfer *state* bits were removed; their always-false
diagnostic events remain available to existing reports. Contact-based
footholds, supported shared-edge crossings and repeated lip rewards
remain intact.

### PPO evaluator ownership (4 November continuation)

The maintained learner exposes four operations its live callers
need: `make_initial_state`, `train_update`, plus the frozen evaluators
`evaluate_summary` (matched aggregate panels) and `evaluate_trace` (real
prism-exam contact/sensor trajectories). Retired `evaluate_efficiency`,
`evaluate_recovery`, and `evaluate_node_window` were stand-alone research
kernels with **no callers in the current physical exam/training path**;
remove them rather than compiling dead JAX callbacks or maintaining a
second definition of exam competence. Measured efficiency remains in
on-policy physical telemetry; frozen policy exam retains raw signed contact
facts. No PPO optimizer, actor or critic mathematics changed.

### Strict v5045 curriculum/evaluator continuation

The live governor no longer loads incomplete old checkpoints or reconstructs
40M-era pooled terrain, Academy, slope, or command evidence. Its complete
saved-state key set and independently measured per-direction counts must
match exactly; the 24 maintained core tests now exercise a valid
roundtrip and reject old partial layouts. A fresh source-pinned run is
required for this altered checkpoint contract. Historical policy artifacts
remain readable using their matching Git commits, not synthetic defaults.

`advance_terrain` now contains only the real-prism STEP near/easy evidence
controller. Both UP and DOWN must meet their own near-height evidence
thresholds; failures on easier terrain can retreat the height. The
unreachable non-unified terrain governor and its pooled counters were
deleted without changing this active controller's height increments,
success thresholds, or safety limits. The generic velocity/yaw/combo
frontier loop, which had already been disabled in unified training, was
removed; `_advance_specialist_commands` retains sole authority over
those frontiers. Resilience, sensor, and transition control retain their
existing active progression rules.

`results.format_line` now always reports the actual prism-gradient target
and measured terrain occupancy, not a retired gravity/rotated-plane
fallback. The current learner retains the two necessary JAX evaluators:
`evaluate_summary` for fixed panels and `evaluate_trace` for frozen
prism contact evidence. Three uncalled standalone evaluators were
removed. These code deletions are not evidence of a GPU throughput gain.

**Open gate:** the latest head requires full branch CI, same-source CUDA
PPO/checkpoint lifecycle and an exported frozen-actor true-prism exam.
Do not start or resume a 140M campaign on prior checkpoint/source SHAs.
