# EMU core skeleton — run to robot walking in the viewer

**Audit branch:** `feature/v5045-core-skeleton-audit`, 3 November 2026.
**Source of truth for a running 140M campaign:** the exact pinned v5044
training commit, **never this refactor branch**. Do not merge or cherry-pick
v5045 changes into a running training checkout. Champion/main and archived
sensor contracts are still used for reproducibility.

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
          physical_exam.py                     |
          physical_quality.py                  v
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

**Retain but isolate as optional interfaces:** `evaluation.py`,
`physical_exam.py`, `physical_quality.py`, `capability_cloud.py`,
`capability_field.py`, the capability atlas viewer, showcase publisher,
independent physics probes and diagnostics. Most of these are imported by
the training implementation even if a particular 140M recipe disables
their heavier operations. They must not be dropped on a size guess.

## Proven obsolete in the active path (removed from this branch)

- `emu/terrain_spike.py`: old standalone finite-step spike world.
- `emu/terrain_carpet.py`: pre-MJX deterministic specification, not
  consumed by actual prism contacts or curriculum. Its obsolete unit
  tests are replaced by tests of `terrain_carpet_mjx.py`.
- `scripts/smoke_step_course.py`, `scripts/probe_carpet_hfield.py`:
  old slab and hfield spikes. The current probe validates 18 real
  six-vertex prism meshes, physical walls, actor reset and stepping.
- `emu/viewer_src/src/training-carpet.js` old raster-heightfield
  generator: removed, with a fail-closed assertion that browser carpet
  must have actual collision solids.
- `scripts/smoke_unified_focus.py` and old box/hfield probes: removed;
  the active single-policy lifecycle and prism physics probes replace
  them. Their old source remains in Git history.
- The viewer now accepts only schema-3 **360-input array-only** actors.
  Old 220/232 tripod-based policy readers and sensor simulators were
  deleted. The published v5032 260M champion was inspected and
  confirmed to use the retained exact array-only contract.
- The old single finite-platform terrain examiner and `terrain-exam`
  CLI route were deleted. The current independent 360-input capability
  examiner and signed prism-step course remain.

## Legacy, not automatically dead

- `environment.py` still supports older 220/232-input policies,
  legacy step platforms and non-carpet terrain paths. The v5044 prism
  mode does NOT use their active collision surfaces, but existing
  training/source and frozen-policy reproducibility depend on them.
- **Prism-only collision scene no longer creates the nine old box
  bodies at all.** Tile recycling uses a conservative test against
  each *actual* half-metre prism square in XY, expanded by each foot's
  spherical radius and a 4mm safety margin, and only moves abandoned
  slots once both feet clear. Nine inert 2×2 hfields remain as **source
  vertex samples** for the analytic height, sensors and mesh deformation;
  they are not collision surfaces. The old nine-box model still exists
  only as a separate benchmark mode and is a later removal candidate.
- `step_academy_world.py` is the older separate slab machinery. The
  v5044 Academy uses the shared prism carpet, but old tests and policy
  worlds still require these definitions.
- `directional.py` is large but is used by live PPO/contact
  computation **and** physical exams; it is not safe to delete wholesale.
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
| P0 | Freeze running v5044 SHA; archive oversized handover; keep one short current handover | original handover and all links preserved; no Python behaviour change |
| P1 | Draw/import-audit actual runtime dependencies; label legacy/dev-only/public showcase paths | import graph, CLI dispatch, tests and viewer imports |
| P2 | Extract physics groups from `environment.py` into focused modules **without changing numerical meaning** | exact old-vs-new contacts, ToF, reward, PPO and checkpoint shape tests |
| P3 | Retire obsolete box/slab paths **from the prism-only runtime** after replacing recycler guard with prism-true clearance | 4cm and 40cm walls, diagonal contacts, first-life support, safe 3×3 recycling |
| P4 | Consolidate CLI run/export/viewer boundaries; split UI/WebGL physics from policy/schema | fixed exported actor byte/field parity, browser sensor parity, champion showcase regression |
| P5 | Archive tested-unused modules and redundant old docs; delete only proven unreachable code | affected tests green, release/legacy path comparison and documented rollback |

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

v5045 first experiment replaces **18 sequential prism mocap scatter
updates** at birth/recycle with **one batched scatter**. The second
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
for changed source; never load v5045 into the active pinned v5044 run.

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
