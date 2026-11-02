# EMU — active engineering handover

**31 October 2026 · New baseline: v5032 Pinnacle 260M**

The owner elected to promote the completed v5032 360-input
Pinnacle as the new `main` model, **with its known weaknesses
documented** rather than starting another experimental branch.
The 24-arm v5033 science work is **paused** and not released.

## What is the robot?

One six-servo dynamically unstable MJX biped, 360-input
4×4 forward ToF + IMU/optical-flow / proprioception observations,
one shared PPO actor. No programmed walk cycles. Real signed
slope and sharp step contacts, six independently growing
teacher focus groups. The actor does not receive privileged
teacher terrain truth. See [ARCHITECTURE](docs/ARCHITECTURE.md).

## The frozen trained baseline

| Fact | Value |
|---|---|
| Completed world steps | **260,046,848** |
| Training source SHA | `9b11bbe596aea7201d29a6beb2d192922f8e2e9a` |
| Source branch at training | `release/v5032-pinnacle-candidate` |
| Environments and seed | 2048, 761 |
| Observation / action | 360 / 6 |
| Policy artifact SHA-256 | `ef929c0220c38e5519391320e5f0629a200e1ef37a5808086f6383cfb63f7ec8` |
| Checkpoint SHA-256 | `238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b` (report; full file is on WSL) |
| Local run directory | `~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/` |
| Published policy location | `showcase/champion.emupolicy` |

`main` may have later documentation/release commits; the
actor's **training source stays frozen at its original SHA**.
Do not relabel those weights as a different training recipe.
Keep the whole `final.pkl` separately backed up outside Git.

**Important recipe discrepancy:** the launch manifest requested
verified-ascent reward reallocation 3.0, but the actual saved
`TrainingConfig` had **0.0** because the trainer's CLI
forwarding was missing. The released actor did not learn that
treatment. Recheck **resolved training config**, not CLI printouts,
before any future PPO campaign.

## What does the measured robot do?

Independent 16-world CPU MJX physical exams on real collision
surfaces (seed1701, 7s terrain, 14s speed):

- **Step-up:** 1–2cm 8/8; 2.5cm 4/8; 3cm 0/8.
- **Step-down:** 3cm 8/8; 3.5cm 1/8.
- **Slopes:** uphill12° 15/16, both cross-slopes12° 16/16;
  downhill12° survival 16/16 but heading-controlled 4/16.
- **Speed:** at requested 0.6m/s, 16/16 survived but 0/16
  passed full heading-controlled mission; at 0.8m/s survival
  was 12/16 and strict mission 0/16.
- **Frozen overall score:** 0.8296 at 260M.

Do not turn these into a claim of 1.0 m/s or 3cm mastery.
In particular, downhill 12° at 0.15m/s was NOT a sprint or
balance failure: mean signed yaw drift disqualified the
heading-control mission.

See [full release measurements](docs/V5032_PINNACLE_RELEASE.md)
and the [research backlog](https://github.com/mdt733/emu/issues/17).

## Publication and rollback

The intended release changes the *code branch and the portable
policy together*, using the exact policy SHA and approved
`showcase/release_approval.json`. The frontend is validated
by `scripts/verify_showcase_release.py` and GitHub Actions.
The GitHub Pages workflow uses explicit manual release
dispatch; merging source alone does not deploy it.

**Previous main rollback:** `2ec9e9db0a816c658131311129ef89ec7a595a12`
(the old 232-input champion, approval file, viewer and code).
Retain this immutable Git SHA, not a casually renamed ZIP.

The original long handover was preserved as
[Handover v5032 before release](docs/history/HANDOVER_V5032_PRE_RELEASE.md).
The new README and [architecture map](docs/ARCHITECTURE.md)
are the entry points for somebody new to the project.

## What to work on next (not yet)

**Read-only diagnostics first.** In order:

1. Separate downhill signed yaw-rate bias, alternating gait
   jitter and **world-frame heading/path drift**; use a 5-speed
   × 4-grade frozen test before changing rewards.
2. Trace real supported 2.5–3cm step-up crossings: knee/hip
   extension, upper-contact duration, lower-foot stance,
   torso dip/rebound and spring energy *release*.
3. Investigate the persistent height reward after foundation.
   A 5cm pre-step crouch is heavily discouraged even though
   springs can physically load; test a temporary near-lip
   allowance only in a future **single controlled experiment**.
4. Study 1mm step-frontier adjustment versus historical
   +18%/-18% jumps, with more completed evidence per direction.
5. Diagnose high-speed heading/turn authority, damping and
   meaningful shaft J/m without rewarding stationary behaviour.
6. Only afterwards revisit organic hills, traversability
   refusal and a future from-scratch Pinnacle.

The complete categorised backlog is
[GitHub issue #17](https://github.com/mdt733/emu/issues/17).
It includes negative tests and acceptance criteria. No new
PPO science arms have been launched in this promotion.

## Guardrails

- `main` is the new **reference**, not a scratchpad for training
  variations. Work on one explicitly justified change at a time.
- The abandoned `feature/v5033-260m-plateau-science` branch
  contains **unvalidated experimental code**. Do not execute
  its 24-arm launcher or mistake its changes for released EMU.
- Keep the old 232 policy recoverable from rollback SHA.
- Published actor `.emupolicy` is inference-only; full
  checkpoint `final.pkl` has optimizer/teacher/normaliser state.
- Use common independent exams with actual supported step
  crossing and achieved velocity, not curriculum headlines.

[README](README.md) ·
[Release measurements](docs/V5032_PINNACLE_RELEASE.md) ·
[Architecture](docs/ARCHITECTURE.md) ·
[Documentation index](docs/README.md).


---

## Experimental v5040 STEP improvement — 2 November 2026

Draft [PR #28](https://github.com/mdt733/emu/pull/28) on
`feature/v5040-multi-lip-academy-world`; keep `main` and all checkpoints
unchanged. The existing STEP specialist now samples four consecutive real
up/down lips, with fresh random ordering per episode, real contact-gated
per-lip incremental rewards, 4×4 physical ToF, a separately checkpointed
height frontier, and original shared 360-input PPO. Other specialties and
the 20M foundation remain. Prior CPU checks exercised physics, not learning.
Added research launchers, test suite, and Academy CI matrix have been
removed from this branch to keep it maintainable; historical research
remains in older branches. **Not GPU-profiled or trained yet.**
See [short v5040 guide](docs/V5040_REAL_MULTILIP_ACADEMY.md).

### v5040 refactor hygiene

The active research branch now separates model uncertainty (`emu/domain_randomization.py`),
learner health (`emu/training_health.py`) and CLI diagnostics (`emu/doctor.py`).
The obsolete v5037 Stage A flags and historical v5032 launcher were removed
from this branch (their original Git history remains). The full Python/browser
checks are optional via `scripts/check.sh --full`; CI retains one actual
physical MJX/PPO step and the browser build. No v5040 GPU training has run.

### v5040 streaming terrain (untrained)

A ring of three physical 50cm Academy supports now recycles the trailing
platform after five real supported far-side frames, both feet safely
beyond the edge and no trailing-slab contact. Each new lip chooses UP
or DOWN independently, without a balancing pattern. The Academy governor
holds distinct UP/DOWN supported-crossing frontiers, each initially 3mm,
advancing/retreating by 0.5mm on that direction's physical evidence.
New steps grow from 50% to 150% of the directional frontier over an
episode. The default aspiration is 15cm, NOT demonstrated capability.
There is no fixed course finish and no auxiliary CONTROL effort shaping.
The generalist's other specialists and 20M foundation are unchanged.
Must still pass actual 2048-environment GPU PPO preflight before training.

### v5041 actual viewer integration — 2 November 2026

Research child branch `feature/v5041-unified-terrain-carpet`, based on
v5040. Adds `Training Carpet` **inside** EMU's existing 3D policy viewer,
not a new standalone visualizer. Original `Terrain Park` remains a
separate mode with its trees and all eight physical obstacle fixtures.
The carpet mode reuses browser MuJoCo, grass/solid trees, the existing
policy/360 sensor/ToF and camera stack. Settings expose specialist limits,
real `metrics.csv`/JSON import and live `/api/live/terrain` readings via
`emu viewer --live --root <run_dir>`. Changes are applied only on user
request to avoid resetting a good walk mid-episode.

The 3×3 golden guide follows EMU and shows 18 conceptual triangles.
This browser implementation has a finite sampled **physical hfield**:
its steep step edges are approximations, not exact vertical risers;
the 3×3 collision ring is not yet present in GPU training. Source includes
`emu/terrain_carpet.py` specification and a 2×2 MJX mutable-hfield CPU
proof. Do not infer 2048-env performance or learned competence from this.
See [v5041 actual viewer runbook](docs/V5041_TRAINING_CARPET_VIEWER.md).

### v5041 main GPU terrain generator — first training candidate

Branch `feature/v5041-unified-terrain-carpet` extends the v5040 feature
without touching `main`. `emu train --terrain-carpet` instantiates NINE
real, reusable 2x2 hfield colliders (18 triangles), with per-world 36
JAX-mutable height values and modulo-three tile coordinates. The shared
six-role sampler drives grade/step envelopes with gentle cross-branch
carryover. Separate ascent/descent Academy evidence is still physically
supported. Old long slabs and Academy box supports are buried in carpet
births. The actor retains exact terrain_360 observations. The 4x4 ToF,
optic depth and support-height computations use the tile samples. GPU
2048-env VRAM and exact lip-riser contacts must be validated on the RTX
4060 Ti before a long training run; see
`docs/V5041_GPU_CARPET_TRAINING.md`. One 4-world CPU PPO update passed
on GitHub Actions. Continue only on feature branch. DO NOT assume the
viewer finite heightfield is the GPU engine.

### v5041 latest physics hardening

Removed obsolete three-platform Academy collision bodies from carpet
mode, disabled the two nonphysical legacy long-slab contact flags,
preserved teacher Academy births separately from terrain geometry and
ensured they share exactly one sampled assignment. Extended 4x4 ToF
simulation to the real vertical hfield sides, added locally varying
shared-corner slope gradients with configurable maximum transition
angle, and introduced 64-element per-environment exact recently visited
tile memory to prevent rocking/revisiting reward farming while allowing
real new multi-lip crossings before/after Academy allocation.
Contact parity tested with real MJX and CPU MuJoCo at 4cm and 15cm.
The remaining acceptance gate is full 2048 CUDA PPO on RTX4060Ti 8GB,
plus a source-pinned bounded lifecycle and physical commanded crossings
before large cold run. Keep `main` and v5040 intact. Runbook:
`docs/V5041_GPU_CARPET_TRAINING.md`.

v5041 safety gate: synthetic MJX+CPU tests must prove horizontal
step-riser normals and upward-facing foot-top normals. The live
carpet STEP reward explicitly requires a physically supported sphere
bottom at the *actual* collision triangle top, and refuses side-contact,
diagonal phantom-lip or repeat-visited tile rewards. Off-grid support
reports finite fall height instead of infinity. Execute the
`scripts/smoke_gpu_carpet.py --full-update` 2048-CUDA acceptance
before any substantial run; no such acceptance has yet been run by
the assistant on the user's GPU.

### v5041 collider redesign — real solid tiles

After strict contact tests exposed incorrect MJX hfield-skirt upward
normals, an isolated real-MJX test established **box-only 50cm tiles**
give the correct horizontal riser normal and ~0.024mm z drift over
eight steps. The earlier hfield+12-wall combination retained the
spurious upward hfield contact. The experimental v5041 trainer now
derives nine sloped solid-box top surfaces from the same 18-analytic-
triangle height samples. All hfield collision flags are OFF; the
hfield sample vectors remain read-only source inputs to the sensor
and reward estimators. No old Academy slabs; no 12 seam wall geoms.
This is a per-tile planar collision approximation, not perfect
noncoplanar triangle collision; compare <8mm divergence and high-step
foot contacts. Preserve v5040 and main; 2048 CUDA still untested.
See docs/V5041_GPU_CARPET_TRAINING.md.

Latest v5041 verification: 9 tilted solid-box MJX physical surfaces
passed real CPU 4-world PPO and stricter 4/15cm top/side contacts in
Actions run 41433542901. ToF (all 16 4x4 rays) and optical-flow depth
were then switched to EXACT ray/OBB intersections against the active 9
MJX box geoms, not inactive HField samples; teacher support and STEP
footholds use the same exact boxes. The analytic triangle height contract
is only used for initial source geometry, with bounded discrepancy.
Run new full CI before requesting 2048 CUDA preflight; remember to
update source SHA in acceptance commands. v5040 and main untouched.


### v5041 first actual 2048-world CUDA gate — 2 November 2026

The user cloned source SHA `c876240f935d1acc4346b229f84381286f314ace`
in a separate WSL checkout and exercised the RTX 4060 Ti 8 GB:

- 128-world physics, finite 360 observations and real PPO: **PASS**.
- 2048-world physics plus 32-frame PPO rollout, 4 epochs and 8 minibatches:
  **PASS**. No GPU OOM; real 9-box recycling and top-surface checks passed.
- Source triangle/physical box top sample mismatch: 2.61mm (128 worlds),
  2.71mm (2048 worlds), both within the 8mm smoke threshold.
- JAX allocator reported 1,075,855,360 bytes peak in use against
  6,440,353,792 bytes device limit (not measured sustained-run VRAM).
- Warnings about unavailable optional warp modules and CUDA timer accuracy
  were nonfatal.

**Important source-pinning distinction:** The passing CUDA code was
`c876240f`, not any subsequent feature-branch commits. Later improvements
add a positive PPO *actor* parameter-change assertion, timing diagnostics
and real rotated-box/18mm spherical-foot recycler clearance plus tests.
CPU CI and then both CUDA smokes need repeating on the final source SHA
before accepting that newer variant.

**Next required gate:** a separate 2-update `EMU_UNIFIED_ACCEPTANCE=
bounded-ppo-lifecycle` CUDA `emu train` run that verifies complete optimizer
checkpoint, resume/source provenance, exporter and 360-observation actor.
Then independent physical commanded step/slope exams and only afterward
a new, source-pinned cold-start 140M run. See
`docs/V5041_GPU_CARPET_TRAINING.md`; main and v5040 stay untouched.


### v5041 update — CUDA acceptance AND warm lifecycle PASSED

**Supersedes the earlier "CUDA/lifecycle still pending" text above.**
The owner reran exact source
`728f0139b3f032dcc68770372730ccfb36494ac6`:

- 128 and 2048 world CUDA smokes PASS, including real 9-solid
  geometry, true box/ball-foot clearance, finite 360 observations,
  and positive actor PPO delta (0.001431 at 2048).
- Production-shaped PPO = 2048×32=65,536 world steps, four epochs,
  eight minibatches; 64.68 seconds INCLUDING first compilation.
- JAX peak memory 1,075,855,360 bytes/limit 6,440,353,792 bytes.
- True carpet first/continue/verify lifecycle PASS at 2048 worlds:
  65,536→131,072 total steps, source-pinned optimiser/parent-preserving
  warm continuation, obs normaliser count advances identically,
  completed-resume is a no-op, 360 schema-3 export/readback PASS.
- Novice checkpoint does NOT walk: ~77.6% early falls are genuine.
  The ~2 million FPS and reward=0 console prints were reporting bugs,
  not evidence of ultra-fast/zero-reward training. The wall timer
  excluded compile+first PPO, and the one-update reward list was empty.

**Engineering work subsequently added ON THE SAME EXPERIMENTAL BRANCH
but NOT yet independently GPU-approved:**

- Correct final reward and compilation-inclusive small-run wall FPS;
  exclude warmup steps from perf's postwarmup-only train-FPS numerator.
- Frozen `evaluate_trace` now gathers actual live nine-box contact
  pairs, local tilted normals and source/solid geometry. It must not
  `stack` the absent old Academy platform bodies.
- Exported `.emupolicy` records nine-box training topology and
  Academy settings. The three-stage checker refuses a carpet export
  that lost this provenance.
- `scripts/exam_step_course.py --carpet`: provenance-gated read-only
  per-lip exam, counts only first-life supported contacts, reports
  observed physical UP/DOWN heights and uncategorised attempts; zero
  attempted lips does not imply success. It ALSO independently rescores
  five-frame supported new-tile transitions using raw box transforms,
  actual foot-contact normals, ball-foot sphere clearance and the signed
  physical edge rise. Teacher and independent counts remain separate,
  with disagreements flagged rather than silently credited.
- `scripts/smoke_gpu_carpet.py --full-update --benchmark-updates 4`
  measures synchronized throughput from 3 post-compilation full
  PPO updates, no checkpoint or long-run launch.
- Regression tests cover first-life censor, real 9-box frozen traces,
  and earlier sphere/solid overhang recycler defence.

**Next:** get the latest Actions CPU CI green, pull/pin the new exact
SHA, run bounded GPU steady-throughput smoke, re-export existing tiny
untrained checkpoint with its real terrain provenance, and probe
read-only frozen carpet exam for pipeline soundness. Do not claim
trained 15cm skill or launch 140M merely from a two-update checkpoint.
Only after frozen independent contact/slope exams and real speed
are validated launch a fresh 140M lineage. `main`, v5040 and all
legacy champions/checkpoints remain untouched.
