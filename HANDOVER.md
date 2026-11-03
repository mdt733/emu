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

## Historical cold-start learning milestones (not curriculum gates)

**Preserve this baseline when interpreting short GPU smoke runs.**
The owner reports EMU generally needs around **5,000,000 world steps
before it starts learning to stand on simple flat ground**. In prior
discussions the owner described roughly **20M as barely standing** and
**40M as an infant/early gait**, not mature reliable walking. These are
**qualitative experience across prior runs**, not newly measured
source-pinned frozen-exam milestones; seeds and recipes vary.

The code's **20M main-style foundation** is a training-mixture
transition, **not** evidence that a 20M policy has mastered standing
or command-following. Don't infer learned locomotion or failure from
65,536/131,072-step PPO lifecycle smoke checkpoints. Do not claim
independent step ability from a single early downhill crossing.
Do not make all six specialist groups wait at an arbitrary global
competence gate. Compare actual physical standing, command-following,
survival and supported contacts at matched ages before judging a new
cold start. README also summarises this early learning curve.

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


### v5041 first frozen CUDA nine-box independent STEP exam — 3 November 2026

User pulled exact source SHA `39573722fc4b86ddabd9c2d2ec489d069129c79f`
and re-exported the 131,072-step bounded lifecycle checkpoint with
carpet provenance. Then ran the real CUDA read-only examiner on 8
nine-box worlds (200 frames, forward command 0.10m/s, nominal 1cm
UP/DOWN challenge, no slope, seed 1701):

- **Teacher credits:** UP 0, DOWN 1.
- **Independent raw-contact/box-plane scorer:** UP 0, DOWN 1.
- **Teacher approached-denominator:** UP 0/1, DOWN 1/3.
- **Falls:** 8/8 worlds, as expected for a nearly untrained policy.
- Independent scorer / teacher aggregate agreement: **PASS for this
  ONE smoke case**, not general scorer equivalence or demonstrated
  terrain capability; nominal 1cm is not an exact every-lip height.
- Optional Warp import warnings were harmless.

Frozen collision topology, terrain source provenance, physical exam
execution and separate independent-vs-teacher reporting have now
passed an owner-run CUDA case. No checkpoint was retrained or altered;
`main` stays frozen.

**Remaining pre-140M gates:** (1) real synchronized postcompile
2048-world PPO throughput (`scripts/smoke_gpu_carpet.py --device cuda
--worlds 2048 --frames 2 --ppo --full-update --benchmark-updates 4`);
(2) inspect observed signed heights and per-world independent raw
crossings in the existing `academy_exam.json` artifact; (3) broaden
independent terrain contact/step/slope exams on additional seeds or
early-stage real gait if gate evidence warrants. Avoid inferring
competence from a 131k-step random-ish actor or pretending the
first-update ~64s is steady training speed.


### v5041 complete sustained 2048-CUDA full-PPO gate — 3 November 2026

The owner ran `python scripts/smoke_gpu_carpet.py --device cuda
--worlds 2048 --frames 2 --ppo --full-update --benchmark-updates 4`
on source `39573722fc4b86ddabd9c2d2ec489d069129c79f`,
RTX 4060 Ti 8 GB / WSL2. **PASS**: 9 tilted solid boxes (actual
collision, 2.71mm source/plane difference), safe recycling,
real PPO actor gradient delta 0.0014353991, first including
compile 71.985s, followed by 3 synchronized *full* PPO updates
at **3.897s/update** or **16,818 world steps/s**, JAX peak in-use
**1,395,523,840 bytes** against JAX memory limit 6,440,353,792.
This measures four 32×2048-step, four-epoch/eight-minibatch PPO
updates, not just physics stepping. Raw 140M PPO estimate ~2.31h
plus training/eval/checkpoint overhead.

The earlier same-source 131k-stage training lifecycle and 8-world
read-only physical exam had already passed: teacher UP 0/DOWN 1,
independent UP 0/DOWN 1, 8/8 fell (expected untrained).
No proven uphill or slope skill yet. **Pre-long-run engineering smoke
gates are now satisfied**, and long **monitored research training may
start** from scratch with this branch in a NEW run directory after
checking `emu train --plan`, clean Git SHA and config. The current
README and GPU runbook correct the previous stale claims of 9 active
heightfield colliders and untested 2048 CUDA. Keep early growth
historical expectations (~5M standing onset, 20M barely standing,
40M immature gait), one PPO actor, six independent specialties,
20M ordinary foundation, and 60/80/100M STEP Academy share ramp.
Do not change source SHA midlineage; never merge experimental
topology into `main`. No 140M has yet been launched here.


### v5041 live physically equivalent viewer and safer 5M cold-start gate — 3 November 2026

Operator correctly challenged the initial "launch 140M" claim: CUDA
topology/gradient/memory/throughput and ONE 131k-step frozen examiner
demonstrate *engineering readiness*, not that a new source reliably
learns standing or six terrain skills. Correct research plan is a
**genuine cold-start ~5M** first, judge posture and survival against
historical 5M standing emergence, then **source- and optimiser-identical
extensions** to 20M and only then 140M. The validated existing
`emu/run.py` `_validated_same_source_extension` path allows
**finished** 5M `final.pkl` to `--resume auto --steps 20000000`
(and subsequently 140M); same clean Git SHA/config, new dedicated run
root, unchanged actor/critic/normaliser and teacher. This avoids a 2h+
all-or-nothing run, without transplanting weights.

The old browser Training Carpet was a finite 22m rasterised hfield
with decorative moving guide only; it was NOT the real GPU trainer
(the latter has 9 tilted moving solid boxes with actual vertical step
faces). The feature branch now builds **NINE real MuJoCo solid boxes**
with trainer-matched half-sizes (.258/.258/.270m), 50cm cells,
per-role signed step / smooth slope generator, 3x3 modulo-slot ring,
actual MuJoCo mocap recycling and conservative foot-overhang guard.
The previous eight-fixture Terrain Park remains separate.
`Settings → Terrain → Training Carpet → Auto` uses *read-only*
`/api/live/terrain` source/metrics and config to show **FLAT through
20M ordinary foundation**, then a *representative* slope specialist,
then STEP/multi-lip Academy after intro (~60M default). Real specialist
frontiers continue to evolve independently; a single browser cannot
replay all 2048 randomly seeded worlds. Small metric changes don't
reset the viewer; substantial changes do, without touching training
weights or checkpoints. Browser terrain is a representative physical
topology, **not exact seed/birth/reward/ToF parity**.

Critical sensor fix: the browser 4x4 array raycast previously filtered
out non-world-body geoms; nine moving terrain mocap boxes were
invisible to policy ToF. `actorTerrainDistance` now includes world and
mocap boxes but not articulated robot bodies, and removes a synthetic
heightfield ray fallback for carpet while keeping it for the old
heightfield Terrain Park. Added Node regressions and native MuJoCo
CI check of **actual 9-box** XML and mocap movement, replacing an
obsolete CI hfield assertion.

CI status and latest source SHA must be checked before the owner pulls;
native nine-box model CI success first seen on commit
`84e68970925dea7229f91706c31a36b2cb0263df`, with sensor
regressions committed later. User should run
`emu viewer-build` on pulled clean feature code and
`emu viewer --live --root ~/robot_lab/emu_runs/v5041_carpet_lifecycle_s735
--no-open` **before** any 5M run; the old 131k policy should see
a flat nine-box grid. Real browser WASM/manual observation remains
a final sanity check; do not describe CI-only validation as interactive
viewer proof. Full runbook: `docs/V5041_TRAINING_CARPET_VIEWER.md`
and staged launcher: `docs/V5041_GPU_CARPET_TRAINING.md`.


### Qwen prelaunch audit triage — 3 November 2026

The owner requested a second review of v5041 before the first 5M
cold-start stage. **Do not confuse retired v5040/browser prototypes
with the source-pinned active MJX trainer**:

- v5041's 360-input 4x4 ToF and optical-flow depth sample the **nine
  actual rigid box colliders** via `_ray_box_distance`, including
  vertical risers. Disabled 2x2 hfield geom data is kept as analytic
  terrain-generation/teacher metadata; `terrain_carpet_mjx.ray_surface`
  is NOT used by the actor's current `_terrain_4x4_ranges` path. The
  owner measured max 2.71mm source-triangle versus solid-box top
  discrepancy on 2048 CUDA worlds; this does not claim perfect
  identity of the slightly curved source triangles and planar box tops.
- `step_academy_world.signed_lip_height(progress)` is for the retired
  three-platform recycler. `_academy_birth` returns immediately for
  `terrain_carpet`, and `_recycle_academy_platform` is NOT called in
  its step path. Actual carpet lip profiles are global-cell-index and
  seed based; waiting in place cannot raise a current step.
- Frozen read-only `ppo.evaluate_trace` logs raw MJX contacts,
  vertically oriented normals, geometry and world cells but does not
  participate in PPO `jax.value_and_grad(loss_fn)`. It is independently
  JIT compiled, only in the exam path.
- Qwen's suspicion about anti-rocking uncovered a REAL subtler bug:
  the history used global cells and prevented immediate re-payment,
  **but the write counter also advanced on already-seen supported
  returns**. After sufficiently many redundant returns, older
  legitimate cells could be evicted early and paid again. Fixed by
  gating `note_confirmed_tile` **inside the helper** with
  `unseen_tile`, independent of the calling teacher, so ONLY first
  novel confirmed cells consume the 64-entry ring. The fixed-size
  history remains an intentionally bounded last-64-UNIQUE-cell ring,
  not unbounded lifetime memory. Added 320-revisit regression.
- Added **real CPU and MJX** normal/contact seam test with a sphere
  foot exactly at x=0 shared tile boundary for flat, 15mm and 8cm
  rises. Tested exact seam contacts, NOT dynamic learned step crossing.
- Existing `nan_grad_skips_total`, `exploration_std_mean`,
  `ppo_kl_stopped_frac`, `ppo_adv_tail_frac` and per-role
  curriculum events are real training metrics. Do not use a hard
  "adv_tail_frac >5%" alarm: `flat['adv']` is standardized
  before this records the fraction beyond FIVE standard deviations,
  so a 5% rule is not a calibrated detector. Monitor temporal
  changes, finite updates, KL and learning/standing outcomes.

**Release gates:** after all new source-code CI tests pass, rerun one
bounded 2048-CUDA full-PPO source-pinned smoke on the final SHA before
launching a 5M cold-start stage; it never saves a training checkpoint.
Check the browser nine-solid carpet separately using an existing
untrained actor. The true cold-start first learns flat standing around
5M historically; stop and inspect before extending the same optimiser
and teacher lineage to 20M and later 140M. No automatic merging to
`main`.


### First actual v5041 ten-million-step cold start and 7m final exam stall — 3 November 2026

Owner completed fresh 2048-CUDA sim2real/unified carpet cold
training on **source `9070238b3c81a6805296f867a7849e8799f4a2ff`**,
seed 735, `~/robot_lab/emu_runs/v5041_carpet_s735_cold`.
Requested 10M, actual `10,027,008` PPO boundary. The user
uploaded `results_bundle(10).zip`, containing 16 metrics rows,
complete fixed exam, performance summary and source-provenanced
run config. At 5.24M stood for ~694/700 frames, with
intermittent later setbacks; at 9.83M fell 1.1% in the completed
rollout and reached 700 frames. Last full 10.027M training sample:
reward +2.488, `fall_rate=0`, length 700, forward +.027 vs
requested +.035m/s; this is **standing, not robust walking**.
Final held-out exam core survival 1.000, dynamic80 0.109,
resilience75 0.078, broad score -0.255. Don't compare
advanced 80% command/disturbance panels to mature learner
until the foundation progresses beyond 10M.

**Important finalisation performance finding:** The user saw no
console messages for ~10min after last training log (~9.83M),
thought process hung, but it finished. `performance.json` proved
**433.200 seconds** in 10-panel, 2-cohort, 32-world held-out
frozen evaluation, with 4-5 first-use different MJX evaluator
worlds costing ~64-70 seconds each. PPO updating itself took
600.37s measured (152 postcompile updates) and final
checkpoint saving was **0.032 seconds**. No data loss or
checkpoint write problem.

Feature branch afterwards adds progress to *every* held-out
evaluation (not only final): target reached announcement, exam
BEGIN, panel/cohort start + completion with wall and
survival/MAE summaries, and a **20-second daemon reporting
heartbeat while blocking JAX compiles**. The ten-panel/two-cohort
scientific protocol and scores remain unchanged: don't silently
shorten the exam by excluding difficult panels. A fake-JAX
regression tests the unchanged metrics and a heartbeat expiry.

**Critical source lock:** the saved 10M source **9070238b**
must NOT be resumed under the newer branch HEAD (which contains
the unique-tile anti-rocking fix plus progress). Resume it only
from an unchanged, clean 9070238b checkout or deliberately
start a completely independent later-source cold training run.
Do not edit in-place source or `main` to acquire new console
messages. The full measured timings and safe handling are also
documented in `docs/V5041_GPU_CARPET_TRAINING.md`.


### v5041 Viewer UI consolidation and specialist lesson inspector — 3 November 2026

The owner requested a substantial viewer UI cleanup during an active
140M GPU training campaign: two obvious worlds, natural field and 3×3
magic carpet; one coherent settings drawer; FRONT DEPTH 4×4 no longer a
floating secondary settings panel; specialist role inspection that
shows the actual **training attempt distribution** as closely as
possible rather than only geometry; visible shove arrows.

Feature branch code now provides:
- **Field** and **Training Carpet** top-level world buttons. Field
  defaults to trees, natural radial slopes, no grid/numbered challenge
  obstacles (each can be toggled under Settings). Existing eight
  physical park features, camera controls and navigation are retained.
- **One Settings drawer**, with physical Field settings, focused
  Carpet inspector, collapsible advanced steering, and a collapsed
  off-by-default 4×4 Sensor inspector. Legacy single-surface
  diagnostics remain nested rather than another top-level world.
  Floating telemetry shows essential gait stats; extras in
  `More telemetry`.
- In Carpet, ONE persistent **Training Role** chooser sits beside
  the 3D world, not hidden in Settings. Specialist `Auto / Standing /
  Commands / Resilience / Slopes / Steps / Transitions / Control`
  controls a genuinely moving/recycled nine-solid MuJoCo box world
  AND one deterministic representative exercise: speed/yaw commands,
  slopes, signed steps, or resilience perturbations. An exercise is NOT an exact GPU world
  replay or evidence of mastery. Joystick/keyboard takes over manual
  control; specialist preview auto-resets after falls.
- `/api/live/terrain` now exposes saved push ceiling, alongside
  live learning step, role frontiers, signed up/down limits, speed
  forward/reverse/lateral/yaw, and measured push frontier. Inspector
  selects ordinary STEP's max(terrain, Academy) heights before
  Academy births, Academy-specific frontiers after introduction, and
  role factors plus actual curriculum exposures. Manual override
  section can intentionally test unearned taller steps/steeper grades.
- Browser `MjData.xfrc_applied` on the torso makes an **actual five
  control-frame horizontal shove**, held over real MuJoCo substeps.
  A directional 3D amber arrow shows the applied world-frame force.
  Resilience spectacle timing is deliberately deterministic and more
  frequent than GPU's rare random push schedule, clearly labelled
  `viewer demonstration`. Only applies to browser data.
- Source-pinned trainer is UNCHANGED; this is viewer/diagnostic-only
  code, with Node regressions for live frontiers, six specialist
  command scenarios, two modes, 4×4 integrated inspector and force
  bindings. Actual browser WASM interaction still needs user
  observation on the separate viewer checkout. Do not pull/reset or
  reinstall the running trainer's checkout or venv. The clean,
  separate viewer checkout can be installed with
  `bash scripts/setup_wsl.sh --viewer-only --install-node`
  (Python 3.13+, Node24) and served with `emu viewer --live --root
  ~/robot_lab/emu_runs/v5041_carpet_s735_140m_fork10m --no-open`.
  Do not update the source-pinned active GPU training checkout.

See `docs/V5041_TRAINING_CARPET_VIEWER.md`. Preserve the training
140M run source SHA and live checkpoint metadata independently of
whatever UI branch HEAD is used to watch.


### Viewer usability hardening — 3 November 2026, specialist-first revision

Further owner browser feedback corrected three practical flaws in the
initial two-world viewer:

1. **Retired depth courses were still visible.** The optional 4×4
   Sensor inspector now always reads the CURRENT Field or Carpet.
   Removed its old Ground selector (flat, step presets, ridge, log,
   v5023 lab), separate slope/step lab sliders and the browser UI
   commands that entered the retired `sensor` terrain mode. The main
   viewer's internal mode selector exposes ONLY Field and Carpet.
   The read-only 4×4 diagnostic still exists.
2. **The stale “Manual control — autonomous navigation resumes in 0s”
   overlay covered the Carpet role chooser.** Field navigation now
   explicitly requires `terrainMode==='park'` in `syncGoalMode`;
   Carpet never enters navigation tick, never displays the manual
   countdown, and hard-hides the old HUD as a CSS fallback.
   **Training specialist is now the first control INSIDE Settings**
   rather than a floating selector. A single consolidated drawer, no
   overlapping secondary navigation or inspector cards.
3. **One checkbox/number-form did not expose the specialist's real
   scaled assignment.** Settings now has a source-linked set of ten
   role-effective sliders below Training specialist: slope angle,
   transition, up/down lips, forward/reverse/strafe/yaw speed,
   disturbance probability/intensity, and push cap. Current training
   assignment is shown by default, with the role's relative scale
   recorded. Drag any slider to leave live-follow mode and override
   that role's **viewer-only** physical geometry/commands, including
   deliberately unearned tall lips at a still-flat foundation age.
   Overrides are keyed per role; Reset restores the source frontier.
   No source-policy/teacher/optimizer changes.

The old deterministic shove rotation and six-second timer are
**removed**: `training-shoves.js` samples a real 50Hz MuJoCo
`xfrc_applied` impulse after a stochastic Bernoulli event, with
random XY direction, magnitude bounded by the chosen Newtons, and
2–8 control-frame duration. Event probability is
`0.0015 * selected resilience intensity` per control frame;
default explicitly-labelled 8× inspection speed makes rare pushes
observable, with 1× training-style frequency selectable.
The **Apply one random shove** button is only an immediate manual
preview, no longer the main way to see perturbations. Amber arrows
are bright only when real force is applied, muted after completion.
This is a browser observation of a representative task, NOT the
hidden per-world GPU stochastic RNG replay.

New `tests/training-tuning.test.mjs` and an expanded specialist
test assert source role multipliers, custom box geometry independent
of teacher source, manual foundation challenge, stochastic shove
distribution/strength/frame bounds, removed retired UI and Field-only
navigation gating. Source-isolated viewer on
`~/emu-v5041-viewer` is still required. Do not alter the running
140M source-pinned trainer checkout. Docs updated in
`docs/V5041_TRAINING_CARPET_VIEWER.md`. Browser MuJoCo/WASM
interaction needs owner acceptance after automated CI.


### IMPORTANT: 140M resilience frequency audit and viewer recovery workout

The owner observed the new Training Carpet spectator rarely getting
pushed even in "training-like" mode and expects roughly two recovery
challenges per second during a resilience lesson. Audited the actual
v5041 trainer source at `emu/environment.py::_simulate_physics`:

- `PUSH_PROB_PER_STEP=0.0015` at 50Hz, conditional on no active push,
  times the per-episode `T.resilience` task fraction (which is NOT
  `resilience_difficulty`). At full intensity, 0.075 starts/s =>
  **one per 13.3s per robot**; at multiplier .36, **one per 37s**.
- In open-frontier unified training, `T.resilience` of a RESILIENCE
  regime is `max(.08 / configured_push_max_n,
  lerp(birth_resilience_d*.70 / configured_push_max_n,
       frontier_push_n / configured_push_max_n,
       specialist_mix))`. With physical maximum 4.0N, frontier .65N
  and full specialist exposure, the effective multiplier is .1625,
  yielding **one per ~82s per robot** (subject to actual world RNG and
  task birth). Other regimes usually borrow 25% intensity.
- Genuine resilience training ALSO randomizes the initial episode
  pose (roll/pitch), position, linear and angular velocity and
  joint targets, with full recovery perturbations for the focused
  RESILIENCE birth and partial borrow elsewhere. Across 2048 worlds
  many independent recovery examples run simultaneously. A high
  mid-walk shove frequency is an interesting intervention, not a
  necessary correction to the existing run.
- Fixed a viewer conceptual error: it used the global
  `resilience_difficulty` as per-frame push probability instead of
  the trainer's `T.resilience` physics intensity. The inspector
  now projects the source frontier, 4N configured ceiling and
  20M–30M specialist exposure ramp; labels it an estimate because
  the per-world birth/field RNG cannot be replayed from metrics.
- Added `training-shoves.js` explicit `training`, `inspection` and
  `drill` schedules. **Training** retains rare original probability;
  **Inspection** multiplies it 8x; **Recovery drill** (default)
  applies roughly 1–2 stochastic pushes/s as severity increases
  (2/s at full slider), with random magnitude/direction and
  2–8 control-frame genuine MuJoCo `xfrc_applied` force. The HUD
  shows expected shove starts/sec and per-robot interval.
  The drill is explicitly NOT what the current GPU trainer does.
- Future controlled training science, only AFTER the unchanged
  source-pinned current 140M run is preserved: compare opt-in
  0.25, 0.5, 1.0 and 2.0 shove starts/s with standing/command/step
  retention and verified recovery, and separate push force and
  timing effects. Never silently increase the running trainer's
  stochastic force frequency.


### Important: 110M Carpet viewer looked flat despite nonzero console frontier

Owner reported at 109.6M–110.3M training:
`slope≤0.7°(0%) step≤0.4cm(15%)` and no obvious slope/steps
in the 3×3 browser viewer. In `results.py`, slope parentheses are
**observed physical slant occupancy**, whereas step parentheses
report **`terrain_exposure` = 0.15**. The visible 4mm is a
*governor ceiling*, not the actual height of a born STEP specialist
world. `environment._carpet_birth` calls
`terrain_carpet_mjx.role_envelope` with
`actual_up_m=frontier_up_m × role_factor × terrain_exposure`
and `actual_slope_deg=frontier_slope_deg × role_factor ×
slope_exposure`. At ordinary STEP factor 1.0, a 4mm frontier
and 15% exposure produce **0.600 mm reference step**, further
varied by seeded directional profiles. A 0% reported physical
slope occupancy is NOT itself a reliable number for the separate
`slope_exposure` parameter; read metrics to know exact source
assignments. This may explain why the active long-run Academy
still reports zero signed crossings and little slope practice,
but don't change its source while training.

Found separate VIEWER problems that made it even more confusing:
- Source-follow geometry signature rounded STEPS at 2mm increments
  (`round(meters*500)`) and slopes at .5°, meaning every new
  submillimetre change could be skipped without a physical rebuild.
  Added `src/carpet-refresh.js`: signature at **0.025mm** and
  **0.01°**, including distinct STEP birth programme.
- Settings step sliders had `step=1` **millimetre**, rounding
  real ~0.600mm assignments to zero/one. Now `step=.025` mm,
  and printed millimetre values have three decimals.
- The viewer had assumed every STEP birth after Academy
  introduction was an Academy birth. Actual GPU
  `academy_active` is a sampled per-episode boolean; ordinary
  STEP uses `max(general terrain,Academy signed frontiers)`,
  Academy uses its separate UP/DOWN frontiers. The viewer now
  has a visible `Ordinary STEP / Academy STEP` chooser.
- Top of Carpet settings now separates **global reported ceiling**,
  **physical exposure factors**, **source-assigned role geometry**
  and **browser collider geometry** with signed heights in mm.
  A tiny readout also remains in the gait HUD outside Settings.
  Default remains the real role×exposure physics. `Global frontier
  maximum` is an EXPLICIT viewer-only diagnostic override
  bypassing role/exposure; an actual physically taller preview
  without claiming that the current GPU worlds are that hard.
- New focused `tests/carpet-live-geometry.test.mjs` reproduces
  110M 4mm×15%→0.6mm, the separate STEP/Academy signed caps,
  upgraded physical-refresh precision, and DOM/slider controls.
  `docs/V5041_TRAINING_CARPET_VIEWER.md` contains the complete
  explanation and usage. Actual 140M trainer checkpoint/source
  unchanged, `main` still frozen. If new branch CI is pending,
  do not claim final Git SHA green before checking.

Potential independent SCIENCE implication: `four_lips=0/43`,
`foothold=0`, `transfer=0`, little physical slope exposure
at 110M despite a very step-heavy allocated curriculum. Source
curriculum frontier versus actual birth severity needs a careful
post-140M analysis before any new source-pinned training experiment.
