# EMU handover — Live Skill School

**Updated 26 November 2026.** Read this first when resuming the project in a
new conversation. This file is the current research entry point, not an
instruction to launch training. Check the live branch head before editing;
the source IDs below identify implementation/evidence milestones.

## Current repository and owner direction

Repository: [mdt733/emu](https://github.com/mdt733/emu).
Active research branch: `research/ks150-live-skill-school-v1`,
[draft PR #106](https://github.com/mdt733/emu/pull/106), based on PR #105.
The most recent implementation commit at this handover is
`87a130e0782b53aac54c3beb8da295a0188a6473`; later documentation commits
may follow. Keep work on this branch. No merge, champion replacement,
Pages publication or hardware qualification without explicit owner approval.
[Previous handover (23 Nov)](https://github.com/mdt733/emu/blob/372bb57bc59ab7ca4bcfb0f99d4460ff23d3c646/handover.md)
preserves older history. See issue #62 and docs/RESEARCH_FINDINGS.md for
historical context; do not assume their older roadmap supersedes this one.

The owner wants a single unified robot that learns individual skills,
retains them, and transfers them. Prioritize educational quality and
efficient GPU use over elaborate CI rituals. CI is a watchdog, not the
scientific objective. The owner explicitly wants **4096 simultaneous
environments** (previous 2048 gave diminishing throughput returns),
live feedback from those same training worlds, and independent exams
only at meaningful checkpoints. When a campaign is ready, provide one
unattended, source-pinned launch and a complete results archive; avoid
routine approval stops and babysitting. Do not claim a GPU run happened
unless the operator actually ran it or its evidence is available.

## Immutable physical and deployment constraints

- One six-actuator H40 robot, one PPO actor/critic/optimizer, 360 inputs
  (four-frame sensor history, IMU, optical flow, 4x4 ToF, proprioception
  and movement command), 256/256 hidden sizes, six motor position outputs.
  No teacher labels, privileged world XY, extra networks or exam geometry
  in the actor. All ordinary, reference and practice worlds train PPO.
- Research uses `EMU_PHYSICAL_PROFILE=h40-spring-only`, calibrated
  KS150 passive knee rocker, model SHA256
  `21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470`.
  Real rolling 2x2 carpet, eight MJX triangular prism colliders.
  Do not conflate research physics with protected original `h40`.
- Main champion stays unchanged: original 120M seed735 actor,
  source `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`, actor SHA256
  `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673`.
  Browser Infinite Meadow is a demo, not the independent physical exam.
  Simulation evidence is not hardware validation.
- 4096 worlds x 32 rollout frames = 131072 experiences/update.
  16 minibatches preserve 8192 samples/minibatch vs 2048 x 32 / 8.
  Four PPO epochs unchanged. Policy updates per fixed step budget halve;
  actual sustained throughput and learning efficiency still need measuring.

## What the completed A/B/C study taught us

Completed source: `7b6c3f9e10b9b6ad2f7302dad00d9d25bbd5d574`,
PR #105 `research/h40-developmental-cross-stairs-v1`.
Operator's WSL study root:
`~/emu-data/studies/ks150-abc-unattended-v2`.
Final evidence: `results-140m.zip` (previously supplied in chat; not
assumed accessible in a new chat or committed to Git). Full checkpoint
binaries and actor exports live on the operator's machine, not in this ZIP.
Ask the operator for the archive if exact raw measurements are needed.
Do not invent a file path in a new session or assert files remain available.

A = original school; B = shared stairs; C = B + registered broad
mission refresher. Training seeds 9501 and 9502, discovery exam seeds
6901/6902, final holdout 7901/7902 (held out of this campaign's teaching
decisions, not necessarily unseen project-wide). All six students completed
140,050,432 steps each; 840,302,592 total, 30/30 stages at 90/100/110/120/140M.
Both C students received real refreshers at 110M. The prior chat verified
included stage hashes and actual intervention telemetry.

Final compact mission pass percentages (pooled two training seeds and two
final exam seeds, descriptive, not independent replicate counts):

| Mission | A | B | C |
|---|---:|---:|---:|
| Command | 95.77 | 98.57 | 94.99 |
| Resilience | 84.67 | 78.61 | 71.97 |
| Slope | 69.21 | 83.72 | 60.42 |
| Steps | 44.09 | 54.35 | 32.86 |
| Transition | 99.71 | 98.63 | 99.80 |
| Control | 100 | 98.44 | 100 |

B was already substantially ahead at 90M before meaningful shared-stair
exposure; do NOT attribute its final advantage causally to treatment.
A improved stair scores more consistently. C's recovery dropped during
extra practice (9501 -13.1pp, 9502 -39.8pp at 110→120M), while slope
improved. Broad specialty practice did not reliably preserve the actual
missing recovery task. Need matched from-parent comparisons. Only two
independent training seeds per arm.

The persistent uphill bottleneck: zero controlled held-out traversals
in 1560 attempts at 32mm and above, all earned 55mm downhill, uphill
frontiers only 3–3.5mm. At 28mm, ~96% leading-foot placement, ~53%
independent bilateral contact, ~14% five-frame traversal (120 trials per
height across six students); at 32mm leading contact itself fell sharply.
Contact is not measured weight transfer. All 96 deep 60s standing trials
survived and passed balance; none passed bounded standing. Distinguish
survival, physical movement and precision; assess observability before
teaching absolute XY station keeping.

Historical 13-arm lab PR #104 used full-state 120M seed736 forks and
130/140/160M exams. Exact per-arm results were not recovered; prior root
`~/emu-data/studies/ks150-stair-school-go-v2/results.zip`. Do not claim
specific hypothesis effects without the raw archive.

## Implemented in PR #106 (read before changing)

Detailed evolving specification:
[research/KS150_LIVE_SKILL_SCHOOL_V1.md](research/KS150_LIVE_SKILL_SCHOOL_V1.md).
Its phases 1–4 describe the earlier slices; the current handover supersedes
older statements saying teacher memory resets or uphill is static-only.

| Component | Module / integration |
|---|---|
| Passive counts | emu/live_skill_observer.py; environment.py + capability_field.py + training.py |
| Fixed recovery | emu/live_skill_reference.py |
| Mastery and regression | emu/live_skill_ledger.py |
| Recovery practice | emu/live_skill_teacher.py |
| Physical stair gates | emu/live_skill_stairs.py |
| Exact uphill births and credit | emu/live_skill_uphill.py |
| Offline uphill diagnosis | emu/live_skill_uphill_review.py |
| NEW adaptive uphill teacher | emu/live_skill_uphill_teacher.py |
| Recipe / CLI / packet | emu/training_config.py, emu/run.py, emu/cli.py, emu/contracts.py |
| Resume + rollback | emu/training.py, emu/checkpoint_io.py |
| Tests / bounded GPU checks | tests/test_live_skill_observer.py, scripts/live_skill_smoke.py, scripts/smoke_gpu_carpet.py |

Observer: approximately 1M-step sparse JSONL windows under
`RUN/live_skills/segment-*.jsonl`; actual counts, signed difficulty and
frame exposure; uses existing real contacts and capability event machines.
New process/rollback log segments separate raw evidence. Incomplete
episodes never fabricated. Command windows are not complete missions;
relative recovery bins and terminal slope grade are observational, not
fixed physical tasks. No extra simulation or actor. Prior isolated host
observer test ~0.55ms/update synthetic 4096-world rollout; not actual GPU
overhead.

Recovery: six signed longitudinal offsets [-.40,-.25,-.10,+.10,+.25,+.40]
m/s, fixed .16m/s forward command, flat real prisms, sensor scale 1,
step command .012m, no extra random pushes/velocity carryover. 150 control
frames/3s or failure, 25 consecutive good tracking frames + survival
required for mission. Opt-in flags:
`--live-skill-observer --skill-reference-recovery-share .15
--skill-practice-recovery-share .15 --skill-adaptive-recovery`.
Shares are conditional on eligible mature RESILIENCE births, not all
worlds; total <=.30, practice <=.15. Fixed reference uniform; separate
practice uniform unless adaptive flag. Two supported mastery blocks,
repeated regression, min 40 attempts, descriptive Wilson screening;
half maintenance, gradual <=5pp weight change per review. No extra
recovery reward. Teacher state includes closed/pending evidence, clock,
weights, saved with learner. External research forks start fresh teacher;
same-run resume/rollback restores saved teaching state. Log history is
not replayed as learning memory.

Uphill: six exact positive heights 4/8/12/20/28/32mm, actual single real
seam, birth 22cm before edge, .16m/s request, sensor scale 1, no carryover,
350 frames/7s or failure. Reference and practice each max .15 of
eligible mature Academy births. Independent physical first contact,
first sustained foothold, each foot five consecutive top-contact frames,
five-frame traversal beyond lip, traversal+survival. No torso-crossing
requirement for first-foot contact. Reference/practice distinct from
legacy Academy/steady promotion evidence. Static practice reward stages
`none|first_support|bilateral|traversal`; bonus 4 once per life on new
milestone, practice only, never failure frame. CLI flags
`--skill-uphill-reference-share .15 --skill-uphill-practice-share .15
--skill-uphill-stage none`.

**Adaptive uphill implementation (isolated CPU validation passed; GPU pending):**
`--skill-adaptive-uphill` selects per-height practice stage and bonus
from unchanged reference evidence. Six cases, separate practice draw,
teacher packet appends six weights + six stage IDs + six bonuses.
Stage/bonus latched at episode birth; never change mid-life. Repeated
supported gaps of >=15pp with preceding gate >=30%, target <=80% choose
first support/bilateral/traversal. Half uniform maintenance; <=5pp
allocation movement per review. Reference gate lower Wilson bound >=.70
on two blocks fades reward 4→3→2→1→0; fresh disjoint pairs required for
each fade. Two teacher states are saved/restored in the same learner
checkpoint. This is an initial hypothesis; not proof of mastery.
No contact deficit or post-traversal survival bonus is automatically
selected. Three disjoint block pairs without a 5pp improvement pause
credit; one fresh pair clears cooldown, another fresh pair can restart.
Teacher schema v2. Uniform/adaptive use the same initial RNG stream.
Post-fade retention and actual physical learning still need validation.

## 27 November update — EMU Studio + PPO neutral reset fix

The operator attempted pinned pilot-v2 (df4d624f). All six actual
cross-source checkpoint verification calls passed, and the real
4096-world combined GPU geometry preflight passed recovery birth
isolation (285 reference / 298 practice), exact signed uphill births
(594 / 641), 8 convex prisms, recycler clearance and two physics frames.
PPO construction failed before any research students trained, because
make_trainer calls env.reset({}) and the new practice vector defaults
were all zeros. Commit d922dd1a fixes params_array: omitted entire
enabled teacher vectors are uniform / zero credit; explicit partial
or invalid vectors remain rejected. Exact production host packet
regression test added. Corrected combined PPO preflight has not run.

User requested a Micro-Moth-like web UI and Git-driven launch standard.
Implemented root launch.json (schema emu.launch.v1), scripts/studio.py
(stdlib local control centre), scripts/studio.html (responsive dashboard,
live console, learner charts, evidence windows, teachers, CPU viewer,
results ZIP). See docs/STUDIO.md for contract and security boundaries.
Get update & run explicitly fetches trusted research branch, reads
manifest and starts exact commit in detached worktree. Each SHA gets
a separate study root; resume last run reuses its original source.
No shell commands from manifest or browser. Source and old study
directories untouched. Loopback port 8770; existing CPU viewer 8771.
Tests added for local launch/resume/HTTP guard and fixed packet; exact
HTML script parses, no browser rendering or end-to-end Studio tests
executed in assistant container. The user can run the local stdlib
Studio tests before starting. Actual GPU run still pending. Do not
start another GPU campaign manually while Studio runs its own.

The current manifest registers pilot-v3 with same six-student budget,
same parent hashes, new source-derived directory:
~/emu-data/studies/ks150-live-skill-pilot-v3-<SHA12>/.
Use Studio instead of the old v2 commands below. The previous section
is retained as historical runner rationale. Next: fix only real
preflight blockers, collect results and review learning; preserve main.

## Exact point to continue

The unattended pilot runner is implemented in
`scripts/live_skill_pilot.py`. The 26 Nov launch review supersedes
the earlier launch commit 794f90aa. It removes duplicate trainer
fixed exams for the explicitly registered observer pilot (cadence 0),
combines both active mature lesson families in one 4096-world GPU
preflight of 12 PPO updates, and adds per-case live evidence coverage
to final receipts. A torn unterminated raw JSONL tail is retained,
identified and excluded from counts; invalid complete lines are still
errors. Never pool overlapping restart or rollback segments blindly.
Ordinary legacy training defaults remain unchanged.

The two original A/120M stage receipts and config SHA256s were
rechecked directly against the uploaded A/B/C archive; the full
checkpoint and actor bytes are operator-local and are authenticated
on launch. The matched 120061952 age is aligned at 131072 steps.
U/R/H, 40 PPO updates each, 5242880 extra steps/student, total
31457280, final 125304832, seed 9861 matched independent parent/final
pulse and signed staircase exam. No routine external exams or
manual teacher approvals. A local test of source-copied runner
control-flow functions with mocked children passed six fork checks,
interruption/resume, one sealed combined preflight, correct treatment
order, immutable study registration, separate epochs and torn-tail
handling. Repository tests are added in tests/test_live_skill_pilot.py.
This is not a full executable trainer or real GPU test; MuJoCo,
Optax and Flax are unavailable in the local analysis container.

**Scientific budget warning:** From A's sparse 110–120M training
metrics, each 2048-world rollout completed on average about 3.65–4.18
RESILIENCE episodes and 18.88–19.53 Academy episodes. Applying the
new conditional 15% uniform reference share to historical counts
over 5.24M world-steps suggests only ~7–8 recovery and ~38–39 uphill
attempts per case, before altered short episode durations and new
process births. These are rough exposure estimates, not predictions.
40 trials per evidence block and two supported blocks remain intact.
This is a small feasibility pilot; it may deliver too few actual
adaptive decisions to measure teaching efficacy. Do not lower
mastery thresholds, reinterpret no treatment as no benefit, or
silently extend one arm. Consider a larger matched or more focused
follow-up only after actual frame exposure/attempt evidence.

Run from a clean pinned research worktree with the existing WSL venv
and Spring-only profile. Use a fresh v2 root because the source and
recipe changed since the previous proposed command:

```bash
python -m scripts.live_skill_pilot \
  --root "$HOME/emu-data/studies/ks150-live-skill-pilot-v2" \
  --parent-root "$HOME/emu-data/studies/ks150-abc-unattended-v2"
```

Upload `~/emu-data/studies/ks150-live-skill-pilot-v2/results-live-skill-pilot.zip`
when complete; preserve local checkpoints. The actual source pin is the
current reviewed branch head or a later explicitly reviewed commit,
not the earlier 794f90aa. If preflight fails, capture the console log;
do not bypass physics checks. No real PR #106 GPU acceptance or pilot
completion is known. Next action: run the bounded preflight and pilot
on the operator's GPU, then review actual exposure, teacher actions,
transfer and retention before adding more teaching features.

## Validation and launch status

Historical isolated host tests reported passing: observer counts, recovery
selection/velocity/settling, checkpoint teacher replay, first-lip gates,
fixed uphill assignment and static once-only credit. These were synthetic
NumPy/JAX tests, not end-to-end MJX. Prior chat reported 323 reference and
320 practice births for a 4096-world synthetic uphill assignment. New adaptive uphill validation (hash-matched exact modules):
four fades 4→3→2→1→0, restoration and rollback snapshot, stall/cooldown,
no-contact guard, 291 fixed and 339 practice cohort assignments preserved;
28mm practice 55→203, one-time credit, exact initial uniform/adaptive
4096-element assignment match. No real GPU run of PR #106 is known. Only the registered small U/R/H pilot is prepared for unattended launch; no champion promotion.
The operator's machine previously supported 4096 worlds; no sustained
throughput measurement on this new code is available. The latest full CI
status is not asserted.

Use the existing WSL repository venv (often `~/emu/.venv/bin/python`),
a clean pinned worktree, and `EMU_PHYSICAL_PROFILE=h40-spring-only`.
Do not run the old A/B/C commands on this new branch. The previous
successful A/B/C archive and all checkpoints must remain preserved.
If a run is started on a frozen source, don't change its source mid-run
or silently overwrite study registration. Provide one launch block
only after the actual runner and relevant preflight are reviewed.

## Suggested opening message for a new chat

"Continue my EMU Live Skill School project. Read the root handover.md
on mdt733/emu branch research/ks150-live-skill-school-v1 (PR #106),
then research/KS150_LIVE_SKILL_SCHOOL_V1.md and current source. Resume
at the exact next-work section; don't restart the design or change main.
The matched pilot runner is implemented; physical GPU acceptance is pending.
Continue from actual run evidence or any observed blocker. Focus on efficient
4096-world live education and one unattended matched experiment."

Update this file at meaningful completion points so a fresh chat can
resume without relying on remembered messages. Keep precise measured
evidence and test limitations explicit. The goal is the best teaching
method, not merely more steps, more exams or more commits.

## Studio progress and live diagnostic export (27 November)

Operator's v3 log confirmed all fork checks, combined birth/geometry
checks and the first real 4096 x 32 PPO update (131072 steps; 86.84s
including compile). No later acceptance update or research student
training is established by that log. Do not infer completion or failure
from silence. Studio now separates total student training from current
phase (fork checks, GPU acceptance, parent exams, training, student exams,
packaging). Old acceptance logs only establish a lower bound; new smoke
prints each completed update, new trainer prints each observed rollout.
Output age and process lock are shown without claiming GPU health.

Final ZIP remains the sealed runner output. Partial ZIP is always available
once a job exists, containing a bounded console log and available metrics,
lesson JSONL, manifests and exam evidence; omitted oversized files are
listed. No checkpoint or actor binaries. Partial snapshot is explicitly
marked incomplete and not a sealed result. A hash-matched copy of
studio_progress.py passed local tests covering the operator's old log,
retry boundaries, per-update progress, phase transitions, archive CRC,
partial JSONL and exclusion of checkpoints. Browser/HTTP integration
and actual corrected full GPU run remain unverified.

A read-only monitor can run alongside the original Studio on port 8772:
`python -m scripts.studio --repo "$HOME/emu" --monitor --port 8772`
from the updated Studio checkout using the existing EMU venv. It reads
the same state receipts and does not launch, stop or modify the pinned
job. Keep the old controlling terminal open while it is active. Once
idle, the new checkout can replace the old controller on port 8770.
Do not click Get update & run to merely refresh a running job: it selects
a new source/study. Resume last run preserves the original pin.

## Operator reported overnight acceptance stall (27 November)

The operator confirmed the v3 attempt remained at the first real PPO
update overnight and treats it as broken. The exact cause is not yet
established; no further trace was supplied. Review found that smoke
retained obsolete full MJX state, previous-policy state, raw observation
matrix and previous rollout arrays across the next update. Commits
9b44c80c and 2a718c5c release these references, reuse one host diagnostic
transfer in live smoke, log dispatch/synchronization/observer phases and
allocator counters, and arm faulthandler watchdogs (900s initialization
and first update, 300s subsequent updates and final checks; exit with
traceback rather than silent overnight wait). Acceptance criteria,
physics, teacher policies and 4096-world budget unchanged. Real GPU
validation still required; this is a plausible memory-pressure fix,
not a verified diagnosis. Stop the failed job in its existing Studio,
wait for the process to exit, then use Get update & run. The new Git
source automatically has a new isolated study directory. Original
failed evidence remains. Do not use Resume last run for this fix.
Older Studio can fetch and launch the new experiment; the Studio UI
itself is loaded from its bootstrap checkout and requires the newer
controller/monitor checkout for progress/partial ZIP features.


## Completed Live Skill School pilot — 27 November 2026

**Actual run succeeded.** Source dc6bd19923c3d03e873f816a6c51d466515517a1,
study ~/emu-data/studies/ks150-live-skill-pilot-v3-dc6bd19923c3.
Operator uploaded results-live-skill-pilot.zip (3.3 MB, 110 entries,
ZIP CRC clean; all 88 receipt-referenced file SHA256s matched).
Combined GPU acceptance receipt present. Six students all final
125304832, five complete live windows each, one epoch/segment each,
5242880 world steps each (31457280 total), no recorded rollback or
nonfinite gradient skip. Final sustained PPO rates 31779–32569/s,
sum of six trainer wall times 1189s excluding external exams/acceptance.
Configs within each matched parent differ only by run_dir and the two
adaptive flags. Full checkpoint and actor binaries not in archive.

**Neither adaptive treatment actually activated.** Both R teachers
made five uniform-maintenance reviews each; both H teachers closed
only one block per height, no events, all zero stage/bonus. All final
weights exactly uniform. Thus endpoint variation is not evidence of
adaptive teaching benefit/harm. Two independent parent seeds only.

Reference coverage: recovery 43–54 completed attempts per student
(4–14 per case), uphill 388–412 total (57–77 per case). Actual
recovery reference share ~0.13–0.17% frames, practice ~0.17–0.19%;
uphill reference ~2.66–2.97%, practice ~2.81–2.95%. Conditional
birth percentages are NOT global experience quotas. Two 40-attempt
blocks insufficient for both teachers. Avoid lowering evidence criteria.

Pooled matched pulse mission pass % (one independent exam seed 9861,
two parents, 4096 attempts each across all tasks):
| Mission | Parent | U | R | H |
|---|---:|---:|---:|---:|
| Command |95.18|96.48|96.22|95.44|
| Control |99.80|99.41|99.61|99.22|
| Resilience |85.55|79.69|86.33|83.20|
| Slope |72.79|73.83|65.36|70.96|
| Step |39.84|43.36|42.38|39.84|
| Transition |99.02|98.44|98.83|98.63|
| All |77.00|77.47|76.49|76.29|

Signed matrix traversal up parent/U/R/H = 71/67/60/67 of 400 each;
down 160/141/161/153 of 400 each. At >=32mm uphill zero of 260
per group (780 student trials). Academy frontier 3–3.5mm up,55mm down.
Live uphill pooled reference counts across six students:
height, attempts, approach, first_contact, first_support, bilateral,
traversal, mission, survival:
4,388,388,388,388,362,338,338,388
8,363,363,363,363,338,317,317,363
12,416,416,413,413,390,369,369,415
20,413,413,411,407,367,335,335,385
28,412,412,276,240,154,103,101,271
32,409,40,1,1,1,0,0,392
Reference traversal gate is near lip (10cm beyond); matrix is 60cm
beyond seam and vx .25 vs .16 live. Do not equate rates.

Raw independent matrix across six final students, 60 trials/height:
28mm median best forward progress 1.320m, 56/60 leading contact,
33/60 both feet,11/60 traversal,45/60 survived. 32mm median forward
progress .080m,2/60 leading contact,0 bilateral/traversal,57/60 survived.
Approach collapse/avoidance or inability to engage is a hypothesis,
not verified intent or mechanism. 28mm weight-transfer/bilateral
support deficit distinct from 32mm approach deficit. Inspect actual
sensor/command/reward trajectory, toe path, signed displacement and
yaw before prescribing new shaping. 32mm high survival is NOT mastery.

Next recommendation (not implemented or launched): matched focused
28mm practice intervention with enough reference births to actually
activate, plus short 28–32mm approach diagnosis (e.g. 28/29/30/31/32)
to localize cliff; keep held-out exam unchanged, separate recovery
repair study with explicit global exposure target rather than tiny
conditional birth share. Approximate recovery reference 2% world-frame
budget could yield ~700 150-frame trials per 5.24M steps, ~116/case
if uniform/full horizons; actual occupancy and block timing need
measurement. Retain all ordinary training, same budgets/parent,
minimum 40 attempts and two disjoint blocks. No champion promotion
or automatic conclusion about adaptive teachers. Evidence not yet
committed as raw archive; request operator ZIP again if needed.


## Cliff School — focused next experiment (28 November 2026)

Implemented on the existing PR #106 branch. New Studio manifest
`ks150-live-skill-cliff-v1` (commit 3c0caf41), launching
`scripts.live_skill_pilot --classroom cliff`. The completed pilot is
preserved at its exact old source/root; main and champion unchanged.
User can click **Get update & run** in the existing Studio. A new
source-derived root is created. Do not resume the old pilot into this
study. Final archive is `results-live-skill-cliff.zip`.

Design: same two authenticated original A120M full-state parents and
matching training seeds, U (uniform practice) vs H (adaptive uphill).
Four sequential students, order U/H for 9501, H/U for 9502, 4096 worlds,
100 PPO updates each = 13,107,200 additional steps/student; target
133,169,152; total 52,428,800. Exact combined GPU preflight uses the
new cliff profile and same .25/.25 conditional Academy birth shares.
Original ladder remains default for other recipes. Fixed reference
heights (20,28,29,30,31,32) mm; all reference/practice births have same
task/physical distribution before adaptive decisions. 20mm maintenance,
five heights around observed cliff. Recovery reference/practice unchanged
at .15/.15 conditional RESILIENCE births, no adaptive recovery arm.
All worlds train PPO. Half Academy births remain ordinary. Actual
global frame share is measured, not guaranteed. Independent matched
pulse and signed matrix endpoints use fresh registered exam seed 9862,
same examiner geometry and scoring; no duplicate trainer exams.

Preserve minimum 40 completed attempts/block, two supported blocks,
bounded prerequisite bonus and fading. No new approach bonus or reward
for simply stopping near a lip. Teacher checkpoint schema v3 binds
profile to physical heights; never restore ladder evidence as cliff
evidence. This tests combined adaptive allocation+prerequisite credit,
not their separate causal contributions. If insufficient evidence
or zero intervention, explicitly report that rather than claiming no effect.

Passive cliff diagnostics: for each fixed reference case, actual
post-step root x<.85m (seam x=1m), record finite world-X velocity
forward >.05m/s, retreat <-.05m/s, slow |vx|<=.05m/s, yaw alignment
|heading|<=.35 rad and command within .02m/s of .16. Frame-denominator
statistics, not independent episodes; include retreat frames. Existing
approach/contact/bilateral/traversal/survival mission gates retained.
New compact cliff_windows.csv preserves separate raw windows, epoch,
height, cohort, frame exposure and physical gates; no blind pooling
across rollback. Studio source displays added diagnostics (existing
bootstrap can launch; UI upgrade itself requires new checkout).
Archive reinspection: 28mm vs 32mm median horizontal speed .222 vs
.084m/s, support .954 vs .984, speed MAE .066 vs .239. Not proof of
intentional avoidance. No full sensor/toe trajectory trace yet.

Validation: hash-matched exact uphill assignment module passed JAX CPU
4096 test with 2048 eligible worlds, 513 reference/539 practice IDs,
all reference IDs and practice membership unchanged when weights
shifted, 28mm practice 96→322. Exact teacher module tested with copied
Block and allocation constants: two weak blocks selected bilateral
at 28mm, no false support credit at 32mm with zero contact, all four
fades and exact checkpoint replay; wrong physical profile rejected.
New repository tests cover observer frame/episode separation and
runner plan. Those full tests, end-to-end trainer/runner and actual
cliff CUDA preflight have not been executed here. Container GitHub DNS
unavailable, no local MuJoCo/Optax/Flax. The mandatory operator-side
preflight remains authoritative. No training launched by this turn.

Next: collect cliff results, diagnose approach cliff before prescribing
approach shaping, evaluate actual dose and adaptive activation versus
matched uniform. Separate recovery exposure experiment still pending:
design a global experience target and bounded feedback allocation,
not merely increase conditional recovery birth probability. Do not
mix its treatment into this uphill comparison or claim global frame
control has already been implemented.
