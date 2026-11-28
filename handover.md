# EMU handover — Live Skill School

**Updated 28 November 2026.** Read this first when resuming the project in a
new conversation. This file is the current research entry point, not an
instruction to launch training. Check the live branch head before editing;
the source IDs below identify implementation/evidence milestones.

## Read first — current status (28 November)

**Goal: a good school, not a selected good student.** Main and champion
frozen. Keep one focused branch. The next registered study is Simplicity
School V1 (U/H/B), not Retention. Same authenticated A/120M parents,
4096 worlds, 240 updates/student, 151519232 final age, 188743680 new steps,
exam seed 9866. Dense uniform recovery .50/.25 for all; fixed uphill
.25/.25 and 20/28/29/30/31/32mm. U has uniform uphill practice,
H activates uphill teacher, B also activates novice fall relief.
No adaptive recovery. H-U isolates uphill teaching, B-H isolates relief
with feedback effects; B repeats previous Broad practice treatment.
No new physics, global rewards, PPO parameters or exam changes.
Results: results-live-skill-simplicity.zip. No automatic promotion.

Operator wants one magic button, not extra terminals. Legacy controller
on 8770 cannot hot-reload its API/HTML. The new runner starts a
read-only ETA companion automatically at http://localhost:8772/.
Original controller remains authoritative for Stop/Resume/final ZIP.
Companion closes at runner exit. New ETA-aware controllers skip it.
See docs/STUDIO.md. Never update an active pinned worktree or restart
a GPU run to refresh the UI. New ETA separates current vs future
student throughput and includes compile/exam/packaging time.
Current GPU/browser integration requires operator evidence.



Validation of this update: copied exact studio_bridge.py and
test_studio_bridge.py (blob hashes f93133dfff7978c7c883837bcb9f8ba5a54962cc
and 32290773c1173a98540997de6a33f8ca19f676d9) to local isolated harness.
Both unit tests passed. A real temporary Git repository and detached
worktree also verified discovery of the original repo/state, start of a
dummy child, and child termination on context exit. This does not
exercise the actual HTTP server or WSL browser. New study plan, updated
future-rate estimator, archive export and GPU preflight not executed
end-to-end locally. The operator should click the original magic button,
then use the automatic 8772 monitor if the controller is legacy. The
existing 8770 controller and completed Retention archive remain intact.

### Latest completed evidence: Retention School

Archive results-live-skill-retention.zip (4696051 bytes, 116 entries,
CRC clean), source b8656ad3219a66da3f51cc15728890a24869bd0f.
118/118 available receipt hashes match; 38 referenced ancestor files
not bundled. All six students completed 151519232, zero recorded
rollback/nonfinite gradient skips, no torn observer tails.
Final PPO throughput B9501/F9501 22901/18991, other four
41192–42969 steps/s. Cause unknown, don't infer GPU problem.

Matched independent pulse counts parent/F/B/T (2048 per seed):
9501 1608/1598/1652/1523; 9502 1539/1626/1623/1669.
Pooled family counts parent/F/B/T:
command 736/736/732/745 of768; control 510/512/512/512 of512;
resilience 428/481/468/472 of512; slope 550/509/577/507 of768;
step 415/474/474/446 of1024; transition 508/512/512/510 of512.
Overall 3147/3224/3275/3192 of4096. All three improved resilience
vs matched parents; dense B and T did not outperform F on resilience.
B best overall, mainly slopes; T inconsistent by seed.
Reference recovery completed F/B/T=601/1957/1973, passes 451/1507/1544;
practice 600/999/1038, passes 462/757/840.
B/T ~980 reference attempts/student, actual ~0.47% of global frames;
F ~300, ~0.14%. Most cases 3–4 closed evidence blocks.
T9501 6 acquisition, 5 uniform, 19 waiting decisions; T9502
5 acquisition, 4 uniform, 21 waiting. **No repair decisions**.
T9501 four mastered reference cases, T9502 zero; B9502 three,
B9501 zero. Not a demonstrated retention treatment; recovery reference
is narrow signed longitudinal kick, not all resilience tasks.

Late reference windows starting >=145227776, traversal successes
over attempts, heights 20/28/29/30/31/32mm:
F 204/222,218/256,167/234,132/227,83/214,48/224;
B 193/210,172/240,134/225,109/202,107/250,57/219;
T 220/235,185/266,142/263,113/246,93/231,60/239.
Independent signed uphill totals per seed parent/F/B/T:
9501 32/34/33/32; 9502 29/44/33/36 (200 each).
Downhill 9501 69/67/78/55; 9502 69/106/70/93.
At +32mm F/B/T 3/2/1 of20; no traversals at >=36mm.
Live progress does not establish independent transfer.
No promotion: two seeds only, exam seed changed across studies.
Next test remove complexity before adding a shared budget controller.
If simpler U/H wins, favour simpler school; later qualify cold
starts with multiple seeds, matched independent exams and retention
across all competencies. Don't confuse references with independent exams.

## Current repository and owner direction

Repository: [mdt733/emu](https://github.com/mdt733/emu).
Active research branch: `research/ks150-live-skill-school-v1`,
[draft PR #106](https://github.com/mdt733/emu/pull/106), based on PR #105.
The current registered implementation is Simplicity School V1; use the
branch HEAD and launch.json as the source of truth. Historical source IDs
below are pinned evidence, not launch instructions. Keep work on this branch. No merge, champion replacement,
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


## Studio live viewer fallback fix (28 November)

The owner reported a black scene with joysticks/settings visible. Source
inspection found the live startup branch ignored the supplied initial policy
(`config.initial_policy && !config.live`); an empty live watcher therefore
never created the simulator. Browser startup now loads the supplied champion
before polling, labels it as fallback, and continues polling even if fallback
loading fails. Existing live checkpoint validation and training remain unchanged.
The viewer-only change does not alter launch.json, teacher or physics. A
running study remains pinned; do not restart training to apply a viewer fix.
The already-running old viewer can load showcase/champion.emupolicy manually.
A browser/WASM acceptance check on the operator's machine remains necessary.


## Completed Cliff School — 28 November 2026

Operator uploaded results-live-skill-cliff.zip (2,755,520 bytes, 81
entries, CRC clean, 62 receipt-referenced file SHA256s verified).
Source 71d61544f9e73683b93c252734b885adcff9823c, root
~/emu-data/studies/ks150-live-skill-cliff-v1-71d61544f9e7.
Four students completed 133169152, 100 updates/13107200 new steps each;
52428800 total. Combined GPU acceptance receipt present; no reported
rollback or nan_grad_skips_total. Each run has one observer segment,
epoch 0, no torn tails. Config pairs differ only in run_dir and
skill_adaptive_uphill. Last reported train_steps_per_sec ~16.7k for
U9501 versus ~41.5–41.8k for other three; no cause established.

Exposure: 6434 reference and 6342 practice uphill episodes; 255–284
reference completions per height/student. Actual reference frame share
4.15–4.42%, practice 4.04–4.40%; not global 25% allocations.
Recovery reference 128–138 per student for H9501/U9502/H9502
(see raw receipts for exact counts), still too sparse for two blocks
per case. No adaptive recovery treatment.

One actual teacher intervention: H9502 selected stage 2 bilateral
support at 28mm, 125304832 (40 updates into 100). Its practice birth
weight rose from 1/6 to .466667 by 132644864, bonus remained 4, no
fade/pause. In last three complete windows (start >=129499136),
28mm practice frame share within practice cohort was 38.9% vs U9502
16.9%. Actual payment counts not exported; do not claim bonus delivery
frequency from stage alone. H9501 never selected any stage; weights
uniform and bonus zero throughout, despite ample exposure. It is an
inactive teacher, not evidence that shaping caused its improvement.

Pooled live fixed references (all four evolving learners; not endpoint
or independent replicates):
height mm, n, approach %, first contact %, sustained first support %,
both feet %, traverse %, survive %, forward/reverse/slow before approach %:
20,1097,100,99.2,98.8,91.4,83.0,95.4,99.3/0.0/0.6
28,1069,99.9,79.7,73.1,56.6,43.8,79.1,68.9/9.2/22.0
29,1080,97.0,52.2,47.7,35.0,25.0,84.4,43.5/19.8/36.7
30,1070,80.9,29.2,25.8,18.9,13.8,93.8,35.2/26.7/38.1
31,1068,53.4,10.9,9.6,5.9,4.5,96.8,34.1/33.9/32.0
32,1050,24.2,3.2,2.7,1.6,1.0,97.8,36.3/38.2/25.4
Pre-approach command maintained 100% rounded for all heights.
These are pooled frame ratios, not independent observations or
evidence of intention. Heading aligned 99.7/74.0/52.8/57.1/61.6/67.5%.
At 32mm H9501 alone traversed 10/263 overall, 9/67 in late windows;
all other students zero. This was without intervention. All four
improved live 28mm traversal from early to late (windows ending <=
125304832 vs starting >=129499136): U9501 45/119 ->41/71,
H9501 48/113 ->54/69, U9502 22/117 ->23/75, H9502 25/124 ->36/67.
No endpoint 32mm independent traversal by any student.

Independent matched seed 9862 signed matrix traversal counts
(each direction 200 per learner):
seed, direction, parent/U/H:
9501 up 30/31/26, down 81/74/69
9502 up 27/9/40, down 79/57/133
At 28mm uphill, 9501 parent/U/H=1/1/1 of 10;
9502=0/0/4 of 10. >=32mm all zero (verify raw matrix for any
other height before claiming larger breakthroughs).
Pulse missions pooled across two learners per arm (4096 attempts):
parent/U/H total 3142/3138/3268; per family percentages
command 94.66/96.22/97.27, control 99.41/99.61/100,
resilience 85.74/87.50/92.38, slope 69.92/64.45/68.23,
step 40.92/42.58/49.02, transition 99.80/99.61/99.61.
Per seed total: 9501 1607/1578/1529 of 2048; 9502
1535/1560/1739 of 2048. Seed9502 H improved substantially but
one active treatment replicate, combined allocation+bonus, and
inactive H9501 differs from matched U even without intervention.
Cannot yet establish causal efficacy or generalization. No promotion.

Owner emphasized the goal is a good school, not a good student.
Use mature matched forks to test mechanisms cheaply, then require
fresh randomly initialized multi-seed learners under an autonomous
school to establish sample efficiency, retention, transfer,
consistency and reduced human intervention. Do not bake the specific
28–32mm ladder or 120M checkpoint into the eventual school.

Next recommendation, not yet implemented: teacher must diagnose
approach/contact gaps, not just support after contact. First audit
reward/physical trajectory (heading, toe path, tracking, survival
versus forward progress) and instrument actual per-stage payment
counts and practice births. Test a general prerequisite engagement
lesson with conservative fade and matched controls (separate
allocation-only from shaping when practical); avoid rewarding merely
standing near an obstacle or reckless falls. Consider reference
coverage/retention by physical difficulty, not a handpicked cliff.
Recovery global exposure study remains separate. Keep launch manifest
unchanged until a new experiment is genuinely ready. Viewer-only
fix at 166c38a4 does not invalidate this pinned completed study.


## Engagement classroom implementation (28 November)

Next bounded study retains the same physical heights and two authenticated
parents, with a new opt-in engagement profile. U is uniform; H has the
expanded adaptive teacher. Stage IDs 1–3 retain their meaning; 4 is a
controlled moving approach (forward world speed >.05, aligned heading,
upright, supported, no bad contact), 5 is actual first destination top
contact. Both are once per life, practice only, never on failure frames.
Contact is not assumed to require torso approach. Two supported blocks,
40 attempts minimum, existing fade/stall/cooldown and maintenance floor.
The old cliff/ladder teacher is preserved. No global reward or physics
changes, no new policy input. Frame reward contrast (total PPO reward and
locomotion component) and stage-specific births, frames and actual bonus
payments are exported by height; these are descriptive, not causal.

160 PPO updates/student = 20,971,520 new steps each, four students,
83,886,080 total; target 141,033,472. Exam seed 9863, same independent
examiner. New manifest engagement-v1; output results-live-skill-engagement.zip.
Do not resume prior Cliff School into this study. This comparison evaluates
combined expanded allocation and shaping, not a separate allocation-only
ablation. Frozen height grid remains an experimental diagnostic, not the
ultimate autonomous school curriculum. GPU acceptance and full repository
tests pending on operator machine. No training launched here.

Validation before publication: hash-matched exact candidate
live_skill_uphill_teacher.py and live_skill_stairs.py executed in a local
CPU harness with source-copied Block/constants, gate helpers and payment
function. Three tests passed (approach -> contact, both four-step fades,
checkpoint replay/profile guard, preserved original no-contact behaviour,
stall cooldown, rejected reverse/turned/reference/failure/repeat credit).
4096-world JAX CPU JIT check produced 1345 expected first-event payments
and zero repeated payments. This is synthetic, not MJX physics. The
candidate also fixes the telemetry name to reward_weighted_locomotion
(the existing environment exports weighted payments under that prefix),
seals the engagement profile on raw windows and adds repository observer
and runner regression tests. Full observer integration tests, production
PPO and real GPU acceptance are still pending; container cannot clone
GitHub and has no MuJoCo/Optax/Flax. Studio preflight exercises the actual
production path before accepting training. Do not treat test success as
evidence that the educational intervention works. The longer budget was
selected to increase the opportunity for actual teaching and fading; it
does not guarantee enough disjoint mastery pairs.


## Completed Engagement School — 28 November 2026

Operator archive results-live-skill-engagement.zip, source 87f54fd7,
81 entries, CRC clean; 86 file hash comparisons across training, matrix,
endpoint and pulse receipts passed. Four students completed 141033472,
160 updates / 20971520 additional world steps each; 83886080 total.
GPU combined acceptance receipt present. No rollback or skipped NaN
gradients; final reported train_steps_per_sec 41029–41912.
Single observer segment / epoch 0 per student. 9813 fixed uphill
reference and 9813 practice completed episodes (reference counts
U9501/H9501/U9502/H9502=2453/2449/2497/2414; practice
2452/2476/2442/2443). Actual global frame share ~3.91–4.15%
reference, ~3.92–4.15% practice.

Both teachers prescribed lessons at 124256256. H9501: first contact
at 29/31mm, controlled approach at 32mm; later bilateral at 28mm and
contact at 30mm. H9502: contact at 29mm, approach at 31mm; later
contact at 28/30mm and approach at 32mm. H9501 faded contact at 29mm
4->3 at final update. H9502 faded contact at 28mm 4->3->2.
No fully faded lessons or stall pauses. Both mastered 20mm reference
mission. Actual births / payments / sum of reward for stage 2,4,5:
H9501 302/231/924, 361/180/720, 1003/471/1884;
H9502 0/0/0, 645/247/988, 975/627/2418. Combined 3286 assisted
births, 1756 actual payments, 6934 reward units. U zero credit.

Late windows start >=135790592 (pooled two students per arm):
height mm, U/H attempts, U/H controlled approach %, first contact %,
bilateral %, traversal %, survival %:
20 201/184 100/100 100/100 96.5/97.8 90.0/92.4 99.0/100
28 193/186 99.0/99.5 99.5/96.2 85.5/84.4 68.9/71.0 92.2/92.5
29 186/180 98.9/98.3 86.6/83.3 69.9/66.7 52.2/57.2 91.9/96.1
30 183/198 91.3/94.9 61.7/66.2 45.9/51.0 33.9/31.8 91.8/92.4
31 207/189 71.0/73.5 29.0/34.4 16.9/22.8 10.6/16.9 96.1/95.2
32 189/220 46.0/49.1 4.2/8.2 1.6/3.6 1.1/2.7 98.9/98.6.
Early windows end <=124256256: 32mm U/H controlled approach
13.3/11.9%, contact and traversal zero. Both arms learn; no robust
causal or transfer benefit. 32mm reference approach definition differs
from prior Cliff study (controlled gate), so do not directly compare.

Reward contrast pooled by actual pre-approach reference frame count:
height mm; forward/reverse/slow total mean reward:
20 2.333/NA/2.691 (only 111 slow frames)
28 2.505/2.488/2.894
29 2.728/2.552/2.890
30 2.913/2.600/2.873
31 3.064/2.631/2.855
32 3.160/2.628/2.833 (191924/202917/141385 frames).
At 32mm weighted locomotion means .987/.518/.667.
Forward generally pays more at hardest heights, yet retreat remains
common. Frame rewards do not establish action-conditional expected
return or fall risk. Direct early fall penalty source 65 + .1 *
max(0,700-age), in addition to losing future reward on termination.
Even at end of 350-frame fixed classroom life a fall costs ~100;
one +4 milestone is small relative to this. Hypothesis: failure
economics, not just missing positive incentive. Need explicitly test
practice-only attenuation while keeping real failure/reset and unchanged
references, ordinary training and independent exams; may also need to
address foregone continuation reward, not just explicit penalty.
User emphasized self-preservation belongs after learning skill limits,
not before exploration. Skill maturity, not absolute policy age.

Independent matched fresh seed 9863 pulse mission totals parent/U/H:
9501 1623/1624/1543, 9502 1576/1643/1720, each 2048.
Pooled family counts parent/U/H:
command 730/739/736 of 768; control 509/512/511 of 512;
resilience 435/463/470 of 512; slope 562/570/562 of 768;
step 455/471/474 of 1024; transition 508/512/510 of 512.
Overall 3199/3267/3263 of 4096; essentially tied U vs H.
Signed stair traversal parent/U/H of 200 each:
9501 uphill 22/33/26, downhill 76/56/62;
9502 uphill 25/34/27, downhill 76/73/84.
28mm uphill 9501 2/4/2, 9502 2/3/4; all zero at >=32mm.
No adaptive promotion. Prior exam seed differed, use within-study
matched comparison rather than direct delta to earlier Cliff exam.
Next proposed experiment: same parent and matched training protocol,
isolate novice failure-cost treatment from milestone shaping, ideally
include uniform, engagement and engagement+reduced-fall-cost arms.
Don't change global reward or launch manifest before registering and
testing the new mechanism. No new training treatment implemented here.


## Freedom School V1 (28 November)

Registered three matched arms, U (uniform), H (existing engagement),
F (identical H plus direct novice fall cost relief). Original two A/120M
parents, 160 updates each, six students = 125829120 new world steps,
target 141033472, matched independent exam seed 9864. No new geometry,
actor inputs, optimizer or global reward changes. Existing engagement
height profile, reference/practice distributions and teacher remain.
F only: actual practice birth with active prerequisite stage and bonus
b pays fraction 1-.90*b/4 of original explicit early-fall penalty.
b=4,3,2,1,0 => .10,.325,.55,.775,1. This uses existing birth-latched
lesson support as a local maturity proxy, not age or proof of total
task mastery. No relief before diagnosis, during cooldown or when the
lesson is inactive. Physics divergence always pays full original cost.
Real falls still terminate, reset and lose continuation return.
No compensation for lost future reward, no reward for falling. Treat
reset exploitation / rapid failure as a failure mode to test rather
than assuming it is impossible. H vs F isolates enabling relief
(including the resulting adaptive feedback), not separately a different
teacher allocation policy. U anchors ordinary practice.

All engagement arms publish actual baseline penalty, actual paid
penalty, relief, and divergence. Observer aggregates counts, actual
relieved falls, both cost sums and fall age sums per stage, cohort,
height. Raw windows and cliff_windows.csv retain those denominators.
GPU preflight must observe both relieved and full-cost terminal
falls and verify actual payment equals expected before accepting.
Original global reward, references, independent exam untouched.
New manifest ks150-live-skill-freedom-v1; results-live-skill-freedom.zip.
No training launched in this chat. Full GPU acceptance pending.


Validation: copied the exact candidate penalty helpers and original constants
into an isolated CPU harness; ran the candidate test_novice_fall_relief.py
(blob hash d1f021ced05f34d50628c68e93ae202a327540b7 matched GitHub).
Both NumPy and JAX CPU tests passed (4096-world jitted payment check).
Not a full repository test or real physics test. Added a synthetic
observer fall-payment test and archive checks for invalid or out-of-arm
relief; these integration tests are committed but not locally executed.
GPU smoke is mandatory and must check relieved and ordinary terminal
falls. The 10% floor was selected as a deliberately strong single
treatment after the prior modest +4 bonus failed to beat uniform practice;
not a proven optimal value. No changes to lost continuation return.


## Completed Freedom School V1 — 28 November 2026

Archive results-live-skill-freedom.zip (4219002 bytes, 111 entries,
CRC clean), pinned source dad0d048ae86e92fc6b7d0303b16232c8179003c.
114 available receipt-referenced hashes matched; 38 ancestral files
listed in stage receipts intentionally not included in the archive
(cannot independently verify those bytes here). Both packaged parent
config hashes match registration. Combined GPU acceptance receipt present.
All six students reached 141033472, 160 updates / 20971520 new steps
per student, 125829120 total. Final throughput 41233–42611 world
steps/s; no rollback or NaN gradient skips; per-arm paired configs
differ only in run_dir and the registered adaptive/relief flags.
Fixed uphill reference counts U/H/F 4895/4910/4946, practice
4887/4873/4937. Approx 4.04–4.10% of global frames for each cohort.

Actual practice falls U/H/F=355/340/373, of which 0/0/200 received
relief; 98 and 102 in the two F students. F full-cost counterfactual
sum 43551.70, actual 22935.85 (20615.85 direct penalty units avoided);
all recorded uphill practice falls had zero divergence. F relief
at 20/28/29/30/31/32mm=0/27/75/39/37/22. F assisted births 3200,
actual positive milestone payments 1874; H assisted births 3265,
payments 2474. F practice fall fraction 7.56%, H 6.98%, U 7.26%;
F mean age at fall 182.4 frames, H 174.2, U 171.5. No clear evidence
of rapid-reset exploitation; no large increase in actual risk-taking.
F relieved falls only ~4.05% of completed practice episodes. No
change to lost continuation return. Teacher milestone bonus remains
a proxy for maturity, not task mastery. All adaptive students mastered
20mm reference mission, none fully faded all assistance. H fade
events 7+2, F 2+1; no stall pauses.

Late unassisted references (window start >=135790592), pooled U/H/F
traversal percent by mm:
20 78.0/90.2/90.5; 28 67.3/80.4/67.8; 29 54.7/65.4/62.9;
30 52.0/56.3/37.5; 31 33.3/41.5/19.0; 32 20.0/24.6/4.0.
At 32mm, controlled approach U/H/F 62.4/65.1/38.8; actual top
contact 37.6/35.9/10.0; survival 98.0/96.9/99.0. H versus F at
32mm per seed: 43.3 vs 6.2% (9501), 3.3 vs 1.9% (9502).
Pre-approach forward/reverse/slow frame fractions U 34.9/33.6/31.4,
H 35.7/34.3/30.0, F 35.2/36.2/28.5. F did not cure obstacle
engagement; both reference and practice traversal pooled were worse
than H. These are changing-policy in-training measures, not frozen
exam replicates.

Independent pulse mission counts (parent/U/H/F):
9501 1590/1542/1455/1595; 9502 1538/1557/1555/1639, each 2048.
Pooled counts by family:
command 721/725/734/743 of 768;
control 508/511/512/511 of 512;
resilience 434/480/458/376 of 512;
slope 529/462/437/583 of 768;
step 426/410/357/510 of 1024;
transition 510/511/512/511 of 512.
Overall 3128/3099/3010/3234 of 4096. F beat H by 224 missions
(+5.47pp), U by 135 (+3.30pp), parent by 106 (+2.59pp).
Slope and step improve in both paired F versus H comparisons, but
resilience drops in both (9501 H222 -> F159, 9502 H236 -> F217).
Resilience survival H467 -> F479; poor mission competence, not
merely more falls. U survival 486. The F treatment has a meaningful
tradeoff and must not be promoted as a whole school.

Independent signed stairs traversed_5frame counts parent/U/H/F of
200 per direction:
9501 up 23/45/26/29, down 81/55/55/66;
9502 up 25/33/30/42, down 79/68/53/89.
At +32mm U/H/F successes of 20=3/4/1 (only F9502 passed);
none at +36mm or higher. Use within-study matched seed 9864
comparisons; don't directly compare counts from prior exam seeds.

Interpretation: selective relief changes optimization and broad
transfer, but did not solve hard uphill engagement. H vs F tests
the relief-enabled teacher including feedback-mediated changes in
assignments, not fixed identical per-world treatment. Only two
training replicates. The large recovery loss despite good survival
is an example of missing cross-skill retention. Fixed recovery
references remain sparse (F 9501 199, F9502 203 across six cases,
29–45 per case), too little for two 40-attempt blocks/case. Do not
automatically escalate relief or reward for attempts. Proposed next:
improve general teacher evidence allocation and a conservative
retention budget; protect previously acquired skills using fixed
references, and use supported deterioration to trigger repair.
Audit continuation-return economics before testing a separate
intervention. Keep matched controls and independent exam. User's
goal remains a good school from cold starts, not selection of a
good final seed. No code/launch changes or promotion in this review.


## Retention School V1 — next matched experiment

Keep the Freedom recipe including novice penalty relief in ALL three
arms. F repeats its original .15/.15 recovery reference/practice
birth shares; B raises to .50/.25, with uniform practice; T uses
exactly B's budget and fixed tasks but activates the existing
recovery teacher. These shares are conditional on eligible mature
resilience births, not global frames. At least 25% remain ordinary
even in B/T. Fixed reference is still a 3s flat signed kick/settle
task; it is not the whole resilience specialty. No independent
exam feedback to training. The larger exposure tests whether it
is a useful proxy and whether more experience or adaptive allocation
is helpful. Do not claim retention merely because a reference improves.

240 updates per student, 31,457,280 additional steps, target 151,519,232,
six students = 188,743,680 steps, new independent exam seed 9865.
Same A/120M parents and all other treatments held fixed. The longer
budget (vs Freedom 160) was selected to improve the chance of four
40-attempt blocks per recovery case (two mastery, two regression).
Freedom archives had ~200 reference episodes/student at .15 share
and 160 updates; naive scaling to .50 and 240 gives ~1000/student,
~167/case, not a guaranteed count. No parent mastery inferred from
a different independent exam. If a reference never establishes
mastery, acquisition is possible but a repair decision is not.
F is a matched longer-run control, not the earlier completed student.

Recovery teacher now changes allocation only on newly closed blocks
and requires fresh evidence from a selected case before increasing
or decreasing a targeted dose. Existing 50% uniform practice floor,
.05 per-review weight change, Wilson mastery/repair thresholds,
checkpoint history and rollback semantics remain. No extra policy
inputs, no new reward, no physics change. Expanded share validation
is common to CLI and environment; no new configuration fields.
GPU preflight exercises the denser recovery share alongside Freedom
uphill lessons. Recovery windows CSV exposes actual global frames,
attempts, passes, closed blocks, mastery anchors and repair status;
raw teacher logs preserved. Manifest retention-v1.
Primary contrast T vs B; secondary B vs F. Independent resilience
mission plus other specialties and signed stairs must be reviewed.
A full school still needs more representative fixed references in
every specialty and a cross-specialty budget controller; this is
a narrow mechanism test, not that finished controller.


Validation before publication: local isolated CPU harness used exact
candidate teacher and ledger blobs (68b1b11afa96557d00b4593abb873d7b491d685f
and 1a82eb082296297ea0f36764cb7008dc80b67492; git hash-object matched)
with extracted unchanged reference constants and new share validator.
Two pytest tests passed: mastery/repair, no stale-dose escalation,
recovery of uniform allocation, share bounds, checkpoint replay/rollback.
The extracted recovery CSV function was also exercised on the real prior
Freedom ZIP; 720 rows and all 1,204 reference trials accounted for.
Full repository integration, actual GPU preflight and new-study launcher
execution remain untested locally; MuJoCo/Optax/Flax unavailable.
Studio source displays fresh cases, block counts and statuses when using
the updated UI bootstrap. Old UI still supports the generic manifest.
No training launched, no champion promotion. Main remains frozen.


## Studio ETA / progress upgrade (28 November)

UI/runner timing only; retention training recipe and launch.json unchanged.
See docs/STUDIO.md. Existing active job remains pinned; read-only monitor
on 8772 can observe it without stopping the original controller.


Validation: hash-matched exact studio_eta.py and test_studio_eta.py
executed locally with unittest; four tests passed (remaining workload,
wall-rate measurement, receipt transitions, resume, retry and stopped
states). Retrospective check using all 14 training metrics series from
Cliff, Engagement and Freedom: at midpoint of logged training rows the
median absolute remaining-training-time error was 1.49s (0.68%); range
covered 13/14, maximum error 435s. At three-quarter point median absolute
error was 0.61s (0.46%); range covered 14/14, maximum error 266s.
The slow Cliff run is a reminder that large throughput changes can
invalidate any estimate. This tests training-only predictions, not
end-to-end completion forecasts or browser/GPU integration. The complete
new UI has not been exercised in the operator's browser or live WSL job.
