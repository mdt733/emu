# EMU next-school research: shared stair practice across specialists

**Status:** first implementation, CI/CPU contracts only; **not** a trained policy,
not proven to prevent regression, not merged into released `main`.

Parent research source: `research/h40-unified-school-cold-v1`.
New branch: `research/h40-developmental-cross-stairs-v1`.
Physical system remains the same Spring-only KS150, six actuators, 360
sensor inputs and eight real MJX triangular-prism colliders.

## Why change the school?

The finished 13-arm experiment (25 reviewed stages) exposed a tradeoff:
several late-stair treatments improved small-height transfers while
independent recovery/slope mission performance declined. The unchanged
control also suffered a severe late recovery-*mission* regression. This
supports investigating **shared skill maintenance**; it does NOT prove
that specialist mixing prevents forgetting. The uphill Academy remained
around 3–4mm while independent 32mm+ tests almost universally failed.

Older source already placed weak terrain in all roles: the old
`role_envelope` scaled step difficulty by focus, sometimes below
the physical lip-recognition threshold. A new-age-independent
`cross_stair_blend` now creates explicit additional **episode births**
with a physical signed stair in each role, except STEP (which keeps its
separate Academy). There is **no new reward or privileged actor channel**.

## New developmental lesson plan

`--cross-specialty-stairs` is an opt-in **fresh cold-school** treatment.
The original foundation and old rewards are the control. Exposure begins
at 90M world steps, rises smoothly to the full role budgets at 110M, and
never increases **difficulty** by age. The signed UP and DOWN heights
must still be earned by the existing physical curriculum. Shared lips
are capped by role, even when the downhill Academy has reached 55mm.

| Specialty | Extra role-birth chance at 110M | Max injected signed lip | What is *still* tested |
|---|---:|---:|---|
| COMMAND | 8% | 12mm | Command tracking, turns and reversals on mild steps |
| RESILIENCE | 12% | 16mm | Recovery from actual perturbations on uneven terrain |
| SLOPE | 15% | 18mm | Incline handling plus small authentic lips |
| STEP | No extra; existing Academy | Earned UP/DOWN Academy frontier | Foothold, two-foot transfer and course continuation |
| TRANSITION | 10% | 14mm | Start/stop and speed switching on uneven terrain |
| CONTROL | 6% | 8mm | Clean efficient gait with occasional mild steps |

Budgets are **conditional on the role**, not whole-batch shares; athletic
rehearsal and inactive foundation worlds are explicitly excluded. The
actual generated difficulty may be considerably lower than those caps,
and zero earned physical height cannot be fabricated by the schedule.
The original ordinary terrain stays available. The teacher samples
new role assignments only at episode birth; old assignments are not
reclassified when the dose ramp changes mid-episode.

## Isolation and evaluation obligations

The baseline differs from the treatment ONLY by enabling the new flag.
Train two independent **from-zero** students per selected seed, with
identical source, policy, PPO, sensor/physics, generalist and Academy
controls. Use matched two-seed independent physical examinations at
90M, 100M, 110M, 120M, and later; a first validation run is a screen,
not certification. A true retention success requires *mission* success
in COMMAND, CONTROL, TRANSITION, both signed slopes and RESILIENCE,
not merely survival.

Only increase lesson complexity if the student demonstrates each step:
first physical upper foothold → sustained bilateral support → controlled
post-lip traversal → multiple genuine lips → uninterrupted course.
Keep upward and downward promotion independent. **Investigate the
32mm uphill mechanical/teaching cliff separately**; no increased height
is authorised by this intervention.

The stage-one implementation logs `cross_stair_blend`,
`cross_stair_extra_birth_prob_<role>`, and actual completed
`cross_stair_<role>_episodes/survived`. These counts must be audited
before interpreting physical treatment effects. A high survival
fraction is NOT sufficient retention proof.

## Newborn/toddler roadmap (not yet implemented)

- Independent physical mission sentinels trigger targeted role refresher
  allocations, while a protected flat/generalist floor prevents starvation
- First-lip uphill feasibility probe at 28→32mm; test motor/geometry vs
  progression and reward timing without claiming that hardware can climb
- Demonstrated-ability reward fading: small bilateral support bonus
  only during transfer acquisition, then gradual reliance on
  follow-through and robust command tracking
- Short near-lip encounters interleaved with full clean recovery and
  multi-lip course tasks, evaluated in independent first-life worlds

**Do not launch an automatic full-cost new school until the contracts,
physical sampling and baseline-vs-treatment comparison pass review.**


## Stage two prototype: independent mission refresher planning

`emu.developmental_school.mission_refresher_plan` is an offline, opt-in
pure host planning function, not a live allocation change. It accepts
per-task **mission pass counts** from an earlier frozen checkpoint of
the same student and a later frozen checkpoint, on the same two exam
seeds, matching task IDs and equal denominators. This detects regression
even when the unchanged control forgets too. Same-age control contrasts
remain necessary for causal claims. Separate signed slope tasks must
remain separate; do not average away a downhill or uphill failure.

Provisional planning rules (not statistically calibrated):
- A loss exceeding five percentage points on either seed protects that
  specialty from donating lessons. A loss on both seeds requests refreshers.
- Rank by the worst repeated task loss, using the smaller of its two
  seed losses; improvement on another task cannot cancel this warning.
- Redistribute at most five percentage points of the eligible role
  birth cohort, taking no more than 10% of any healthy donor's existing
  share. STEP never donates or receives this budget. If all potential
  donors show regression, make no allocation change and report the alarm.
- Reserve added births as **clean** mission lessons; they are separate
  from the ordinary allocation, which retains the existing mixed lessons.
  No new reward, lip height, Academy share, actor or PPO change.
- Require at least ten attempts per task per seed by default; inadequate
  or mismatched evidence is rejected. Two seeds are a screening rule,
  not a confidence interval or replication across training seeds.

The function validates the provided counts, not the authenticity of
their source. **Do not wire it into PPO yet.** Next implement the
source/checkpoint/task/world-identity receipt adapter, establish a
fixed pre-ramp reference exam and suitable sample sizes, compare
per-task mission scores on two independent cohorts, persist and audit
the planned clean births, and decide in advance how often to review
and when to stop/reduce refreshers. Validate with offline replay and
separate matched from-zero experiments; never silently enable this in
the shared-stairs-only treatment. The initial implementation and
synthetic contract tests do not prove that targeted teaching works.


### Offline examination-to-advisory bridge

`python -m scripts.mission_refresher_review registration.json --output advisory.json`
is a CPU-only review of four previously completed specialist pulse ZIPs:
two matched independent seeds at each of two frozen ages of one student.
It does not run an exam, load a checkpoint, modify a teacher or train.
The registration schema is `emu_mission_refresher_review_v1`:

- `seeds`: two distinct registered exam seeds; `base_weights`: six
  eligible ordinary role weights in FOCUS_NAMES order, summing to one.
- `model_sha256`, `sensor_sha256`, `source_sha256`: independently pinned
  robot, sensor and examiner file identities; the latter must match the
  current review checkout's examiner dependency files.
- `reference` and `current`: each contains the exact frozen export's
  `actor_source` object, `artifact_sha256`, and two `exams` entries
  with `seed`, `path`, `sha256` of the completed pulse ZIP.
  Pin these identities at completion, not by copying untrusted manifests
  after the fact. The checkpoint custodian must establish common student
  lineage and training configuration before registration.

The bridge checks archive bytes, source/actor/model/sensor identities,
GPU first-life contract, runtime, ordered checkpoint ages, canonical
2048-world assignment, all 46 separate task cells and physical mission
flags by re-scoring the raw world scalars. It rejects unmatched or
incomplete evidence; a survival-only result cannot trigger a pass.
The advisory records the registration digest, exact per-task counts,
the proposed budget and `applied: false`. Existing advisories cannot
be overwritten. No teacher budget is changed by this command.

The bridge is not a cryptographic signature or independent verification
that a GPU ran; its integrity depends on trusted digests and checkpoint
custody. Synthetic archive tests are not physical validation. Next:
capture a real paired pulse at the pre-ramp checkpoint, then at the
next agreed milestone, audit the resulting advisory against raw task
counts, and separately implement a persisted, opt-in allocation
consumer only after receipt and replay validation. Keep shared stairs
and adaptive refreshers as separate experimental treatments.


### Approved refresher replay and eventual birth consumer (offline)

The review CLI now optionally creates a bounded, explicit approval receipt:

```bash
python -m scripts.mission_refresher_review registration.json \
  --approve-until 110000000 --academy-probability 0.6 \
  --output approved_schedule.json
```

The numbers above are examples only, not a recommended schedule. In
addition to the four pinned exam archives, approval requires separately
trusted `training_config_sha256` and `current.checkpoint_sha256` in the
registration, plus the actual conditional Academy probability at the
activation checkpoint. The command repeats the evidence review before
creating the receipt; it never accepts an edited advisory as authority.
No output file is overwritten. The expiry is exclusive and no more
than 10M steps after the current **actual PPO checkpoint step**.

The pure host `replay_refresher_schedule` checks the receipt digest,
registered source/run/seed/config/activation-checkpoint identities, step
window and unchanged base birth allocation. The caller must obtain
these identities from authenticated checkpoint state, not the approval
file. It returns the historical allocation when disabled or expired,
and refuses an active receipt with changed baseline or lineage.
**This is a replay contract, not checkpoint integration.**

`refresher_birth_parameters` combines ordinary and extra-clean shares into
one role distribution and six conditional clean-birth probabilities.
The STEP share and conditional Academy probability are invariant, and
no ordinary donor loses more than 10% of its previous allocation.
The maximum extra-clean budget is 5 percentage points **of the
eligible non-athletic role cohort**, not of all world frames.

A future caller can use `clean_refresher_assignment` with a new independent
birth RNG salt, excluding inactive/athletic/Academy/STEP worlds, and
`clean_refresher_terrain` before building actual prism profiles. The latter
removes both signed lips and the mixed-stair flag only on clean births;
SLOPE retains its existing earned incline, other clean roles are flat.
These helpers are not yet called by the MJX environment; they do not
guarantee exact mission-specific command or disturbance sampling.

Before enabling any live consumer:
- Verify the new synthetic review/replay regression checks in CI.
- Record real matched pulse archives from the same fresh student at the
  pre-ramp and later ages; audit each advisory against physical outcomes.
- Persist the approval hash, activation ancestor and expiry in the
  authenticated training state; test resume/rollback and reject unpinned
  receipt swaps. Choose a fresh opt-in research arm with independent
  configuration identity, rather than changing an existing run.
- Add teacher parameter slots and birth-only episode flag/telemetry;
  verify no changes to disabled-path worlds, rewards, actor observations,
  Academy share, ordinary terrain or old checkpoint compatibility.
- Compare the combined method against shared stairs alone and the
  unchanged school. No automatic training or promotion is authorized.

The 90–110M shared-stair ramp remains independent of this prototype.


### MJX clean-refresher delivery contract

The research environment now accepts six appended teacher-only
`clean_refresher_prob_<role>` conditional probabilities. All default to
zero and historical parameter indices remain unchanged. The host rejects
nonfinite/out-of-range probabilities, STEP clean births, non-mature
specialist allocation, exact-exam overrides and a weighted clean birth
budget exceeding 5% of the non-athletic role cohort. These are low-level
packet guards, not approval authorization.

The real prism birth generator samples an independent salted draw after
role/Academy allocation and before terrain profile construction. Active
clean births exclude STEP, Academy, athletic and inactive worlds. Both
signed lips and the competing cross-stair assignment are removed; the
SLOPE grade is preserved. The episode-local Boolean lives in EnvState
and is replaced only for resetting worlds. Ordinary birth RNG draws,
reward definitions, actor input width and Academy mechanics are unchanged
by design. Exact physical examinations explicitly exclude the new cohort.

Rollout reporting adds active clean fraction, per-role completed clean
episodes and survived counts, and scheduled conditional probabilities.
These are delivery/closure diagnostics, **not independent mission passes**.
Like the existing cross-stair metrics, sparse rollout snapshots are not
whole-campaign birth totals.

The existing CPU-MJX prism probe now includes a deterministic matched-key
real recovery-birth comparison: an eligible 5% recovery role is selected
without changing the draw law, then the same reset key is exercised with
and without a conditional clean assignment. Assertions check zero signed
lips/grade/profiles for the assigned clean worlds, unchanged other worlds'
terrain and observations, no competing cross-stair flag, and the same
360-wide actor observation shape. Exact fixture tests check that the
new packet defaults off and rejects an active clean override. These
checks have been committed but need CI confirmation.

The training loop still emits zero clean probabilities (via defaults);
it does **not** load or consume an approval receipt. Existing checkpoint
serialization persists learner/optimizer/normalizer/governor/health/field
state and deliberately restarts live worlds on resume. The future
approval consumer must persist a separate trusted approval hash and
activation checkpoint identity, with explicit rollback/resume tests;
it must not try to serialize the transient MJX episode flag. This
preserves the shared-stairs-only experiment and existing training recipes.
Do not launch an adaptive refresher training arm until that consumer,
real examination receipts and CPU/GPU delivery checks are validated.


### First live consumer: one approved programme, checkpoint-pinned

The research trainer now has `--mission-refreshers`, off by default and
requiring `--cross-specialty-stairs`. Register it at cold birth, not
mid-run or in an external research fork. It changes no lesson until a
valid approval is explicitly supplied during same-run resume. Do not
use it in the shared-stairs-only arm.

The first implementation deliberately supports **one** approved
programme per student, not an autonomous repeatedly mutating teacher.
Activation must be at or after the completed Academy allocation ramp
and at least 40M after foundation, with a mature stable host allocation.
A student at 90M can provide the reference exam but a live approval
waits for the stable allocation. The next PPO rollout uses the
prescribed combined role distribution and conditional clean probabilities.

Protocol (not an instruction to launch unvalidated GPU training):

1. Freeze the same student's matched independent exams and full
   activation checkpoint. Register the substantive config identity using
   `refresher_recipe_sha256(TrainingConfig.load(config_path))` from
   `emu.developmental_school`. This canonical digest excludes only
   runtime device, resume path and final target steps; it includes
   run directory, training seed, teacher settings and opt-in flag.
   It is not the literal JSON file hash.
2. Use the approved schedule CLI above, with actual checkpoint digest,
   actual role weights and conditional Academy probability from
   `birth_lesson_weights` at the actual PPO boundary. Place the approved
   bytes in `<run-dir>/mission_refresher_approval.json`, and explicitly
   supply their independently verified SHA256 in
   `EMU_MISSION_REFRESHER_SHA256` on activation and subsequent resumes.
   The trainer does not read a changed approval during an active process.
3. The trainer checks the selected same-run checkpoint's full bytes,
   same source/run/seed/config identity, age and actual baseline
   allocation before exclusively publishing
   `mission_refresher_pin.json`. Every subsequent checkpoint metadata
   includes that pin, including fixed-exam best and recovery snapshots.
   Crash before the first new checkpoint can revalidate the exact
   activation ancestor. Later resume requires identical local and
   checkpoint pins. Missing/changed receipts or a pre-activation
   checkpoint fail closed, and another programme cannot replace this one.
   Back up the approval and pin with the checkpoint; hashes assume
   trusted operator custody, not protection from deliberate deletion or
   forgery of all local artifacts.
4. Expiry removes new clean allocations on the first host rollout
   boundary at or after the registered exclusive step; live episodes
   retain their birth assignment until natural reset. Rollback refuses
   any recovery checkpoint with a different or missing programme pin.
   This may reduce available early rollback candidates, so monitor
   collapse/retreat and validate rollback behavior before a GPU arm.

The ordinary cold learner, old source forks and shared-stair-only arm
retain zero clean dose. No automatic approval or schedule replacement
exists. Synthetic startup/replay tests cover wrong digest, wrong seed,
wrong baseline, unapproved receipt, pre-activation resume, crash before
new checkpoint, and changed receipt. CPU/MJX CI and a real staged resume
with independently captured exam archives remain mandatory before
calling the end-to-end path validated.

### Next large experiment: learning principles, not a tournament

Proposed three-arm cold-school screen, at least two matched independent
training seeds per arm, same source/hardware/PPO/budget and matched
restart milestones:

| Arm | Teacher |
| --- | --- |
| C | Original unchanged school |
| S | Shared-stair exposure only |
| R | S plus one preregistered, mission-triggered clean refresher opportunity |

Compare S-C for cross-subject stair transfer and R-S for targeted
maintenance; R-C alone cannot isolate the mechanisms. All three arms
use the same milestones/restart protocol. Proposed examination ages
90M (pre-mixing baseline), 100M (ramp), 110M (full mixing, potential
activation), 120M (post-programme) and 140M (durability), rounded to the
same true PPO boundaries. Only activate R when the registered two-seed
mission criteria request practice; if no loss qualifies, record the
no-intervention result. Assess absolute performance against earlier
checkpoints AND age-matched control. Use separate untouched final exam
seeds never used for the teacher's decisions. Two training seeds are
a screen, not statistical certification.

Measure individual signed stair first-foot, bilateral support,
controlled continuation and same-life survival; independent task pass
counts at each signed slope, command, recovery and transition difficulty;
and performance on new compound missions. Record scheduled vs observed
clean/mixed births and their exposure, not only frontier or reward.
The 28-to-32mm mechanical question requires separate physical feasibility
probes and must not be inferred from the learning curve alone.
No new reward fading or uphill promotion treatment should be silently
bundled into these three arms. A timing-ramp comparison and ability-based
uphill mastery ladder are separate follow-up hypotheses. No automatic
launch, promotion or release is authorized by this protocol.


### Executable A/B/C comparison (first campaign runner)

The executable arm names supersede the earlier C/S/R notation:
A = original school, B = shared stairs, C = B + one optional approved
mission refresher. Six cold students: training seeds 9501 and 9502 in
each arm. Discovery exam seeds 6901/6902; final-only holdout 7901/7902.
The study compares at requested 90, 100, 110, 120 and 140M, rounded to
true PPO boundaries. The 90M exam is at the first complete update at
or after the ramp start (not mathematically before it). Nominal total
training budget is 840M world steps, excluding exams and smoke.
The 350-frame signed stair matrix uses 10 trials per signed height per
seed; the 600-frame pulse uses 2048 worlds per seed. The final milestone
adds both holdout seeds and independent deep capability exams (8 worlds
per seed). These are screening sample sizes. The existing 1200-frame
school diagnostic is locked to a historical actor, so this campaign
does not claim paired 350/1200-frame stair evidence or a new compound
stair-plus-command examination. Those remain separate follow-ups.

From a clean checkout of the exact reviewed branch commit, using its
installed project virtual environment, the intended operator interface is:

```bash
python -m scripts.ks150_abc_school plan
python -m scripts.ks150_abc_school lock
# Only after confirming CI green for this exact source:
python -m scripts.ks150_abc_school smoke --ci-verified-sha "$(git rev-parse HEAD)"
python -m scripts.ks150_abc_school advance --stage 90000000
python -m scripts.ks150_abc_school advance --stage 100000000
python -m scripts.ks150_abc_school advance --stage 110000000
python -m scripts.ks150_abc_school review
```

Default root: `~/emu-data/studies/ks150-abc-v1`; use the same
`--root` for all commands. Plan writes no study files. Lock requires a
new root and freezes exact recipe, seeds, source and model. Smoke
requires explicit operator attestation of green CI for the exact SHA
and then actually checks 2048-world GPU clean births and one PPO
update; it does not create student checkpoints. Smoke success is
necessary, not full proof of checkpoint resume or learned performance.

Inspect both C advisories at 110M before continuing. For each training
seed separately, use the registered decision `approve` if the planner
has a nonzero qualifying allocation, otherwise `no_intervention`:

```bash
python -m scripts.ks150_abc_school decide --seed 9501 --action approve
python -m scripts.ks150_abc_school decide --seed 9502 --action no_intervention
```

The choices above are **examples**, not predicted results. The runner
rejects a decision inconsistent with its preregistered dose rule.
The operator can halt the study if the evidence is unsafe or invalid;
never manufacture a qualifying effect. The approval lasts at most 10M
actual steps; expiry is evaluated at the next host PPO boundary.

```bash
python -m scripts.ks150_abc_school advance --stage 120000000
python -m scripts.ks150_abc_school advance --stage 140000000
python -m scripts.ks150_abc_school report --stage 140000000
```

Each advance command trains **only one common milestone** for all six
students. It cannot skip a milestone or train past 110M until both C
decisions are sealed. The same command can resume a partially completed
stage. Completed stage checkpoints are hard-linked to immutable
evidence (same filesystem), with frozen actor/config and raw evidence
digests; interrupted unsealed outputs fail closed and must be preserved
under a different name before retry. Do not modify the live checkpoint
or frozen hard-linked ancestor in place. Use `status` for progress.

At the 110M-to-120M transition, archive the previous recovery candidate
for **all six** students, regardless of C dose. This is a matched
restart/rollback policy change relative to an uninterrupted historical
run: it prevents the C pin from uniquely disqualifying old rollback
candidates. Students retain the selected full PPO checkpoint and must
earn a new recovery candidate under the post-review teacher.

Report writes an exclusive `results-140m.zip` (or an earlier common
stage ZIP), with raw pulse/stair/deep evidence, frozen recipe and stage
receipts, sparse training telemetry, decisions, and per-task matched
B−A/C−B/C−A percentage-point contrasts separately for each training
seed and exam seed. It excludes large checkpoint and actor binaries.
No ranking, promotion, automatic decision or new run is triggered by
reporting. C without a qualifying intervention remains part of the
intention-to-treat comparison. A/B/C differ only by registered
teaching flags; shared pauses and recovery-epoch reset are common.

**Readiness:** runner, synthetic contract tests and GPU smoke gate are
committed; CI and actual local GPU execution must still be confirmed.
The first real staged checkpoint review is a further validation gate.
No autonomous launch or claim of end-to-end experimental validation.


#### Simplified gated launch

After lock and the exact-source smoke gate, the operator may use
`python -m scripts.ks150_abc_school go` instead of issuing individual
advance commands. GO completes the common 90/100/110M milestones,
then stops if either C decision has not been frozen. After reviewing
and recording both decisions, GO resumes the 120/140M milestones and
writes the final results ZIP. It neither approves a decision itself
nor skips a partially completed cohort. Individual advance remains
available for deliberate milestone-by-milestone review.

The smoke gate additionally performs a separate eight-world,
two-update bounded lifecycle acceptance: cold C-configured student,
one complete PPO checkpoint, same-source target extension to two
updates, then validated actor export. The acceptance directory is
outside the six research students and never supplies their ancestry.
This verifies basic staged resume, not a mature active refresher.
The first live 110M intervention remains a monitored validation gate.

### Launch policy clarification

CI remains a watchdog, not a substitute for scientific validation. The
ABC smoke command no longer requires a green CI attestation as an
operator input. An optional `--ci-verified-sha` records a verified exact
commit; it cannot claim success for another commit. The registered
source, actual 2048-world mixed/clean/Academy GPU PPO acceptance and
the bounded checkpoint lifecycle remain mandatory for GO. No training
is launched by planning or locking. Inspect any substantive watchdog
failure, rather than treating a green badge as physical evidence.

Minimal WSL launch sequence (in the exact reviewed clean checkout):

```bash
.venv/bin/python -m scripts.ks150_abc_school plan
.venv/bin/python -m scripts.ks150_abc_school lock
.venv/bin/python -m scripts.ks150_abc_school smoke
.venv/bin/python -m scripts.ks150_abc_school go
```

GO is the separate explicit large-run command. Stop after smoke if
anything fails or the GPU acceptance output is physically unexpected.


#### Final practical acceptance before the campaign

The `smoke` command now runs three full 2048-world PPO updates with
mixed, clean and Academy cohorts, including two post-compile throughput
measurements and the existing physical motor-limit checks. It then
completes the isolated eight-world checkpoint/extension/export lifecycle
and exercises the **actual** discovery pulse and signed stair examiner
with that frozen acceptance actor (seed 9582, not a study seed). The
runner re-scores the pulse archive and checks the signed matrix identity
and 292 comparison cells. The acceptance actor is not expected to walk
well; this tests evidence delivery, not ability. It does not supply
ancestry to the six students.

No large-run launch is needed to test the complete path from PPO to
frozen physical evidence. The operator can inspect the steady GPU FPS,
acceptance archive and console before explicitly issuing GO. The local
GPU results are not yet available, and the first mature live C
intervention remains a separate review/validation gate.
