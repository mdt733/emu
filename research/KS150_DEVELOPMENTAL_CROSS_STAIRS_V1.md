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
