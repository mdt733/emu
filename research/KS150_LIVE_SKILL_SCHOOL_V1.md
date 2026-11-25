# KS150 live skill school — phase 1

Base: the completed A/B/C research source
`7b6c3f9e10b9b6ad2f7302dad00d9d25bbd5d574`. No champion change.

## Why

The six A/B/C learners showed a persistent uphill acquisition barrier and
substantial recovery regression during broad refreshers. Use the ordinary
training worlds as the primary source of developmental feedback. The actor
continues to receive 360 observations; all training worlds continue PPO.
Independent frozen exams remain useful at registered start/end milestones,
not as routine teacher queries.

## Implemented, opt-in

`--live-skill-observer` records sparse integer attempts and passes in
`RUN/live_skills/segment-*.jsonl`. Each PPO rollout contributes once,
including the last rollout. Windows cover roughly 1M world-steps and end at
rollbacks and process exits. A new process starts a new segment, so evidence
from rewound checkpoints is never silently presented as uninterrupted
learning. Incomplete episodes are not invented as attempts. The observer
does not feed the governor, field, reward or actor.

The existing command-window, birth-recovery and transition-edge state
machines in CapabilityField supply their raw event verdicts. We reuse their
counts rather than duplicating their frame loops. The ledger also records
role survival, Academy first-birth signed lip height and first-lip arming,
bilateral readiness, safe bilateral transfer and goal completion; ordinary
first-encounter safe crossing; and slope stability and command success by
terminal physical grade and direction. Gate counts share their stated
denominators. These are distinct observations, not one composite score.

The header stores the physical bin edges, command lattice, relative recovery
bin count, edge directions and field-taught evidence policy. Command windows
are not complete missions; relative recovery severity is not an absolute
kick; terminal grade is not guaranteed to be the episode's initial grade.
Do not compare changing exposure distributions as though they were fixed
reference cohorts. No ledger cell yet grants mastery or triggers repair.

The 4096-world configuration retains 32 rollout frames and four PPO
epochs, increases minibatches from eight to sixteen, and records the
effective count in run_meta.json. Both 2048 and 4096 therefore have
8192 experiences per minibatch, but 4096 has half as many policy updates
per fixed world-step budget. The larger classroom is the target, not a
claim of measured throughput improvement. Existing 2048 doctrine is
unchanged. Additional counters reuse the rollout's existing host transfer
except two physical scalar metrics per frame; no additional simulator or
actor is created.

## GPU acceptance (before an actual experiment)

From this exact clean research checkout, using the existing project venv:

```bash
export EMU_PHYSICAL_PROFILE=h40-spring-only
export EMU_UNIFIED_ACCEPTANCE=bounded-ppo-lifecycle
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/emu/jax-live-school"

python -m scripts.live_skill_smoke --root "$HOME/emu-data/studies/live-skill-smoke-v1"
```

The smoke runs two real PPO updates at each of 2048 and 4096, writes
checkpoints and observer logs and checks integer evidence integrity and
the recorded effective PPO minibatches. This tests resource and integration
compatibility, not mastery. GPU timings are not yet measured.

## Next development slices (not active yet)

1. Capture absolute birth disturbance vectors and approach-command context.
   Separate clean reference tasks from mixed practice and actual terrain.
2. Reserve stable reference births inside the same PPO classroom. Preserve
   their distribution across updates and treatment arms; record the assigned
   task even if the robot never reaches the intended obstacle.
3. Introduce a windowed skill ledger with minimum evidence, uncertainty,
   spaced retention tests and separate acquisition / regression flags.
   Reference tasks supply the decision denominator, not changing ordinary
   difficulty. No IID assumptions about repeated windows.
4. Bounded task-specific birth allocation and prerequisite-based stair
   practice. Match practice budgets, protect generalist/maintenance shares,
   and test shaping/assistance removal. Start with advisory decisions and
   then enable the teacher only in registered research arms.
5. Matched warm-fork acquisition and retention labs; retain parent learner
   and optimiser, restart live physical episodes, authenticate source and
   checkpoint bytes. The existing A/B/C C pinned programmes require
   deliberate compatibility work before a new research fork; do not bypass
   their lineage guard. Independent exams only at major registered endpoints.

No new large-run launch is authorized by this phase. The observer is passive
until the measurement and reference birth contracts are validated.


## Phase 2 — fixed-budget, task-specific recovery teacher

Implemented opt-in switches:

- `--live-skill-observer --skill-reference-recovery-share .15`: stable
  uniform six-case signed longitudinal birth offsets at a .16 m/s
  forward command on flat prisms. Sensor scale 1, step command .012m,
  no added random pushes, tilt or carried-over joint/base velocity.
- `--skill-practice-recovery-share .15`: a separate equal-budget cohort
  learning the same six cases, uniformly by default.
- `--skill-adaptive-recovery`: redirect only the practice cohort using
  evidence from the unchanged reference cohort. Maximum .15 practice and
  .30 combined share of eligible mature RESILIENCE births, not whole
  classroom frames. Ordinary, STEP/Academy and athletic role draws remain
  unchanged. Reference assignment uses an independent fixed case draw.

Reference/practice episodes close at exactly 150 control frames (3s) or
earlier physical failure. A mission requires 25 uninterrupted stable
tracking frames within that time AND surviving to the deadline. The
observer records settling, survival, mission and actual PPO frame exposure
separately. These short trials must not be treated as equivalent to
long-horizon survival exams. They are excluded from relative-severity
capability field events and ordinary resilience frontier evidence.

The teacher uses disjoint blocks of at least 40 completed reference
trials, two supported mastery blocks and repeated statistically screened
regression before repair. Acquisition selects developing tasks with two
recent blocks in the 30–80% range. It prefers repair, preserves half the
practice distribution as uniform maintenance, and changes each case's
weight by at most 5 percentage points per review window. No new rewards.
The uniform and adaptive arms have identical birth budgets, not
necessarily equal realized frame occupancy; report both.

Teacher decisions are appended to `live_skills/teacher-segment-*.jsonl`.
For now, the live teacher deliberately restarts from uniform practice
and a fresh evidence epoch on process restart or rollback. Previous
raw evidence remains on disk. Checkpoint-bound teacher restoration is
still needed before a long segmented adaptive campaign; do not silently
join contradictory policy histories. The online reference ledger is a
descriptive training measurement, not a held-out test or independent
training replication.

### Active mature GPU acceptance

The earlier cold two-update smoke cannot exercise mature references.
In the clean research checkout, run this additional bounded acceptance
using the project venv (not a research training launch):

```bash
EMU_PHYSICAL_PROFILE=h40-spring-only \
python scripts/smoke_gpu_carpet.py --worlds 4096 --device cuda \
  --prisms --ppo --full-update --stair-survival-curriculum \
  --live-recovery-smoke --benchmark-updates 5
```

It verifies identical reference IDs and non-practice observations
under different practice distributions, actual flat prism geometry and
birth velocities, active real PPO and terminal reference/practice
verdicts. This GPU acceptance is not yet executed. Exact committed
reference, ledger and teacher blobs were copied to an isolated host
and verified by Git blob hash; CPU JAX 4096-world assignment/velocity
and settling/deadline tests plus repeated-regression, weight bounds and
epoch-separation tests passed. This is not a substitute for MJX physics.

Next: checkpoint-bound teacher memory and uphill acquisition teaching,
then matched uniform/adaptive recovery experiments from authenticated
parents. No main or champion changes.


## Phase 3 — checkpoint-owned teacher and physical stair evidence

The adaptive recovery teacher now owns a compact evidence clock, its six
practice weights, closed reference windows and the unfinished reference
counts. These are saved with the actual learner in latest, rotating,
final, evaluated-best and recovery checkpoint metadata. The teacher's
review interval uses observed world-steps, independent of report-file
flush boundaries. Same-run restart restores the saved state; no log
replay or pooling of abandoned episodes. A new external research fork
starts a new teacher. A rollback restores the selected checkpoint's
teacher before selecting new reset births, discarding the subsequent
teacher history. Raw logs keep their separate process/rollback segments.
An evaluated recovery snapshot carries the teacher state from before the
rollout that evaluated that policy, consistent with the existing learner
and field snapshot timing. Live simulation episodes restart on resume.

The passive stair observer now collects physically distinct stages on
the original first target tile: approach zone, authentic top contact
(including a leading foot before torso crossing), five consecutive top
contact frames per foot independently, five consecutive safe traversal
frames at least 10cm beyond the edge, and traversal plus episode survival.
All use the actual signed first birth lip and the existing real prism
contact matrix. These measures are not the old teacher's root-gated
first-lip arming or disabled stationary goal reward. Bilateral contact
is not a direct measurement of transferred load. No stair reward,
frontier, assignment, actor input, or active stair teacher changes yet.
Next: fixed signed uphill reference births and prerequisite practice,
matched budget against the current Academy. GPU acceptance remains
outstanding. The teacher state and first-lip gate tests are added;
the full physical lifecycle still needs the real GPU run.


## Phase 4 — isolated uphill prerequisite laboratory

New optional recipe switches:

- `--skill-uphill-reference-share .15`
- `--skill-uphill-practice-share .15`
- `--skill-uphill-stage none|first_support|bilateral|traversal`

These reserve at most 15% + 15% of eligible mature Academy births,
not all training worlds. Both cohorts sample the same six exact upward
lip heights uniformly: 4, 8, 12, 20, 28 and 32mm. They start 22cm before
the single real +X prism seam, centered in Y, .16m/s forward request,
zero pitch/roll/yaw/lateral command, fixed sensor scale 1, no extra
resilience disturbance or velocity carryover. Both end after 350
control frames (7s) or physical failure. The same eight real prism
colliders, actor, normalizer and PPO are used. Ordinary Academy births
retain their existing course and teacher. Isolated uphill trials do not
count toward the legacy Academy promotion or ordinary steady frontier
gates. Global training health still sees all actual experience.

The reference cohort has the original reward. In the practice cohort
only, the optional selected prerequisite pays four reward units once
per life on a newly demonstrated milestone, never on the failure frame.
The `none` control uses identical physical exposure with no added bonus.
No permanent mastery claims, adaptive height weights or automatic
reward fading are enabled yet. A new advisory function
`diagnose_uphill` identifies repeated deficits from two disjoint blocks
of at least 40 reference attempts per height, using shared denominators
and a 15 percentage-point minimum repeated stage loss. An initial
first-support deficit may instead be a failure to reach the lip; inspect
the approach/contact counts before selecting a treatment. Post-traversal
survival gaps are reported but are not a selectable bonus stage.

The observer separates first contact, first supported foot, each foot
independently supported, five-frame traversal and traversal plus
survival. It reports actual sampled frame exposure by case and cohort.
The extra prerequisite credit is exposed in rollout metrics. Stable
reference counts are distinct from shaped practice counts. Neither
is an independent held-out examination.

Exact committed uphill assignment and advisory modules were copied to
the local CPU test environment and checked by Git blob hash. A
4096-world JAX synthetic test preserved all 323 reference assignments
when adding 320 practice births; height-index assignment, no-repeat
bonus, reference/ordinary/failure-frame exclusion and repeated-gap
advice passed. Additional repository ledger tests are included; the
actual physical integration remains unexecuted.

GPU acceptance (not a training campaign):

```bash
EMU_PHYSICAL_PROFILE=h40-spring-only \
python scripts/smoke_gpu_carpet.py --worlds 4096 --device cuda \
  --prisms --ppo --full-update --stair-survival-curriculum \
  --live-uphill-smoke --benchmark-updates 8
```

This checks exact physical lip heights, spawn positions, flat sides,
active reference/practice cohorts, reward isolation and terminal
verdicts when present during actual PPO. Eight updates cover 256
frames, shorter than the 350-frame horizon; a longer acceptance must
verify completed surviving episodes before research launch. The next
slice is the bounded checkpoint-owned stair stage/difficulty allocator
and a matched uniform-versus-prerequisite experiment. No champion
promotion or large-run launch is authorized.
