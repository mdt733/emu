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
