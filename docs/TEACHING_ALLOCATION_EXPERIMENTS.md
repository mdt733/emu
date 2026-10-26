# EMU — teaching-allocation experiment design

**Implementation status:** v5023 completed a real 260M-step owner-GPU
campaign. Its role weights are sampled once per episode and stored
until the episode ends. v5024 adds opt-in `steering` (35/15/20/15/10/5)
and separately measures incline stability versus commanded tracking.
See [V5024_RUN_PLAN.md](V5024_RUN_PLAN.md) for the next controlled
3×60M comparison and frozen terrain matrix.

## Two independent adaptive questions

1. **Difficulty within each specialty:** each physical command/recovery/
   slope/step/transition target makes small evidence-driven advances, holds
   and retreats. Failures at harder-than-target severity are normal.
2. **Attention across specialties:** what fraction of all PPO experience
   emphasises each of Commands, Resilience, Slopes, Steps, Transitions and
   Control/Grace? This is a *schedule of lesson probabilities*, not proof of
   walking ability, and should initially be treated as an experimentally
   selected configuration rather than another complicated governor.

A control/grace world has **no efficiency difficulty target**. It gets
easier versions of useful commands and terrain while the shared eight
baseline locomotion payments remain primary and mild additional motor/smoothness
shaping discourages waste *without rewarding stationary behaviour*.

No separate specialist policies, neural-network changes, PPO engines, model
templates or reward variants just to vary attention. User expects one
continuous cold-start learner and a compact, understandable codebase.

## Suggested allocations (experimental, not a fixed doctrine)

All rows describe percentage of **all** training worlds; each sums to 100%.
For the very first 70 PPO updates retain the original steady-only bootstrap
unless a separate explicitly identified experiment tests its removal.

| Steps | Commands | Resilience | Slopes | Steps | Transitions | Grace | Experimental intent |
|---|---:|---:|---:|---:|---:|---:|---|
| 0–20M | 75 | 10 | 0 | 0 | 10 | 5 | Establish gait and small realistic perturbations |
| 20–40M | 40 | 25 | 5 | 0 | 10 | 20 | Stabilise movement and improve quality |
| 40–60M | 25 | 20 | 25 | 5 | 10 | 15 | Learn sustained inclined contact |
| 60–80M | 20 | 15 | 20 | 20 | 10 | 15 | Practise discontinuous up/down crossing |
| 80M+ | 20 | 15 | 20 | 20 | 10 | 15 | Continue growing every local physical target |

**These times are candidate interpolation anchors, not success gates or
curriculum unlocks.** Use a smooth transition window instead of an abrupt
teacher switch. Age changes attention only; the within-skill targets use
their own physical results. Preserve modest rehearsal of previously learned
skills and occasional harder exploratory probes.

The 20M entry can be replaced with 10M if a cold-start experiment demonstrates
a usable gait earlier; the later attention schedule should be measured
*relative to the chosen foundation* instead of inheriting arbitrary 20M
constants. The three early recipes `main`, `command-heavy` and `curious`
already exist as separate saved experiment configurations.

## Candidate attention policies to compare

- **Balanced:** reference all-six 20/20/20/20/10/10 allocation after the
  foundation, ramping in without changing local target controllers.
- **Staged:** the table above, gently shifting emphasis through recovery/
  grace, slopes and steps. Rehearsal fractions remain nonzero.
- **Terrain-early:** introduce small amounts of real slope and step learning
  much sooner. Is early experience valuable even while the robot is immature?
- **Recovery-first:** spend more initial attention on starting disturbances,
  pushes and smoothness before geometric challenges.
- **Static-mixed:** choose one reasonable allocation and hold it forever.
  This is a useful control arm against elaborate developmental assumptions.

Vary only **one meaningful design dimension at a time** in the first
comparison: same robot and reward terms, same actor sensor contract, same
policy/optimiser, same evaluation cases, same seed and step budget.

## Implementation specification: minimal, honest dynamic allocation

1. Keep a small **named attention schedule** (knots: cumulative training
   steps and a six-element nonnegative weight vector, sum 1.0). Save the
   schedule name/full settings with the immutable training configuration.
   No arbitrary CLI forest and no Python callback inside the JAX rollout.
2. Host computes interpolated six focus weights at PPO rollout boundaries
   and passes them in the existing JAX **teacher-only task packet**.
   Fixed robot topology and JIT shapes must not change as weights move.
3. At episode birth, JAX draws **one** focus identity from the weight
   distribution and stores it in EnvState until the episode ends. Assign
   `specialist_active` separately if the post-foundation ramp is active.
   Do NOT recompute/relabel the role every physics frame, which would
   corrupt episode-level reward attribution and continuation logic.
4. Birth tasks, continuing transition targets, collisions, efficiency
   shaping and reported completion evidence must all use that **stored
   episode identity**. Teacher labels remain absent from the actor's
   360-value observation.
5. Zero-probability families must truly receive zero focus births and no
   focused evidence. Keep fallback ordinary gait lessons and use a
   nonzero grace share when scheduled. Non-focused physical challenges
   continue at reduced current difficulty (about half; grace lower).
6. Logs and offline report show **requested focus weights, actual focus
   births, completed episodes, physical task severities and outcomes**.
   Allocation is not equivalent to completed experience when survival
   differs across cohorts. Do not infer competence from requested mix.
7. Maintain strict source/config/checkpoint provenance. At a fork, any
   trial claiming a shared parent must restore **the full same-source
   actor, PPO optimiser, normaliser, teacher and health state**. Never
   pretend weights-only transfer is an identical learning continuation.

If these seven requirements cannot be met cleanly, prefer a simpler
stepwise experiment outside the training loop over half-implemented
dynamic scheduling.

## Experiment logic and cost control

**Question A: best early gait.** Screen `main`, `command-heavy`,
`curious` at 10M with one matched seed and frozen physical exam.
Promote promising combinations to seeds 733/734/735 and compare
standing, supported travel, tracking, fall rate, real contact, work
per useful distance and seed variability. 10M standing is not proof
of mature locomotion; do not punish natural early trial failures.

**Question B: best later sequence.** Choose one useful common early
policy/checkpoint then run balanced, staged and terrain-early curricula
as separate **same-source, fully restored** continuation roots when
that lifecycle is validated. This paired-parent design reduces variance
and avoids repeatedly relearning the same early gait. The full-fresh
end-to-end run remains a separate necessary check for cold-start ability.

**Question C: allocation can be responsive.** Only after the scheduled
arms are measured, try reallocating *small* percentages based on
opportunity, meaningful physical progress and persistent starvation.
Keep an exploration floor and bound allocation changes. Do NOT reward
a teacher simply for serving easy tasks or optimise away hard skills.

For each arm examine trajectories at 20/40/60/80M (or stages shifted to
10M), not just the final scalar. A high average reward cannot substitute
for genuine combined vx/vy/yaw, recovery, independent slopes, step
crossing, and energy quality. A curriculum that gains new skills but
forgets old ones is not the winner.

## Surface laboratory separate from the running experiment

The viewer-only branch `feature/v5023-surface-lab` is based on the
exact d6f3622 training source. It adds four 0.5–12 degree continuous
planes (uphill, downhill, left/right camber) and 0.1–6 cm real
step transitions for a loaded array-only 360-input policy.

This browser MuJoCo comparison is **not bit-identical** to training MJX,
nor is manual driving a valid held-out statistical qualification. Do not
pull new source into the active source-pinned trainer: launch the viewer
from an independent Git checkout with an exported read-only policy.

The existing slope governor requires 8 completed near-target episodes
in EACH of four directions, with >=65% success in every direction,
before increasing the ONE shared target by 0.08 degrees. An easy-direction
success rate below 30% can retreat that one target by 2.5% and reset
the accumulated windows. This may be unnecessarily conservative for the
shared angle, but the live training source must stay unchanged; gather
actual 2–5 degree viewer evidence before changing the teacher logic.

## Migration guard

Do not modify `main` or the approved Pinnacle policy/showcase. Keep the
current known source-pinned cold training path as fallback until the new
dynamic sampler passes Python/JAX CPU and 2048-world CUDA PPO smoke.
Neither this design document nor CPU CI demonstrates a successful GPU run.
