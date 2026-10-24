# EMU — unified learning contract

**Purpose:** one durable explanation of *what we are building and why*. For current progress and the next task, use [HANDOVER.md](../HANDOVER.md). Older experiments are preserved in [historical research](CAPABILITY_GROWTH.md) and Git history.

## The idea

**One robot. One brain. Many complementary lessons. Earn the next difficulty.**

Train one six-actuator biped from scratch with **one shared actor and PPO optimizer**, learning simultaneously in **2,048 MJX worlds**. The objective is not the highest scalar reward, speed, step height or slope. It is the *breadth and quality of reliable locomotion*: how many combinations of commands, terrain, disturbances and transitions the robot can handle **while walking with support, accuracy and reasonable mechanical effort**.

No fixed timetable automatically unlocks hard tasks. We maintain **evidence-backed capability frontiers**, sample near them, and expand, hold or retreat as real competence warrants. All worlds contribute to the same network every update.

## Five distinct responsibilities

```text
Physical world + robot  →  realistic observations  →  ONE actor
          ↑                                          ↓
   Task/curriculum teacher  ←  completed-episode evidence
          ↑                                          ↓
   Safety/learning governor       PPO + eight locomotion rewards
                             ↘
                    independent frozen-policy exam
```

1. **Physics** decides real contact, ground shape, support, displacement, step crossing and falls. Do not replace terrain contact with visual labels or tilted-gravity tricks.
2. **Actor** receives only physically plausible delayed measurements and user commands; it produces six actuator targets. No curriculum label, physical ground-truth normal or specialist ID leaks into its inputs.
3. **Reward/PPO** teaches useful behaviour *within each sampled task*. Keep the proven eight locomotion payments and one optimizer; reward is a teaching signal, never proof of mastery.
4. **Curriculum** chooses the next task distribution from actual completed-episode attempts, quality and severity. A single small governor owns difficulty decisions; maintain one shared total growth budget and faster retreat. **The original regime budget works; integrating new physical frontiers into that same budget remains work to do.**
5. **Examiner** independently measures frozen-policy physical competence on fixed, paired, held-out challenges. It reports strengths and failure modes; exam outcomes do not silently update the live training curriculum.

The browser viewer is an inference/debugging tool, not a representation of all GPU training worlds.

## Six simultaneous lesson emphases

These are **teacher-side emphases within one policy**, not separate brains or exclusive modes. The allocation is the *mature target*, progressively activated after basic walking is earned.

| Emphasis | Target worlds | Harder lesson |
| --- | ---: | --- |
| Commands | 20% (~410) | Signed forward/reverse, lateral, yaw and mixtures |
| Resilience | 20% (~410) | Pushes, recovery and awkward starts |
| Continuous slopes | 20% (~410) | Actual uphill, downhill and both cross-slope directions |
| Discrete steps | 20% (~410) | Real step up/down, rotated lips and supported landings |
| Transitions/posture | 10% (~204) | Start/stop, acceleration, reversals and chassis attitude |
| Control/efficiency | 10% (~204) | Smooth, economical **moving** gait on easier supported tasks |

Every specialist still encounters easier doses of the other challenges. Neither zero-exposure silos nor maximum-difficulty mixtures from birth are acceptable. Mix challenges more strongly **after** component competence, while keeping a small early crossover. Proportions may later adapt to information needs, but a capability must not silently starve.

Control worlds are not idle rest worlds. They reward supported motion, accurate tracking, smoothness, low slip and **bounded** simulated mechanical shaft work. Freezing must never be the efficient solution. Work/metre is a useful mechanical proxy, **not measured battery energy**.

## One evidence-and-growth rule

Use the same compact logic for each capability rather than inventing six governors:

1. **Birth:** sample a teacher-only focus, direction, severity and supporting challenges from the *earned* frontiers. Record what was actually presented.
2. **Attempt:** the actor receives realistic observations, walks, and produces measurable physical outcomes.
3. **Close the episode:** record attempted severity, supported progress, command error, survival/falls and relevant quality. An interrupted/failed attempt is failure evidence, not a success inferred from partial displacement.
4. **Attribute locally:** pool sufficient completed episodes in the *same meaningful direction and severity context*. No single lucky win, no recycled evidence after a frontier changes, no mixing easier attempts with frontier proofs.
5. **Decide:** advance gently on adequate repeated success; hold with sparse/ambiguous evidence; retreat faster on sustained failure. Reserve global gait retreat for genuine widespread collapse. Safeguard PPO health and share a bounded **overall** growth budget so several skills cannot jump together.

**Directions matter.** For velocity commands, the signed `vx/vy/yaw` limits define an **L1 octahedral envelope** (`|vx|/Vx(sign) + |vy|/Vy + |yaw|/Yaw <= 1`). Probe interiors, edges and modestly beyond, including pure-axis vertices. A successful diagonal is evidence for *that combination*, not permission to increase every independent axis. Both signs of lateral/yaw motion need evidence before expanding a symmetric limit.

For terrain, uphill does not equal downhill. An actual step ascent requires **supported upper-foot contact**, not merely the body passing a lip. A slope lesson needs real inclined **foot-ground contact** in all four directions, independently of step crossings. Report commanded **and achieved** velocities.

Keep **configured physical limits**, **sampled challenge severity**, **earned curriculum frontier**, and **independently demonstrated competence** as four separate facts. Never call one the other.

## Cold start: preserve what already worked

Historical `main` foundation: `2ec9e9db0a816c658131311129ef89ec7a595a12`.

- **20M steps; 2,048 worlds; seed 733; 32-frame PPO rollout; 4 epochs × 8 minibatches; LR 2e-4; maximum 700 control frames/episode.**
- First **70 updates (~4.59M steps) steady-only**. Initial `vx` requests were **+0.020…+0.060 m/s**. Later mix steady, transition and resilience in response to achieved gait rather than a schedule.
- Preserve the eight original gait/contact/clearance/smoothness/safety rewards, pooled completed-episode fall EMA, faster retreat, shared regime growth limit and PPO recovery behaviour.
- Specialist *birth activation* starts at zero while steady difficulty is ≤0.10 and gradually reaches its planned share around 0.32. Gentle real terrain can enter **when walking is earned**, not at an arbitrary 20M unlock. If competence retreats, challenge exposure can retreat.

These numbers are a reference to reproduce deliberately, not permission to copy mature training-profile defaults. The new 360-input actor will **not** reproduce the historical 220/232-input weights. Compare *walking quality* after its own fresh 20M foundation, not weight identity.

## One honest physical world and sensor contract

Unified training uses **two solid, oriented collision slabs** representing adjoining terrain. Per-episode poses vary actual pitch/roll, step rise/drop and independently rotated edge; feet and rays encounter **the same geometry**, with no per-world mesh recompilation. Flat/control episodes must have no ghost collisions. Physics and ray/height/support calculations must agree, especially at sloped step lips and descending edges.

The new actor remains **90×4 = 360 observations, six actions**. Its terrain sensing is a delayed **4×4 = 16-cell ToF array** with validity, body IMU and optical flow; local grade/confidence is *inferred* from sensor returns. The three old tripod ToFs are not part of this new actor. **8×8 is not implemented** and would need an explicit versioned redesign. Existing 220-, 232-, and earlier 360-input policies retain their own schema/replay semantics; same tensor length does not mean interchangeable sensor meanings.

## How we know it is better

A credible examiner reports at least:

- **Gait:** actual displacement, supported locomotion, falls, slip, tracking, posture and movement smoothness.
- **Envelope:** separately evidenced pure and mixed signed commands, accelerations, pushes and recovery.
- **Terrain:** real incline grade/direction, paired up/down step heights, upper-foot support, edge angle and failures; compare combined slopes+steps, not just isolated successes.
- **Efficiency:** mechanical work per *achieved* metre alongside movement, tracking, stability and action jerk.
- **Comparison:** the same frozen tasks/seeds for old vs new, with valid sensor/physics protocols and explicit confidence/attempt counts.

The **first fresh 20M unified actor** is a major checkpoint: compare it with main's proven foundation on matched flat tasks and inspect earned physical capabilities. If the basic gait deteriorates substantially, fix the teaching distribution before committing to 100M+. A later ~10-hour run is a **single resumed lineage of the same shared actor**, not parallel bespoke speed/slope/step networks.

## Boundaries

- Development stays on `feature/v5023-capability-growth`. **Do not edit or merge `main`**, its approved **260M Pinnacle**, or the public showcase without explicit approval.
- Keep historical source/checkpoint lineages pinned. A new actor contract or source requires its **own cold-start lineage** unless a migration is deliberately proved.
- Prefer the existing `environment.py`, `curriculum.py`, `rewards.py`, `ppo.py`, `training.py`, `results.py` and viewer/evaluation owners. Replace duplication; do not create new `v502x` managers/launchers to paper over it.
- **No unified overnight GPU campaign until** full physical-contact acceptance, joint examiner, actual RTX 4060 Ti CUDA PPO/checkpoint/resume/export/viewer lifecycle and cold-start settings are validated. The source deliberately fails closed for now.

**That is the design.** Status, verified commit, outstanding acceptance gates and *the next single engineering action* belong only in [HANDOVER.md](../HANDOVER.md).
