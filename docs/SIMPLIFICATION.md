# EMU simplification — the one living worklist

**Aim:** a technically serious robot-learning system that reads like a book.
The reward and curriculum maths can remain sophisticated; the *ownership,
interfaces and top-level workflow* should be obvious in a few minutes.

**Canonical reference:** [architecture.md](architecture.md).
**Work branch:** `refactor/emu-readable`, forked from
`feature/v5016-directional` at `18b566d`. The ongoing old-policy GPU exams
should continue on their original pinned source; do not update their checkout.
**Do not merge this branch, publish a showcase or resume an existing PPO
lineage using its modified source** until the independent GPU acceptance gate.
The current 20 software checks only show contract compatibility.

## Definition of simple

- The first screen of `training.py` explains the whole loop.
- Each decision has one owner and one obvious place to change it.
- Physical measurements, reward payments, learning updates and examination
  judgements are separate concepts with traceable data flow.
- A new training hypothesis changes a recipe/config, not the PPO optimiser.
- A run writes an independently understandable story, not just rankings.
- The viewer loads a frozen brain and cannot secretly affect training.
- Less code/files is welcome when real duplication disappears, but no target
  line count justifies sacrificing physical correctness.

## Passes (update *this file*, not a new pass-N document)

- [x] **A. Make the map and name the owners.** Record the seven responsibilities,
  their actual current owners, data boundaries, missing separations and
  verification requirements. Rename the misleading `scoring.py` training
  reward module to `rewards.py` without altering numbers. Simplify the
  main documentation entry point. **Status: complete; 16 Python + 4 browser checks and viewer build passed on `d2abc8b`.**
- [ ] **B. Make the universe legible.** **B1 verified:** task generator and schema moved to `emu/tasks.py` with original JAX methods and RNG split order. [CI](https://github.com/mdt733/emu/actions/runs/38835597285) passed 16 Python + four browser tests. **B2 source verified:** `_step_impl` is a readable five-stage control frame, `_reset_with_keys` composes task → birth physics → initial observation, and `tasks.py` advances transition commands. Same intended RNG, reward timing, 220-input actor and auto-reset rules. [Code CI 38836739692](https://github.com/mdt733/emu/actions/runs/38836739692) passed **16 Python and four browser checks** (including real MJX CPU smoke); fresh CUDA update/save/resume remains outstanding. Trace one `reset → task → step →
  physical observation` path; group `environment.py` by actual
  responsibilities. Separate task sampling from physics where it reduces
  coupling; keep batched model leaves and JIT behaviour efficient. Clearly
  distinguish *physical truth* and a sensor available to the real robot.
  **Gate:** small real MJX CPU smoke + isolated CUDA step; no changed reward.
- [ ] **C. Rewards and diagnostics.** **Source refactor prepared; CI pending.** The teaching formula is now a short top-level narrative with three coherent ingredients, physical metrics are explicitly grouped, and weighted payment names/reward-only fields have one owner. This does not enable an electrical power reward. Make `rewards.py` accept documented
  physical measurements and expose the eight component payments plus
  fall cost. Keep `rollout_metrics.py` responsible for independent
  measurable facts. Remove duplicate logging conversions. **Gate:** reward
  arithmetic equivalence (including falls) and honest telemetry.
- [ ] **D. Training reads top to bottom.** Expose a short visible lifecycle:
  prepare or resume → create many worlds/learner → collect/update →
  observe/revise curriculum → log/save → finish. Move coherent responsibilities
  out of the 975-line orchestration without creating wrapper mazes.
  **Gate:** fresh isolated GPU update/save/resume and identical step/seed
  semantics; keep atomic checkpoints.
- [ ] **E. Curriculum and teaching capability.** Draw one path from rollout
  evidence to allowed task distribution. Decide which governor, field,
  recovery and health complexity *earns its keep*; remove overlapping or
  unproven machinery only after comparison. **Gate:** visible physics
  command envelopes, retreat and no hidden held-out leakage.
- [ ] **F. Meta-learning/experiment recipe.** Make shared foundation,
  independent teaching strategies, branch budgets and common exams readable
  in one recipe. Consolidate planning/promotions and avoid duplicating
  trainer/config logic. **Gate:** validated ancestry and equal-budget
  apples-to-apples results.
- [ ] **G. Reports and viewer, then prune.** One narrative `results.md`
  + bundle that shows learning curves, setbacks, commands, falls, measured
  load and policy ancestry; separate frozen-policy export/viewer. Retire
  stale wrappers, redundant historical pass docs and dead commands. **Gate:**
  policy inference parity, viewer build and realistic held-out exam.
- [ ] **H. Release-level confidence.** Run the 20 retained code checks and
  fresh-root real CUDA training/checkpoint/resume. Run 0°, 0.25°, 0.5° and
  carefully approached 1° directional exams against frozen policies.
  Compare source SHA, policy hashes, course progress and actual footfalls.
  **Gate:** no main/champion overwrite and documented behaviour differences.

## Before deleting any code

Answer in this order:

1. **Who owns it?** Is it serving physics, reward, learner, curriculum,
   experimental design, reporting or frozen-policy viewing?
2. **Is it executed?** Locate an actual call site, not only a mention in docs.
3. **What factual behaviour would disappear?** Find the input/output contract.
4. **Can one owner replace the duplication?** Prefer deleting a branch over
   adding an adapter.
5. **How do we know it's safe?** One strong check or isolated run is more
   useful than dozens of shallow tests.
6. **Does its source belong to an active checkpoint lineage?** If yes, do
   not change that lineage; experiment in a separate revision/run root.

## Next “keep going” starting point

**Pass D**: make the ~1000-line `training.py` read top-to-bottom. Start by mapping preparation/resume → batched rollouts/PPO → curriculum and health → held-out exams → reporting/checkpoint/finalisation. Extract only coherent ownership boundaries; no wrapper maze and no checkpoint schema changes. **The outstanding B/C CUDA acceptance gate still applies** to this new source.

## Evidence ledger

| Revision | Actual change | Verification |
|---|---|---|
| `18b566d` | Baseline v5016 experimental branch, 20 retained checks | CI: 16 Python + 4 browser passed |
| `d2abc8b` | Pass A: seven-layer framework, reward rename, readable entry points and single worklist | [GitHub CI 38834067868](https://github.com/mdt733/emu/actions/runs/38834067868): **16 Python + 4 browser passed**, build passed; no GPU run |
| `3102062` | Pass B1: separate lesson sampler into `tasks.py`, keep same RNG and physics; add root `HANDOVER.md` | [GitHub CI 38835597285](https://github.com/mdt733/emu/actions/runs/38835597285): 16 Python + 4 browser passed; viewer build passed; GPU run still required |
| `df82901` | Pass B2: explicit physical/lesson/reward stages; small reset/step orchestrators | [GitHub CI 38836739692](https://github.com/mdt733/emu/actions/runs/38836739692): 16 Python + 4 browser passed, viewer built; no GPU run yet |
