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
- [x] **C. Rewards and diagnostics.** **Source code/CPU CI verified at `39651e4`** ([run 38837791165](https://github.com/mdt733/emu/actions/runs/38837791165)); isolated CUDA acceptance still owed before merging. The teaching formula is now a short top-level narrative with three coherent ingredients, physical metrics are explicitly grouped, and weighted payment names/reward-only fields have one owner. This does not enable an electrical power reward. Make `rewards.py` accept documented
  physical measurements and expose the eight component payments plus
  fall cost. Keep `rollout_metrics.py` responsible for independent
  measurable facts. Remove duplicate logging conversions. **Gate:** reward
  arithmetic equivalence (including falls) and honest telemetry.
- [ ] **D. Training reads top to bottom.** **D1–D3 code checks verified:**
  [resume and rollout ownership](https://github.com/mdt733/emu/actions/runs/38842290106),
  [frozen exams](https://github.com/mdt733/emu/actions/runs/38842871586),
  [metrics and field artifacts](https://github.com/mdt733/emu/actions/runs/38843252714).
  `training.py` is ~750 lines (down from 993); 20 tests retained, with stronger
  resume and delivery assertions. D4 still needs to clarify recovery and PPO
  lifecycle without wrapper mazes or altered rollback semantics. **D4 CI verified:** 
  [38844333684](https://github.com/mdt733/emu/actions/runs/38844333684)
  passed 16 Python + 4 browser and build. Complete policy/teacher/normalizer/LR
  rewind plus fresh world reset lives in `run_support.py`, and its existing
  checkpoint test exercises the restored state. **D5 source submitted:**
  KL-adaptive LR moved to `health.py`, preserving deadband and clamp;
  no new test files or training payment changes. **GPU acceptance remains due.**
  **Gate:** fresh isolated GPU update/save/resume, identical step/seed
  semantics, and atomic checkpoint validation; no changed reward.
- [ ] **E. Curriculum and teaching capability.** **E1 source pending CI:** separated the established physical-frontier update into a named method in the same `curriculum.py` file, preserving every coefficient, EMA, hysteresis and safety limit; strengthened one of the existing 20 checks for adaptive-on/off behaviour. Next E2: trace capability-field evidence to real task sampling before removing anything. Draw one path from rollout
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

**Pass E2:** physical frontier update is now a single readable subdecision. Examine `capability_field.py`'s evidence, decay, teaching-packet and task-sampler call paths for actual overlap or unused complexity, then cut only what can be proven safe. Trace rollout evidence to the sampled task envelope, identify unused or overlapping controls by real call sites, then simplify only where behaviour and matched-policy comparisons justify it. D5 code CI and independent GPU acceptance remain open. Keep exact rewind semantics; run isolated CUDA learning/resume/export and held-out physical exams before merging. The 220-input actor and schema-7 checkpoint are unchanged.

## Evidence ledger

| Revision | Actual change | Verification |
|---|---|---|
| `18b566d` | Baseline v5016 experimental branch, 20 retained checks | CI: 16 Python + 4 browser passed |
| `d2abc8b` | Pass A: seven-layer framework, reward rename, readable entry points and single worklist | [GitHub CI 38834067868](https://github.com/mdt733/emu/actions/runs/38834067868): **16 Python + 4 browser passed**, build passed; no GPU run |
| `3102062` | Pass B1: separate lesson sampler into `tasks.py`, keep same RNG and physics; add root `HANDOVER.md` | [GitHub CI 38835597285](https://github.com/mdt733/emu/actions/runs/38835597285): 16 Python + 4 browser passed; viewer build passed; GPU run still required |
| `df82901` | Pass B2: explicit physical/lesson/reward stages; small reset/step orchestrators | [GitHub CI 38836739692](https://github.com/mdt733/emu/actions/runs/38836739692): 16 Python + 4 browser passed, viewer built; no GPU run yet |
| `974d857` | Pass D2: fixed/cloud exam implementation moved to their evidence owners | [GitHub CI 38842871586](https://github.com/mdt733/emu/actions/runs/38842871586): 16 Python + 4 browser, build passed; CUDA unverified |
| `22e8edb` | Pass D1: resume ownership and rollout provenance | [CI 38842290106](https://github.com/mdt733/emu/actions/runs/38842290106): 16 Python + 4 browser passed; no GPU run |
| `ebd1153` | Pass D3: canonical training metrics and field artifact writers, strengthened delivery check | [CI 38843252714](https://github.com/mdt733/emu/actions/runs/38843252714): 16 Python + 4 browser passed; no GPU run |
| `a1b43f9` | Pass D4: complete recovery rewind and step-seeded physical restart; existing checkpoint test enlarged | [CI 38844333684](https://github.com/mdt733/emu/actions/runs/38844333684): 16 Python + 4 browser passed; no GPU run |
| `12d47b2` | Pass D5: soft PPO KL controller owned by health; existing assertions strengthened | [CI 38844623465](https://github.com/mdt733/emu/actions/runs/38844623465): 16 Python + 4 browser passed; no CUDA training |
| `39651e4` | Pass C: readable reward computation, single source for payment names/reward-only fields, clear host diagnostics, strengthened existing servo-power invariant test | [GitHub CI 38837791165](https://github.com/mdt733/emu/actions/runs/38837791165): 16 Python + 4 browser passed, viewer built; 24 MuJoCo warnings; GPU gate open |
