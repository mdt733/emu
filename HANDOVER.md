# EMU — active handover

**Branch:** `feature/v5023-capability-growth` · **Updated:** 24 October 2026  
**Design and permanent non-regression contract:** [Unified learning contract](docs/UNIFIED_CAPABILITY_GROWTH.md#non-regression-contract--what-must-never-be-traded-away)  
**Protected reference:** `main` at `2ec9e9db0a816c658131311129ef89ec7a595a12`

## Mission

**One cold-startable 360-input actor, one PPO optimizer, 2,048 MJX worlds.** Preserve the working **20M-step main-style foundation** (original gait physics, eight reward payments, 70-update steady-only bootstrap, conservative commands, growth/retreat and checkpoint safety), then grow coherent command, recovery, real slope, real step, transition/posture and economical control capabilities. All experiences train the **same** robot brain; specialist labels remain teacher-only.

The six mature emphases target **20/20/20/20/10/10%**. Every world still sees gentler doses of other challenges. Every completed episode adds directional, severity-aware evidence; frontiers expand only on enough real success and share one bounded growth budget. No invented step crossings or PPO reward masquerading as independent walking ability.

**Foundation→growth now implemented (unverified walking):** the unified 360 actor starts with main's command ramp and foundation resilience burden; its teacher-only specialist dose is zero through 20M, then blends in over ~2M steps weighted by earned steady difficulty. This controls role assignments, real slope/step births, and the switch toward evidence-backed command limits. Sensor schema remains 360 throughout; no 220-to-360 checkpoint migration.

## What is implemented

- **Learning:** one governor with the original four core difficulties, separate closed-window signed L1 command, four-direction slope and genuine step-up/step-down evidence. Specialist expansions now pass **one combined growth arbitration**, with full retreats and a PPO-health expansion veto. One `advance_curriculum` entry point handles normal and final rollout decisions.
- **Robot/world:** 6 actions, delayed **4×4 ToF array** and IMU/flow within a versioned **90×4 = 360** actor. Two real inclined solid collision slabs permit sampled pitch/roll, rise/drop and rotated edge. Old 220/232/360 actor replay is a separate preserved contract.
- **Single run:** `emu train` → `emu/run.py` → `training.py`; source-pinned plan, checkpoint/resume, reporting and viewer. `foundation_config` is intact. The public 260M Pinnacle and guarded Pages release remain protected.
- **Repository cleanup:** removed old ladder/lab5015, Pinnacle campaign+upgrade runners, multi-arm evaluator and directional-batch runner, their exclusive tests/scripts, oversized dated historical documents, and dead viewer stage sorting. Old code and research remain available in Git history; `main` was not changed. Active docs are README, this handover, the learning contract, INSTALL and PLAYGROUND. Tournament results replaced by a small single-run on-policy evidence report.

## Verified versus pending

- **Previous integrated engineering SHA `b7893fd`: four CI jobs passed** (Python, 12-world real MJX/PPO CPU, browser, clean Ubuntu viewer). The larger cleanup commits require their **own latest-HEAD CI**; don't describe old green checks as proof of newer source.
- **Not yet verified:** actual cold unified 20M first-stage gait parity vs main; accurate frozen **combined** slope+step+command+recovery examiner and browser simulation parity for a freshly trained array-only actor. **The 2,048-world CUDA PPO/checkpoint/resume/export lifecycle PASSED on owner hardware at `6653c03`; see dated evidence below.**
- The unified trainer **intentionally fails closed**. There is no authorised ten-hour unified GPU campaign and **no trained unified successor** to Pinnacle yet. Existing frozen older 360 runs climbed poorly: supported ascent stayed 0/12 in matched +0.15 m/s exams despite better flat survival and descent. Do not label far-side root motion as an ascent.

## Live preservation checklist (details in the design contract)

**From main — preserve, not merely resemble**

- [x] Code-level **20M foundation** core: 70 steady-only updates, 32×4×8 PPO, LR2e-4, seed733/2048 envs, eight weighted gait payments, phase/command cap and effective early 0.70N push exposure. Existing `tests/test_learning.py` now checks source-level numerical parity. **Does not prove gait quality.**
- [x] Intended 360 actor born with same versioned realistic sensors; no specialist or hard physical terrain before 20M; gentle competency-driven fade thereafter. **Physical/training distribution only verified in small CPU smoke.**
- [ ] **Real first-20M parity:** a fully trained cold 360 actor must demonstrate support, achieved vx/vy/yaw tracking, stability, commanded pace and acceptable motor burden comparable to main on *matched flat frozen exams*.
- [ ] Verify that early **posture, command schedule, normalizer and capability-field shadow behavior** remain observationally equivalent enough to main; similar configuration values alone do not establish this.

**From our unified vision — earn, don't invent**

- [x] Six teacher emphases, one actor/optimizer, signed L1 command probes, direction-aware closed windows for slope/steps and shared expansion arbitration implemented.
- [ ] Demonstrate each teacher's **actual** task mixture and evidence attribution from completed episodes. Check underrepresented directions, sparse evidence, and whether successful diagonal commands wrongly expand unproven pure axes.
- [ ] Validate frozen array-only **19-case** physical exam on a real exported unified policy: all four signed slopes, rotated/paired supported up/down steps, mixed surfaces, commands, pushes, transitions, grip/slip, action changes and mechanical J/m. No such trained policy exists yet.
- [x] Full **2,048-world CUDA** PPO/checkpoint/resume/export and strict array-only actor decoder PASS at `6653c03` (source-pinned). [ ] Frozen 360-actor physical examiner and browser simulation parity still pending; sustained FPS still unmeasured.

**Never claim a replacement is better** using reward, a green CPU smoke, an examiner with zero trained-policy runs, or a single terrain case. Preserve `main`, approved 260M Pinnacle, public showcase and archived checkpoints. **Next action:** finish current-HEAD CI, then isolate GPU lifecycle and frozen policy examination. 

## 25 Oct WSL acceptance — observed evidence and correction

- **GPU preflight PASS on RTX 4060 Ti:** real 2,048 mixed-physics worlds, 360 observation channels, six roles (410/410/409/409/205/205), one full 32×4×8 PPO update/65,536 frames; 63.5s compile-inclusive and reported JAX peak allocation ~811MB. No learned walking claimed; that FPS includes compilation.
- **First actual trainer stage at 8 worlds reached its PPO update**, then failed on best checkpoint save with `sensor contract fingerprint/geometry mismatch`. The independent checkpoint reader was reconstructing *legacy 360 tripod+array* while training supplied *new 360 array-only*. The mismatch, not PPO divergence, was the fatal traceback. CUDA preallocator logged unsuccessful large reservation attempts, but the first update completed.
- **Corrected on branch:** `checkpoint_io.py` now reconstructs canonical geometry from the declared sensor revision and still requires caller-pinned lineage; archived 360 tripod and new array-only 360 remain strictly distinct. Fixed evaluation worlds also now instantiate the actual unified two-slab, array-only sensor physics. Added regression tests covering on-disk checkpoint save/load, tamper rejection, old-vs-new 360 separation and frozen-world topology. Bounded lifecycle acceptance also uses the same non-preallocating allocator mode as the successful GPU preflight.
- **Still pending:** current HEAD CI and a *fresh-root* 8-world lifecycle first→continue→verify, followed by fresh-root 2,048-world lifecycle and independent 360 exported exam. **Do not reuse the failed `unified_accept_7d3d6ae_8` root, silently patch its provenance or delete its evidence.** No new trained 20M policy exists; long-run guard remains closed.

## 25 Oct follow-up — first complete CUDA trainer update

- User ran a fresh 8-world unified `first` stage at SHA `162f079`. **The actual trainer completed 256 steps and saved `first/checkpoints/final.pkl`**. The *acceptance wrapper* then incorrectly called `load_checkpoint(...)` with implicit legacy schema-7 defaults and rejected the valid schema-9 terrain actor. This is a **harness regression, not a new neural-network failure**.
- The 8-world smoke also wasted **467.8 seconds** on the final 10-panel fixed walking exam, unsurprisingly reporting 100% falls for an almost-untrained 256-step policy. That is *not* evidence of foundation quality.
- Fixed together: the smoke now validates every saved checkpoint against `TrainingConfig.observation_schema` and independently reconstructed `training_sensor_contract` (including the 4×4-only revision). The explicitly two-update acceptance path now **skips building/running** the expensive fixed walking panels; `main` and all normal, non-bounded evaluation schedules remain unchanged. Its purpose is exclusively complete PPO/checkpoint/resume/normalizer/export/replay plumbing; the real `capability-exam` follows separately.
- Added regression test that invokes the **actual lifecycle checkpoint loader** on a schema-9 array-only checkpoint and confirms it rejects the archived 360-tripod identity. CI needs to pass on the final SHA before repeating a *fresh-root* 8-world first→continue→verify. No overnight authorisation yet.
- Developer test package is optional: plain `setup_wsl.sh --cuda13 --install-node` does **not** install pytest. To run pytest locally, explicitly `python -m pip install -e '.[dev]'`; CI installs it already. Don't treat missing pytest as CUDA or PPO failure.

## 25 Oct production CUDA lifecycle — VERIFIED on owner hardware

- **Clean feature SHA `6653c0386d2d4bb10718cb364c815eb1f4eb7f92`: PASS.** RTX 4060 Ti ran the real unified 360-input trainer at **2,048 simultaneous MJX worlds**, from cold state for 65,536 environment steps, saved a validated schema-9 checkpoint, then resumed a NEW run directory from that exact ancestor and completed at 131,072 steps.
- **Verified evidence:** `PASS stage 1`; `PASS stage 2`; checkpoint lineage `65,536 → 131,072`; observation-normalizer count `65,536 → 131,072`; ancestor checkpoint content unchanged; completed resume no-op; `.emupolicy` exported and decoded through array-only 4×4 sensor contract. Artifact: `~/robot_lab/emu_runs/unified_accept_6653c03_2048/tiny_smoke_actor.emupolicy`. Small 8-world lifecycle at the SAME SHA also passed.
- **Source CI at that exact SHA:** all four jobs successful (Python, unified MJX CPU, Ubuntu viewer, browser). Protected `main` remains `2ec9e9db`.
- **Do not mistake the logged 2.4–3.2 million FPS for sustained throughput.** In one-update training, the reported runtime clock starts after the expensive first update, so the denominator is nearly zero. Separate 2,048-world production-shaped PPO preflight took 63.5s **including compilation**. Measure sustained throughput over a longer real run.
- **Meaning of the reported 63–66% falls:** only two PPO updates of a fresh learner; these are *not* evidence of a regressed trained gait. No new 20M policy exists.
- **Next executable gate:** frozen read-only `emu capability-exam` on the **already exported 131,072-step smoke actor**, quick case set with actual two-slab geometry, 4 worlds, 120–160-frame horizon on CPU; this verifies examiner/replay plumbing and physical geometry, **not skill**. Full 19-case scores and 20M gait parity require a genuinely trained policy.
- **After successful examiner plumbing:** allow the first explicitly approved source-pinned **20M cold foundation**, with an intermediate measured rollout/throughput/logging check and held-out flat-gait comparison to `main` before any six-teacher overnight continuation. Normal long unified trainer guard is still CLOSED. Do not promote the public policy.

## Path to an overnight growth run (current priority)

1. **Source gate:** latest feature-branch CI must be green (Python, physical MJX, unified gradient, browser, viewer). The numeric main-foundation test is now in `tests/test_learning.py`. Record *the exact passing SHA*.
2. **GPU capacity:** on the 8GB RTX 4060 Ti, run `python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update` from a clean checkout. A two-frame smoke is not proof of production-shaped 32×4×8 capacity.
3. **Real 360 lifecycle: COMPLETE.** Verified `first → continue → verify` at 8 and **2,048** worlds on RTX 4060 Ti. Source `6653c03`, normalizer resume, untouched ancestor and array-only browser export PASS. Next execute frozen `capability-exam` with the small untrained export **without altering its source**. See [INSTALL](docs/INSTALL.md#overnight-readiness--exact-sequence-same-source-sha).
4. **Cold 20M:** once 1–3 pass, obtain owner approval for a source-pinned fresh 360-input foundation. Compare the actual frozen 20M gait to `main` on supported flat locomotion, tracking, falls and mechanical effort. **No trained unified policy exists yet.**
5. **First overnight continuation:** only after a credible 20M walker and owner approval, create a **new run directory** resuming that exact checkpoint on the **same SHA**, with a larger absolute target. Let all six specialist frontiers develop under the one budget; monitor failures and preserved basic walking. Don't publish over Pinnacle.

**Current:** source commit `6653c03` has all four CI jobs green and both 8-world and 2,048-world physical GPU lifecycle fully passed. Next is the frozen exported 360-policy examiner, then the owner-approved cold 20M foundation. Nothing here authorises a long run.

**Protected:** `main` at `2ec9e9db`, approved 260M Pinnacle, showcase and historical source-pinned runs. No merge, policy promotion or long-run guard removal without explicit approval. Keep docs/contract and this handover concise; work in the existing core.
