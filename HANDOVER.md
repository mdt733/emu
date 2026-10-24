# EMU — current handover

**Updated:** 24 October 2026 · **Work only on:** `feature/v5023-capability-growth`  
**Read the design once:** [Unified learning contract](docs/UNIFIED_CAPABILITY_GROWTH.md)  
**For old results:** Git history (the previous capability-growth notes and working `main` reference).

**Mission in one sentence:** train **one** cold-startable, six-actuator policy in **2,048 MJX worlds**, with complementary lessons for commands, recovery, real slopes, real steps, transitions and efficient walking, expanding difficulty only on physical evidence. The charter owns *why and how*; this file owns *where we are and what is next*.

## 24 Oct cleanup: fewer paths, same foundation

- Retired historical `emu ladder`, `lab5015`, `campaign5016` and upgrade runners, their exclusive tests/scripts, and the old bulky dated runbooks. Everything remains in earlier Git commits (and `main` for the old campaign).
- The single maintained launcher lives in `emu/run.py` and backs `emu train`; `foundation_config` and its 70-update steady bootstrap are retained, along with PPO, physics, reward, checkpoints and viewer/release safeguards.
- Documentation is now `README` (entry), `UNIFIED_CAPABILITY_GROWTH` (design), `HANDOVER` (status), `INSTALL` (commands), `PLAYGROUND` (viewer). Research and old campaigns are Git history, not live instructions.
- **Do not confuse successful legacy 220-input foundation with a successful new 360-input cold start.** The latter remains fail-closed until physical/GPU acceptance.
- **Next:** inspect remaining reporting/directional/field code for genuinely unused functions, then close joint physical examiner, cold-start parity and CUDA lifecycle. Do not launch a long campaign yet.

## What works in the branch

- **Training design:** six teacher-only emphases (20/20/20/20/10/10% target), gradually activated after basic steady competence; one shared 360-input actor/PPO optimizer, not six learners.
- **Actual code:** signed L1 command-sector probing with checkpointed completed-episode evidence; separate slope direction evidence; **separate step-up and step-down evidence already exists** (do not reimplement it); bounded moving-control mechanical-effort lesson and cohort telemetry.
- **Physics/sensors:** two real inclined solid contact slabs with independently angled step edge; sampled pitch/roll and rise/drop; real feet/ToF intersection. New policy uses delayed **4×4 ToF**, IMU/flow and **90×4 = 360** observations, not 8×8. Old policy contracts remain distinct.
- **CI:** at code SHA `8a9ff47`, all four jobs **passed**: Python, physical unified MJX/PPO CPU smoke, browser, fresh Ubuntu viewer. This verifies **that SHA on CPU**, not GPU capability or walking quality. Later documentation-only commits do not establish fresh CI evidence.
- **Safety:** `emu train --unified-capabilities` is intentionally **fail-closed**. No full unified GPU campaign or approved overnight launcher exists.

**Known research finding:** matched older 360-policy terrain exams showed 20M→34M improved flat survival **4/8→8/8** and descent success **5/12→9/12**, but ascent remained **0/12** at requested 0.15m/s. A faster small-step demonstration is not proof of general climbing. Evaluate **supported upper-foot contact** and keep both directions separate. Historical runs/checkpoints belong to their pinned source; never resume them blindly on this evolving branch.

## What has NOT been proved

- Unified **2,048-world RTX 4060 Ti CUDA** memory/performance, PPO update, checkpoint/resume/export and live viewer replay.
- A frozen **joint slope+step+command+recovery examiner**, tested on supported physical contact and paired task severity.
- First-20M gait parity with proven `main` or sustained efficient walking on mixed terrain.
- One truly cohesive **shared growth budget and episode-evidence mechanism** across original regimes, command/slope/steps, recovery and posture. Existing methods work in parts; some frontier growth still uses separate decisions.

## Next actions (ordered)

- [x] **Map the existing governor:** `update()` shares one growth budget across steady/transition/resilience/posture. `_advance_specialist_commands` (signed command windows), `advance_slope` (four bearings), and `advance_terrain` (up/down windows) already exist but **grow outside that budget**. Push/transition frontiers still use separate EMAs; posture is inferred from global falls. Step growth is not passed the same PPO-health veto as command/slope growth. **Do not rebuild any of these.**
- [ ] **Next: reconcile expansion decisions inside the existing governor.** Agree on one bounded growth/health arbitration across these paths without flattening physical units or conflating directional evidence. Keep local windows, prompt retreats, original steady bootstrap and checkpoint state compatibility. Then make a small code change—not a new subsystem.
- [ ] **Check foundation parity:** historical `main` reference `2ec9e9db`: 20M, seed733, 2,048 worlds, 32-frame rollout, 4×8 PPO, LR2e-4, first 70 updates steady-only, initial vx +0.02…+0.06m/s, eight core rewards. Audit the actual unified config (especially inherited mature defaults) before modifying cold-start behaviour.
- [ ] **Complete one physical examiner:** flat vs dormant-slab baseline; actual up/down/cross slopes; paired up/down steps with upper support; mixtures, disturbances, smoothness and shaft J/m. Avoid invented or pooled successes.
- [ ] **Freeze source and accept GPU lifecycle:** small CUDA PPO → 2,048-world memory/fps → checkpoint/resume → array-only policy export → viewer parity, preserving old 220/232/360 replay.
- [ ] **Only then consider one fresh 20M unified foundation**, compare with `main`, and seek approval for the larger **single-lineage** training run.

**Protected:** `main`, approved 260M Pinnacle, showcase and historical WSL runs/checkpoints. No main merge, promotion or long GPU run without explicit owner approval. Work in existing core modules; avoid new `v502x` wrappers, competing documents or large tests-for-tests' sake.

**Next concrete step:** specify the smallest shared **growth arbitration** for the existing governor methods, retaining their proven direction-specific evidence. No new learner, manager, or training run.

At the end of each session, change **status, blockers and next action here**, not the enduring design charter.
