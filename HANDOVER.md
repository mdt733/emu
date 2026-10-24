# EMU — active handover

**Branch:** `feature/v5023-capability-growth` · **Updated:** 24 October 2026  
**Design authority:** [Unified Capability Growth](docs/UNIFIED_CAPABILITY_GROWTH.md)  
**Background/results:** [Prior capability research](docs/CAPABILITY_GROWTH.md) · [Git history](https://github.com/mdt733/emu/commits/feature/v5023-capability-growth/HANDOVER.md)

This is the **short entry point for the next session**, not a research diary. Update status and the next action; keep lengthy experiment narratives in the linked research document or Git history.

## Goal and non-negotiables

Train **one cold-startable, six-actuator biped policy** with **one PPO optimizer** across **2,048 MJX worlds**. Each world has a teacher-only emphasis, eventually allocated **20% commands / 20% resilience / 20% slopes / 20% steps / 10% transitions and posture / 10% control and efficiency**. All worlds retain easier exposure to other skills. Difficulty expands or retreats on **completed-episode physical evidence**, within one shared growth budget; no separate specialist brains.

- Real signed **vx/vy/yaw mixtures** use a directional **L1 (octahedral)** competence envelope, not a rectangular command box. Success at one mixture does not unlock unrelated directions.
- Real inclined contact surfaces, genuine **step up/down with upper-foot support**, disturbances, transitions, posture and graceful *moving* efficiency all matter. Reward alone is not proof of capability.
- New policy: **360 observations (90×4)**, six actions, delayed **4×4 ToF array** plus realistic IMU/flow sensing. **Not 8×8**, and no legacy three-ToF tripod in this *new* actor. Teacher-only focus and ground truth are never policy inputs. Preserve replay of older 220/232/360 contracts.
- **Protect `main`**, its approved **260M Pinnacle / public showcase**, and historical checkpoint lineages. No merge, replacement, or long GPU launch without owner approval. Development/commits stay on this feature branch.

## Preserve the foundation that worked

Reference `main`: `2ec9e9db0a816c658131311129ef89ec7a595a12`.

The proven 20M cold start used **2,048 worlds, seed 733, 32-frame rollouts, 4 PPO epochs × 8 minibatches, LR 2e-4**, and the same eight core locomotion rewards. Its first **70 PPO updates (~4.59M steps)** were steady-only. Initial commanded forward speed was just **+0.020 to +0.060 m/s**. Thereafter, steady/transition/resilience were introduced gradually with pooled fall evidence, *faster retreat than expansion*, and a **shared total difficulty-growth budget**.

The new specialist births start **inactive** and ramp with earned steady competence (0 at steady ≤0.10; full around 0.32). Preserve gait, contacts, anti-freezing, PPO and reward dynamics before introducing hard terrain. The new 360-input network cannot reproduce the old actor bit-for-bit: **compare actual first-20M flat walking** instead of assuming the architecture is better.

## Current implementation and evidence

**Present in source:** teacher-only six-cohort assignment; gradual activation; signed L1 command probes and persisted command evidence; bounded mechanical-work/control lesson; telemetry for occupancy, command quality and work/m. **Physical mixed terrain** uses two oriented *solid slabs* with per-reset pitch/roll, step height and independently rotated lip, matching foot contacts and 16 ToF rays. New array-only sensor revision is explicit; legacy policies remain distinct. Slope exposure/frontier tracks qualified up/down/cross-left/cross-right episodes. See `emu/capability_focus.py`, `environment.py`, `curriculum.py`, `rewards.py`, `training.py`, `results.py`.

**Verified:** the code HEAD immediately before this handover edit, `8a9ff47`, passed all four GitHub CI jobs (Python, unified MJX CPU/PPO smoke, browser, fresh Ubuntu viewer). Intermediate mixed-slab CPU contact/optimizer checks also passed. **Not yet proven:** a fresh **2,048-world RTX 4060 Ti CUDA lifecycle**, sustained mixed-terrain walking, full joint-terrain examiner, new-policy live viewer parity, or long-run efficiency. The unified launcher is not ready; the trainer currently **fails closed** on a unified campaign.

**Useful old-policy reality check:** on matched 0/1/2/4cm CPU courses at **requested +0.15m/s**, previous pinned 360-input training from 20M → 34M improved flat survival **4/8 → 8/8** and downhill crossings **5/12 → 9/12**, but uphill crossings stayed **0/12**. Some faster 1cm runs had genuine supported crossings; this does **not** establish reliable climbing. Mere root progress over a lip must never count as a step-up pass. Mechanical shaft J/m is **not electrical battery energy**.

Historical 34M checkpoint: `~/robot_lab/emu_runs/v5023_window_fresh733/growth34/checkpoints/final.pkl` (source `411b841de08ebfbb65c29b357c8ee0d39d577dd1`). Leave the pinned `~/emu-v5023-window` checkout and checkpoint untouched; **do not directly resume it on evolving branch code**.

## Next work — in order

- [ ] **Close the curriculum loops:** make step-up and step-down evidence/growth direction-specific; bring resilience, transitions and posture into the same small persisted episode-evidence pattern as commands/slopes. Keep the shared difficulty budget, gentle crossover and rollback; avoid six bespoke governors.
- [ ] **Audit cold-start parity against `main`:** initial commands, rollout/PPO settings, all eight rewards, action/servo burden, pushes, posture, normalizer and first 70 updates. Fix regressions before training.
- [ ] **Build one frozen joint examiner:** flat baseline plus real up/down slopes, cross-slopes, matched-height step up/down, rotated lips, command mixtures, recovery and mechanical J/m. Require supported upper-foot contact and report attempts *and* failures separately.
- [ ] **Accept a real GPU lifecycle at one frozen source SHA:** 4/12-world CUDA PPO, 2,048-world memory/fps, checkpoint→resume→export, and live viewer agreement for the array-only actor. Keep old viewer contracts working.
- [ ] **Only then cold-start one source-pinned 20M unified run.** Compare against `main` on matched flat gait and the new physical exam. If sound, build and authorize **one** checkpointed ~10-hour continuation of the **same actor/optimizer**, not multiple A/B brains.

**Operating rule:** inspect current HEAD and CI once per meaningful change; keep commits small and code cohesive. Avoid new one-off `v502x` launchers, test systems and duplicated runbooks. Do not weaken real physics or the first-20M gait to make a report green. Record what changed, what is *actually* verified and the **next single task** here.
