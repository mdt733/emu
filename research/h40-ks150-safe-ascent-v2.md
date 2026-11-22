# H40-KS150 — Safe Ascent V2 (research, NOT a training launch)

This work is stacked on the frozen V1 result in PR #101. It does **not**
change the immutable 120M, 150M or 160M actors, their physical model or their
optimiser states. It does **not** yet implement another PPO curriculum.

## Why V2 exists

The uploaded V1 first-life signed-stair matrix (seed 1701, 50 physical
trials/height/direction) exposed a safety regression that V1's preregistered
screen did not penalise:

| Frozen model | Up 4–40mm traversal | Up 4–40mm survival | Up 44–56mm traversal | Up 44–56mm survival |
|---|---:|---:|---:|---:|
| KS150 CORE-120M | 111/500 | 490/500 | 0/200 | **196/200** |
| KS150 STAIRS-BAL-150M | **245/500** | **490/500** | 26/200 | 107/200 |
| KS150 STAIRS-BAL-160M | 139/500 | 476/500 | **29/200** | 89/200 |

The preserved 150M actor is the preferred *research starting point* for a
controlled ≤4cm negotiation skill. It is **not** a certified 55mm climber and
the 160M actor is not certified either.

Completing a strict parked global-world XY goal 60cm beyond the lip is
separate from five-frame supported **traversal**, as H40 has velocity feedback
but no global-position control. The new screening rules require *both* actual
controlled crossing **and** surviving first life, never GPS-like parking.

## Gate 1 — replicate with a genuinely fresh, held-out physical seed

Run this before ANY new PPO training:

```bash
cd ~/emu
git fetch origin
git switch research/h40-ks150-safe-ascent-v2
git pull --ff-only
bash scripts/run_h40_ks150_heldout_wsl.sh plan
bash scripts/run_h40_ks150_heldout_wsl.sh preflight
bash scripts/run_h40_ks150_heldout_wsl.sh all
```

One source/actor-authenticated WSL command performs, for each original 120M,
balanced-stairs 150M and 160M **unchanged actor**:
- 2048 first-life worlds × 600 frames, six-specialty compact pulse;
- 2048-world exact v5 signed prism matrix, 20 physical heights at 4mm spacing
  from +4…+80 and −4…−80mm, 50 independent worlds per height/direction;
- 8-world 60s dynamic standing and eight 30s signed/transition commands.

The RNG seed is **2701**, not the previous 1701. That is a NEW *evaluation*
seed, not an independent cold PPO training seed. The output root is uniquely
`~/emu-data/exams/h40-ks150-heldout-s2701` and the source run is read-only.
`all` is resumable for complete immutable exams; missing outputs are
recomputed. No checkpoint is edited and there are zero optimiser updates.

The resulting `heldout-report.md` and its supporting exam artifacts are
needed before any new training choice. The `all` mode automatically packages
**`~/emu-data/exams/h40-ks150-heldout-s2701.zip`**, including all three
full raw stair matrices, pulses, deep ZIP archives and the scorecard.
Neither old results nor a previous bundle are overwritten.

Report and manual bundle modes:

```bash
bash scripts/run_h40_ks150_heldout_wsl.sh report
bash scripts/run_h40_ks150_heldout_wsl.sh bundle
```

## Gate 2 — safe height frontier, *not* an optimistic reward score

These are engineering targets for the *future* candidate, NOT retrospective
changes to the V1 qualification screen and not statistical guarantees.

1. Hold **≥96% uphill first-life survival** over 4–40mm, provided the matched
   120M independent reference is ≈98%. General rule: no >2 percentage point
   drop versus contemporaneous 120M reference. Preserve the substantial
   five-frame controlled-traversal gain.
2. Require **≥90% uphill first-life survival** at 44–56mm before describing
   that height band as reliably climbable. Report attempted traversals
   separately; a robot that simply avoids crossing is not a climber.
3. Protect same-seed downhill crossings/survival and all six compact
   specialties; guard 30s actual vx/vy/yaw tracking, upright balance and
   motor shaft work. Global XY route endpoints remain diagnostic only.
4. Treat ≥60–80mm uphill as **held-out stretch capability**, not a
   graduation target. Never infer 8cm climbing from leading-foot contact.
5. Confirm independently again at a second physical-evaluation seed and
   eventually cold training seed before anyone calls a generalist champion.

The seed2701 scorer now **rejects** V1-style crossing improvements with
large first-life survival regressions, but it makes no automatic promotion.

## Gate 3 — source-pinned Safe Ascent V2 continuation (IMPLEMENTED, NOT RUN)

**The seed2701 replication passed:** unchanged V1 150M traversed 236/500
+4–40mm uphill with 488/500 first-life survivors, versus 120M 120/500
traversals and 485/500 survivors. But at +44–56mm the V1 150M robot
survived only 104/200 versus 196/200 for 120M. This is a documented
hazard, **not** a qualified safe ≥5cm robot.

The V2 research branch now implements a separate **150→170M** protected
continuation. It has NOT trained or been locally GPU validated:

```bash
cd ~/emu
git fetch origin
git switch research/h40-ks150-safe-ascent-v2
git pull --ff-only
bash scripts/run_h40_safe_ascent_v2_wsl.sh plan
bash scripts/run_h40_safe_ascent_v2_wsl.sh preflight
bash scripts/run_h40_safe_ascent_v2_wsl.sh gpu-smoke
# Only after those gates pass:
bash scripts/run_h40_safe_ascent_v2_wsl.sh all
```

### Training treatment

- Parent: immutable full actor+critic+optimizer+normalizer+governor+field+
  health checkpoint `stages/150m/checkpoints/final.pkl`, step 150011904,
  actor SHA256 `9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240`,
  pickle SHA256 `c66d664d251d7ee441f42e6ca322cd3240e04f45087bad1a1543a7e2ce855d4f`,
  parent training SHA `4504ac2c7219e5df8a106a224b77825fee219ead`.
- Research child `~/emu-data/runs/h40-ks150-safe-ascent-s735`
  restores this exact parent with explicit cross-source authorization.
  **One continuous PPO process** to 170M, 2048 GPU worlds, seed 735,
  unchanged 360-channel actor and servo/physical model; permanent 160M
  full-state milestone saved while learning, final at 170M.
- At the *one-time external fork*, reset only stale stair
  evidence; clamp Academy **uphill** and ordinary (mixed-up/down) step
  frontiers to **40mm**. Preserve separate Academy **downhill** up to its
  already-earned height, capped at 55mm.
- Independent and persistent **24-completed-first-life-episode**
  uphill birth windows at each *actual first physical lip height relative
  to its own birth frontier*. For a promotion, require both ≥55% real
  near-height supported crossings and ≥90% completed-first-life survival;
  fewer than 24 survival trials means HOLD. Retreat on sustained near
  survival below 85% or easy-band survival below 94%. Academy DOWN still
  learns separately. These are preliminary research thresholds.
- The ordinary STEP classroom cannot silently bypass the safety mechanism
  by expanding its *shared signed* height beyond 40mm: the ordinary
  governor receives the protected 40mm cap throughout V2.
- V2 survival state is saved in the same full PPO checkpoint; the
  checkpoint schema migrates from the exact V1 150M governor only on
  an authorised external fork. Same-run resume preserves the window,
  never regenerates it. Physical metric `safe_ascent_*` records full
  episode outcomes, never per-frame survival or a rewarded foothold.
- PPO loss, optimiser, action space, physics, solver, struts, sensor
  contract, locomotion and existing step rewards remain unchanged from
  the original Balanced Stairs run. Only uphill teacher difficulty
  progression and its host-side evidence change.

### After training

Without interrupting PPO, the driver then freezes and independently
re-examines the 150M original, new 160M and 170M actors, all on
**seed3701**, with one same-source 2048-world signed stair matrix,
six-specialty pulse and 60s/30s dynamic exam per actor. The V2 scorecard
requires, relative to the same-seed 150M parent: ≤2pp loss of 4–40mm uphill
survival, ≤5pp loss of controlled 4–40mm uphill crossing and downhill
4–80mm survival/crossing, ≥90% first-life survival AND ≥15% supported
crossing at 44–56mm, and locomotion/recovery retention. A robot that avoids
the hard climb cannot pass merely by surviving. No global-XY parking gate.

The launcher builds an immutable uploadable evidence bundle at
`~/emu-data/runs/h40-ks150-safe-ascent-s735/v2-evidence-s3701.zip`.
No automatic promotion.

### Constraints

This is **one warm-start training seed**, not causal evidence of V2 versus
the unchanged V1 teacher. A same-source, same-parent 150→170M control
would be needed before attributing any observed benefit to the survival
gate; another independent cold seed is needed for reproducibility.
Only the owner can initiate costly training with the explicit `all` mode.
Do not merge either research PR or replace the showcase champion yet.
