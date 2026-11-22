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

\`\`\`bash
cd ~/emu
git fetch origin
git switch research/h40-ks150-safe-ascent-v2
git pull --ff-only
bash scripts/run_h40_ks150_heldout_wsl.sh plan
bash scripts/run_h40_ks150_heldout_wsl.sh preflight
bash scripts/run_h40_ks150_heldout_wsl.sh all
\`\`\`

One source/actor-authenticated WSL command performs, for each original 120M,
balanced-stairs 150M and 160M **unchanged actor**:
- 2048 first-life worlds × 600 frames, six-specialty compact pulse;
- 2048-world exact v5 signed prism matrix, 20 physical heights at 4mm spacing
  from +4…+80 and −4…−80mm, 50 independent worlds per height/direction;
- 8-world 60s dynamic standing and eight 30s signed/transition commands.

The RNG seed is **2701**, not the previous 1701. That is a NEW *evaluation*
seed, not an independent cold PPO training seed. The output root is uniquely
\`~/emu-data/exams/h40-ks150-heldout-s2701\` and the source run is read-only.
\`all\` is resumable for complete immutable exams; missing outputs are
recomputed. No checkpoint is edited and there are zero optimiser updates.

The resulting \`heldout-report.md\` and its supporting exam artifacts are
needed before any new training choice. Report mode:

\`\`\`bash
bash scripts/run_h40_ks150_heldout_wsl.sh report
\`\`\`

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

## Gate 3 — proposed V2 PPO research hypothesis (NOT YET IMPLEMENTED)

Only after the heldout physical replication:
- Fork from the immutable stage
  \`~/emu-data/runs/h40-ks150-stairs-bal-s735/stages/150m/checkpoints/final.pkl\`
  with full policy, critic, optimiser, normaliser, governor, health and field.
  Checkpoint step **150,011,904**, actor SHA256
  \`9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240\`.
  Exact parent checkpoint SHA256:
  \`c66d664d251d7ee441f42e6ca322cd3240e04f45087bad1a1543a7e2ce855d4f\`.
- Begin at **≤40mm uphill frontiers**, preserving retained easy steps
  and meaningful downhill and regular-command births. Remove any inherited
  over-high Academy/terrain frontier and reset its old near/easy evidence
  at the source-pinned fork.
- Investigate instrumenting **completed first-life episode survival/fall
  counts** stratified by actual physically measured uphill lip height;
  do NOT equate contact or crossing "success" to survival. An extended
  learner must hold the upper frontier unless both traversal AND a
  sufficiently sampled first-life survival threshold pass. Penalise/retreat
  on actual easy-band falls, not a global aggregate over easy flat births.
  Closed evidence windows and explicit minimum attempts are essential.
- Keep signed uphill and downhill goals separate and keep the 360-channel
  actor, motor limits, feedback, phase, and physical springs unchanged.
- Prefer a narrow, bounded **150→170M** warm continuation as V2 A/B,
  with an unchanged-teacher same-source 150M→170M control if attributing
  any performance change to the new curriculum. Preserve a 160M snapshot.
  No train/launch script is authorised by this document.

**Do not merge this branch, alter \`showcase/champion.emupolicy\`, or promote
a policy based on a single evaluation seed or the old screen's pass flag.**
