# v5041 — 140M terrain-exposure postmortem (3 November 2026)

## Source and observed facts

Research parent: `feature/v5041-unified-terrain-carpet`.
Uploaded `results_bundle(20261103-003923).zip` is a **140,050,432**-step continuation
of the original cold-start seed 735 actor, with training source SHA
`601da9d22f6b20f60ecba9f9696116b92c3783ad`, parent checkpoint
at 10,027,008 steps (`9070238b3c81a6805296f867a7849e8799f4a2ff`).

* Final held-out fixed-evaluation broad score 0.7858, core survival 1.0.
  Neither is an independent 10–12 degree slope exam.
* Governor remained at slope target **0.7 degrees** and exposure **0.15**
  for its entire post-foundation history; measured maximum **0.0935 degrees**.
  No completed up/down/cross-slope episode satisfied its evidence bands.
* Ordinary step frontier stayed **4mm** and exposure **0.15**; Academy
  directional frontiers stayed **3mm**. All 199 sampled metric rows
  show zero terrain up/down attempts, zero Academy lip credit and zero
  supported foothold/transfer milestones, even while Academy occupied
  about two thirds of the last rollout's frames.
* Command tracking and flat survival improved, but this run did **not**
  establish success or failure on real 10+ degree grades.

## Mechanism and code-level fix

`terrain_carpet_mjx.role_envelope` mistakenly multiplied slope angle and
step height by their **exposure fractions**. Historical two-slab code
uses those parameters as the probability that an episode receives a
terrain challenge, *not* obstacle-size multipliers. At 0.15 this
produced ~0.105-degree slopes and 0.45mm Academy lips, below the
teacher's 0.10-degree slope recognition / 0.78x near-grade band and
1.5mm signed-lip threshold respectively. A further 0.5x profile
reduction around the start cell made early lips even less detectable.

Fix on `feature/v5041-carpet-exposure-unlock`:
1. Salted, deterministic per-birth Bernoulli gates now interpret ordinary
   slope/step exposures as *probabilities*. Selected worlds get
   full specialist-scaled physical severity; worlds not selected stay
   flat for that dimension.
2. Dedicated Academy births always get their earned signed step
   frontiers, independent of ordinary STEP exposure.
3. X-profile starts at 0.95x of the 3mm seed rather than 0.5x, still
   grows smoothly to 1.5x by tile 14, and retains independent up/down
   signs, deterministic global indices, and visit anti-rocking guard.
4. The viewer's representative *exposed* solid carpet now mirrors the
   corrected physical amplitude and profile; exposure rates remain
   available as probability labels. This viewer is not a replay of
   a particular CUDA training world.

No actor dimensions, PPO weights, other regime controller, collision
shapes, sidewall tests, or model checkpoint bytes are changed.

## Historical verification plan, for the original v5041 source only

The following command names and tests belong to the **3 November v5041
source**, and were deleted from the active v5045 prism branch. Do not
paste them into a current checkout. For current physical acceptance use
the [v5045 prism runbook](V5045_LEAN_PRISM_RUNBOOK.md).

* Run `python -m pytest -q tests/test_terrain_carpet.py
  tests/test_carpet_solid_geometry.py tests/test_carpet_visit_rewards.py`
  and `npm test --prefix emu/viewer_src`.
* Run the bounded 128-world CPU/CUDA birth/geometry smoke then a
  **2048-world CUDA PPO smoke** on the committed SHA. In particular,
  check the 3mm early signed lips are physically present, actual
  maximum slope is in the requested range, and slope/step cohort
  attempts become **nonzero** after suitable settled episodes.
* Run a **short, separately named, explicit SHA-256-verified research
  fork** of the original 140M checkpoint, or a fresh cold-start smoke.
  The research fork is a new source lineage, never `--resume auto`
  into the original run directory. See the explicit
  `--research-parent-source` and `--research-parent-sha256` CLI
  provenance requirements.
* Compare first complete slope/step exposure cohorts against the
  140M parent. Neither synthetic tests nor the fixed-eval broad
  score prove real 10-degree walking or 3mm uphill contact mastery.
* Only after real evidence arrives should the slope teacher advance
  its 0.7-degree frontier toward 10–12 degrees. Do not hard-code 10
  degrees immediately, change `main`, or rewrite old metrics.

## Secondary evidence audit (not fixed by severity correction)

The ordinary STEP teacher in `environment._step_impl` currently
defines `terrain_active` using `academy['height_m']`. In carpet mode
`_carpet_events()` sets that height only while the robot is moving
into a new, not-yet-confirmed tile. The legacy
`terrain_episode_attempt` and `terrain_episode_{near,easy}_attempt`
then require `done & terrain_active`. Unless episode termination
coincides with those entry frames, *real encounters can disappear
from the episode-end curriculum denominator*. This is a distinct,
source-observed **risk of undercounting** after geometry is unlocked,
not proof that actual crossings occurred in the original 140M run.

Also, `academy_lip_attempt_new` reports failed approaches at terminal
frames, while the near/easy classifier uses this frame's signed
`academy['height_m']`; the independent read-only exam already has
special logic to recover the *last encountered lip height* for this
reason. We must verify live teacher attribution, not just physical
collision presence. A later fix should preserve per-encounter signed
UP/DOWN difficulty across rollouts, apply one verdict per unique
approach, and test against independent raw-contact traces. Do not
prematurely loosen top-contact or anti-rocking safeguards.

**Known remaining question:** Does the five-frame physical far-side
contact gate actually accumulate credited up/down crossings after the
birth-severity fix? This needs a frozen real-policy MJX exam and
step-by-step physical trace; don't assume positive encounters imply
crossings. Note the current `scripts/exam_step_course.py --carpet`
forces STEP focus: its effective slope is HALF the global slope target.
Do not present `--slope-deg 10` in that exam as proof of a **physical
10° SLOPE-specialist** challenge; use a separately validated full-grade
SLOPE-focus experiment.
