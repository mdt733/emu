# EMU — current engineering handover

**Updated:** 30 October 2026  
**Protected published reference:** `main` — 232-input Pinnacle, approved browser policy  
**Completed 80M A/B:** `feature/v5028-considered-steps` at exact SHA
`38713791023681dc9883f6c374d92e46261b3baf` (frozen, do not modify)
**Current fresh next run:** `feature/v5029-speed-restoration` (new 120M
cold start, seed737; do not run before exact-source CUDA preflight).
**Old cleanup/integration branch:** `feature/v5028-merge-polish`
**Golden rule:** **do not pull/change the running checkout; do not merge or
replace the public champion without explicit owner approval.**

This is the concise read-first handover. The full pre-polish chronology
remains in [historical handover](docs/history/HANDOVER_PRE_POLISH.md).

## The mission

Build a **single 360-input, six-servo, dynamically unstable walking policy**
that eventually **outperforms protected `main` in two independent ways**:

1. Regain the incumbent's roughly **1 m/s requested forward-command envelope**
   with good **measured** velocity tracking, genuine dynamic gait, controlled
   lateral/yaw commands, low falls, recovery and useful J/m.
2. Improve supported real-terrain competence: 2–4cm-plus **up/down**
   discontinuities, multi-direction grades, compound terrain, post-cross
   recovery and eventual ToF-aware conservative route selection.

Neither climbing slowly forever nor displaying a teacher's high target
counts as success. Fast actual walking **and** careful climbing are the
joint graduation criteria. Commands/terrain should grow based on physical
evidence, not hard-coded maturity age or inflated frontiers.

## v5029 speed-restoration mission (30 Oct 2026)

**Root cause, source audited:** At v5028 80M the command frontier only
grew from 0.06 to ~0.369m/s. Six active specialist roles rewrote virtually
all birth commands to low speeds. Configured mature 10% athletic rehearsal
was *zero* in the rollout because resuming the 20M foundation restored
the zero-valued curriculum state. Ordinary locomotion reward coefficients
were unchanged from protected `main`; the new per-specialist proxy
cost is small. Flat v5028 collision support was only 4m from centre,
so 1m/s for 14s would mean falling off the floor.

**v5029 code changes on a separate new branch**:
- Two actual convex collision slabs remain, now 32m half-length and
  16m half-width. The buried platform is identified from its **full
  known buried pose**, not its centre height (sloped box centres can
  be lower than -1m once their boxes become long).
- 20M first-stage foundation keeps original main-style births.
  Afterward `--athletic-rehearsal-share .25` gives 25% real flat,
  STEADY, unattenuated fast-gait births, chosen before six specialist
  assignments. All worlds still share *one actor/optimizer*.
  The phase/gait clock is recomputed for true rehearsal commands.
  Fast rehearsal targets start up to .6m/s and can extend
  `1.15*earned_forward_frontier`; no permanent .6 hard stop.
- Explicit run-config athletic share wins over a restored ancestor
  controller's zero when constructing new task packets.
- Telemetry reports the fraction requested at >.3/.5/.8m/s and
  measured velocity and tracking error in each band, besides frontier
  value. Independent read-only speed ladder covers .1–1m/s.
- One `scripts/v5029_speed_restoration.py` launcher,
  **120M steps fresh**, seed737, 2048 CUDA worlds, 20M cold nursery,
  six balanced focus roles, considered STEP and gentle push programme,
  with source and dirty-checkout guards, CPU smoke, actual 2048 CUDA
  PPO full-update preflight, SHA-matched resume and read-only speed/
  slope-step exams; bundle remains under a NEW run root.

**Launch after green GitHub Actions and first-time local GPU preflight:**
`cd ~/emu-v5029 && source ~/emu-graduation/.venv/bin/activate &&
python -u scripts/v5029_speed_restoration.py --run`.

No claim that speed improved yet; v5029 needs actual measured commands,
supported long-floor motion, real step retention and broader fixed exams.

## What actually ran / what is known

- v5027 A/B/C were a shared source-frozen seed735, 20M nursery plus
  three 20→60M forks (140M aggregate); archived training source
  `5dec60c4`. Their **independent 16-world per physical condition**
  extended exam (0.15m/s, 350 frames) is documented in
  [v5027 extended findings](docs/history/V5027_EXTENDED_PHYSICAL_FINDINGS.md).
- A supported climbed **2.0cm uphill 8/8; 2.5cm 6/8; 3.0cm 0/8;
  3.5cm 0/8**. Up/down outcomes must not be mixed. Surviving/stalling
  in front of 3cm is *not* evidence of a learned safe refusal.
- At `cmd_vx=+0.15m/s`, stopping briefly forfeits much of the historical
  dense locomotion reward. The one-time +10 crossing/+9 progress credit
  plausibly has too little incentive to teach difficult deliberate
  footholds. This was a **mechanistic hypothesis**, not causal proof.
- v5028 adds a carefully bounded **STEP-UP near-lip** underspeed
  exception (90% relief, no more than 120 eligible 20ms frames/episode),
  keeping overspeed, sideways/yaw, posture, slip, joints and falls.
  Supported landing contact may momentarily settle; flat, downhill,
  slopes, and other roles retain standard tracking economics.
  No new actor input, no hard-coded servo climbing manoeuvre.

## Historical v5028 training experiment (complete; preserved)

The owner reported **the launcher started running** on the morning of
29 October. The original `~/emu-graduation/.venv` lacked `pytest`;
using its direct Python reward-function test avoided that ancillary
dependency. User has **not yet shared GPU preflight success or metrics**;
do not claim CUDA validation or learned improvements.

- Checkout: `~/emu-v5028`; **leave it untouched while running**.
- Exact source: `38713791023681dc9883f6c374d92e46261b3baf`.
- Launch: `python -u scripts/v5028_considered_steps.py --run`
  in foreground (no tmux). The source checker refuses dirty/mismatched
  checkouts; launcher does CPU + 2048-world CUDA PPO preflight.
- Design: a **fresh seed735 shared 20M foundation**, original coherent
  A 20→80M, considered-step B 20→80M; **140M aggregate world steps**.
  Same 360 policy shape, same gentle commands, other specialist lessons,
  PPO and real collision physics.
- Run output: `~/robot_lab/emu_runs/v5028_considered_steps_s735/`,
  `overnight_results.zip`, metrics, fixed exams, physical policies,
  independent per-world JSON and console logs.
- **Original pinned launcher passes `--worlds 8` to each final terrain
  condition** (four UP and four DOWN if paired). A previous summary
  mistakenly promised 16. The **polish branch** corrected its future
  launcher to 16, but the running source remains at eight. Plan a
  separate read-only **16–32-world** re-exam afterwards; do not
  touch running source to fix presentation.
- Exact-source interrupted runs may resume their own compatible
  checkpoints. **Never resume archived v5027 weights into changed source**.

## 30 Oct physical examiner boundary — corrected diagnosis

Both source-pinned 80M checkpoints completed. The A 8-world frozen exam
failed upon reaching 15° because the environment enforces
`np.rad2deg(np.arctan(g)) > 15.0` after **float32** quantisation.
One legitimate 15° axis yields approximately **15.000001** and triggers a
false over-limit exception. The checkout/actor is not faulty.

The independent polish branch fixes only the numerical guard using a
1e-5° tolerance, and adds a no-MJX regression test including exact
uphill/downhill/cross-slope 15° and real >15° rejection. Running v5028
SHA and checkpoints stay untouched. To recover on the original exact source,
run an explicit 14.9° frozen A/B matrix in fresh separate directories, label
the results truthfully and collect them in a standalone comparison bundle.
Do not ask the original hard-coded 15° campaign launcher to rerun until the
source-bound 15° guard can be corrected under a new experiment contract.

## 29 Oct campaign incident — corrected diagnosis

The original SHA `38713791` successfully completed the matched A policy at
80,019,456 steps (frozen broad 0.759, core 1.0). B did **not** start PPO:
the source-pinned CLI `emu/cli.py` accidentally omitted
`coherent-careful` from `--terrain-lesson` options, while environment
and run config already supported the mode. This is a real **CLI parser defect**
in the original committed SHA, **not** an old installed package.

**Do not amend original branch/working tree:** source-pinned parent and A
checkpoint require that exact SHA. For recovery run the original unchanged
campaign using a strictly targeted, external `sitecustomize.py` argparse
compatibility shim. Verify `--help` exposes the one additional permitted
choice; then `--run --skip-smoke` reuses preflight, 20M foundation and
completed 80M A; starts only B and later frozen exams. No reward or actor
source changes; the shim modifies argparse choices only, in memory.

Permanent one-line CLI fix and regression test live **on the independent
polish branch only** and must not be pulled into the current experiment.

## 29 Oct maintenance — done on separate polish branch only

- [x] Inventory 105 Git tree entries, including 20 Markdown documents
  and four retired versioned experiment launchers.
- [x] Move completed/superseded v5024–v5027 work under
  [docs/history](docs/history/README.md), preserving findings.
- [x] Replace outdated 693-line v5024-centred README with a current,
  external-reader-friendly project overview.
- [x] Fold the sole `velocity_i_step` command controller into `emu/environment.py` with the same function body, retire `emu/velocity_trim.py`, and keep the single focused controller test. No PPO, reward or actor change intended.
- [x] Group the overly granular evaluation_* files into two clear entrypoints, `emu/evaluation.py` and `emu/physical_exam.py`; return the tiny training-health component to the governor. Kept public imports unchanged. Structure intentionally follows actual robot/training/evaluation responsibilities, **not file-length targets**. Pending full CI acceptance.
- [x] Extract the former 1,827-line evaluator into two meaningful owners: fixed/PPO evaluation and atlas in `emu/evaluation.py`, with headed motion and unified 360 physical exams in `emu/physical_exam.py`. Existing public CLI imports re-exported, numeric scoring unchanged. `tests/test_evaluation_split.py` checks identity and stable cases. **Pending CI confirmation.**
- [x] Retire v5026-only `scripts/rescore_unified_exam.py` into `docs/history/`; leave current release build/review scripts intact.
- [x] Enable GitHub Actions checks for `feature/v5028-merge-polish` and cover extracted files in compile checks.
- [x] Move/retain the PPO health/rollback gates with curriculum governance (same serialized state-dict); the brief standalone module was folded back after reviewing file layout.
- [x] Split the former 2,647-line curriculum monolith without changing the governing equations: `emu/curriculum.py` now owns regime governance and training health; `emu/capability_field.py` owns event evidence; `emu/capability_cloud.py` owns read-only held-out probes. `emu.curriculum` retains historical imports through re-exports. **Not yet CPU/JAX validated.**
- [x] Refactor live campaign helper dependency away from old
  `v5024_campaign.py` to `scripts/campaign_support.py`.
- [x] Retire four obsolete v5024/v5025/v5026/v5027 launchers *on
  polish branch only*. Old source SHAs continue to preserve reproducibility.
  Retire associated historical launcher-only tests and add a current
  v5028 campaign-plan contract. Update workflow compile targets.
- [x] Add [maintenance rules](docs/MAINTENANCE.md) and
  [release readiness gates](docs/RELEASE_READINESS.md).
- [ ] **Validation still required** on an actual checkout/CI: syntax,
  selected learning contracts, browser build, exact-source experiments.
  The GitHub connector can edit/inspect source but is not the owner's
  CUDA device; no GPU/test success is claimed here.
- [ ] **Do not merge/cherry-pick this branch into active v5028 while it
  runs.** Post-run code integration creates a new source SHA.

## Next actions in order

1. Finish or collect a recoverable v5028 run without modifying its
   working tree. Obtain the original `overnight_results.zip`.
2. Compare A/B **directionally supported ascents** and 3cm frontier,
   approach time, after-cross velocity/heading, survival, push recovery
   and shaft-work. Inspect 1m/s capability separately; training teacher
   envelopes are not actual speed certification.
3. Run source-pinned, independent broader physical exams; ensure
   the cleaned runner's new 16-world setting does not mislabel
   historical eight-world outputs.
4. Evaluate ToF detection distance, actual braking/hold ability and
   credible **refusal versus stalling**. The route supervisor described
   in [TRAVERSABILITY_REFUSAL](docs/TRAVERSABILITY_REFUSAL.md) is design
   only; stopping may need stepping in place on this unstable robot.
5. Maintain parallel capability/retention experiments, choose a
   reproducible winner using a **common 232-vs-360 physical exam**,
   then only with owner approval merge a tidy candidate into `main`.

## Documents

[Architecture](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Capability map](docs/CAPABILITY_DIMENSIONS.md) ·
[Current v5028 experiment](docs/V5028_CONSIDERED_STEPS.md) ·
[Post-polish maintenance](docs/MAINTENANCE.md) ·
[Research archive](docs/history/README.md) ·
[Public release checks](docs/RELEASE_READINESS.md)
