# EMU — current engineering handover

**Updated:** 31 October 2026
**Protected public actor:** `main` is still frozen (232-input Pinnacle).
**Active research:** v5031 same-B80-ancestor reward/control experiment;
its source-pinned training checkout, original B80 checkpoint and results
MUST NOT be modified or merged during viewer work.
**Viewer-only fix:** `fix/viewer-crest-contact-seam`, based on
`fix/viewer-park-optical-flow`. Repairs two 1-cm *actual collision gaps*
in the park's 6-cm crest (ascent→flat top→descent) without changing the
neural policy or experiment code.
**Next locomotion research:** `docs/TERRAIN_TRANSITION_GAP.md` audits
the two-planar-slab training/evaluation domain and proposes held-out
crest/valley/camber/organic transitions before modifying training.
The old planar slope and sharp-step exams remain mandatory controls.
**Fresh-PC install guide:** latest maintained version is on
`docs/v5031-fresh-install-20261031`; older experiment branches
do not contain that docs-only maintenance update.

## Historical status captured 30 October 2026 (superseded)

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
  These births request positive .06..adaptive-forward speed, with only
  +/- .05m/s lateral and +/- .25rad/s yaw so the exponential tracking
  reward is not drowned in unrelated full-power steering errors.
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

## Completed v5028, source preserved

The A/B study completed both 80M policies at frozen source
`38713791023681dc9883f6c374d92e46261b3baf`. The matched
16-world exams at 0.15m/s and 14.9° showed real 2cm supported ascent
7/8 each, 2.5cm 4/8 A versus 6/8 B; neither reached 3cm.
The B near-lip underspeed treatment improved some upsteps but
downhill/overall tracking were mixed. Both remained at a +0.369m/s
requested command frontier despite main's roughly 1m/s ambition.
See `docs/V5028_CONSIDERED_STEPS.md` and
`docs/V5028_SPEED_FRONTIER_AUDIT.md` for complete findings.

The earlier CLI omitted the coherent-careful choice, then the physical
exam rejected an exact 15° fixture from float32 roundoff. Both are fixed
on later branches; the original immutable training source is retained.
A temporary external CLI shim was used to finish the original
checkpoint without editing the running source. The reward audit
identified fast command starvation, zero actual athletic rehearsal
despite config, and the four-metre collision-slab edge.

## v5030 housekeeping and reward-lab design

Only `feature/v5030-reward-lab` is being modified while v5029 runs.
The old v5028 launcher (and its launch-only helper/test) is retired
from the latest tree; use its original pinned source SHA to reproduce.
The v5029 launcher and the shared evaluator/physics are kept. Design
and pure, inactive helper functions are in
[reward experiments](docs/V5030_REWARD_EXPERIMENTS.md).

Proposed *conditional* matched reward studies: A unchanged, B causal
25-frame (0.5s) blended forward tracking, C B + verified CoG and
support-surface height transfer from a fixed step budget, or D B +
one-shot 0.5-0.8s near-lip underspeed allowance. No new reward is
wired into PPO; choose A/B vs A/C vs A/D from final v5029 measured
speed ladder and supported up/down step evidence. No cross-SHA
continuation without explicit source and state-contract acceptance.

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

1. Let v5029 finish at its **original SHA**, without pulling this
   housekeeping branch into its running checkout. Read the actual
   `overnight_results.zip` and both independent exam JSON files.
2. Confirm actual vs commanded speed by >.3/.5/.8m/s band, fall
   locations and 700-frame speed-ladder support. Distinguish target
   frontier from proven 0.6–1.0m/s velocity.
3. Check retained 2/2.5/3/3.5cm supported uphill/downhill crossing,
   5–14.9° grade tracking, yaw, resilience and shaft J/m.
4. Select **one** small matched A/B reward comparison from v5030
   design. Integrate and test its reward-only state/physical gating
   under a *fresh pinned source*, never mutate v5029 checkpoint
   semantics silently. Use GPU only after quick CPU and 2048-world
   acceptance.
5. Only after matched actual-speed + terrain examinations, consider
   replacing protected `main` by explicit approval.

## Documents

[Architecture](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Capability map](docs/CAPABILITY_DIMENSIONS.md) ·
[Current v5028 experiment](docs/V5028_CONSIDERED_STEPS.md) ·
[Post-polish maintenance](docs/MAINTENANCE.md) ·
[Research archive](docs/history/README.md) ·
[Public release checks](docs/RELEASE_READINESS.md)


## 30 Oct — main physical baseline + controlled A/B build

A source-isolated `feature/v5030-reward-lab` now supports
`emu capability-exam --legacy-main-baseline` to run protected
`showcase/champion.emupolicy` (232 sensor actor, original tripod/flow
sensor semantics) on the SAME v5029 32m two-slab collision topology,
seed, task schedules, and scoring as the 360-input candidate.
`scripts/v5030_compare_main.py --run` generates independent 16-world,
0.1–1.0m/s speed ladder and slope/step matrix for both actors.

An exact reward-only 25-frame forward-command history and opt-in
`--temporal-vx-reward-share` implement B without touching baseline
reward arithmetic when share=0. The A/B campaign uses a fresh, common
20M foundation at one source commit then matched continuations; no
v5029 checkpoint transplant across changed source/physical state.
The C ascent credit remains an offline, bounded physics-gated prototype
pending A/B evidence. Source commits must be pinned and CI verified
before local GPU training.


### Current post-v5029 operational deliverables

- `scripts/v5030_compare_main.py` is the **read-only**
  same-physics 16-world benchmark for main (original schema-2 232
  actor) and v5029 (schema-3 360 array actor). Sensors are NOT
  up/down-converted, and the protected main showcase blob and XML
  are identical across main and the v5030 source tree.
- `scripts/v5030_temporal_reward_ab.py` defines the focused
  cold 20M foundation + two matched 20M→60M reward continuations
  (A share0, B share0.5, 25 causal frames/0.5s). The 20M starting
  checkpoint is *source-compatible* for both; no v5029 transplant.
  Physical exams are automatically scheduled at the end.
- `--temporal-vx-reward-share` is **OFF by default**.
  C verified-height and D temporary-pause rewards are still design
  hypotheses and should not be mixed into A/B.
- Do not treat the new main physical cross-schema comparison as a
  claim that the same visual information is available to both
  networks: it intentionally tests their *real* distinct sensor
  contracts on matching physical challenges.

Before using CUDA, check new branch CI, then the launcher’s CPU
and 2048-world GPU preflight. **Never pull this new source into
the archived v5029 training checkout.**


## 31 October 2026 — v5030 80M learning-horizon verdict and v5031 ready-for-preflight

- Source-pinned v5030 A/B at 80,019,456 steps each, from original \`a7be11b\`.
  Broad frozen score A 0.745292, B 0.758657 (B retains advantage,
  although both nearly plateaued at 60→80M). B is now selected as the
  **research baseline**, not promoted to main.
- Matched B80 physical facts: at 0.6m/s request actual 0.541m/s,
  15/16 survived; at 0.8m/s 10/16 survived; at 1m/s 0/16 sustained;
  12-degree uphill missions 13/16. 2cm uphill supported ascent 8/8,
  2.5cm 3/8, 3cm 0/8, 3.5cm 0/8. On 3cm all 8 survived, but
  far-side reached 0/8, mean vx ~0.010m/s: **stalling at the lip**.
  Yaw-rate stride oscillation remains diagnostic, not a suppression target.
- New independent branch \`feature/v5031-verified-ascent\` from immutable
  \`a7be11b\`. See [v5031 study](docs/V5031_VERIFIED_ASCENT.md).
  A single opt-in, reward-only scalar tracks a per-episode high-water
  mark of physically verified upper-slab contact plus torso elevation.
  Default-off is the B reward; treatment reassigns 3 of the existing 9
  ascent shaping points to transfer, preserving the +10 real supported
  completion bonus. No actor input/PPO optimizer or physics changes.
- \`scripts/v5031_verified_ascent_ab.py\` source-pins and forks both
  candidates from the SAME completed original B80 final checkpoint;
  80→100M each, seed747, 2048 envs. Mandatory CPU/JAX and both
  2048 CUDA full-update preflights; immutable new root, per-arm exams
  and intermediate report. This is an explicitly validated cross-source
  checkpoint fork, not a silent in-place resume.
- **Before GPU launch:** require green GitHub Actions on exact source SHA
  and successful launcher plan against the user's on-disk B80 checkpoint.
  **No GPU preflight or v5031 training success has been claimed yet.**
- If validated height-transfer payment is zero in rollouts or 3cm
  supported crossing does not improve without speed/heading regression,
  inspect foot-contact and near-lip traces; do not keep adding rewards
  indiscriminately. After this study, progressive 0.6→1.0m/s ramp
  assessment is the next evaluator refinement; big 200–300M cold-start
  Pinnacle waits for validated improvements.
