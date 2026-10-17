# EMU — v5016 sim-to-real living handover

**Authoritative baseline:** GitHub `main` at `77c468c5bd78eedfaedaa755937616fe5743cfe8` (verified 17 Oct 2026).
**Development branch:** [`feature/v5016-sim2real`](https://github.com/mdt733/emu/tree/feature/v5016-sim2real).
**CI-only draft review:** [PR #8](https://github.com/mdt733/emu/pull/8), **not for merge**.
The approved Pitch15 120,061,952-step showcase remains protected by Git blob
`067d6156f0f9de629eabb1a69cf1b0ceb86db24c`. Actor contract: 55 × 4 = 220
observations, six actions; checkpoint schema 7. No changes to `main`, the
published demo or any historical training directory without express approval.

## Physical-reality programme: execution gates, not a blind training schedule

**Pass A — measurement correctness (IN PROGRESS).** Use the *existing*
`evaluation.py`, `directional.py`, `results.py` and `ppo.py` tracer.
Verify world-fixed signed distance, path error, complete alternating contact
touchdowns, contact-conditioned foot slip, startup/braking/reversal response,
yaw/heading error, falls and matched seed reproducibility. Overlay per-joint
actual MJX shaft force and joint velocity, peak/p95 torque, nominal
0.638 N m duty, absolute/positive/braking shaft work, squared-torque duty
proxy, and stored spring energy. Keep reward, broad held-out score and
physical evidence in distinct tables. Audit frame alignment and early
termination with real MJX CPU traces **before** asserting physical results.
Synthetic unit tests alone are not acceptance. Nominal duty is not a
validated safe current/heat envelope; spring storage is not recovered energy.

**Pass B — frozen baseline exam.**
A 10-course protected-export plan is available through
`emu directional-batch --policy showcase/champion.emupolicy --profile physical
--device cuda --worlds 4 --seed 1701 --output "$NEW_EXAM_ROOT"`.
It **prints only** unless `--run` is supplied; separate results roots are
required for independent seeds 1702/1703 and new source revisions.
The legacy four-arm tournament profiles retain `--root` instead of `--policy`.
 First run CPU shape and determinism
checks (2 worlds, short horizon). On WSL CUDA, expand to bounded
4–8-world, 350-frame exams on flat ground with precisely fixed seeds;
collect steady forward/backward, sidestep both directions, rest→walk,
walk→stop→resume, reversals and turns, both controller OFF/ON.
Then add held-out sensor perturbations and gravity-equivalent slope trials.
A true ramp geometry is a separate later exam. Repeat seeds 1701, 1702,
1703 rather than promoting one lucky run. Preserve raw world-level results,
report per-world missingness, and do not collapse into one composite score.
The approved `showcase/champion.emupolicy` can now enter the existing
directional exam directly with `--policy`: the reader verifies its actor,
normalizer, 55-channel order, delivery timing and exact `robot.xml` identity
before running frozen JAX inference through the same MJX course trace.
This is an actor-only baseline and cannot rewind PPO training. The selected
historical full PKL checkpoint SHA remains
`ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`;
if later available, compare exported actor against it, then retain the PKL
outside Git. No run restarts or checkpoint changes are necessary.

**v5016 pitch doctrine (17 Oct 2026):** The Pitch15 tournament win did not isolate
whether pitch was the useful improvement or simply a less stressful workload;
no claim of trivial learning or confirmed physical headroom follows from that
ranking. The new opt-in `emu train --profile sim2real` resolves a shared
±30° chassis pitch-command envelope for future training (v5015 task
sampler/transition reward and unchanged 220-input actor). This is a COMMON
teaching setting, not a fifth arm or a repeat of Pitch15-vs-Pitch30. The
historical `emu ladder` / `lab5015` recipe remains locked to its original
four study arms for research reproducibility; it is **not** the v5016
training tournament. Actual ±30° commanded motion still needs held-out
posture/locomotion and servo-load examination before hardware assumptions.

**v5016 motor-guidance experiment (opt-in, untrained):** A bounded
`motor_guidance_weight` 0..0.10 is now a logged ninth teaching term.
Its per-frame magnitude is `weight*(0.65*clip(over-continuous fraction,0,1)
+0.35*clip(absolute simulated actuator shaft watts/20,0,1))`, subtracted
from PPO reward. Default weight zero retains the historical incentives and
physics. It is not current/thermal data; a future matched experiment can
compare zero against e.g. 0.06 with identical pitch30, physics and evaluator.
Do not teach spring *motion* as a goal: measure energy cycling and useful
physical progress first. `--plan` verifies parameters without training.

**Pass B2 — passive springs and actual contact geometry (IMPLEMENTED, CPU acceptance pending).**
The frozen MJX trace now records eight separate spring angles with the XML's
actual reference angles and stiffnesses. From consecutive pre-reset poses it
measures elastic charging, releasing, right/left passive-shin release during
sustained stance contact, and non-stance release. This establishes neither
beneficial return nor physical damping/heat; reward currently has NO spring
bonus. Keep signed travel, actuator work, gait and contact alongside the
spring evidence before forming a hypothesis.

Read-only `emu directional-exam --policy showcase/champion.emupolicy
--angle 2 --contact-slope --direction uphill --device cuda ...` now rotates
the actual floor collision plane once, BEFORE compiling the MJX model, while
gravity stays (0,0,-9.81). The report records `ground_mode`,
`actual_gravity_world`, and the grade reference used for course scoring.
The 13-case protected-policy physical batch now includes three real-contact
2° slope cases (up/down/cross) alongside the 10 flat cases. This is an
**opt-in test world**, not per-world rough terrain randomization, and not a
claim that the robot can ascend hills. MuJoCo vs MJX contact-plane normals
and the actual frozen actor are exercised by new CPU integration tests.
A follow-up may add per-world physical slopes/heightfields only if the
CPU/GPU contact proof is sound and reset heights remain safe. A rotated
infinite plane is a real slope but not uneven terrain.

**Pass C — actuator model sensitivity.** Inspect the existing
±1.913 N m forcerange, damping-based torque-speed approximation, friction,
spring terms and one-frame action delivery in the actual XML/MJX path.
DO NOT apply a second torque-speed cap. Define a short measured/justified
uncertainty table for torque-speed under battery voltage, tracking delay,
gear backlash/stiction, motor torque-duty and sensor delay. Model only
identifiable quantities, separately log nominal vs perturbed performance,
and label any temperature proxy as uncalibrated. Add physical collision
sentinels and actual contact slope/terrain only after a geometry parity check.

**Pass D — adaptive teaching.** Keep dynamic locomotion, not static pose
holding, as the learning objective. Grow acquisition, acceleration, braking,
reversing, yaw changes and push recovery according to observed course/
contact/motor feasibility, *not* reward alone. Use bounded per-world
variation and avoid changing sensor/actuator hardware assumptions inside
one comparison. Explicitly test whether the 55 observable channels could
be supplied/estimated on a real robot.

**Pass E — controlled training tournament (NOT STARTED).** First complete
a fresh small CUDA two-update save→resume→fixed-eval→export lifecycle.
Default to 512 worlds for resource acceptance, then assess 1024; only
use 2048 if confirmed within the RTX 4060 Ti 8 GB limit. Reserve a
new per-source-revision run root outside Git.

Proposed first exploration budget, in total *environment transitions*:
- 12M shared foundation at one seed, trained from cold start.
- Four matched hypotheses continued 12M each: A hardware-aware
  nominal control; B motion acquisition/transitions; C torque/power
  guidance; D gentle contact/robustness (only once terrain is validated).
- Two physically credible finalists gain 20M each; one finalist gains
  a further 20M only if the independent examiner still improves.
- Total budget: 12 + 4×12 + 2×20 + 20 = **120M aggregate steps**,
  not one 120M unexamined trajectory. This is a staged *budget ceiling*,
  not an instruction to start jobs. Earlier stop conditions are desirable.
- After pilot evidence, rerun the winning teaching difference under at
  least two further seeds with matched setups; avoid a single-seed
  causal conclusion.

At every checkpoint freeze policy and matched evaluator. A candidate
must (i) produce positive signed progress, (ii) exhibit alternating
contact-stride evidence, (iii) preserve survival/recovery and
command tracking under held-out tasks, and (iv) improve or at minimum
not hide worsening actuator load/shaft work. Publish the individual
Pareto dimensions and worst-world distributions, not a hidden scalar
promotion score. Use identical environment model and evaluation seeds
for each controlled contrast.

**Pass F — real hardware identification and safety (PLANNED).** Map all
55 observation channels to real sensors/estimators, identify servo
torque-speed and delay on a guarded bench, measure voltage/current/
temperature and tracking latency, then calibrate uncertainty ranges.
Watchdogs, joint travel/torque limiting and fall-safe shutoff live
outside the neural controller.

## Validation responsibilities

GitHub PR CI verifies imports, unit tests, browser tests and CPU-level
source contracts; it does **not** establish hardware or CUDA/MJX training
validity. WSL checks must demonstrate compiled physics, full lifecycle,
8 GB allocation margin, and directional physical evidence. Keep a
different run root for each source revision, never resume an older
experiment into new source, and do not disturb any separate active
120M WSL campaign. Every release gate records: hypothesis, source SHA,
policy SHA, seeds, physical parameters, measured outcomes, known
limits and next decision. Promote via reviewed PR only after explicit approval.

---

## Historical v5015/refactor handover (preserved for context)

**Current stable release:** `main` at merge commit `252f6102dd813a451daad29fecf2fbebd586da6b`, containing the refactored trainer, single browser viewer, and real **Pitch15 120,061,952-step** `showcase/champion.emupolicy`. The public GitHub Pages site deploys only after manually dispatching the Pages workflow; do not confuse a code merge with a published site.

**Current refactor:** `refactor/emu-minimal`. Target is roughly fifteen cohesive Python modules with obvious ownership, not a fixed line-count reduction. Large files are acceptable when they tell a coherent story. All development occurs on this branch; never edit or merge `main` without explicit user approval. Keep the working original available in Git history; do not rewrite `main` or active WSL run roots before CI and local acceptance.

## The essential ideas

Robot and universe → rewards → PPO/training → curriculum / capability evidence → branch experiment design → independent results → frozen browser viewer. Every retained file needs an understandable purpose. No one-off pass documents, old CSV result bundles or import-only compatibility stubs.

## Proven experiment and lessons

User's study from source commit `a5996b8cda2c394c7c65e3e938282a3808ded121`, seed 733, 2048 MJX worlds: shared 20M intro, four independent 80M arms, two 100M finalists and Pitch15 120M winner, total 320M aggregate environment steps.

Pitch15 matched frozen-screen scores: .683903 at 80M; .712062 at 100M; .723497 at 120M (91.992% hard survival, 86.328% growth survival). Shadow second at earlier rounds; slope practice did not dominate the common gravity-equivalent slope exam. **These rankings do not prove actual hardware walking, and one seed isn't a universal result.** Late Pitch15 servo over-rated-torque sample fraction ~74.12%, aggregate simulated shaft |power| ~12.81 W. Real hardware motor limits/efficiency and signed footfall progress remain unresolved.

The selected raw PKL checkpoint (local-only) SHA-256:
`ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`.
Trained browser file tracked at `showcase/champion.emupolicy`, Git blob SHA
`067d6156f0f9de629eabb1a69cf1b0ceb86db24c`, policy actor SHA-256
`c836f0f179445c309aea3b9c87eb6b63c75a7cd2666801e262087b7663d5849c`.
Preserve exact 220-input/six-action contract and checkpoint schema 7.

## Operational safety

- Keep large checkpoint roots outside Git; preserve source commit and normaliser/optimiser lineage. Never open an active trainer's branch for destructive experiments.
- Changes to MJX environment, PPO, observation ordering, checkpoint storage and exporter need real CUDA save/resume/export acceptance, not just unit tests.
- The static browser uses `emu/viewer_src/`; local live watcher uses validated read-only checkpoints. The GitHub Pages champion is a frozen exported brain, never the original `.pkl`.
- Pages deployment is intentionally manual and pins the exact champion Git blob SHA.

## Minimalisation status and honest scope (17 October 2026)

**PR #6:** [refactor/emu-minimal](https://github.com/mdt733/emu/pull/6), branched from the working 120M `main` release. **Do not merge yet.** The independent GitHub Pages champion Git blob remains `067d6156f0f9de629eabb1a69cf1b0ceb86db24c`, byte-for-byte the same as `main`.

Tracked inventory: **132 → 56 files** (after empty media removal), `emu/` **42 → 15 Python files** (including `__init__.py`), `docs/` **30 → 2 Markdown files**, five maintained setup/check/showcase scripts, four test files, 20 browser source/dependency assets, one protected champion. All previous pass notes and tracked legacy result bundles can be found in Git history.

**Crucial honesty:** Python source volume only changed **609,546 → about 599,518 bytes**. Most gains are fewer module boundaries and discoverability, **NOT** a 70% reduction in logic. To shrink further, audit live usage and eliminate or simplify real training/exam features after careful CUDA acceptance, rather than coalescing into unreadable multi-thousand-line modules.

### Current code ownership (no import-only alias files)

- `robot.xml` + `environment.py`: physical universe, task commands, per-world domain variation, 220-dim actor observation.
- `rewards.py`: eight training payments, with servo power diagnostic rather than a hidden PPO penalty.
- `ppo.py`: PPO actor/value optimisation, JAX backend and performance timing.
- `training.py`: one learning loop, rolling measurements, explicit checkpoint calls.
- `curriculum.py`: regime governor, field teacher/evidence, held-out cloud probes, training health.
- `branch_lab.py`: readable 20/80/100/120M hypothesis recipe, single-run and tournament/expt executor.
- `evaluation.py`: frozen evaluation, fair matched branch screens, capability atlas, signed course/contact/gait and multi-branch batch plans.
- `results.py`: rollout scoring metrics, CSV/log artefacts, training stories and optional posture drift diagnostics.
- `viewer_server.py` + `viewer_src/`: actor export, local snapshot watching, field visualizer, browser physics/UI.
- `directional.py`: independent pure geometry/heading/contact oracles imported by PPO; kept separate to avoid circular evaluation dependencies.
- `contracts.py`, `training_config.py`, `checkpoint_io.py`, `cli.py`: explicit small boundaries for shared shapes/constants, user-controlled knobs, complete rewind/run lineage, public commands.

### Failures caught while truly moving source

- Removed-field import in `try_restore_recovery`: corrected after checkpoint round-trip CI failed. **Reason:** changing a class module does not automatically update every lazy import.
- Duplicate `summarize()` in moved posture/report logic silently overrode the campaign summary. Corrected by naming the optional function `summarize_posture()`; existing matched-scorecard roundtrip passed again.
- Source audit found old `capability_field` and `domain_randomization` doctor imports, stale atlas `paths.py` import, and a stale policy-export import in the real CUDA acceptance verification path. Corrected; added explicit CPU doctor, CLI planning and import-path smoke to GitHub Actions.
- Never declare the branch ready based on green tests alone. Exercise real WSL CUDA: 256→512 steps save/resume/verify, 20M cold start only if desired, export a tested actor and inspect the browser locally, including independent signed course/footfalls.

**No change to PPO reward weights, physical motor parameters, trained actor source files or schema-7 checkpoint payload is intended.** Source relocation still changes training source provenance; use a fresh run root, do not resume a historical lineage with this revision.

## Continuing in another chat

Inspect `git diff main...refactor/emu-minimal` and `git status`; don't trust the summary alone. Read `README.md`, `docs/INSTALL.md` and this file. Make actual, CI-verified reductions, not docs-only passes. Before any merge, check 16 Python + four browser tests, real WSL CUDA acceptance and browser model playback. Keep the 120M champion's file bytes unchanged.


## Research record and viewer work

A curated, evidence-aware experiment history lives in
[docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md). Keep public-facing
instructions in README / INSTALL / PLAYGROUND; keep internal observations and
future research hypotheses in that history. Preserve old raw logs through Git
history or external run roots rather than stuffing them into documentation.

Branch-only viewer additions: inverted pitch, split/hybrid joystick option,
browser yaw-heading P hold and physical radial terrain park. These change no
policy weights, source training environments, saved checkpoints or protected
showcase actor. Browser/MuJoCo physics playback needs hands-on verification
after automated JS checks. **Do not merge this branch yet.**

## Audit refinements (v5016, 17 Oct 2026)

Physical batch case recipes now carry explicit `speed`, `lateral`,
`yaw_rate`, `turn_rate`, and `settle_frames`. A zero-lateral sidestep
is rejected instead of producing a false pass. The same `exam_request()`
contract defines recorded protocol, recovery and batch skip/resume decisions.
A completed OFF/ON pair can be summarised offline after an interruption;
an incomplete or stale run is deliberately untouched. All logic stays in
existing `evaluation.py`; no new module architecture has been introduced.
