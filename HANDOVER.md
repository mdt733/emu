# EMU — active handover (24 October 2026)

> **Read first:** [Unified capability-growth architecture](docs/UNIFIED_CAPABILITY_GROWTH.md). **Research history:** [prior v5023 notes](docs/CAPABILITY_GROWTH.md), older versions of this file in Git history. This file records the **current state**, not every past experiment. Update it as the implementation advances.

## Mission and branch guardrails

Train **one** deployable, cold-startable six-actuator biped controller in **2,048 simultaneous MJX worlds**, using *one actor, one PPO optimizer*, physically honest 360-input sensing, and adaptive evidence-driven growth across commands (signed vx/vy/yaw and mixtures), resilience, **real inclined surfaces**, **real up/down steps**, transitions/posture, and energy-efficient/graceful normal walking. The six specialist emphases are **20/20/20/20/10/10%**, but the robot is not asked to solve six extreme skills from the first update. Every world may see easier versions of the others. The control/efficiency emphasis rewards **moving smoothly, tracking commands, maintaining support, and avoiding needless motor effort**, not standing still.

- **Active development:** `feature/v5023-capability-growth` at `mdt733/emu`; inspect latest HEAD before touching. Use reviewed branch-only commits/PR; **never edit/merge `main`, replace the approved 260M Pinnacle or showcase/demo without owner approval.**
- **`main` working comparison:** `2ec9e9db0a816c658131311129ef89ec7a595a12`; preserved as cold-start baseline.
- **No unified GPU run has been launched or authorized yet.** The old v5023 flat/1cm 10-hour control+A/B launcher and two obsolete runbooks have been **deleted** from this branch, and its old speed-only/foothold dual-arm training path removed; do not revive it.
- **Source-commit pinning remains critical.** Existing WSL checkouts and policies stay on their own historical source commits; any altered source needs its own independent clean run root.

## Why preserve main's foundation

Main's successful first 20M did **not** involve throwing the full mature curriculum at a newborn robot:

- Cold `foundation_config`: 20,000,000 total steps, seed733, 2048 worlds; fixed actor/PPO shape; 32-frame rollouts; **4 epochs ×8 minibatches**, LR 2e-4; max 700 control frames/episode.
- **70 PPO updates** (~4.59M world-steps at 2048 worlds) were **steady-only bootstrap**. Thereafter STEADY/TRANSITION/RESILIENCE allocation adapted to measured competence; posture was a separate global scalar. No turn-taking that starved a dimension.
- Initial vx request band **+0.020…+0.060 m/s**, with slowly earned foundation task-envelope limits `vx_hi=.40, vx_lo=-.12, vy=.14, yaw=.90`; phase-speed gain **.75**, max push **.70N**. These are sampling ceilings, *not evidence that the actor could already track them*.
- The **same eight original payments** teach command following, real supported moving gait, posture, contacts, anti-slip, foot clearance, smoothness, joint safety, and early-fall penalties. Small/zero new motor burden at cold birth.
- Controller uses pooled episode-count fall EMA (not ratios of differently sized rollouts), qualified expansion, faster retreat, and **one shared total difficulty-growth budget**. This protected the successful gait after earlier runs collapsed when too many dials grew together.
- The new 360-input physical-sensor network **cannot be bit-identical** to historical 220/232-input foundation. Preserve main's training dynamics/physics/reward doctrine and measure flat-gait parity at first **20M**; do not assume theoretical improvements are harmless.

## Single current design and implementation

**Canonical:** [docs/UNIFIED_CAPABILITY_GROWTH.md](docs/UNIFIED_CAPABILITY_GROWTH.md) (includes all six roles, L1/octahedral command envelope, evidence rules, main parity doctrine, acceptance gates, and implementation checklist).

- `emu/capability_focus.py`: concise six-cohort assignment and signed **L1** vx/vy/yaw probes, intentionally mixing interior, near-boundary, and mild extrapolation. No rectangular max vx+max vy combined request. Only one JAX actor.
- `emu/environment.py`: opt-in teacher-only `--unified-capabilities` world emphasis and per-birth activation. **Foundation preservation:** the proportion of actual specialist births starts at zero and rises as the *existing* steady difficulty earns competence (0 below steady 0.10, up to full by 0.32). Inactive births use main-style steady/transition/recovery tasks, not a new PPO policy. Active births label their actual specialist focus **only in teacher metrics**, never in 360 actor channels. Command specialists draw L1 probes; resilience active births stress learned push/recovery; control active births carry softer tasks and rewarded economical walking. Step specialists receive a higher birth probability, with lower-dose crossover in others. **This is partial implementation, NOT full slope+step training.**
- `emu/curriculum.py`: new opt-in serialized, *closed-window* command-sector evidence. Separate forward/reverse, signed sideways, signed yaw and mixed probes; one completed episode supplies genuine attempted/successful evidence, then growth only after adequate sample count/quality. Diagonal success never automatically grows pure axes, symmetrical vy/yaw needs **both signs** evidenced. Existing main adaptive steady difficulty, rollback and PPO health remain. Unified command frontiers seed gently, not `vx=+.60` at birth.
- `emu/rewards.py`: baseline main eight payments stay; an opt-in stronger but bounded mechanical-effort proxy and moving/tracking/supported smoothness bonus affect **active control** worlds; all other worlds retain normal gait smoothness/safety. Shaft power is **simulated mechanical work, not verified electrical/battery energy**.
- `emu/results.py` and `emu/training.py`: per-specialty actual occupancy, completed episodes/falls, L1 command-local quality and control shaft-work/m telemetry; `--unified-capabilities` config is saved with checkpoint provenance. Foundation-friendly unified settings retain main-ish slow `phase_speed_gain=.75`, `difficulty_base_step=.0014`, `regime_command_frac=.60`, no premature mature athletic/rehearsal/field-teaching.
- The live browser viewer already supports schema-9 **terrain_360** actor, delayed 4×4 ToF valid/range sensing and 1/2cm step courses. **It illustrates inference, not literal 2048 GPU worlds.** Do not regress the accepted old 220/232 public viewer.
- New focused `scripts/smoke_unified_focus.py` checks six actual MJX cohorts, 360 actor, one PPO update and governor round-trip on CPU; CI job `unified-focus-cpu` runs it. Confirm *current* commit CI rather than assuming an earlier revision's green run is valid.

## Physical and sensor integration — accomplished on CPU, still GPU-gated

**No six open triangles or per-world mesh recompilation.** Unified-mode MJX now uses two *solid tilted contact slabs*, both carrying the same sampled 2-D ground normal and an independently angled sharp lip. Each reset controls physical pitch, roll, step rise and edge yaw through movable mocap poses. Legacy plane/box models and archived actor contracts are unchanged. The 16 actor depth rays intersect exactly these physical oriented boxes; foot support and step scoring use surface-normal rather than world-z height. The three historical tripod ToFs were **removed from the NEW unified physical sensor path** (and historical policies retained separately). An explicit new versioned **4×4-array-only** terrain_360 contract derives ground plane/confidence and optical-flow depth from delayed 16 depth cells, without privileged slope normals. It stays **360 channels, 6 actuators**; 8×8 is NOT currently the actor sensor.

**Verified intermediate CI:** commit `f6fd2218` passed real 12-world CPU MJX mixed-slab + physical upper-foot contact + a PPO optimizer update (the browser and fresh Ubuntu viewer-install checks also passed). Additional slope governance/telemetry and test code were committed subsequently, **so recheck final HEAD CI before trusting it**. GitHub workflow is `.github/workflows/emu-v5015-checks.yml`, job `unified-focus-cpu`.

**New independent slope growth:** `RegimeParams.slope_exposure` and `slope_max_deg`, fully saved/resumed. After 70-update steady bootstrap and demonstrated core walking, it begins gentle actual <1° slope exposures, automatically explores up/down/cross-left/cross-right inclines plus low-dose overlapping steps, and only expands on completed, quality-qualified episodes with minimum counts in **all four real physical directions**. A mixed step/riser cannot masquerade as a successful slope. Console line shows **sampled physical incline occupancy** and earned angle, not the old static `slope=0.0°` when unified is active. Historical checkpoint shapes and governor states remain recognized.

**Still fail closed:** `emu/training.py` rejects `--unified-capabilities` before creating checkpoints until the **fresh final HEAD** passes CI, local **RTX4060Ti 2048-world** CUDA compilation/real PPO update, accurate viewer replay and a curriculum lifecycle smoke. Do not start a ten-hour run yet. The old `emu terrain-exam` was intentionally blocked from mis-scoring the new two-slab array-only actor; a new frozen *joint incline+step+speed* examiner is required, not a false pass from the old single-platform scorer.

**Next milestones:**
1. Re-run and close final `unified-focus-cpu` and Python CI on a frozen source commit; fix physical, sensor, reward and governor issues before any CUDA campaign.
2. Generalize the same compact persisted episode-evidence logic to resilience-force, transition/reversal and posture while protecting **main's 70-update easy gait** and shared total difficulty growth budget; inspect specialist severity crossover instead of piling on more custom task systems.
3. Add matched held-out mixed-terrain/speed/recovery examiner for **physical upper-platform support, sustained genuine incline walking, falls, command tracking, gait smoothness, mechanical shaft J/m**. Test GUI viewer schema9 array-only policy with real generated checkpoint; archived 232/360/220 policy loading must still work.
4. Freeze SHA in a **new WSL checkout**, run real 4/12-world CUDA PPO smoke and 2048-world memory/fps diagnostic (no policy/checkpoints destroyed). Only when it passes, lift fail-closed guard for one clean cold-start 20M source-pinned lineage.
5. Compare 20M vs main's proven reference using matched flat-gait and new terrain exams, then expand same one shared policy to a measured 10-hour training budget if quality and capability pass. No auto-promotion or main changes.

## Relevant prior trained brains and lessons — PRESERVE

Existing 360 actor `terrain_360` from **source `411b841de08ebfbb65c29b357c8ee0d39d577dd1`**: third fresh 20M→34M governor campaign `~/robot_lab/emu_runs/v5023_window_fresh733/growth34/checkpoints/final.pkl`. SHA256 checkpoint `9da0d47c6ced78e2d74ed3132ea0d05280255368cc0f519ff6b1ba2938d420f4`, exported policy SHA256 `e05264e3ec471bdb8c7798f366fa6dc483bac4a9b3435b0e86ec638882efd7f3`. WSL checkout `~/emu-v5023-window` must remain detached and untouched.

- Original governor 34M new actor at **0.15m/s command** found a verified supported ~**0.73cm** climb on seed1701. At seed1702 it got far-side root motion but failed the required supported upper-foot contact; do not mislabel far-side reach as mastery.
- **Same frozen actor**, seed1701, **0.30m/s command** got all **8/8 genuine supported** paired 1cm-course crossings, including actual **0.87cm and 0.95cm uphill** cases. Across two seeds and 0/1/2/4cm courses, paired aggregate crossing successes increased from **24/48 at .15** to **28/48 at .25 and .30**; higher speeds incurred more 2/4cm falls. 0.30 is *requested* speed, not measured speed at lip. Step shorter than 18mm spherical foot radius may still be mostly rolling/pushing.
- Additional 34M baseline actor source `19d2bc7` tested on 1701/1702. New governor policy modestly increased uphill totals **4/24→5/24** at .15 but worsened flat mechanical J/metre by ~25%; no reason to promote the old branch on reward alone.
- GPU throughput at 2,048 worlds on RTX4060Ti 8GB historically ~**29–32k env-steps/s** in PPO, but source compilation/frozen exams cost many minutes. The previous stopped 10h plan was a *design*, not a running job.

## Next work order — keep it clean

1. **Compare first-20M main behaviour against the new foundation**, keeping original PPO, contact, anti-freezing, reward and core governor. Treat `main` as working baseline. Review dormant 360 sensing effects independently.
2. **Solve and accept genuine mixed real slope+step physics** including per-world directions/severities and correct ToF/contacts. This is the hard launch blocker; don't weaken it.
3. **Bring the other specialist frontiers under the same compact persistent episode evidence contract**, not one bespoke subsystem per skill. Maintain overlapping challenges and a shared total expansion budget.
4. **Validate one true unified PPO/CUDA lifecycle**, build the *new* single-lineage 10h runner, demonstrate viewer live following and held-out composite exams. Only then request GPU time; never auto-promote main/showcase.

**Repository hygiene:** former dual-arm G6 knobs/teacher/launcher/tests and redundant runbooks have been culled from active branch. Old detailed research is available in `docs/CAPABILITY_GROWTH.md` and Git history. Avoid creating a replacement proliferation of 'v502x' one-off campaign files; use the one canonical charter, this handover and existing core Python modules.
