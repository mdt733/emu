# EMU v5023 — LIVE HANDOVER / start here

> **Last updated: 24 October 2026 (Australia/Sydney).** This is the **current** short handover for the ongoing capability-growth project. Update this file after each meaningful experiment, engineering change or new decision. The exhaustive charter, design rules, acceptance gates and dated session notes live in [docs/CAPABILITY_GROWTH.md](docs/CAPABILITY_GROWTH.md); do **not** duplicate that entire history here.
>
> **Important:** This document replaces a stale v5016 handover formerly at the same path. The prior text remains recoverable in Git history; it is not the active plan.

## 1. Project, ownership and inviolable constraints

- Repository: **https://github.com/mdt733/emu**
- Active **development** branch: **[feature/v5023-capability-growth](https://github.com/mdt733/emu/tree/feature/v5023-capability-growth)**; draft **[PR #14](https://github.com/mdt733/emu/pull/14)**. It is a work-in-progress branch, **not** approved for merge.
- **Never modify/merge protected `main`, existing 260M v5016 Pinnacle, showcase policy, or GitHub Pages without explicit owner approval.** Main is the known working baseline; this branch must earn promotion in independent physical exams.
- **CRITICAL: training checkpoint provenance is SHA-pinned.** The two real GPU campaigns below used exactly **`0ba009fe7a046ff1e5a28cbdd55e2d05e8e4b75e`**, regardless of how far the active GitHub feature branch has moved since. **DO NOT `git pull` in `~/emu-v5023` before further resume/fork from those checkpoints**, and do not weaken the source-provenance guard. Do new feature development in a separate clone/worktree; a migration to newer source requires a deliberate compatibility/evidence plan. The GitHub branch contains newer log/report-only and handover commits not present in the pinned training checkout.
- User's machine: Windows 11 / WSL2, NVIDIA RTX 4060 Ti **8 GB**, CUDA 13 JAX backend `gpu`. Project's original `bash scripts/setup_wsl.sh --cuda13 --install-node` works. **2,048 simultaneous worlds have already trained full PPO at ~30–32k environment steps/s** with the 360-input actor. A `warp` import warning was nonfatal.
- Development style: **read and fix the existing coherent code first**; minimal meaningful smoke tests, not a proliferating test harness or sidecar trainers. Small foundation, bounded continuations and small controlled forks. Save results as Markdown/JSON and share bundles.

## 2. Vision and architecture

Train **one physically honest six-actuator biped** whose *reliable usable capability* grows across forward/reverse/sideways/yaw commands, combinations, acceleration/stop/reversal, posture, pushes/recovery, real inclines, physical upward/downward finite steps, sensor uncertainty and mechanically graceful movement. Reward, actual task success, safety, efficiency and confidence must remain distinct. No success from freezing, falling or exploiting scalar reward. Expand fronts on measured evidence, retreat struggling dimensions while preserving easy gait.

- `robot.xml` + `emu/environment.py`: robot/MJX world, physical contacts, true episode terrain geometry, sensor acquisition and observations.
- `emu/contracts.py`: versioned fields and immutable sensor contract. Current new `terrain_360` actor = **90 fields × 4 frames = 360**, six action targets. 58 existing sim2real fields + **16 4×4 slant-ToF ranges + 16 validity bits** per frame. Simulated rays/noise/latency/foot occlusion; **no privileged terrain map** in actor. Checkpoint schema **9**, export schema **3**; historical 220/232 remain unchanged. Hardware sensor placement is *provisional*, not calibrated.
- `emu/rewards.py`: locomotion/commands/safety payments; current terrain-aware actor gets **sparse +10 only after real far-side crossing**.
- `emu/ppo.py` + `emu/training.py` + `emu/checkpoint_io.py`: one PPO trainer, training/health/normalization, validated checkpoint/resume, strict provenance.
- `emu/curriculum.py`: one `RegimeGovernor` for task difficulty plus `CapabilityField`. Finite terrain is layered onto walking (>=75% ordinary-world births), 20M totally flat foundation, then initial 1cm / 5% terrain dose if core gait healthy.
- `emu/evaluation.py` + `emu/results.py`: **frozen, independent** held-out scores, MJX real foot/contact step examiner and offline `results.md`. `emu/viewer_server.py` handles .emupolicy export; browser parity of new schema-3 policy still needs actual verification.

**Known physics constraint:** current v5023 disallows finite physical steps and actual inclined contact floor together. Future mixed terrain requires an intentional, versioned world/sensor/evaluator upgrade, *not* claiming tilted-gravity alone is uneven ground.

## 3. Important completed engineering

- G3 finite-step physical geometry with a single episode-moved finite MJX platform, including actual **up and down** ledges, angle variation, physics reset, 16-zone actual simulated sensor returns.
- Eliminated a **serious dormant-floor contact regression** using non-overlapping floor/platform collision masks. Frozen original-flat and dormant-platform controls must agree exactly. 34M independent exam: both **8/8 survival and +0.7137 m**, per-world progress deltas 0.
- G5 360-input full PPO gradient + sensor acquisition confirmed on CPU CI; owner confirmed CUDA 4-world gradient and 2048-world sensor physics ~30,080 world-steps/s; **real 2048-world PPO trained successfully**.
- Automated report generation, shorter console lines (vx frontier, slope, terrain exposure/height, resilience, mechanical shaft J/m approximation) were **implemented on newer feature commits**. They are NOT in the pinned 0ba009f training code, which still prints legacy v5015 lines and needs the offline reporter run manually.
- Recent feature-branch reports/README/charter changes must not silently rewrite the source lineage of the two trained policies. The protected main and champion were not changed by this work.

## 4. Two real owner-GPU policies — preserve the checkpoints

Both: source **`0ba009fe7a046ff1e5a28cbdd55e2d05e8e4b75e`**, `terrain_360`, seed **733**, **2,048** worlds, `--terrain-step-max-m 0.04 --pitch-deg 8`, JAX/CUDA, same 360-input contract.

| Experiment | Training result | Exact WSL checkpoint location | SHA-256 of final.pkl |
| --- | --- | --- | --- |
| **20M cold foundation** | `20,054,016` steps; `--profile foundation`; final reward +3.036, 700-frame episodes, 0% final training falls, `core` held-out survival 1.000. **No physical step lessons** before 20M. | `~/robot_lab/emu_runs/v5023_foundation20_0ba009f_20261024_081113/checkpoints/final.pkl` | `287da5a235c2ba1b8534ada2af8744c5027b4aeb1c7bf6b1628398c022983f46` |
| **34M grown control** | `34,013,184` steps, `--profile sim2real`, resumed exact foundation weights/optimizer; final training reward +2.786; 100% held-out core survival; adaptive **real 1cm physical steps**, final ~16.7% terrain exposure, **no slope**. | `~/robot_lab/emu_runs/v5023_growth34_control/checkpoints/final.pkl` | `c0320b46d83d78642c3f1108943d893c2c2a02c7c9148ddf9db9cf76258edaa7` |

Original uploaded reports: `results_bundle(5).zip` = foundation; `results_bundle(6).zip` = growth. Each run root also has `metrics.csv`, `fixed_eval.csv`, `training_config.json`, `run_meta.json`, checkpoint and manually generated `results.md`, `results_data.json`, `results_bundle.zip`. These ZIPs **do NOT contain policy checkpoints**.

**Matched frozen fixed evaluation** (higher overall score but score is not a universal gait grade):

| Independent panel (mostly survival fraction) | 20M | 34M |
| --- | ---: | ---: |
| Overall broad exam score | **−0.045** | **+0.638** |
| Core survival | 100% | 100% |
| Dynamic80 survival | 24.0% | 98.3% |
| Combo70 survival | 51.0% | 99.8% |
| Resilience75 survival | 24.1% | 76.9% |
| Robust40 survival | 56.4% | 97.3% |
| Growth110 survival | 18.9% | 73.0% |
| Growthcombo110 survival | 29.3% | 97.2% |

34M has **significant broader survival and improved step-down behaviour** but still imperfect command precision. Capability field has 194 known/0 “competent” command nodes and 1528 known/0 “competent” transition edges at 34M despite high held-out survival; investigate tracking-confidence thresholds before granting it heavy curriculum authority. Servo loading is appreciable (late training shaft positive power ~8.6 W; ~57.5% samples over nominal continuous torque), **not battery electrical energy**; compare cost only under matched physical workloads.

## 5. CRITICAL NEW FINDING — matched frozen real-terrain exams, now both completed

Two original uploads: **34M** = `terrain_exam(1).json` / `.md`; **20M** = `terrain_exam(2).json` / `.md`. These are two distinct **frozen actor** exams on the **same source, seed 1701, 8 worlds, 350 frames (7 s), vx command 0.15 m/s, CPU**, and ceilings **0/1/2/4cm**, in the SAME cohort/physical geometry. Actual step height is sampled **30–100%** of the ceiling (identical sampled heights per world across these two exams). Four even-index **step-up** / four odd-index **step-down** worlds per nonflat height. Each true pass requires real far-side 100ms foot support AND completing the horizon without resetting.

| Exam condition | 20M flat trained | 34M terrain trained |
| --- | ---: | ---: |
| Original flat survival | **4/8**; +0.583 m mean progress | **8/8**; +0.714 m |
| Dormant terrain platform 0cm | 4/8, flat-parity control | 8/8, flat-parity control |
| **1cm step-ups** | **0/4** | **0/4** |
| **1cm step-downs** | **2/4** | **4/4** |
| **2cm step-ups** | **0/4** | **0/4** |
| **2cm step-downs** | **1/4** | **3/4** |
| **4cm step-ups** | **0/4** | **0/4** |
| **4cm step-downs** | **2/4** | **2/4** |
| **All ascent / descent** | **0/12 up; 5/12 down** | **0/12 up; 9/12 down** |

**Interpretation:** 34M is unambiguously better in same-seed broad locomotion/flat control and step-down success; a portion of improvement may come from improved general walking, not a separately proven sensor-learning effect. Neither model has demonstrated **ANY real upward crossing**, at any sampled height. **34M survived all 12 ascents but STALLED** (1cm avg +0.159m progress, 2cm +0.031m, 4cm −0.049m). This is an **ascent/approach/lifting skill failure**, not principally a falling/recovery failure. Blocked ascent has high simulated J/m due to small denominator; not battery energy. Direction cohorts have **different sampled height magnitudes**, although exactly reproducible across these 20M vs 34M runs; future examiner should pair **equal magnitudes and seeds** for up/down.

**Root-cause hypothesis in current code:** `environment._terrain_crossing()` and `results.summarize_rollout()` aggregate successful up/down traversal into `terrain_episode_success`; `RegimeGovernor.advance_terrain()` advances on the **combined** rate >=50%. With down-only success this can mistakenly unlock harder terrain even when upward ability is zero. Sparse +10 is awarded only after a complete crossing, so an actor with zero upward successes gets little *direction-specific* learning signal. At 34M, final terrain height was still **held at 1 cm**, not 4cm: the governor's exposure grew ~16.7% while average training-world 1cm completions remained modest. Across **22 logged** rollout samples: 245 completed terrain attempts / 70 complete successes (~28.6%, sample totals, NOT all 14M steps). **This is a verified asymmetry and a concrete hypothesis, not proof that the ray sensor itself is faulty.**

## 6. Exact next task for new chat — fix learning before more blind GPU time

**Start with targeted read-only inspection; do not auto-run a long GPU continuation or modify main.**

1. Review **`environment._episode_terrain`, `_terrain_crossing`, `_terrain_4x4_ranges`, the +10 crossing reward, `results.summarize_rollout`, `curriculum.RegimeGovernor.advance_terrain` and `evaluation.evaluate_terrain_course`**. Trace the actual physical up/down world reset/geometry, contact score and sensor signals. Especially why a policy can survive at an upward lip yet fail to lift/clear it.
2. Fix **direction-conditioned evidence**: independent up/down completed attempts, supported crossing success, falls, tracking and cost. Gate future height/exposure expansion on **both directions or an explicit ascent minimum**, rather than pooled down-only success. Never give actor privileged geometry.
3. Add **careful, bounded ascent progress/clearance teaching**, using actual physical progress/contact from teacher-only MJX state so the robot has learnable incremental ascent behaviour before a rare all-or-nothing +10. Require *real forward displacement* and support, not foot-flailing, spinning, reversing, standing at the riser or simply moving while falling. Protect flat locomotion and existing movement rewards.
4. Improve **independent frozen exam** with matched up/down *actual heights*, early-lip approach and foot contacts, multiple seeds, true successful crossing and command/effort evidence. Preserve original-flat vs dormant-flat parity gate. Audit ToF 4×4 view of the approaching lip (occlusion/angle) before blaming training.
5. Run one small cold/source-compatible smoke & **bounded 34M→~46M experimental fork** only after reviewing how the new source can validly inherit the pinned old weights/optimizer **without silently bypassing source provenance**. Another small from-scratch trial is acceptable if migration cannot be justified. Compare independently against 34M and 20M at the same fixed exams before granting further height.
6. Broader roadmap after meaningful upward skill: staged command/resilience accuracy, physical slopes, graceful shaft work, held-out combined challenges, 1–3 seed replication, challenger vs approved 260M champion; keep one bounded multi-axis teaching budget. Absolutely no unexamined 120/400M marathon and no unapproved main merge.

## 7. Operational commands and caveats

**Pinned training checkout (`~/emu-v5023`): leave it at `0ba009f`; do not pull.** To produce archived reports, existing code uses **offline** reporter, because automatic `results.md` generation was implemented on later feature branch:

~~~bash
cd ~/emu-v5023 && source .venv/bin/activate
git rev-parse --short HEAD                # MUST be 0ba009f
python -m emu.results --root ~/robot_lab/emu_runs/v5023_growth34_control
~~~

`emu export-policy <checkpoint.pkl> <new-file.emupolicy>` exports a frozen actor. Existing independent exam syntax:

~~~bash
emu terrain-exam \
  --policy ~/robot_lab/emu_runs/v5023_growth34_control/growth34.emupolicy \
  --output ~/robot_lab/emu_runs/v5023_another_unique_exam_dir \
  --device cpu --worlds 8 --horizon 350 --speed 0.15 \
  --heights-cm 0 1 2 4
~~~

**New source development:** use another checkout, e.g. `git clone -b feature/v5023-capability-growth https://github.com/mdt733/emu.git ~/emu-v5023-dev`. Read the *latest* branch head/CI first. Make any code changes **only** there / through connected GitHub branch tools. New branch commits may be ahead of both actual trained checkpoints; after editing, the branch must pass its own CI and physical evidence before any new long experiment. Never assume that today's remote HEAD equals the exact code used for the 20M/34M runs.

### 2026-10-24 — directional terrain evidence correction

- Source commit: [153d087](https://github.com/mdt733/emu/commit/153d08744c266c321160b699733f48299a2b0bfc), plus focused governor test. All changes exclusively on the v5023 feature branch; main, champion, showcase and Pages unchanged.
- Code-first review: real step-up/down geometry and 4x4 ray/foot-occlusion already exist; crossing requires sustained far-side MJX foot support. But training outcome metrics and governor pooled up/down. Old matched frozen result is STILL 0/12 up for 20M and 34M, with descent 5/12 vs 9/12.
- Now published per direction: completed attempts, genuine support-confirmed crossing success, falls, active-frame support/bad-contact, vx tracking MAE and mean mechanical shaft-drive W (not battery power). Original pooled counts remain available. Both training report paths carry new episode counts.
- New governor growth gate: **at least three completed episodes for EACH direction in a rollout** and **>=50% successful supported far-side crossings for BOTH**. All-down successes and old rows without directional evidence no longer advance terrain. Sparse evidence holds the frontier; failed ascent/descent retreats terrain only. Existing 360 observation, sensor/physics geometry, core rewards and governor state schema remain unchanged.
- Limitations: per-rollout counts may be sparse at 5% terrain dose; do not infer competence or relax the gate without validated serialized accumulation. New feature-branch source is NOT yet approved to resume the 20M/34M source-pinned 0ba009f checkpoints. No new GPU run or new ascent claim.
- **Exact next task:** implement bounded monotonic ascent progress and supported clearance teaching without privileged actor information or reward loops for rocking/flailing; next verify actual height-matched up/down exams, near-lip ToF visibility and original-flat/dormant-flat parity before planning source-compatible bounded GPU fork.
## 8. Completion and handover discipline

**Current status:** 20M and 34M GPU runs **COMPLETED**, both checkpoint-validated; independent 20M-vs-34M terrain exams **COMPLETED**; **ascending real steps 0/12 in both**. Directional reports/governor fixed on updated feature source; no new physical training result yet. Main and champion **protected**. Next work is **code-first ascending-learning correction and independent paired exam**, not more of the unchanged trainer.

**Carry-over checklist:**
- [x] Save the mission/architecture, protected branches, pinned source, both checkpoint paths/hashes and exact results.
- [x] Validate real CUDA 360 actor/2,048 worlds; finish cold foundation and mature continuation.
- [x] Frozen independent terrain A/B at both 20M and 34M with original/dormant physics parity.
- [x] Direction-separated up/down evidence and curriculum gates (implemented; GPU verification pending).
- [ ] Bounded ascent guidance and actual confirmed up-crossings.
- [ ] Matched-height, multiple-seed independent physical exam; preserved command/core ability.
- [ ] Safe reviewed source/checkpoint migration or fresh suitable cold-start; short training fork.
- [ ] Mixed actual contact slopes and finite terrain, servo/efficiency and browser 360 parity.
- [ ] Independent broad comparison and **explicit owner approval** before PR merge or champion/showcase changes.

**New-chat first instruction:** “Read `HANDOVER.md` and `docs/CAPABILITY_GROWTH.md` on branch `feature/v5023-capability-growth`. Preserve `main` and the Pinnacle champion. Verify the current branch HEAD, read the physical up/down code, and begin the direction-aware ascent teaching fix. The owner's trained 20M/34M checkpoints are pinned to `0ba009f`; do not casually `git pull` that WSL checkout or bypass provenance.”

**Every future session:** after any change, update **this very file**: newest evidence, blockers, exact next task, any new run path/commit/CI result; move long narratives into the dated charter. Keep it short, factual and reproducible.
