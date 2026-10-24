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
- Console follow-up: commit [14dc43a](https://github.com/mdt733/emu/commit/14dc43a0ae5012d7829a8a0a050f4c7eea9208b7) also displays up/down supported completions as passes/attempts (or a dash when unsampled) on the existing training headline. Detailed evidence stays in metrics.csv.
- New governor growth gate: **at least three completed episodes for EACH direction in a rollout** and **>=50% successful supported far-side crossings for BOTH**. All-down successes and old rows without directional evidence no longer advance terrain. Sparse evidence holds the frontier; failed ascent/descent retreats terrain only. Existing 360 observation, sensor/physics geometry, core rewards and governor state schema remain unchanged.
- Limitations: per-rollout counts may be sparse at 5% terrain dose; do not infer competence or relax the gate without validated serialized accumulation. New feature-branch source is NOT yet approved to resume the 20M/34M source-pinned 0ba009f checkpoints. No new GPU run or new ascent claim.
- **CI state at this update:** original handover-head workflow 40032619063 was green; new engineering-head checks [40034943266](https://github.com/mdt733/emu/actions/runs/40034943266) are queued/pending, so these corrections are NOT yet a verified full CI pass.
- **Exact next task:** implement bounded monotonic ascent progress and supported clearance teaching without privileged actor information or reward loops for rocking/flailing; next verify actual height-matched up/down exams, near-lip ToF visibility and original-flat/dormant-flat parity before planning source-compatible bounded GPU fork.
### 2026-10-24 — ascent teaching built; owner approves NEW 34M cold-start campaign

**Owner decision:** Rather than attempting to migrate the old 0ba009f 34M optimizer/brain across source revisions, train a new terrain_360 policy from scratch using the latest reviewed v5023 source. First **20M physically flat foundation**, then 20M→**34M cumulative sim2real growth**, 2,048 GPU worlds, seed 733, max terrain topology 4cm, 8deg pitch. NO cold-start of hard sim2real profile from step zero. Run only from ONE pinned clean Git commit for both phases; keep original 0ba009f checkout/checkpoints unchanged.

**New code on feature branch:** `0385dc9` adds one-episode high-water marks for real forward ascent approach and a properly lifted/reaching foot while the *other* foot has a real MJX floor-contact pair. Requires commanded positive vx, controlled vy/yaw, a supported nonfallen body and physically forward root displacement. Both gains are monotonically one-time: max **6.6** points root + **2.4** foot per ascent episode, plus unchanged **+10** only for genuine sustained far-side contact. Flat/foundation and descent rewards get zero new shaping. No changes to 360 actor channels, actions or old checkpoint schema; new state is process-local episode state. `3a037da` logs the ascent guidance separately from actual crossings.

**Better frozen exam:** `3f4a0d3` adds optional **paired_terrain_heights** in `EmuMJXEnv`, used ONLY by the frozen terrain examiner. Each adjacent up/down pair is assigned identical **actual** 30–100%-of-ceiling height and edge yaw from seed/pair index, whereas training retains normal per-episode randomness. `4c42c6a` turns this on for the exam and rejects mismatched pair heights, records new v2 protocol. Old v1 exams can still be compared as historical baselines, but they do NOT have direction-matched heights. Run v2 on both the new 34M and original 34M policy (using updated separate examiner code if available) for truly matched A/B.

**Review caveats:** None of the new code proves climbing yet; a successful independent ascent remains 0/12 on BOTH archived policies. New episode-state paths and paired frozen examiner require current GitHub CI physical/PPO smoke before committing overnight GPU time. CPU GitHub checks on intervening commits may be cancelled as the branch advances; the FINAL head needs its own clean pass. No full 2048-GPU run was performed in-chat. The expanded teacher compute may change throughput. Leave protected main, 260M Pinnacle, showcase and GitHub Pages untouched.

**Run recipe, from clean WSL checkout (new folder only):**

~~~bash
git clone --branch feature/v5023-capability-growth --single-branch https://github.com/mdt733/emu.git ~/emu-v5023-ascent
cd ~/emu-v5023-ascent
# After checking the branch CI, detach at the checkout's exact SHA and never pull during this run.
git checkout --detach "$(git rev-parse HEAD)"  # freeze new checkout exactly once
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ROOT="$HOME/robot_lab/emu_runs/v5023_ascent_fresh733"
emu doctor --require-gpu
emu train --run-dir "$ROOT/foundation20" --profile foundation --steps 20000000 --envs 2048 --seed 733 --device cuda --terrain-actor --terrain-step-max-m .04 --pitch-deg 8
emu train --run-dir "$ROOT/growth34" --profile sim2real --steps 34000000 --envs 2048 --seed 733 --device cuda --terrain-actor --terrain-step-max-m .04 --pitch-deg 8 --resume "$ROOT/foundation20/checkpoints/final.pkl"
emu export-policy "$ROOT/growth34/checkpoints/final.pkl" "$ROOT/growth34/growth34.emupolicy"
emu terrain-exam --policy "$ROOT/growth34/growth34.emupolicy" --output "$ROOT/exam34_seed1701" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1701
emu terrain-exam --policy "$ROOT/growth34/growth34.emupolicy" --output "$ROOT/exam34_seed1702" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1702
~~~

**Expected evidence:** first 20M contain NO real steps (even though allocated terrain topology); new code's up/down console indicators show dashes until episodes complete; thereafter look at actual completed ascents vs descents, near-lip progress, standard core/speed tracking and failure rates. Each phase is separately SHA-pinned and checkpoint validated, and standalone runs create automatic results. If stopped mid-run, `--resume auto` for the SAME run directory/config is supported, but do not change SHA. Checkpoint source lineage must remain identical between the phases.

**Exact next engineering task once run results arrive:** compare frozen v2 paired-height multiple-seed ascents and core gait to old policy; if upward crossing remains zero, inspect visibility/timing of 4x4 rays and actual floor/foot impulses at the lip before increasing reward or terrain height. Do not grant promotion merely from positive training reward.
### 2026-10-24 — uploaded NEW 34M source-pinned run, first bounded ascent evidence

Owner uploaded `results_bundle(7).zip` and `terrain_exam(3).json` / `.md`. Full 34M *fresh source* path is `~/robot_lab/emu_runs/v5023_ascent_fresh733/growth34`, final **34,013,184** steps; 2,048 CUDA worlds, seed **733**, code **`19d2bc7f13dcc3a5edfc16c23a6074de788957cd`**, `terrain_360`, `--pitch-deg 8`, 4cm allocated topology, physical slope disabled. Checkpoint `checkpoints/final.pkl` digest **`d9a937a85828122f961abf848f6e36cce3b3b3cebde27abb374bf2d3a7ceb868`**. Frozen actor artifact SHA-256 **`c61a28ee96bd639438c7a4dc1e2e727134c0420c2bbae9022b02df8cbe401803`**. Foundation lineage appears as the explicit `--resume .../foundation20/checkpoints/final.pkl` in config; uploaded ZIP has no raw checkpoint or foundation artifact to independently hash.

**v2 matched-up/down independent physical exam, seed 1701, 8 worlds × 350 frames at requested +0.15m/s, actual matched height & edge yaw per pair:** Original flat and dormant topology **8/8 survival**, identical **+0.8540m** and *zero per-world progress/failure delta*. At 1cm ceiling: step-up **2/4** (ONLY actual **0.37 and 0.46cm** ascents; actual 0.87/0.95cm still stalled), step-down **4/4**. At 2cm ceiling: step-up **0/4**, step-down **4/4**. At 4cm: step-up **0/4**, step-down **2/4**, two downhill height+tilt falls. TOTAL **2/12 ups; 10/12 downs**. An actual ascent success is now observed, but do **not** call this 1cm+ mastery or attribute causally to shaping without rerunning historical 34M actor under same v2 protocol. The old frozen exam had 0/12 up and 9/12 down, but it used unpaired actual heights.

**Broad fixed held-out exam:** core survival **100%**, dynamic80 **97.9%**, combo70 **99.61%**, resilience75 **71.34%**, growth110 **74.56%**, growthcombo110 **97.05%**; broad score **+0.630** vs prior old-source 34M **+0.638** (different trained policies, inspect per-panel quality instead of overreading scalar). Core vx/vy/yaw MAE **0.0149/0.0599/0.1383**. Late training falls around **8.1%**, +2.739 reward, positive shaft drive power around **8.56 simulated W**, motor over-nominal fraction around **56%** (not measured electrical watts or thermal qualification). Capability field 191 known / zero competent command nodes, ~1555 known edges / zero competent; unresolved calibration/quality issue.

**Exact governor starvation diagnosed:** of 22 *logged* rollout snapshots (NOT all 212 training updates), physical attempts total **40 up / 48 down** with only **4 up / 33 down completions**. Nineteen of 22 snapshots returned `await_3_terrain_episodes_each_direction`, two `terrain_retreat_failed_ascent`, one `terrain_hold_directional_learning`. Even with 2,048 worlds, 5% terrain births yield too few completed *per-direction episodes in one 32-frame PPO batch*. The governor remained at **5% exposure, 1cm height ceiling** for all logged snapshots; this is a curriculum evidence aggregation flaw separate from genuine ascent skill limitations. The *mean shaped ascent payment* is tiny when averaged over all rollout frames and should not be interpreted as training performance. Need bounded, serializable **multi-rollout directional evidence**, not pooled down-only success, before next fresh trial. Keep old 34M checkpoint unchanged; new source changes forbid cross-commit resume without explicit migration.

**Next precise task:** correct `RegimeGovernor.advance_terrain` using a bounded, strictly checkpoint-persisted directional observation window with minimum effective counts in BOTH directions, retain hard physical crossing criteria, cap initial terrain exposure to protect flat practice; only grow real height when up AND down supported-crossing performance is credible. Update one focused regression test and live handover. Then frozen v2 exam historical 34M exported actor (read-only) under identical seed/heights before deciding whether more GPU time is justified.
### 2026-10-24 — sparse-evidence governor correction after new 34M result

**Engineering commits:** `a9a136f` (+math import `111c199`), focused directional cohort/restore test `4ed3625`, telemetry integration `2464626`. Work only on `feature/v5023-capability-growth`; do not merge into main or edit archived source-pinned `~/emu-v5023-ascent` (`19d2bc7`) or old `~/emu-v5023` (`0ba009f`) during an existing checkpoint lineage.

- **Root cause, directly visible in uploaded metrics:** existing per-PPO-rollout requirement >=3 complete up AND down terrain episodes is too sparse at 5% terrain births (19/22 recorded summary rows said await); exposure/height remained exactly 5%/1cm. This masks real but rare partial ascent progress. Correct the signal sampling, NOT the actor sensor dimensions/PPO.
- `RegimeGovernor` now owns one **serialized 40-update-max up/down evidence window**, accumulating only physical completed episode attempts, sustained crossing successes and falls. A decision needs >=**20 real completed attempts per direction**. Unknown/missing old pooled data cannot certify either direction. After each evaluated decision the window resets (no replay of easy evidence for new height); core decline also retreats/reset independently. Versioned governor state accepts exactly recognized old shape with empty window, but **training still enforces exact Git SHA checkpoint provenance**; do not take this as approval of old 34M resume.
- Severity is deliberately separated from teaching dose: **both directions >=50%** physically complete supports growth (first exposure max 25%, then height up +0.025cm/decision); if ascent >=10% and descent >=50% but either <50%, permit +0.5 percentage point practice births per evidence window **up to 10% exposure**, holding 1cm height. Normal walking remains >=90% in practice stage. Zero-ascent/worse performances still locally retreat/hold. These thresholds are an explicit next controlled curriculum hypothesis, **not** proof of learning improvement.
- `training.py` writes `terrain_window_*` counters to metrics/report for visible evidence accumulation, on both periodic and final paths. A focused governor test checks same outcome before/after `state_dict` restore, zero-up block, unlabeled refusal, small-up practice, both-way real crossing growth and sparse window expiry. No actor/observation/action/robot geometry/reward changes in these commits.
- **Verification status:** code committed; new-head GitHub CI workflow has been queued but must pass its Python/MJX path before authorizing a fresh larger GPU campaign. No new training or physical exam has yet been run on this updated governor; all numeric physical results above come from original `19d2bc7` 34M run.

**Low-cost immediate matched baseline and replication (RUN ONLY using pinned `~/emu-v5023-ascent` examiner, which already has v2 physical pairing, not the newer unverified branch):**

~~~bash
cd ~/emu-v5023-ascent && source .venv/bin/activate
ROOT="$HOME/robot_lab/emu_runs/v5023_ascent_fresh733"
OLD="$HOME/robot_lab/emu_runs/v5023_growth34_control/growth34.emupolicy"
# Genuine apples-to-apples old 34M frozen actor under exact v2 1701 cohort
emu terrain-exam --policy "$OLD" --output "$ROOT/old34_v2_seed1701" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1701
# Repeat new 34M frozen actor on a second independent seed
emu terrain-exam --policy "$ROOT/growth34/growth34.emupolicy" --output "$ROOT/exam34_seed1702" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1702
~~~

Send `old34_v2_seed1701/terrain_exam.json` and `exam34_seed1702/terrain_exam.json` to compare old vs new and replication. New-source cold-start (20M flat →34M growth) should be scheduled only after review of these comparisons and green CI, retaining original trained artifacts for A/B.
### 2026-10-24 — FAIR v2 A/B CONFIRMED: new 34M still no improved crossing count

Owner supplied `terrain_exam(5).json` = **OLD 34M actor** (artifact SHA256 `9dd983dea211893c9aaba57db2fbaa6f431a59328720eba6ec45cd5725a647ba`, seed 1701), and `terrain_exam(4).json` = **NEW 34M actor** (artifact SHA256 `c61a28ee96bd639438c7a4dc1e2e727134c0420c2bbae9022b02df8cbe401803`, seed 1702). Examiner source SHA **`19d2bc7`** for both; old actor trained at source **`0ba009f`**, new actor trained at **`19d2bc7`**. Previous `terrain_exam(3).json` is NEW 34M actor seed 1701. Frozen v2 examiner uses exact matched physical height/yaw between ascent and descent per adjacent world, and same cohort across both actor versions at seed 1701.

| Frozen seed / actor | 1cm UP/DOWN | 2cm UP/DOWN | 4cm UP/DOWN | UP total | DOWN total | original flat progress |
|---|---|---|---|---|---|---|
| OLD 34M, seed 1701 | 2/4, 4/4 | 0/4, 4/4 | 0/4, 3/4 | **2/12** | **11/12** | **0.7137m**, survive 8/8 |
| NEW 34M, seed 1701 | 2/4, 4/4 | 0/4, 4/4 | 0/4, 2/4 | **2/12** | **10/12** | **0.8540m**, survive 8/8 |
| NEW 34M, seed 1702 | 2/4, 4/4 | 0/4, 4/4 | 0/4, 2/4 | **2/12** | **10/12** | **0.8928m**, survive 8/8 |

**Critical correction to previous optimism:** The old policy also cleared exactly the same small upward 0.37cm and 0.46cm steps in seed 1701. Thus improved UP crossing count from ascent shaping is **NOT established**. Seed 1702 new policy cleared only 0.58cm and 0.67cm; failed 0.69, 0.90cm steps, then all larger upward steps. These are ceiling categories, NOT fixed true heights. Under the strict matched seed 1701 comparison the new actor travels further on every ascending category (1cm up average progress **0.473 vs 0.319m**, 2cm **0.144 vs 0.085m**, 4cm **0.009 vs -0.005m**), which suggests partial improved approach locomotion but not proven climbing. At 4cm ceiling new actor has 2 downhill falls vs old 1; do NOT trade descent robustness for apparent reward gains. Two seeds are still a tiny skill sample.

**Updated engineering inference:** The unchanged per-rollout 3+ cohort governor at source `19d2bc7` never left 5% exposure or 1cm training height; better flat gait/near-lip progress did not become real large-step clearance. The new checkpoint-serialized multi-rollout up/down evidence and max-10%-exposure ascent-practice correction already exists on branch (`411b841`), and full GitHub workflow **40041202515 succeeded** (browser, CPU-viewer, Python/MJX). This correction has NOT been GPU trained or empirically evaluated. Recommend one new reproducible 20M flat→34M cold start pinned to the reviewed branch SHA **only if owner wants next experiment**, with independent seed 1701/1702 frozen v2 exams and strict <=1cm actual ascent analysis; don't continue an old checkpoint on a different SHA, don't touch `main`/champion.

**Next tasks:** (1) decide new-source short campaign and evaluate whether practice dosage makes upward contact succeed beyond 0.67cm; (2) check teacher-only ascent shaping actual payout over lip distance and ToF validity near each ledge; (3) inspect physical foot clearance/contact and anti-rocking behaviour for cases that approach but stall, especially ~0.69–1.00cm. The governor can learn to practice but cannot fix missing sensor visibility or impossible foot lift. (4) Freeze matched challenge exams across revisions; do not declare first-ever ascent again based on unpaired historical samples.
### 2026-10-24 — code audit: is gait phase preventing actual foot placement?

Owner is training the independently cloned `~/emu-v5023-window` pinned to **`411b841`**. **Do not change that checkout or tell owner to interrupt the running 20M→34M campaign.** Read-only source audit of exactly `411b841`:

- **No hard-coded gait pattern in actual control.** Actor is a feed-forward **360→256→256→6** tanh policy with **six independent normalized joint-position actions at 50Hz** (both hip roll/pitch and knees), bounded servo ranges and exactly one 20ms delivery delay; `environment._simulate_physics` applies previous raw action to servo position targets, never synthesizes a gait trajectory. Passive shin springs respond to physics, not a forced periodic actuator.
- **Soft gait-clock pressure EXISTS:** teacher advances `phase` at positive commanded `phase_rate`, actor sees sin/cos phase each observation, reward's `organic_step_shape` includes `0.30*phase_sync*step_evidence` and `clearance` then includes `0.45*organic_step_shape` with total reward term coefficient `0.16`. Thus maximum direct phase-sync clearance contribution is `0.16*0.45*0.30=0.0216` per frame; NOT a hard foot-lift rule. Actor can disobey clock and hold/move any joint. Clock itself cannot be paused by the actor. Policy has only **4×20ms=80ms** raw observation history and no recurrent hidden state, but can react to ongoing proprioception/ToF.
- **Real learning-budget caveat:** foundation is all flat for first 20M. A 34M run permits just 14M post-foundation steps. If exposed terrain births remain 5% and up/down split evenly, *roughly* **350k ascent world-frames** (2.5% of continuation) are ever seen; 10% practice dose could at best approximately double them. Active-frame fractions and finished episodes differ because of terminations, so these are order-of-magnitude estimates, not measured exact counts. The governor begins with a **1cm maximum**, each active episode samples only 30–100% of that. A 2cm/4cm independent exam is mostly **held-out out-of-distribution** until the governor grows terrain height, regardless of total steps.
- **Important future bottleneck:** new directional window governor first grows exposure 0.005 per sufficiently successful batch until 25%, then height only **+0.00025m = 0.025cm per successful batch**. Thus from 1cm to 2cm requires 40 eligible height decisions *after* reaching the exposure gate. Longer training may be needed, but the ramp is also algorithmically very conservative, not necessarily a sign of policy inability. At 1cm, the scarce supported near-lip reaching bonus can pay once (up to 2.4 foot points) but does not directly reward the FIRST supported landing on top; +10 crossing still needs stable far-side support. Event-specific foot-on-top and weight-transfer diagnostics/reward hypotheses may be needed before changing architecture.
- **After this run:** check the last terrain height/exposure, completed ascent attempts and successes, `terrain_window_*` and ascent guidance, near-lip foot contact and ToF. Do not conclude '34M insufficient' from raw steps alone. Consider an event-based phase-ablation A/B and first-foot-on-top witness to distinguish clock bias, uninformative sensor, servo reach, reward gaps, and mere insufficient practice. Never silently migrate checkpoint source; no code changes to in-progress run.
### 2026-10-24 — new serialized-governor 34M GPU campaign complete (`results_bundle(8).zip`)

Source SHA **`411b841de08ebfbb65c29b357c8ee0d39d577dd1`** (clean detached WSL checkout `~/emu-v5023-window`), training seed **733**, **2,048 worlds**, actor **terrain_360**. Lineage foundation **20,054,016** → final growth **34,013,184** environment steps. Training directory **`~/robot_lab/emu_runs/v5023_window_fresh733/growth34`**; final checkpoint **`checkpoints/final.pkl`** hash SHA256 **`9da0d47c6ced78e2d74ed3132ea0d05280255368cc0f519ff6b1ba2938d420f4`**. Saved `run_meta.json` confirms source clean and parent 20M resume. Uploaded ZIP does NOT contain checkpoint bytes or browser actor; cannot inspect the policy itself or verify cold-start foundation artifact from this bundle.

**Governor finding confirmed:** exposure grew from **5.00%** to **8.3748%** (new practice policy did operate), but maximum *train-time* terrain height stayed **0.010m (1cm)**. Across the **22 sampled PPO-rollout CSV rows** (NOT an exhaustive tally of all training updates), **63 up attempts / 7 supported successes (~11.1%), versus 61 down attempts / 34 successes (~55.7%)**. `terrain_event` in these snapshots = 21 `terrain_collect_directional_evidence` and 1 `terrain_retreat_failed_ascent`; other interim decisions occurred between rows (the dose changes prove it). End-of-run serialized terrain evidence window **up 17 attempts / 0 passes; down 16 attempts / 7 passes**, so evidence remains sparse and ascent is far from 50% mastery. The new governor *increased the dose*, did **NOT** confer larger obstacle competence.

**Full held-out conventional fixed evaluation:** `eval_broad_score` **+0.6410**, core survival **100%**, dynamic80 **97.22%**, combo70 **99.44%**, resilience75 **72.85%**, growth110 **74.22%**, growthcombo110 **95.85%**. Previous source-19d2bc7 34M broad +0.630 and original 0ba009f +0.638, so this is modest, one-seed, non-terrain improvement; gait/core preserved, not a proven successor. Early→late training reward **+2.739→+2.834**, fall fraction **9.38%→4.88%** (logged samples). Shaft drive power ~**8.745 simulated W**, servo over-rated fraction **56.59%** in late samples (NOT measured battery watts). Capability field snapshot ~198 known nodes / **1 competent** and ~1521 known edges / 0 competent: competence calibration/frontier still weak.

**Interpretation/action:** The new 34M experiment confirms the previous one-rollout evidence-starvation fix now increases terrain opportunity; however actual uphill physical crossing training remains only ~11% for sub-1cm obstacles. Any claims of improved stair climbing require independent **v2 frozen paired-height 1701 and 1702 exams** of this new 34M policy. Do not blindly assume another 26M steps will teach 2–4cm while 1cm mastery is still poor: governor only grows height after both-direction >=50% and high exposure. Next instrument 4x4 ToF lip visibility, supported first-foot landing, per-direction contact/foot clearance, and local phase/gait control, then decide whether the same-source 34→46/60M continuation, a careful reward milestone, or curriculum adjustment is justified. Do not touch `main`, champion, original 0ba009f or 19d2bc7 runs. All new code changes must be on branch separate from pinned `411b841` WSL checkout.

**Owner's next exact *read-only* WSL commands**, using the same pinned checkout; outputs are unique under this campaign:

~~~bash
cd ~/emu-v5023-window && source .venv/bin/activate
ROOT="$HOME/robot_lab/emu_runs/v5023_window_fresh733"
emu export-policy "$ROOT/growth34/checkpoints/final.pkl" "$ROOT/growth34/growth34.emupolicy"
emu terrain-exam --policy "$ROOT/growth34/growth34.emupolicy" --output "$ROOT/exam34_seed1701" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1701
emu terrain-exam --policy "$ROOT/growth34/growth34.emupolicy" --output "$ROOT/exam34_seed1702" --device cpu --worlds 8 --horizon 350 --speed .15 --heights-cm 0 1 2 4 --seed 1702
~~~

Ask owner to upload those two `terrain_exam.json` files; use alternate **new** exam output names if any directory already exists (`terrain-exam` intentionally refuses overwrite). Historical old/new 34M seed-1701/1702 exams are recorded above. Source `411b841` code has passed full CI run 40041202515, but this uploaded bundle has no independent terrain exam.
### 2026-10-24 — INDEPENDENT FROZEN v2 EXAMS FOR THIRD 34M POLICY (new governor)

Owner supplied `terrain_exam(6).json`/`terrain_exam(4).md` = seed **1701**, and `terrain_exam(7).json`/`terrain_exam(5).md` = seed **1702**. Both use **same new actor** SHA256 **`e05264e3ec471bdb8c7798f366fa6dc483bac4a9b3435b0e86ec638882efd7f3`**, training source **`411b841de08ebfbb65c29b357c8ee0d39d577dd1`**, schema `terrain_360`; frozen CPU evaluator physically paired up/down per-height worlds, 8 worlds, 350 frames at +0.15m/s, real support required >=5 consecutive 20ms frames after root >10cm beyond edge and within ±27cm lateral, plus full 7s survival. Do not mistake max-height *ceiling* for actual sampled obstacle heights.

| Frozen policy + seed | up passes 1cm / 2cm / 4cm | DOWN passes 1cm / 2cm / 4cm | all UP | all DOWN | terrain survivals (1,2,4cm) | flat mean progress |
|---|---|---|---|---|---|---|
| previous 19d2bc7 policy, 1701 | 2/4,0/4,0/4 | 4/4,4/4,2/4 | **2/12** | **10/12** | 8/8,8/8,6/8 | +0.8540m |
| NEW 411b841 policy, 1701 | 2/4,**1/4**,0/4 | 4/4,4/4,2/4 | **3/12** | **10/12** | 8/8,8/8,6/8 | +0.7842m |
| previous 19d2bc7 policy, 1702 | 2/4,0/4,0/4 | 4/4,4/4,2/4 | **2/12** | **10/12** | 8/8,8/8,6/8 | +0.8928m |
| NEW 411b841 policy, 1702 | 2/4,0/4,0/4 | 4/4,4/4,**1/4** | **2/12** | **9/12** | 8/8,8/8,5/8 | +0.8219m |

**Genuine but tiny improvement:** NEW source-411b841 policy crossed a physically sampled **0.73cm upward step** in the 2cm-ceiling panel, seed1701 world0, first pass at frame271 (~5.4s), 7s full survival and real far-side platform-foot contact. All earlier 34M policies had 0/4 upward crossings on that same seed/height panel. Old 19d2bc7 brain highest verified ascent across these 2 seeds was **0.67cm**, new **0.73cm**. Combined two-seed UP crossings **4/24 old → 5/24 new**; DOWN **20/24 old → 19/24 new**. No statistically compelling general skill win; 1cm+ actual ascents all fail. At NEW seed1702, 0.69cm up reaches root far-side (progress +0.3637m) but fails required sustained upper-platform support despite staying upright; the report's `reached_far_side` is only 'ever root >10cm past lip', NOT foot placement. At 4cm ceiling NEW seed1702 downhill suffers **3 falls / 1 crossing** vs prior 2 falls / 2 crossings. Full flat control 8/8 alive in both seeds; dormant-platform parity within millimetres.

**Efficiency regression:** same-seed old vs new flat progress seed1701 **0.8540→0.7842m**, seed1702 **0.8928→0.8219m**; mean **0.8734→0.8031m** (-8%). Mean positive servo shaft work ~**52.6→60.4J per 7s** (+15%), cost ~**60→75J/m** (+25%). These are mechanical shaft-work proxies, NOT battery energy. Cross-seed small-sample comparison, but reproducible across both matched seeds. Do not automatically promote the highest ascent policy as the new walking champion.

**Next engineering hypothesis (do not edit existing pinned training checkout):** physically annotate near-lip events in a DIAGNOSTICS-ONLY frozen examiner, using available real per-foot MJX `terrain_platform_contact`, `floor_contact` and root relative lip, to distinguish (a) saw/approached obstacle, (b) lifted foot over top, (c) FIRST supported upper-foot touchdown, (d) weight transfer / root far side, (e) stable >=100ms upper support / uninterrupted survival. Current JSON only reports root far-side and final sustained far-side contact; cannot tell *why* 0.69cm root-far-side trial failed. Compare old and new frozen policy under identical seeds before introducing another training reward. Keep actor interface, core reward, governor, main and champion unchanged while checking mechanics. Given only +1 UP out of 24 and decreased efficiency, do not promise 46/60M will fix climbing; if choosing same-source continuation, observe physical skill and efficiency checkpoints at 46M before longer expansion. Owner has not requested a fourth training launch yet.
## 8. Completion and handover discipline

**Current status:** original source `0ba009f`, newer source `19d2bc7`, and serialized-governor source `411b841` have each completed separate 20M→34M GPU campaigns, and the new `411b841` actor now has **both frozen seed-1701/1702 exams complete**. Only one additional up-crossing (0.73cm) gained across both seeds vs `19d2bc7`, while flat distance/shaft-efficiency declined and one down-crossing was lost. Exact matched v2 exam confirms **2/12 upward crossings in BOTH old and new 34M brains on seed 1701**; new seed 1702 also 2/12. All successes are actual steps <=0.67cm; no proven improvement in full upward crossing. New brain has better flat progress but somewhat weaker 4cm downhill survival. On source `411b841`, the multi-rollout directional governor/practice-dose fix now has a completed 34M GPU run; exposure reached 8.37%, height still 1cm, frozen 1701/1702 independent exams completed, marginal +1/24 ascent and efficiency downside. Main/champion remain untouched. Next is an optional controlled fresh campaign after inspecting ToF/lip mechanics, never an unapproved main merge.

**Carry-over checklist:**
- [x] Save the mission/architecture, protected branches, pinned source, both checkpoint paths/hashes and exact results.
- [x] Validate real CUDA 360 actor/2,048 worlds; finish cold foundation and mature continuation.
- [x] Frozen independent terrain A/B at both 20M and 34M with original/dormant physics parity.
- [x] Direction-separated up/down evidence and curriculum gates (implemented; GPU verification pending).
- [x] Bounded ascent guidance and small-step confirmed up-crossings (only <=0.67cm; NOT improved vs old matched cohort).
- [x] Matched-height multiple-seed independent physical exam; flat physics parity and 8/8 survival. Larger ascent ability NOT proven.
- [x] Source-pinned 20M→34M new-brain campaign at `19d2bc7`; fresh campaign of new governor code NOT yet run.
- [ ] Mixed actual contact slopes and finite terrain, servo/efficiency and browser 360 parity.
- [ ] Independent broad comparison and **explicit owner approval** before PR merge or champion/showcase changes.

**New-chat first instruction:** “Read `HANDOVER.md` and `docs/CAPABILITY_GROWTH.md` on branch `feature/v5023-capability-growth`. Preserve `main` and the Pinnacle champion. Verify the current branch HEAD, read the physical up/down code, and begin the direction-aware ascent teaching fix. The owner's trained 20M/34M checkpoints are pinned to `0ba009f`; do not casually `git pull` that WSL checkout or bypass provenance.”

**Every future session:** after any change, update **this very file**: newest evidence, blockers, exact next task, any new run path/commit/CI result; move long narratives into the dated charter. Keep it short, factual and reproducible.
