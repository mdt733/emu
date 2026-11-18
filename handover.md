# EMU H40 — 140M development handover

**Goal:** one six-servo, 360-input dynamically unstable biped learns useful commanded walking, resilience, slopes and both step directions *without forgetting earlier skills*.

## The promoted baseline

- **Original 140M learner:** source `22266821bdbfc82a529ef708e5171c82c6c9f29c`, training seed `735`, actual step `140,050,432`, complete PPO/normaliser/curriculum state at `~/emu-data/runs/h40-holistic-cold-s735/stages/140m/final.pkl`. Its signed `stage_manifest.json` and exported `actor.emupolicy` remain with it **outside Git**.
- **Maintained code:** `main` is the curated source-equivalent Spring V1 training system up to 140M, derived from PR #96. Core robot, observations, PPO and rewards retain the original 140M-producing implementations. This integration commit is *not* the original checkpoint source SHA.
- **Profile:** `EMU_PHYSICAL_PROFILE=h40-spring-v1` (150 N/m effective knee strut, revised anchors and ROM, original six rated ±0.980665 N·m motor caps). Keep separate from historical `h40` physical model. Real spring cartridge geometry and hardware performance are unvalidated.
- **Public website:** `showcase/champion.emupolicy` remains the older approved 120M actor. New source promotion does **not** silently publish a new actor or change release approval.

## How the 140M learner was built

| World steps | Lesson |
| --- | --- |
| 0–20M | Walking and balance with gait guidance |
| 20–40M | Fade gait guidance to zero |
| 40–120M | Outcomes: commands, slopes, recovery, uphill/downhill practice |
| 120–140M | Step commitment: approach a lip, attempt progress, allow temporary slowdown |

One actor, one optimizer, 2,048 MJX environments, no warm-start actor imported. The canonical runner **stops at 140M**. The source still describes the original after-140M transfer phase for historical parity; it is **not** a supported mainline continuation.

## Physical evidence

Three *evaluation* seeds (1701/1711/1729; only one training seed, 735) yielded means: **57.4%** flat lane control, **50.5%** uphill 44–60mm, **19.1%** uphill 64–80mm, **71.6%** downhill 44–80mm; essentially 100% flat survival. Full sequential multi-step reliability is not demonstrated.

The historical 160M actor lost flat directional control to **7.2%**; the later 175M actor gained hard-uphill capability (**43.6%**) but flat control fell to **12.3%**. The remaining challenge is **retaining the old skills while learning new ones**, not blindly increasing run length. These numbers are from source-pinned first-life true-prism independent exams, not PPO rewards.

## Reproduce, evaluate and extend

~~~bash
# Maintained mainline: plan is default, no checkpoint opened.
bash scripts/run_h40_holistic_cold_wsl.sh plan
# New random seed and isolated output directory:
EMU_COLD_SEED=736 bash scripts/run_h40_holistic_cold_wsl.sh all
# Original trained development actor (local only):
EMU_PHYSICAL_PROFILE=h40-spring-v1 emu viewer   ~/emu-data/runs/h40-holistic-cold-s735/stages/140m/actor.emupolicy --no-open
~~~

The new default run path is `~/emu-data/runs/h40-main-140m-s<seed>/`; never overwrite the original source-pinned run. Continue from the 140M full PPO checkpoint only on a **single isolated research branch** with explicit cross-source checksum/source authentication. For new curriculum proposals, measure flat/UP/DOWN/command/slope/resilience; no promotion by reward or frontier. Reproduce the 140M recipe across other *training* seeds before claiming robust training reproducibility.

[README](README.md) · [Core skeleton](docs/CORE_SKELETON.md) · [Measured findings](docs/RESEARCH_FINDINGS.md)
