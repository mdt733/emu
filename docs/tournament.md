# Experiment lab — v5015 and historical tournaments

## Current tinyEmu GPU comparison

**The executable guide is [tinyEmu GPU campaign](TINYEMU_GPU_CAMPAIGN.md).** Keep training commands, WSL install steps and run-directory conventions in that single guide rather than duplicating procedures here. The accepted `main` is the working reference; this draft branch must pass its own WSL GPU acceptance before the longer campaign.

| Stage | Cumulative policy steps | Additional training | Decision |
|---|---:|---:|---|
| Foundation | 20M | 20M once | Review handoff and physical telemetry |
| Four controlled forks | 80M each | 4 × 60M | Compare all arms on common held-out panels |
| Optional finalists | 100M each | 2 × 20M | Only after round-1 review |
| Optional winner | 120M | 20M | Only after further review |

The first comparison costs **260M aggregate environment steps**; the complete optional funnel costs **320M** before rounding to PPO update boundaries. All branches inherit the **same protected, validated 20M checkpoint**, but occupy distinct run directories and have separate checkpoint SHA-256 provenance.

The four hypotheses are intentionally narrow: `baseline` (10% graph teacher); `shadow` (teacher inactive); `slope` (gravity-equivalent terrain-domain practice); and `pitch15` (15° requested chassis-pitch ceiling). See [the human-editable recipe](../emu/ladder_recipe.py) for the **exact named TrainingConfig overrides**. PPO, reward coefficients, actor interface and robot XML are not changed between arms.

The existing `emu ladder` / `emu lab5015` runner is the only ladder owner. Its `foundation`, `auto --stop-after round1`, `round2`, `round3` and `score` paths preserve source, parent, config and evaluated-checkpoint identity. The default human review boundary is **after round1**, before automatically investing in finalists.

### Interpreting the scoreboard

Every 80M policy faces the same independent v5015 15° command/recovery/growth/gentle-slope panels. Survival, command tracking, posture and height contribute to the existing promotion score; the lower-trust step-lift/support measures are visible but **not calibrated gait-certification gates**. New servo-load, shaft-power and spring-energy telemetry is currently recorded from **training rollouts**, which vary by arm and therefore must **not** be treated as a directly comparable held-out efficiency exam or added silently to promotion.

A single-seed 80M sweep selects **promising hypotheses**, not scientific winners. We still need completed-stride/displacement evidence and additional seeds. Reported crouching near 100–200M per lineage remains a separate later validation problem. Never publish a candidate simply because its score is high or the public viewer looks good.

See also [evaluation contracts](evaluation.md), [validation gates](validation.md), [current status](STATUS.md), and [historical learning notes](history.md).

---

## Archived experiment: v5013 capability-teacher tournament

The v5013 tournament compared graph-teacher doses, node/edge/recovery mechanisms and staged teacher schedules. Seed confirmation was mixed; it did **not** prove one universal winning teaching mechanism. The original source, ranking rules, prompts and runner remain inspectable in [Git history](https://github.com/mdt733/emu/blob/b0f31d865198f451414ec0d6e69e1793c69478a7/emu/tournament.py), and its actual evidence is preserved in [results/v5013_tournament](../results/v5013_tournament/README.md). Key lessons are retained in [history.md](history.md).

Historical `emu tournament` commands in older docs do **not** exist in the current executable. Use a separate historical Git checkout for exact reproduction; do not point it at current run roots.
