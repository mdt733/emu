# EMU documentation — active core only

**Current code, not a chronological research scrapbook.**

1. [**Run → physics → PPO → exam/export → browser: core skeleton**](CORE_SKELETON.md).
2. [**v5045 lean-prism training and GPU acceptance**](V5045_LEAN_PRISM_RUNBOOK.md).
3. [**v5044 historical 140M recipe**](V5044_PRISM_140M_COLDSTART.md) and [**v5043 real-prism baseline**](V5043_MJX_PRISM_CARPET.md).
4. [**Latest engineering handover**](../HANDOVER.md).
5. [**Windows/WSL installation**](INSTALL.md).

## Current reference

| File | Why it stays |
|---|---|
| [v5041 exposure postmortem](V5041_140M_TERRAIN_EXPOSURE_POSTMORTEM.md) | Prevents the exposure-probability × difficulty bug recurring |
| [Six-specialist curriculum](UNIFIED_CAPABILITY_GROWTH.md) | Teaches the one actor through six focus cohorts |
| [Physical quality](PHYSICAL_QUALITY.md) / [capability dimensions](CAPABILITY_DIMENSIONS.md) | Measurable locomotion and resilience criteria |
| [v5032 released 260M reference](V5032_PINNACLE_RELEASE.md) / [physical baseline](PHYSICAL_BASELINE_260M.md) | Public baseline against which the new policy must be judged |
| [Release readiness](RELEASE_READINESS.md) | Requires independent supported crossings, command tracking, survival |
| [Protected playground](PLAYGROUND.md) / [graduated park](VIEWER_GRADUATED_PARK.md) | Current 360-input browser deployment and diagnostics |
| [Maintenance](MAINTENANCE.md) | Source identity, branch and checkpoint safety |

## Old experiments

**Removed from this branch deliberately.** No historical experiment,
three-platform curriculum, old paper geometry design or 50KB handover
needs to live in the working tree. They remain in the exact immutable
[pre-cleanup v5044 Git tree](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/history/README.md),
along with its [full handover](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/HANDOVER.md),
[historical architecture](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/ARCHITECTURE.md)
and [older nine-box trainer instructions](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/V5041_GPU_CARPET_TRAINING.md).

Use the original source SHA and full archived checkpoint to reproduce
a historical result. Never use the cleaned source to interpret an old
optimizer state by shape coincidence.
