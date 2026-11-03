# EMU engineering handover — lean prism v5045

**3 November 2026.** The earlier v5044 140M experiment was manually
stopped; its source commit and checkpoints remain historical. New work
lives only in draft PR #34 on `feature/v5045-core-skeleton-audit`.
`main` and the published 260M champion are unchanged.

## One current model

- Six-servo dynamically unstable EMU; **one** 360-input 4×4-ToF
  array-only actor, critic and PPO optimizer, 2,048-world GPU target.
- **One** 3×3 infinite logical carpet: 18 physically colliding
  triangular prisms, nine noncolliding 2×2 vertex samples, no old
  two-slab/nine-box/three-platform collision bodies.
- First 20M: cold ordinary walking. 20–30M: gradual six-specialist
  exposure. Afterward the governor updates physical command, slope,
  up/down step, recovery and transition frontiers from completed
  episode evidence, not the calendar.
- STEP Academy is another birth cohort in that same carpet and PPO;
  the CONTROL role practices useful low-cost walking, not zero-motion
  energy farming. Teacher geometry never becomes policy input.

## Cleanup completed

- Over **21k deleted lines** vs frozen v5044, the historic tests reduced
  to 18 Python and six browser maintained contracts. Historic science
  stays in old Git SHAs rather than runtime compatibility branches.
- `environment.py` is prism-only; zero extra slab/box bodies.
  `ppo.py` no longer emits old slab and box exam traces.
- Obsolete physical exam, physical-quality, old commands and duplicate
  prism scorer retired. `emu prism-exam` runs one frozen
  true-prism actor and scores raw supported signed up/down crossings,
  actual terrain exposure, velocity, survival and shaft work.
- `policy_export.py`, `viewer_server.py` and
  `capability_viewer.py` separate actor, local HTTP and field UI.

## Current release/launch gates

1. GitHub 24-test + browser-build + four-world MJX physical probe **must
   pass on the exact final source SHA**.
2. On the RTX 4060 Ti, repeat the matched **2,048-world, 3-update
   full-PPO benchmark**. Old prism reference: 7,492 world-steps/s
   after JIT. No new measured v5045 speed yet.
3. Same-source CUDA two-update `first/continue/verify` checkpoint,
   normalizer, optimizer and exported actor acceptance.
4. Only then start a **new empty** 140M run root. Never resume old
   v5044 state with v5045 code. Frozen-policy real-prism examination
   later validates walking, not the mere presence of PPO gradients.

**[New v5045 runbook](docs/V5045_LEAN_PRISM_RUNBOOK.md)** has exact
install, check, benchmark, lifecycle and launch commands.
**[Core source skeleton](docs/CORE_SKELETON.md)** explains file ownership
and retirement decisions. The historical full handover remains
[at the pinned v5044 commit](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/HANDOVER.md).
