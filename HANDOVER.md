# EMU engineering handover — lean prism v5045

**4 November 2026.** The earlier v5044 140M experiment was manually
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

- Over **23k deleted lines** vs frozen v5044, the historic tests reduced
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
   after JIT. First measured v5045 GPU benchmark: **7,536.1
world-steps/s** on two steady 2,048-world updates (+0.6%; not a
statistically convincing speedup), peak 1,818,866,944 device bytes.
Next required: exact-source 8-world checkpoint/resume/export and
frozen-actor true-prism exam smoke.
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

## 4 November — make the physical control frame understandable

- `EnvState` was **59 fields, now 52**. The retired single-slab
  approach counters and ascent peaks never changed in real-prism training;
  they no longer occupy an MJX state. Reset populates fields by name and
  per-frame updates use `state._replace(...)` with named live changes,
  rather than counting 59 positional constructor arguments.
- The only supported `params_array` always packs the full terrain/six-role
  teacher packet. Frozen exams and the trainer both construct the real
  18-prism model; `evaluation._world` rejects other geometry before JIT.
- The environment has no alternate rotated-ground plane, tripod/220-input
  sensor contract, or dead standalone finite-step platform dimensions.
  Robot and hardware model randomization are preserved. Exact 4×4 ToF
  acquisition, servo delivery, physical prism contacts and supported
  terrain events remain the active simulation.
- Removed the inactive v5031 single-lip careful-approach/crouch reward
  and unused offline counterfactual ascent diagnostics from `rewards.py`.
  The actual verified-prism crossing payment and the optional 25-frame
  forward tracking blend remain. Retired config fields were removed from
  `training.py` and `training_config.py`; this is a **new source-locked
  checkpoint lineage** and old branch checkpoints cannot be resumed.
- The 438-line `_step_impl` is now about 224 lines. Its reporting dictionary
  and physical slope/STEP episode grading moved into `_step_report`,
  which reads the pre-auto-reset episode accumulator and the same physical
  contact facts. A byte-for-byte comparison confirms the 8,032-character
  reporting dictionary and reward invocation were preserved.
- The existing four-world prism probe now requires finite, world-shaped
  rewards, episodes, signed-step, slope and stance evidence. Two more
  never-set Academy flags were removed; state construction now requires
  all 52 fields, without optional None defaults.
- Remaining large concepts: `_step_report` still collects substantial
  telemetry, `curriculum.py` still has a long
  evidence governor, and `results.py` still records legacy metric names.
  Consolidate only with fresh physical equivalence proof; do not change
  reward economics or teacher thresholds as part of cosmetic refactoring.
- On any latest commit, require core Python/JS and 4-world MJX CI green,
  then same-SHA CUDA first/continue/verify plus exported frozen-actor
  prism examination **before** starting a fresh 140M run.
