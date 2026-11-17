# EMU maintenance — one supported system

These rules keep EMU understandable as a small-team project. Git history and result ZIPs are the archive; current source code is **not** an anthology of every research variant.

## Clear ownership

- `robot.xml` and `physical_variants.py` define H40 geometry and mechanics. `contracts.py` fixes **one 360-input, six-output actor** and ToF geometry.
- `environment.py` owns MJX world physics and physical event measurements. Terrain meshes stay in `terrain_carpet_mjx.py` / `carpet_prism_mjx.py`; not duplicate box approximations.
- `ppo.py` owns one actor/critic optimiser. `rewards.py` pays physical outcomes; `capability_focus.py` / `curriculum.py` choose age stages and competency-based challenges. Six specialists are lesson allocations, **not six networks**.
- `run.py`, `training.py` and `checkpoint_io.py` protect source-pinned PPO training/resume. `evaluation.py`, `prism_exam.py`, matrix exam and result summariser read frozen policies without learning.
- `policy_export.py` and `viewer_src/` serve only a validated actor. `console_dashboard.py` is read-only.

## Rules for changes

1. **One focused development branch per experiment**, not a branch per reward weight; small commits are fine. Preserve `main` until capability is demonstrated.
2. **Never edit code underneath a running checkpoint's source SHA** or silently resume a different physical model or sensor contract. A cleaned project has a different commit identity even if training-critical files are identical.
3. **One canonical cold command**: `bash scripts/run_h40_holistic_cold_wsl.sh all`. It starts random weights, not any archived actor. Keep one read-only exam and viewer workflow, not versioned repair/tutor launchers.
4. **Delete historical experiment files from the proposed main working tree** after carrying forward only durable science and useful tests. Retain the completed PRs, SHA-pinned runs and ZIP evidence in Git/results storage.
5. **Test the scientific boundaries**: 360 observation contract, rated motor/spring physics, real MJX top-contact detection, two-foot transfer, signed downhill as well as uphill, no phase reward after fade, PPO lifecycle, clean resume, examiner invariance, browser policy export. CI and a CUDA PPO smoke validate software; independent fixed trials and a second seed support competence/reproducibility.
6. **No automatic source or showcase promotion.** A new `main` needs user review and preserved evidence. Public GitHub Pages has a separate strict approval file and workflow.
7. **Avoid documentation sprawl.** Keep one short README, architecture, installer, physical-quality spec, viewer guide, release guide, one condensed research findings page, docs index and a concise `handover.md`. Historical roadmaps belong in Git/issues, not duplicate source files.

## Release versus physical hardware

The simulated H40 has six chassis-mounted servo motors, knee actuation through a modelled joint but an incompletely measured transmission, rated torque caps and provisional carbon limbs/springs. **There is no complete validated CAD/parts BOM/wiring/calibration package.** This repository can reproduce a *simulated* robot, policy and examiner; physical rebuild instructions will need hardware drawings and measurements rather than invented dimensions.

For development: `bash scripts/check.sh --quick` or `bash scripts/check.sh --full`; check `emu doctor --require-gpu` and a real 2,048-world GPU PPO smoke before a long cold train.

Handover: [../handover.md](../handover.md) · [Architecture](CORE_SKELETON.md) · [Physical quality](PHYSICAL_QUALITY.md) · [Research evidence](RESEARCH_FINDINGS.md).
