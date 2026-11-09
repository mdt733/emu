# EMU maintenance contract

The [core skeleton](CORE_SKELETON.md) owns architecture; the [H40 handover](../handover.md) owns the current baseline and release boundary.

## One authority per concept

- **Hardware:** `robot.xml` defines six powered joints, two passive shins, springs and motor limits; `physical_variants.py` deterministically scales the selected profile.
- **Physics:** `environment.py` steps one true-prism MJX world; `terrain_carpet_mjx.py` and `carpet_prism_mjx.py` own terrain; `domain_randomization.py` owns manufacturing spread.
- **Actor:** `contracts.py` fixes the 360-field observation and six actions; `ppo.py` owns the single actor/critic and optimiser.
- **Teacher:** `rewards.py` pays, `curriculum.py` selects challenge, `capability_focus.py` allocates six cohorts, `capability_field.py` aggregates evidence. There are not six independent networks.
- **Run and evidence:** `run.py`, `training.py`, `checkpoint_io.py`, `evaluation.py`, `prism_exam.py`, `results.py` retain source identity and distinguish learning from held-out physical exams.
- **Viewer:** `policy_export.py`, `viewer_server.py`, and `viewer_src/` deliver inference. The illustrative browser carpet is **not** an exact MJX terrain replay.

## Cleanup rules

1. **One implementation.** Do not leave alternate simulator geometries, duplicate learner paths or experimental flags after an experiment is resolved. Git is the historical archive.
2. **One run identity.** Full checkpoints include optimiser, normalization, teacher/health and source/model/observation identity. A different XML or sensor geometry requires a genuinely new run.
3. **Freeze trained lineage.** H40 120M is the `main` baseline. Preserve its exact source commit `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`, full `final.pkl`, training logs and actor export. Historical v5058/v5059 experiments remain in Git. A new robot/sensor/actuator variant requires a distinct cold start and matched exams, never silent checkpoint reuse.
4. **Tests prove boundaries.** At most 24 core pytest cases, plus real MJX prism checks, GPU PPO/checkpoint smoke and browser tests. Keep tests for failure modes; retire superseded per-version test scaffolding.
5. **Evidence before capability claims.** Owner approval made H40 the software mainline despite unresolved steps. Training reward, curriculum frontiers and viewer animation are not held-out capability. For future improvements, measure physically supported motion, command tracking, torque saturation, solver parity, falls and recovery.
6. **No stale roadmap.** One root `handover.md`, one skeleton, one install page. Completed findings may stay in release records or the Git commit that produced them; don't present experiment runbooks as current.

Quick validation: `bash scripts/check.sh --quick`; full Python/browser checks: `bash scripts/check.sh --full`. Both always validate the checked-in H40 actor. The separate GitHub Pages approval is verified only when `showcase/release_approval.json` exists, and Pages itself **always** requires it. Real CPU/MJX contact CI and operator-side GPU acceptance are separate; a passing unit test is not proof of CUDA or real hardware.

### Current hardware placement contract

**All six servo motors are mounted in the chassis.** The two knee drives
act through an unmodelled transmission. No explicit knee-mounted motors,
motor masses, fake viewer braces or height-dependent chassis translations.
A source change to `robot.xml` changes all generated physical profiles
and demands new checkpoint ancestry. Retain 24 tests plus physical CPU/GPU
acceptance. Nominal h40 ~506g is a measured-value TODO, not a certified
parts BOM.
