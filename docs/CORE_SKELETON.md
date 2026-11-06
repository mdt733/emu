# EMU — live system skeleton and ownership audit

This is a **map of the current working system**, not an experimental
diary. The released v5032 260M 360-input policy on `main` is preserved.
The v5048 50M-parent / eight-fork training campaign runs on its **own
frozen source SHA**; the v5049 release-preparation branch does not
retrain or silently migrate its checkpoints.

## One system, seven responsibilities

```text
CLI: emu train / emu prism-exam / emu export-policy / emu viewer
                    │
      configuration + Git provenance + checkpoints
                    │
        six-servo robot.xml and true MJX world
         ┌──────────┼──────────┐
         │          │          │
     2×2 carpet    sensors    birth cohorts
    8 prisms      360 inputs  six specialists
         └──────────┼──────────┘
                    │
       ONE actor / ONE optimiser / PPO
          reward + curriculum + health
                    │
        source-pinned saved checkpoint
                 ┌──┴──────┐
       frozen independent   inference-only
       contact examiner     .emupolicy → browser
```

The browser is a separate MuJoCo/WASM simulation of the released actor
and representative park; it is **not** a substitute for the independent
MJX exam of the actual terrain curriculum. There is no scripted
climbing gait, second network, cloud evaluator or privileged terrain
geometry in the deployable sensor observations.

## Owners of the 27 live Python modules

| Owner | Files under `emu/` |
|---|---|
| Entry point, commands and saved configuration | `__init__.py`, `cli.py`, `run.py`, `training_config.py` |
| Real physics and sensor geometry | `contracts.py`, `environment.py`, `terrain_carpet_mjx.py`, `carpet_prism_mjx.py`, `domain_randomization.py`, `directional.py` |
| One PPO policy and teaching | `training.py`, `ppo.py`, `rewards.py`, `curriculum.py`, `capability_focus.py`, `capability_field.py`, `training_health.py`, `step_goal.py` |
| Checkpoint, metrics and physical examinations | `checkpoint_io.py`, `results.py`, `evaluation.py`, `prism_exam.py`, `step_autopsy.py` |
| Actor delivery and viewer | `policy_export.py`, `viewer_server.py`, `capability_viewer.py` |
| Hardware/system diagnosis | `doctor.py` |

The physical robot is `emu/robot.xml`; the browser lives in
`emu/viewer_src/`. The [read-only source-ownership inventory](../scripts/audit_core_dependencies.py) checks that all 27 modules
have exactly one declared owner and no retired imports reappear. Run
it to get live counts, instead of trusting an old numbered file list:

```bash
python scripts/audit_core_dependencies.py --json
```

### Actual terrain and observation invariants

- **Four** recyclable 75cm tiles in a 2×2 midpoint-controlled window,
  each divided into **two** independently physical convex triangular
  prism mesh colliders: eight terrain contact solids, not 18.
- Four non-colliding 2×2-sample heightfields are source-height samplers,
  not alternative foot-support collisions. Independent frozen
  `prism_exam.py` checks real foot-on-physical-prism support.
- All six specialist cohorts use this **same** JAX/MJX terrain
  generator. Four frames × 90 deployable history fields = **360 actor
  inputs**. The policy outputs six servo targets.
- The feet-only solver saves contact work, while torso/leg penetration
  is watched by virtual hard-strike safety probes. These are numerical
  approximations that must pass physical plausibility checks.
- Newton/linesearch `2/4` is an experimental measured performance
  option, **not** a certified hardware-fidelity default. Physical
  penetration, step support and CUDA compact/full reset parity remain
  release-review gates.
- An independent frozen exam must use the **actor's exported solver
  identity**, actual 75cm terrain corners and source-pinned sensor
  contract. The retired `(num_envs,9,4)` frozen-trace bug was fixed
  in v5048 and caught before full training.

## Active paths and outputs

| Responsibility | Stable entry / owner |
|---|---|
| Install and diagnostics | `scripts/setup_wsl.sh`, `emu doctor`, [INSTALL](INSTALL.md) |
| One source-pinned training run | `emu train --run-dir ~/emu-data/runs/<study>-s<seed>/` |
| Checkpoint and recovery | `emu/checkpoint_io.py`, `emu/run.py`; same SHA/config is mandatory |
| Physical reproducibility | `scripts/probe_carpet_prism_mjx.py`, `emu prism-exam` |
| GPU acceptance | `scripts/smoke_gpu_carpet.py`, `scripts/smoke_training_lifecycle.py` |
| Metrics and research findings | `emu report`, [v5046 optimisation results](V5046_OVERNIGHT_RESULTS_20261105.md), [issue #17](https://github.com/mdt733/emu/issues/17) |
| Protected public model | `showcase/champion.emupolicy`, `showcase/release_approval.json` |
| Release verification | `scripts/verify_showcase_release.py`, [RELEASE_READINESS](RELEASE_READINESS.md) |
| CI and publishing | `.github/workflows/emu-checks.yml`, `emu-playground.yml` |

Every training run owns its config, run_meta, console, metrics,
checkpoints, frozen exams and exported actors in **one run directory**.
Scratch experiments go under `~/emu-data/scratch/`. Never create
additional working roots or hide trained data in the Git checkout.

## Core release acceptance (not a claim of learned competence)

1. `bash scripts/check.sh --full` must pass **on the exact release
   candidate SHA**, including browser checks.
2. `python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2`
   must pass genuine MJX collider / recycle checks.
3. On the operator's CUDA GPU, verify the same source with 2048-world
   full-PPO smoke, checkpoint continue/verify, actor export, and
   actual frozen step exam. CPU-only CI does not prove CUDA physics
   fidelity or locomotion.
4. Separately examine one frozen 200M candidate: support on both up
   and down steps, post-lip controlled progress, still/slow drift,
   fast achieved speed and useful stride, steering, falls and J/m.
   Compare against the old main champion on **equivalent missions**,
   not merely one reward or procedural terrain target.
5. Keep the released 260M policy and approval JSON **byte-for-byte
   unchanged** until explicit owner-reviewed model promotion. Code
   merge and public GitHub Pages deployment are different approvals.

Full gates, measured 260M reference and rollback are in
[RELEASE_READINESS](RELEASE_READINESS.md),
[V5032_PINNACLE_RELEASE](V5032_PINNACLE_RELEASE.md) and
[PHYSICAL_BASELINE_260M](PHYSICAL_BASELINE_260M.md).

## Retirement rule

There is one teacher (`curriculum.py`), one reward owner
(`rewards.py`), one physical generator and one PPO. A temporary
research script is deleted **after its measurements are written to
the permanent results/issue**. The v5045 140M, v5046 20M and v5048
eight-arm launchers are preserved by the corresponding historical
source commits but intentionally **absent from this future-main
preparation branch**. The v5045-only duplicate CI workflow and
obsolete runbook are likewise retired.

The detailed original [v5045 cleanup audit](https://github.com/mdt733/emu/blob/feature/v5045-core-skeleton-audit/docs/CORE_SKELETON.md)
remains in Git history, including its deleted-file accounting.
Current architecture should be understandable without importing
the historical experiment scaffolding.
