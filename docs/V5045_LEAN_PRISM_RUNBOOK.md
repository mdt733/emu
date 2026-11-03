# v5045 — lean true-prism CUDA acceptance and cold-start

**Only source branch:** `feature/v5045-core-skeleton-audit` (draft PR #34).
**This replaces the old v5044 launcher in this branch only.** v5044
checkpoints remain intact and MUST NOT be resumed with the v5045
prism-only MJX model. The previous v5044 large run was stopped.

The goal is **one 360-input policy, one PPO, one procedural physical
18-prism carpet and six episode-birth teaching specialties**, with
20M foundation and independent evidence-based command/terrain growth.

## 1. Separate clean WSL checkout

Use the existing working GPU JAX virtual environment; do not install
CPU-only JAX over an existing CUDA environment.

~~~bash
git clone --branch feature/v5045-core-skeleton-audit --single-branch \
  https://github.com/mdt733/emu.git ~/emu-v5045-prism
cd ~/emu-v5045-prism
source ~/emu-graduation/.venv/bin/activate
python -m pip install -e .
git rev-parse HEAD
git status --short
python -c 'import jax; print(jax.__version__, jax.devices())'
python scripts/run_prism_140m.py plan
~~~

Source must be **clean**, on the exact v5045 feature branch and never
silently pulled after starting a checkpoint lineage. The new launcher's
default root is `~/emu_runs/v5045_prism_140m_s735`, not the old v5044
experiment. No GitHub login is needed to clone this public repo.

## 2. Source, physics and PPO preflight

~~~bash
python -m pytest -q tests
npm test --prefix emu/viewer_src
python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2
python scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 \
  --frames 2 --ppo --full-update --benchmark-updates 3 --prisms
~~~

The 2,048-world benchmark performs full PPO, excluding its first compile
from steady throughput. Record **post-JIT world-steps/s**, compilation,
and peak memory separately. The old full-PPO **v5044** reference was
**7,492 world-steps/s** on RTX 4060 Ti. This is a *comparison target*,
not a claimed v5045 result.

These checks prove MJX dynamics, terrain, commands, sensors, gradients
and memory capacity, **not learned walking**.

## 3. Same-source, bounded two-update CUDA checkpoint proof

~~~bash
python scripts/smoke_training_lifecycle.py first \
  --root ~/emu_runs/v5045_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735

python scripts/smoke_training_lifecycle.py continue \
  --root ~/emu_runs/v5045_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735

python scripts/smoke_training_lifecycle.py verify \
  --root ~/emu_runs/v5045_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735
~~~

The first step creates a fresh directory and fails closed if it already
exists. The next two verify source hash, exact sensor contract, optimizer,
normalizer, checkpoint continuity, no-op completed resume and a real
exported actor. If the Git SHA changes, use a **new unique** acceptance
root; never relabel or delete previous evidence.

## 4. Start the clean 140M training lineage

Only after **both** physical/PPO preflight and the bounded lifecycle
have passed on the exact same commit:

~~~bash
python scripts/run_prism_140m.py start
~~~

This runs in the **foreground** (no tmux). At 2,048 worlds, seed 735:
20M main-style gait, 20–30M specialist blend, later dynamic terrain
frontiers and 60–100M Academy ramp. All six specialties share one PPO.
The real model contains **18 active prism colliders**, nine *noncontact*
vertex-source samples, and no old slab or nine-box collision scene.

Run metadata, `metrics.csv`, `fixed_eval.csv`, `results.md`, latest
and rotating checkpoint files are written **outside the repo**. Monitor
whether issued tasks actually encounter physical slopes and signed
steps; a curriculum target is not proof of exposure or mastery.

To examine an interruption without changing weights, use
`python scripts/run_prism_140m.py verify`. On the SAME source SHA, use
`python scripts/run_prism_140m.py resume` to continue. Do not pull a new
cleanup commit into that checkout while training.

## 5. Honest frozen-policy prism examination

When a true-prism actor is available (exported from a matched checkpoint):

~~~bash
python -m emu.cli export-policy \
  ~/emu_runs/v5045_prism_140m_s735/checkpoints/final.pkl \
  ~/emu_runs/v5045_prism_140m_s735/final.emupolicy

python -m emu.cli prism-exam \
  --policy ~/emu_runs/v5045_prism_140m_s735/final.emupolicy \
  --output ~/emu_runs/v5045_prism_140m_s735/prism_exam_mixed \
  --worlds 8 --horizon 400 --device cuda \
  --mode mixed --grade-deg 3 --lip-cm 1 --speed .12
~~~

The exam runs a **frozen actor** on freshly sampled real prisms.
It reports *actually measured* root-triangle grades, command tracking,
real supported new-tile up/down crossings, survival and mechanical
work per forward progress. The `--grade-deg` and `--lip-cm` values
are teacher frontiers, **not guaranteed fixed fixtures**. This is
deliberately **not comparable case-for-case** to the historical two-slab
exam. Use old frozen source for old policy baseline reproducibility.

## Release boundary

Current CI proves only a small four-world physics probe and the
maintained 24-test suite. A fresh 2,048-world benchmark, same-source
8-world first/continue/verify and independent actual trained-policy
examination remain release gates. Keep PR #34 **draft** until all pass.
