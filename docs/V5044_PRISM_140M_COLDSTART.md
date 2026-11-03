# v5044 — 140M real-prism cold-start campaign

**Feature branch:** feature/v5044-prism-140m-coldstart, based on the v5043
real MJX prisms. This is a new source-pinned 140M-step learning lineage;
main, old nine-box runs, policy weights and saved checkpoints are untouched.

## One actual teacher/actor/optimizer

- 2,048 CUDA worlds, 360-input actor, 18 dynamic convex triangular-prism
  collision meshes (2 true physical triangles per 50cm square), 3x3
  recyclable terrain carpet. The old nine tilted-box colliders are inactive
  in this experiment. All six specialists use this same physical world:
  COMMAND, RESILIENCE, SLOPE, STEP, TRANSITION, CONTROL.
- First 20M: quiet main-style walking foundation on near-flat physical
  prism tiles. 20–30M: blend six specialist roles from 0 to 100%.
  From 20M onward use the terrain-early lesson weighting: 16% command,
  9% resilience, 35% slope, 30% step, 5% transition, 5% control before
  Academy/retention adjustments.
- Same-prism multi-lip Academy: 60–80M gradually introduces 0–10% of
  episode births; 80–100M ramps to 50%; 100–140M targets 50% of all
  births. Each height must be earned from physical crossing evidence.
  15cm is a *ceiling*, not a promised achieved step.
- Athletic fast-gait rehearsal 10%, CONTROL retention target 15%,
  main-mature growth, coherent-careful STEP lessons, gentle pushes,
  8-degree pitch cap, transition cap 3 degrees between neighbouring
  smooth triangles, independently signed 2D step offsets.
- PPO checkpointer writes latest.pkl every 20 updates, or about 1.31M
  world steps; rotating snapshots about every 2M. Fixed evaluation every
  10M uses **actual 18-prism physics** (a legacy nine-box evaluator bug
  was fixed on this branch). A final report uses recorded learning evidence.
- One original Git SHA and full immutable configuration across start,
  restart and final. Running new source on an old lineage is refused.

The matched 2048-world benchmark on RTX 4060 Ti measured 7,492.3
post-compilation world-steps/s on 18 prisms versus 17,774.2 on boxes.
Device memory peak was 1.90 GB on prisms. 140M / 7,492.3 gives about
5.19 h of *steady PPO-only throughput*; this does not include compilation,
checkpoint I/O, evaluations, startup, physical exam or rate drift.
Do not mistake it for a guaranteed wall-clock completion time.

## 1 — Install a fresh v5044 feature checkout

~~~bash
git clone --branch feature/v5044-prism-140m-coldstart \
  --single-branch https://github.com/mdt733/emu.git ~/emu-v5044-prism
cd ~/emu-v5044-prism
source ~/emu-graduation/.venv/bin/activate
python -m pip install -e .
python -c "import jax; print(jax.__version__, jax.devices())"
git rev-parse HEAD
git status --short

# Prints the entire resolved 140M configuration; writes nothing
python scripts/run_prism_140m.py plan
~~~

Use the user's already working JAX/CUDA venv. The script refuses to
start from an incorrect editable checkout, dirty tree, non-v5044 branch
or root inside the repo. Once a real run starts, **never git pull in
this checkout**. Existing checkpoints require this exact full SHA.

## 2 — Mandatory bounded REAL prism checkpoint lifecycle

Run these three commands independently, all from the same exact SHA.
They test two PPO updates only (8 worlds by 32 frames each); **they do
not start an overnight run**. This checks the real 18-prism trainer,
checkpoint restoration, source identity, finite physical telemetry,
advancing observation normalization, completed resume no-op, and an
array-only exported actor carrying prism collision provenance.

~~~bash
cd ~/emu-v5044-prism
source ~/emu-graduation/.venv/bin/activate

python scripts/smoke_training_lifecycle.py first \
  --root ~/emu_runs/v5044_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735

python scripts/smoke_training_lifecycle.py continue \
  --root ~/emu_runs/v5044_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735

python scripts/smoke_training_lifecycle.py verify \
  --root ~/emu_runs/v5044_prism_lifecycle_s735 \
  --unified --carpet --prisms --envs 8 --seed 735
~~~

The first stage refuses an existing acceptance directory. If you need
a repeat, use a distinct new acceptance root and pass its path as
--acceptance-root when starting the big run; do not erase previous
evidence. The full launcher requires both validated checkpoints,
the exported 360-input policy with true prism provenance, matching seed
and the exact source SHA before it will start.

## 3 — Start ONE actual cold 140M run

~~~bash
cd ~/emu-v5044-prism
source ~/emu-graduation/.venv/bin/activate
python scripts/run_prism_140m.py start
~~~

This launches ONE foreground CUDA training process using the existing
single-run trainer; no tmux and no background daemon. Its dedicated
default run directory is:

    ~/emu_runs/v5044_prism_140m_s735

The start mode refuses a nonempty run directory. The script enables
the existing approved long-run launch gate, checks all checkpoint and
source preconditions, and prints the exact source SHA and command.
All six specialist cohorts, recovery, real ToF and physical trajectory
metrics remain active. Ordinary terminal interruption is safe after a
validated latest or rotating snapshot has been written.

## 4 — Read-only validation and same-source resume

After an interruption, keep exactly the same source commit and venv:

~~~bash
cd ~/emu-v5044-prism
source ~/emu-graduation/.venv/bin/activate
python scripts/run_prism_140m.py verify
python scripts/run_prism_140m.py resume
~~~

The verify mode checks the config/source/checkpoint without updating
weights. The resume mode restores the SAME optimizer, normalizer,
curriculum and training health to the original 140M target. It does
not quietly fork the nine-box policy. Do not run resume against a
different config or source SHA; those deliberately raise exceptions.
If there is no saved checkpoint yet, resume fails and a NEW empty run
directory is required for a genuine fresh start.

## 5 — Evidence and exported policy

Inspect these files under the 140M experiment directory:

- console.log — canonical uncropped training stdout.
- metrics.csv — PPO, six cohorts, physical step evidence, true birth
  allocation, motor energy, recovery and commanded velocity.
- fixed_eval.csv — frozen held-out evaluations every 10M on real prism
  physics (not the nine-box approximation).
- checkpoints/latest.pkl and rotated snapshots — source-validated
  checkpoint restore points; best.pkl and final.pkl on progression.
- results.md / report outputs on final completion.

Export the trained actor only after final.pkl validates:

~~~bash
cd ~/emu-v5044-prism
python -m emu.cli export-policy \
  ~/emu_runs/v5044_prism_140m_s735/checkpoints/final.pkl \
  ~/emu_runs/v5044_prism_140m_s735/prism_140m_final.emupolicy
~~~

The exported actor records the 360 observation schema and a
terrain_carpet_prisms=true training provenance flag. The original
live browser still approximates terrain visually; it must NOT claim
that its representative geometry is exact per-world MJX prism physics.

## Known limitations

A bounded GPU PPO benchmark and CPU prism physical probes passed on
v5043. This specific v5044 run must still pass its own 8-world
first/continue/verify GPU lifecycle on the user's machine. No 140M
policy has yet trained on these prisms; command/slope/step capability
remains to be proven with frozen physical examinations. The old
nine-box terrain remains available for a controlled baseline.
