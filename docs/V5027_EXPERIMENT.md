# v5027 — coherent commands × dynamic retention

**Status:** implemented on `feature/v5027-coherent-retention`, awaiting user WSL CPU/2048 CUDA preflight and GPU experiment. **Not** a new champion; protected `main` unchanged.

## What the frozen v5026 physics actually showed

The original strict all-axis report was **0/8 even on flat** because the policies' instantaneous yaw-rate MAE (~0.115–0.126 rad/s) exceeded the hard 0.08rad/s limit. All three actually survived flat 8/8 with ~0.022–0.024m/s forward tracking MAE. Strict heading precision remains a real, separate task; it should not conceal supported travel.

In the matched v5026 frozen terrain matrix (8 worlds/condition, 0.15m/s, 350 control frames) **D/mixed** performed 0/4 on *2cm uphill* crossing while **E/coherent** and **F/coherent+aggressive** performed 3/4. Both E/F crossed 7/8 paired 2cm steps versus D's 4/8. E/F maintained forward command on four-way 5° inclines (provisional moving mission ~31–32/32), whereas D remained physically stable but generally failed command tracking. This is early physics evidence for coherent teaching, **not** 28mm mastered stairs or 15° controlled travel. See [audited v5026 results](V5026_OVERNIGHT_FINDINGS.md).

## Experiment: isolate the still-hidden step teaching inconsistency

Original `coherent` teacher samples 0.10–0.22m/s forward stepping intent, *then* generic non-command teaching attenuates it around 0.04m/s:

\[
v_x' = 0.04 + 0.5 (v_x-0.04) \approx 0.07\ldots0.13\ \mathrm{m/s}.
\]

The independent ascent-progress guidance only becomes eligible above `cmd_vx >= 0.08m/s`. Thus even well-sampled lessons can miss the teacher's physical approach signal.

The new explicit `--terrain-lesson coherent-direct` path keeps the full 0.10–0.22m/s forward command **for focused STEP and SLOPE only**, zeros their sideways/turn intent and recomputes the correct gait phase rate. It does not change `coherent` or other specialists, and does not change the observation schema, PPO, physical collision geometry, sensor noise or reward definition.

`--retention-share 0.25` already exists: a quarter of birth focus is low-disturbance **moving** rehearsal (forward, some reverse, sideways, yaw and gentle pushes). We test it separately to see whether these old skills survive coherent terrain training.

## Fresh source-pinned one-brain A/B/C study

**One** 20M cold 360-input foundation (seed735, legacy gentle and mixed; old successful first 20M style) and **three** matched forks 20→60M, sharing exactly the same parent checkpoint. **140M aggregate environmental transitions**, not 140M age per policy. All use the restored `main-mature` command/phase/athletic settings, balanced six-focus schedule, 2048 worlds, gentle force program and 8cm geometry headroom.

| Arm | Terrain teacher | Retention | Question |
|---|---|---|---|
| **A — baseline** | `coherent` (old attenuated command) | 0% extra | Can we reproduce v5026 E-like terrain within the new same-source study? |
| **B — direct** | `coherent-direct` | 0% extra | Does preserving real crossing command improve actual step-up skill over A? |
| **C — direct + retention** | `coherent-direct` | 25% | Does moving rehearsal preserve general commands/recovery while retaining B's terrain growth? |

Comparisons **B−A** isolate the direct command treatment. **C−B** isolate dynamic retention. Actual latent old-skill recovery is still tested only by fixed physical exam—not presumed from `res=` teacher value.

**No old checkpoint transplantation:** source SHA pin deliberately prevents resuming the saved v5026 policy from `ab64cac` after training semantics change. Running from a fresh branch is a new experiment, not silently continuing E.

### Before training

Launcher is plan-only by default, requires clean named feature branch, writes results outside the repo, and performs **3 CPU variants + a full CUDA PPO gradient update at 2048 worlds** before creating the foundation learner. The smoke has an explicit assertion that post-specialist focused step commands genuinely remain in 0.10–0.22m/s. CPU smoke also verifies real paired incline/step collisions and 360-input actor geometry.

The launcher uses one console/one sequential GPU task; Ctrl+C terminates the child group, and **same-source** checkpoints can resume safely.

### After training

Every 10M the existing fixed exam records generalist tracking, dynamic80, resilience75 and core vx MAE. At the end, each 60M actor is exported and the **read-only 0.15m/s frozen physical matrix** runs on eight CPU worlds with identical physical cases, seeds and horizon. Reports distinguish strict heading tracking from true supported crossings and provisional physically useful movement.

Review **before any further training**:

- 1cm and 2cm step-UP/DOWN supported crossing, progression into 3cm with additional physical validation later
- 1/2/3/5° slopes four bearings, including commanded motion (not only stable contact)
- signed vx/vy/yaw core and dynamic tracking, resilience75, real push-conditioned recovery (still needs extended matrix)
- fall rate and servo shaft-work per useful metre
- no regressions at already-strong conditions, independent from raising frontier targets

Three policies × eight worlds is a *screen*, not mastery. Repeat promising cases with 32 worlds / alternate seeds; separately add signed heading/path consistency and force-versus-speed recovery grid before promotion.

### Launch from WSL

**Do not run this in or modify the old `~/emu-graduation` checkout**. The owner already has a separate `~/emu-research` checkout; refresh it onto this branch, using the old venv only for installed dependencies. The launcher calls `python -m emu.cli` from the new repository root, so it imports this source.

```bash
cd "$HOME/emu-research"
git fetch origin feature/v5027-coherent-retention
git switch --track -c feature/v5027-coherent-retention origin/feature/v5027-coherent-retention
source "$HOME/emu-graduation/.venv/bin/activate"
python scripts/v5027_coherent_retention.py
python -u scripts/v5027_coherent_retention.py --run
```

The new run root is `~/robot_lab/emu_runs/v5027_coherent_retention_s735`; the shareable report becomes `overnight_results.zip`, including `physical_summary.md` and each `unified_exam.json/md`.

**Do not `git pull` or modify source in the middle of a run.** Any branch commit changes the required training source SHA. The original 232-input Pinnacle remains protected and the public playground remains unchanged. No 400M graduation is proposed until controlled physical evidence supports it.
