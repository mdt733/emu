# v5051 physical science battery — operator runbook

**Research branch:** `research/v5051-gait-autopsy`. This is **not an RL training campaign** and does not change `main`, actor weights, curriculum, solver or checkpoint. Do not merge while acceptance is failing or unreviewed.

## What it tests

| Test | What is physically measured | Limitation |
|---|---|---|
| Reward economics | Exact deterministic PPO gate payments and zero-request exception | Counterfactual arithmetic, no policy |
| CPU standing screen | 10s bounded servo PD across gains, noise and delays, ball-foot support and fall | Flat plane, not full deployment controller; failed PD does not prove impossibility |
| CPU supported swing | Left/right open-loop commanded hip/knee sweeps, real ball-foot contact, supported vertical lift, forward reach, torque | Real MuJoCo flat plane, but no riser and no proof of weight transfer |
| CPU fixed wall screen | Real colliding risers at actual 1/2/2.5/3/3.5/4cm, opposite foot support, foot clearance, upper top-contact streak and collision diagnostics | Open-loop probe, not frozen actor or completed climb; base robot rods remain non-contacting |
| Frozen-policy flat panels | Survival, actual velocity, drift, footfall Hz, same-foot stride, phase coherence and shaft-work proxy | Same saved actor; no learning-variance estimate |
| Frozen-policy prism panels | Actual signed prism-top support, separate post-contact recovery and 50cm commanded continuation | Procedural heights, not fixed 3cm fixtures; detection, riser clearance and CoM load transfer not instrumented |
| Phase/ToF intervention | Same held-out seeds and command: valid phase clock hold/invert/random; 4x4 ToF range+validity masked | Input-only frozen-policy *sensitivity*, not proof an alternative-trained actor could do without inputs |

The post-lip mission scorer **requires simultaneous bilateral actual top contacts for five frames** and then an additional 50cm along the original heading, with a <=0.35rad whole-interval heading bound. No eligible physical supported contacts means the denominator is not measured, not a zero-percent climb success rate. Confidence intervals are Wilson intervals conditional on independent-trial assumptions, **not confidence across independent PPO training seeds**.

## Fresh WSL checkout and deterministic source pin

Use the maintained `docs/INSTALL.md` if a Python 3.13 / CUDA venv is missing. No ZIP or extra repo. Foreground shell only.

```bash
cd "$HOME/emu"
git fetch origin
git switch research/v5051-gait-autopsy
git pull --ff-only origin research/v5051-gait-autopsy
git status --porcelain
git rev-parse HEAD
source .venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Before running: commit SHA must match the version of PR #42 you intend
# to certify. Do not switch branches or modify source mid-battery.
python -m unittest discover -s tests -p 'test_v5051_step_autopsy.py' -v
python -m scripts.v5051_reward_economics
```

All resulting files live under `~/emu-data/runs`, not in the source repository. Run the cheap mechanical screens before the long frozen-policy matrix.

```bash
ROOT="$HOME/emu-data/runs/v5051-audit-s735"
mkdir -p "$ROOT"
python -m scripts.v5051_balance_screen --quick --output "$ROOT/mechanics-quick"
python -m scripts.v5051_swing_screen --quick --output "$ROOT/mechanics-quick"
python -m scripts.v5051_fixed_wall_screen --quick --output "$ROOT/mechanics-quick"
python -m scripts.v5051_balance_screen --output "$ROOT/mechanics"
python -m scripts.v5051_swing_screen --output "$ROOT/mechanics"
python -m scripts.v5051_fixed_wall_screen --output "$ROOT/mechanics"
```

**Acceptance smoke:** requires a valid frozen 360-input `.emupolicy`; the public viewer export is suitable for *inference*, not resuming PPO. Check source/export identity in the manifest. The first one-panel CPU test compiles the same MJX prism world; a larger CUDA test uses the RTX 4060 Ti.

```bash
POLICY="$PWD/showcase/champion.emupolicy"
test -f "$POLICY"
python -m scripts.v5051_science_battery \
  --policy "$POLICY" --device cpu --profile pilot \
  --worlds 4 --horizon 100 --max-panels 1 \
  --output "$ROOT/prism-smoke"
python -m scripts.v5051_science_battery \
  --policy "$POLICY" --device cuda --profile pilot \
  --worlds 16 --horizon 650 \
  --output "$ROOT/prism-pilot"
```

The **pilot** runs 44 separate frozen-policy panels: 2 held-out simulation seeds, 4 flat speeds, 3 step-frontier settings at 2 approach speeds, plus 12 intervention panels per seed. It emits a complete `SCIENCE_REVIEW.md`, `science_summary.json` and one `prism_exam.json` per case. Run is foreground and resumable: rerun the same command to skip *completed, identity-verified* panels. Existing partial/incomplete panel dirs stop instead of being overwritten. If you must retry one broken panel, move it outside the battery directory to `~/emu-data/scratch/` first; never silently reuse or delete evidence.

If the smoke, pilot and independent physical review pass, the **full** frozen battery is a separate result directory: **102 panels**, 3 held-out environment seeds, 8 flat commands and 7 requested lip frontiers. It is still NOT a three-PPO-seed campaign.

```bash
python -m scripts.v5051_science_battery \
  --policy "$POLICY" --device cuda --profile full \
  --worlds 16 --horizon 650 \
  --output "$ROOT/prism-full"
```

Use `--max-panels N` in a new output directory for bounded integration testing. Do not change `--worlds`, `--horizon`, `--profile`, device, actor or source when resuming an existing battery: `identity.json` rejects identity changes.

## Review decision — mandatory stop before PPO

1. Confirm CPU screens work without collision disabling. Judge mechanical evidence as preliminary, including cases with failed PD or missed opposite-foot support. Examine high-lift sets for believable actuator loads.
2. Compare baseline frozen-policy gait and perturbations: 0-command footfalls/s, drift, phase locking; 0.70m/s stride and survival; masked ToF effect on stair episodes. One cohort does not establish PPO learning variance.
3. Inspect the signed *actual* lip heights and mission event failure labels, not the nominal requested label or the saturated v5048 composite score.
4. Run source-pinned GPU smoke and existing CI on the PR before additional work. No claim of full wall-scale climb without a **fixed actual-height**, controlled-approach examiner that records riser collision and body/foot loading.
5. **Do not launch the multiseed PPO campaign yet.** Phase dependence, rewards, physical feasibility, effective task distribution and fixed-height mission protocol first need a written go/no-go review. With reusable optimiser checkpoints confirmed, select few compatible matched arms plus 2–3 PPO seeds, rather than spending hundreds of millions on uninterpretable mixtures.

**Known unfinished work:** fixed-height 1/2/2.5/3/3.5/4cm **frozen-actor** riser fixture (the CPU mechanical riser probe is now implemented), support-wrench/CoM load-transfer examiner, true acquisition-age/ToF geometry probe, a nonlinear standing-controller feasibility follow-up, independent PPO seed variance, and eventual temporary train-arm runner. This initial diagnostic battery alone does **not** solve stairs or authorize a champion replacement. All physical science on this PR remains unverified by the owner's GPU until the commands above are run and logged.
