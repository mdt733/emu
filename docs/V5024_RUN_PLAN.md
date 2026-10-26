# v5024 — terrain stability, commanded control, and next run

**Frozen GPU training branch:** `feature/v5024-terrain-control`
at `d26ae0c2c4097d7c7124759197a94504abc44995`.
The separate `feature/v5024-release-polish` branch is for
cleanup/review and **must not resume this experiment's checkpoints**.

**Do not edit/merge** `main`, the public showcase, archived v5023
policies, or the completed 260M campaign. This branch includes the
`feature/v5023-surface-lab` viewer changes **already**, plus the new
teacher, reporting and frozen examiner. No cross-source checkpoint resume.

## Why v5024 exists

The 260M v5023 experiment completed five independent runs. At 80M, the
balanced `curious` policy led the matched generalist exam in the
first seed (broad score approx .725), while terrain-early learned
larger slope/step targets at some expense to broad generalist performance.
The source report's uphill near-boundary tracking pass rate was
consistently much worse than downhill. Manual viewer driving showed the
robot can often STAY UPRIGHT on a 2–5° plane while drifting downhill
rather than following the requested body-frame velocity.

These observations distinguish three things:
1. Stable support on true inclined contact.
2. Forward progress against/with gravity.
3. Accurate vx, vy and yaw control in the presence of gravity.

A frontier is a **practice request**, not a certified capability envelope.
A larger slope target on its own is never the winning metric.

## Implemented changes — ONE shared actor/optimizer retained

- Slope training now emits two separate episode outcomes per real
  direction and near/easy severity: `stability_successes` (supported,
  moving, no fall) and the original strict `successes` (including
  command-following).
- The shared **physical-practice** incline target consumes completed
  near/easy stability episodes, requiring >=8 in EACH of four directions
  and >=80% supported motion to increase by only 0.08°. An easier
  cohort below 45% stable motion can retreat. Poor tracking remains
  counted as poor tracking; it is not renamed a success.
- Repair misleading `frontier_event_vy/yaw=off` logging: the true
  symmetric signed pairs were already trained but the display queried
  nonexistent aggregate keys.
- Add opt-in `--teaching-schedule steering`: 35% command, 15%
  resilience, 20% slope, 15% step, 10% transition, 5% grace **among
  active specialist episodes**. In COMMAND-focused worlds, increase
  the pure-axis vertex share from 27% to 45% and preferentially sample
  lateral/yaw signs. Balanced/staged/terrain-early keep their existing
  distribution unchanged; actor dimensions stay 360.
- Fix gait-phase recomputation for modified easy commands in
  non-command specialist worlds.
- Keep the last actually measured terrain-normalizer clipping fraction
  and its acquisition step in subsequent log rows instead of filling
  every ordinary row with unavailable `-1`. This remains diagnostic,
  not a changed normalizer.
- Read-only `emu capability-exam --terrain-matrix`: matched 1°, 2°,
  3°, 5° uphill/downhill/left/right real MJX slopes; 0.3, 0.5, 1,
  2 cm up/down; flat, lateral ±0.05m/s and yaw ±0.2 rad/s.
  Each slope report has **separate stable and tracking pass counts**
  with signed downhill/cross-slope drift and vx/vy/yaw errors.
  No teacher/optimiser feedback from exams.
- The v5023 **surface viewer is INCLUDED**: real MuJoCo continuous
  user-set uphill, downhill, camber planes and up/down step geometry
  and grade-relative fall recognition.

## Proposed next experiment: 3×60M, independent cold starts

| Arm | First 10M teaching | 10M→60M teaching | Controlled question |
|---|---|---|---|
| A · `main_balanced` | Main adaptive foundation | Balanced 20/20/20/20/10/10 | Does trusted main-style infancy feed v5024 well? |
| B · `commandheavy_balanced` | 90/5/5 foundation | Same balanced | Is command-heavy infancy better than main-style? |
| C · `commandheavy_steering` | IDENTICAL to arm B's foundation recipe | Steering 35/15/20/15/10/5 plus pure lateral/yaw emphasis | Does focused control beat balanced growth? |

**Total 180M world steps**, 2,048 worlds sequentially on the RTX 4060
Ti. Same seed 733, PPO, 360 inputs, real terrain contacts, reward model
and ten-million-step fixed generalist evaluations. No failed arms hold
the next arm hostage, but CPU/2048-world CUDA preflight failure aborts.

Arm A vs B isolates **foundation mix**. B vs C isolates the added
**steering training emphasis**. The stability/tracking teacher and
other v5024 repairs are shared among ALL arms. For a clean comparison
with the v5023 slope algorithm, export the archived v5023 policies
and use the SAME new fixed terrain matrix; do not confuse two source
families as identical training conditions.

The campaign's 60M endpoint is **exploratory**, not a final winner.
If a promising pair emerges, repeat it at seeds 734 and 735 and extend
same-source policies rather than guessing from one stochastic seed.
Keep one policy per run; no premature champion promotion.

### Safe plan and optional launch

Use a NEW checkout, do not alter the historical v5023 training checkout.
The commands below are for WSL, **after exact-SHA CI is green**.

```bash
git clone --single-branch -b feature/v5024-terrain-control \
  https://github.com/mdt733/emu.git "$HOME/emu-v5024"
cd "$HOME/emu-v5024"
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q tests
python scripts/v5024_campaign.py       # PREVIEW only; no GPU work
```

To run, verify GPU and free disk first; launch in tmux so closing the
terminal does not kill training:

```bash
nvidia-smi
df -h "$HOME"
tmux new -s emu-v5024
cd "$HOME/emu-v5024"
source .venv/bin/activate
python scripts/v5024_campaign.py --run
```

The launcher runs 12-world CPU MJX + 2,048-world FULL CUDA PPO preflight,
captures the source/runtime manifest, freezes 10M exams and then trains
each arm. Detach tmux with Ctrl-B then D; reconnect with
`tmux attach -t emu-v5024`. It writes
`~/robot_lab/emu_runs/v5024_cold_steering/overnight_results.zip`.
Same SHA/config/root supports interruption-safe resume. No CUDA
training is launched from this chat.

### Independent terrain exam of existing 80M policies (no training)

The new examiner can first test v5023's archived 80M balanced and
terrain-early policies under identical physics and commands. In the
v5024 checkout with Python dependencies installed:

```bash
OLD="$HOME/robot_lab/emu_runs/overnight_teaching_v5023_audited"
LAB="$HOME/robot_lab/emu_runs/v5024_archived_exams"
mkdir -p "$LAB"
JAX_PLATFORMS=cpu python -m emu.cli export-policy \
  "$OLD/growth_balanced_s733/checkpoints/final.pkl" \
  "$LAB/balanced_80m.emupolicy"
JAX_PLATFORMS=cpu python -m emu.cli export-policy \
  "$OLD/growth_terrain-early_s733/checkpoints/final.pkl" \
  "$LAB/terrain_early_80m.emupolicy"

JAX_PLATFORMS=cpu python -m emu.cli capability-exam \
  --terrain-matrix --policy "$LAB/balanced_80m.emupolicy" \
  --output "$LAB/balanced_matrix" --worlds 8 --horizon 350 --device cpu
JAX_PLATFORMS=cpu python -m emu.cli capability-exam \
  --terrain-matrix --policy "$LAB/terrain_early_80m.emupolicy" \
  --output "$LAB/terrain_early_matrix" --worlds 8 --horizon 350 --device cpu
```

Each output contains `unified_exam.md` and `unified_exam.json` with
slope-specific stability AND tracking scores, signed drift and
genuine step-supported crossings. **Do not** run these against the
archived trainer's Python process or rewrite its checkpoints.

## Acceptance criteria before another full night

1. Exact-SHA Python, browser/viewer and unified MJX CPU CI green.
2. `v5024_campaign.py` plan lists three independent 60M arms
   and 180M steps; **no** unintended `--run`.
3. A 2048-world CUDA PPO update passes on the owner's GPU before
   any long arm begins.
4. At the 10M handover, birth roles and terrain stability/tracking
   counters start appearing without tensor-contract changes.
5. The frozen 1/2/3/5° examiner proves actual collision geometry and
   keeps **supported-stable** distinct from **precisely tracked**.
6. Training `main`, v5023 archives, public Pinnacle and showcase
   remain unchanged.

**Do not claim more incline competence simply because the reported
teacher limit expands.** Judge matched held-out progress, physical
support, uphill velocity, lateral downhill drift, yaw, resilience,
supported step-ups AND downs, and energy per useful distance.
