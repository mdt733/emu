# EMU foundation experiments — one learner, multiple early recipes

No distinct trainer, PPO, actor or policy branch. Every run uses the same
360-input actor, physical MJX robot and the original eight locomotion reward
coefficients. The run config records the foundation recipe and its handover
point, and is source-pinned for reproducible resume.

| --foundation-mix | Foundation lessons after 70 steady-only updates |
|---|---|
| main | Original adaptive steady / transition / resilience proportions |
| command-heavy | 90% steady commands, 5% transitions, 5% resilience |
| curious | 84% steady, 8% transition, 8% resilience in the other 95% of worlds, and ~5% easy control/grace |

Choose `--foundation-steps 10000000` or `20000000` (10M or 20M).
The same growth specialty lessons fade in over the following 10M world steps,
then independently nudge their physical difficulty targets. A 10M run ending
at the foundation boundary can **extend its existing run** later, with
`--resume auto` and exactly the same source/settings.

## First comparison

Start with three separate 10M cold runs, seed 733, one for each recipe.
Compare *actual* standing, supported stepping, seed-command tracking, falls,
foot activity and simulated shaft work per meaningful distance. Do not decide
from reward alone. The winner can be repeated with seeds 734/735, compared
against frozen `main` on identical physical exams, then extended to 30M+.
A 10M policy is an infant, not a terrain qualification exam.

### Safe read-only plans for all three

```bash
(
set -euo pipefail
if git -C "$HOME/emu-capability-growth" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cd "$HOME/emu-capability-growth"
elif git -C "$HOME/emu" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cd "$HOME/emu"
else
  echo "STOP: no valid Git checkout; see setup instructions"; exit 1
fi
test -z "$(git status --porcelain)" || { echo "STOP: dirty tree"; exit 1; }
git switch feature/v5023-capability-growth
git pull --ff-only
source .venv/bin/activate

for mix in main command-heavy curious; do
  emu train \
    --run-dir "$HOME/robot_lab/emu_runs/foundation_${mix}_s733_10m" \
    --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
    --unified-capabilities --foundation-mix "$mix" \
    --foundation-steps 10000000 --steps 10000000 \
    --envs 2048 --seed 733 --device cuda --plan
done
)
```

### Actually train the curious candidate after CUDA acceptance

This is a REAL GPU launch. Review exact-SHA CI and run the 2048-world CUDA
PPO smoke in INSTALL.md first.

```bash
(
set -euo pipefail
if git -C "$HOME/emu-capability-growth" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cd "$HOME/emu-capability-growth"
else
  cd "$HOME/emu"
fi
test "$(git branch --show-current)" = "feature/v5023-capability-growth"
test -z "$(git status --porcelain)"
source .venv/bin/activate
nvidia-smi
RUN="$HOME/robot_lab/emu_runs/foundation_curious_s733_10m"
test ! -e "$RUN" || { echo "STOP: run directory already exists"; exit 1; }
XLA_PYTHON_CLIENT_PREALLOCATE=false \
EMU_UNIFIED_CONTINUOUS=cold-growth-approved \
emu train --run-dir "$RUN" \
  --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
  --unified-capabilities --foundation-mix curious \
  --foundation-steps 10000000 --steps 10000000 \
  --envs 2048 --seed 733 --device cuda
)
```

To compare other mixes, change BOTH the mix flag and run directory name.
To repeat a candidate, change BOTH seed and run directory. To extend an
already completed, same-source 10M run, retain all settings and root,
raise `--steps` (e.g. 30000000) and add `--resume auto`; the existing
final checkpoint is validated before any training resumes.

Protected: `main`, approved Pinnacle and public showcase; no promotion
without owner review.
