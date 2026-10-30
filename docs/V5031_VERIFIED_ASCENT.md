# V5031 — verified uphill transfer, controlled B→C experiment

**Status:** v2 cross-source guard repair prepared; 2048-world GPU acceptance and 100M training pending at new pinned source.
**Protected:** `main` champion and the completed v5030 80M A/B study.
**Source:** `feature/v5031-cross-source-fork`; preserve its exact commit SHA in the new campaign manifest. The original `feature/v5031-verified-ascent` remains preserved at `93cb5cf`.
**Common ancestor:** v5030 B half-second reward, 80,019,456 steps, seed747, original SHA `a7be11b587f0912e642e100b22ec6d617f548f5b`.

## Why this experiment

The v5030 B 80M physical exam showed 8/8 supported 2cm uphill
crossings, 3/8 at 2.5cm, and **0/8 at both 3 and 3.5cm**.
For 3cm, the eight robots survived but none reached the far side;
mean forward motion was only about 0.010m/s, evidence of *stalling*
rather than falling. B had 13/16 uphill 12-degree mission successes,
15/16 survived the 0.6m/s flat ladder, 10/16 survived at 0.8m/s,
but 0/16 sustained a 1m/s request.

The next experiment therefore targets the **actual supported climb**,
not appearance of stillness, yaw oscillation or a more ambitious
curriculum frontier. Keep the half-second velocity reward B (share .5).
Do not introduce yaw jitter penalties.

## Two equal continuations, one treatment

Both arms resume from the **same exact B80 final checkpoint**, which
contains policy/value weights, optimiser, observation normalisation,
governor/field and training health. Environment physics episodes
reset on a research fork by design; no previous episode is replayed.

| Arm | 80→100M | Existing ascent shaping | New verified credit | Completion |
|---|---|---|---|---|
| `B80_control` | 20M | identical v5030 B (max 9) | 0 | +10 same as before |
| `C80_verified_ascent3` | 20M | existing max 9 reduced to 6 | max 3, nonrepeatable | +10 same |

**Nothing else changes:** seed747, 2048 CUDA worlds, 25-frame
velocity reward 0.5, 25% fast rehearsal, all six focus cohorts, push
programme, physical slabs, actor/ToF observation schema, optimisers,
normalisation and direction/terrain lessons.

The C credit requires *actual upper-slab foot contact*, a healthy
supported posture and positive torso elevation **along the lower
slab's real normal**, relative to its level-ground neutral height.
The cap is real step height (up to 4cm), not a teacher frontier.
The episode-local high-water mark prevents repeated tapping or rocking
from earning multiple payments. Flat worlds, step-down worlds, falls,
and unsupported jumps earn zero. The +10 fully supported crossing
payment is unchanged; three of the original nine progress/foothold
points are moved to verified transfer instead of creating new rewards.

The source adds one new **reward-only** scalar to MJX episode state;
the 360-input actor and learnable PPO parameter and optimizer shapes
stay identical. A cross-source B80 fork is explicitly authorised
**only after** exactly matching source/config/parent checkpoint
hashes and both CPU and 2048-world CUDA preflight checks. Do not
silently resume v5030 with this changed source inside its old run dir.


## 31 October — source-guard repair and safe re-launch (v2)

The initial `feature/v5031-verified-ascent` campaign passed all CPU/2048-world
CUDA preflights, but its **first training attempt never started**: the
central `emu.run.inspect_run` correctly rejected the original v5030 B80
checkpoint's different source SHA. The original launcher recorded
cross-source intent in its manifest but had not supplied a matching
explicit proof to this central guard. Never remove the central guard
or edit an ancestor checkpoint to fool provenance.

**Corrected source branch:** `feature/v5031-cross-source-fork`.
This code permits an external cross-source *research* fork only when
the command expressly includes BOTH the ancestor's complete 40-character
Git SHA and the full SHA256 digest of the actual immutable `final.pkl`.
The full schema/360-sensor/NN/optimizer-state rewind validation still
runs; original-parent `TrainingConfig` compatibility is rechecked,
and the ancestor must be older than the requested target. Missing,
mismatched or gratuitous flags fail closed. Same-source
`--resume auto` never uses or requires these flags.

This experiment's launcher pins the known v5030 B80 Git SHA and the
actual local ancestor SHA256 before starting. New campaign results go
to a **fresh root** so the old failure manifest remains untouched:

`~/robot_lab/emu_runs/v5031_verified_ascent_fork_v2_s747/`

**No previous B80 or first failed v5031 data needs deleting.**
The unchanged ancestor checkpoint remains outside the new root and
is only READ; new v5031 PPO/checkpoint state is written under the v2
root. Current CI and exact-SHA CUDA preflights must pass again before
either B or C runs. This is essential because the repaired source has
a new Git SHA; previous preflight stamps do not authorize it.

From a WSL Ubuntu terminal (inside your persistent GNU Screen session):

```bash
git clone --single-branch --branch feature/v5031-cross-source-fork \
  https://github.com/mdt733/emu.git "$HOME/emu-v5031-fork"
cd "$HOME/emu-v5031-fork"
source "$HOME/emu-graduation/.venv/bin/activate"
python -m pytest -q tests/test_cross_source_research_fork.py tests/test_v5031_verified_ascent.py
python scripts/v5031_verified_ascent_ab.py

# Only after clean plan and successful source-specific checks:
python -u scripts/v5031_verified_ascent_ab.py --run
```

Do not pull the new branch into `~/emu-v5031`, modify v5030's source,
reuse `~/robot_lab/emu_runs/v5031_verified_ascent_s747/` or change the
published `main` actor. Once either v2 arm has begun, do not pull new
commits in the v2 checkout either.

## Historical v1 remote launch (superseded; do not use)

Everything is on GitHub. No downloadable ZIP is needed.

Inside a terminal on the GPU's WSL Ubuntu:

```bash
if [ ! -d "$HOME/emu-v5031/.git" ]; then
  git clone --single-branch --branch feature/v5031-verified-ascent \
    https://github.com/mdt733/emu.git "$HOME/emu-v5031"
fi
cd "$HOME/emu-v5031"
source "$HOME/emu-graduation/.venv/bin/activate"
git branch --show-current
git rev-parse HEAD
python scripts/v5031_verified_ascent_ab.py
```

The plan prints the resolved source/parent contracts without
training. Check the ancestor:

`~/robot_lab/emu_runs/v5030_ab_learning_horizon_s747/at80M/B_halfsecond25/checkpoints/final.pkl`

After a clean plan and successful CI, launch **inside Screen**:

```bash
cd ~/emu-v5031
source ~/emu-graduation/.venv/bin/activate
python -u scripts/v5031_verified_ascent_ab.py --run
```

It executes a pure reward helper test, two CPU MJX smoke probes, and
two source-stamped full 2048-world CUDA PPO preflights, **then** runs the
matched continuations sequentially. It exports policies and runs the
same 16-world speed ladder and terrain matrix for both arms, writes
short/medium/full reports, and saves all outputs outside the checkout:

`~/robot_lab/emu_runs/v5031_verified_ascent_s747/`

The physical comparison ZIP is called
`v5031_verified_ascent_physical_comparison.zip`.
The original B80 checkpoints and published main remain untouched.

## Decision rule after results

- Physically verified **3cm UP** crossings are the primary target.
  Count separate UP/DOWN outcomes; do not mistake 8/8 upright or a
  3cm teacher frontier for a single successful climb.
- Guard against regressions in 2cm ascents, 0.6m/s actual speed,
  0.8m/s sustained survival, slopes, commanded turning and meaningful
  shaft-work J/m. Use sustained signed yaw bias and path heading, not
  gait-induced rapid yaw oscillation.
- Reject the treatment if credit is zero in training (no actual
  contact/transfer events) or if it increases stalling or unsafe falls.
  Both total paid credit and event count are in rollout telemetry.
- A one-seed 20M continuation is discovery, not a final proof.
  Repeat important improvements on an independent seed before
  incorporating them into the 200–300M fresh Pinnacle candidate.
- If C does not solve it, inspect step-lip foot-contact traces to
  distinguish insufficient reach, no target contact, and failed
  load transfer; do not blindly increase the reward.
- Independently test a progressive-speed ramp to 1m/s to determine
  whether the remaining high-speed failures are abrupt-start failures
  or unsustainable fast gait. No oscillation penalty by default.

**Never replace protected main or its published policy automatically.**
