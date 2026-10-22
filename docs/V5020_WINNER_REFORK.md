# v5020 — five-way winner-and-refork, with earned obstacle height

**One integration Git branch:** `feature/v5020-coldstart-terrain`.
**Historical main/reference:** successful v5016 Pinnacle methodology.
**Historical failed v5020 campaign:** unchanged; `emu campaign5020` remains reproducible.

We now make **five different learning strategies**, not five Git branches. All
contenders start from the **same frozen schema-9, 312-input policy**, including
actor, value network, optimiser, observation normaliser, governor, training
health and capability field. At each 40M/60M/80M policy-step milestone we
evaluate all five, promote **one qualified winner**, and refork its *entire*
learning state into five new independent training continuations.

## Five contenders, every generation

| Contender | What it teaches |
|---|---|
| `general` | Main-style conservative level/general gait; no physical boxes |
| `turns` | More mixed forward/lateral/yaw commands |
| `recovery` | More disturbance pushes and recovery lessons |
| `efficiency` | Mild servo mechanical-effort costs |
| `steps` | Real collidable box steps **on level ground**, with an **earned continuously growing step-height frontier** |

The 312-input sensor package remains identical across all five. **The steps
arm does not have a prescribed 3mm, 5mm or 20mm final height.** It begins
with a **3mm frontier** only when the parent has no previously earned one,
and trains on a mixture of no-obstacle births, intermediate step heights and
frontier-edge steps. Some episodes are flat to rehearse general walking;
non-forward tasks are also flat because a box placed on global +X would
otherwise train avoidance and falling instead of climbing.

The preexisting `RegimeGovernor` accumulates real *completed* episodes
near the frontier and advances by up to **1mm per qualifying evidence batch**
when true crossing, contact and fall criteria pass. Falls can retreat height;
lack of evidence holds it still. The height field lives in the saved governor
and transfers with whichever trained policy wins. The safety ceiling is
**150mm** under the *current fixed-geometry MJX box contract*. That's an
engineering limit, **not a target**: if real, independently replicated
competence reaches it, the next research change can enlarge the collision
geometry and carefully test heights beyond 150mm, perhaps eventually 200mm.
Just lifting a numeric cap would not make the robot capable.

The optional `terrain100` generation is now **another five-way competition**
from the 80M winner, not a special 3-versus-5mm shootout. The first launch
still defaults to stopping at 40M for evidence review.

## Evaluation: the same physics for every contender

We use *identical frozen-policy screens* for all five at each generation:

- General/core survival, commands, posture, stepping evidence and recovery
- Two real-contact slope distributions
- Comparable shaft-effort, overload and action-smoothness measurements
- **True collidable step crossings at 5, 10, 20, 40, 80 and 150mm**

These 6 heights are **testing milestones**, not ceilings in the teaching
distribution. All models—including `general`, which trained no boxes—are
evaluated at every height. This preserves the ability of naturally agile
walkers (such as the older 260M demo) to score obstacle competence even
without explicit step lessons.

The frozen examiner now sets **the exact requested collision height at birth**,
not a randomized lesson height capped by the exam label. Each height requires
at least **8 completed eligible attempts** for tournament grading, with
explicit contact and crossing rates; insufficient evidence fails closed.
This is a screening threshold, not a hardware safety certification.

The examiner requires a real box-foot contact and **supported, upright
passage beyond the measured box**, with a matched seed and course geometry.
Neither merely moving forward nor reporting a large training reward counts.

The step-grade score rewards crossings across the six heights (weighted
towards 20–40mm, with credit also at 80–150mm). It contributes **25% of a
positive, unnormalised score component**, but general locomotion, recovery,
contact-slope survival, gait, posture and motor power remain important.
The weights are fixed for everyone at every generation. Independent physical
course results can therefore advantage the steps arm **without permitting
a stumbling, stationary or unstable policy to win**.

The step specialist also has a small **reward-only** training incentive:
supported, upright progress in the obstacle corridor, and a single bonus
when a previously contacted real obstacle is genuinely cleared the first
time in an episode. This never adds privileged obstacle truth to actor
observations; the sensor contract stays 312 inputs. The tournament never
uses this PPO teaching payment as a measured crossing.

## Guard rails after the failed 380M experiment

Each contender first runs a 2M-step probe under its proposed lesson
distribution. **Probes log every PPO update** (roughly 16 updates at
4096 worlds or 31 at 2048, 32 physics/control frames each), instead
of the ordinary 10-update CSV cadence. Longer surviving continuations
retain the efficient 10-update cadence.
It reads **every logged row**, including all sparse-log
intervals' **worst per-world spring-energy and shaft-power peaks**.
Unphysical values at any point veto continuation, even if the final rows
look healthy. The final ~20% of rows are evaluated using the number of
completed episodes as their weight: >92% falls or <70-frame episode
lengths reject persistent collapse. Too few completed episodes or
nonfinite evidence also prevent qualification. Earlier collapse followed
by sustained recovery is reported as a warning, not automatically a
rejection. This is intentionally a coarse **research safety screen**,
not a physical hardware certification.
A failed probe **never trains the next ~18M**. Each surviving
continuation completes in its own pinned training-config phase.

At the 40M, 60M and 80M boundaries, every final policy faces the same
frozen-policy tests and is rejected if its general locomotion fails the
stage-specific core/recovery/contact-slope/gait/height gates. The parent
checkpoint is also tested under the same conditions. If all contenders
are rejected or regress badly against a qualified incumbent, the campaign
**stops and preserves the parent** rather than selecting a false champion.

One GPU subprocess runs at a time. `flock`, run/source identity, parent
SHA-256, validated checkpoint schema and selection SHA checks enforce
resumability. Existing output roots and old policy files are never
reinterpreted under changed training source.

## Read-only plan / tests

```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git fetch origin feature/v5020-coldstart-terrain
git merge --ff-only origin/feature/v5020-coldstart-terrain
source .venv/bin/activate

emu campaign-refork
python -m pytest -q tests/test_v5020_refork.py tests/test_v5020_campaign.py
```

Inspect available foundation snapshots before choosing a parent:

```bash
find ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints \
  -maxdepth 1 -name '*.pkl' -printf '%f\n' | sort
```

The input may be any compatible, **validated 12–22M foundation checkpoint**,
including a healthy earlier snapshot. The completed 20M final is not
automatically best.

## First GPU acceptance: STOP at 40M

**Only after Python CI and real MJX examiner smoke pass.** Example uses the
4096-world foundation in the previous campaign and preserves that environment
count for its descendants:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
emu campaign-refork --run \
  --foundation ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints/final.pkl \
  --envs 4096 --seed 733 \
  --screen-envs 32 --screen-horizon 300 \
  --stop-after walk40 \
  --root ~/robot_lab/emu_runs/v5020_refork
```

The full default evening run now continues through successive **five-probe
40M/60M/80M generations**, and conditionally a 100M terrain generation
when the 80M policy is physically coherent. Use `STOP_AFTER=walk40`
to limit it to the first round. Each generation runs five probes,
then only survivors' continuations and frozen exams. A promising but
not-yet-qualifying policy may become a **provisional research parent**
for more training; it is explicitly **not** promoted as a certified
champion. A qualified existing incumbent is carried forward when new
contenders regress rather than prematurely ending a healthy lineage. Expect the five-arm 40M round to require
more total environment steps than the previous four-arm proposal. Inspect
`walk40/scoreboard.json`, `walk40/scoreboard.md` and the winner before
using the same pinned commit/root to increase `--stop-after` to `walk60`,
`walk80`, then optional `terrain100`.

Changing from the original 4096 training worlds to 2048 is a separate
experiment and requires a 2048-world foundation checkpoint and fresh root.
At equal aggregate environmental steps, 4096 worlds produce about half
as many 32-frame PPO updates, so don't equate a 20M policy step counter with
proof of curriculum maturity.

## Implementation and verification

- `emu/campaign_refork.py`: five-arm lineage, probe/finish separation,
  score eligibility and single-winner promotion
- `emu/evaluation.py`: frozen same-geometry, six-height MJX examination
- `emu/terrain.py` and `emu/curriculum.py`: existing per-world adaptive
  births, earned height growth, retreat and training-evidence pooling
- `emu/environment.py` and `emu/rewards.py`: reward-only,
  physically verified first crossing
- `tests/test_v5020_refork.py`: five-arm recipes, capability/score gates,
  probe aborts, DAG promotion, no altered sensor contract
- `tests/test_v5020_coldstart.py`: actual MJX obstacle/crossing/efficiency
  compilation smoke

The code and documentation are under review on the v5020 feature branch.
No changes to `main` or the showcase, and no locally trained weights
are written to Git. A clean source checkout and successful Python/real-MJX
acceptance remain required before the next large run.
