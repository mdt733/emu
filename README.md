# EMU · Learning to walk

**Six actuators. Two feet. No scripted gait.** EMU studies how an unstable
biped learns locomotion in thousands of parallel MuJoCo/MJX worlds.

> **Two different states of the project:**
>
> - [`main`](https://github.com/mdt733/emu/tree/main) is the **untouched,
>   proven reference**, with the approved 232-input v5016 Pinnacle 260M
>   showcase. No experimental changes are automatically published there.
> - [`feature/v5020-coldstart-terrain`](https://github.com/mdt733/emu/tree/feature/v5020-coldstart-terrain)
>   is **experimental**: 312-input terrain sensing, real collidable steps,
>   adaptive step height and the five-way Champion Ladder. Its source tests
>   pass, but it has **not** produced a qualifying stronger trained policy.
>
> **Do not use the v5020 source to resume a run pinned to a different commit.**
> Keep all checkpoints and training outputs outside the source checkout.

**[Interactive EMU Playground](https://mdt733.github.io/emu/)** ·
**[Architecture](docs/EMU_ARCHITECTURE.md)** ·
**[Install & run safely](docs/INSTALL.md)** ·
**[Research findings](docs/RESEARCH_HISTORY.md)** ·
**[Current cleanup checklist](docs/V5020_CLEANUP_PLAN.md)**

## What the current research system does

```text
robot.xml and environment.py   Physical biped, servo and contact simulation
             │
       2048/4096 MJX worlds    One shared actor, independent sampled lessons
             │
      ┌──────┴───────────┐
      │                  │
   rewards.py        results.py        Per-step payments vs physical episode facts
      │                  │
    ppo.py         curriculum.py       Gradient update vs evidence-driven teaching
      └──────┬───────────┘
             │
       validated checkpoint
             │
   campaign_refork.py           Five 20M continuations of one common parent
             │
       evaluation.py            One independent exam for every frozen contender
             │
       choose ONE winner        Or stop and preserve the incumbent
             │
      repeat after 20M
```

The five teaching emphases are **General, Turning, Recovery, Efficiency and
Steps**. These are *training continuations*, not five Git branches. The
trained champion's complete checkpoint (network, optimiser, normaliser,
governor, health, field) is the parent for the next five arms.

**Actor contracts are versioned, not interchangeable:** 220 (historical),
232 (proven sim-to-real reference), and 312 (experimental rolling terrain
preview) inputs; all have six action outputs. Training/teacher truth must
not leak into deployable observations.

The 312-input physical Steps specialist starts with small obstacles,
then may earn harder heights from verified crossings. Its **current
simulator geometry ceiling is 150mm**, not a certified capability claim.
The independent examiner tests every contender at 5/10/20/40/80/150mm,
in addition to general walking, recovery, slopes and mechanical effort.

## Quick start

Python 3.13+ and a suitable JAX/MuJoCo installation are required. For a
**new WSL checkout** of the research source:

```bash
git clone --single-branch -b feature/v5020-coldstart-terrain \
  https://github.com/mdt733/emu.git emu-v5020
cd emu-v5020
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate

emu doctor --v5015-smoke --domain-smoke
emu campaign-refork                         # READ-ONLY plan
emu campaign-refork --status                # READ-ONLY status
python -m pytest -q tests/test_v5020_refork.py tests/test_v5020_campaign.py
```

`bash scripts/setup_wsl.sh` performs installation and tests. Actual GPU
training requires explicit `--run`, a validated 312-input foundation
checkpoint, a clean pinned source revision, and a **new empty** output
directory. See [current training instructions](docs/INSTALL.md).
The default is to stop at the first 40M comparison, not launch an
unexamined overnight campaign.

For the **existing proven demo**, stay on `main` or visit the public
Playground. Neither source cleanup nor v5020 research modifies that policy.

## Where to look

| Question | Current implementation |
|---|---|
| What is physically happening? | `robot.xml`, `environment.py`, `terrain.py` |
| What does the actor sense? | `contracts.py`, `forward_tof.py`, `rolling_tof.py` |
| What actions are rewarded? | `rewards.py` |
| How does PPO learn? | `ppo.py`, `training.py` |
| What happened physically? | `scoring.py` (shared event contracts), `results.py` (rollout reports) |
| How are lessons adjusted? | `curriculum.py` |
| What does the logged command envelope suggest? | `competence_shadow.py` (read-only evidence; no certifications) |
| What can a frozen policy demonstrate? | `evaluation.py` |
| How are champions selected? | `campaign_refork.py` |
| How is one run planned, guarded and resumed? | `train_cli.py` |
| Where are checkpoints and run identities guarded? | `checkpoint_io.py`, `experiment_io.py` |
| Why did previous experiments fail or succeed? | `docs/RESEARCH_HISTORY.md` |

Old experiment-specific source remains accessible through the repository's
Git history. Current source comments should explain the current contract,
not reproduce that history.
