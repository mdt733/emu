# V5033 — 260M Pinnacle plateau science

Status: branch-only research, not yet GPU-launched. Refer to [issue #17](https://github.com/mdt733/emu/issues/17). Do not modify main, the published champion export, or the rollback source.

## Rewind identity

Parent: original v5032 completed checkpoints/final.pkl — **not** the .emupolicy viewer export.

- Original training source: 9b11bbe596aea7201d29a6beb2d192922f8e2e9a
- Exact parent step: 260,046,848; original seed 761; 2048 worlds; terrain_360.
- Exact checkpoint SHA256: 238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b (already preserved by release-approval CI).
- Rewind all actor/critic/optimizer/normalizer/governor/health/field state. Keep the acquired frontier heights.
- Clear any *inherited partial step-evidence cohort once* when forking the released source, identically for all arms. Do not clear evidence when resuming that arm's own checkpoint.

The launcher validates the bytes, checkpoint schema, exact original source and age, original training configuration, clean experiment source, and its immutable manifest. No source-mismatched automatic resume.

## Design

24 continuations from 260,046,848 to an absolute target of 280,000,000 steps, each with its own result directory. This is ~19.99M additional world steps per arm, about 479M new training world steps total. **No new 20M foundation.**

First 16: resolution-IV fractional-factorial screening of eight named switches, main effects balanced and pairwise orthogonal. Two-factor interactions are NOT all separately identifiable.

| Factor | Default at 260M | Active level |
|---|---|---|
| Step promotion | +max(1mm,18% frontier) | +1mm |
| Step retreat | ×0.82 | −1mm, existing 1.5mm floor |
| Near/easy evidence | 4 attempts per direction | 12 per direction |
| Verified ascent | 0 | 3.0 points reallocated from existing nine-point shaping |
| Crouch reward | Original height penalty | 4cm additional reward tolerance at a real uphill lip |
| Fast rehearsal | 25% | 40% |
| Temporal velocity reward | 0.50 | 0.80 |
| Steering | Balanced | Steering specialist attention |

The bounded crouch option changes **only** height-related reward shaping on a supported, non-fallen, actively approaching true uphill step, at most 120 control frames. It does not relax physical fall/contact constraints, script a jump or change the actor observation.

Eight extra targeted comparisons: control with seed 762; ascent credit 1.5 and 4.0; 10% fast rehearsal; terrain-early attention; integral velocity multiplier 1.5; stronger push frontier; motor guidance penalty 0.025.

Every arm keeps all other resolved settings from the actual original saved config. An independent-seed control helps estimate training randomness, but one seed per treatment is exploratory evidence only.

## Foreground WSL commands

Do not switch away from a working branch with unsaved changes. From your usual EMU checkout and Python environment:

~~~bash
cd ~/robot_lab/emu
git status --porcelain
git fetch origin
git switch --track origin/feature/v5033-260m-plateau-science
git rev-parse HEAD
source .venv/bin/activate

# Show the 24 arms without a checkpoint or GPU:
python scripts/run_plateau_science.py --matrix

# Check the exact original final.pkl, source and run identity:
python scripts/run_plateau_science.py --verify-only --arm 0

# Run just the baseline control first (2048-world real GPU PPO):
python scripts/run_plateau_science.py --run --arm 0

# Then run/restart all 24 sequentially, one GPU training at a time:
python scripts/run_plateau_science.py --run --all

# Regenerate the report and comparison ZIP without training:
python scripts/run_plateau_science.py --report
~~~

If a local branch of that name already exists, use git switch feature/v5033-260m-plateau-science followed by git pull --ff-only instead of --track.

By default the launcher searches ~/robot_lab/emu_runs for the *unique* matching completed checkpoint. If yours is elsewhere, append:

~~~bash
--parent /absolute/path/to/checkpoints/final.pkl
~~~

This is an argument to the Python command, **not** a standalone shell command.

The explicit --run switch is required. No tmux, daemon or automatic background work. Ctrl+C stops the foreground campaign. Reissue the same command with the same clean Git commit to resume from the last valid snapshot, not from a different source or recipe.

Artifacts go to ~/robot_lab/emu_runs/v5033_plateau_260_to_280/, outside the repo. It includes the source-and-parent-sha manifest, arm-specific config, metrics, held-out fixed_eval, checkpoints, console logs, read-only exports and report ZIP. Finishing an arm does not promote any public policy.

## Examination before claiming progress

All arms run the same built-in frozen fixed evaluation at 10M cumulative-step intervals. Evaluate promising exports again using genuine CPU MJX terrain matrix and speed-ladder exams, on independent seeds 1701 and 1702, and summarise through emu physical-quality.

The primary step plateau question is whether 3cm *UP supported crossing* improves beyond the released 0/8 while 2cm crossings, controlled heading, downhill path, signed speed and recovery stay intact. Also assess world-frame route drift, spring load-and-release traces, gait yaw jitter versus persistent signed bias, and J per **useful** supported metre. PPO reward, teacher ceiling, legacy heading jitter gate and survival alone are not valid ranking criteria.

A successful arm is a hypothesis, not a replacement Pinnacle. After identifying and replicating good changes, combine them in a new clean cold-start run and compare the resulting physical capability profile to the unmodified released 260M actor.
