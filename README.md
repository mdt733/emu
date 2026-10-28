# EMU — learning to walk, recover and negotiate the world

**One dynamically unstable, six-servo biped. One neural policy. One physical learning curriculum.**

![EMU learning architecture](docs/images/emu-architecture.svg)

EMU learns to balance and move using reinforcement learning in MuJoCo/MJX. It has no hand-authored sequence of footsteps: a single policy learns how to coordinate six actuators in response to motion commands, inertial and leg feedback, and a simulated forward-looking depth array.

[Try the public playground](https://mdt733.github.io/emu/) ·
[Install / operate](docs/INSTALL.md) ·
[Learning architecture](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Research status](HANDOVER.md) ·
[Documentation](docs/README.md)

## Two realities: published robot and research robot

| | Published Pinnacle (`main`) | Research successor (`feature/v5028-considered-steps`) |
|---|---|---|
| Status | Protected reference and live browser demo | Experimental; **not released** |
| Observation contract | 232-input sensor policy | 360-input terrain-array policy |
| Actions | Six servo targets | Six servo targets |
| Research focus | Proven dynamic gait and high command envelope | Retain fast, confident walking **and** learn supported slopes, steps, resilience and energy-aware control |
| Policy replacement | None without owner approval | Must first beat the incumbent on independent **common physical tests** |

The predecessor's roughly **1 m/s requested forward command envelope** is an important target to regain. A large requested speed is **not** evidence of accurate measured speed: the comparison must record actual forward velocity, command error, falls, headings and mechanical effort at the same physical task. The new policy must not improve steps by becoming a permanently slow or timid walker.

**Nothing on an experimental branch replaces the public policy automatically.**
A candidate is promoted only after reproducible evaluation, review and explicit approval.

## What the robot learns

EMU uses one shared PPO actor/optimizer across six complementary teaching emphases, not six different brains.

| Teaching emphasis | What success actually means |
|---|---|
| **Commands** | Accurate forward, reverse, lateral and yaw requests across an expanding speed envelope |
| **Slopes** | Supported uphill/downhill/camber motion **and** accurate commanded progress |
| **Steps** | Genuine supported step **up and down**, separated by direction, height and approach |
| **Resilience** | Recover from measured disturbances and retain earlier competence |
| **Transitions** | Respond to changing commands without unstable overshoot or falls |
| **Control / efficiency** | Smooth, useful movement with bounded actuator effort and slip |

Every emphasis still rehearses other capabilities at easier difficulties. Frontiers are expanded based on observed physical outcomes, not simply elapsed steps. Harder practice is not a certificate of mastery.

### Real steps, not a decorative terrain label

The terrain course has physically colliding support surfaces, adjustable incline and an elevated/depressed step. A real crossing requires supported contact on the far side, not just surviving near the obstacle.

The 360-input policy receives simulated delayed/noisy measurements, including the forward **4×4 ToF array**. Privileged simulator coordinates may be used by the *teacher* to verify success, but do not become deployable neural inputs.

We score at least four different outcomes separately: physical crossing, mission/velocity tracking, heading/path retention, and safe recovery. A step **down** does not count as a step **up**.

## Current experiment: considered step-ups

The v5027 physical exam found that the strongest prior coherent policy crossed **2.0 cm uphill in 8/8 trials**, **2.5 cm in 6/8**, and **3.0 cm in 0/8**, under the tested +0.15 m/s, 16-world paired conditions. Surviving in front of 3 cm was **not** a deliberate, verified refusal.

The v5028 study tests whether per-frame forward-velocity pressure prevented careful foothold preparation:

- **A — coherent control:** original STEP reward, gentle coherent commands.
- **B — considered uphill:** same commands, but a tightly limited, near-lip reduction in *forward underspeed* pressure and supported-settling acceptance. No free standing reward, no relaxation of sideways/yaw, slip, overspeed or fall safety.

The two branches start from the **same fresh 20M-step cold foundation**, then train separately to **80M each** on **2,048 MJX environments**: **140M aggregate world steps**. Frozen physical exams independently compare real ascents, descents and grades. The user has reported launching the campaign; **results are not yet audited**.

See [experiment contract and source-pinned launcher](docs/V5028_CONSIDERED_STEPS.md). Don't pull commits into its checkout while it runs.

## Reading training results without fooling ourselves

The console reports time, rollout/reward, fall rate, throughput, command envelopes, terrain exposure and actuator-effort proxies. These are useful telemetry, but **curriculum difficulty is not measured proficiency**.

The key independent outcomes are:
- **Speed:** actual signed velocity and command tracking, including up to the ~1 m/s target.
- **Terrain:** supported UP/DOWN crossings at each exact height; true four-bearing slope missions.
- **Safety:** falls, slip, supported settlement, retreat/braking distance and recoverability.
- **Control:** sideways/yaw accuracy, smoothness, and shaft-work per useful metre.
- **Retention:** do all of those remain competent on easy known tests when harder skills grow?

The current training experiment does **not** implement reliable ToF-conditioned navigation refusal. That separate future milestone is described in [traversability and safe refusal](docs/TRAVERSABILITY_REFUSAL.md).

## Project map

```text
emu/
  environment.py       physical MJX worlds, sensor readouts, teacher outcomes
  rewards.py           PPO teaching payments
  curriculum.py        active difficulty and teaching governor
  capability_field.py  learned command/recovery competence evidence
  capability_cloud.py  read-only held-out command/recovery probes
  capability_focus.py  six teacher roles and episode allocations
  ppo.py               single shared policy, value model and optimiser
  training.py          rollouts, fixed evaluations, logging and checkpoints
  training_config.py   validated learning settings
  run.py, cli.py       guarded run/export/exam entrypoints
  evaluation.py        independent, read-only physical examinations
  viewer_src/          browser simulation and playground
scripts/
  v5028_considered_steps.py   active source-pinned experiment launcher
  smoke_unified_focus.py     CPU / CUDA physics + PPO preflight
  smoke_training_lifecycle.py
docs/
  README.md            current documentation index
  history/             source-pinned historical findings (not live recipes)
showcase/
  champion.emupolicy   protected, publicly approved Pinnacle
```

**Recommended entry points:** [Installation](docs/INSTALL.md), [physical capability inventory](docs/CAPABILITY_DIMENSIONS.md), [research handover](HANDOVER.md), [playground](docs/PLAYGROUND.md), and [release gates](docs/RELEASE_READINESS.md).

## Contributing and release discipline

Every behavioural experiment uses a **new feature branch** and an exact source SHA. Running branches and checkpoint lineages are never silently modified. Prefer simplifying or replacing old machinery when a new approach is proven; do not accumulate multiple overlapping "current" systems. Keep evidence and the commands that generated it in research history, not in the quick-start guide.

A beautiful merge is not just smaller code: it is **one legible implementation**, a believable physical comparison against `main`, reviewable provenance, clean docs and a working playground. The original champion stays protected until that standard is reached.

[Engineering maintenance rules](docs/MAINTENANCE.md) · [Graduation checklist](docs/RELEASE_READINESS.md)
