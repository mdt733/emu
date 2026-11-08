# EMU documentation — current system

Start at [CORE_SKELETON](CORE_SKELETON.md), [the active v5058 handover](../handover.md), and [INSTALL](INSTALL.md). Those three are the navigational source of truth.

| Question | Canonical page |
| --- | --- |
| How do the body, sensors, world, actor, PPO and evaluator fit together? | [Core skeleton](CORE_SKELETON.md) |
| Which robot and run are we working on? | [Current handover](../handover.md) |
| How do I install, train, verify or recover? | [Install](INSTALL.md) |
| What code and tests earn their place? | [Maintenance](MAINTENANCE.md) |
| What counts as physical success? | [Physical quality](PHYSICAL_QUALITY.md) and [release gates](RELEASE_READINESS.md) |
| What is the browser actually simulating? | [Playground](PLAYGROUND.md) |

Older experiment writeups are historical evidence, **not** parallel training instructions. Git preserves every implementation and retired one-off runner. The current candidate is the **40 cm C018 rated-torque** body on v5058. `main` and the public actor remain protected pending independent, equivalent tests.
