# Documentation

Start with the [project introduction](../README.md), then choose the guide for the work you're doing.

| Need | Guide |
|---|---|
| What has actually been demonstrated? | [Status and limitations](STATUS.md) |
| Set up Python, GPU and browser dependencies | [Installation](INSTALL.md) |
| Understand module responsibilities and contracts | [Architecture](architecture.md) |
| Understand the PPO/curriculum/evaluation lifecycle | [Training](training.md) |
| Edit and run the experiment ladder | [Experiments](tournament.md) |
| Compare and interpret policy capability | [Evaluation](evaluation.md) |
| Preview/publish the protected browser demo | [Playground](PLAYGROUND.md) |
| Check code, GPU integration and release evidence | [Validation](validation.md) |
| Understand the robot and observations | [Hardware model](hardware-model.md) |
| Understand capability-map semantics | [Capability map](capability-map.md) |
| Understand integrated physical randomisation | [Domain randomisation](domain-randomization.md) |
| Inspect browser policy and controls | [Policy format](../viewer/POLICY_FORMAT.md), [Viewer](../viewer/README.md) |
| Read historical experiments and lessons | [History](history.md), [v5014 recipe](v5014-production.md), [v5013 data](../results/v5013_tournament/README.md) |

[Contributing](../CONTRIBUTING.md) explains the review/test expectations.

**Source-of-truth rules:** edit `viewer/`, not its generated packaging mirror. Update the current `ladder_recipe.py` *between* training campaigns, not during an active lineage. Code-contract claims belong with tests. Historical explanations should be marked as historical, not copied into live installation instructions.
