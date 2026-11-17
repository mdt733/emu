# EMU documentation

The supported project is a **single H40 robot, 360-input actor, PPO learner, holistic cold-start curriculum, independent real-prism examination, and browser viewer**. The checked-in champion may be older than the best experimental actor; source promotion and public showcase publication are deliberately separate.

| Need | Canonical document |
| --- | --- |
| Understand components and sensor/teacher boundaries | [Core skeleton](CORE_SKELETON.md) |
| Install on Windows/WSL, recover a source-pinned run safely | [Installation](INSTALL.md) |
| What counts as reliable balancing, stairs and recovery | [Physical quality](PHYSICAL_QUALITY.md) |
| Explore Meadow, physical physics limitations and sensor lab | [Playground](PLAYGROUND.md) |
| Coding, branch and cleanup rules | [Maintenance](MAINTENANCE.md) |
| Approve and publish a validated browser policy | [Release readiness](RELEASE_READINESS.md) |
| Evidence from older hypotheses and measured regressions | [Research findings](RESEARCH_FINDINGS.md) |
| Where we are now and what remains | [Current handover](../handover.md) |

The one training recipe and quick commands are in the [root README](../README.md). Training and exam results belong under `~/emu-data/`, not in the source tree. Git and result ZIPs preserve finished experiments; there are no permanent per-iteration runbooks.
