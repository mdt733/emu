# EMU documentation — current H40 mainline

The maintained entry points are the [core skeleton](CORE_SKELETON.md), [H40 handover](../handover.md), and [install/recovery guide](INSTALL.md). These are the source of truth, not historical experiment notes.

| Question | Canonical page |
| --- | --- |
| What is the 40 cm robot, sensor, world, actor and learning loop? | [Core skeleton](CORE_SKELETON.md) |
| Which checkpoint/policy and source are the current baseline? | [H40 handover](../handover.md) |
| How to install, run, verify, or safely realign a diverged local checkout? | [Install and recovery](INSTALL.md) |
| Which code and tests remain supported? | [Maintenance](MAINTENANCE.md) |
| What actually counts as physical competence? | [Physical quality](PHYSICAL_QUALITY.md) |
| What is live in the browser versus the MJX trainer? | [Playground](PLAYGROUND.md) |
| What still separates source promotion from GitHub Pages publication? | [Release readiness](RELEASE_READINESS.md) |

**Current source:** H40 120M was promoted to `main` via [PR #53](https://github.com/mdt733/emu/pull/53), commit `1b796144b128b326be1ac54f6a289ad87857cde6`. The policy is checked into `showcase/champion.emupolicy`; the GitHub Pages site remains independently gated pending the original training checkpoint SHA-256 and release approval.

**Known limitation:** the owner reports a good gait and about 15° slopes in the viewer, but robust stair traversal and held-out equivalent MJX physical exams remain open.

Older experiment writeups are historical evidence only; Git preserves their original source and PR discussions. Do not execute an old experiment launcher as if it were a current training recipe.
