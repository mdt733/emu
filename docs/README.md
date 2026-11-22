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
| What did earlier experiments actually prove or disprove? | [Research findings](RESEARCH_FINDINGS.md) |

**Current source:** H40 120M was promoted to `main` via [PR #53](https://github.com/mdt733/emu/pull/53), commit `1b796144b128b326be1ac54f6a289ad87857cde6`. The policy is checked into `showcase/champion.emupolicy`; its original training checkpoint SHA-256 and strict release approval are recorded. Actual GitHub Pages publication remains a separate manual workflow.

**One active improvement branch:** `research/h40-unified-school-cold-v1`
([draft PR #103](https://github.com/mdt733/emu/pull/103)).
It keeps the same one 360-input actor/PPO and explores a coherent
survival-aware school on opt-in Spring-only physics. Its ongoing
0→120M sanity run was launched from **source
`1fa832fc7cd9241e2772b44ad434613c6080adfa`**, not the
subsequently edited branch head. New source commits are **not
interchangeable** with that run's resumable checkpoint. The full
current state is in the [handover](../handover.md). No other research
branch is actively maintained.

**Known limitation:** the owner reports a good gait and about 15° slopes in the viewer, but robust stair traversal and held-out equivalent MJX physical exams remain open.

Older experiment writeups are historical evidence only; Git preserves their original source and PR discussions. Do not execute an old experiment launcher as if it were a current training recipe.
