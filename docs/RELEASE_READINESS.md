# EMU release readiness — source, actor and hardware are separate

**The checked-in public demonstration is still the previously approved 120M H40 actor** at `showcase/champion.emupolicy`. The holistic 0→160M curriculum is a candidate training source. It is not itself a demonstrated 160M success, a new approved actor, or validated real hardware.

## Source promotion gate

Before a cleaned branch enters `main`, require:

1. The original cold-run SHA and complete 20/40/120/140/160M checkpoint/actor records remain archived and untampered.
2. Training-critical modules have demonstrably identical Git blob hashes to the source that trained the candidate, or changes have separately passed physical/reward/sensor and PPO parity tests. A different Git commit is still a different *checkpoint provenance*.
3. Clean WSL installation, CPU true-prism physics tests, 2,048-world CUDA PPO smoke and checkpoint/export/resume checks work without archived policies or secret local file paths.
4. The fresh seed-735 actor has credible independent uphill and downhill 4–80mm results **and** retains flat, command, slope and resilience competence. Treat failure to maintain downhill as a failed gate even if uphill improves.
5. Ideally the same frozen curriculum/source reproduces broadly similar capabilities at a **second random seed**. Network weights need not be byte-identical.
6. Explicit owner approval for one curated `main` PR, not five historical experimental PRs merged wholesale.

## Public actor publication gate

Merging training source does **not** replace the browser champion. Separately review the new `.emupolicy`, its full PPO source checkpoint, model/sensor contract, independently measured capabilities and matching `showcase/release_approval.json`. Run `python scripts/verify_showcase_release.py` and the separate approved GitHub Pages workflow; review the browser rendering and ToF interpretation. Do not auto-publish or overwrite `showcase/champion.emupolicy` during research cleanup.

## Physical hardware boundary

Simulated competence, MJX eight-prism contacts and a pretty WASM viewer cannot certify actuator thermals, physical spring preload, actual ToF zone calibration, collision parity or reliable stair traversal on hardware. A real-robot release also requires measured torque/geometry/BOM/wiring, fault handling and safe experimental hardware testing.

[Core skeleton](CORE_SKELETON.md) · [Physical quality](PHYSICAL_QUALITY.md) · [Research findings](RESEARCH_FINDINGS.md) · [Current handover](../handover.md).
