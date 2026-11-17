# EMU H40 — concise handover

**Product goal:** reproducible, one-policy locomotion for the 40 cm H40 six-servo biped: stand, walk commanded vx/vy/yaw, tolerate disturbances and slopes, then negotiate physical uphill *and* downhill steps without sacrificing easier skills.

## Current authority

- `main` retains the prior approved H40 120M actor in `showcase/champion.emupolicy`. Source promotion does not replace the public demonstration.
- The newer **holistic cold curriculum** is one source-pinned **0→160M random-initialisation** learner, with teacher stages at 20M (gait fade), 40M (outcomes), 120M (physical step approach), 140M (both-foot transfer), and a final 160M held-out exam. It does not import a 120M/140M/160M research checkpoint.
- The **first candidate run** is against the unmodified `work/emu-holistic-coldstart` source. Treat its own `run_meta.json`, immutable stage manifests and Git SHA as authoritative. Don't silently resume its checkpoints on the separate clean-integration branch.
- `work/emu-main-clean-integration` is the **curated, main-based proposal**: carry across the new training implementation unchanged, remove one-off experiment scripts, refresh docs and tests. This branch has a *different Git SHA*, despite byte-identical training-critical files. It is not yet approved for merging.
- Preserve all 120M, 140M and 160M historical reference actors, their source hashes and result ZIPs outside Git. Findings are in `docs/RESEARCH_FINDINGS.md` and original Git/PR history.

## Evidence that drives decisions

Earlier matched 2,000-stair + 2,000-flat physical exams:
- At **140M**, 44–60mm UP had 20.0% verified both-foot support and 1.6% aligned traversal; DOWN 44–80mm had 50.0% aligned traversal.
- At **160M**, UP 44–60mm improved to 34.4% both-foot support and 8.8% aligned traversal, but UP survival fell to 47.2% and DOWN aligned traversal regressed to 27.2%.
- The holistic experiment specifically addresses that tradeoff with **uphill Academy practice capped at 55mm, downhill practice at 80mm, extra core-control retention after 140M**, and full independent tests. Higher Academy reward or assigned step height is not proof of real crossing.

## Next decisions

1. Do not change the source or delete checkpoints underneath the running original cold training.
2. Compare independent held-out UP, DOWN, flat and general capability results at 120/140/160M; inspect failures by actual height and foot-transfer stage.
3. Verify **main-based clean integration** with a fresh install, source parity, real CPU MJX contacts and 2,048-env CUDA PPO/export/resume smoke. Treat passing CI as software validation, not successful cold locomotion.
4. Reproduce with a second independently randomised seed on **the same frozen revision** before asserting seed robustness.
5. Request explicit owner approval before a curated merge to `main`; separately approve a new showcase actor.

## Operator shortcuts

```bash
bash scripts/setup_wsl.sh --cuda13 --install-node --check
bash scripts/run_h40_holistic_cold_wsl.sh plan
bash scripts/run_h40_holistic_cold_wsl.sh all
source .venv/bin/activate
emu dashboard --run-dir ~/emu-data/runs/h40-holistic-cold-s735 --once --plain
emu viewer showcase/champion.emupolicy --no-open
```

Architecture: [docs/CORE_SKELETON.md](docs/CORE_SKELETON.md). Hardware assumptions and unresolved calibration: [docs/MAINTENANCE.md](docs/MAINTENANCE.md). Source-specific history lives in Git; do not accumulate new branch-per-tweak launcher files.
