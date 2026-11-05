# EMU engineering rules — one understandable system

**Current baseline:** v5032 Pinnacle 260M, 360-input actor,
training-source commit `9b11bbe596aea7201d29a6beb2d192922f8e2e9a`.
The previous 232-input public policy remains recoverable at
`2ec9e9db0a816c658131311129ef89ec7a595a12`.
Future work is prioritised in [issue #17](https://github.com/mdt733/emu/issues/17);
the 24-arm v5033 branch is **paused**. The v5046 34-arm benchmark
campaign completed, and its temporary runner is removed on PR #36;
the 21 actual core pytest cases remain.

## One owner per concept

| Responsibility | Code |
|---|---|
| Body, six servos and passive springs | `emu/robot.xml` |
| Actor 360-input contract and sensor schema | `emu/contracts.py` |
| Real MJX physics, contacts, sensors and resets | `emu/environment.py` |
| Per-world model randomisation | `emu/domain_randomization.py` |
| Teaching reward | `emu/rewards.py` |
| Teacher difficulty and six focus groups | `emu/curriculum.py`, `emu/capability_focus.py` |
| PPO KL stress, collapse and recovery | `emu/training_health.py` |
| One neural actor/critic and PPO updates | `emu/ppo.py`, `emu/training.py` |
| Source-verified checkpoints and resume | `emu/run.py`, `emu/checkpoint_io.py` |
| Held-out training metrics and independent frozen-policy prism exam | `emu/evaluation.py`, `emu/prism_exam.py` |
| Browser emulator and approved public export | `emu/viewer_server.py`, `emu/viewer_src/`, `scripts/verify_showcase_release.py` |

[Full architecture and data flow](CORE_SKELETON.md).

The actor gets only sensor/command observations; the teacher
can use privileged physical contacts for reward and curriculum.
Independent exam results NEVER feed back into its teaching.

## Code and experiment hygiene

1. **One owner, not another wrapper.** Change curriculum when
   selecting lessons; rewards when paying for performance;
   environment/contracts when physical or sensory semantics change.
2. **Source-pinned research.** A trained model's original Git
   SHA, saved learner/config and checkpoint hashes are immutable.
   Updating today's `main` does not retroactively update its
   released weights.
3. **Do not duplicate experiments.** The finished 260M launcher is kept at the original Git SHA, not
   executed from active `scripts/`. New training starts with `emu train`.
   The large old A/B launchers are retained at their Git commits.
   The former v5045 140M, v5046 20M and v5048 eight-fork scripts are
   removed from the future-main cleanup branch; the running v5048
   experiment remains unchanged at its original source SHA.
4. **Fix the real bug first.** A historical v5031-era attempt
   trained with `verified_ascent_reallocation=0.0` despite
   its manifest requesting 3.0. That option is now retired. Future commands must prove
   that **resolved TrainingConfig and actual environment**
   match the requested treatment, not just a CLI echo.
5. **Real physical evidence.** Report actual supported upper
   contact, distance past a lip, speed vs requested motion,
   steering bias vs gait jitter, survival and mechanical J/m.
   Harder curriculum numbers alone are not ability.
6. **Back up full learner state.** Git stores source, exported
   browser weights and report identities, not a full `.pkl`
   optimizer. The complete `final.pkl` lives outside Git.
7. **Release deliberately.** An actor update requires a
   provenance-matching `showcase/release_approval.json`,
   validated browser build, explicit owner decision, and a
   tested old-actor rollback source. GitHub Pages deployment
   is intentionally separate from a `main` merge.

## Canonical workspace and outputs

One Git checkout `~/emu/` owns source plus `.venv`, never trained data.
All new training, checkpoint recovery, physical exams, reports and exports
belong together under `~/emu-data/runs/<study>-s<seed>/`.
Short-lived GPU A/B outputs go under `~/emu-data/scratch/`;
**do not invent extra run, result, artifact or work roots**.
The `--run-dir` argument in `emu train` and `--root` in `emu report`
refer to the same run directory. The absolute run path and exact Git
SHA are printed and stored; a run is never silently migrated between
roots. No wrapper per experiment unless the CLI genuinely cannot
express the physical recipe. The one-source setup and backup guide is
[INSTALL](INSTALL.md).

## Documentation ownership

The [README](../README.md) welcomes new contributors;
[ARCHITECTURE](CORE_SKELETON.md) maps code;
[HANDOVER](../HANDOVER.md) reports the current baseline;
[release measurements](V5032_PINNACLE_RELEASE.md) record
the exact policy, physical limitations and SHA;
[INSTALL](INSTALL.md) explains setup;
[issue #17](https://github.com/mdt733/emu/issues/17)
holds future experiments. Completed old research belongs in
[history](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/history/README.md) or an immutable Git commit.

Avoid new permanent docs for every small experiment. A
stranger should have ONE clear answer to *what is current*
and *which file owns it*. Remove replaced implementations in the
same change; never keep duplicate experimental launchers in active code.
The maintenance workflow is: **hypothesis → temporary runner →
measurements → issue/release findings → delete runner**. Never delete
the measurements when retiring the experiment.
`bash scripts/check.sh` is quick by default, including the source
ownership inventory and current-document links; `--full` runs the
maintained Python and browser contracts. The complete
[live ownership map](CORE_SKELETON.md#owners-of-the-25-live-python-modules)
records current owners, retired files and outstanding physical gates.
Code promotion to `main` is separate from selecting a replacement
inference actor. Both require explicit review; an ordinary `main`
merge must not silently publish a new GitHub Pages champion.

## Preserved historical incident

The first v5032 run stopped before PPO training because
the launcher set `EMU_UNIFIED_CONTINUATION` instead of the
required `EMU_UNIFIED_CONTINUOUS`. The repaired second
cold-start run completed at 260,046,848 world steps.
The old run logs, source hash and handover are archived at
[docs/history/HANDOVER_V5032_PRE_RELEASE.md](https://github.com/mdt733/emu/blob/a2f59bb31bec049204b60a3eb86500b100db039f/docs/history/HANDOVER_V5032_PRE_RELEASE.md).
