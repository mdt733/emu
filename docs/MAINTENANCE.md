# EMU engineering hygiene — add a capability, retire an obsolete path

**Goal:** a public-ready project whose active implementation and documentation
can be understood without knowing the history of v5015 through v5028.

## Non-negotiable constraints

1. **Training provenance:** a launched PPO campaign is tied to a clean,
   exact Git SHA, actor/input contract and checkpoint lineage. No commit,
   pull, rebase, hotfix, cherry-pick or test cleanup in its checkout until
   the source-pinned campaign has ended. Polish happens on a sibling branch.
2. **One policy:** the six-servo 360-input successor has one actor and
   one PPO optimiser. Specialist focus is teacher-side; privileged terrain
   facts may teach/evaluate but cannot silently enter deployment inputs.
3. **Evidence:** teacher targets, learned physical capability and a published
   policy are distinct states. The 1m/s commanded envelope is a target
   to recover with *measured tracking*, not a reason to trade away
   2.5–3cm step ascents, resilience, heading or sensible shaft work.
4. **Protection:** never overwrite `main`, the approved 232-input
   `showcase/champion.emupolicy`, source-frozen policies, or checkpoints
   during cleanup. Promotion needs a separate common-physics comparison,
   explicit approval and a rollback path.

## Before merging any new capability

- Identify the one canonical owner module; remove stale alternate
  implementations and experimental wrappers, unless genuinely shared.
- State exactly what is actor input, teacher-only privileged state, PPO
  reward, evaluator-only evidence, or browser navigation logic.
- Keep **baseline behaviour bitwise or contractually unchanged** when
  the experiment is off. Verify what happens on flat, down-step, slope,
  non-STEP and frozen checkpoint cases.
- Preserve only meaningful checks of reward mathematics, real contacts,
  source pin, export/resume and schema compatibility. Avoid tests
  tied solely to an obsolete launcher or historical filename.
- A cold 2048-world CUDA gradient smoke, same-SHA resume, policy
  export and independent physical exam must pass on the future
  candidate before promotion. Python syntax checking is not a GPU test.
- Update **one** maintained doc, the canonical
  [handover](../HANDOVER.md), and the current
  [README](../README.md) only when externally meaningful.
  Completed experiment narratives belong in [history](history/README.md).

## File and entrypoint policy

- `emu/environment.py`: true physics, observations and physical evidence
- `emu/rewards.py`: teaching payments only
- `emu/curriculum.py` / `emu/capability_focus.py`: difficulty and roles
- `emu/training_health.py`: PPO health/rollback decision logic
- `emu/capability_field.py`: stored training-only physical competence evidence
- `emu/capability_cloud.py`: held-out evaluation; never writes back into teaching field
- `emu/training.py` / `emu/ppo.py`: one learner and policy lifecycle
- `emu/evaluation.py`: independent, frozen and directionally honest exams
- `scripts/campaign_support.py`: shared foreground logging/reporting,
  never an experiment itself
- `scripts/v5028_considered_steps.py`: current study, explicitly SHA pinned
- `docs/history/`: prior run papers/proposals and immutable numeric findings

The v5024/v5025/v5026/v5027 standalone launchers were **retired on the
independent polish branch**; they remain reproducible from their historical
Git SHAs. Their shared reporting functions live in
`scripts/campaign_support.py`. Nothing here is a request to mutate
the *already running* v5028 experiment.

## Definition of a worthy replacement for `main`

Measured signed forward speed/heading near the incumbent's high-command
envelope **plus** improved physically supported terrain traversal.
Specifically preserve/expand walking reliability, command tracking,
step-up (not just down), four-direction inclines, perturbation recovery
and effort per useful metre. Require same task conditions, per-world
traces, more than one seed, and a 232-versus-360 neutral benchmark.
The candidate should be better at *walking and solving obstacles*,
not merely better at one teaching score.

## After the ongoing v5028 campaign ends

1. Keep its exact training SHA and checkpoints immutable.
2. Audit real A/B signed ascent, tracking, recovery, log/provenance,
   and any unexpected report failures from its original code.
3. Decide which careful-step behaviour to retain; do not infer that a
   training-reward improvement is a physical competence gain.
4. Reconcile this **post-run** polish branch with the selected later
   research source. Any PPO-altering edit requires a new training
   lineage; do not resume old weights across incompatible source.
5. Only then consider a protected `main` PR and new public policy
   after the release checklist is satisfied.
