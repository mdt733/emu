# v5020 winner-refork campaign — preserve the good walker

This is a **new training campaign on the existing
\`feature/v5020-coldstart-terrain\` integration Git branch**. It does not alter
\`main\`, the GitHub Pages showcase or the historical \`emu campaign5020\`
380M-step experiment. "Branches" here mean PPO checkpoint continuations,
not Git branches.

The design borrows the successful v5016 Pinnacle approach from \`main\`:
start from one genuine sensor-trained actor, keep a **general walking
control**, challenge its skills in small controlled variations, and compare
policies only in independent physics. What we deliberately improve is winner
reuse: **one** winner from four candidates becomes the ancestor for **all
four** candidates in the next generation.

## Actual generations

| Generation | Policy target | Four experimental variants | New physical obstacles? |
|---|---:|---|---|
| walk40 | 40M | general, turns, recovery, efficiency | No |
| walk60 | 60M | general, turns, recovery, efficiency | No |
| walk80 | 80M | general, turns, recovery, efficiency | No |
| terrain100 (opt-in) | 100M | flat_control, step3, step5, step5_efficiency | 0/3/5mm |

The actor is **312 inputs / six actions throughout**. Its ToF/optical-flow
sensor contract and physical robot hash are validated at every checkpoint
handoff. 20M is *not* a magical competence boundary: a validated **12–22M**
foundation snapshot can be used, including an earlier healthier checkpoint
from a run that deteriorated before its final 20M save.

Each stage runs:

1. Four independent 2M-environment-step **probe phases**, all warm-started
   from the *same* frozen parent checkpoint.
2. Each probe must show a mean recent fall rate at most 92%, mean episode
   length at least 70 frames, and no logged mean spring potential above 50J.
   Values are preliminary safety limits, not robotics performance claims.
   A rejected probe **never trains its full stage**.
3. Surviving probes finish their continuation to the stage's absolute
   40/60/80/100M policy step boundary. The probe/finish phases have separate
   immutable training directories to avoid mutating a training-config's
   \`steps\` mid-run.
4. Frozen candidates face **identical**, truly held-out locomotion tests:
   easy core, disturbance recovery, command combinations, two *real-contact*
   slope distributions, plus servo shaft-effort telemetry. **No obstacle
   geom exists in the walking examiner.**
5. The strongest **eligible** candidate becomes the one ancestor for the next
   generation. The previous incumbent is evaluated on the same tests to
   catch gross regression. If no candidate qualifies, **STOP** and preserve
   the parent, rather than guessing or inventing a winner.
6. Promotion is locked to exact checkpoint SHA-256 and saved scoreboard;
   running again cannot quietly change the chosen ancestor.

No adaptive physical step can enter training until \`walk80\` has an eligible
winner with >=80% core survival, >=5% hard recovery survival, >=10% real
contact-slope survival and adequate stepping evidence. These floors should
be revised only with measured training evidence, **before** locking a new
campaign root. The first terrain generation uses a matched flat control and
only 3–5mm step lessons. Its independent physical examiner checks a
**genuine 5mm supported crossing**; a 20mm obstacle is *not* the first lesson
any more.

## Why the old 20M-to-80M launch was unsafe

The failed v5020 campaign inherited a poorly generalised 4096-world
foundation and instantly combined: mature command envelopes, strong pushes,
heavy combined command sampling, new graph teaching, real slopes and adaptive
3–30mm physical boxes. Every branch reported near-immediate catastrophic
falling. The new campaign retains true 312-channel sensing and light slopes
from the start, but makes the first 20M continuation essentially the old
foundation's **level/general locomotion** curriculum. It ramps command range,
combined commands, disturbances and graph teaching conservatively between
generations. Terrain training is a later, **separately gated** skill.

4096 worlds use only about 153 PPO updates per 20M versus about 305 at
2048 worlds, with the existing 32-frame rollout. The governor's 70-update
steady-only bootstrap therefore occupies proportionally more of 4096-world
foundation. This coordinator deliberately **preserves the selected
foundation's world count**. Changing world count should be a separate study;
it is not a silent checkpoint migration.

## Inspect first; no GPU training

Read-only plan:

\`\`\`bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only
source .venv/bin/activate
emu campaign-refork
\`\`\`

Read-only status, including before a campaign exists:

\`\`\`bash
emu campaign-refork --root ~/robot_lab/emu_runs/v5020_refork --status
\`\`\`

Inspect the existing foundation snapshots before deciding which to reuse:

\`\`\`bash
find ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints \
  -maxdepth 1 -name '*.pkl' -printf '%f\n' | sort
\`\`\`

A final 20M policy **is not necessarily the best foundation**. Choose a
validated snapshot from the actual independent locomotion evidence where
possible; this coordinator does not pretend the best checkpoint can be
selected merely by its highest step count.

## Foreground GPU launch, only after smoke tests/CI

With the original **4096-world** foundation:

\`\`\`bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
emu campaign-refork --run \
  --foundation ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints/final.pkl \
  --envs 4096 --seed 733 \
  --screen-envs 32 --screen-horizon 300 \
  --stop-after walk40 \
  --root ~/robot_lab/emu_runs/v5020_refork
\`\`\`

\`--stop-after walk40\` is an intentional first acceptance point. Inspect
the four probes, the four independent score reports, and the actual winner
before scheduling 60M or 80M; passing source tests alone is **not**
evidence of learned locomotion.

After a satisfactory 40M generation, use the **same arguments and source
commit**, changing only \`--stop-after walk60\` (then \`walk80\`). All
completed phases, probe verdicts and sealed selections are reused.

After the 80M winner passes its distinct terrain-entry gate, the same
command may be resumed with \`--stop-after terrain100\`. This first physical
stage contains a matched no-step control and an actual collidable 5mm step
exam. It does **not** imply 20mm or taller obstacle competence.

If using a 2048-world foundation instead, change \`--envs\` to 2048 **and
start in a different fresh root**. Input schema and environment count are
both checked.

## Artifacts and safety

\`\`\`text
v5020_refork/
  refork_context.json              # pinned SHA, source, actor count, examiner settings
  walk40/
    general/
      probe/{training_config.json,metrics.csv,checkpoints/final.pkl}
      probe_health.json
      finish/{training_config.json,metrics.csv,checkpoints/final.pkl}
    turns/... recovery/... efficiency/...
    scoreboard.json                 # immutable SHA-verified examination verdict
    scoreboard.md
    selection.json                  # winner SHA, parent SHA and scoreboard SHA
  walk60/                            # four new children of walk40 winner
  walk80/                            # four new children of walk60 winner
  terrain100/                        # only after walk80 qualification
\`\`\`

There is **one foreground GPU job at a time**, guarded by \`flock\`. No
training weights are stored in Git, and no checkpoint is overwritten by
another experiment. Training/process failures retain the last validated
checkpoint for exact resume. Source changes during an active campaign are
refused by \`refork_context.json\`: complete, stop, or create a **new**
campaign root before changing the code.

The 2M probe is designed to catch the kind of immediate collapse in the
failed 80M tournament; it cannot guarantee later training stability.
Physics-divergence diagnostics and finite-but-absurd spring-energy metrics
remain an active research issue. A competent gait in the browser is also
not a replacement for the real MJX examination.

## Regression checks

\`\`\`bash
python -m pytest -q tests/test_v5020_refork.py tests/test_v5020_campaign.py
\`\`\`

These tests cover configuration/observation consistency, four candidates
per generation, single-winner checkpoint lineage, early rejection, no
unearned terrain entry, and safe plan/status before GPU work. The full
repository/CI test suite and a small **real MJX** acceptance are also needed
before an overnight run.
