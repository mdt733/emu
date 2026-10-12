# Training lifecycle

## Foundation: 0→20M target

The foundation learns basic dynamic locomotion and creates the exact common ancestor for later comparisons. PPO updates are indivisible vectorised batches: the handoff is therefore the first complete update boundary at or beyond 20M (20,054,016 steps for the default 2048×32 batch), and both mature branches must use that same validated checkpoint.

- capability field observes in `shadow` mode;
- curriculum may expand or retreat;
- useful snapshots are saved;
- **network/optimiser rollback is disabled**;
- no historical reference checkpoint is required.

## Mature training: handoff→target

The exact shared foundation handoff checkpoint can continue as:

- `shadow` — field measures but does not steer;
- `graph` — field proposes bounded node and maneuver practice.

The governor still owns legal physical frontiers in both arms. Training health remains separate from curriculum and may request rollback only in mature training.

The mature profile freezes observation-normalizer running statistics at 25M absolute steps. This deliberately fixes the actor's coordinate system before the mature command envelope expands much further; it is a testable profile hypothesis, not an accidental trainer default.

## One update

1. Collect one vectorised MJX rollout under the current policy/task packet.
2. Perform PPO optimisation from that rollout.
3. Reduce the rollout into training metrics and semantic capability events.
4. Update training health and curriculum using the exact rollout provenance.
5. Build the next fixed-shape capability teaching packet.
6. Periodically checkpoint and run quarantined held-out exams.

The metrics, normaliser, curriculum state and field state used to judge a rollout are kept aligned with the policy that actually generated it.

## Resume

Resume is an exact learning-state continuation, not an exact physics-frame replay. Current schema 7 restores network, optimiser, normaliser, curriculum, training health and capability field or fails loudly; live MJX episode buffers restart fresh by design. Rotating-snapshot fallback considers only fully schema-valid snapshots.

Run-local evaluation history is also restored from its artifacts so evaluator cadence and best-checkpoint selection do not reset merely because the Python process restarted.

## Rollback

Rollback is an emergency health mechanism for a policy that had useful competence and later collapses. Curriculum retreat is the ordinary response to tasks becoming too hard; rollback rewinds the learned state itself.

The foundation disarms rollback. Mature training keeps one canonical `recovery.pkl`, ranked monotonically so process resume or branch fork cannot replace a stronger recovery point with a weaker one.

## v5013 teacher experiment surface

The normal mature profile remains deliberately narrow. The v5013 tournament varies only capability-teacher dose, teaching family, evidence feedback and influence timing. These fields live in `TrainingConfig` because the experiment genuinely varies them; PPO, reward, robot physics, governor/frontier doctrine and observation-normalizer freeze remain fixed.

A teacher-only `field_taught` task provenance bit records whether the current command target was chosen by the capability field. It is not part of the 220-value actor observation and does not enter reward. The `B4_clean_evidence` arm uses that provenance to prevent self-selected practice from updating the field while still allowing the same experience to train PPO.

See `docs/tournament.md` for the complete funnel.
