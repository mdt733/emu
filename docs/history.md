# EMU — lessons worth keeping

This is intentionally not a changelog of every experiment. Git contains the archaeology. This file records the ideas and failures that should continue to shape the project even after old implementation code is deleted.

## 1. Dynamic locomotion is the task

EMU has small feet, compliant mechanics and no reason to solve walking by freezing into a static pose. Training and evaluation should reward stepping/catching locomotion rather than accidentally making “stand still and avoid falling” the easiest solution.

A useful policy must remain controllable while moving, not merely survive.

## 2. Train with deployable information

The simulator knows far more than the real robot will. The actor therefore uses the deployable 55-value observation contract with four-frame history rather than privileged simulator truth.

Changes to this observation contract are architectural changes and should be explicit and tested.

## 3. Target slew was harmful

Earlier software command slew made training easier but taught the policy a world unlike the intended controller interface. Removing it improved real command responsiveness. If transitions need curriculum, train the transition itself rather than hiding it behind an artificial actuator-independent filter.

## 4. Separate global safety from local teaching

The governor owns broad difficulty and physical frontiers. The capability field owns evidence about where the policy has practised and what appears learnable.

The field may propose practice only inside the governor's legal envelope. Neither subsystem should silently take over the other's job.

## 5. Capability evidence must be semantic

A major failure in the first self-map treated thousands of correlated frames as if they were independent capability trials. This produced impressive sample counts with almost no statistical meaning.

Modern evidence is event-based:

- one sustained command window = one node verdict;
- one perturbation/recovery attempt = one recovery verdict;
- one complete source→target transition = one edge verdict.

Continuous tracking/posture remains useful **quality**, but quality is not Bernoulli competence.

## 6. Walking metrics must tolerate gait oscillation

Instantaneous velocity thresholds punished healthy stepping because body velocity naturally oscillates through the gait cycle. The v5011 field therefore believed a strong policy was almost completely incapable.

Node competence now uses windowed **mean absolute error** against windowed dynamic tolerances. Do not use `abs(mean(error))`: positive and negative errors would cancel.

The shared physical thresholds live in `tracking_contract.py`. The field and held-out node exam must agree on their event definition.

## 7. Calibrate like with like

A node probability cannot be validated against a recovery probability merely because both are between zero and one.

Keep separate calibration questions:

- node field ↔ held-out node-window exam;
- recovery field ↔ held-out recovery attempts;
- edge field ↔ held-out maneuver trials.

v5011 violated this principle and produced meaningless calibration numbers even aside from its overly strict node gate.

## 8. One-frame maneuvers are real maneuvers

A v5011 transition flag was computed after advancing the ramp. When the learned ramp duration reached one control frame, the transition completed before it was ever marked as active. The maneuver graph then stopped receiving new events precisely at its hardest frontier.

Command resampling is now explicit telemetry. A one-frame jump must start an edge event and count toward transition/frontier evidence.

## 9. Recovery means return to competent locomotion

Survival alone is not recovery. Recovery should require a sustained return to broadly upright, supported command-tracking locomotion.

Keep state recovery, external disturbance and model uncertainty conceptually separable even when a report later combines them.

## 10. The exam is quarantined

Held-out evaluation is an exam, not curriculum data. It must never mutate the capability field, governor or optimiser.

A field fingerprint is checked around held-out cloud evaluation as defence in depth. Architecture should also keep evaluator APIs read-only wherever practical.

## 11. Frontiers must refer to physical capability

Legacy fixed suites such as `dynamic80` are useful regression coordinates but are not physical limits. Report meaningful frontiers in m/s, rad/s, newtons, disturbance angles and related units.

If a frontier does not move, determine whether the policy is incapable, the governor is starved of valid evidence, or a safety cap is binding before drawing conclusions.

## 12. A shared foundation enables causal comparisons

The useful trainer experiment is:

```text
same seed → same exact foundation handoff ancestor → shadow vs graph
```

Within the pair, network, optimiser, curriculum, training-health, normaliser and field ancestor are identical. The field's permission to teach is the intended difference.

Environment-step targets are not necessarily PPO update boundaries. Never treat the mere existence of a resumable checkpoint as proof that a phase is complete: a partial checkpoint may resume the phase, while the fork may use only the validated checkpoint on the exact shared update boundary.

Run independent seeds rather than pretending tiny PPO differences are deterministic effects.

## 13. The nursery does not need time travel

During early cold-start training, falling is normal and competence can improve continuously while a collapse detector still fires. Rewinding an immature learner to another immature snapshot adds intervention without protecting anything valuable.

The foundation may retreat curriculum difficulty and save snapshots, but network/optimiser rollback is disarmed until the foundation handoff.

## 14. Resume must be strict

A long GPU experiment must never silently interpret “resume failed” as “start a fresh run.” Missing metadata, incompatible observation/action contracts, invalid field state or a corrupt checkpoint should fail loudly.

Rotating snapshots may provide explicit validated recovery from a damaged latest checkpoint.

## 15. Provenance must describe executable truth

Source version, Git commit, dirty state, robot model hash, checkpoint schema, observation/action dimensions and field schema matter.

Do not maintain fragile hand-written lists of which files count as source when Git can answer that question.

## 16. Evaluation cost is part of trainer design

Earlier campaigns spent a majority of wall time evaluating rather than learning. Continuous training-derived instrumentation can reduce routine examination, but held-out final evidence still matters.

Always distinguish raw train throughput from whole-process wall throughput.

## 17. Avoid opaque automated winner selection

A lexicographic ranking was useful when EMU ran large automated branch tournaments. The current scientific question is smaller and clearer.

For paired shadow-vs-graph experiments, report the deltas directly: capability calibration, recovery, worst direction, physical frontiers, falls, tracking and efficiency. Do not compress the conclusion into an arbitrary winner scalar unless a future automation problem genuinely needs one.

## 18. Delete experimental archaeology

Old phase importers, compatibility filenames, abandoned viewers, tournament logic and one-off ablation flags once served useful purposes. They do not need permanent residency in the current architecture.

Git is the archive. Current code should express the current system. Reintroduce old experiments one at a time when a concrete question requires them.

## 19. Derived quantities are not rewind state

If a quantity can be reconstructed exactly from authoritative state, do not persist a second decaying copy merely for convenience. Duplicate denominators, weights and reporting-only timing arrays make checkpoints larger and create invariants that can drift.

Likewise, a diagnostic belongs in checkpoint state only when future training behaviour actually depends on it. Held-out reporting can recompute its own summaries; the training rewind should contain the smallest complete state needed to continue the same learning process.

## 20. Learning-state rewind is not physics-frame replay

A PPO checkpoint needs the state that can change future learning: network and optimiser, observation normaliser, curriculum, training-health/LR controller and capability field. Live MJX environment buffers, current observations and rollout RNG are process-local episode state. Resume deliberately starts fresh episodes, so serialising those buffers only makes checkpoints larger and more device-fragile without making continuation more correct.

Be precise in language: EMU supports an exact **learning-state** rewind, not a claim that a restarted process reproduces the exact same simulator frame.

## 21. Fixed defaults are doctrine, not hidden knobs

A default argument that every supported run leaves untouched is not useful configurability. If it defines current PPO/network behaviour, give it a name, use it directly, record it with the run, and make a future change explicit. This applies just as much to gamma, GAE lambda, clip epsilon, gradient norm, network shape and exploration bounds as it does to curriculum thresholds.

This keeps the public/internal configuration surface small while improving provenance: supported profiles contain only real choices; fixed scientific assumptions remain visible and reviewable.

## 22. A teacher can corrupt its own evidence distribution

v5012 ran two exact shared-ancestor shadow-vs-graph comparisons (seeds 733 and 1742). At 65% maximum field influence, graph teaching consistently increased training-derived node competence/frontier coverage and sometimes improved hard growth tests, but it also consistently weakened hard recovery/robustness. More importantly, the field's held-out node calibration worsened in both seeds while its internal confidence increased.

That is the signature to watch for when a self-map both chooses experience and learns from the experience it chose. The field remains useful, but teaching dose and evidence provenance must be treated as causal variables. v5013 therefore tests dose first, then teaching family, then schedule. One arm explicitly allows field-selected practice to train PPO while withholding that self-selected experience from field evidence.

Do not infer that more known/mastered field cells imply a better general policy. Held-out recovery, robustness and calibration remain the independent judge.

## Milestones

- **v5007** — no-slew training and repaired physical frontier logic established a strong reference policy.
- **v5008/v5009** — recovery experiments and evaluator profiling exposed the cost of broad campaign machinery.
- **v5010** — introduced the self-mapping trainer and demonstrated the value of exact shared-ancestor forks; frame-weighted evidence proved statistically misleading.
- **v5011** — event-based Honest Map; cold-started a strong policy, but exposed two critical measurement failures: an overly strict instantaneous node doctrine and invisible one-frame maneuvers.
- **v5012** — windowed node competence, explicit one-frame transitions, strict resume/quarantine checks and the first proper shadow-vs-graph comparison.
- **v5013** — current-only cleanup: real package/resources; compact training profiles plus named doctrine; separate PPO, curriculum and training-health ownership; read-only data-driven fixed evaluation; schema-7 exact learning-state rewind; field schema 6; an explicit named 55-channel actor contract; a 317-value task packet with dead assist/rehearsal dimensions removed; strict loaders; compact learning-state-only checkpoints; canonical checkpoints; and removal of obsolete experiment/compatibility machinery.
