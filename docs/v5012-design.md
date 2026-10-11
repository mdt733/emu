# v5012 design — Trustworthy Measurement

v5011 proved that a cold-start EMU policy can become a strong dynamic controller, but its live self-map did not agree with the held-out behaviour. v5012 therefore changes measurement and experimental trustworthiness before adding any new curriculum mechanism.

## Findings that motivate v5012

The v5011 seed-733 graph run reached roughly saturated survival on the ordinary held-out suites and good command tracking, while the field ended with zero mastered command nodes and zero mastered maneuver edges. Mean node quality improved strongly but mean Bernoulli competence remained near zero. The final calibration compared node competence with recovery success, which was also a semantic mismatch.

Separately, the acceleration frontier eventually reached one-frame command ramps. The environment only marked a transition after advancing the ramp fraction, so a 0→1 one-frame ramp was never marked as a transition. New edge evidence consequently disappeared at exactly the hardest maneuver frontier.

## Semantic fixes

### Command nodes

A node event remains one clean, sustained one-second command window. Its tracking verdict now uses the mean absolute vx/vy/yaw error across the complete window compared with the mean dynamic tolerance on each axis. Broad upright/support/contact health must be good on at least 90% of the window.

This makes the event gait tolerant without turning continuous quality back into pseudo-Bernoulli evidence. A stepping controller may oscillate around a requested velocity every stride while still having excellent mean command tracking.

Any post-settle fall is negative node evidence even if the full one-second window has not completed. Fast failures must become failures, not unknown cells.

### Maneuver edges

The environment exports an explicit `cmd_resampled` metric and defines `transition_frac` from the pre-advance ramp state. Therefore a one-frame ramp is still one real transition frame. The edge graph and acceleration frontier see the same event.

### Calibration

Node competence is calibrated against a held-out node-window exam that uses the same one-second aggregate definition as the live field. Recovery remains a separate held-out recovery exam. The evaluator stays quarantined: its outcomes never update the live field.

## Fail-loud contracts

A missing required field metric raises immediately. Capability-field checkpoint arrays must all be present and shape-compatible. Unknown future optional FieldConfig keys are filtered rather than crashing.

`--resume` is exact-resume only: missing checkpoints, missing metadata, schema/robot/observation contracts, non-finite network state, field restore failures, optimizer restore failures and observation-normalizer restore failures all stop the process. `resume=auto` may use a validated rotating snapshot when the primary latest checkpoint is poisoned, but it never silently starts at step zero.

## Experiment

The intended v5012 result is a causal shadow-vs-graph comparison, not a one-arm selection:

```bash
emu compare
```

For each seed (default 733 and 1742):

1. train one genuine 0→20M foundation;
2. fork the exact checkpoint, including network, optimizer, governor and normalizer;
3. continue one branch with the field in `shadow` mode;
4. continue the other with the field in `graph` mode;
5. compare on identical held-out evaluation.

The longer hard fixed suites run at the forced final exam so mature policies are not all compressed at approximately 100% survival.

## Deliberate non-changes

v5012 does not add terrain, new observations, reward changes, near-fall replay, a learned curriculum network or new disturbance families. Those would confound the measurement repair and shadow-vs-graph test.
