# Capability graph

## Geometry

The default grid is `21 × 15 × 25` over `(vx, vy, yaw-rate)`: 7,875 command nodes.

Velocity capability is a **node property**. Agility is an **edge property**.

## Node competence vs quality

The trainer stores these separately.

**Competence** asks a binary physical question over a sustained window: did the robot remain upright, supported and within command-tracking tolerances?

**Quality** records how accurately and cleanly it performed. A robot can therefore be highly reliable but mediocre quality, or vice versa, without corrupting the probability scale.

## Confidence and mastery

Success/failure evidence is exponentially forgotten with a default 20M-step half-life. The field reports both posterior mean and a conservative Wilson-style lower confidence bound.

A node is mastered only when it has enough recent effective trials **and** the conservative bound clears the mastery threshold. This prevents one lucky trial from turning a voxel green.

## Recovery

Recovery is currently an **integrated recovery** layer: the normal resilience birth may combine body tilt, position error, joint disorder, linear/angular velocity mismatch and pushes. The trainer intentionally does not pretend these are separate maps until the training curriculum actually separates them.

Each command node has six severity bands. One resilience birth produces one recovery verdict when competent locomotion is sustained again or the attempt fails first.

## Maneuver edges

A maneuver starts at a source command and has a target command/ramp rate. The field waits for the entire ramp and subsequent settling period before issuing one success/failure event.

Each source node has 26 directional sectors. Successful edges also accumulate demonstrated normalized command-change rate. Settling duration is part of the edge verdict, not separately stored telemetry.

## Sampling

Teaching anchors are chosen from a mixture of:

- learnable frontier cells;
- stale/regressing skills;
- unknown neighbors of confidently mastered territory;
- under-covered but already reachable regions;
- a small amount of mastered rehearsal.

The governor clips all field proposals to globally earned command/recovery safety envelopes.
