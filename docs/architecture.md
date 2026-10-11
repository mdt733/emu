# Architecture

EMU has four deliberately separate control loops.

## 1. Vectorised MJX worlds

The default trainer advances 2,048 robot worlds in parallel. An environment that falls or times out is reset in-place while the rest of the vector batch continues. PPO collects 32 control frames per update, so one update contains 65,536 fresh transitions.

## 2. PPO

The actor/critic is on-policy. Each rollout is used for PPO updates and then discarded as training data. There is no giant replay buffer silently retraining on ancient experience.

## 3. Regime governor

The governor controls global training safety and progression: steady/transition/recovery allocation, global difficulty, command frontiers, push/sensor frontiers and collapse/retreat logic.

It answers **how adventurous training is allowed to be**.

## 4. Capability graph

The capability graph answers **where that training budget should be spent**.

Its geometry is a low-resolution 3-D lattice over command space:

```text
node = [vx, vy, yaw rate]
```

Each node stores clean command competence, quality, confidence and recovery evidence. Twenty-six directional edge sectors from every node store complete maneuver outcomes and demonstrated transition rate.

The graph produces a tiny fixed-size teaching packet (24 node anchors + 24 edge anchors) for JAX. It never enters the actor observation.

## Held-out evaluation

Evaluation is a one-way observation path. Its results are written to reports but never update the graph/governor/policy. This is what lets us detect self-map overconfidence and curriculum overfitting.
