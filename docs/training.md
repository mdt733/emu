# Training lifecycle

## Cold start

The normal product path always supports random actor/critic weights.

- **0→20M:** conservative nursery. The capability field observes but cannot steer.
- **20M onward:** field influence ramps in and the graph trainer becomes the default curriculum locator.
- **120M:** primary finite-GPU trainer benchmark.
- **220M (optional):** asymptotic extension for physical-limit hunting.

## One PPO update

At default settings:

```text
2,048 envs × 32 frames = 65,536 fresh transitions
```

The update then:

1. computes PPO advantages/returns;
2. updates actor/critic over minibatches/epochs;
3. summarizes rollout health for the governor;
4. converts ordinary rollout telemetry into semantic capability events;
5. updates the decaying capability graph;
6. builds the next fixed-size teaching packet;
7. advances the next rollout.

No periodic evaluator is needed to make this inner loop adaptive.

## Capability events

v5012 deliberately avoids treating correlated frames as independent trials.

- **Node event:** about one second at a stable command → one success/failure verdict.
- **Recovery event:** one integrated resilience birth → one recovered/failed verdict.
- **Edge event:** one command ramp + settling period → one maneuver verdict.

Continuous tracking/support/upright quality is recorded alongside these events but does not alter the meaning of success probability.

## Resume semantics

A resume restores network/value parameters, optimiser state, observation normalization, governor state/curriculum and the capability graph. MJX physics buffers are intentionally recreated as fresh episodes because device-specific `mjx.Data` is not a stable checkpoint contract.
