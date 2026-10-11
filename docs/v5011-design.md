# v5011 design — Honest Map

v5010 answered the architecture question: continuous node geography helped, the maneuver graph helped more, and the graph trainer reproduced from an independent cold start.

v5011 therefore changes the **measurement semantics and project architecture**, not the robot/network/reward fundamentals.

## Acceptance criteria

- two independent cold starts remain capable of reaching mature locomotion;
- policy capability does not materially regress against matching v5010 graph seeds;
- genuinely mastered nodes and maneuver edges become nonzero;
- predicted command competence calibrates sensibly against quarantined held-out controls;
- mastered-boundary expansion actually activates;
- edge evidence represents complete transitions, not frame quality;
- recovery evidence represents complete recovery attempts;
- field overhead remains small relative to PPO/MJX training;
- checkpoint/resume preserves network, optimiser, governor, normalizer and field exactly at the supported contract level;
- a new user can install, validate, train and view EMU without knowing the v500x history.

## Explicit non-goals

v5011 does not add terrain, near-fall replay, new sensor families, a learned curriculum neural network or broad new actuator randomisation. Those are valuable later experiments, but they would confound the test of whether an honest capability graph improves the already successful trainer.
