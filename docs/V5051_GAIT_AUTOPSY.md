# v5051 — gait hypotheses and falsification

This file is the **research rationale**, not another test runner. The
[60M → 120M training campaign](V5051_TRAINING_CAMPAIGN.md) owns the executable
protocol. The released `main` policy remains unchanged.

## Source-grounded reasons the tippy-tap may be difficult to unlearn

- **Activity reward:** after one simulated second, the episode-average
  stepping-activity gate falls to 0.30 if activity is absent. It multiplies
  posture (0.68) and contact (0.50). For otherwise perfect qualities the
  maximum payment difference is `(0.68 + 0.50) * 0.70 = 0.826` per
  20 ms control frame. This is an economic bound, **not causal proof**.
- **External phase:** the 360-input actor includes sin/cos clock observations.
  The source clock advances at low/zero movement requests by default.
  Phase sync also enters a weighted lift payment with maximum explicit
  contribution `0.16 * 0.45 * 0.30 = 0.0216` per frame.
  The phase is an input/reward cue, not a direct actuator oscillator.
- **Crouch economics:** current height shaping begins penalising shortfall
  beyond 12 mm (with some pitch allowance), potentially discouraging
  spring-loading or deliberate lowered posture.
- **Perception:** the 4×4 ToF produces 16 range and 16 validity observations
  with acquisition delay and dropout. Having a sensor does not establish
  useful early step recognition or causal usage.
- **Contact physics:** foot-prism physical top contact is measured, but
  fast-simulator thigh/shin rods are noncontacting with virtual strikes.
  Upper-foot touchdown is not itself centre-of-mass load transfer.
- **Prior evidence:** the v5048 100M 1-mm ordinary-frontier arm produced
  21/32 worlds with ≥2 cm supported uphill encounters in nominal-3cm panels.
  These are **actual physical height world incidences**, not completed
  3cm step missions. The 200M temporal candidate still had roughly
  8.5 combined footfalls/s at zero command. See
  [v5048 shared-parent results](V5048_SHARED_PARENT_RESULTS_20261105.md).

## Specific learnable experiments

The campaign's 15 source-pinned arms test, against their own unmodified
60M→120M control: zero-only/all-commands activity gate removal; low-speed
phase slowing/pausing; removal of phase synchronisation pay; explicit
stillness births; STEP-only bounded crouch allowance; actual supported
foothold, bilateral contact and 50cm forward follow-through payments;
ToF-conditioned authentic foothold teaching; ordinary-frontier increments;
Academy lesson density; useful forward progress per lift; and two clearly
labelled non-isolated combinations.

The phase features are **not deleted** from the 360-input network;
architecture-changing phase deletion would need matched cold-start actors.
The ToF arm does **not** implement a privileged critic or new sensors.
The 1-mm frontier arm changes the *ordinary* step frontier, while the
Academy already uses 0.5 mm growth. Only new physical measurements, not
training rewards or requested lip labels, can establish improvement.

## Preserve / retire

**Keep:** one real-mission autopsy (`emu/step_autopsy.py`), the existing
independent prism examiner, one actual-height wall physics screen, and one
small test module covering study config, genuine MJX stepping,
reward gating and negative physical mission verdicts.

**Temporary:** the v5051 experiment driver itself and all per-arm outputs
outside Git. Delete the runner after experiments and verified results have
been archived. The older 44-panel frozen-only campaign is removed from this
research branch: it does not train the robot and duplicated the examiner.

A supported top contact is evidence of contact. A completed 50cm mission
is evidence of controlled post-lip progress. Neither alone proves a full
weight transfer, toe/riser clearance or success on an **exact** 3cm wall.
