# Evaluation

Evaluation is deliberately quarantined from training.

## Fixed regression panels

`emu/evaluation.py` defines deterministic panel data and fixed evaluation worlds once. Core, dynamic, resilience, combination and growth panels are longitudinal regression rulers: the questions stay fixed while the policy changes.

These panels can select `best.pkl`, but their answers never update PPO, curriculum, health or the capability field.

## Live field calibration

Routine held-out reviews choose a few command coordinates from different predicted competence bands and ask: **is the self-map calibrated?**

Reported calibration includes raw predicted/observed pairs, mean absolute calibration error and a Brier-style squared error. Node predictions are compared with held-out node-window events; recovery predictions are kept as a separate question.

## Common recovery exam

The forced final recovery exam uses the same moderate 3-D command panel for every policy. Undisturbed controls are screened first, and expensive disturbed rollouts are spent only on commands the policy can already execute competently.

The common recovery exam compares recovery under common commands; it is not a physical-limit search.

## Physical atlas

`emu/atlas.py` is the expensive final map of physical-unit command and recovery frontiers. It is appropriate at major milestones, not as a curriculum signal.

For lateral and yaw sweeps the atlas keeps a +0.08 m/s forward command, so those reported frontiers mean "while moving slowly forward" rather than "from static standing."
