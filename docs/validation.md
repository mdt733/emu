# Validation

A release candidate should pass, at minimum:

- Python compilation for every active module;
- the semantic/unit test suite;
- checkpoint exact-resume round trip and canonical discovery order;
- foundation/shadow/graph profile-contract tests;
- exact named 55-channel observation schema + shared task-packet layout test;
- tracking-contract tests, including oscillating error that must not cancel;
- one-frame maneuver registration;
- held-out field quarantine/fingerprint test;
- `emu doctor` on the target JAX/MJX machine;
- one tiny training/evaluation smoke before an expensive campaign.

The public CLI is intentionally small: `doctor`, `train`, `compare`, `view`, `evaluate`.
