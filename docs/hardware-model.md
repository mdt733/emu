# Robot/model notes

EMU models a dynamically unstable six-actuator biped. The learning target is stepping/catching locomotion rather than static standing.

The actor is intentionally restricted to deployable sensor-style observations rather than unrestricted simulator truth. The current observation contract is defined in `environment.py`; checkpoint metadata records the observation/action dimensions so incompatible source/checkpoint combinations fail clearly.

`robot.xml` is the single physical-model source of truth. v5011 checkpoints also record its SHA-256 so a resume can warn when the physics source has changed.
