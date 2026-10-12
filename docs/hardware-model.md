# Robot/model notes

EMU models a dynamically unstable six-actuator biped. The learning target is stepping/catching locomotion rather than static standing.

The actor is intentionally restricted to deployable sensor-style observations rather than unrestricted simulator truth. The exact 55-channel observation order is defined once in `emu/contracts.py`; checkpoint metadata records the observation/action dimensions so incompatible source/checkpoint combinations fail clearly.

`emu/robot.xml` is the single physical-model source of truth. the current trainer checkpoints also record its SHA-256 so resume rejects a checkpoint when the physics source has changed.
