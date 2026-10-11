# Validation status

Before packaging, the v5012 source passed:

- Python bytecode compilation for all active modules;
- nine semantic/unit tests covering command competence, bad tracking, recovery events, maneuver events, conservative mastery, field persistence/schema refusal, teaching-packet shape, and the common matched recovery panel;
- public CLI parsing/listing checks;
- editable-install entry-point check in the build Python environment;
- live viewer HTTP checks, including a JSON waiting response instead of the old HTML-as-JSON failure;
- a synthetic full-size host capability-field benchmark at 32 frames × 2,048 environments: about 11.6 ms per PPO update on the build CPU.

The artifact build environment does **not** contain MuJoCo/MJX, so the real physics/JAX reset-step smoke cannot be executed here. `emu doctor` and the training preflight run those checks on the target machine before a long run begins.

The performance benchmark is a build-machine sanity check, not a guarantee for another CPU. It is intended to catch an obviously expensive host-side field implementation.
