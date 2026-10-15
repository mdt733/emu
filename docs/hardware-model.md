# Robot and deployable observation model

EMU models a dynamically unstable six-actuator biped whose primary task is **stepping, steering and recovering**, not freezing in a stable pose. The robot and simulator model are defined by [`emu/robot.xml`](../emu/robot.xml) (15 generalized coordinates, 14 velocities and 6 actuators in the operator's installed MuJoCo 3.15.0 smoke test).

The policy deliberately sees only deployable sensor-like signals. **55 named per-frame features × four oldest-to-newest frames = 220 inputs.** The canonical names/order are in `emu/contracts.py`; the exported browser actor has **six clipped action outputs** that map to joint targets through the control contract. Do not change observation order or servo semantics without an explicit schema migration and parity tests.

Checkpoints record contracts and robot-model provenance so an incompatible resume can be refused. Changing a robot XML parameter during an active production lineage is not a harmless cosmetic edit. By contrast, the browser's grid, trail, sky, lights and shadows are **render-only** and do not alter the training world or policy observations.

For physical variation that is currently integrated, see [domain-randomization.md](domain-randomization.md). For the actor-only browser JSON format, see [viewer/POLICY_FORMAT.md](../emu/viewer_src/POLICY_FORMAT.md). No real-hardware transfer result is established by this document.
