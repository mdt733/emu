# `.emupolicy` format — schema 1

The file is UTF-8 JSON. It is deliberately not pickle, ZIP, ONNX, or a Python-specific object graph.

Required top-level keys:

```text
format = "emu-policy"
schema = 1
source
policy
observation
control
features
robot_xml
```

Float32 tensors use:

```json
{
  "dtype": "float32-le",
  "shape": [220, 256],
  "data_b64": "..."
}
```

Schema 1 actor contract:

```text
base observation: 55
history:          4, oldest -> newest
actor input:      220
MLP:              220 -> 256 -> 256 -> 6
hidden activation:tanh
actor output:     deterministic mean, clipped [-1,1]
```

`observation.base_fields` is authoritative. A viewer must reject an incompatible field list/schema rather than silently reinterpret it.

`control` records simulation timestep, policy/control cadence, joint names, target spans, delay doctrine, command limits and nominal command defaults.

`features.pitch_tracking_trained` distinguishes **an input channel existing** from **a policy having been explicitly trained/evaluated to obey it**. This matters because v5013 already contains `cmd_pitch` but does not explicitly reward chassis pitch tracking.

## v5014 compatibility rule

If v5014 retains the 55-field observation order and 6-action actuator contract, it may continue to emit schema 1 even when domain randomisation changes training physics. New optional metadata can be added without changing schema.

Increment the schema only when a viewer would otherwise misinterpret the actor interface, for example:

- observation field ordering/dimensionality changes
- action semantics/dimensionality changes
- network architecture is no longer the schema-1 tanh MLP
- control-delay/history semantics change

The viewer should fail closed on an unknown schema.


## Optional evaluation metadata (v0.2)

Bundles exported from a tournament may include `evaluation`, `tournament`, and
`training` objects. They are informational and never enter policy inference.
Older schema-1 bundles without these objects remain valid.
