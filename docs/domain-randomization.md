# v5014 domain randomisation — physical diversity without hidden controls

**Status:** the fast MJX/JAX-valued parameter randomiser is integrated into the v5014 `emu forge` lineage. A host-built morphology generator and a low-frequency terrain-heightfield generator also exist, **but their full bank/heightfield physics are not integrated in the default production training run**. Don't describe those generators as active training variation.

## Why it exists

The same deployable **220-value actor input** must control modestly different robots and contact conditions. The actor is not given the randomisation seed, sampled parameters or domain severity. Ordinary sensor observations must reveal the resulting physical effects. v5013 capability teaching answers *where to practise*; v5014 domain variation answers *which imperfect physical world to practise in*. Their boundaries stay distinct.

## Three independent radii

- **R — robot (0–1):** mass/inertia-style spread, gravity-magnitude variation.
- **T — terrain (0–1):** equivalent floor tilt via world-gravity orientation, floor friction and independent left/right foot friction.
- **H — hardware (0–1):** actuated-joint damping/friction/armature, servo gain, torque ceiling and zero offset.

`emu/domain_randomization.py` defines the numerical caps. At radius 1 they are:

| Variation | Upper magnitude |
|---|---:|
| Equivalent pitch/roll slope | ±6° |
| Floor friction | ±35% |
| Each foot friction (independent) | ±20% |
| Overall body mass | ±8% |
| Per-body mass | ±4% |
| Gravity magnitude | ±1.5% |
| Actuated-joint damping | ±18% |
| Actuated-joint friction loss | ±25% |
| Actuated-joint armature | ±10% |
| Servo proportional gain | ±12% |
| Servo torque ceiling | ±15% |
| Servo zero position | ±1.5° |

The default mixture reserves a near-nominal population and a small difficult tail. Individual environment instances in the current fast path hold their sampled model throughout that vector-world lifetime; the actor receives no explicit domain ID. Slope is gravity-vector equivalent on a flat collision plane, **not actually sloped collision mesh geometry**. At the production phase boundaries, the ±6° per-axis cap scales with terrain severity: at 60M (`T=0.08`) that is at most ±0.48° per axis, rising to ±4.2° at 400M (`T=0.70`). A browser 1° physical ramp introduces height transitions and contacts that the infinite-plane approximation does not teach. That approximation is intentional to avoid structural model changes inside the JAX compile.

The production severity schedule is in [v5014-production.md](v5014-production.md): R/T/H are zero in foundation, then increase at 60/140/260/400M. Changing radii in an existing run root invalidates its intended learning-lineage contract; see `emu/production.py`.

## Separate development layers, not yet default forge

- `emu/morphology.py` can generate *host-compiled*, coherent robot variants with changes to limb lengths, hip width, foot radius, torso mass and COM. Structural geometries need MuJoCo's derived inertial/collision data recomputed before an MJX stack is built. This generator is **not equivalent to training a 16-morphology bank**.
- `emu/terrain.py` can generate smooth low-frequency 33×33 heightfields and nuisance features; the default forge fast path does **not** train on that terrain. No exteroceptive preview exists, so tall stairs, gaps and blind jump planning are outside the current problem.

Avoid claiming simulated hardware-transfer robustness without frozen held-out evaluations. Candidate integration must demonstrate acceptable MJX throughput, nominal-regression preservation, and a measurable improvement on separately seeded morphologies and actual heightfields. A proposed cost gate is under 5% slowdown for simple parameter variation, roughly 10–15% for gentle heightfields, and a redesign review above 25%; **these are engineering budgets, not measured v5014 results**.

See [evaluation.md](evaluation.md) for the examiner/learner separation and [history.md](history.md) for why deployment truth is more important than flattering simulation metrics.
