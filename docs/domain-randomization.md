# Domain randomisation — the integrated physical diversity controls

**Current:** only JAX-valued robot, slope-equivalent, friction and actuator perturbations are part of training. The old host-side morphology/heightfield generator prototypes were **not integrated** and have been removed from current source; recoverable in Git history at commit `b0f31d8`. Do not describe them as training or evaluated capabilities.

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

The *historical* 400M severity schedule is documented in [v5014-production.md](v5014-production.md). The current v5015 ladder specifies arm-specific radii through named `TrainingConfig` overrides in [`emu/ladder_recipe.py`](../emu/ladder_recipe.py). Changing these radii inside an existing checkpoint run root violates the strict resume contract; see `emu/phase_config.py`.

## Archived, unintegrated experiments

The former `emu/morphology.py` could render host-compiled variations in limbs, hip width, foot size and torso distribution, while `emu/terrain.py` generated smooth 33×33 MuJoCo heightfield inputs. They were **standalone prototypes**: no trainer ever interleaved a morphology bank or placed those heightfields under EMU's current MJX worlds. Retaining the unused modules in the everyday code made those imagined capabilities look implemented.

The source and unit tests remain recoverable from [the pre-cleanup Git tree](https://github.com/mdt733/emu/tree/b0f31d865198f451414ec0d6e69e1793c69478a7/emu). Reintegration requires real MuJoCo contact correctness, a defined task distribution, measured MJX throughput and independent held-out validation; a generator by itself is not a training feature.

Avoid claiming simulated hardware-transfer robustness without frozen held-out evaluations. Candidate integration must demonstrate acceptable MJX throughput, nominal-regression preservation, and a measurable improvement on separately seeded morphologies and actual heightfields. A proposed cost gate is under 5% slowdown for simple parameter variation, roughly 10–15% for gentle heightfields, and a redesign review above 25%; **these are engineering budgets, not measured v5014 results**.

See [evaluation.md](evaluation.md) for the examiner/learner separation and [history.md](history.md) for why deployment truth is more important than flattering simulation metrics.
