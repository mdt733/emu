# v5014 domain envelope — learn how a biped survives

v5014 deliberately asks a different question from the v5013 capability-teacher
tournament: **can one deployable 220-channel policy survive many slightly
different EMUs on many slightly different ordinary surfaces?** The goal is not
to prescribe one robust-looking gait. It is to make exact simulator-specific
gaits unreliable enough that the useful solution is feedback-driven biped
survival.

This work is additive. The v5013 tournament remains frozen and all new domain
randomisation defaults to OFF. When v5013 finishes, its evidence about capability
teaching/dose/schedules can be incorporated without invalidating that comparison.

## Three hidden radii, not dozens of knobs

Training exposes only three 0..1 severities:

- **R — robot:** mass/inertial uncertainty now; coherent manufactured morphology
  variants through a host-compiled bank.
- **T — terrain:** floor/foot friction and slope-equivalent gravity now; smooth
  low-frequency heightfield roughness after the throughput gate.
- **H — hardware:** per-servo gain/torque/zero spread and actuated-joint
  damping/friction/armature spread.

The actor never observes R/T/H or any sampled physical parameter. A large
near-nominal fraction protects the clean gait; a small hard tail makes the edge
of the learned envelope fuzzy rather than a cliff.

## Runtime MJX path (implemented in dev1)

MJX permits leading batch dimensions on JAX-valued model leaves. v5014 samples a
fixed physical world for every vector environment and batches only those leaves,
keeping one structural model and one XLA compile. At radius 1 the current
ceilings are intentionally moderate:

| Family | Radius-1 envelope |
|---|---:|
| equivalent ground pitch/roll | ±6° |
| floor friction multiplier | ±35% |
| each foot friction multiplier | ±20%, independently |
| global body mass scale | ±8% |
| per-body mass scale | ±4% |
| actuated-joint damping | ±18% |
| actuated-joint frictionloss | ±25% |
| actuated-joint armature | ±10% |
| actuator Kp | ±12% |
| actuator torque ceiling | ±15% |
| servo zero offset | ±1.5° |
| gravity magnitude | ±1.5% |

Slope uses a rotated gravity vector with the collision plane left flat. For an
infinite plane this is the same rigidly-rotated physical problem, but it avoids
mutating collision geometry inside JIT. The projected-gravity observation uses
the actual per-environment gravity vector, so the actor receives exactly the IMU
cue that a real tilted robot would receive; it is not told the slope angle.

These domain instances are fixed per vector world for dev1. With 2048 worlds the
policy still sees 2048 different physical plants in every PPO batch, while no
observation exposes world identity. Episode-by-episode model resampling can be
considered later if it proves useful enough to justify carrying model state more
dynamically.

## Coherent manufactured robots (generator implemented in dev1)

`emu.morphology` creates deterministic XML variants *before* `mjx.put_model`.
This is important: changing geom dimensions directly on device can leave MuJoCo
collision/inertial derived quantities inconsistent. Host compilation lets MuJoCo
recompute those quantities normally.

A variant jointly changes attachment positions, link collision geometry, link
mass (constant-density length scaling), feet and reset height. Common-mode and
left/right error are sampled separately, so the bank includes subtly crooked
robots rather than only perfectly symmetric big/small copies.

Current radius-1 bank envelope:

- thigh and shin scale: ~±3% common plus up to ±1.2% side asymmetry
- hip width: ±3%
- foot radius: ±4% common plus asymmetry
- torso mass: ±8%
- torso COM x offset: ±3 mm

The intended mature setup is a small bank (initially 16) of fixed held-out and
training morphologies. Compile them once, validate identical topology/structural
fields, then interleave groups/blocks rather than recompiling every episode.

## Reactive terrain, not blind stair climbing

EMU has no exteroceptive terrain sensor. Therefore terrain must be something a
biped can plausibly survive *after contact reveals it*:

- slope and cross-slope;
- broad low-frequency waviness;
- shallow bumps and hollows;
- raised seams / cable-like nuisance features;
- small local height mismatches under one foot;
- friction changes and left/right traction mismatch.

Stairs, gaps, stepping stones and tall curbs are deliberately excluded. Those
are perception/planning tasks, not proprioceptive robustness tasks.

`emu.terrain` now generates deterministic smooth 33×33 physical height maps with
millimetre-scale RMS roughness and a few broad nuisance features. Production
hfield wiring is intentionally behind a benchmark gate: MJX-JAX throughput is
very sensitive to contact complexity.

## Performance gates

v5013 flat training is roughly 32k env-steps/s on the current machine. New
physics does not get a free pass just because it is realistic.

- batched dynamics/mass/servo spread: target <5% loss;
- slope/friction: target <5% loss;
- smooth hfield roughness: target <10–15% loss;
- roughness + discrete nuisance features: tolerate ~15–20% only if held-out
  robustness gain justifies it;
- >25% loss: redesign, reduce contact complexity, or move that family to sparse
  training/evaluation blocks.

## Evaluation quarantine

Training randomisation must not grade itself. The v5014 evaluator should add
frozen, never-trained panels containing nominal, morphology, asymmetric,
slope, rough, low-friction, weak-motor, combined and intentionally-outside-envelope
cases. A useful headline is not just a scalar DR score but e.g. **15/16 frozen
virtual manufactured EMUs complete the panel**.

The nominal panel remains a hard regression gate: robustness is not a win if it
destroys ordinary locomotion.

## Integration after v5013

The capability-field tournament answers *where/when to practise*; the domain
envelope answers *which physical world practises it*. They remain orthogonal in
dev1. Once v5013 reports which teacher dose/mechanism survives held-out tests,
v5014 can decide whether R/T/H severity should be governor-owned, field-aware,
or simply sampled under a slow global schedule. Do not let the capability map
learn from a changing domain distribution until that choice is explicit.
