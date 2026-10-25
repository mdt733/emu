# EMU — complete growing-capability map

Updated 26 October 2026. This is a **physical-task inventory**, not a promise
that every listed axis has an independent curriculum dial. The canonical
philosophy is [UNIFIED_CAPABILITY_GROWTH.md](UNIFIED_CAPABILITY_GROWTH.md);
implementation status belongs here. All learning uses **one actor/optimizer**.

## The common rule

A target is where the *teacher trains*, **not proof of mastery**. For each
capability, sample easy practice + around-target lessons + occasional hard
probes. Attribute actual completed physical outcomes to the exact attempted
severity and direction. Repeated near-target success nudges up; failed harder
probes usually hold; failure of genuinely easier challenges nudges down.
Use a small increment and independent slowly changing evidence windows.
Do not wait for one specialist to graduate another, and do not claim success
from requested (rather than achieved) motion. No target must continually
rise; it can oscillate and resume growing. The same NN learns every cohort.

Cold source: first 20M environment steps mainly teach basic gait. Specialist
activation then blends from 0 to 35% across another 10M steps. This changes
the **frequency of challenging lessons**, not earned proficiency. No later
absolute-step unlocks. A same-source completed run can extend its target
and resume without changing physical state or optimiser identity.

## Capability-dimension inventory

The third column is the proposed independent target count, **not** the count
implemented today. Directions continue to be sampled even when limits share
one dial.

| Family | Real physical challenges | Independent targets today | Candidate independent targets |
|---|---|---|---|
| Commands | +vx, −vx, +vy, −vy, +yaw, −yaw; combined axes | Forward / reverse independently, one ±vy, one ±yaw, one combo radius = **5** | Six signed direction limits + mixed radius = **7** |
| Recovery / resilience | Joint-angle birth offset; body tilt; root xyz offset; linear velocity kick; angular velocity kick; torso shove | One linked resilience-difficulty scalar and one separately controlled push-force limit = **2** | Six disturbance magnitudes = **6** |
| Slopes | Uphill, downhill, cross-left, cross-right | Four *measured* outcomes, one shared maximum angle = **1** | Four directional maxima = **4** |
| Sharp steps | Up, down, oblique up, oblique down | Up/down contact outcomes, one shared maximum height and random lip yaw = **1** | Four near-height envelopes = **4** |
| Dynamic commands | Time before switching, ramp speed, size of command jump | One transition difficulty plus two coupled timing values; **one practical family** | Hold interval, ramp time, command delta = **3** |
| Posture / chassis | Chassis pitch, lateral roll, commanded height | One shared posture-difficulty scalar = **1** | Three separately controllable envelopes = **3** |
| Grace / efficiency | Useful work per distance while tracking; rate of actuation / jerk | Dedicated control lesson and measured shaft work, **zero earned limits** | Work budget and smoothness budget = **2** |
| Sensor robustness | Measurement noise / corruption scale | One scalar = **1** | One limit = **1** |

**Working goal: ~30 candidate targets** (7+6+4+4+3+3+2+1).
This is a useful map of physical challenges, **not 30 new trainer classes**
or a demand to immediately alter the sensor/network contract. Do not
multiply this into a 30-dimensional curriculum tensor: sample one or a few
challenge dimensions per episode and combine them gradually.

### Evidence and dependencies

**Commands:** ask for a signed command and measure *sustained actual*
body-heading vx/vy/yaw error within tolerance, contact quality, support,
falls and episode duration. A requested +0.4m/s is not success if the robot
actually shuffles +0.1m/s. For combined commands, track the vector, not just
its largest axis. The current signed command ledgers are complete-episode
near/easy probes, with an independent small target nudge. Left/right lateral
and yaw limits are still symmetrical **by policy**, not separate dials.

**Resilience:** physical resets already randomize roll/pitch, horizontal/vertical
position, six actuated joint starts, linear velocity and angular velocity.
The code still scales most together through `resilience_difficulty`;
`RESET_JOINT_NOISE` and `RESET_QVEL_NOISE` are small fixed background terms
beside those difficulty-scaled kicks. A separate push-force frontier exists;
the push is applied to the robot in physics. No direct physical evidence
controller yet knows the maximum tolerable *individual* birth noise.
Eventually score recovery from each family and nuisance direction at its
attempted intensity while protecting reliable normal gait.

**Slopes:** real two-slab contact inclines; azimuth determines uphill,
downhill and lateral camber. Today the four episode outcomes independently
contribute to evidence but must all succeed to increase **one** maximum
angle. Current target uses measured actual physical inclines, distinguishes
easy versus near-target, can retreat below the initial 0.7° probe.
A four-direction target array is future work, not implemented yet. Angles
should be relative to robot heading for stringent directional exams.

**Steps:** real discontinuous slab lip, not an obstacle sticker. Actual
up/down surface difference is measured, and only physically completed,
supported near-height crossings may advance the shared height. Current
training lip orientation is random at about ±20°, **not** ±45°; a fixed
exam override can request a specified lip yaw. Four distinct height/approach
targets need an intentionally broadened sampler and separate attribution.
Do not credit flat, partial, or down-only contact as step-up competence.

**Transitions:** currently command hold duration and ramp length move
together using tracking and fall-based feedback; command jump magnitude is
not its own adjustable target. Eventually distinguish a sudden 0.1m/s change
from a 0.5m/s reversal and track convergence time plus support.

**Posture:** requested chassis pitch, roll and commanded height exist in
task sampling, but share one posture scalar. Step swing-height variation is
also tied to this scalar. Keep chassis pitch distinct from the robot's
feet-midpoint-to-torso angle. Do not manufacture a new degree of freedom
solely to have another number to display.

**Control/efficiency:** control-focused worlds get a modest additional
motor-effort penalty and smoothness shaping, and produce shaft-work and
movement-quality diagnostics. This is **not an adaptive efficiency frontier**
yet. A safe candidate target is *mechanical work per achieved supported
distance, at adequate commanded speed and tracking accuracy*. A second
candidate limits action-rate / jerk (and perhaps peak servo overload).
The economy target may slowly tighten after good accurate episodes or relax
when it starts degrading reliable locomotion. Zero motion is never efficient
walking. Shaft-work proxy is not battery energy.

**Sensors:** sampled observation corruption scale has an earned frontier.
Separate sensor delay, dropouts, mass, actuator and contact-friction domain
randomization remain optional future *robustness envelope* axes; they are
not currently 30 extra dimensions and should not be enabled just to inflate
the count.

## How to count honestly

- **Targets**: trainable dials. One up/down slope target counts as one even
  when evidence is measured for four directions.
- **Challenge directions and severities**: data bins for decisions and exams;
  not automatically independent targets.
- **Lesson allocation**: how frequently a focused cohort is born. Not a
  skill and not earned proof.
- **Quality**: accuracy, support, fall rate, mechanical work, smoothness,
  recovery duration. These qualify attempts; they are not all frontiers.
- **Proficiency map**: independent held-out physics exams; it must not
  silently replace the teacher's local completed-episode evidence.
- **Global protection**: numerical PPO health, real collapses, source-pinned
  checkpoint/recovery. Do not mistake normal infant falls for failure.

## Implementation order before the first long cold GPU run

1. Keep cold gait, 10M specialist exposure fade, source-pinned one-run
   checkpoint and target extension boring and reliable.
2. Confirm command, slope, step evidence is **actually arriving**, in
   both directions, and every target can expand, hold and retreat.
3. Give resilience distinct recovery evidence and make control quality
   conditional on useful movement, without multiplying generic counters.
4. Split symmetric/directional bounds only after the evidence pipeline
   is reliable; hard-code no step-specific training ages.
5. Run a CUDA 2048-world PPO smoke, then a fresh cold campaign and
   inspect actual training distributions and supported locomotion.
