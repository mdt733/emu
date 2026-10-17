# EMU — research history and open questions

This is a curated internal record, not a chronological dump of development
passes. Numerical results are tied to the experiment that produced them.
Archived raw files and earlier notes remain recoverable in Git history; full
.pkl training checkpoints are maintained outside the repository.

## Goal and interpretation

The project is a testbed for learning **how to teach** a dynamically unstable,
six-actuator biped to walk, not simply optimizing a demonstration score.
Separate **training reward** (an incentive), **held-out frozen-policy scores**
(a proxy for generalization), and **independent physical motion evidence**
(signed travel, alternating foot contact, hardware feasibility).

Current deployed actor: 55 sensor channels × 4 history frames = 220 inputs,
six action outputs. Physical simulation and the WebAssembly viewer may differ.

## Accepted Pitch15 campaign — October 2026

- Training source: `a5996b8cda2c394c7c65e3e938282a3808ded121`.
- Seed: 733; 2048 MJX environments per run.
- Shared 20M foundation, four branches to 80M, two finalists to 100M,
  and one winner to 120M: **320M aggregate environment steps**.
- Four teaching hypotheses: baseline (field guidance), shadow (field off),
  slope (gentle variation), and pitch15 (15° chassis command).
- On matched frozen exams, Pitch15 led at 80M (0.683903, versus shadow
  0.645141, baseline 0.642302, slope 0.638627), 100M (0.712062 versus
  shadow 0.692671) and 120M (0.723497).
- Final held-out panels: hard survival 91.992%; growth survival 86.328%;
  height shortfall ~4.87 mm. Gentle gravity-equivalent slope survival 100%.
- Final training telemetry: **74.12%** of servo-rated torque samples above
  threshold; simulated aggregate absolute shaft power about **12.81 W**.
- Approved exported demo: 120,061,952 steps, Pitch15. Actor blob pin
  `067d6156f0f9de629eabb1a69cf1b0ceb86db24c`.

### What these observations support

Increasing the chassis-pitch command ceiling helped this specific training
campaign under its matched screens. Shadow was competitive, so the result
does **not** establish that graph guidance is necessary. The slope branch did
not beat Pitch15 on the common tests. Do not infer that the latter is
general-purpose terrain-ready: these experiments used one seed and selected
screens, not verified signed uphill progress and contact-stride completion.

The high motor torque overrun is a serious real-hardware deployment blocker.
Shaft power is not directly measured electrical input power, and smoothness
reward is not an efficiency objective.

## Earlier development — preserved lessons

- Original public playground sometimes served a zero-weight fixture instead
  of the checked-in trained model. Make release provenance explicit: source
  revision, policy SHA and training step, pinned in the manual Pages workflow.
- A previous ambitious 400M single-run continuation showed deterioration
  in walking posture/height as training progressed. Prefer short shared
  foundations, compared branches, then selectively extending finalists.
- Curriculum breadth, physical capability evidence, slope adaptation and
  weak heading hold motivated explicit separated evaluation and direction
  tracking checks.
- Earlier refactor attempts scattered state across similarly named helper
  files. The current approach prefers **few files with clear ownership**,
  even when an individual module is over 1,000 lines long.
- Source consolidation exposed stale lazy imports and colliding function
  names; verify real command paths in addition to import-only tests.

## v5016 — first independent physical-exam instrumentation (October 2026)

- The pre-existing directional examiner already measured world-fixed signed
  progress, heading, reset-safe contact-stride candidates, command transitions,
  and optional gyro-delay/noise effects. No second examiner was created.
- v5016's protected Pitch15 actor can now be examined directly from the
  repository's `showcase/champion.emupolicy`: the read-only loader rejects
  incompatible XML, channel order, normalization, servo delivery and altered
  actor bytes. JAX exam inference uses the identical actor as the browser.
  The policy is not a resume checkpoint and the model itself is unchanged.
- Added explicit rest-to-walk, acceleration-step and brake-to-stop courses
  alongside existing reversals/turns. Transition response and braking travel
  are measured from actual trajectory arrays, not derived from PPO reward.
- The existing directional batch planner now accepts a protected exported
  actor with a ten-course `physical` profile and explicit `--run`. This is
  the same per-case matched command/controller comparison, not a new exam
  implementation or a new training framework. Plan mode is read-only.
  Seeds 1701/1702/1703 remain *planned*, not claimed measurements.
- Branch-only instrumentation now carries six MuJoCo actuator output torques
  and joint speeds through the existing frozen-policy MJX trace. The reducer
  reports per-joint peak/p95 torque, exposure above nominal 0.638 N·m,
  squared-duty thermal *proxy*, mechanical work, contact-pair slip proxy and
  changes in stored spring energy. Neither heat nor regeneration is inferred.
- Motor caveat: MuJoCo's position actuator output is measured BEFORE the
  joint-level 0.435 N m s/rad damping term that approximates speed-dependent
  back-EMF; the 0.08 N m Coulomb loss and spring torques are also separate.
  Thus reported "shaft work" is the actuator-force × joint-speed accounting
  convention, **not** calibrated net joint output or electrical consumption.
  No second torque-speed law has been implemented.
- Known limitations: pre-step actuator samples lag one control interval;
  a terminating step's post-step actuator force is unavailable after reset.
  The post-step aggregate power metrics still cover the terminating step.
  Contact-pair proximity is not a ground-reaction-force measurement.
  Total spring energy variation does not establish energy returned to gait.
- GitHub's earlier real-MJX CPU trace tests passed. The protected 120M
  actor now has a full CPU MJX examiner acceptance test, but local CUDA
  physical acceptance and meaningful multi-seed frozen-policy measurements
  must still be verified before asserting locomotion or hardware viability.
  Exported actor weights/normalizer cannot resume or serve as a full PKL
  training checkpoint.
- Source, reward, curriculum, actor interface and physical delivery model are
  intentionally unchanged. No hardware feasibility verdict has been made.

## Open questions and next experiments

1. **Can EMU travel where commanded?** Verify signed course progress,
   continuous stride pairs, heading consistency and turn-stop behavior
   independently of high simulator reward.
2. **Can it walk with realistic servo constraints?** Examine torque saturation,
   motor speed, energy, temperature-related risk and spring assistance.
3. **Do the viewer and MJX agree?** Compare dynamics under equivalent
   commands and perturbations; a browser demonstration is not physical parity.
4. **Which curriculum grows capability most reliably?** Compare multiple
   seeds and matched tests before making causal claims from a single winner.
5. **Can it tackle true uneven terrain?** Distinguish tilted gravity tests
   from real contact geometries, hills, bumps and transitions.

## Validation policy

A branch must pass its current Python/browser CI, a clean isolated CUDA
save-resume-evaluate-export lifecycle and local viewer playback before it can
be proposed for main. Never resume an old training root from new source.
`main` and the trained showcase are read-only without explicit approval.

### v5016 next teaching work (same date)

- User's pitch hypothesis is sensible as experiment design: share a ±30°
  chassis pitch target across future arms rather than spending a comparison
  slot on a posture-only variant. This was implemented as an opt-in
  `sim2real` profile; the old v5015 Pitch15 win is not evidence that the
  actual movements were trivial or safely reproducible at 30°.
- A disabled-by-default, explicit motor-effort incentive is available for
  controlled comparison once the GPU exam and real baseline are measured:
  `motor_guidance_weight` ∈ [0,0.10]. It reports a separate weighted payment
  and does not alter the robot model, observations or checkpoint schema.
  It is *not* an optimisation of measured electrical energy or calibrated
  motor heat. A spring-charging reward is intentionally absent: cycling a
  spring without useful progress would be a reward hack.
- The public `emu directional-exam --policy ...` and
  `emu directional-batch --policy ... --profile physical` CLI handlers
  were repaired; previously only lower-level calls understood this mode.
  Tests exercise the public argument forwarding and no-write planning.

### v5016 B2 real-contact and elastic-mechanism audit

- Exam trace now samples eight actual MuJoCo spring hinge qpos values
  and the model's stiffness/reference vector. The reducer computes
  Δ(0.5 k(q-q_ref)^2) on the EXACT scored pose intervals, stopping before
  episode reset. Right/left passive-shin release coincident with contiguous
  stance flags is reported separately from non-stance release.
- Neither stance timing nor elastic release proves that springs helped
  forward progress. Spring charges/releases cannot be summed into a net
  battery saving without energy-flows, damping losses and a motor model.
  No extra training reward for spring cycling has been added.
- A single static MJX world-contact plane may now incline by 0..5 degrees
  via `contact_plane_quaternion` applied to the real floor geom before
  `mjx.put_model`, without changing robot.xml. The old tilted-gravity
  tests remain explicit and untouched. Future terrain sampling must
  separately verify per-world contact orientation, geom collision behavior,
  spawn clearance, and GPU VRAM impact.
- The approved frozen actor physical batch includes 13 cases: ten flat
  direction/maneuver cases, three separate REAL contact 2° grade cases.
  These are a physics audit, not a completed demonstrated climb.


### v5016 real GPU examination user feedback — null summaries and recovery

- A frozen Pitch15 four-world steady course completed its OFF and ON MJX
  rollouts on the user's WSL/CUDA machine, but the paired report crashed
  because an unobserved command-transition latency is JSON null. This is
  expected on a steady command (no maneuver), not evidence of instantaneous
  response. Nulls are now skipped by optional averaging, not forced to zero.
- The existing paired exam can explicitly accept `--compare-heading
  --reuse-completed` to reconstruct its **missing** paired report from
  existing `off/` and `on/` exam JSON, without another simulation. It checks
  SHA256 of the actual frozen policy/checkpoint, complete operator protocol,
  heading mode, seed, world count and shared source revision; it records
  the original physics and later analysis revisions separately.
- A public CLI integration test runs the protected Pitch15 actor with the
  same `--compare-heading` flag, verifies null reporting, deletes only
  the derived paired summaries, and reconstructs them with MJX evaluation
  monkeypatched to fail if any new rollout is attempted. CPU CI is required;
  real CUDA evaluation/feasibility remain separately verified by WSL.


### v5016 code-first audit correction (17 Oct 2026)

- Found by direct source walkthrough, not by trusting existing green tests:
  steady schedules discarded their lateral request, yaw-only maneuvers were
  never measured, and both gait and training proxies interpreted absolute
  world-Z as ground clearance on the newly inclined contact plane.
- The examiner now collects the real MuJoCo floor plane normal/origin from
  the *scored MJX Data*, requires it for real-contact experiments, grades
  real hills using their collision normal (not vertical gravity), and uses
  signed ball-to-plane clearance and tangent-plane displacement for gait
  evidence. The old tilted-gravity/flat-floor exam remains explicitly distinct.
- The environment computes floor-relative height, foot clearance, low-body
  pressure and height-sensor approximation from actual collision geometry.
  Flat worlds retain old world-Z geometry; new inclined-plane births are
  height-adjusted to the local floor. Neither makes the unfinished hardware
  sensor estimator reliable, nor turns the original domain-randomized
  tilted gravity into true per-world terrain.
- Pure change-oracle tests cover lateral and yaw-only requests and real-slope
  grading; an actual CPU MJX reset checks ground-relative geometry. No PPO
  reward coefficients, robot.xml, actor layout, schema-7 checkpoint or
  published Pitch15 policy were changed. GPU acceptance still required.
