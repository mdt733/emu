# EMU research history — findings, failures and archived experiments

This is the curated human record behind the **current** architecture. Historical
source still exists in Git commit history and, for the known-working
232-input reference, untouched [`main`](https://github.com/mdt733/emu/tree/main).
This document intentionally replaces the need for old version-by-version
explanations embedded throughout active Python source.

**A green test suite validates implementation contracts, not learned walking.**
The known-working 260M reference demo and its behaviour are not replaced
by any unproven 312-input experimental policy.

## Summary of experiment families

| Family | Research question | Findings worth retaining | Disposition |
|---|---|---|---|
| v5013–v5015, 220-input | Can one six-action actor walk with command combinations, a command field and sensible curriculum? | A trained 60M public demo worked; command sign conventions, field teaching dose, temporal recovery, observation normalisation and readable viewer mattered. Graph/nodes-only arms were not universally superior across seeds. | Preserve lessons and main/reference history; no new training should depend on old recipes |
| v5016 Pinnacle, 232-input | Can ToF/flow grounding, gentle contact terrain, energy-awareness and matched candidate selection improve general locomotion? | Working reference and 260M demonstration. Four experiments: baseline, motor guidance, contact slopes, combined commands; choose candidates on independent physical examinations, not reward. A general-walk control is essential. | **Known-good in untouched `main`**; keep as behavioural baseline, not duplicated campaign code forever |
| v5017–v5018 terrain/navigation | Can simulated contacts, genuine supported obstacle crossings, CPU/browser paths and robot-relative terrain agree? | Must use real MJX collision and real foot contact; broad-phase candidate pairs are not force or supported contact. A viewer-only obstacle is not a training obstacle. Height sampling and physical contact semantics require smoke tests. | Retain terrain geometry, tests and current sensor/physics contracts; retire research-specific orchestration when no longer needed |
| v5019 forward/rolling ToF | Can a deployable, history-aware terrain preview give the actor enough information for steps? | 312 inputs = 78 sensor/command fields × 4 frames; preserve 220/232 schema meaning, sensor delays/confidences and checkpoint fingerprint. No simulator-only obstacle height may leak into the actor. | Retain working 312 observation contract, sensor models and tests; still needs sim-to-real validation |
| Early v5020 four-arm lab | Would sending a ~20M 312-input foundation directly into demanding combined commands, real slopes and physical boxes produce a stronger walker? | **No successful winner established.** Foundation deteriorated late; contenders near start of next stage rapidly collapsed, reaching almost universal falls and extremely short episodes. Abnormally large finite spring-energy readings also occurred; causality unproven. | Keep result bundles, original run snapshots, failure analysis and original source commit; do not use this as proof PPO/terrain physics is sound |
| v5020 Champion Ladder (current experiment) | Will five narrower teaching emphases, short failure probes, earned step-height growth and common independent exams consistently improve a proven parent? | Implemented and source regression-tested; **not yet proven by real long GPU training**. Five arms: general, turning, recovery, effort, steps. 40/60/80M winner-refork; step-height physical ceiling 150mm. | **Active research on `feature/v5020-coldstart-terrain` only**. Keep the previous champion until verified improvement |

## Historical quantitative reference (archived Pitch15 experiment)

The historical pre-260M Pitch15 experiment used source
`a5996b8cda2c394c7c65e3e938282a3808ded121`, seed 733,
2048 MJX worlds and a 20M foundation. Four experimental arms to 80M,
two finalists to 100M and the best to 120M consumed **320M aggregate
world-steps**. Pitch15 led the matched screens: 0.683903 (80M),
0.712062 (100M), and 0.723497 (120M). These are **study-specific
screen scores**, not percentages of physical walking competence.

The 120M final-screen hard survival was 91.992%; growth survival 86.328%;
height shortfall approximately 4.87mm. Telemetry showed **74.12%** of
nominal servo-rated torque samples above the configured threshold and
approximately **12.81W** aggregate absolute simulated shaft power.
These numbers do not demonstrate safe physical actuators: shaft power
is not calibrated electrical consumption. The original exported
120,061,952-step Pitch15 policy and its provenance are documented in
historical Git commits and score reports, not reintroduced as the current
public champion.

**Additional physical-sensor lesson:** MuJoCo freejoint angular velocities
are body-local, while translational velocities are world-frame. A
forward-model optics/ToF experiment identified a double-rotation risk in
legacy gyro processing. Sensor extrinsics, timing, calibration, occlusion
and JAX-vs-real data semantics require explicit tests. Old checkpoints
must not be silently reinterpreted after such changes.

## Lessons to retain as engineering rules

### Physics

- A collision candidate is not load-bearing contact. Differentiate contact
  geometry, penetration and true physical success; box crossing requires
  actual forward passage, both feet and torso beyond the obstacle, support
  and upright stability.
- Finite but implausible physics values must be monitored too; merely
  rejecting NaNs is insufficient for safe, meaningful learning evidence.
- The current fixed-topology adaptive box permits per-world height
  changes without JAX shape recompilation. 150mm is the existing
  geometry/safety ceiling, **not** an experimentally reached competence.
- Simulator truth can teach and evaluate. The actor may only see physically
  plausible sensors, with dropout, delay, confidence and observable history.

### Training and curriculum

- Begin with attainable tasks and grow challenges from actual evidence;
  changing commands, resilience, terrain and teacher influence
  simultaneously at a stage boundary can obliterate learned gait.
- An environmental step count alone is not maturity. 4096 worlds versus
  2048 at the same number of total environment steps means about half as
  many 32-frame PPO updates; changing that variable mid-lineage is not a
  controlled comparison.
- Short early health probes should stop failures cheaply. They do not
  replace longer independent evaluations.
- PPO reward is a **teaching payment**. It is not a measure of certified
  physical walking, robustness or generalisation.
- Capability-field experience, adaptive lesson difficulty and
  independent examination must not compete for curriculum authority.
- Evidence missing from completed episodes is **unknown**, not a zero and
  never automatic permission to grow a frontier.

### Promotion

- Every candidate and the incumbent face identical held-out conditions.
  Failing the basic stability/walking gate disqualifies a specialist
  regardless of its obstacles or power score.
- Measure success with counts and the tested command/environment context.
  One apparent 20mm crossing is not a certified 20mm capability.
- Preserve the entire checkpoint lineage and the original source hash;
  never try to combine the neural weights of different contenders without
  a separate, controlled and evaluated study.
- Keep one general-purpose control with every tournament. Select one
  verified winner, or stop and retain the incumbent.

### Tooling and project hygiene

- A canonical Git `main` is a reference, not a scratchpad.
- Version-specific campaigns exist for archaeological reproducibility in
  Git history. Current source should describe **current behaviour**, not
  chronicle a dozen old revisions in inline comments.
- Store historical results and training lessons here; code is removed
  only after active imports/CLI/tests have been disentangled.
- Local run roots, checkpoint `.pkl` files and training outputs are not
  deleted by source cleanup, and must never be overwritten or silently
  reinterpreted by a newer source commit.

## Deferred research after the first verified 312-input champion

A separate October 2026 architecture review compared the existing sensor
contracts, terrain and the old four-arm failure. Its strongest conclusion:
**prove the current rolling-ToF actor can walk and improve real step results
before adding more senses or obstacle families.** We agree. None of the
following hypotheses is activated for tonight's v5020 experiment.

**Potential new senses, only after a passing held-out 312-input reference:**

- Foot-contact threshold or force/current measurement: possibly the most
  economical extra signal for supported stepping; needs realistic contact
  bounce, load and actuator calibration.
- IMU linear acceleration: potentially helps impact/slip classification;
  use measured frame/delay, not integrated world-truth velocity.
- A downward depth/short-range range-grid: may extend field of view beyond
  the two rolling ToF lanes; must model mounting extrinsics, occlusion,
  noise, quantisation, latency and dropout. A high-resolution raw depth
  stream is not automatically a useful actor input.

**Any real actor addition requires a new, explicitly versioned observation
schema**, fresh checkpoint/normaliser contract, masked-vs-unmasked ablation
and independent same-condition exams. Never reinterpret the existing
220/232/312-input checkpoint weights as if inputs had been appended.
Sim-only obstacle labels and exact geometry remain teacher/evaluation
truth, not actor senses.

**Potential future obstacle families:** authentic inclined ramps/slopes
(with actual normal-force support and slip checks), a sequence of steps,
then gaps/uneven patches. The *current* v5020 already has gentle real
contact slopes and a movable box. Do not conflate showing terrain in
the browser with testing its MJX collidability. Fixed-topology physics
and exact-height/angle exam conditions remain prerequisites. Any new
family gets an independently measured, evidence-driven frontier; no
simultaneous expansion of height, grade, velocity and sensor corruption
on the same immature foundation.

**No new Git branch or actor schema is authorised here.** The only
current experimental branch remains `feature/v5020-coldstart-terrain`,
and proven `main` is protected until explicit approval. The next
engineering milestone is an evaluated foundation, five bounded probes,
one 40M Champion Ladder exam, and honest diagnosis if nobody qualifies.

## Reproducibility pointers

- Reference: `main` → `emu/campaign5016.py` and pinned 232-input policy.
- Historical four-way v5020: `emu/campaign5020.py` on its original source
  revision and its original `robot_lab/emu_runs/...` root. This campaign is
  a research **failure**, not the current recommended training strategy.
- Current 312-input five-way prototype: `emu/campaign_refork.py` with
  `docs/V5020_WINNER_REFORK.md`.
- Version-specific research details were captured in older handovers,
  Git commits and CI logs. This history is the compact decision record.

**Do not use a currently edited checkout to resume an existing training
directory from a different pinned commit.** Start a fresh run root when
changing behaviour, or return to the pinned original source.
