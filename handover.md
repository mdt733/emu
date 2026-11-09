# EMU current handover — 40cm central-servo physical build

Updated 2026-11-09. This is the **only active handover**. Completed experiments and superseded recipes live in Git history; do not read their launch commands as current instructions.

## Repositories / promotion boundary

- `main` is the frozen proven v5050-lineage champion until physical exams justify replacement.
- Historical v5058 results/checkpoints live in the previous Git revisions and data folders; the successor does **not** carry old policy format or sensor compatibility.
- The **candidate** is [draft PR #52](https://github.com/mdt733/emu/pull/52), branch `maintenance/v5058-clean-skeleton-sensor-audit`. This creates an entirely new physical/sensor/terrain model and must train **from zero**, with an independent checkpoint identity.
- Never merge directly to `main` without an equivalent signed step/slope/command/recovery physical exam and evidence of real torque limits.

## Physical h40 robot: one authoritative generator

`emu/robot.xml` plus `emu/physical_variants.py` produce the actual MuJoCo robot; the CPU browser sensor lab loads **that identical generated h40 XML**, not a painted substitute.

- Nominal hip/root height: **400mm**, six actively controlled joints and two passive shin-flex hinges. Ball feet diameter **36mm**.
- All **six Feetech C018 12V motors are chassis-mounted** (330g combined); both knees are driven by remote transmissions, provisionally idealized by joint-space actuators. No knee motor mass.
- Chassis: **460g estimated**, centred **5mm forward** of hip axes. Its rear face overlaps the hip attachment points; no visual braces. Additional original hip carriers and spherical feet ~40g.
- Each thigh: **one solid 2×2mm square-section carbon rod**. Each shin: **one 2mm round rod** split at the existing passive flex hinge. Approximately 6g total carbon at h40, subject to measured density.
- Total about **506g estimated**, *not a weighed bill of materials*. Three batteries are provisionally AA/14500-sized, not AAA/10440; regulators and servo bus current remain unverified.
- Rated actuator hardcap **±0.980665 Nm**. Original compression-knee support, passive shin flex and hip-roll centring retained. The 0.624 Nm·s/rad heuristic joint damping and remote transmission loss/backlash need calibration.

## Terrain / eyes: one physical contract

- MJX **four 1m tiles** in a 2×2 rolling window; **eight actual convex triangular prism colliders**. Every support measurement, height sample, lip, sidewall and step exam derives from that generator. Recycler waits for feet to clear abandoned tiles.
- ST **VL53L8CX** model: **4×4**, **45°×45° square FoV**, **70° downward**, **0mm forward**, **−30mm vertical**, zero lateral/yaw, with 16 *centre-of-zone* rays, nearest foot-sphere occlusion and delayed/noisy range validity.
- Full 360-input (90×4) actor architecture unchanged, but the new ray and mounting geometry has a **new hashed sensor contract**, not a compatible old checkpoint.
- ST **4×4 SPAD-array** physical layout shows IDs 12–15 on its top row and 0–3 on its bottom row, but the *receiving lens reverses BOTH axes*. In the nominal package orientation, the actor's scene rows map to raw IDs `3,2,1,0 | 7,6,5,4 | 11,10,9,8 | 15,14,13,12` (ST UM3109 §2.2). Hardware code must check actual board rotation and apply the corresponding transform; raw zone 0 observes the scene's **top-right**.
- The user originally preferred bottom **inner** zones occluded by the feet. Geometric preview of this actual 70°/−30mm mount suggests **outer** bottom corners instead. DO NOT fabricate occlusion by remapping values. Confirm with the real sensor before hardware finalization.
- The CPU sensor lab is a **fixed** 2×2 1m eight-prism calibration fixture; the trainer is *recycling* MJX terrain. The generic trained-policy browser carpet is still an illustrative nine-box world; only independent MJX exam can prove terrain capability. Each real ToF zone collects more than one optical measurement: training's one central ray is an approximation.

## Learning and evidence

- Exactly **one 360→256→256→6 actor**, one critic and PPO. Six teaching roles share that actor: command, resilience, slope, step, transition, control.
- Future target: **120 million cold world-steps**, seed 735, 2048 environments; **20M foundation**. Existing mature recipe introduces Academy after 20M and ramps births from 40M toward 60% by 55M. This has not been launched.
- Freeze and report at 20/40/60/80/100/120M (run with `--fixed-eval-every-steps 20000000`); keep metrics, source SHA, XML and sensor hashes, exact checkpoints, independent signed-step course and viewer export outside Git at `~/emu-data/runs`.
- NO warm-start from v5057, v5058 or v5059. NO `v5057_physical_tournament.py` 20M launcher for this new 120M recipe.

## Required gates **before** 120M

1. `bash scripts/check.sh --full` plus browser build/tests.
2. CPU compiled true-prism 1m geometry, exact h40 mass/contacts, two-triangle normal agreement, signed lip exam, midpoint recycler and occlusion contract tests.
3. GPU on the operator's 4060Ti: 2048-world 2-frame full PPO/update smoke; physical sensor probe, checkpoint restore/source-identity check, throughput/VRAM measurement.
4. A clean unique run root. No invisible checkpoint resumption, and no change to `main` or published actor.

If a gate fails, fix the single source, rerun the gate, and leave the 120M campaign untouched. Record any remaining fidelity approximations explicitly in this handover rather than claiming hardware equivalence.

## Rated servo damping correction — new h40 source only

The old v5058 motor-joint `damping=0.624` supplied passive braking torque
outside the ±0.980665Nm rated motor limit; at 2rad/s that added 1.248Nm of
unbounded viscous force. The successor removes this from driven hinge
joints and puts the SAME coefficient in each position actuator as `kv`
(with MuJoCo's rated `forcerange`). The existing 18% hardware variation
now randomizes `actuator_biasprm[...,2]`, not the removed passive damping.
This correctly bounds the combined PD motor output but **does not**
constitute a measured motor torque-speed/thermal curve or validated
remote-knee transmission; 0.08Nm mechanical frictionloss is still
uncalibrated. New cold-start model identity; no older checkpoint reuse.

## One successor; Git is the archive

The active parser, exporter and public release verifier admit **only the
physical 360-input VL53L8CX actor**. The CPU sensor laboratory is generated
without any retired champion weights. The former Pinnacle exported policy,
release approval and version-named 20M launcher are removed from this
candidate branch; they remain recoverable from older Git revisions. Manual
GitHub Pages deployment requires a NEW trained 120M actor, verified
checkpoint, source SHA and separate release approval. No code-level
backward compatibility and no automatically deployed stale champion.
