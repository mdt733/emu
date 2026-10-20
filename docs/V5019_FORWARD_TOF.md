# EMU v5019 — forward-looking ToF sensing

**Isolated branch:** feature/v5019-forward-tof-profile, from v5018 commit b5548aa177. Do not alter the ongoing v5018 overnight GPU run, protected champion, main, or published site.

## Target: measure before climbing

The existing 220 and 232 actor schemas remain unchanged. The new opt-in forward_tof_280 schema adds six **raw slant ranges** (right 5/12/22cm nominal floor footprints, left 5/12/22cm) and six corresponding validity bits to each historical 58-element frame. Four frame histories make 280 inputs. Checkpoint schema 9 and exported policy schema 3 cannot silently be read as older policies.

Two *three-zone* forward/downward narrow-beam modules are assumed, located ±50mm laterally and 35mm ahead of torso reference. A pair of single-point ToF chips cannot provide all six simultaneous rays; future physical sensor selection must match the model or use separately timed scans. All extrinsics are provisional, not confirmed CAD or hardware data.

emu/forward_tof.py calculates first intersections of six rays with the actual MuJoCo floor plane and collidable box, including the front face. The actor receives raw noisy and quantized slant lengths and independent validity bits. Its existing gravity/gyro/downward tripod inputs supply body orientation and ground context. The optional inverse can derive relative heights for diagnostics, but is not on the actor path: losing the downward tripod estimate does not erase six otherwise-good forward beams. Unknown returns are **range 0 with validity 0**, never a measured flat floor. Range-dependent noise, provisional 1mm quantization, 25Hz acquisition, one-control-frame latency and independent ray dropout are fingerprinted.

**A/B control:** --forward-preview uses the full measured 280-channel actor; --forward-preview-masked keeps identical physical world, actor width and acquisition model but zeros the 12 preview values. No new reward shapers, adaptive governors, or six-arm campaigns have been introduced.

## Minimal code ownership

- emu/forward_tof.py: pure measured geometry forward/inverse.
- emu/contracts.py and emu/ppo.py: 280 schema and normalization (raw ranges normalized, validity unnormalized).
- emu/environment.py: sampled range → bounded delayed ring → raw slant range/validity → actor observation.
- emu/training.py, emu/training_config.py, emu/branch_lab.py, emu/cli.py, emu/evaluation.py: opt-in single-run physics and 280 masked/visible contracts.
- emu/checkpoint_io.py and emu/viewer_server.py: checkpoint schema9 and strict Python-side schema3 export/import, old schemas retained.

## Deliberate limitations

This is a **first-stage isolated box+plane scanner**, not a general hfield/ramp/log ray engine. Self-occlusion by moving legs, reflectance, multipath and motion-compensated rolling scans are not yet implemented. The published browser's existing 220/232 observation generation does NOT reconstruct 280 channels, so schema3 Python export is not yet an interactive browser policy. No real hardware mount, real sensor class or step-crossing competence is proven.

## Review without starting a training job

From a separate local checkout of this feature branch, use `emu train --profile sim2real --forward-preview --terrain-box-height-mm 30 --run-dir "$HOME/robot_lab/emu_experiments/v5019_preview" --steps 5000000 --envs 1024 --device cuda --plan` to print a config **only**. Add --forward-preview-masked and a different run directory for the matching control plan. These are research configuration examples, **not instructions to run GPU training while v5018 is active**.

Next: review actual ray intersections and two-world MJX observations, measure preview visibility as box enters the field of view, check real physical hardware sensor choices and compare masked vs visible 280-policy PPO only after current run results can inform gait feasibility. Keep the comparison small until it proves the sensory advantage.

## Falsifiable sensory result, before training

At least one of six 280-preview range fields **and** its validity must change when a 20–50mm collidable step enters a calibrated beam. The six visible fields must remain physically measurable even if the original three-downward-ToF planar estimator becomes invalid. Compare flat, shifted step approach and ±10° heading **and pitch**. The original 5/10/15cm angles lost a 40mm step under +5–10° pitch; the provisionally revised 5/12/22cm set saw the same physical step in the fixed-box numerical model at ±10° yaw and pitch. No claim that 5/12/22cm are exact hit footprints under bounce/pitch/yaw; they are nominal beam targets at body height 0.2584m.

A 0.5m/s approach gives a nominal 0.44s preview at 22cm, **before** subtracting 25Hz sample scheduling (worst ~40ms), one 20ms delivery frame and actuation response. The beam/step first-hit location will also shift as the torso bounces. Therefore a forward range channel alone is not enough to guarantee EMU can react in time. At slower 0.1m/s this same 22cm nominal separation is 2.2s.

When a trained 280-policy exists, run the same no-detour physical examiner with its original observations and `--ablate-forward-preview` (new **read-only** frozen-actor counterfactual) and compare flat ground, held-out 20/30/40mm steps, command velocity tracking, fall rate, and simulated servo effort. An unchanged crossing result after ablation is evidence the policy is not using the forward preview effectively; it does not justify another height-governor redesign.
