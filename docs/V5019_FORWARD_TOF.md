# EMU v5019 — forward-looking ToF sensing

**Isolated branch:** feature/v5019-forward-tof-profile, from v5018 commit b5548aa177. Do not alter the ongoing v5018 overnight GPU run, protected champion, main, or published site.

## Target: measure before climbing

The existing 220 and 232 actor schemas remain unchanged. The new opt-in forward_tof_280 schema adds six signed height readings (right 5/10/15cm, left 5/10/15cm) and six corresponding validities to each historical 58-element frame. Four frame histories make 280 inputs. Checkpoint schema 9 and exported policy schema 3 cannot silently be read as older policies.

Two *three-zone* forward/downward narrow-beam modules are assumed, located ±50mm laterally and 35mm ahead of torso reference. A pair of single-point ToF chips cannot provide all six simultaneous rays; future physical sensor selection must match the model or use separately timed scans. All extrinsics are provisional, not confirmed CAD or hardware data.

emu/forward_tof.py calculates first intersections of six rays with the actual MuJoCo floor plane and collidable box, including the front face. Sensor outputs are raw noisy lengths. A separate inverse derives signed relative heights using only delayed ray ranges and the existing ToF tripod measured ground plane, without passing perfect terrain height to the actor. Invalid ranges give zero height **and zero validity**, not confidently flat readings. ToF sampling cadence, one-frame latency, individual dropout and calibration are versioned in the fingerprint.

**A/B control:** --forward-preview uses the full measured 280-channel actor; --forward-preview-masked keeps identical physical world, actor width and acquisition model but zeros the 12 preview values. No new reward shapers, adaptive governors, or six-arm campaigns have been introduced.

## Minimal code ownership

- emu/forward_tof.py: pure measured geometry forward/inverse.
- emu/contracts.py and emu/ppo.py: 280 schema and normalization (heights normalized, validity raw).
- emu/environment.py: sampled range → bounded delayed ring → local-plane relative height → actor observation.
- emu/training.py, emu/training_config.py, emu/branch_lab.py, emu/cli.py, emu/evaluation.py: opt-in single-run physics and 280 masked/visible contracts.
- emu/checkpoint_io.py and emu/viewer_server.py: checkpoint schema9 and strict Python-side schema3 export/import, old schemas retained.

## Deliberate limitations

This is a **first-stage isolated box+plane scanner**, not a general hfield/ramp/log ray engine. Self-occlusion by moving legs, reflectance, multipath and motion-compensated rolling scans are not yet implemented. The published browser's existing 220/232 observation generation does NOT reconstruct 280 channels, so schema3 Python export is not yet an interactive browser policy. No real hardware mount, real sensor class or step-crossing competence is proven.

## Review without starting a training job

From a separate local checkout of this feature branch, use `emu train --profile sim2real --forward-preview --terrain-box-height-mm 30 --run-dir "$HOME/robot_lab/emu_experiments/v5019_preview" --steps 5000000 --envs 1024 --device cuda --plan` to print a config **only**. Add --forward-preview-masked and a different run directory for the matching control plan. These are research configuration examples, **not instructions to run GPU training while v5018 is active**.

Next: review actual ray intersections and two-world MJX observations, measure preview visibility as box enters the field of view, check real physical hardware sensor choices and compare masked vs visible 280-policy PPO only after current run results can inform gait feasibility. Keep the comparison small until it proves the sensory advantage.
