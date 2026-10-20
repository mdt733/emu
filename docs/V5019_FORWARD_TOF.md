# EMU v5019 — two single-point ToF sensors + optical-flow rolling terrain memory

**Branch:** `feature/v5019-forward-tof-profile`, isolated from the active v5018 25M/fork campaign. Do NOT edit that running checkout or old checkpoints, main, champion or Pages.

## User's concept, implemented as the v5019 contract

Two forward/downward **single-beam** ToF modules (right/left, body y = ±50mm) aim at a nominal **20cm** distance in front of the chassis. Each 25Hz sample is a real beam/riser/box/plane **first-hit** return, subject to time delay, dropout, range-dependent noise and 1mm provisional quantization.

- Existing asynchronous tripod (3 downward ToF), optical flow and IMU remain. Delayed gyro-compensated optical-flow estimates **local forward and lateral velocity**, while measured IMU angular speed estimates yaw.
- The sensor only gets *measured* slant distance and measured local tripod ground plane. It projects each return to a relative local height + actual along-body-X hit point. A riser hit can be closer than the far 20cm aim; it must land at its *observed* X rather than always being inserted at 20cm.
- Fixed JAX **21×2 map**: 0,1,2...20cm on each side. Every 20ms the cached samples shift toward the feet using estimated motion. One new real return per lane is deposited on its actual observed location at the ToF update rate. Fractional-centimetre displacement uses confidence-weighted interpolation. An unmeasured gap keeps near-zero confidence; it is not filled with invented flat values.
- Maps reset every new episode. Flow failure stops odometry and decays confidence. Yaw receives a side-specific displacement correction; yaw/lateral movement also decays confidence because two straight tracks cannot perfectly represent lateral reprojection or rotating views. Old measurements decay with age.
- The **new actor contract is 296**, not 280: original 58 per frame + 2 sides × 4 sampled heights (0,5,10,15cm) + 8 confidence values = 74 per frame, over four history frames. Historical 220 and 232 schemas, old checkpoints and 6 servo outputs remain separate.
- A matched `--forward-preview-masked` variant keeps the same physical world, map update, 296 input shape and optimizer settings but feeds zeroes in the 16 new NN slots. Once sighted weights exist, `--ablate-forward-preview` on the existing read-only physical crossing examiner masks those slots *without retraining the policy*.

The map is *not* global SLAM, and internally synthesized heights are estimates from raw calibrated rays plus a noisy measured local plane; no true world terrain height, box size, global root pose, or true MuJoCo robot velocity enters the rolling actor map. Contact/reward teacher truth remains separate.

## Practical limits

A narrow beam can miss the step during extreme body pitch/yaw. Rolling memory helps retain earlier valid observations but cannot create data that no beam ever measured. This is a **first-stage plane+box** model: no robot-leg occlusion, optical properties, multizone array FOV, hfield/log/capsule ray consistency, or confirmed real hardware mount. The current browser viewer still only generates older 220/232 actor observations; a 296 policy is **not interactive in-browser yet**.

At 0.5m/s and 20cm lookahead, maximum nominal warning is ~0.40 seconds; 25Hz measurement intervals can add up to 40ms waiting plus 20ms delivery delay, control smoothing and servo lag. At 0.1m/s 20cm corresponds to 2s. Actual scan hits under bounce/pitch or riser geometry change that window. The right response to missed obstacles is to measure physical beam coverage, not increase PPO budgets.

## Observable completion and no premature training

A known real collidable 20–50mm step must produce changed TWO measured ToF ranges and update 2×21 rolling map. After 5cm estimated forward motion, an original far 20cm sample should appear around the actor's 15cm channel with decaying positive confidence; before movement 0/5/10/15cm channels should remain *unknown*, not prefilled flat. 232 checkpoint sample structures must remain seven-field historical `SensorSample`, not silently switched to new schema-9 `PreviewSensorSample`.

The feature includes pure sensor/map checks and an optional two-world MJX reset/step, plus strict 296 policy checkpoint/export/exam handling. **This does not prove a trained robot can climb or that its simulated sensing matches installed hardware.** Keep training curriculum/reward/governors frozen; wait for v5018 physical run results. Then compare ONE reasonable pair of equally trained 296-input masked and sighted policies on held-out step heights, flat retention, yaw ±10°, velocity tracking and simulated servo stress; additionally freeze a sighted policy and test zero-ablation. No further experimental campaign layers until that result exists.
