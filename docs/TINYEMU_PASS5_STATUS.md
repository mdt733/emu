# tinyEmu Pass 5 — Physical observability and showcase identity

**Date:** 15 October 2026. **Review:** draft [PR #3](https://github.com/mdt733/emu/pull/3). **Accepted reference:** main `40c5d0c`. This is **not** a Pass 9 transplant or a new PPO objective.

## Files changed and why

| File | Actual change |
|---|---|
| `emu/environment.py` | Read modeled six-actuator torque, joint velocity and domain-aware force limits in `_metrics`. Resolve spring stiffness/reference from the loaded XML model. Two small pure helpers calculate read-only effort/elastic-energy metrics. |
| `emu/rollout_metrics.py` | Average seven physical diagnostics into existing run telemetry; no separate reducer. |
| `tests/test_servo_effort.py` | Validate torque×speed, sign split, zero shaft power at stall, spring energy/reference, actual loaded robot XML, JAX jit and rollout reducer. |
| `scripts/prepare_showcase.py` | Add optional validated Git SHA to generated `showcase.json`. |
| `scripts/verify_showcase_dist.py` | Check expected build Git SHA before any Pages upload/deploy. |
| `emu/viewer_src/src/showcase.js`, `src/main.js`, `tests/showcase.test.mjs` | Safe manifest parsing and visible short frontend source revision on static showcase (no policy mutation). |
| `.github/workflows/emu-playground.yml` | Stamp/verify `GITHUB_SHA` in manual deployment. Workflow remains manual, with existing champion protection. |
| `tests/test_showcase_product.py`, `tests/test_showcase_build_assets.py` | Metadata generation and mismatch regression tests. |
| `docs/WHERE_THINGS_LIVE.md`, `docs/PLAYGROUND.md` | Document units/limitations, how build revisions differ from policy checkpoints, fix an obsolete `cd viewer` command. |

## Measurements introduced (each averaged across observed rollout frames)

- `servo_shaft_abs_power_w`: sum of absolute mechanical actuator torque × joint rate.
- `servo_shaft_drive_power_w`: positive part of mechanical shaft power.
- `servo_shaft_brake_power_w`: magnitude of negative mechanical shaft power (**not energy regeneration**).
- `servo_over_rated_frac`: fraction of six actuators with modeled torque magnitude above nominal STS3215 continuous 0.638 N·m.
- `servo_peak_torque_limit_frac`: peak demand fraction of the **actual per-world** model force limit (domain randomization included).
- `spring_potential_energy_j`: elastic potential of six motor hinge springs + two passive shin springs, using actual XML stiffness and `qpos_spring`.
- `shin_spring_energy_j`: the two passive compliant shin springs alone.

**Caveats:** mechanical zero-velocity shaft power can be **zero at high stall torque**, while actual motor electrical power/heating remains significant; these are not battery-watt, winding-temperature, net spring-work, cost-of-transport or calibrated lifetime estimates. The measurements have not been added to PPO reward, deployable observations, governor, held-out rankings or automatic branch promotion.

## Tests actually executed

- [CI run #97](https://github.com/mdt733/emu/actions/runs/38684628941): static showcase source revision feature and existing contracts — successful.
- [CI run #103](https://github.com/mdt733/emu/actions/runs/38684917624): failed **two new test assertions**, one float64/float32 exact equality and one too-broad string rejection of the pre-existing `servo_smoothness` reward. No implementation failure identified; both assertions corrected rather than removing the safety check.
- [CI run #106](https://github.com/mdt733/emu/actions/runs/38685041786): **167 Python tests passed plus 5 subtests, 18 JS passed**, browser production build, checked-in champion/WASM package and checkout-clean checks passed.

These are CPU/JAX/robot-XML and browser-package checks, **not a full real-MJX or CUDA training acceptance on the user's RTX 4060 Ti**. A real rollout is needed to verify the new `data.actuator_force` fields and that the telemetry's performance/values remain sensible at training scale.

## Retained / removed

**Retained:** accepted PPO/GAE, 220-input six-output actor, original eight rewards and fall penalty, schema-7 resume/checkpoint flow, current experimental ladder and fixed evaluations, domain randomization, protected 60M champion, manual Pages deployment.

**Removed:** no additional training modules this pass; a stale local preview command and uncertainty about which source revision *future* Pages builds deploy are addressed. Earlier passes removed the duplicate frontend tree and original environment reward owner.

**Publication status:** current public GitHub Pages source revision could not be directly inspected with the available live-site source. An old site won't yet display a revision; the new field is available only on a subsequently approved manual deployment. This branch was **not published**.

## Next work before merge

1. WSL **isolated worktree**, fresh-root CUDA/MJX/PPO doctor and accepted 0→256→512 checkpoint→warm resume→fixed exam→no-op→export acceptance. Existing long runs stay untouched.
2. Sample/read newly logged motor and spring fields from a bounded genuine physics rollout; cross-check `servo_peak_torque_limit_frac` under hardware domain variation and ensure these signals remain absent from PPO gradients.
3. Build separate read-only *completed-stride* and forward-displacement evidence, including negative-control policies that fall, hop in place or stand frozen.
4. Only after benchmark calibration, consider effort/per-joint sustained-load scorecards, derived spring work, a simple experiment winner criterion and whether/how reward weights should change. Energy shaping without verified locomotion would create an incentive to freeze.
5. Revisit the very large remaining `environment.py` and `training.py` structures after the GPU gate, and replace rather than wrap their overlapping responsibilities.

**Do not merge or publish the draft yet.**
