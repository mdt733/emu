# EMU v5020 — from-zero training with V-memory, recovery, real slopes and steps

**Code:** [feature/v5020-coldstart-terrain](https://github.com/mdt733/emu/tree/feature/v5020-coldstart-terrain), forked from v5019's working diagnostic viewer. No changes to `main` or public champion. Previous v5018 run is stopped; these are new independent 312-input policy lineages.

## Physics → sensors → PPO

- A new policy starts with random weights, 312 physical observations (78 per frame × four 20ms frames), six actuators, standard JAX/MJX PPO. No 232→312 weight transplant and no synthetic privileged step channels.
- All phases keep the **existing resilience curriculum**, moving vx/vy/yaw commands, transitions, push recovery, attitude and motor-effort reward infrastructure. It is NOT a terrain-only walker.
- The `foundation` phase adds a gentle distribution of **genuinely rotated contact planes**, not gravity pretending to be a hill. Later phases maintain these varying slopes while adding a solid, collidable box.
- Each non-flat world may place one fixed-shape MJX step at an independently sampled X (22–38 cm), Y (±2.5 cm) and yaw (±~10°) **at episode birth**, and at a height determined by the existing, physical-success-gated terrain frontier. The box is set at the **local actual slope elevation** and never moved mid-episode.
- The actor's V memory sees physically raycast step and floor contacts, including the rear ToF. It receives only measured sensor signals, not actual box height or simulator floor normal; those privileged quantities remain with teacher/scoring.
- Existing 220/232 policies retain their old world and sensor contracts, with legacy slope+box forbidden. New slope+box code runs only with `rolling_tof_312` and an adaptive MJX box course.

## One cold-start lineage, five progressive phases

Launch `bash scripts/run_v5020_coldstart.sh STAGE` to inspect resolved `emu train --plan`. Append `--run` to run **in the same terminal**, not tmux/nohup.

| Stage | Absolute step target | New lessons |
|---|---:|---|
| `smoke` | 4,096 at 64 worlds | Prove 312 PPO compiles while real slopes and variable-height boxes collide |
| `foundation` | 20M | From random weights: steady/transition/recovery + gentle real slopes, no box |
| `slope` | 35M | Carry over full policy/optimizer/governor state; wider vx/vy/yaw and real slopes |
| `steps` | 60M | Keep locomotion/resilience/slopes, add variable 0–25 mm physical boxes (55% exposure ceiling) |
| `mixed` | 85M | Wider 0–50 mm step frontier, more challenging slope distribution, strong recovery commands |
| `stretch` | 105M | Optional harder slope and attitude commands; keep flat rehearsal, do not go if 85M deteriorates |

**Targets are candidate review boundaries, not automatic permission to continue.** The existing terrain frontier still decides individual learned heights based on actual crossings; a ceiling of 50mm does not mean the walker has achieved 50mm.

Use `EMU_V5020_ENVS=256` for VRAM/performance acceptance; `1024` is the initial target on 8GB GPU, not a guaranteed safe limit. `EMU_V5020_ROOT`, `EMU_V5020_SEED` are configurable. Stage run roots are separate so a later stage cannot overwrite its parent. The script chooses `--resume auto` for same-stage restarts and external parent `final.pkl` for the next stage. **All stages in a lineage must use the exact same committed source SHA**: checkpoint resume rejects different source revisions. Freeze code before the first 20M run.

## Controlled experiments

You can use `--arm masked` with the same script to train a completely fresh, dimension-matched 312-input actor with all 20 terrain-memory channels zeroed. It retains the same two ToFs, slopes, box collisions, rewards, PPO and seed. Do *not* compare sighted 60M against masked 20M as proof of sensing benefit.

First test that 312 observations have nonzero measurements/confidence during actual movement, plus true contact slopes and box alignment per world. After the first good 60M result compare a complete masked lineage of **matching steps** if compute is justified; evaluate the frozen sighted actor again with its inputs zeroed.

Report: flat commanded velocity and energy; sideways/yaw tracking; pushes/tilt recovery; genuine 10/20/30/40/50mm step crossings at held-out distances/headings; best/worst slopes; falls; sensor visibility. `emu evaluate` is a mostly flat/core examiner; use the independent **physical step examiner** for step score, not only reward, curriculum ceiling or sensor-panel appearance.

## Preflight and known boundaries

1. Fix the v5019 rear-reference CI failure: invalid rear observations must cause an *exact* no-op on map contents. This is in the new training branch only.
2. Run tiny actual MJX with per-world contact-plane geometry + step body and check true box-top height above local floor; smoke validates source and policy shape. Real slope+box is a **new physical combination** and must be verified for contact/solver stability before GPU scaling.
3. Check browser-vs-MJX ray probes at known pitch and step poses; the viewer diagnostic is useful but not proof of 312-policy browser inference. That decoder still needs a full browser actor implementation before a trained 312 brain can run in it.
4. Train a true from-scratch 312 foundation; inspect its frozen-policy gait. Move to the next phase only when core command following and resilience have survived.

No long GPU job has been launched from the repository changes. The user controls the WSL run and its cadence.
