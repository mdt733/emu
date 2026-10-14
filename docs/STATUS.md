# Project status

*Last reviewed: 14 October 2026. This page records evidence, limitations and next verification gates—not plans represented as completed results.*

## Public demonstration

- The [Playground](https://mdt733.github.io/emu/) is a static MuJoCo/WebAssembly demonstration of an exported **60M-step actor**, not an ongoing learner.
- The trained policy is tracked as `showcase/champion.emupolicy`. The Pages workflow requires a manual dispatch and checks the pinned Git identity of that policy.
- Browser controls, appearance and physical ramp settings are for inspection. Demonstration on a ramp is **not** established locomotion competence under varied physical conditions.

## Training and source status

- The observed WSL smoke test used Python 3.13, JAX GPU, MuJoCo/MJX, one real PPO update and domain randomisation; it passed.
- The first v5015 foundation reached **20,054,016 environment steps** and reported completion. The subsequent ladder handoff failed because the trainer wrote to `00_foundation/` while the runner looked in `foundation20/`. This was an output-directory bug, **not** a demonstrated PPO convergence failure.
- The draft repair aligns run destinations and checks them before training; it has CPU regression coverage. **The original GPU run has not been confirmed recovered or rerun to completion.**
- Earlier long-horizon behaviour reportedly deteriorated in height around **100–200M steps per lineage**. The current 72M-aggregate-step ladder only follows one candidate to ~46M, so it cannot test that failure mode.

## What is currently validated

GitHub pull-request CI runs Python compilation and tests; Node/JavaScript tests; browser build; and a static policy/WASM packaging verification. The training suite also tests a deterministic exported actor against the real browser inference implementation with a NumPy reference. Those tests are necessary **code checks**, not proof of healthy gait or unchanged GPU learning dynamics.

## Known limitations and open acceptance gates

1. **Full small GPU lifecycle:** one update, checkpoint, warm resume, matched held-out evaluation and checkpoint export in the refactored checkout have not been run end to end on the target GPU.
2. **Gait quality:** current scores emphasise survival; promotion gates do not yet require a calibrated minimum for genuine commanded stepping. Do not claim that winning an early screen proves walking ability.
3. **Physical robustness:** the `robust40` fixed panel stresses *commands*, not independent physical-domain variation. Slope-equivalent domain screening exists; actual ramp and morphology transfer remain unproven.
4. **Cross-runtime physics:** Python/JavaScript deterministic actor inference can be checked numerically, but equivalent contact dynamics, sensor construction and observations in MJX versus WASM remain a separate unverified contract.
5. **Resume semantics:** schema-7 checkpoints restore learnable, optimiser, normaliser, governor and capability state; MuJoCo episodes and rollout RNG restart. This is a **warm learning-state resume**, not a deterministic continuation of the same physical simulation episode.

The [validation checklist](validation.md) specifies how to establish each gate. Historic results and retired recipe code remain available in [history](history.md) and Git history.
