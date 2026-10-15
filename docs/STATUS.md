# Project status

*Last reviewed: 15 October 2026. This page records evidence, limitations and next verification gates—not plans represented as completed results.*

## Public demonstration

- The [Playground](https://mdt733.github.io/emu/) is a static MuJoCo/WebAssembly demonstration of an exported **60M-step actor**, not an ongoing learner.
- The trained policy is tracked as `showcase/champion.emupolicy`. The Pages workflow requires a manual dispatch and checks the pinned Git identity of that policy.
- Browser controls, appearance and physical ramp settings are for inspection. Demonstration on a ramp is **not** established locomotion competence under varied physical conditions.

## Training and source status

- **15 October 2026 fresh-Windows/WSL validation:** Ubuntu 24.04.5, Python 3.13.16, JAX 0.11.2 on the RTX 4060 Ti (CUDA GPU backend), MuJoCo 3.16.0, 8-world MJX reset/step, four-world real PPO optimiser update and 11-field evaluation, and domain randomisation all passed. The local static Playground successfully ran the protected 60M policy. Optional Warp imports were absent, but MJX worked. Browser ramp walking appeared poor to the operator; this is qualitative feedback, not a matched physics exam.
- **15 October 2026 GPU lifecycle acceptance (fresh WSL):** an isolated v5015 test trained 8 environments for one 32-frame PPO rollout (256 aggregate steps), wrote a validated `final.pkl` and held-out `fixed_eval.csv`, then warm-resumed in a new process to step 512, preserved the ancestor checkpoint, restored and advanced observation-normaliser count **256 → 512**, verified an already-complete resume did **no PPO update**, and exported `tiny_smoke_actor.emupolicy`. All three documented stages (`first`, `continue`, `verify`) **PASS**. Both near-untrained policies fell 100% in the tiny check; this does **not** demonstrate a newly learned gait. The fixed examination took about 219 and 215 seconds in the two stages.
- The first v5015 foundation reached **20,054,016 environment steps** and reported completion. The subsequent ladder handoff failed because the trainer wrote to `00_foundation/` while the runner looked in `foundation20/`. This was an output-directory bug, **not** a demonstrated PPO convergence failure.
- The draft repair aligns run destinations and checks them before training; it has CPU regression coverage. **The original GPU run has not been confirmed recovered or rerun to completion.**
- Earlier long-horizon behaviour reportedly deteriorated in height around **100–200M steps per lineage**. The earlier accepted-main ladder only reached ~46M per survivor; the **draft tinyEmu recipe** extends four separate continuations to 80M and optional finalists to 100/120M, but still cannot establish robustness to 200M without later tests.

## What is currently validated

GitHub pull-request CI runs Python compilation and tests; Node/JavaScript tests; browser build; and a static policy/WASM packaging verification. The training suite also tests a deterministic exported actor against the real browser inference implementation with a NumPy reference. Those tests are necessary **code checks**, not proof of healthy gait or unchanged GPU learning dynamics.

## Known limitations and open acceptance gates

1. **Full small GPU lifecycle passed:** the documented 256 → 512 step warm-resume experiment completed on the target GPU and exported a policy. The **new draft branch's 20M foundation and proposed 260M aggregate four-way comparison have not run** on the user's GPU; its own bounded MJX/PPO acceptance is also pending. Do not transfer the earlier `main` GPU PASS label to this source commit.
2. **Gait quality:** current scores emphasise survival; promotion gates do not yet require a calibrated minimum for genuine commanded stepping. Do not claim that winning an early screen proves walking ability.
3. **Physical robustness:** the `robust40` fixed panel stresses *commands*, not independent physical-domain variation. Slope-equivalent domain screening exists; actual ramp and morphology transfer remain unproven.
4. **Cross-runtime physics:** Python/JavaScript deterministic actor inference can be checked numerically, but equivalent contact dynamics, sensor construction and observations in MJX versus WASM remain a separate unverified contract.
5. **Resume semantics:** schema-7 checkpoints restore learnable, optimiser, normaliser, governor and capability state; MuJoCo episodes and rollout RNG restart. This is a **warm learning-state resume**, not a deterministic continuation of the same physical simulation episode.

The [validation checklist](validation.md) specifies how to establish each gate. Historic results and retired recipe code remain available in [history](history.md) and Git history.
