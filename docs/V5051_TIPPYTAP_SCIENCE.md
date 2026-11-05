# EMU v5051 — tippy-tap cause audit and step-certification protocol

**Research date:** 2026-11-05 (Sydney)  
**Protected baseline:** `main` `6152cdc266c39b1cc7a7c5cc07bced53ee50be12`  
**Research branch:** `research/v5051-tippytap-autopsy` · [draft PR #41](https://github.com/mdt733/emu/pull/41)  
**Primary evidence:** `docs/V5048_SHARED_PARENT_RESULTS_20261105.md`, issue #17, and the live source in `emu/{environment,rewards,training,prism_exam,ppo}.py`.

This is a **mechanism investigation**, not a new champion release. The public 200.016M actor and its exact browser release are untouched. The 200M `.emupolicy` is for inference only; without its complete `final.pkl` optimizer checkpoint, it cannot supply a matched PPO parent.

## Findings verified against current source

| Priority | Mechanism | Evidence and quantitative consequence | Falsifying/isolating test |
|---|---|---|---|
| 1 | Stepping-activity gate | `rewards._gait_maturity` multiplies posture (weight 0.68) and foot contact (0.50) by `0.30 + 0.70*clip(episode_mean_step_evidence/0.15)` after 1s. It can withhold **0.826 reward/frame**, or **41.3/s at 50Hz**, at maximum physical qualities. This is an upper bound, **not** a measured actual loss in the champion. | Same-policy, same-trajectory counterfactual reward replay; then a single-variable reward-gate continuation. |
| 2 | Inadequate true slow/zero exposure | CONTROL explicitly samples positive forward velocity, ordinarily **0.04m/s or more**; coherent STEP selects ~0.10–0.22m/s before possible attenuation. Dedicated zero-births occur only if `stillness_birth_frac > 0` and are assigned **after** specialists. This existing option **also enables a zero-command gate exception**, so it is NOT a pure single-variable arm. | Log **final** after-overrides command distributions; separate stillness exposure and gate economics in a later treatment. |
| 3 | Phase-dependent shaping | `phase_rate` is sampled with 0.85–1.20Hz base, speed gain, and has no zero-speed pause. `phase_sin/cos` are in all four actor observation frames. `organic_step_shape` contains **0.30 × phase_sync × step_evidence**, contributing at most **0.0216/frame** through the 0.16 clearance weight. This is an explicit rhythmic incentive, but not a motor oscillator. | Frozen-actor phase observation perturbation first; then matched periodic shaping ablation and low-speed phase rate tests independently. |
| 4 | Costly crouch and COM shift | At 12mm height shortfall the healthy-height exponential starts declining; at 40mm shortfall relative to neutral it is approximately `exp(-1000*0.028²)=0.4566`, and the additional locomotion-height multiplier is `exp(-1200*0.028²)=0.3902`. Uprightness and feet-to-COM cone add more pressure. | Log posture/reward on actual approaches; compare a bounded, step-only softening with unchanged hard fall limits. |
| 5 | Perception too late / ignored | 4×4 ToF from torso, pitch rows 40/60/80/100°, yaw columns ±7.5/±22.5°, default 2-frame acquisition and 1-frame delay at 50Hz. Rays may be forward, nearly downward or rearward depending on pitch. Sensor availability does **not** establish learned use. | Same seeds and physical terrain, frozen actor with original vs raw terrain channels ablated; record measured detection-distance and action changes. |
| 6 | Hard mechanical constraints | Six position servos, 18mm spherical feet, spring hip/knee support. Thin thigh/shin collision geoms are **disabled** in the fast solver; a virtual penetration sentinel checks some strikes but does not supply full hardware contact reaction. Current source does **not** prove 3cm dynamic feasibility on real hardware. | Actual MuJoCo-supported swing workspace + constrained dynamics, riser/sentinel checks, 2/4 vs 4/4 parity before training. |

**200M champion provenance verified, not guessed:** the approved `showcase/champion.emupolicy` contains `source.step=200015872`, `source.git_commit=088a29be47f624be9e2c322c504ec98049b3f9bf`, `training.solver_iterations=2`, `training.solver_ls_iterations=4`, and **`training.stillness_birth_frac=0`**. The approved 200M policy therefore did not train with the optional zero-command birth or its reward-gate exception. The release manifest explicitly says its resumable `final.pkl` checkpoint has **not** been provided.

The gait proxy threshold deserves special attention. With a 12mm target swing and one lifted foot, a *sustained* roughly **2.2mm** foot-height increase beyond nominal contact makes per-frame `step_evidence≈0.15`. The episode mean still matters, so this is **not** a claim that one 2.2mm twitch alone clears the gate. Nonetheless, this demonstrates that the gate asks for activity, not useful horizontal travel or tall riser clearance.

**Important counterevidence against a simple clock-lock story:** the v5048 frozen 200M exam measured about **8.5 detected footfalls/s across both feet at zero request**, far faster than a 0.85–1.2Hz full clock cycle. The clock may bias posture and alternation, but it does not directly command every touchdown at that cadence.

## What was wrong with the old examiner

The previous `score_prism_crossings` considered a lip independently supported after five consecutive frames of actual top-surface contact. This is valuable evidence, but it can credit a single upper foot even when the other foot never follows, the body falls, or forward navigation stops. The old score also never defined **eligible approaches**, and nominal 3cm teacher requests sampled actual supported rises of only ~1.8–2.9cm. Therefore **16/32 worlds** with ≥2cm upper support for the 200M policy is **not** 50% mission completion and does **not** prove an actual 3cm climb.

## v5051 physical evidence now implemented

The existing `emu/prism_exam.py` owns all evidence; no additional engine module, physics world or deployable actor was added.

1. `score_prism_crossings` additionally identifies the leading physical foot on each top-supported lip.
2. `score_post_lip_missions` requires a sustained top contact for the **other** foot on the same real destination tile, then **50cm** commanded-direction body progress with heading within 30° and lateral deviation within 25cm, before a fall. It reports the first trailing support frame, first distance-crossing frame, final qualified frame, failed stage and actual signed lip height.
3. The exam JSON preserves both **supported events** and **completed among supported events**. This must never be relabelled an eligible-approach pass rate. Neither binary foot contact nor torso travel independently measures the load fraction carried by each foot; a true impulse/COM-transfer examination remains to be built.
4. Read-only ablations in `ppo.evaluate_trace` can hide terrain-facing inputs or phase inputs for **all four** observation history frames without changing the simulated physical world. ToF ablation does **not** remove implicit ToF use within the unchanged optical-flow depth estimator; results are about the terrain channels, not total sensory blindness.

Regression tests cover a completed course, a one-foot contact, heading failure, fall after placement and no crossing.

### Still not certified

- Actual *fixed measured* lip heights 1 / 2 / 2.5 / 3 / 3.5 / 4cm, uphill and downhill, straight/oblique, and comparable solver contacts.
- Every **eligible approach**, including stalls, aborted attempts, sidewall collisions, false footholds, and actual body-weight transfer. Successful events alone cannot supply that denominator.
- Dynamic support-limited foot workspace with current torque/speed and spring storage/release.
- Sensor-channel ablation findings and clock-dependence findings from actual frozen GPU/CPU policies.
- Training improvements: **no new PPO has been run**.

## Evidence gates, in order

**A. Baseline and falsification, zero PPO training.** Run five matched frozen exams (normal, phase frozen at valid sin=0/cos=1, phase reference rotated +180° without changing phase cadence, terrain channels hidden, both frozen/hidden) for zero/slow and step commands, identical seeds and solver, plus a held-out fast panel. Compare the original and ablated actor's footfalls, drift, support, actual supported heights, verified post-lip continuation, yaw and falls. An ablation is an out-of-distribution input intervention; strong effect establishes sensitivity, not automatically causation of the trained strategy. Weak effect only shows insensitivity under that exam, not that the sensor is unnecessary generally.

**B. Mechanical feasibility.** `python scripts/probe_stair_kinematics.py --output ~/emu-data/scratch/v5051-static-reach.json` now performs a static, read-only MuJoCo FK scan with the other foot in floor contact, both legs independently, and actual 1/2/2.5/3/3.5/4cm clearances. This checks a *necessary geometric* condition only. Next build and run MJX/MuJoCo **dynamics** and constrained supported-foot tests for specified actual risers. Report joint targets, measured servo torque/speed saturation, COM position, spring potential-energy history and failed collision/virtual-sentinel conditions. If 3cm is unattainable at the actual motors, don't spend 300M PPO steps trying to discover it.

**C. Physical approach autopsy.** Add eligible-approach and attempt accounting on real observed walls, including geometry, lead foot riser clearance, upper landing, body transfer, rear-foot lift and completed 50cm. Keep all denominators and body-/foot-contact evidence separate. Avoid changing the old support rule while benchmarking.

**D. Matched interventions, *only* after A–C.** First unchanged control, then isolated gate, zero-command exposure, phase shaping, phase rate, bounded crouch and 1mm Academy frontier as evidence justifies. Do not combine modifications until the first pass establishes effects. Keep actual source/config/seed/checkpoint ancestry and same-source evaluation. `--stillness-birth-frac` alone changes two related mechanisms and must be reported as a composite arm unless split before use.

**E. Finalists.** Prefer a compatible validated 30–50M PPO parent; if unavailable, produce a new 50M source-pinned cold foundation. Estimate a budget around 6×30M matched continuations plus 1–2×100M extensions **only after go/no-go gates**. Examine at fixed milestones and preserve Pareto results for stairs, stillness, fast travel, yaw/turning, slopes and mechanical cost. No automatic scalar-score winner; no automatic `main` promotion.

## Fresh WSL research checkout and read-only GPU starting point

Run in a **foreground** WSL Ubuntu terminal; the files below are outside source. This launches **no PPO training**:

```bash
git clone https://github.com/mdt733/emu.git ~/emu-v5051
cd ~/emu-v5051
git switch research/v5051-tippytap-autopsy
git rev-parse HEAD
git status --short
bash scripts/setup_wsl.sh --cuda13 --install-node --check
source .venv/bin/activate
emu doctor --require-gpu
python scripts/probe_stair_kinematics.py \
  --output ~/emu-data/scratch/v5051-static-reach.json

mkdir -p ~/emu-data/runs/v5051-baselines-s1701
emu prism-exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/runs/v5051-baselines-s1701/step-normal \
  --worlds 16 --horizon 650 --seed 1701 --device cuda \
  --mode step --speed 0.12 --lip-cm 3
emu prism-exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/runs/v5051-baselines-s1701/step-phase-hidden \
  --worlds 16 --horizon 650 --seed 1701 --device cuda \
  --mode step --speed 0.12 --lip-cm 3 --phase-ablation
emu prism-exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/runs/v5051-baselines-s1701/step-phase-rotated \
  --worlds 16 --horizon 650 --seed 1701 --device cuda \
  --mode step --speed 0.12 --lip-cm 3 --phase-offset-deg 180
emu prism-exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/runs/v5051-baselines-s1701/step-tof-hidden \
  --worlds 16 --horizon 650 --seed 1701 --device cuda \
  --mode step --speed 0.12 --lip-cm 3 --tof-ablation
emu prism-exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/runs/v5051-baselines-s1701/step-both-hidden \
  --worlds 16 --horizon 650 --seed 1701 --device cuda \
  --mode step --speed 0.12 --lip-cm 3 \
  --phase-ablation --tof-ablation
```

The above 3cm remains a **requested procedural frontier**, not fixed-actual-height certification; the protocol reports actual signed heights separately. Use a new output directory for each repeated seed and condition; the examiner refuses to overwrite an existing populated result. Verify source SHA after switching, and avoid editing this checkout mid-exam.

No full GPU PPO campaign command is authorised until A–C pass and a resumable `final.pkl` parent is verified. Preserve raw `prism_exam.json` for source-pinned comparison and bundle only small evidence artifacts.

**Operating rule:** experiments are disposable; verified physical findings are permanent.
