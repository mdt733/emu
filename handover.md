# EMU handover — H40 120M mainline

**Updated 2026-11-10. START HERE.** This is the single maintained entry point for resuming work after any chat or interruption. It records the current implementation, source identities, measured/unmeasured claims, operating rules and next action. The canonical **checkbox tracker** is [EMU master roadmap — issue #62](https://github.com/mdt733/emu/issues/62). Do not maintain a second, competing checkbox list here.

**At the start of each new chat:** read this file, [issue #62](https://github.com/mdt733/emu/issues/62), current `main` and any open PRs. Select the next unchecked, unblocked item; work on a short-lived branch; merge and remove experimental scaffolding; update the issue and this handover whenever the baseline, findings, restrictions or next action change. Git is the archive, not the current working tree.

## Fast orientation

| Area | Current answer |
| --- | --- |
| Robot / actor | H40 ~40 cm; 6 chassis-mounted 12 V C018 servos; 360 sensor history inputs → 256 → 256 → 6 position targets |
| Active policy | **120,061,952 steps**, seed **735**; `showcase/champion.emupolicy`; original training source `301d18b7b1cd1b165e6578fc53f9f8cda94fb680` |
| Terrain | Training: **4 rolling 1 m tiles / 8 true convex MJX prisms**. Viewer: **one Infinite Meadow** with **225 streaming CPU heightfields** |
| What works | Owner-reported good fast gait, slopes around 15° in viewer; not independently certified MJX capability |
| What does not work reliably | Up/down genuine steps, controlled post-step progress and heading; current H40 independent physical capability matrix missing |
| Hardware truth | Mass, transmission, motor thermal/torque-speed, ToF PCB mapping and sim2real are provisional |
| Source protection | Keep `main` champion and original private `final.pkl` untouched; code merges never silently retrain or republish actor |
| Planning authority | [Issue #62](https://github.com/mdt733/emu/issues/62) controls task checkboxes and completion references; this page controls handover/context |
| Next work | **Frozen H40 CUDA baseline smoke**, using the corrected read-only physical examiner from [PR #65](https://github.com/mdt733/emu/pull/65). First maintenance copy fix [PR #64](https://github.com/mdt733/emu/pull/64) is complete; deeper audit and broader exam remain open in #62. No PPO megacampaign yet |

## Roadmap and branch strategy

**Use two lanes, not a stack of eight speculative branches.** Issue #62 has the detailed task-level checkboxes, acceptance criteria, research questions and completion record. Here is the dependency map:

```text
Step 0: unified roadmap/hand-over merged via PR #63
  |
  +-- Lane A: maintenance/h40-source-tidy
  |       -> viewer/h40-laboratory-ui (Explore / Experiment / Diagnostics)
  |
  +-- Lane B: science/h40-capability-exam
          -> science/h40-stair-autopsy
          -> ONE evidence-led stair-learning experiment (20M acceptance first)

Independent later: physical-model/calibration ledger; measured GPU performance.
```

**Safe parallelism:** a maintenance/GUI change and independent, read-only GPU examiner can proceed on separate fresh-`main` branches. **Serialise** overlapping edits in `environment.py`, `ppo.py`, `training.py`, `handover.md`, and browser `main.js`. Merge small complete PRs, delete branches, and refresh the next branch from the merged `main`. Do not stack multiple unfinished PRs or maintain alternate learner/terrain implementations.

**Immediate priority gates:**
1. **Planning only** — link the one roadmap checklist, save the known baseline and branch rules; no actor, model, PPO, trainer, curriculum, terrain or viewer changes.
2. **Capability exam (P0)** — the 1 m seam spawn/lane, true-prism CPU checks, dual UP/DOWN trial allocation and separate actor/policy/XML hashes were corrected in [PR #65](https://github.com/mdt733/emu/pull/65) ([green CI](https://github.com/mdt733/emu/actions/runs/42801912985)). **Next:** run the CUDA smoke below on the frozen H40 actor (zero PPO updates). Only after it passes, collect 2048-world paired stairs/flat and broader command/heading/slope/recovery/effort evidence. Keep raw trials and a source-pinned summary.
3. **Housekeeping (P1)** — first copy/release fixes landed via [PR #64](https://github.com/mdt733/emu/pull/64) ([green CI](https://github.com/mdt733/emu/actions/runs/42801111504)); disabled/dead research switches, large-module ownership, outdated comments and test-value audit remain open. No numerical physics changes by stealth. Maintain ~24 substantive Python tests and the eight current docs.
4. **Viewer lab (P1)** — one Infinite Meadow with cleaner Explore / Experiment / Diagnostics settings, deterministic signed-step/scenario controls, sensor/foot/force/failure overlays and measured rendering/physics performance. Distinguish browser scenes and inspector targets from GPU exam facts.
5. **Stair autopsy (P1, exam-dependent)** — inspect obstacle visibility/lookahead and ToF ablations, gait phase and stationary behaviour, actual PPO terminal/advantage credit, C018 joint torque/hip-roll saturation, knee support, foot clearance, and virtual non-foot collisions; isolate the first failed stage.
6. **Targeted PPO (P2, autopsy-dependent)** — one controlled hypothesis and matched no-change comparison; 2048-world GPU/contact/solver + source-pinned checkpoint gate; 20M cold acceptance before a long run. No improvement claim without independent signed UP/DOWN traversal **and** preserved flat gait/heading/resilience.
7. **Physical fidelity + efficiency (P2–P3)** — measure hardware assumptions and small-batch full-contact/solver parity before changing high-speed approximation; then profile CUDA/JAX/MJX/WASM and keep setup and results workflow lean.

**Acceptance for every PR:** `bash scripts/check.sh --quick`, `bash scripts/check.sh --full` as applicable, GitHub CI, read-only policy identity verification and actual physics/GPU smoke if that PR touches it. Publishing GitHub Pages is a separately gated, user-reviewed action. A good viewer clip, a training frontier or a successful workflow alone does not certify physical locomotion.

**Completion convention:** tick a task **in issue #62 only when it passed its gates**; reference its PR/commit, the run/exam evidence (or 'docs-only') and cleanup. Update this file when the next recommended action or authoritative source changes. Leave retired experiments in Git history and concise conclusions in [research findings](docs/RESEARCH_FINDINGS.md), not as permanent one-off launch scripts.

## Research facts worth carrying into every stair discussion

- Historical pre-H40 experiments distinguished first upper-foot contact, independent five-frame top support of each foot, forward traversal and strict stop/parking. One older 20 mm screen showed **89/100 two-foot support** and **86/100 downstream crossing**, yet only **1/100 strict parked arrival**. Those are **not H40 measurements**.
- Older longer-training stair attempts sometimes regressed on 32.5–40 mm obstacles. Never assume more steps means more competent stair crossing.
- A stationary-command activity gate and phase shaping may influence the tippy-tap gait; that is a **testable hypothesis**, not an established cause of stair failure.
- PPO terminal success can have negative advantage if the critic expected a larger value. Existing diagnostics must be checked against **actual H40 learner events** before claiming wrong credit.
- **Fast MJX contacts are foot-focused**; knees/shins/thigh/torso use virtual intrusion sentinels rather than full collision impulses. Full-contact parity on small worlds may be needed to test hardware realism, and 2/4 solver settings are not automatically certified physically equivalent.
- The retired v5053 matrix **previously tested uphill only** and carried 75 cm exam spawn/lane assumptions. [PR #65](https://github.com/mdt733/emu/pull/65) corrected the H40 1 m exam-only spawn to 0.78/0.50 m and `stairs` now tests **both** signs; tested with actual four-world CPU MJX. **Real 2048-world GPU and H40 capability outcomes remain unmeasured.** Training, browser rendering and physical exam must be named separately.
- Save full checkpoint + optimiser/normaliser/teacher health outside Git in `~/emu-data/runs/`; preserve immutable original H40 files. A `.emupolicy` is **not** a resume checkpoint.

## Status and lineage

- **Chosen successor:** H40, 400 mm nominal root height, six central C018 actuators, new VL53L8CX sensor contract, four 1 m terrain tiles.
- **Complete cold-start run:** `h40-1m-tof70-120m-s735`, seed 735, **120,061,952** world-steps, trained at exact commit `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`. Source, final checkpoint, metrics, and intermediate checkpoints in `~/emu-h40-120m` and `~/emu-data/runs/h40-1m-tof70-120m-s735/` must remain frozen.
- **Proven policy export:** `showcase/champion.emupolicy`, uploaded owner-supplied bytes SHA-256 `178c201af661d3172d701a7d1d8a7711f9b2106e1f7058bc4902aa9c64a37c7a`, Git blob `eef79b1010f2a9561d0b72dd7aae3e0fcef512ae`; 360→256→256→6 network; actor SHA-256 `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673`.
- **Research code:** [PR #52](https://github.com/mdt733/emu/pull/52) at `f7e3fc851c2942b5f2785aaf068882c6fd7d32c5`; training physics used for the 120M policy was frozen at the earlier exact source commit. Subsequent code includes CPU/browser fixture improvements and must not be represented as having retrained the policy.
- **Promotion completed:** H40 source and the byte-identical 120M export entered `main` via [PR #53](https://github.com/mdt733/emu/pull/53), commit `1b796144b128b326be1ac54f6a289ad87857cde6`; its 24 Python tests, CPU MJX checks and browser CI passed. Pages publication is **a separate, manually approved workflow**, not a side-effect of a source merge. The [10 Nov 2026 Pages workflow](https://github.com/mdt733/emu/actions/runs/42794739700) and [source/browser CI](https://github.com/mdt733/emu/actions/runs/42794505755) both report success at `79c9f43`. This confirms workflow results, **not an independent live-site/browser acceptance**; verify the deployed actor in the actual browser before marking that task complete.

### What is demonstrated

The operator reports a good fast gait and roughly **15° slopes** while inspecting the exported trained actor. This is a qualitative owner observation, not a controlled held-out MJX examination. **Up/down step traversal remains unreliable**, including the baseline's historical limitations; no successful 3–4 cm stair capability can be claimed. No robot has been physically built or electrically/thermally validated.

The original `final.pkl` is **not in this repository or in the chat upload**. It is not interchangeable with the 863 KiB inference-only `.emupolicy`. The checkpoint checksum was measured on the original training PC as `632c09788613e0e5fdc077637efd0a4bf4fc1be7ae0c07dc3662087c6c12970b` and recorded in strict `showcase/release_approval.json`. The checkpoint bytes remain private/off GitHub; do not loosen the release gate.

```bash
sha256sum ~/emu-data/runs/h40-1m-tof70-120m-s735/checkpoints/final.pkl
```

## The one H40 physical design

`emu/robot.xml` + `emu/physical_variants.py` generate the model, used identically by the local sensor laboratory. No separate painted browser-only robot or fake structural braces.

- Six **12 V Feetech ST-3215-C018** motors are inside the **460 g estimated chassis** (330 g of motors included), centered 5 mm forward of hip axes. Remote knee transmissions are represented by idealized joint-space actuators. No knee-mounted servo mass.
- Two original small hip carriers, 36 mm-diameter ball feet, one 2×2 mm square carbon thigh rod and one split 2 mm round carbon shin per leg. Total estimated mass **~506 g**. Batteries and mass distribution are estimates; real BOM, backlash, transmission friction, electronics and cooling are unverified.
- Passive compression-knee support, shin-flex hinges and hip-roll centering remain. Position actuator `kv=0.624` is **inside** each ±0.980665 N·m `forcerange`; it is not unlimited passive joint braking. Motor torque-speed/thermal curve and 0.08 N·m joint friction are not characterized.

## The one sensor and world

- VL53L8CX 4×4 approximation: 45°×45° FoV, 70° down, 0 mm forward, 30 mm below torso, no lateral/yaw offset. Sixteen centre-of-zone rays, delayed/noisy confidence and nearest foot-sphere occlusion.
- ST receiving lens inverts both axes. Nominal scene-to-raw zone order: `3,2,1,0 | 7,6,5,4 | 11,10,9,8 | 15,14,13,12`; real PCB orientation and which bottom zones feet occlude still need hardware checks.
- Canonical sensor SHA-256: `d8f559641d240f6eb4bd54a3dc7e96f414994682ebc1b478209df202cc903995`. Older sensor/checkpoint schemas are rejected.
- **GPU training and frozen physical exams:** rolling 2×2 four **1 m** tiles, eight genuine convex triangular prisms, with recycler support safety and signed step goals. **Browser viewing:** [PR #61](https://github.com/mdt733/emu/pull/61) replaced retired Field/Park with one **Infinite Meadow**: deterministic endless global coordinates inside a moving **15×15 m / 225 physical CPU heightfield** window, with prettier scenery. This is **not** the GPU contact topology and cannot certify stair ability. The untrained sensor/calibration laboratory still exists separately.

## Run it

```bash
# CPU browser with the trained export (not the untrained sensor-lab flag)
python -m emu.cli viewer-build
python -m emu.cli viewer showcase/champion.emupolicy --no-open --port 8765

# Separate geometry and ray diagnostic: untrained
bash scripts/start_sensor_lab_wsl.sh
```

The local trained actor may also be loaded directly from `~/emu-data/actors/emu-h40-120m.emupolicy`. Do not invoke `--sensor-lab` to view neural inference. The old champion, older compatible interfaces and research data stay available through Git history, not active code.

## Maintained references and checks

- [Master roadmap, open tasks and completion record](https://github.com/mdt733/emu/issues/62)
- [Core skeleton](docs/CORE_SKELETON.md) · [Physical-quality examiner criteria](docs/PHYSICAL_QUALITY.md) · [Research findings](docs/RESEARCH_FINDINGS.md)
- [Maintenance contract](docs/MAINTENANCE.md) · [Clean WSL installation](docs/INSTALL.md) · [Playground](docs/PLAYGROUND.md) · [Release gates](docs/RELEASE_READINESS.md)
- [H40 promotion PR #53](https://github.com/mdt733/emu/pull/53) · [Infinite Meadow PR #61](https://github.com/mdt733/emu/pull/61) · [Housekeeping PR #64](https://github.com/mdt733/emu/pull/64) · [Examiner PR #65](https://github.com/mdt733/emu/pull/65)

```bash
cd ~/emu
git pull --ff-only
bash scripts/check.sh --quick
bash scripts/check.sh --full
# Read-only, one-time 64-world CUDA examiner smoke against the FROZEN H40 actor:
python -m scripts.v5053_matrix_exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/exams/h40-matrix-smoke64-s1701 \
  --mode stairs --worlds 64 --levels 20 --trials-per-level 1 \
  --horizon 350 --seed 1701 --device cuda
# If smoke passes, see issue #62 for 2048-world signed stairs + paired flat.
# GPU training acceptance only on the approved source-pinned research branch:
python -m emu.cli doctor --require-gpu
```

**Checkpoint rule:** Original training source SHA `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`, exact robot XML/sensor contract, `~/emu-data/runs/h40-1m-tof70-120m-s735/checkpoints/final.pkl` and policy `showcase/champion.emupolicy` are protected. Do not assume the present `main` training config can resume the frozen original checkpoint. Creating a changed physics/sensor source requires a new identity and independently tested actor.

**Fresh-chat prompt:** “Read `handover.md` and [issue #62](https://github.com/mdt733/emu/issues/62), inspect current `main` and open PRs. Pick the next unblocked checkbox, protect the H40 120M baseline, use one short-lived branch, validate and merge, delete old implementation/temporary scripts, and update the issue and handover with actual evidence. Avoid new training until the examiner and stair failure are understood.”
