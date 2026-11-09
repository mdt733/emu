# EMU release readiness — H40 120M

**As of 9 November 2026:** H40 is the **current source and inference actor on `main`**, promoted in [PR #53](https://github.com/mdt733/emu/pull/53) (commit `1b796144b128b326be1ac54f6a289ad87857cde6`). That merge is **not** a claim of hardware readiness, reliable stair climbing or a GitHub Pages deployment.

The public playground is a **separate manual release decision**. Its [workflow](../.github/workflows/emu-playground.yml) only deploys after `scripts/verify_showcase_release.py` finds an exact policy/approval pair.

## The one candidate

| Identity | Recorded value |
|---|---|
| Robot / source profile | 40 cm H40, six chassis-mounted C018 servos, rated ±0.980665 N·m limit |
| Original training commit | `301d18b7b1cd1b165e6578fc53f9f8cda94fb680` |
| Run / seed / age | `h40-1m-tof70-120m-s735` / 735 / **120,061,952** steps |
| Committed actor | `showcase/champion.emupolicy` |
| Actor SHA-256 | `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673` |
| Policy SHA-256 | `178c201af661d3172d701a7d1d8a7711f9b2106e1f7058bc4902aa9c64a37c7a` |
| Policy Git blob | `eef79b1010f2a9561d0b72dd7aae3e0fcef512ae` |
| Sensor contract SHA-256 | `d8f559641d240f6eb4bd54a3dc7e96f414994682ebc1b478209df202cc903995` |

The owner reports a pleasing fast gait and approximately 15° slopes in the **viewer**. These are qualitative observations. A properly matched independent MJX physical exam, robust signed step crossing, heading/command fidelity and real 12V actuator/remote transmission validation remain **unproven**. An old actor's independent 2 cm step success must not be transferred to H40.

## Source and browser gates

- [x] Source, current H40 actor bytes and documentation entered `main` via [PR #53](https://github.com/mdt733/emu/pull/53).
- [x] PR Python/CPU MJX and browser CI passed, including the retained **24** core Python cases.
- [x] 120M inference-only export retains the exact robot XML and sensor contract. Full training state is retained on the operator's computer, **not** inside this repository.
- [ ] Independently run a matched physical exam for all meaningful step/grade/command/recovery claims. This is a **capability validation**, not an excuse to label viewer evidence as real-world proof.

## Public GitHub Pages gate (not complete)

- [x] The new actor bytes are committed unmodified.
- [ ] Compute **SHA-256 of the original full `final.pkl`** in `~/emu-data/runs/h40-1m-tof70-120m-s735/checkpoints/final.pkl` on the operator's training computer.
- [ ] Generate a reviewable `showcase/release_approval.json` from that exact SHA-256 and the checked-in trained export. Do not manufacture a checkpoint hash or silently fall back to a weaker historical approval format.
- [ ] Run `python scripts/verify_showcase_release.py` and prepare/build/verify the exact approved `dist` assets.
- [ ] Explicitly dispatch the protected/manual GitHub Pages workflow **from `main`**; verify the deployed artifact and actual browser gait. A normal source merge must never publish an actor automatically.

Until the approval exists, `bash scripts/check.sh --quick` and `--full` validate the current actor contract and source independently, while **publication remains blocked**.

The frozen training source, complete checkpoint, results and stats are not altered by viewer-only commits or docs maintenance. A new step-solving experiment must use a new source-pinned branch and its own checkpoints rather than silently rewriting H40.

## Measured quality for future challengers

Use [physical quality](PHYSICAL_QUALITY.md) as the independent examiner definition: useful forward/reverse/lateral/yaw movement, heading and route, energy per *useful supported* metre, legitimate signed step up/down, slopes/camber and disturbance recovery. Report real contact and continued commanded progress beyond a lip, not merely terrain requests or curriculum reward. Compare paired tasks only under compatible physics/geometry/sensors and report misses as misses.

The [old October/early-November release checklist](https://github.com/mdt733/emu/blob/6152cdc266c39b1cc7a7c5cc07bced53ee50be12/docs/RELEASE_READINESS.md) preserves the detailed v5032/v5048 results, old release exceptions and prior decisions. It is **historical evidence**, not current instructions.
