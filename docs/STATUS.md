# Project status

*Last reviewed: 17 October 2026. This records observed evidence and
remaining gates, not a guarantee of physical locomotion.*

## Current experimental result: 20M foundation → 120M Pitch15 finalist

The completed v5015-lab1 campaign was run from source commit
`a5996b8cda2c394c7c65e3e938282a3808ded121` (seed 733,
2,048 worlds), and its output was supplied as `results.md` and
`results_data.json`. All four 80M arms were eligible under the same
256-world/350-step held-out screen, with a 15° commanded pitch envelope.

| Stage | Pitch15 held-out score | Comparison |
|---|---:|---|
| 80M | 0.683903 | ahead of shadow 0.645141, baseline 0.642302, slope 0.638627 |
| 100M | 0.712062 | ahead of shadow 0.692671 |
| 120M | **0.723497** | only continued finalist; hard survival 91.992%, growth survival 86.328% |

**Provisional champion:** `round3/pitch15/checkpoints/final.pkl`,
120,061,952 steps. Expected raw-checkpoint SHA-256:
`ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`.
The trained checkpoint remains on the user's local WSL disk; **it is not
contained in the report uploads and has not yet been committed/exported to
the public showcase.** Do not replace the current 60M actor with a fixture.

The results support pursuing the 15° pitch-command teaching hypothesis,
but not an assertion that graph teaching or gravity-slope training is
universally inferior: this is one seed and one matched-screen protocol.
All 80M arms achieved 100% *gentle gravity-equivalent slope* screen survival,
which is distinct from walking a collision ramp. The current screen does
not have a calibrated signed-progress plus completed-footfall gate.

**Hardware warning:** final Pitch15 on-policy training telemetry gives
about 12.81 W aggregate shaft absolute power and **74.12% of samples
above the configured servo rated-torque threshold**. These are simulation
diagnostics, not measured real motor power. Servo safety and actual walking
still need their own evaluations.

## Public demonstration

The [Playground](https://mdt733.github.io/emu/) still uses the protected
**60M** `showcase/champion.emupolicy` as of this report. Browser playback
is not an independent examiner; the user's qualitative report that the new
models look excellent is encouraging but does not substitute for measured
gait/contact and signed-direction results.

The plan is **one reviewed merge** with the completed simplified code,
latest reviewed viewer, and independently validated Pitch15 120M actor,
together with an updated pinned Git blob ID in
`.github/workflows/emu-playground.yml`. GitHub Pages deployment remains
**manual after merge**. Draft [PR #5](https://github.com/mdt733/emu/pull/5)
is not to be merged before the new policy is staged and the bundle checked.

## Implementation and validation

The refactored code passed 16 Python and four browser checks. The real WSL
CUDA lifecycle revealed a missing `ACTION_DIM` import in post-evaluation
diagnostics; that was fixed and tested. The full campaign then completed
and produced source/parent checkpoint hashes and held-out scoreboards.
Legacy 220-observation/six-action schema-7 policy contracts are preserved.

Remaining important checks are independent real-MJX signed progression and
footfall pairs on representative commands, multiple-seed reliability,
servo torque and spring energy feasibility, and Python/MJX versus
JavaScript/WASM full-observation/contact parity. Neither the current
held-out ranking nor a good viewer run proves hardware readiness.

The [handover](../HANDOVER.md) records the experimental lessons, provenance
and next release steps. See [PLAYGROUND.md](PLAYGROUND.md) for publication.
