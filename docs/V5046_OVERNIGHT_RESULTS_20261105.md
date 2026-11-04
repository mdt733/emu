# v5046 — overnight optimisation results and promotion gates

> **Completed research, not a standing experiment suite.** 34/34 arms passed:
> 17 configurations × seeds 735 and 761, each with eight full PPO updates
> (first includes JIT and is excluded from speed timings), on 2,048 RTX 4060 Ti
> worlds. Original campaign source SHA:
> `7f2edd988fcb2e637577b7c5aa23b3e274aac856`.
> The operator's raw `overnight_results(20261104-194041).zip` remains the
> external evidence archive; the one-off sweep runner is deleted after use.

## Findings worth keeping

| Trial | Median steady steps/s | Relative to default | Decision |
|---|---:|---:|---|
| Default solver (100 Newton / 50 linesearch) | 31,974 | 1.00× | Reference |
| 8 / 8 | 66,669 | 2.09× | Conservative physics candidate |
| 4 / 4 | 89,921 | 2.81× | Strong reference for training |
| **2 / 4** | **119,662** | **3.74×** | **First 20M research candidate** |
| 2 / 2 | 121,665 | 3.80× | Marginal gain; not yet worth fidelity risk |
| 1 / 1 | 162,930 | 5.10× | **Reject as default: physical warnings** |
| 4 / 4 + torso collisions | 51,429 | 1.61× | Contact-cost control |

All figures are **synchronised, post-JIT full-PPO throughput**, not end-to-end
training FPS. Extra sweeps found 4/default-LS ≈45,243, default-Newton/4-LS
≈75,130, 4/8 ≈81,532, 8/4 ≈73,421 and 4/2 ≈93,451 steps/s. Repeated
baseline arms drifted enough that single-digit FPS differences are inconclusive.

- **1/1 physics is suspect.** Against 4/4 on short *untrained-policy*
  rollouts: stance slip +19.7%, virtual nonfoot strikes +117%, forward velocity
  −8.2%. Similar fall/event rates do not certify acceptable contact dynamics.
- **2/4 looks promising, not validated.** Against 4/4: slip proxy +0.22%,
  virtual nonfoot strikes +6.4%, velocity −1.3%; this short random-policy
  comparison cannot establish locomotion or accurate impulse resolution.
- **Foot-only contacts saved substantial work.** Re-enabling the torso raised
  padded contact slots 17→51, constraint rows 82→218 and reduced 4/4
  throughput ≈43%. Preserve feet-only topology for the learning experiment,
  but test that the virtual torso/knee strikes cannot be gamed.
- **Reset capacity remains 128.** Capacity 192 averaged +5.7% but changed
  throughput +14.2% in one seed and −1.9% in the other; sampled final
  rollouts had no 128-capacity overflow. No established reason to raise it.
- **No successful walking claim.** Short PPO sweeps were initialised afresh.
  `terrain_cross_new` was zero in final sampled rollouts. The completed
  campaign is evidence for *engineering throughput*, not a gait result.

### Open GPU reset-forward parity issue

Compact versus full `mjx.forward` on the 4060 Ti disagreed in **`.qacc`**
(generalised linear/angular/joint accelerations). The later named-field probe
showed **9/28 elements outside rtol=2e−4, atol=2e−5**, max absolute
difference **0.0109558** and safeguarded relative difference **0.009756**.
A prior archived CUDA preflight showed a smaller discrepancy; CPU preflight
passed. Neither observation proves the error harmless. Compare later-step
positions, velocities, support, contact impulses and reset reproducibility
at identical initial states before claiming exact GPU parity.

## Keep versus delete

**Keep:** one unified real-MJX 360-input trainer, 2×2 window of four 75cm
terrain tiles represented by eight exact convex triangular prisms, foot-only
contact plus virtual-body detection, compact-reset fallback, solver caps
explicitly saved with training/evaluation physics, 20M source-pinned gait
runner, 21 core Python tests (ceiling 24), and the GPU performance audit.

**Delete:** the 415-line `scripts/overnight_solver_campaign.py` and its
121-line one-off launch document. The 34 configurations were **benchmark
arms, not 34 automated tests**. Keep results and lessons, not a permanent
experiment framework. Future single-purpose research scripts should also
be retired after conclusions are documented.

## Replacing main — candidate pathway, not an automatic merge

The old `main` policy is the preserved champion. The new branch offers
large engineering value even if its trained walker merely **matches**
the champion on observable locomotion: ~3.7× measured default-solver
throughput in the trial, dynamic 75cm rolling terrain, unified step/slope
curriculum, reset acceleration, a simpler core and maintainable tests.

1. **Prove it learns:** complete the source-pinned 20M 2/4 cold-start run;
   review evolving `vx`, survival, foot contacts, shaft energy and actual
   supported terrain crossings at checkpoints. The 10M foundation means
   early terrain exposure is brief; do not mistake a weak 20M result for
   proof the architecture cannot learn.
2. **Independent frozen-policy parity:** compare champion `main` and the
   new checkpoint on *equivalent physical missions*, with the same requested
   commanded speeds, yaw/strafe control, slopes and step lips. Compare
   achievements, not different reward scales. Require a controllable
   recognisable gait, non-exploited support, and repeatable physical exams;
   raw 1.0m/s need not be the only release criterion.
3. **Physics confidence:** resolve or bound CUDA `.qacc` reset differences,
   ensure solver 2/4 doesn't hide excessive penetration/slip, and verify
   torso/limb contact sentinels and trailing-foot recycling. If necessary
   promote 4/4 instead: accuracy outranks theoretical throughput.
4. **Engineering release gate:** preserve exact source-SHA checkpoint
   provenance, demonstrate uninterrupted and resumed training, protect
   champion assets, run core Python/browser CI, keep ≤24 core Python tests,
   and remove dead experimental paths before any release branch.
5. **Stage promotion:** PR #35 is based on the v5045 core cleanup, not
   directly on main. After the gait and integrity gates, review the stacked
   branches for a clean integration candidate. **Do not merge into main
   without the user's explicit approval**, even if nominal metrics pass.

This report deliberately retains the **decisions** rather than source code
for a research sweep. Larger XProf/HLO detail lives in
`docs/JAX_GPU_WORKLOAD_AUDIT.md`; all test/run code kept in the tree must
justify its continued existence.
