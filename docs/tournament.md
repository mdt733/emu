# v5013 capability-teacher tournament

> **Historical experiment (v5013).** The completed tournament informed v5014 but is not the default production run. Its original results are preserved in [../results/v5013_tournament/](../results/v5013_tournament/README.md).

v5012 showed a repeatable trade: strong graph teaching improved internal capability-map competence and some growth/headroom tests, but hurt hard recovery/robustness and worsened held-out field calibration. v5013 therefore asks three questions in order:

1. **Dose** — is field teaching useful when it does not dominate the training distribution?
2. **Mechanism** — are node births, maneuver edges, recovery-severity teaching, or self-reinforcing evidence responsible for the trade?
3. **Schedule** — should the field teach continuously, arrive late, ramp slowly, or tutor and then get out of the way?

The experiment is a successive-halving funnel. It intentionally does **not** run twelve 120M branches.

## The twelve hypotheses

### A — dose screen, all to 50M total steps

- `A0_shadow` — 0%, field observes only.
- `A1_graph10` — full graph teacher, 10% maximum influence.
- `A2_graph25` — full graph teacher, 25%.
- `A3_graph40` — full graph teacher, 40%.
- `A4_graph65` — full graph teacher, 65%; v5012-strength replication.

If two doses are within 0.015 promotion utility, the lower dose wins the tie. This deliberately prefers the least intervention that produces the same evidence.

### B — mechanism screen, four new arms to 50M

The dose selected by A is inherited automatically.

- `B1_nodes_only` — node-selected births, including the node layer's recovery-severity recommendation; no edge teaching.
- `B2_edges_only` — maneuver-edge targets only.
- `B3_no_recovery` — node + edge targeting, but node anchors cannot alter recovery severity.
- `B4_clean_evidence` — full node + edge + recovery teacher, but field-selected practice may train PPO **without writing evidence back to the field**.

`B4` is the direct self-confirmation test. A teacher-only provenance bit follows field-selected task/transition targets. It is not an actor channel and does not enter reward. In clean-evidence mode, semantic node/recovery/edge event machines are broken at those provenance boundaries so a trial cannot begin on unbiased experience and finish inside self-selected practice.

### C — schedule screen, three new arms to 60M

The best mechanism from A/B is inherited automatically.

- `C1_slow_ramp` — influence ramps from 0 at 20M to full at 60M.
- `C2_delayed` — governor-only through 40M, then ramp to full at 60M.
- `C3_tutor_release` — full influence by 30M, hold through 45M, then fade to zero at 60M.

Shadow and the baseline winning mechanism are continued only the extra steps required to compare at the same 60M boundary.

## Promotion funnel

After C, four **scientifically distinct** representatives earn 80M:

1. shadow;
2. best dose arm;
3. best B mechanism variant;
4. best C schedule variant.

This is deliberate. The 80M set is not a generic top-four score list that could fill itself with nearly identical teachers.

At 80M, promotion becomes ruthless: shadow is always retained, and only the top two non-shadow challengers reach 120M. At 120M, the best challenger is chosen for independent seed confirmation.

Seed 1742 builds its own fresh v5013 foundation and runs only:

- shadow;
- the seed-733 best challenger.

The runner never buys a third seed automatically. A mixed/tied confirmation is reported as such and should be inspected before more GPU time is spent.

## Held-out screen

Every tournament decision point runs a cheap, read-only stress screen with 512 envs by default. It adds four longer-ruler probes:

- `resilience100`;
- `robust70`;
- `growth140`;
- `growthcombo140`.

These probes are **not** curriculum inputs and do not enter the ordinary broad score. They exist because exactly this harder recovery/growth contrast exposed the v5012 over-teaching failure. The current one-frame hardware target-delay doctrine remains unchanged, so they are regression/stress coordinates rather than claims of physical limits.

The screen costs far less than blindly promoting a weak arm another 30–70M training steps.

## Promotion guardrails

A challenger is flagged if, relative to paired shadow at the same stage:

- broad score is more than 0.04 lower;
- mean `resilience100/robust70` survival is more than 0.08 lower;
- valid held-out recovery is more than 0.10 lower;
- field calibration MAE is more than 0.20 worse.

At 80M/120M, a challenger also fails the cloud guard if shadow has valid held-out recovery evidence and the challenger has none.

The transparent promotion utility is:

```text
30% broad score
30% hard recovery (resilience100 + robust70)
15% hard growth headroom (growth140 + growthcombo140)
10% held-out cloud recovery
10% worst-direction recovery
 5% calibration honesty
```

Guardrails outrank utility. If every arm violates a guardrail, ranking still continues so the experiment can finish, but the decision file records the violations rather than pretending the candidate passed.

## Actual teaching exposure

`metrics.csv` records `occupancy_field_taught`, the fraction of rollout frames whose command target came from the field. This is distinct from requested `field_node_influence` / `field_edge_influence`: a requested dose may not be realizable while useful anchors are sparse.

## Running

Inspect the full adaptive plan without touching JAX/GPU:

```bash
emu tournament --plan
```

Run the entire funnel:

```bash
emu tournament
```

Useful inspection breakpoints are restart-safe:

```bash
emu tournament --stop-after A
emu tournament --stop-after B
emu tournament --stop-after C
emu tournament --stop-after 80
emu tournament --stop-after 120
```

Running the same command again continues from validated current-schema checkpoints. `experiment_spec.json` is immutable per arm and `tournament_manifest.json` locks seed/env/evaluation settings, so an interrupted experiment cannot silently resume under changed semantics.

Outputs include:

```text
tournament_manifest.json
tournament_decisions.json
tournament_summary.csv
tournament_results.md
seed733/foundation20/
seed733/arms/<experiment>/
seed1742/foundation20/
seed1742/arms/<confirmation>/
```
