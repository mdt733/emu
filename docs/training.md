# EMU training lifecycle and historical comparison design

The common learning loop serves the current v5015 ladder plus historical v5013/v5014 experiments. For the **actual v5014 phase sequence** and physical radii, use [v5014-production.md](v5014-production.md) rather than this history/reference page.

## A training update

1. Advance vectorised MJX environments under the current observation/control contract.
2. Update the actor/critic using PPO and the collected rollout (no replay buffer).
3. Reduce rollout telemetry into performance, tracking and semantic capability events.
4. Update curriculum and training-health decisions; the field proposes bounded practice inside permitted physical frontiers.
5. Assemble the next fixed-shape task packet, save eligible snapshots and occasionally run held-out exams.

Metric provenance should refer to the policy/normaliser/field state that generated the rollout. The evaluator never trains PPO or writes back into the capability map.

## Foundation and mature training

The foundation ends at the **first complete PPO update boundary at or above 20M environment steps** (at default 2048 worlds × 32 rollout frames, 20,054,016 steps). It learns basic dynamic stepping and permits curriculum retreat, but intentionally disarms network/optimizer rollback. Later mature training can permit health-triggered rollback when there is actually useful competence to preserve.

A **learning-state resume** restores policy/critic, optimizer, observation normaliser, curriculum, training health and capability field from a validated schema-7 checkpoint. The simulator episodes and current rollout buffers start afresh. It is not a byte-for-byte replay of a physical frame. Invalid snapshots must fail loudly, with only validated rotating snapshots considered for explicit recovery.

The shared mature profile freezes running observation-normalizer statistics at its configured absolute-step threshold. Do not change observation ordering, actor dimensions, robot XML or actuator meaning for an ongoing run. Phase compatibility is also enforced by `emu/production.py`.

## Why v5013 had paired branches (historical)

The v5013 research used a common foundation checkpoint as the exact ancestor for **shadow** (capability field observes only) versus **graph** (capability field can select bounded practice) continuations. This isolates teaching permission better than comparing unrelated cold starts. The experiment varied dose, source of teaching, evidence feedback and influence schedule—**not** the current production physical domain or PPO interface.

The `field_taught` provenance marker indicates that practice was field-selected; it is not part of the actor's 220 inputs or a reward. The `B4_clean_evidence` experiment let those rollouts train PPO but forbade them from self-confirming the capability field.

See [tournament.md](tournament.md) and the archived original [results](../results/v5013_tournament/README.md). The tournament is retained for reproducibility. Today's recommended workflow is `emu ladder --stage plan` and `bash scripts/run_ladder.sh`. The older `forge` command still reproduces a historic single-lineage plan.
