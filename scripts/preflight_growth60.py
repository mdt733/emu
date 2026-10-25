#!/usr/bin/env python3
"""Read-only exact 40M->60M full-learning-state preflight. NEVER starts PPO."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from emu.checkpoint_io import load_checkpoint
from emu.curriculum import RegimeGovernor, TrainingHealth, CapabilityField
from emu.run import validate_40m_to_60m_handoff
from emu.training_config import (
    TrainingConfig, UNIFIED_GROWTH_STEPS, training_sensor_contract,
)

def main() -> int:
    root = (Path.home() / "robot_lab" / "emu_runs" /
            "unified_maturity40_54345b2")
    parent = root / "checkpoints" / "final.pkl"
    prior = TrainingConfig.load(root / "training_config.json")
    candidate = replace(
        prior, steps=UNIFIED_GROWTH_STEPS, device="cuda", resume="",
        run_dir=str(root.parent / "unified_growth60_PLANNED_ONLY"))
    # The loader checks shape, finite PPO+Adam tree, array-only sensor revision,
    # physical geometry, normalizer and mandatory rewind fields.
    payload = load_checkpoint(
        parent, expected_observation_schema=candidate.observation_schema,
        expected_sensor_contract=training_sensor_contract(candidate))
    record = validate_40m_to_60m_handoff(candidate, parent, payload)
    governor = RegimeGovernor(
        fall_target=candidate.fall_target,
        difficulty_base_step=candidate.difficulty_base_step,
        regime_command_frac=candidate.regime_command_frac,
        athletic_rehearsal_frac=candidate.athletic_rehearsal_frac,
        open_frontier=candidate.open_frontier,
        unified_capability_focus=candidate.unified_capability_focus)
    governor.load_state_dict(payload["curriculum_state"])
    health = TrainingHealth()
    health.load_state_dict(payload["health_state"])
    CapabilityField.from_state_dict(payload["field_state"])
    summary = {
        "result": "PASS: read-only ancestor identity and learning-state preflight",
        "parent": record, "training_contract": asdict(candidate),
        "frontiers": governor.params_dict(),
        "normalizer_count": float(payload["obs_norm"]["count"]),
        "health": health.state_dict(),
        "launch_authorised": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("60M training remains PRE-LAUNCH; no run directory or checkpoint was changed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
