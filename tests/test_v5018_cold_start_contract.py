"""Cold-start is a supported first-class training path, not a champion fork.

The frozen 260M actor is an experimental control. Both legacy and current
sensor actor shapes must construct a fresh PPO lineage from source+seed with
NO checkpoint, policy export or external foundation requirement.
CPU-only contracts: these do not claim 20M steps learned to walk.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from emu.branch_lab import inspect_run, one_run_config
from emu.checkpoint_io import resume_training_lineage
from emu.contracts import OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR
from emu.ppo import expected_param_shapes
from emu.training_config import training_sensor_contract


def _args(root: Path, *, sensor: bool, profile: str = 'foundation'):
    return SimpleNamespace(
        run_dir=str(root), steps=256, envs=8, seed=733, device='cuda',
        profile=profile, pitch_deg=None, motor_guidance_weight=0.,
        contact_slope_deg=0., contact_terrain_radius=0.,
        sensor_actor=sensor, resume='', plan=False,
    )


@pytest.mark.parametrize('sensor,expected_schema,expected_dim', [
    (False, OBS_SCHEMA_LEGACY, 220),
    (True, OBS_SCHEMA_SENSOR, 232),
])
def test_fresh_foundation_requires_no_champion_or_parent(
    tmp_path, sensor, expected_schema, expected_dim,
):
    run_dir = tmp_path / ('cold_sensor' if sensor else 'cold_legacy')
    args = _args(run_dir, sensor=sensor)
    cfg = one_run_config(args)
    assert cfg.observation_schema == expected_schema
    assert cfg.resume == ''
    assert cfg.seed == 733
    assert cfg.envs == 8
    assert cfg.steps == 256
    assert Path(cfg.run_dir).resolve() == run_dir.resolve()
    meta = training_sensor_contract(cfg)
    assert (meta is not None) is sensor
    assert expected_param_shapes(obs_dim=expected_dim)['pi'][0]['w'][0] == expected_dim

    # If a fresh run ever starts resolving a checkpoint, this test must
    # fail. No policy/checkpoint/showcase file should be needed.
    provenance = dict(available=True, dirty=False, commit='cold-start-contract-test')
    with patch('emu.checkpoint_io.load_checkpoint',
               side_effect=AssertionError('cold start opened a pretrained checkpoint')):
        parent, completed = inspect_run(cfg, provenance=provenance, resume='')
        payload, step, done = resume_training_lineage(
            run_dir, resume='', target_steps=cfg.steps, envs=cfg.envs,
            expected_observation_schema=cfg.observation_schema,
            expected_sensor_contract=meta)
    assert parent is None and not completed
    assert payload is None and step == 0 and not done
    assert not run_dir.exists()  # pure planning; no hidden files written


def test_auto_resume_is_not_silent_cold_start(tmp_path):
    cfg = one_run_config(_args(tmp_path/'empty', sensor=True))
    provenance = dict(available=True, dirty=False, commit='cold-start-contract-test')
    with pytest.raises(RuntimeError, match='resume auto requested'):
        inspect_run(cfg, provenance=provenance, resume='auto')


def test_future_sim2real_can_initialise_without_foundation(tmp_path):
    cfg = one_run_config(_args(tmp_path/'sim2real_fresh', sensor=True,
                               profile='sim2real'))
    assert cfg.resume == ''
    assert cfg.observation_schema == OBS_SCHEMA_SENSOR
    parent, completed = inspect_run(
        cfg, provenance=dict(available=True, dirty=False,
                             commit='cold-start-contract-test'), resume='')
    assert parent is None and not completed
    # This only proves allowed initialisation, not that skipping easy
    # foundation would produce competent locomotion or safe curriculum.
