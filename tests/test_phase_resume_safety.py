"""Existing phase reuse must preserve critical seed/shape/domain choices."""
from dataclasses import replace
from pathlib import Path
import pytest

from emu.phase_config import _phase_config, verify_existing_phase_config


def test_rejects_changed_seed_env_count_and_domain(tmp_path):
    root=tmp_path/'run'
    phase='10_adapt'
    c=_phase_config(phase,60_000_000,.10,.08,.12,root=root,seed=733,device='cuda',envs=2048,resume=None)
    run_dir=root/phase
    c.save(run_dir/'training_config.json')
    ck=run_dir/'checkpoints'; ck.mkdir()
    (ck/'latest.pkl').write_bytes(b'checkpoint placeholder')
    verify_existing_phase_config(run_dir,c)
    # Changing location/device/resume is allowed; not learning-contract changes.
    verify_existing_phase_config(run_dir,replace(c,device='cpu',resume='/different/path'))
    for bad in (replace(c,seed=1742),replace(c,envs=512),
                replace(c,domain_hardware_radius=.5),replace(c,field_max_influence=.65)):
        with pytest.raises(RuntimeError,match='different training configuration'):
            verify_existing_phase_config(run_dir,bad)

def test_missing_config_prevents_checkpoint_reuse(tmp_path):
    phase=tmp_path/'10_adapt'; (phase/'checkpoints').mkdir(parents=True)
    (phase/'checkpoints'/'latest.pkl').write_bytes(b'bad')
    c=_phase_config('10_adapt',60_000_000,.10,.08,.12,root=tmp_path,seed=733,device='cuda',envs=2048,resume=None)
    with pytest.raises(RuntimeError,match='without training_config'):
        verify_existing_phase_config(phase,c)


def test_ladder_uses_shared_phase_compatibility():
    from emu import phase_config, branch_lab
    assert branch_lab._phase_config is phase_config._phase_config
    assert branch_lab.verify_existing_phase_config is phase_config.verify_existing_phase_config
