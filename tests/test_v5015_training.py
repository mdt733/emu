"""CPU-only v5015 training contract tests. No GPU needed."""
from dataclasses import asdict
import json
import math
import numpy as np
from emu.training_config import TrainingConfig
from emu.phase_config import _phase_config
from emu.ladder_recipe import FOUNDATION, ROUNDS, ARMS
from emu.evaluation import broad_score
from emu.tracking_contract import node_window_success_numpy


def test_legacy_production_and_actor_contract_unchanged():
    from emu.contracts import BASE_OBS_DIM, OBS_DIM, ACTION_DIM, PITCH_CMD_MAX_RAD
    assert (BASE_OBS_DIM, OBS_DIM, ACTION_DIM, PITCH_CMD_MAX_RAD) == (55, 220, 6, .14)
    assert FOUNDATION == 20_000_000


def test_v5015_ladder_recipe_and_phase_config(tmp_path):
    assert tuple(ROUNDS.values()) == (26_000_000, 34_000_000, 46_000_000)
    assert len(ARMS) == 4
    cfg = _phase_config('20_breadth', 140_000_000, .3, .25, .35,
                        root=tmp_path, seed=733, device='cpu', envs=8, resume=None,
                        profile='v5015', pitch_deg=25.)
    assert cfg.attitude_profile == 'v5015'
    assert cfg.pitch_command_max_deg == 25
    assert cfg.domain_terrain_radius == .25
    old = _phase_config('20_breadth', 140_000_000, .3, .25, .35,
                        root=tmp_path, seed=733, device='cpu', envs=8, resume=None)
    assert old.attitude_profile == 'v5014'


def test_historical_v5014_config_loading_stays_strict(tmp_path):
    config = asdict(TrainingConfig())
    del config['attitude_profile']
    del config['pitch_command_max_deg']
    file = tmp_path / 'config.json'
    file.write_text(json.dumps(config))
    assert TrainingConfig.load(file).attitude_profile == 'v5014'
    del config['allow_rollback']
    file.write_text(json.dumps(config))
    try:
        TrainingConfig.load(file)
    except ValueError:
        pass
    else:
        raise AssertionError('old required config field was silently accepted')


def test_45_degree_chassis_cannot_alone_fail_healthy_robot():
    z = np.zeros(50, np.float32)
    one = np.ones(50, np.float32)
    pitch_gz = np.full(50, math.cos(math.pi/4), np.float32)
    assert not node_window_success_numpy(z,z,z,z,z,z,pitch_gz,one,z)
    assert node_window_success_numpy(z,z,z,z,z,z,pitch_gz,one,z,
                                     structural_attitude_ok=one)
    assert not node_window_success_numpy(z,z,z,z,z,z,pitch_gz,one,z,
                                         structural_attitude_ok=z)


def test_v5015_heldout_scoring_penalises_crouch_and_pitch_error():
    standard = broad_score({}, profile='v5014')
    assert standard == broad_score({}, profile='v5015')
    panels = ('dynamic50', 'dynamic80', 'combo70', 'resilience75')
    bad = {f'eval_{p}_height_shortfall_m': .040 for p in panels}
    bad.update({f'eval_{p}_pitch_mae': .4 for p in panels})
    assert broad_score(bad, profile='v5014') == standard
    assert broad_score(bad, profile='v5015') < standard - .20


def test_ladder_plan_does_not_start_training(capsys):
    from emu.branch_lab import main
    assert main(['--stage','plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['version'] == 'v5015-lab1'
    assert plan['foundation'] == FOUNDATION


def test_atlas_reads_phase_contract_and_root_final_lineage(tmp_path):
    from emu.atlas import checkpoint_training_profile
    phase = tmp_path / '20_breadth'
    (phase / 'checkpoints').mkdir(parents=True)
    (phase / 'training_config.json').write_text(json.dumps({'attitude_profile':'v5015','pitch_command_max_deg':25.0}))
    assert checkpoint_training_profile(phase/'checkpoints'/'final.pkl') == ('v5015', 25.0)
    (tmp_path / 'lineage.json').write_text(json.dumps({'version':'v5015','phases':[{'pitch_max_deg':35.0}]}))
    assert checkpoint_training_profile(tmp_path/'final.pkl') == ('v5015', 35.0)


def test_pre_knobs_config_defaults_remain_backward_compatible(tmp_path):
    from emu.training_config import PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST
    old = asdict(TrainingConfig())
    for key in ('ppo_lr','ppo_entropy_cost','ppo_value_cost'):
        old.pop(key)
    filename = tmp_path/'old.json'
    filename.write_text(json.dumps(old))
    cfg=TrainingConfig.load(filename)
    assert (cfg.ppo_lr,cfg.ppo_entropy_cost,cfg.ppo_value_cost) == (
        PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST)
