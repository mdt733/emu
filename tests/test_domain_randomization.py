from dataclasses import asdict


from emu.domain_randomization import DomainRandomizationConfig, doctrine_dict as domain_doctrine
from emu.training_config import TrainingConfig, foundation_config, mature_config


def test_v5013_profiles_keep_domain_randomization_off_by_default(tmp_path):
    f = foundation_config(run_dir=tmp_path / 'f', seed=733, device='cuda', envs=64)
    m = mature_config('graph', run_dir=tmp_path / 'm', steps=120_000_000,
                      seed=733, device='cuda', envs=64, resume=tmp_path / 'x.pkl')
    for cfg in (TrainingConfig(), f, m):
        assert cfg.domain_randomization is False
        assert cfg.domain_robot_radius == 0.0
        assert cfg.domain_terrain_radius == 0.0
        assert cfg.domain_hardware_radius == 0.0


def test_domain_config_clamps_only_public_radii_and_mixture():
    cfg = DomainRandomizationConfig(True, 1.5, -0.2, 0.7, 2.0, -1.0).normalized()
    assert cfg.enabled
    assert cfg.robot_radius == 1.0
    assert cfg.terrain_radius == 0.0
    assert cfg.hardware_radius == 0.7
    assert cfg.near_nominal_fraction == 0.90
    assert cfg.hard_tail_fraction == 0.0
    assert domain_doctrine()['MAX_SLOPE_DEG'] > 0


def test_domain_parameters_are_not_added_to_actor_observation_contract():
    from emu.contracts import BASE_OBS_FIELDS, BASE_OBS_DIM, OBS_DIM, HISTORY_LEN
    assert not any('domain' in name or 'slope' in name or 'friction' in name or 'mass' in name
                   for name in BASE_OBS_FIELDS)
    assert BASE_OBS_DIM == 55
    assert OBS_DIM == BASE_OBS_DIM * HISTORY_LEN == 220
