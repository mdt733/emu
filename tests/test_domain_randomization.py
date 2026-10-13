from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from emu.domain_randomization import DomainRandomizationConfig, doctrine_dict as domain_doctrine
from emu.morphology import (
    FOOT_RADIUS_M, HIP_HALF_WIDTH_M, SHIN_HALF_COMPONENT_M, THIGH_COMPONENT_M,
    MorphologySpec, render_variant_xml, sample_specs,
)
from emu.paths import ROBOT_XML
from emu.terrain import make_surface, to_mujoco_hfield_data
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


def test_morphology_sampling_is_deterministic_bounded_and_asymmetric():
    a = sample_specs(16, seed=5014, radius=1.0)
    b = sample_specs(16, seed=5014, radius=1.0)
    assert a == b
    assert a[0] == MorphologySpec()
    assert any(abs(x.r_thigh_scale - x.l_thigh_scale) > 1e-6 for x in a[1:])
    for x in a:
        assert 0.94 < x.r_thigh_scale < 1.06
        assert 0.94 < x.l_thigh_scale < 1.06
        assert 0.94 < x.r_shin_scale < 1.06
        assert 0.94 < x.l_shin_scale < 1.06
        assert 0.90 < x.torso_mass_scale < 1.10


def test_morphology_xml_changes_dimensions_coherently_without_topology_change():
    base = Path(ROBOT_XML).read_text(encoding='utf-8')
    spec = MorphologySpec(
        r_thigh_scale=1.02, l_thigh_scale=0.99,
        r_shin_scale=1.03, l_shin_scale=0.98,
        hip_width_scale=1.02, r_foot_scale=1.04, l_foot_scale=0.97,
        torso_mass_scale=1.06, torso_com_x_shift_m=0.002,
    )
    out = render_variant_xml(base, spec)
    a = ET.fromstring(base)
    b = ET.fromstring(out)
    names_a = sorted((e.tag, e.get('name')) for e in a.iter() if e.get('name'))
    names_b = sorted((e.tag, e.get('name')) for e in b.iter() if e.get('name'))
    assert names_a == names_b

    r_shin = b.find(".//body[@name='r_shin']")
    assert r_shin is not None
    pos = [float(x) for x in r_shin.get('pos').split()]
    assert np.isclose(pos[0], -THIGH_COMPONENT_M * 1.02)
    assert np.isclose(pos[2], -THIGH_COMPONENT_M * 1.02)

    r_lower = b.find(".//body[@name='r_shin_lower']")
    assert r_lower is not None
    pos = [float(x) for x in r_lower.get('pos').split()]
    assert np.isclose(pos[0], SHIN_HALF_COMPONENT_M * 1.03)

    r_foot = b.find(".//geom[@name='r_foot_ball']")
    assert r_foot is not None
    assert np.isclose(float(r_foot.get('size')), FOOT_RADIUS_M * 1.04)

    r_hip = b.find(".//body[@name='r_hip']")
    assert r_hip is not None
    assert np.isclose(float(r_hip.get('pos').split()[1]), -HIP_HALF_WIDTH_M * 1.02)

    torso = b.find(".//body[@name='torso']")
    assert torso is not None
    expected_z = max(
        THIGH_COMPONENT_M * 1.02 + 2 * SHIN_HALF_COMPONENT_M * 1.03 + FOOT_RADIUS_M * 1.04,
        THIGH_COMPONENT_M * 0.99 + 2 * SHIN_HALF_COMPONENT_M * 0.98 + FOOT_RADIUS_M * 0.97,
    )
    assert np.isclose(float(torso.get('pos').split()[2]), expected_z)


def test_rough_surface_is_deterministic_bounded_and_zero_mean():
    a = make_surface(123, 0.8, rows=33, cols=33)
    b = make_surface(123, 0.8, rows=33, cols=33)
    assert a.shape == (33, 33)
    assert np.array_equal(a, b)
    assert abs(float(a.mean())) < 1e-6
    assert float(np.ptp(a)) > 0.0
    packed = to_mujoco_hfield_data(a, elevation_z_m=0.024)
    assert np.all((packed >= 0.0) & (packed <= 1.0))


def test_domain_parameters_are_not_added_to_actor_observation_contract():
    from emu.contracts import BASE_OBS_FIELDS, BASE_OBS_DIM, OBS_DIM, HISTORY_LEN
    assert not any('domain' in name or 'slope' in name or 'friction' in name or 'mass' in name
                   for name in BASE_OBS_FIELDS)
    assert BASE_OBS_DIM == 55
    assert OBS_DIM == BASE_OBS_DIM * HISTORY_LEN == 220
