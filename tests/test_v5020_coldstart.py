"""v5020 cold-start: real per-world slope, yawed step, 312 sensor and no old-policy changes."""
from pathlib import Path

import numpy as np
import pytest

from emu.contracts import OBS_SCHEMA_PREVIEW, OBS_SCHEMA_SENSOR
from emu.environment import DomainRandomizationConfig, EmuMJXEnv
from emu.terrain import adaptive_height_course


def _course(tmp_path):
    from emu.contracts import ROBOT_XML
    return adaptive_height_course(ROBOT_XML, tmp_path / 'v5020_true_contact_course.xml')


def _episode_config():
    return {
        'frontier_terrain_height_m': .020,
        'probe_cmd_vx': .16,
        'probe_cmd_vy': 0.,
        'probe_cmd_yaw': 0.,
    }


def test_two_world_varied_step_with_real_contact_slopes_and_memory(tmp_path):
    pytest.importorskip('jax')
    import jax.numpy as jnp
    cfg=DomainRandomizationConfig(
        enabled=True, terrain_radius=.35, terrain_mode='contact')
    env=EmuMJXEnv(
        str(_course(tmp_path)), num_envs=2, seed=733,
        observation_schema=OBS_SCHEMA_PREVIEW,
        terrain_contact_mode='box_surface', terrain_adaptive=True,
        terrain_random_placement=True, terrain_height_exposure=.95,
        domain_config=cfg)
    state,obs=env.reset(_episode_config())
    assert obs.shape==(2,312)
    assert state.sensor_state.terrain_heights.shape==(2,2,51)
    data=state.data
    boxid=env.terrain_box_geom_id
    xyz=np.asarray(data.geom_xpos[:,boxid,:])
    n=np.asarray(env.floor_normal_world)
    assert xyz.shape==(2,3) and n.shape==(2,3)
    assert np.isfinite(xyz).all()
    assert np.isfinite(n).all()
    h=np.asarray(state.task[:,env.T.terrain_height_m]) if hasattr(env, 'T') else None
    # No obstacle may spawn under the feet. Flat-rehearsal worlds park
    # their obstacle at x=5m.
    active=(xyz[:,0]<1.0)
    assert np.all((xyz[active,0]>=.17)&(xyz[active,0]<=.43))
    top_z=xyz[:,2]+float(np.asarray(env.terrain_box_half_size)[2])
    floor_z=-(xyz[:,0]*n[:,0]+xyz[:,1]*n[:,1])/n[:,2]
    # Physical box top lies above that world's actual collidable plane.
    assert np.all((top_z-floor_z)[active]>=0.)
    assert np.all((top_z-floor_z)[active]<=.021)
    step=env.step(state,jnp.zeros((2,6),dtype=jnp.float32),
                  env.params_array(_episode_config()))
    assert step.obs.shape==(2,312)
    assert np.isfinite(np.asarray(step.obs)).all()


def test_232_policies_must_not_use_varied_step_or_combined_contact_slope(tmp_path):
    pytest.importorskip('jax')
    path=str(_course(tmp_path))
    cfg=DomainRandomizationConfig(
        enabled=True, terrain_radius=.15, terrain_mode='contact')
    with pytest.raises(ValueError, match='variable obstacle location|312-input'):
        EmuMJXEnv(path,num_envs=2,observation_schema=OBS_SCHEMA_SENSOR,
                  terrain_contact_mode='box_surface',terrain_adaptive=True,
                  terrain_random_placement=True)
    with pytest.raises(ValueError, match='312-input'):
        EmuMJXEnv(path,num_envs=2,observation_schema=OBS_SCHEMA_SENSOR,
                  terrain_contact_mode='box_surface',terrain_adaptive=True,
                  domain_config=cfg)


def test_training_cli_plans_new_slope_and_step_contract(tmp_path,capsys):
    from emu.cli import main
    root=str(tmp_path/'plan')
    code=main([
        'train','--run-dir',root,'--steps','20000000','--envs','256',
        '--device','cuda','--profile','sim2real',
        '--forward-preview','--terrain-varied-placement',
        '--terrain-height-ceiling-mm','20',
        '--terrain-height-initial-mm','3',
        '--terrain-height-exposure','.55',
        '--contact-terrain-radius','.25','--plan'])
    out=capsys.readouterr().out
    assert code==0
    assert 'rolling_tof_312' in out
    assert 'terrain_random_placement' in out
    assert 'contact' in out
