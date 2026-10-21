"""v5020 cold-start: real per-world slope, yawed step, 312 sensor and no old-policy changes."""
from pathlib import Path

import numpy as np
import pytest

from emu.contracts import OBS_SCHEMA_PREVIEW, OBS_SCHEMA_SENSOR
from emu.environment import DomainRandomizationConfig, EmuMJXEnv
from emu.terrain import adaptive_height_course, box_crossing_position_flags


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


def test_three_world_varied_step_with_real_contact_slopes_and_memory(tmp_path):
    pytest.importorskip('jax')
    import jax.numpy as jnp
    cfg=DomainRandomizationConfig(
        enabled=True, terrain_radius=.35, terrain_mode='contact')
    env=EmuMJXEnv(
        str(_course(tmp_path)), num_envs=3, seed=733,
        observation_schema=OBS_SCHEMA_PREVIEW,
        terrain_contact_mode='box_surface', terrain_adaptive=True,
        terrain_random_placement=True, terrain_height_exposure=.95,
        domain_config=cfg)
    state,obs=env.reset(_episode_config())
    assert obs.shape==(3,312)
    assert state.sensor_state.terrain_heights.shape==(3,2,51)
    data=state.data
    boxid=env.terrain_box_geom_id
    xyz=np.asarray(data.geom_xpos[:,boxid,:])
    n=np.asarray(env.floor_normal_world)
    assert xyz.shape==(3,3) and n.shape==(3,3)
    assert np.isfinite(xyz).all()
    assert np.isfinite(n).all()
    # No obstacle may spawn under the feet. Flat-rehearsal worlds park
    # their obstacle at x=5m.
    active=(xyz[:,0]<1.0)
    assert np.all((xyz[active,0]>=.17)&(xyz[active,0]<=.43))
    top_z=xyz[:,2]+float(np.asarray(env.terrain_box_half_size)[2])
    floor_z=-(xyz[:,0]*n[:,0]+xyz[:,1]*n[:,1])/n[:,2]
    # Physical box top lies above that world's actual collidable plane.
    assert np.all((top_z-floor_z)[active]>=0.)
    assert np.all((top_z-floor_z)[active]<=.021)
    step=env.step(state,jnp.zeros((3,6),dtype=jnp.float32),
                  env.params_array(_episode_config()))
    assert step.obs.shape==(3,312)
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

def test_batched_box_crossing_geometry_is_jittable_with_64_worlds():
    """Reproduce actual GPU smoke dimensions N=64, F=2 (and N=2 silent case).

    This guard checks spatially different obstacles and world-local lane
    offsets; earlier N=2 tests accidentally broadcast [N] over feet.
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def classify(feet,torso,centre,rotation,half):
        return box_crossing_position_flags(
            jnp,foot_xyz=feet,torso_xyz=torso,box_center=centre,
            box_rotation=rotation,box_half_size=half)

    for n in (2,3,64):
        cx=np.linspace(.20,.39,n,dtype=np.float32)
        cy=np.linspace(-.02,.02,n,dtype=np.float32)
        centres=np.stack([cx,cy,np.zeros(n,dtype=np.float32)],axis=1)
        rot=np.broadcast_to(np.eye(3,dtype=np.float32),(n,3,3)).copy()
        half=np.array([.04,.20,.075],dtype=np.float32)
        # World i has its own box far edge: box X + .04.
        feet=np.zeros((n,2,3),dtype=np.float32)
        torso=np.zeros((n,3),dtype=np.float32)
        torso[:,0]=cx+.04+.19
        torso[:,1]=cy
        feet[:,0,0]=cx+.04+.05
        feet[:,1,0]=cx+.04+.06
        feet[:,:,1]=cy[:,None]
        flags=classify(jnp.asarray(feet),jnp.asarray(torso),
                       jnp.asarray(centres),jnp.asarray(rot),jnp.asarray(half))
        assert all(np.asarray(a).shape==(n,) for a in flags)
        assert all(np.asarray(a).all() for a in flags)
        # Fail only world 0, only one of its two feet: no world leakage.
        feet[0,1,0]=cx[0]+.04+.01
        flags=classify(jnp.asarray(feet),jnp.asarray(torso),
                       jnp.asarray(centres),jnp.asarray(rot),jnp.asarray(half))
        assert not bool(np.asarray(flags[1])[0])
        assert np.asarray(flags[1])[1:].all()
        # Lateral lane follows box Y per world, not a global origin.
        feet[0,1,0]=cx[0]+.04+.06
        feet[-1,:,1]=cy[-1]+.15
        flags=classify(jnp.asarray(feet),jnp.asarray(torso),
                       jnp.asarray(centres),jnp.asarray(rot),jnp.asarray(half))
        assert not bool(np.asarray(flags[2])[-1])
        assert np.asarray(flags[2])[:-1].all()
