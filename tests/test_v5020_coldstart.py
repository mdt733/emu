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


def test_three_world_312_real_step_frozen_exams_compile(tmp_path):
    """The tournament must JIT its REAL physical and motor screens before 80M."""
    pytest.importorskip('jax')
    import jax
    import jax.numpy as jnp
    from emu.ppo import make_trainer
    terrain=DomainRandomizationConfig(
        enabled=True,terrain_radius=.22,terrain_mode='contact')
    env=EmuMJXEnv(
        str(_course(tmp_path)),num_envs=3,seed=1771,
        observation_schema=OBS_SCHEMA_PREVIEW,
        terrain_contact_mode='box_surface',terrain_adaptive=True,
        terrain_random_placement=True,
        terrain_height_exposure=.95,terrain_height_edge_fraction=.80,
        domain_config=terrain)
    trainer=make_trainer(
        env,seed=733,lr=3e-4,rollout=2,epochs=1,minibatches=1)
    ep=_episode_config()
    params=env.params_array(ep)
    # make_initial_state consumes a task-parameter mapping, not the packed
    # JAX array passed directly to frozen examiner methods below.
    state=trainer['make_initial_state'](ep)
    norm_mean=jnp.zeros((312,),dtype=jnp.float32)
    norm_std=jnp.ones((312,),dtype=jnp.float32)
    key=jax.random.PRNGKey(1991)
    course=trainer['evaluate_terrain_summary'](
        state.params,params,norm_mean,norm_std,key,4)
    power=trainer['evaluate_efficiency'](
        state.params,params,norm_mean,norm_std,key,4)
    assert course.shape==(3,)
    assert power.shape==(5,)
    assert np.isfinite(np.asarray(course)).all()
    assert np.isfinite(np.asarray(power)).all()
    assert 0<=float(course[0])<=3
    assert 0<=float(course[1])<=1 and 0<=float(course[2])<=1



def test_heldout_exact_step_exam_never_samples_shorter_height(tmp_path):
    """At certification, a 20mm label means a physical 20mm box top.

    Ordinary training must remain free to sample flat and interior
    obstacles. This test uses the same fixed-topology MJX mocap course.
    """
    pytest.importorskip("mujoco")
    jax=pytest.importorskip("jax")
    from emu.environment import T
    env=EmuMJXEnv(
        str(_course(tmp_path)), num_envs=3, seed=1701,
        observation_schema=OBS_SCHEMA_PREVIEW,
        terrain_contact_mode="box_surface", terrain_adaptive=True,
        terrain_random_placement=True, terrain_exact_height_exam=True)
    for mm in (5, 20, 80):
        request={**_episode_config(), "frontier_terrain_height_m":mm/1000.}
        state,obs=env.reset(request)
        tasks=np.asarray(jax.device_get(state.task[:, T.terrain_height_m]))
        np.testing.assert_allclose(tasks, np.full(3,mm/1000.),atol=1e-6,rtol=0.)
        box=np.asarray(jax.device_get(
            state.data.geom_xpos[:,env.terrain_box_geom_id,:]))
        box_top=box[:,2]+float(np.asarray(env.terrain_box_half_size)[2])
        np.testing.assert_allclose(box_top,np.full(3,mm/1000.),atol=1e-5,rtol=0.)
        assert obs.shape==(3,312)
        assert np.isfinite(np.asarray(jax.device_get(obs))).all()

    # A regular training world has exactly the same real geometry but
    # uses a varied-height sampler and may leave a birth on flat ground.
    training=EmuMJXEnv(
        str(_course(tmp_path)),num_envs=16,seed=1701,
        observation_schema=OBS_SCHEMA_PREVIEW,
        terrain_contact_mode="box_surface",terrain_adaptive=True,
        terrain_height_exposure=.50,terrain_random_placement=True)
    state,_=training.reset({**_episode_config(),"frontier_terrain_height_m":.02})
    heights=np.asarray(jax.device_get(state.task[:,T.terrain_height_m]))
    assert np.all(heights<=.02+1e-6)
    assert np.any(heights<.019)


@pytest.mark.parametrize('direction', ['up', 'down'])
@pytest.mark.parametrize('height_mm', [5, 10, 20])
def test_elevation_fixture_real_surfaces_and_sensor_first_hits(tmp_path, direction, height_mm):
    import mujoco
    from emu.terrain import elevation_course
    from emu.forward_tof import first_hit_ranges
    from emu.contracts import ROBOT_XML
    path = elevation_course(ROBOT_XML, tmp_path/'elevation.xml', height_mm,
                            direction=direction)
    # Add a tiny independent contact probe on the upper platform. The
    # platform must support it without a phantom floor contact beneath it.
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    high_x = .50 if direction == 'up' else 0.
    probe = ET.SubElement(tree.getroot().find('worldbody'), 'body',
        {'name': 'surface_probe', 'pos': f'{high_x} .35 {height_mm/1000.+.009}'})
    ET.SubElement(probe, 'freejoint')
    ET.SubElement(probe, 'geom', {'name': 'probe', 'type': 'sphere',
                                'size': '.01', 'mass': '.01'})
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'v5018_obstacle_step')
    h = height_mm/1000.
    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'probe')
    contacts = [c for c in data.contact if probe_id in (c.geom1, c.geom2)]
    assert contacts
    assert all(box in (c.geom1, c.geom2) for c in contacts)
    assert all(abs(c.frame[2]) > .99 for c in contacts)
    # Disable probe visibility for the sensor parity rays below.
    model.geom_group[probe_id] = 5
    ray_groups = np.array([1,1,1,1,1,0], dtype=np.uint8)
    # Rays stay beside the robot, but well inside the platform width.
    for x, expected in ((0., 0. if direction == 'up' else h),
                        (.50, h if direction == 'up' else 0.)):
        origin = np.array([x, .35, .30])
        normal = np.zeros(3)
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, np.array([0.,0.,-1.]),
                                 ray_groups, True, -1, geom, normal)
        assert .30-distance == pytest.approx(expected, abs=1e-9)
        np.testing.assert_allclose(normal, [0.,0.,1.], atol=1e-9)
        ranges, valid = first_hit_ranges(np, origin[None], np.eye(3)[None],
            np.zeros((1,3)), np.array([[0.,0.,1.]]),
            box_center=data.geom_xpos[box][None],
            box_rotation=data.geom_xmat[box].reshape(1,3,3),
            box_half_size=model.geom_size[box],
            mounts_body=((0.,0.,0.),), beams_body=((0.,0.,-1.),))
        assert valid[0,0]
        assert ranges[0,0] == pytest.approx(distance, abs=1e-8)
    # The vertical face is real and has a horizontal outward normal.
    origin = np.array([0. if direction == 'up' else .50, .35, h/2])
    sign = 1. if direction == 'up' else -1.
    normal = np.zeros(3)
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, np.array([sign,0.,0.]),
                             ray_groups, True, -1, geom, normal)
    assert distance == pytest.approx(.25)
    assert geom[0] == box
    np.testing.assert_allclose(normal, [-sign,0.,0.], atol=1e-9)
    assert model.geom_size[box,0]*2 >= 1.
    assert model.qpos0[2] == pytest.approx(.2584 + (h if direction == 'down' else 0.))


def test_mjx_down_step_birth_is_on_upper_landing(tmp_path):
    from emu.terrain import elevation_course
    from emu.contracts import ROBOT_XML
    from emu.environment import NEUTRAL_ROOT_Z
    path = elevation_course(ROBOT_XML, tmp_path/'down.xml', 20, direction='down')
    env = EmuMJXEnv(str(path), num_envs=2, seed=733,
                   observation_schema=OBS_SCHEMA_PREVIEW,
                   terrain_contact_mode='box_surface')
    state, obs = env.reset({'probe_cmd_vx': .10, 'probe_cmd_vy': 0.,
                            'probe_cmd_yaw': 0., 'steady_difficulty': 0.,
                            'resilience_difficulty': 0., 'posture_difficulty': 0.,
                            'regime_frac_steady': 1., 'regime_frac_transition': 0.,
                            'regime_frac_resilience': 0.})
    assert obs.shape == (2,312)
    np.testing.assert_allclose(np.asarray(state.data.qpos[:,2]),
                               NEUTRAL_ROOT_Z+.020, atol=1e-6)


def test_elevation_certificate_requires_source_contact_landing_and_more_gait():
    """Pure NumPy adversarial examples: no GPU and no alleged obstacle rewards."""
    from emu.terrain import elevation_transition_update

    def run(*, source=True, contact=True, target=True, gait=.4,
            both_feet=True, upright=True, progress=True):
        state = (np.array([False]), np.array([False]), np.array([False]),
                 np.array([0.],dtype=np.float32), np.array([0],dtype=np.int32),
                 np.array([0.],dtype=np.float32), np.array([False]))
        edge = np.array([.25],dtype=np.float32)
        # Two frames of genuine source-side approach.
        for _ in range(2):
            state = elevation_transition_update(np,
                source_seen=state[0],contacted=state[1],landed=state[2],
                landing_x=state[3],supported_frames=state[4],
                post_landing_gait=state[5],crossed=state[6],
                source_support=np.array([source]),
                obstacle_contact=np.array([False]),
                target_support=np.array([False]),torso_x=np.array([0.]),
                feet_x=np.array([[0.,0.]]),edge_x=edge,
                upright=np.array([True]),gait=np.array([0.]),
                active=np.array([True]))
        # Transition then 10 frames on the new level. Traversal happens
        # AFTER first supported landing, not on the approach or mid-air.
        for i in range(11):
            x = .35 + (min(i, 10)*.005 if progress else 0.)
            feet = np.array([[x-.01, x-.02 if both_feet else .20]])
            state = elevation_transition_update(np,
                source_seen=state[0],contacted=state[1],landed=state[2],
                landing_x=state[3],supported_frames=state[4],
                post_landing_gait=state[5],crossed=state[6],
                source_support=np.array([False]),
                obstacle_contact=np.array([contact]),
                target_support=np.array([target]),torso_x=np.array([x]),
                feet_x=feet,edge_x=edge,
                upright=np.array([upright]),gait=np.array([gait]),
                active=np.array([True]))
        return bool(state[-1][0])

    assert run()
    for kw in ({"source":False}, {"contact":False}, {"target":False},
               {"gait":0.}, {"both_feet":False}, {"upright":False},
               {"progress":False}):
        assert not run(**kw), kw


@pytest.mark.parametrize("direction", ["up", "down"])
def test_true_elevation_frozen_mjx_exam_compiles_on_small_worlds(tmp_path, direction):
    """Exercise real collidable geometry and JIT inference; no claim of gait skill."""
    import jax
    import jax.numpy as jnp
    from emu.terrain import elevation_course
    from emu.contracts import ROBOT_XML
    from emu.ppo import make_trainer
    path = elevation_course(ROBOT_XML, tmp_path/f"{direction}.xml", 10,
                            direction=direction)
    env = EmuMJXEnv(str(path), num_envs=2, seed=733,
                    observation_schema=OBS_SCHEMA_PREVIEW,
                    terrain_contact_mode='box_surface')
    trainer = make_trainer(env, seed=733, lr=.0003,
                           rollout=2, epochs=1, minibatches=1)
    cfg = {'probe_cmd_vx': .18, 'probe_cmd_vy': 0.,
           'probe_cmd_yaw': 0., 'steady_difficulty': 0.,
           'resilience_difficulty': 0., 'posture_difficulty': 0.,
           'regime_frac_steady': 1., 'regime_frac_transition': 0.}
    state = trainer['make_initial_state'](cfg)
    measured = trainer['evaluate_elevation_summary'](
        state.params, env.params_array(cfg), jnp.zeros(312), jnp.ones(312),
        jax.random.PRNGKey(1701), 4, direction)
    results = np.asarray(measured)
    assert results.shape == (6,)
    assert np.isfinite(results).all()
    assert results[0] == 2.
    assert results[2] == 0.  # Impossible to land and walk 40mm in 4 frames.
    assert np.all((results[1:] >= 0.) & (results[1:] <= 1.))


def test_public_elevation_exam_cli_is_read_only(tmp_path, monkeypatch, capsys):
    from emu.cli import main
    from emu import evaluation
    calls = []
    def fake_evaluate_runs(paths, **kwargs):
        calls.append((paths,kwargs))
        return {"checkpoint": {"elevation_up_10mm_crossing_rate": 0.0}}
    monkeypatch.setattr(evaluation,"evaluate_runs", fake_evaluate_runs)
    source=tmp_path/"checkpoint.pkl"
    assert main(['screen-elevation','--checkpoint',str(source),
                 '--course-root',str(tmp_path/'geometry'),
                 '--direction','up','--height-mm','10',
                 '--envs','8','--horizon','350','--device','cpu']) == 0
    assert calls[0][0] == [source]
    assert calls[0][1]["v5020_elevation_exam"]
    assert calls[0][1]["write_screen_files"] is False
    assert calls[0][1]["key_by_path"] is True
    assert not (tmp_path/'geometry').exists()
    assert 'elevation_up_10mm' in capsys.readouterr().out


@pytest.mark.parametrize("direction, sign", [("up", 1.), ("down", -1.)])
def test_true_elevation_ray_returns_are_signed_and_visible_in_actor_at_20cm(direction, sign):
    """Hardware-like returns, not privileged obstacle height as policy input.

    At 20cm ahead the 2cm insertion kernel cannot populate the OLD 15cm
    actor tap. The 20cm tap must see both elevated tops and descending
    edges without advancing the robot (and retain confidence=0 when blind).
    """
    from emu.forward_tof import first_hit_ranges
    from emu.directional import (
        fit_tripod_ground, TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY)
    from emu.forward_tof import FORWARD_MOUNTS_BODY_M, FORWARD_DIRECTIONS_BODY
    from emu.rolling_tof import (
        empty_map, insert_measurement, measured_height_and_x, readout,
        OUTPUT_SAMPLE_X_M, OUTPUT_SAMPLE_INDICES)
    from emu.contracts import sensor_observation_contract, PREVIEW_RANGE_FIELDS

    h=.020
    # The box is a 2m long physical landing, with transition edge at x=.16m.
    # Step-up: box begins at .16. Step-down: box ends at .16.
    center=np.array([[1.16 if direction=="up" else -.84, 0.,h/2]],
                    dtype=np.float32)
    rotation=np.eye(3,dtype=np.float32)[None]
    torso=np.array([[0.,0.,.2584+(h if direction=="down" else 0.)]],
                   dtype=np.float32)
    floor=np.zeros((1,3),dtype=np.float32)
    up=np.array([[0.,0.,1.]],dtype=np.float32)
    kwargs=dict(box_center=center,box_rotation=rotation,
                box_half_size=np.array([1.,.5,h/2],dtype=np.float32))
    tof,valid=first_hit_ranges(np,torso,rotation,floor,up,
                              mounts_body=FORWARD_MOUNTS_BODY_M,
                              beams_body=FORWARD_DIRECTIONS_BODY,**kwargs)
    tripod,tripod_valid=first_hit_ranges(np,torso,rotation,floor,up,
                              mounts_body=TOF_TRIPOD_MOUNTS_BODY_M,
                              beams_body=TOF_TRIPOD_BEAMS_BODY,**kwargs)
    estimator=fit_tripod_ground(np,tripod,np.array([[0.,0.,-1.]],
                                                  dtype=np.float32))
    assert valid.all() and tripod_valid.all() and estimator["valid"].all()
    x,y,measured,good,rx,ry,rear_valid=measured_height_and_x(
        np,tof,valid,tripod,estimator["ground_normal_body"],
        estimator["valid"],
        mounts_body=FORWARD_MOUNTS_BODY_M,
        beams_body=FORWARD_DIRECTIONS_BODY,
        tripod_mounts_body=TOF_TRIPOD_MOUNTS_BODY_M,
        tripod_beams_body=TOF_TRIPOD_BEAMS_BODY,
        gravity_body=np.array([[0.,0.,-1.]],dtype=np.float32),
        include_rear=True)
    assert good.all() and rear_valid.all()
    np.testing.assert_allclose(measured,np.full((1,2),sign*h),
                               atol=.003,rtol=0)
    heights,confidence=empty_map(np,1)
    heights,confidence=insert_measurement(
        np,heights,confidence,x,measured,good,
        sample_y_m=y,apex_x_m=rx,apex_y_m=ry)
    # New readings around 20cm cannot reach 15cm when only a 2cm
    # neighbourhood is filled. That was the old actor's blind spot.
    assert np.all(confidence[:,:,25]==0.)
    read_h,read_c=readout(np,heights,confidence)
    assert read_h.shape==(1,10) and read_c.shape==(1,10)
    np.testing.assert_allclose(read_h[:,[4,9]],sign*h,atol=.003,rtol=0)
    assert np.all(read_c[:,[4,9]]>.05)
    assert OUTPUT_SAMPLE_X_M[-1]==.20 and OUTPUT_SAMPLE_INDICES[-1]==30
    assert PREVIEW_RANGE_FIELDS[4].endswith("_20cm_m")
    assert sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW)[
        "rolling_actor_sample_x_m"][-1]==.20

    # Unknown forward return must stay unknown, not be invented as flat.
    z,unknown=empty_map(np,1)
    z,unknown=insert_measurement(np,z,unknown,x,measured,np.zeros_like(good))
    _,confidence_empty=readout(np,z,unknown)
    assert np.all(confidence_empty==0.)
