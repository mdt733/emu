"""V-shaped terrain sensing: measured physics, sparse rear anchor and policy width.

No GPU campaign, browser, servo or learned stepping claims are made here.
"""
import numpy as np
import pytest

from emu.forward_tof import (
    first_hit_ranges, FORWARD_MOUNTS_BODY_M, FORWARD_DIRECTIONS_BODY)
from emu.rolling_tof import (
    GRID_CELLS, GRID_START_M, GRID_LOOKAHEAD_M, OUTPUT_SAMPLE_INDICES,
    SENSOR_MAP_REVISION, empty_map, advance_map, insert_measurement,
    readout, v_track_y, measured_height_and_x)
from emu.contracts import (
    OBS_SCHEMA_SENSOR, OBS_SCHEMA_PREVIEW, OBS_DIM, SENSOR_OBS_DIM,
    PREVIEW_OBS_DIM, SENSOR_OBS_FIELDS, PREVIEW_OBS_FIELDS,
    SENSOR_SETTINGS_DEFAULTS, sensor_observation_contract)
from emu.training_config import (
    TrainingConfig, training_sensor_contract, SENSOR_NOISE_SCALE)


def _physical_ranges(x, *, height=.0, pitch_deg=0., yaw_deg=0., torso_z=.2584):
    n=len(x)
    pos=np.column_stack((np.asarray(x),np.zeros(n),np.full(n,torso_z)))
    pitch,yaw=np.deg2rad([pitch_deg,yaw_deg])
    cy,sy,cp,sp=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch)
    rz=np.asarray([[cy,-sy,0],[sy,cy,0],[0,0,1.]])
    ry=np.asarray([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rot=np.tile(rz@ry,(n,1,1))
    fpos=np.zeros((n,3))
    fnormal=np.tile(np.asarray((0.,0.,1.)),(n,1))
    extra={}
    if height:
        extra=dict(box_center=np.tile([.20,0,height/2],(n,1)),
                   box_rotation=np.tile(np.eye(3),(n,1,1)),
                   box_half_size=np.asarray((.04,.20,height/2)))
    return (first_hit_ranges(
        np,pos,rot,fpos,fnormal,**extra),pos,rot,fpos,fnormal)


def _ground_hits(forward,valid,pos,rot,fpos,fnormal):
    from emu.directional import (
        simulate_tripod_ranges, fit_tripod_ground,
        TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY)
    gravity=np.einsum('nij,j->ni',np.transpose(rot,(0,2,1)),np.array([0.,0.,-1.]))
    tripod=simulate_tripod_ranges(
        np,pos,rot,fpos,fnormal)
    # The three downward rays define a measured local reference.
    plane=fit_tripod_ground(np,tripod['ranges_m'],gravity)
    return measured_height_and_x(
        np,forward,valid,tripod['ranges_m'],plane['ground_normal_body'],
        plane['valid'],mounts_body=FORWARD_MOUNTS_BODY_M,
        beams_body=FORWARD_DIRECTIONS_BODY,
        tripod_mounts_body=TOF_TRIPOD_MOUNTS_BODY_M,
        tripod_beams_body=TOF_TRIPOD_BEAMS_BODY,
        gravity_body=gravity,include_rear=True)


def test_two_physical_beams_measure_real_step_not_oracle():
    ((flat,vf),pos,rot,fo,fn)=_physical_ranges([-.20,0.])
    ((raised,vr),_,_,_,_)=_physical_ranges([-.20,0.],height=.05)
    assert flat.shape==vf.shape==(2,2)
    assert vf.all() and vr.all()
    assert np.allclose(flat[0],raised[0],atol=1e-5)
    assert (flat[1]-raised[1]>.01).any()


def test_gravity_projected_hit_distance_changes_with_pitch_and_chassis_height():
    values=[]
    for pitch in (-8.,0.,8.):
        (r,v),p,rot,fo,fn=_physical_ranges([0.],pitch_deg=pitch)
        x,y,h,good,rx,ry,rear_ok=_ground_hits(r,v,p,rot,fo,fn)
        assert good.any() and rear_ok.any()
        assert np.isfinite(x).all() and np.isfinite(rx).all()
        values.append(float(x[0,0]))
    assert values[0] < values[1] < values[2], values
    (r,v),p,rot,fo,fn=_physical_ranges([0.],torso_z=.30)
    x,_,_,_,rx,_,_=_ground_hits(r,v,p,rot,fo,fn)
    assert float(x[0,0])>values[1]
    assert -0.12 < rx[0] < 0.04


def test_v_paths_meet_at_rear_tof_and_diverge():
    apex_x=np.asarray([-.043],dtype=np.float32)[:,None,None]
    apex_y=np.asarray([.012],dtype=np.float32)[:,None,None]
    assert GRID_CELLS==51 and np.isclose(GRID_START_M,-.10)
    assert np.isclose(GRID_LOOKAHEAD_M,.40)
    assert tuple(OUTPUT_SAMPLE_INDICES)==(5,10,15,20,25)
    rear=v_track_y(np,np.asarray([[[-.043]]],dtype=np.float32),apex_x,apex_y)
    assert rear.shape==(1,2,1)
    assert np.allclose(rear[0,:,0],.012)
    distant=v_track_y(np,np.asarray([[[.30]]],dtype=np.float32),apex_x,apex_y)
    assert distant[0,0,0]<.012 and distant[0,1,0]>.012


def test_rear_tof_seeds_both_arms_at_true_junction():
    h,c=empty_map(np,1)
    rear_x=np.array([[-.04,-.04]])
    rear_y=np.array([[.006,.006]])
    h,c=insert_measurement(
        np,h,c,rear_x,np.zeros_like(rear_x),np.ones((1,2),bool),
        sample_y_m=rear_y,apex_x_m=np.array([-.04]),apex_y_m=np.array([.006]))
    values,confidence=readout(np,h,c)
    assert confidence[0,0]>.1 and confidence[0,5]>.1  # -5cm sample near -4cm
    assert not confidence[0,4] and not confidence[0,9]   # +15cm stays unknown
    assert np.all(values==0.)  # this is a measured local zero, not a flat map


def test_new_scan_advects_toward_15cm_without_fabricated_flat():
    h,c=empty_map(np,1)
    ys=np.asarray([[-.05,.05]])
    h,c=insert_measurement(
        np,h,c,np.array([[.20,.20]]),np.array([[.05,.02]]),
        np.array([[True,True]]),sample_y_m=ys)
    _,before=readout(np,h,c)
    assert np.all(before==0.)  # currently stored only at +20cm
    for _ in range(25):
        h,c=advance_map(np,h,c,np.array([.10]),np.array([0.]),
                        np.array([0.]),np.array([1.]))
    values,confidence=readout(np,h,c)
    assert confidence[0,4]>.005 and confidence[0,9]>.005
    assert np.isclose(values[0,4],.05,atol=.025)
    assert np.isclose(values[0,9],.02,atol=.025)
    assert confidence[0,2]<.10 and confidence[0,7]<.10


def test_reverse_and_turns_do_not_invent_values():
    h,c=empty_map(np,1)
    x=np.asarray([[.10,.10]])
    apex=-.04
    ys=v_track_y(np,x[:,:,None],np.asarray([[[apex]]]))[:,:,0]
    h,c=insert_measurement(np,h,c,x,np.array([[.03,.07]]),
                           np.array([[True,True]]),sample_y_m=ys)
    # Reverse motion should move a fixed patch towards *larger* X.
    h1,c1=advance_map(np,h,c,np.array([-.10]),np.array([0.]),
                      np.array([0.]),np.array([1.]),dt_s=.20)
    grid=np.arange(GRID_CELLS)*.01-.10
    assert grid[np.argmax(c1[0,0])]>.10
    # Pure yaw moves the V arms differently, while keeping unknown cells
    # unknown even under large heading changes.
    h2,c2=advance_map(np,h,c,np.array([0.]),np.array([0.]),
                      np.array([1.]),np.array([1.]),dt_s=.10)
    assert np.isfinite(h2).all() and np.isfinite(c2).all()
    assert c2.min()>=0. and c2.max()<=1.
    assert np.count_nonzero(c2)<2*GRID_CELLS


def test_flow_dropout_does_not_falsely_shift():
    h,c=empty_map(np,1)
    ys=v_track_y(np,np.array([[[.15]],[[.15]]]).reshape((1,2,1)),
                 np.array([[[-.04]]]))[:,:,0]
    h,c=insert_measurement(np,h,c,np.array([[.15,.15]]),
                           np.array([[.05,.02]]),np.array([[True,True]]),
                           sample_y_m=ys)
    old_x=np.argmax(c[0,0])
    h,c=advance_map(np,h,c,np.array([5.]),np.array([0.]),
                    np.array([0.]),np.array([0.]))
    assert np.argmax(c[0,0])==old_x
    assert c.max()<1. and np.isfinite(h).all()


def test_historical_232_manifest_unchanged_new_312_contract_separate():
    assert (OBS_DIM,SENSOR_OBS_DIM,PREVIEW_OBS_DIM)==(220,232,312)
    assert len(PREVIEW_OBS_FIELDS)==78
    assert sum(x.startswith("terrain_height_") for x in PREVIEW_OBS_FIELDS)==10
    assert sum(x.startswith("terrain_confidence_") for x in PREVIEW_OBS_FIELDS)==10
    historical=sensor_observation_contract()
    assert historical['schema']==OBS_SCHEMA_SENSOR
    assert historical['base_fields']==list(SENSOR_OBS_FIELDS)
    assert 'rolling_model_revision' not in historical
    visible=sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW)
    masked=sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW,preview_masked=True)
    assert visible['sha256']!=masked['sha256']!=historical['sha256']
    assert visible['rolling_cells_per_lane']==GRID_CELLS
    assert visible['rolling_model_revision']==SENSOR_MAP_REVISION
    settings={**SENSOR_SETTINGS_DEFAULTS,'sensor_noise_scale':SENSOR_NOISE_SCALE}
    assert training_sensor_contract(TrainingConfig(
        observation_schema=OBS_SCHEMA_PREVIEW))==sensor_observation_contract(
        settings,schema=OBS_SCHEMA_PREVIEW)
    assert training_sensor_contract(TrainingConfig(
        observation_schema=OBS_SCHEMA_PREVIEW,forward_preview_masked=True)
        )==sensor_observation_contract(settings,schema=OBS_SCHEMA_PREVIEW,
                                      preview_masked=True)


def test_mjx_two_world_rolling_and_old_checkpoint_state_separated():
    jax=pytest.importorskip('jax')
    pytest.importorskip('mujoco')
    import jax.numpy as jnp
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from emu.terrain import adaptive_height_course
    from emu.environment import EmuMJXEnv, RollingSensorState, SensorState, SensorSample
    from emu.contracts import ROBOT_XML
    with TemporaryDirectory() as tmp:
        xml=adaptive_height_course(ROBOT_XML,Path(tmp)/'V_course.xml')
        env=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                      observation_schema=OBS_SCHEMA_PREVIEW,
                      terrain_contact_mode='box_surface',terrain_adaptive=True,
                      tof_dropout_prob=0.)
        task={'frontier_terrain_height_m':.05}
        state,obs=env.reset(task) # reset expects DICT, not packed JAX array
        assert isinstance(state.sensor_state,RollingSensorState)
        assert obs.shape==(2,312)
        assert state.sensor_state.terrain_heights.shape==(2,2,51)
        assert state.sensor_state.apex_x_m.shape==(2,)
        stepped=env.step(state,jnp.zeros((2,6),dtype=jnp.float32),
                         env.params_array(task))
        assert stepped.obs.shape==(2,312)
        assert np.isfinite(np.asarray(jax.device_get(stepped.obs))).all()
        assert stepped.state.sensor_state.terrain_heights.shape==(2,2,51)

        legacy=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                         observation_schema=OBS_SCHEMA_SENSOR)
        old,_=legacy.reset(task)
        assert isinstance(old.sensor_state,SensorState)
        assert isinstance(old.sensor_state.samples,SensorSample)
        assert len(old.sensor_state.samples)==7
        masked=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                         observation_schema=OBS_SCHEMA_PREVIEW,
                         forward_preview_masked=True)
        _,obs2=masked.reset(task)
        assert obs2.shape==(2,312)
        assert np.all(np.asarray(jax.device_get(obs2))[:,-20:]==0.)
