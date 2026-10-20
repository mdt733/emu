"""Focused v5019 proof: two REAL rays, measured spatial memory, policy schema.

No long PPO runs. Cannot certify FOV, hardware or learned step crossing.
"""
import numpy as np
import pytest

from emu.forward_tof import first_hit_ranges, FORWARD_MOUNTS_BODY_M, FORWARD_DIRECTIONS_BODY
from emu.rolling_tof import (
    empty_map, advance_map, insert_measurement, readout, GRID_CELLS,
    measured_height_and_x)
from emu.contracts import (
    OBS_SCHEMA_SENSOR, OBS_SCHEMA_PREVIEW, OBS_DIM, SENSOR_OBS_DIM, PREVIEW_OBS_DIM,
    SENSOR_OBS_FIELDS, PREVIEW_OBS_FIELDS, sensor_observation_contract)
from emu.training_config import TrainingConfig, training_sensor_contract


def rays(x, *, step_height=.0, yaw=0.):
    n=len(x)
    torso=np.column_stack((np.asarray(x),np.zeros(n),np.full(n,.2584)))
    a=np.deg2rad(yaw)
    R=np.tile(np.array([[np.cos(a),-np.sin(a),0.],
                         [np.sin(a),np.cos(a),0.],[0.,0.,1.]]),(n,1,1))
    floor=np.zeros((n,3))
    normal=np.tile(np.asarray([0.,0.,1.]),(n,1))
    extras={}
    if step_height>0.:
        extras=dict(
            box_center=np.tile(np.asarray([.20,0.,step_height/2]),(n,1)),
            box_rotation=np.tile(np.eye(3),(n,1,1)),
            box_half_size=np.asarray([.04,.20,step_height/2]))
    return first_hit_ranges(np,torso,R,floor,normal,**extras)


def test_two_single_point_beams_see_physical_box_and_no_far_oracle():
    assert len(FORWARD_MOUNTS_BODY_M)==len(FORWARD_DIRECTIONS_BODY)==2
    flat,valid=rays([-.2,.0])
    stepped,seen=rays([-.2,.0],step_height=.05)
    assert flat.shape==valid.shape==(2,2)
    assert valid.all() and seen.all()
    assert np.allclose(stepped[0],flat[0],atol=1e-6)
    assert np.any(flat[1]-stepped[1]>.01)


def test_flow_advects_far_right_measurement_into_15cm_history():
    h,c=empty_map(np,1)
    h,c=insert_measurement(np,h,c,np.array([[.20,.20]]),
                           np.array([[.05,.02]]),np.array([[True,True]]))
    initial,ic=readout(np,h,c)
    assert np.all(ic==0.)  # 20cm measured; [0,5,10,15] still UNKNOWN
    for _ in range(25):
        h,c=advance_map(np,h,c,np.asarray([.10]),np.asarray([0.]),
                        np.asarray([0.]),np.asarray([1.]))
    values,conf=readout(np,h,c)
    assert conf[0,3]>.03 and conf[0,7]>.03
    assert np.isclose(values[0,3],.05,atol=.01)
    assert np.isclose(values[0,7],.02,atol=.01)
    assert np.all(conf[0,[0,1,2,4,5,6]]<.15)  # no invented far-reaching map


def test_missing_motion_degrades_confidence_not_made_up_forward_progress():
    h,c=empty_map(np,1)
    h,c=insert_measurement(np,h,c,np.array([[.15,.15]]),
                           np.array([[.05,.05]]),np.array([[True,True]]))
    initial,ci=readout(np,h,c)
    for _ in range(5):
        h,c=advance_map(np,h,c,np.asarray([1.]),np.asarray([0.]),
                        np.asarray([0.]),np.asarray([0.]))
    values,after=readout(np,h,c)
    assert after[0,3]<ci[0,3]
    assert values[0,3]>.01   # no shift on INVALID optical flow
    assert after[0,2]==0.


def test_triangulate_sensor_height_only_from_calibrated_raw_measurements():
    from emu.directional import TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY
    flat,v=rays([0.])
    raised,sv=rays([0.],step_height=.05)
    tripod=np.full((1,3),.2334)
    kwargs=dict(mounts_body=FORWARD_MOUNTS_BODY_M,
                beams_body=FORWARD_DIRECTIONS_BODY,
                tripod_mounts_body=TOF_TRIPOD_MOUNTS_BODY_M,
                tripod_beams_body=TOF_TRIPOD_BEAMS_BODY)
    x,z,valid=measured_height_and_x(np,flat,v,tripod,
                                    np.asarray([[0.,0.,1.]]),np.asarray([True]),**kwargs)
    assert valid.all() and np.max(np.abs(z))<1e-4
    x,h,valid=measured_height_and_x(np,raised,sv,tripod,
                                    np.asarray([[0.,0.,1.]]),np.asarray([True]),**kwargs)
    assert valid.any() and (h>.02).any()
    assert (x>0.).all() and (x<=.2).all()
    _,zero,none=measured_height_and_x(np,raised,sv,tripod,
                                     np.asarray([[0.,0.,1.]]),np.asarray([False]),**kwargs)
    assert not none.any() and not zero.any()


def test_old_220_232_shapes_and_manifest_are_unchanged_new_296_distinct():
    old=sensor_observation_contract()
    assert old["schema"]==OBS_SCHEMA_SENSOR
    assert old["base_fields"]==list(SENSOR_OBS_FIELDS)
    assert "rolling_model_revision" not in old
    assert (OBS_DIM,SENSOR_OBS_DIM,PREVIEW_OBS_DIM)==(220,232,296)
    assert len(PREVIEW_OBS_FIELDS)==74
    assert len([k for k in PREVIEW_OBS_FIELDS if k.startswith("terrain_height_")])==8
    assert len([k for k in PREVIEW_OBS_FIELDS if k.startswith("terrain_confidence_")])==8
    visible=sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW)
    hidden=sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW,preview_masked=True)
    assert visible["sha256"]!=hidden["sha256"]!=old["sha256"]
    assert visible["forward_masked"] is False
    assert len(visible["forward_beams_body"])==2
    assert visible["rolling_cells_per_lane"]==GRID_CELLS
    assert training_sensor_contract(
        TrainingConfig(observation_schema=OBS_SCHEMA_PREVIEW))==visible
    assert training_sensor_contract(
        TrainingConfig(observation_schema=OBS_SCHEMA_PREVIEW,
                       forward_preview_masked=True))==hidden


def test_two_world_mjx_roll_map_shape_checkpoint_pytree():
    jax=pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    import jax.numpy as jnp
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from emu.terrain import adaptive_height_course
    from emu.environment import EmuMJXEnv, RollingSensorState, SensorState
    from emu.contracts import ROBOT_XML
    with TemporaryDirectory() as tmp:
        xml=adaptive_height_course(ROBOT_XML,Path(tmp)/"rolling.xml")
        env=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                      observation_schema=OBS_SCHEMA_PREVIEW,
                      terrain_contact_mode="box_surface",terrain_adaptive=True,
                      tof_dropout_prob=0.)
        params=env.params_array({"frontier_terrain_height_m":.05})
        state,obs=env.reset(params)
        assert isinstance(state.sensor_state,RollingSensorState)
        assert obs.shape==(2,296)
        assert state.sensor_state.terrain_heights.shape==(2,2,21)
        nxt=env.step(state,jnp.zeros((2,6),dtype=jnp.float32),params)
        assert nxt.obs.shape==(2,296)
        assert np.isfinite(np.asarray(jax.device_get(nxt.obs))).all()
        assert nxt.state.sensor_state.terrain_heights.shape==(2,2,21)
        legacy=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                         observation_schema=OBS_SCHEMA_SENSOR)
        old,_=legacy.reset(params)
        assert isinstance(old.sensor_state,SensorState)
        masked=EmuMJXEnv(str(xml),num_envs=2,seed=733,
                         observation_schema=OBS_SCHEMA_PREVIEW,
                         forward_preview_masked=True)
        _,masked_obs=masked.reset(params)
        assert masked_obs.shape==(2,296)
        assert np.all(np.asarray(jax.device_get(masked_obs))[:,-16:]==0.)
