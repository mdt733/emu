"""v5016 real-trace adapter and conservative contact-pair gait gates."""
import numpy as np
import pytest

from emu.gait_exam import (
    contact_pair_mask, count_contact_strides, evaluate_trace_arrays)
from emu.directional_exam import exam_gravity,validate_request


def synthetic_walk(*, travel=True):
    n=130
    body=np.zeros((n,2))
    if travel:
        body[:,0]=np.arange(n)*.003
    foot=np.zeros((n,2,3))
    foot[:,:,2]=.018
    ct=np.ones((n,2),dtype=bool)
    for who,start,end,anchor in (
        (0,10,20,.07),(1,38,48,.07),(0,65,76,.14),(1,94,104,.14)):
        ct[start:end,who]=False
        foot[end:,who,0]=anchor
        fraction=np.arange(end-start)/(end-start)
        previous=anchor-.07
        foot[start:end,who,0]=previous+.07*fraction
        foot[start:end,who,2]=.018+.012*np.sin(np.pi*fraction)
    return body,foot,ct


def test_alternating_true_contact_touches_and_displacement_count_strides():
    b,f,c=synthetic_walk()
    assert count_contact_strides(b,f,c) >= 2


def test_tapping_while_stationary_does_not_qualify():
    b,f,c=synthetic_walk(travel=False)
    assert count_contact_strides(b,f,c)==0


def test_sliding_with_both_feet_never_airborne_cannot_qualify():
    b,f,c=synthetic_walk()
    c[:]=True
    assert count_contact_strides(b,f,c)==0


def test_only_one_leg_lifts_no_complete_alternating_pair():
    b,f,c=synthetic_walk()
    c[:,1]=True
    assert count_contact_strides(b,f,c)==0


def test_missing_or_degenerate_foot_packets_rejected():
    b,f,c=synthetic_walk()
    with pytest.raises(ValueError):
        count_contact_strides(b,f[:,:1],c)


def test_real_contact_pair_id_and_distance_gate_both_orders():
    geom1=np.array([[1,3,6,2],[4,9,1,7]],dtype=np.int32)
    geom2=np.array([[3,1,1,8],[1,1,4,1]],dtype=np.int32)
    dist=np.array([[0.,-0.001,.002,0.01],[0.,0.,.01,0.]],dtype=float)
    out=contact_pair_mask(np,geom1,geom2,dist,1,(3,4))
    assert out.shape==(2,2)
    assert out.tolist()==[[True,False],[False,True]]


def fake_trace(n=42,worlds=2):
    pos=np.zeros((n,worlds,2))
    pos[:,:,0]=np.arange(n)[:,None]*.002
    pos[12:,1,0]=20.0+np.arange(n-12)*.02  # teleport AFTER termination
    cmd=np.zeros((n,worlds,3))
    cmd[:,:,0]=.1
    heads=np.zeros((n,worlds))
    feet=np.zeros((n,worlds,2,3))
    feet[:,:,:,2]=.018
    contact=np.ones((n,worlds,2),dtype=bool)
    done=np.zeros((n,worlds))
    fell=np.zeros((n,worlds))
    done[11,1]=fell[11,1]=1
    return dict(position_xy=pos,heading_rad=heads,foot_xyz=feet,
                floor_contact=contact,command=cmd,done=done,fell=fell,
                last_position_xy=np.array([[n*.002,0],[999,0]]),
                last_heading_rad=np.zeros(worlds),
                last_foot_xyz=feet[-1],last_floor_contact=contact[-1])


def test_pre_reset_trace_never_scores_world_reset_teleport():
    result=evaluate_trace_arrays(
        fake_trace(),np.tile([0,0,-9.81],(2,1)),settle_frames=5)
    good,bad=result
    assert good['course_pass'] and not good['walking_pass']
    assert good['full_horizon_survival']
    assert bad['first_fall'] and not bad['course_pass']
    assert bad['first_done_step']==12
    assert bad['frames_recorded']==11
    assert 'uphill_progress_m' not in bad or abs(bad['signed_course_progress_m']) < 1


def test_exact_incline_geometries_and_direction_signs():
    g=exam_gravity(1,'uphill')
    assert g[0] < 0 and g[2] < 0
    assert exam_gravity(1,'downhill')[0] > 0
    assert exam_gravity(1,'cross_left')[1] < 0
    np.testing.assert_allclose(exam_gravity(0,'cross_right'),[0,0,-9.81],atol=1e-6)
    with pytest.raises(ValueError):
        exam_gravity(10,'uphill')


def test_output_directory_guards_are_read_only(tmp_path):
    run=tmp_path/'trained'
    run.mkdir()
    with pytest.raises(ValueError,match='separate'):
        validate_request(run_dir=run,output=run/'exam',worlds=8,horizon=350,
           settle_frames=25,angle_deg=0,direction='uphill',speed=.1,lateral=0,yaw_rate=0)
    out=tmp_path/'new'
    assert validate_request(run_dir=run,output=out,worlds=8,horizon=350,
           settle_frames=25,angle_deg=.5,direction='uphill',speed=.1,lateral=0,yaw_rate=0)[1]==out
    out.mkdir()
    (out/'keep.csv').write_text('untouched')
    with pytest.raises(FileExistsError):
        validate_request(run_dir=run,output=out,worlds=8,horizon=350,
           settle_frames=25,angle_deg=.5,direction='uphill',speed=.1,lateral=0,yaw_rate=0)
    assert (out/'keep.csv').read_text()=='untouched'


def test_real_mjx_evaluator_trace_shapes_cpu():
    """Small real stepping smoke; exercises contact.geom1/geom2/dist plumbing."""
    pytest.importorskip('mujoco')
    pytest.importorskip('optax')
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    from emu.environment import EmuMJXEnv
    from emu.paths import ROBOT_XML
    from emu.ppo import make_trainer
    from emu.contracts import OBS_DIM
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=2,seed=1701,
                  sensor_noise_scale=0.,attitude_profile='v5015')
    trainer=make_trainer(env,seed=1701,lr=.0002,rollout=2,
                         epochs=1,minibatches=1)
    assert trainer['obs_dim']==OBS_DIM
    state=trainer['make_initial_state']({'probe_cmd_vx':.1})
    params=state.params
    mean=jnp.zeros(OBS_DIM)
    std=jnp.ones(OBS_DIM)
    history=trainer['evaluate_trace'](
        params,env.params_array({'probe_cmd_vx':.1,'probe_cmd_vy':0.,'probe_cmd_yaw':0.}),
        mean,std,jax.random.PRNGKey(199),4)
    data=jax.device_get(history)
    assert data['position_xy'].shape==(4,2,2)
    assert data['floor_contact'].shape==(4,2,2)
    assert data['command'].shape==(4,2,3)
    assert data['last_position_xy'].shape==(2,2)
    assert np.isfinite(data['position_xy']).all()
    # Exercise the actual optional gyro-integrated heading controller JIT path.
    on=trainer['evaluate_trace'](
        params,env.params_array({'probe_cmd_vx':.1,'probe_cmd_vy':0.,
                                 'probe_cmd_yaw':.2}),
        mean,std,jax.random.PRNGKey(199),4,0.75,jnp.asarray([.1,0.,.2]))
    off=trainer['evaluate_trace'](
        params,env.params_array({'probe_cmd_vx':.1,'probe_cmd_vy':0.,
                                 'probe_cmd_y':.2}),
        mean,std,jax.random.PRNGKey(199),4,0.0,jnp.asarray([.1,0.,.2]))
    on=jax.device_get(on); off=jax.device_get(off)
    assert on['issued_yaw_rate'].shape==(4,2)
    assert on['estimated_heading_rad'].shape==(4,2)
    assert np.isfinite(on['issued_yaw_rate']).all()
    # Grade original requests: exact same operator command in both conditions.
    np.testing.assert_array_equal(on['command'],off['command'])
    # Matching commands are not enough if the reset bakes in different
    # initial yaw velocities. The physical starting states must also match.
    np.testing.assert_array_equal(on['position_xy'][0],off['position_xy'][0])
    np.testing.assert_allclose(on['root_qvel'][0],off['root_qvel'][0],
                               rtol=0,atol=1e-7)
    np.testing.assert_allclose(off['root_qvel'][0,:,5],.2,rtol=0,atol=.02)
    np.testing.assert_allclose(on['command'][:,:,2],.2,rtol=0,atol=1e-7)
    np.testing.assert_allclose(off['issued_yaw_rate'],
                               off['command'][:,:,2],rtol=1e-6,atol=1e-6)
    assert np.max(np.abs(on['issued_yaw_rate'])) <= 1.5+1e-6
    measured=evaluate_trace_arrays(on,np.tile([0,0,-9.81],(2,1)),
                                   settle_frames=0)
    assert len(measured)==2
    assert all('mean_abs_controller_correction_rad_s' in w
               for w in measured if w['frames_recorded']>2)
