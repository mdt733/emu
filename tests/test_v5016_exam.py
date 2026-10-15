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


    # v5016: explicit mid-episode turn/stop schedule, noisy/delayed gyro.
    # The *environment's* post-step command must match what PPO was told:
    # changing only task/obs is insufficient because env.step advances its
    # own command ramp after computing physics.
    schedule=jnp.asarray([[.10,0.,0.],
                          [.10,0.,.25],
                          [.10,0.,-.25],
                          [.10,0.,0.]],dtype=jnp.float32)
    noisy=trainer['evaluate_trace'](
        params,env.params_array({'probe_cmd_vx':.1,'probe_cmd_yaw':0.}),
        mean,std,jax.random.PRNGKey(199),4,.75,None,schedule,.015,.02,2)
    noisy=jax.device_get(noisy)
    np.testing.assert_allclose(noisy['command'][:,0,:],np.asarray(schedule),atol=1e-7)
    assert noisy['gyro_sample_rad_s'].shape==(4,2)
    assert noisy['gyro_delayed_rad_s'].shape==(4,2)
    assert np.isfinite(noisy['issued_yaw_rate']).all()
    # Ignore reset worlds; their new task is intentionally unrelated to
    # the old episode's request and may have a different birth command.
    valid=~noisy['done'].astype(bool)
    np.testing.assert_allclose(
        noisy['next_task_yaw_rate'][valid],
        noisy['issued_yaw_rate'][valid],rtol=0,atol=2e-6)
    # The schedules intentionally begin with different yaw requests (zero
    # versus .2 rad/s), but otherwise have identical reset physical noise.
    np.testing.assert_allclose(noisy['root_qvel'][0,:,:5],
                               off['root_qvel'][0,:,:5],rtol=0,atol=1e-6)
    np.testing.assert_allclose(noisy['root_qvel'][0,:,5],
                               off['root_qvel'][0,:,5]-.2,rtol=0,atol=2e-6)


# --- Batch campaign planning and shareable results ---
import json
import zipfile

import numpy as np
import pytest

from emu.command_exams import SCENARIOS, make_schedule
from emu import v5016_batch


def test_turn_stop_schedule_has_one_true_turn_and_a_long_heading_hold():
    n=160
    left=make_schedule('turn_stop_left',n,speed=.1,turn_rate=.25)
    right=make_schedule('turn_stop_right',n,speed=.1,turn_rate=.25)
    assert left.shape==(n,3)
    assert np.all(left[:n//4,2]==0)
    assert np.all(left[n//4:2*n//3,2]==.25)
    assert np.all(left[2*n//3:,2]==0)
    np.testing.assert_allclose(right[:,2],-left[:,2])
    np.testing.assert_allclose(right[:,0],left[:,0])
    assert float(np.sum(left[:,2])*.02) > .25


def test_lateral_reverse_stop_resume_are_distinct_operator_requests():
    left=make_schedule('sidestep_left',160,lateral=.07)
    right=make_schedule('sidestep_right',160,lateral=.07)
    back=make_schedule('reverse',160)
    pause=make_schedule('stop_resume',160)
    assert np.all(left[50:70,0]==0)
    assert np.all(left[50:70,1]==.07)
    np.testing.assert_allclose(right[:,1],-left[:,1])
    assert np.all(back[50:70,0]<0)
    assert np.all(pause[50:70,0]==0)
    assert np.all(pause[:40,0]>.0)
    assert np.all(pause[120:,0]>.0)


@pytest.mark.parametrize('bad', ['unknown',None])
def test_bad_schedules_fail_closed(bad):
    with pytest.raises(ValueError):
        make_schedule(bad,160)


def test_invalid_scenario_rates_and_nonfinite_commands_fail_closed():
    with pytest.raises(ValueError):
        make_schedule('steady',20)
    with pytest.raises(ValueError):
        make_schedule('turn_stop_left',160,turn_rate=0)
    with pytest.raises(ValueError):
        make_schedule('steady',160,speed=float('nan'))
    assert 'turn_stop_left' in SCENARIOS


def test_batch_preview_is_small_staged_and_writes_nothing(tmp_path,capsys):
    source=tmp_path/'old'
    dest=tmp_path/'exam'
    assert len(v5016_batch.plan('smoke',source,dest))==2
    assert len(v5016_batch.plan('pilot',source,dest))==20
    sensor=v5016_batch.plan('sensor',source,dest)
    assert len(sensor)==4
    assert all(j['gyro_bias']==.015 and j['gyro_delay_frames']==2 for j in sensor)
    assert len(v5016_batch.plan('stretch',source,dest))==4
    jobs=v5016_batch.plan('smoke',source,dest)
    v5016_batch.execute(jobs,'a'*40)
    assert 'PLAN ONLY' in capsys.readouterr().out
    assert not dest.exists()


def test_batch_never_writes_inside_training_or_reuses_bad_evidence(tmp_path):
    run=tmp_path/'old'
    out=tmp_path/'exam'
    with pytest.raises(ValueError,match='separate'):
        v5016_batch.plan('pilot',run,run/'temp')
    j=v5016_batch.plan('smoke',run,out)[0]
    report=out/'smoke'/'baseline'/'flat'/'heading_comparison.json'
    assert not v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        'checkpoint_sha256':'b'*64,
        'examined_at_revision':'a'*40,
        'protocol':dict(angle_deg=0.,direction='uphill',scenario='steady',
                        worlds=2,horizon=40,seed=1701,gyro_bias_rad_s=0.,
                        gyro_noise_std_rad_s=0.,gyro_delay_frames=0),
        'heading_kp':.75,
    }))
    with pytest.raises(ValueError,match='incomplete'):
        v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    for mode in ('on','off'):
        path=report.parent/mode
        path.mkdir()
        (path/'directional_exam.json').write_text('{}')
    assert v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    with pytest.raises(ValueError,match='stale'):
        v5016_batch.verify_existing(report,j,'c'*40,'b'*64)


def test_results_bundle_contains_raw_paired_evidence_not_checkpoints(tmp_path):
    source=tmp_path/'fourway'
    dest=tmp_path/'out'
    jobs=v5016_batch.plan('smoke',source,dest)
    j=jobs[0]
    folder=__import__('pathlib').Path(j['output'])
    folder.mkdir(parents=True)
    comparison={
        'checkpoint_sha256':'b'*64,
        'protocol':{'scenario':'steady'},
        'off':{'survived_full':.5,'course_pass':.0,'heading_mae_deg':35.},
        'on':{'survived_full':1.,'course_pass':.5,'heading_mae_deg':10.},
        'on_minus_off':{'course_pass':.5},
    }
    (folder/'heading_comparison.json').write_text(json.dumps(comparison))
    for mode in ('on','off'):
        p=folder/mode
        p.mkdir()
        (p/'directional_exam.json').write_text(json.dumps({'worlds':[]}))
    (folder/'secret.pkl').write_bytes(b'unsharable trained checkpoint')
    profile=folder.parents[1]
    bundle,count=v5016_batch.summarize(jobs,profile,'a'*40)
    assert count==1
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert 'results.md' in names
        assert 'baseline/flat/off/directional_exam.json' in names
        assert not any(name.endswith('.pkl') for name in names)
