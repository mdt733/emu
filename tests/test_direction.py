"""Five directional-oracle contracts, including real downhill cheating controls."""
import math
import numpy as np
import pytest
from emu.directional import HeadingState, heading_hold_step, wrap_radians, examine_course
from emu.directional import estimate_ground_plane
from emu.directional import evaluate_trace_arrays

def ideal(commands, initial_yaw=0.0, dt=0.02):
    """Synthetic perfect command following; not a trained policy."""
    c = np.asarray(commands, dtype=float)
    p = np.zeros((len(c) + 1, 2))
    h = np.full(len(c) + 1, initial_yaw)
    for i, (vx, vy, yaw) in enumerate(c):
        p[i+1] = p[i] + dt * np.array([
            vx * np.cos(h[i]) - vy * np.sin(h[i]),
            vx * np.sin(h[i]) + vy * np.cos(h[i]),
        ])
        h[i+1] = h[i] + yaw * dt
    return p, h
def hits_for_world_plane(slope_degrees=0, chassis_pitch_degrees=0):
    """Intersect rigid body-fixed beams with z=tan(slope)*world_x."""
    a=math.radians(chassis_pitch_degrees)
    rot=np.array([[math.cos(a),0,math.sin(a)],
                  [0,1,0],
                  [-math.sin(a),0,math.cos(a)]])
    world_origins=ORIGINS @ rot.T + np.array([0,0,.20])
    world_beams=BEAMS @ rot.T
    t=math.tan(math.radians(slope_degrees))
    ranges=(t*world_origins[:,0]-world_origins[:,2])/(world_beams[:,2]-t*world_beams[:,0])
    gravity_body=rot.T@np.array([0,0,-1.0])
    return ranges,gravity_body
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
ORIGINS = np.array([[.035,-.05,0],[.035,.05,0],[-.04,0,0]], dtype=float)
BEAMS = np.tile([0,0,-1.0],(3,1))

def test_one_degree_directed_climb_is_credited_with_equivalent_elevation():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos, heading = ideal(cmd)
    result = examine_course(pos, heading, cmd, uphill_world_xy=(1, 0),
                            slope_deg=1.0, completed_strides=7)
    assert result['course_pass'] and result['walking_pass']
    assert result['signed_course_progress_m'] == pytest.approx(.7)
    assert result['uphill_progress_m'] == pytest.approx(.7)
    assert result['equivalent_elevation_m'] == pytest.approx(.7 * math.sin(math.radians(1)))
    assert result['heading_mae_deg'] == pytest.approx(0)

def test_turn_downhill_cheat_fails_even_if_body_forward_and_final_yaw_rate_are_good():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos = np.zeros((351, 2))
    heading = np.zeros(351)
    for i in range(350):
        heading[i+1] = min(math.pi, (i+1)*math.pi/75)
        pos[i+1] = pos[i] + .02*.1*np.array([math.cos(heading[i]), math.sin(heading[i])])
    assert abs(heading[-1]-math.pi) < 1e-12
    result = examine_course(pos, heading, cmd, uphill_world_xy=(1, 0),
                            slope_deg=1.0, completed_strides=10)
    assert not result['course_pass'] and not result['walking_pass']
    assert result['uphill_progress_m'] < 0
    assert result['heading_mae_deg'] > 60

def test_heading_wrap_saturation_and_invalid_sensor_reacquisition():
    assert abs(wrap_radians(math.radians(179)-math.radians(-179))) == pytest.approx(math.radians(2))
    state=HeadingState(target_rad=math.pi-.01, has_reference=True)
    state, rate, error = heading_hold_step(state, -math.pi+.01, 0, kp=100, max_rate=.5)
    assert abs(error) < .03 and abs(rate) <= .5
    lost, safe_rate, _ = heading_hold_step(state,float('nan'),.1,measurement_valid=False)
    assert not lost.has_reference and safe_rate == pytest.approx(.1)
    restored, rate, error = heading_hold_step(lost, 2.0, 0)
    assert restored.has_reference and rate == error == 0
    with pytest.raises(ValueError):
        heading_hold_step(restored, 0, 0, dt=0)

def test_tilted_robot_on_flat_floor_is_not_misread_as_inclined_ground():
    ranges,gravity=hits_for_world_plane(0,chassis_pitch_degrees=10)
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid
    assert r.slope_deg == pytest.approx(0,abs=1e-7)

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

def test_physical_trace_duty_work_and_terminal_truncation():
    """No post-reset torque may contaminate a terminal world's evidence."""
    from emu.directional import examine_actuator_trace
    torque = np.zeros((8, 2, 6))
    omega = np.zeros_like(torque)
    torque[1:5, 0, 0] = 1.0
    omega[1:5, 0, 0] = 2.0
    torque[5:, 0, :] = 999.0  # teleport/reset contaminant
    data = dict(
        servo_torque_nm=torque,
        servo_joint_rate_rad_s=omega,
        servo_peak_limit_nm=np.full((2, 6), 1.913),
        servo_shaft_abs_power_w=np.zeros((8, 2)),
        servo_shaft_drive_power_w=np.zeros((8, 2)),
        servo_shaft_brake_power_w=np.zeros((8, 2)),
        spring_potential_energy_j=np.zeros((8, 2)),
        shin_spring_energy_j=np.zeros((8, 2)),
    )
    result = examine_actuator_trace(data, 0, start=0, stop=4, dt=.02, terminal=True)
    assert result['actuator_sample_frames'] == 4
    assert result['servo_peak_abs_torque_nm'][0] == pytest.approx(1.0)
    assert result['servo_over_continuous_seconds'][0] == pytest.approx(.08)
    assert result['servo_longest_continuous_overload_s'][0] == pytest.approx(.08)
    assert result['servo_p95_abs_joint_speed_rad_s'][0] == pytest.approx(2.)
    assert result['servo_near_peak_fraction'][0] == pytest.approx(0.)
    assert result['servo_positive_shaft_work_j'][0] == pytest.approx(.16)
    assert result['servo_thermal_exposure_proxy_s'][0] == pytest.approx(.08 / .638**2)
    assert result['servo_all_sample_over_continuous_fraction'] == pytest.approx(1 / 6)


def test_physical_trace_stall_has_heat_exposure_but_no_shaft_work():
    from emu.directional import examine_actuator_trace
    torque = np.ones((4, 1, 6))
    data = dict(servo_torque_nm=torque,
                servo_joint_rate_rad_s=np.zeros_like(torque),
                servo_peak_limit_nm=np.full((1,6),1.913),
                servo_shaft_abs_power_w=np.zeros((4,1)),
                servo_shaft_drive_power_w=np.zeros((4,1)),
                servo_shaft_brake_power_w=np.zeros((4,1)),
                spring_potential_energy_j=np.zeros((4,1)),
                shin_spring_energy_j=np.zeros((4,1)))
    result=examine_actuator_trace(data,0,start=0,stop=4,dt=.02,terminal=False)
    assert result['servo_measured_interval_abs_shaft_work_j'] == 0
    assert all(x > 0 for x in result['servo_thermal_exposure_proxy_s'])


def test_uninterrupted_physical_trace_includes_final_state():
    """An unbroken horizon has N scored intervals, not N-1."""
    from emu.directional import examine_actuator_trace
    torque=np.ones((5,1,6))*0.1
    speed=np.ones_like(torque)*2.
    trace=dict(
        servo_torque_nm=torque,
        servo_joint_rate_rad_s=speed,
        last_servo_torque_nm=np.ones((1,6))*.2,
        last_servo_joint_rate_rad_s=np.ones((1,6))*2.,
        servo_peak_limit_nm=np.full((1,6),1.913),
        servo_shaft_abs_power_w=np.zeros((5,1)),
        servo_shaft_drive_power_w=np.zeros((5,1)),
        servo_shaft_brake_power_w=np.zeros((5,1)),
        spring_potential_energy_j=np.zeros((5,1)),
        shin_spring_energy_j=np.zeros((5,1)),
    )
    result=examine_actuator_trace(trace,0,start=1,stop=5,dt=.02,terminal=False)
    assert result['actuator_sample_frames']==4
    assert result['servo_peak_abs_torque_nm'][0]==pytest.approx(.2)
    assert result['servo_positive_shaft_work_j'][0]==pytest.approx(
        (3*.1*2 + .2*2)*.02)


def test_report_energy_respects_settle_window_and_progress_direction():
    """No free efficiency credit for reverse drift or warm-up work."""
    n=10
    trace=dict(
        position_xy=np.zeros((n,1,2)),
        heading_rad=np.zeros((n,1)),
        foot_xyz=np.zeros((n,1,2,3)),
        floor_contact=np.ones((n,1,2),dtype=bool),
        command=np.zeros((n,1,3)),
        done=np.zeros((n,1)),
        fell=np.zeros((n,1)),
        last_position_xy=np.array([[.030,0.]]),
        last_heading_rad=np.zeros(1),
        last_foot_xyz=np.zeros((1,2,3)),
        last_floor_contact=np.ones((1,2),dtype=bool),
    )
    trace['position_xy'][:,:,0]=np.arange(n)[:,None]*.003
    trace['foot_xyz'][:,:,:,2]=.018
    trace['last_foot_xyz'][:,:,2]=.018
    trace['command'][:,:,0]=.1
    trace['servo_torque_nm']=np.ones((10,1,6))*.1
    trace['servo_joint_rate_rad_s']=np.ones((10,1,6))*.2
    trace['last_servo_torque_nm']=np.ones((1,6))*.1
    trace['last_servo_joint_rate_rad_s']=np.ones((1,6))*.2
    trace['servo_peak_limit_nm']=np.ones((1,6))*1.913
    for key in ('servo_shaft_abs_power_w','servo_shaft_drive_power_w',
                'servo_shaft_brake_power_w','spring_potential_energy_j',
                'shin_spring_energy_j'):
        trace[key]=np.zeros((10,1))
    trace['servo_shaft_abs_power_w'][:5]=999
    trace['servo_shaft_drive_power_w'][:5]=999
    # Only the scored 5-interval window may feed the spring-cycle proxy.
    trace['spring_potential_energy_j'][:5,0]=100.
    trace['spring_potential_energy_j'][5:,0]=[0.,1.,0.,2.,1.]
    trace['shin_spring_energy_j'][5:,0]=[0.,.5,0.,1.,0.]
    trace['servo_shaft_abs_power_w'][5:]=2
    trace['servo_shaft_drive_power_w'][5:]=1
    report=evaluate_trace_arrays(trace,np.array([[0,0,-9.81]]),
                                 settle_frames=5)[0]
    p=report['physical_evidence']
    assert p['step_integrated_abs_shaft_work_j']==pytest.approx(5*.02*2)
    assert p['step_integrated_drive_shaft_work_j']==pytest.approx(5*.02*1)
    assert p['all_springs_charging_proxy_j']==pytest.approx(3.)
    assert p['all_springs_release_proxy_j']==pytest.approx(2.)
    assert p['passive_shins_charging_proxy_j']==pytest.approx(1.5)
    assert p['passive_shins_release_proxy_j']==pytest.approx(1.5)
    assert report['positive_shaft_work_per_progress_m_j_m']==pytest.approx(.1/.015)
    # Same measured power but movement against the command: no positive
    # metres denominator is available for a physical efficiency verdict.
    trace['position_xy'][:,:,0]=-np.arange(10)[:,None]*.003
    trace['last_position_xy']=np.array([[-.030,0.]])
    opposite=evaluate_trace_arrays(trace,np.array([[0,0,-9.81]]),
                                   settle_frames=5)[0]
    assert opposite['positive_shaft_work_per_progress_m_j_m'] is None


def test_protected_export_reader_refuses_contract_and_actor_tampering(tmp_path):
    """A public browser export is a valid exam source, never a rewindable .pkl."""
    import json
    from pathlib import Path
    from emu.viewer_server import load_exported_actor_for_exam
    champion=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    layers,norm,details=load_exported_actor_for_exam(champion)
    assert details['source']['step']==120061952
    assert details['source']['checkpoint_schema']==7
    assert layers['pi'][0]['w'].shape==(220,256)
    assert norm['std'].shape==(220,)
    assert (norm['std']>0).all()
    obj=json.loads(champion.read_text())
    obj['source']['actor_sha256']='0'*64
    target=tmp_path/'tampered_actor.emupolicy'
    target.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='actor hash'):
        load_exported_actor_for_exam(target)
    obj=json.loads(champion.read_text())
    obj['observation']['base_fields'][0]='wrong_channel'
    target.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='channel names'):
        load_exported_actor_for_exam(target)
    obj=json.loads(champion.read_text())
    obj['source']['robot_model_sha256']='0'*64
    target.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='robot XML'):
        load_exported_actor_for_exam(target)


def test_transition_exam_measures_real_response_not_reward_or_foot_lift():
    from emu.directional import examine_command_transitions
    cmd=np.zeros((100,3))
    cmd[20:60,0]=.1
    dt=.02
    pos,head=ideal(cmd,dt=dt)
    r=examine_command_transitions(pos,head,cmd,dt=dt)
    assert [x['kind'] for x in r['maneuvers']]==['rest_to_walk','brake_to_stop']
    assert r['maneuver_unobserved_count']==0
    assert r['maneuvers'][0]['response_latency_s']==pytest.approx(0)
    assert r['maneuvers'][1]['response_latency_s']==pytest.approx(0)
    assert r['braking_path_first_0p5s_m']==pytest.approx(0)

    # Identical requests, but robot cannot pick up speed for 10 frames
    # after start and coasts for 8 frames after the stop command.
    delayed=np.zeros((101,2))
    for i in range(100):
        vx=(.1 if 30<=i<68 else 0.)
        delayed[i+1]=delayed[i]+np.array([dt*vx,0])
    late=examine_command_transitions(delayed,head,cmd,dt=dt)
    assert late['maneuvers'][0]['response_latency_s']==pytest.approx(.20)
    assert late['maneuvers'][1]['response_latency_s']==pytest.approx(.16)
    assert late['braking_path_first_0p5s_m']==pytest.approx(8*.1*dt)


def test_reverse_exam_does_not_give_credit_to_forward_only_motion():
    from emu.directional import examine_command_transitions
    cmd=np.tile([.1,0.,0.],(100,1))
    cmd[35:,0]=-.1
    pos,head=ideal(cmd)
    pass_result=examine_command_transitions(pos,head,cmd)
    assert pass_result['maneuvers'][0]['kind']=='reverse_or_redirect'
    assert pass_result['maneuvers'][0]['response_latency_s']==pytest.approx(0)
    forward_only=np.zeros_like(pos)
    forward_only[:,0]=np.arange(101)*.002
    failed=examine_command_transitions(forward_only,head,cmd)
    assert failed['maneuvers'][0]['response_latency_s'] is None
    assert failed['maneuver_unobserved_count']==1


def test_explicit_start_speed_and_stop_schedules_are_physical_requests():
    from emu.evaluation import make_schedule
    start=make_schedule('rest_start',120,speed=.1)
    step=make_schedule('speed_step',120,speed=.1)
    brake=make_schedule('brake_stop',120,speed=.1)
    assert np.all(start[:30,0]==0)
    assert np.allclose(start[30:,0],.1)
    assert np.allclose(step[:30,0],.04)
    assert np.allclose(step[30:,0],.1)
    assert np.all(brake[80:,0]==0)
    assert np.allclose(brake[:80,0],.1)


def test_protected_export_physical_campaign_plan_is_read_only(tmp_path):
    from pathlib import Path
    from emu.evaluation import plan, execute, verify_existing
    champion=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    root=tmp_path/'unique_exam_results'
    jobs=plan('physical',None,root,policy_file=champion,worlds=4,seed=1701)
    assert len(jobs)==10
    assert {x['arm'] for x in jobs}=={'champion'}
    assert all(x['is_exported_policy'] for x in jobs)
    assert all(x['run_dir']==str(champion.resolve()) for x in jobs)
    assert all(x['worlds']==4 and x['seed']==1701 for x in jobs)
    assert {x['scenario'] for x in jobs} >= {'rest_start','brake_stop','reverse',
                                               'turn_stop_left','sidestep_right'}
    execute(jobs,'synthetic_revision',actually_run=False)
    assert not root.exists(), 'planning must create no experiment directory'
    assert verify_existing(root/'missing.json',jobs[0],'src','sha') is False
    with pytest.raises(ValueError,match='exported policy'):
        plan('pilot',None,root,policy_file=champion)
    with pytest.raises(ValueError,match='specify training'):
        plan('physical',tmp_path/'old_run',root)


def test_physical_batch_refuses_stale_saved_source_or_changed_protocol(tmp_path):
    import json
    from pathlib import Path
    from emu.evaluation import plan, verify_existing
    hero=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    job=plan('physical',None,tmp_path/'out',policy_file=hero)[0]
    path=Path(job['output'])/'heading_comparison.json'
    path.parent.mkdir(parents=True)
    for mode in ('off','on'):
        (path.parent/mode).mkdir()
        (path.parent/mode/'directional_exam.json').write_text('{}')
    protocol=dict(angle_deg=job['angle'],direction=job['direction'],
        scenario=job['scenario'],worlds=job['worlds'],horizon=job['horizon'],
        seed=job['seed'],gyro_bias_rad_s=job['gyro_bias'],
        gyro_noise_std_rad_s=job['gyro_noise'],
        gyro_delay_frames=job['gyro_delay_frames'])
    payload=dict(schema=2, source_artifact_sha256='expected',
                 examined_at_revision='revision',protocol=protocol,
                 heading_kp=job['heading_kp'])
    path.write_text(json.dumps(payload))
    assert verify_existing(path,job,'revision','expected')
    with pytest.raises(ValueError,match='stale source'):
        verify_existing(path,job,'revision','modified-export')
    with pytest.raises(ValueError,match='stale source'):
        verify_existing(path,job,'modified-revision','expected')
    payload['protocol']['scenario']='wrong-way'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='changed exam parameters'):
        verify_existing(path,job,'revision','expected')


def test_public_cli_routes_exported_actor_and_batch_without_hidden_checkpoint(monkeypatch,tmp_path):
    """Regression: public emu CLI must not require --run-dir for --policy."""
    from pathlib import Path
    from emu import cli, evaluation
    policy=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    received=[]
    def one(*args):
        received.append(('exam',args))
        return 0
    monkeypatch.setattr(evaluation,'directional_main',one)
    assert cli.main(['directional-exam','--policy',str(policy),
                     '--output',str(tmp_path/'one'),'--device','cpu']) == 0
    assert received[0][1][:2] == ('--policy',str(policy))
    def plan_stub(*args,**kwargs):
        received.append(('plan',args,kwargs))
        return [{'sentinel':'read-only'}]
    def execute_stub(jobs,source,*,actually_run):
        received.append(('execute',jobs,actually_run))
    monkeypatch.setattr(evaluation,'plan',plan_stub)
    monkeypatch.setattr(evaluation,'execute',execute_stub)
    assert cli.main(['directional-batch','--policy',str(policy),
                     '--profile','physical','--output',str(tmp_path/'batch'),
                     '--device','cpu']) == 0
    assert received[1][2]['policy_file'] == str(policy)
    assert received[2][2] is False
    assert not (tmp_path/'batch').exists()


def test_v5016_common_pitch30_profile_is_not_an_experiment_arm(tmp_path):
    """New run only; v5015 historical recipe stays pinned and reproducible."""
    from types import SimpleNamespace
    from emu.branch_lab import one_run_config, ARMS
    def config(profile, pitch=None):
        return one_run_config(SimpleNamespace(
            profile=profile, run_dir=str(tmp_path/profile),steps=2048,
            envs=2,seed=733,device='cpu',pitch_deg=pitch))
    modern=config('sim2real')
    assert modern.attitude_profile=='v5015'
    assert modern.pitch_command_max_deg == pytest.approx(30.0)
    assert config('mature').pitch_command_max_deg == pytest.approx(8.0)
    assert config('foundation').pitch_command_max_deg == pytest.approx(8.0)
    assert config('sim2real',18).pitch_command_max_deg == pytest.approx(18.)
    # Pitch15 is only part of the preserved *historical* lab5015 recipe.
    # Future v5016 arms must use identical common 30deg range.
    assert 'pitch15' in {a.name for a in ARMS}
    assert modern.run_dir != config('mature').run_dir
    with pytest.raises(ValueError,match='pitch-deg'):
        config('sim2real',31.0)
    with pytest.raises(ValueError,match='pitch-deg'):
        config('sim2real',0.0)


def test_public_sim2real_plan_resolves_common_pitch_30(tmp_path,capsys):
    import json
    from emu import cli
    run=tmp_path/'fresh_v5016'
    assert cli.main(['train','--profile','sim2real',
                     '--run-dir',str(run),'--steps','2048',
                     '--envs','2','--device','cpu','--plan']) == 0
    output=capsys.readouterr().out
    data=json.loads(output[output.index('{'):output.rfind('}')+1])
    assert data['pitch_command_max_deg']==30.0
    assert data['attitude_profile']=='v5015'
    assert not run.exists()


def test_motor_guidance_keeps_historical_config_compatible_and_rejects_bad_arm(tmp_path):
    import json
    from dataclasses import asdict
    from emu.training_config import TrainingConfig, apply_experiment_knobs
    old=asdict(TrainingConfig())
    old.pop('motor_guidance_weight')
    p=tmp_path/'old_training_config.json'
    p.write_text(json.dumps(old))
    restored=TrainingConfig.load(p)
    assert restored.motor_guidance_weight == 0.0
    arm=apply_experiment_knobs(restored,(('motor_guidance_weight',.06),))
    assert arm.motor_guidance_weight == pytest.approx(.06)
    assert restored.motor_guidance_weight == 0.0
    with pytest.raises(ValueError,match='motor_guidance_weight'):
        apply_experiment_knobs(restored,(('motor_guidance_weight',.25),))
    with pytest.raises(ValueError,match='motor_guidance_weight'):
        apply_experiment_knobs(restored,(('motor_guidance_weight',-0.01),))


def test_passive_spring_accounting_at_stance_and_reset_boundary():
    """Stored energy is conservative geometry, not credited propulsion."""
    from emu.directional import examine_spring_cycle
    q=np.zeros((7,1,8),dtype=float)
    q[1,0,6]=.10   # right passive shin stores 0.02 J (k=4)
    q[2,0,6]=0.     # right releases at a grounded stance interval
    q[2,0,7]=.20   # left charges 0.08 J
    q[3,0,7]=0.     # left releases in non-stance
    q[4:,0,:]=999. # post-terminal episode contaminant
    contact=np.asarray([[1,1],[1,1],[1,0],[1,0]],dtype=bool)
    trace=dict(spring_joint_qpos_rad=q,
               spring_reference_rad=np.zeros(8),
               spring_stiffness_nm_rad=np.array([0.,0.,0.,0.,0.,0.,4.,4.]))
    p=examine_spring_cycle(trace,0,start=0,stop=3,terminal=True,contacts=contact)
    assert p['spring_interval_count']==3
    assert p['passive_shin_charge_j']==pytest.approx([.02,.08])
    assert p['passive_shin_release_j']==pytest.approx([.02,.08])
    assert p['passive_shin_stance_release_j']==pytest.approx([.02,0.])
    assert p['passive_shin_nonstance_release_j']==pytest.approx([0.,.08])
    assert p['passive_shin_stance_interval_count']==[3,1]
    assert p['spring_joint_net_storage_delta_j']==pytest.approx([0.]*8)
    assert max(p['spring_joint_release_j'])<0.1  # excludes the 999rad reset


def test_spring_accounting_survivor_uses_final_state_not_stale_last_frame():
    from emu.directional import examine_spring_cycle
    q=np.zeros((4,2,8))
    q[1:,1,6]=.1
    final=np.zeros((2,8))
    final[1,6]=0
    trace=dict(spring_joint_qpos_rad=q,
               last_spring_joint_qpos_rad=final,
               spring_reference_rad=np.zeros(8),
               spring_stiffness_nm_rad=np.array([0,0,0,0,0,0,4,4]))
    grounded=np.ones((5,2),dtype=bool)
    p=examine_spring_cycle(trace,1,start=0,stop=4,terminal=False,contacts=grounded)
    assert p['spring_interval_count']==4
    assert p['passive_shin_charge_j'][0]==pytest.approx(.02)
    assert p['passive_shin_release_j'][0]==pytest.approx(.02)
    assert p['passive_shin_stance_release_j'][0]==pytest.approx(.02)
    with pytest.raises(ValueError,match='misaligned'):
        examine_spring_cycle(trace,1,start=0,stop=4,terminal=False,
                             contacts=np.ones((4,2),dtype=bool))
