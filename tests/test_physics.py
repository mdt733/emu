"""Three physical truths: torque/shaft power, real servo model, MJX rollouts."""
import numpy as np
import pytest
from emu.environment import servo_effort_metrics
from emu.gait_exam import evaluate_trace_arrays

def test_stall_can_require_torque_without_mechanical_shaft_power():
    tau = np.full((2, 6), 1.913, dtype=np.float32)
    stationary = np.zeros_like(tau)
    result = servo_effort_metrics(np, tau, stationary, np.full(6, 1.913))
    np.testing.assert_array_equal(result['servo_shaft_abs_power_w'], [0., 0.])
    np.testing.assert_array_equal(result['servo_over_rated_frac'], [1., 1.])
    np.testing.assert_allclose(result['servo_peak_torque_limit_frac'], [1., 1.], rtol=1e-6)

def test_loaded_robot_xml_owns_actuator_limits_and_spring_reference():
    import mujoco
    from emu.contracts import ACTUATED_JOINTS
    from emu.paths import ROBOT_XML
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    assert model.nu == 6
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)] == list(ACTUATED_JOINTS)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1],
                               np.full(6, 1.913), atol=1e-6)
    names = (*ACTUATED_JOINTS, 'r_shin_flex', 'l_shin_flex')
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in names]
    assert all(j >= 0 for j in ids)
    positions = model.jnt_qposadr[ids]
    spring_ref = model.qpos_spring[positions]
    stiffness = model.jnt_stiffness[ids]
    assert spring_ref.shape == stiffness.shape == (8,)
    np.testing.assert_allclose(stiffness[-2:], [4., 4.], atol=1e-6)

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
