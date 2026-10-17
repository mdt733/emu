"""Three physical truths: torque/shaft power, real servo model, MJX rollouts."""
import numpy as np
import pytest
from emu.environment import servo_effort_metrics
from emu.directional import evaluate_trace_arrays

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
    from emu.contracts import ROBOT_XML
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


def test_browser_grass_terrain_compiles_in_real_mujoco():
    """Build the exact browser-generated XML and compile actual contact geoms.

    JS unit tests alone cannot establish MJCF correctness. This uses Node
    only to generate the identical terrain source consumed by the browser.
    """
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which('node') is None:
        pytest.skip('Node not installed; browser workflow separately verifies this')
    import mujoco
    repo_root = Path(__file__).resolve().parents[1]
    javascript = (
        "import fs from 'node:fs';"
        "import {withTerrain} from './emu/viewer_src/src/terrain.js';"
        "const xml=fs.readFileSync('emu/robot.xml','utf8');"
        "process.stdout.write(withTerrain(xml,'park',{maxDegrees:10}));"
    )
    completed = subprocess.run(
        ['node', '--input-type=module', '-e', javascript],
        cwd=repo_root, capture_output=True, text=True, timeout=40,
    )
    assert completed.returncode == 0, completed.stderr
    model = mujoco.MjModel.from_xml_string(completed.stdout)
    assert model.nu == 6
    assert model.nhfield == 1
    assert model.hfield_nrow[0] == model.hfield_ncol[0] == 129
    assert model.ngeom >= 19  # Robot + ground + a grove of solid tree trunks
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'viewer_park_ground') >= 0
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'viewer_tree_trunk_0') >= 0
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor') == -1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_real_mjx_evaluator_trace_shapes_cpu():
    """Small real stepping smoke; exercises contact.geom1/geom2/dist plumbing."""
    pytest.importorskip('mujoco')
    pytest.importorskip('optax')
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    from emu.environment import EmuMJXEnv
    from emu.contracts import ROBOT_XML
    from emu.ppo import make_trainer
    from emu.contracts import OBS_DIM
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=2,seed=1701,
                  sensor_noise_scale=0.,attitude_profile='v5015')
    # E2: shared birth/transition envelope preserves exact seed/frontier
    # arithmetic and field-taught birth provenance in the REAL JAX sampler.
    from emu.environment import T
    from emu.contracts import P, SEED_VX_HI, SEED_VX_LO, FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE
    for adaptive in (False, True):
        # No additional MuJoCo world needed for this deterministic JAX
        # contract: the sampler uses the environment's JAX configuration.
        env.open_frontier = adaptive
        packet = env.params_array({
            'steady_difficulty': .35, 'regime_command_frac': .6,
            'frontier_vx_hi': .8, 'frontier_vx_lo': -.25,
            'frontier_vy': .3, 'frontier_yaw': 1.6})
        full, reduced = env._command_envelopes(packet)
        full = np.asarray(full, dtype=np.float32)
        reduced = np.asarray(reduced, dtype=np.float32)
        base_hi = (max(SEED_VX_HI, .8) if adaptive
                   else float(np.asarray(env._lerp(SEED_VX_HI, env.vx_hi_max, .35))))
        base_lo = (min(SEED_VX_LO, -.25) if adaptive
                   else float(np.asarray(env._lerp(SEED_VX_LO, env.vx_lo_min, .35))))
        expected_full = np.asarray([
            base_hi, base_lo,
            (.3 if adaptive else .35 * env.vy_max_mps),
            (1.6 if adaptive else .35 * env.yaw_max_rad_s)], np.float32)
        np.testing.assert_allclose(full, expected_full, rtol=1e-6, atol=1e-7)
        expected_reduced = np.asarray([
            SEED_VX_HI + .6 * (expected_full[0] - SEED_VX_HI),
            SEED_VX_LO + .6 * (expected_full[1] - SEED_VX_LO),
            .6 * expected_full[2], .6 * expected_full[3]], np.float32)
        np.testing.assert_allclose(reduced, expected_reduced, rtol=1e-6, atol=1e-7)
    env.open_frontier = False
    anchors = np.zeros((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE), np.float32)
    anchors[0] = [0.055, 0.0, 0.0, 10.0, 0.0]
    choices = dict(steady_difficulty=.35, regime_command_frac=.6,
                   field_node_anchors=anchors)
    seed_keys = jax.random.split(jax.random.PRNGKey(774), 64)
    free = np.asarray(env._sample_task(seed_keys, env.params_array({
        **choices, 'field_node_frac': 0.0})))
    taught = np.asarray(env._sample_task(seed_keys, env.params_array({
        **choices, 'field_node_frac': 0.95})))
    selected = taught[:, T.field_taught] > 0.5
    assert selected.any()
    assert np.all(free[:, T.field_taught] == 0.0)
    np.testing.assert_allclose(taught[selected, T.vx], 0.055, rtol=0, atol=1e-6)
    np.testing.assert_allclose(taught[selected, T.vy], 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(taught[selected, T.yaw], 0.0, rtol=0, atol=1e-6)
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
    # Real MJX JIT: no synthetic substitute for source actuator fields.
    assert data['servo_torque_nm'].shape==(4,2,6)
    assert data['servo_joint_rate_rad_s'].shape==(4,2,6)
    assert data['last_servo_torque_nm'].shape==(2,6)
    # Nominal MJX shares limits across worlds; domain-randomized worlds
    # may instead expose one six-actuator array per world.
    assert data['servo_peak_limit_nm'].shape in ((6,), (2,6))
    np.testing.assert_allclose(data['servo_peak_limit_nm'], 1.913, atol=1e-6)
    for name in ('servo_torque_nm','servo_joint_rate_rad_s',
                 'servo_shaft_abs_power_w','servo_shaft_drive_power_w',
                 'spring_potential_energy_j'):
        assert np.isfinite(data[name]).all()
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
