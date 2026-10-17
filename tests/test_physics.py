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
                  sensor_noise_scale=0.,attitude_profile='v5015',
                  pitch_command_max_deg=30.0)
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
    # The 30 degree command is a common posture lesson, not a new actor
    # feature, scripted servo offset, or separate Pitch30 study arm.
    from emu.environment import T
    from emu.contracts import P
    pitch_keys=jax.random.split(jax.random.PRNGKey(3016),128)
    pitch_tasks=np.asarray(env._sample_task(
        pitch_keys, env.params_array({'posture_difficulty':1.0})))
    max_rad=np.deg2rad(30.)
    assert np.max(np.abs(pitch_tasks[:,T.pitch])) <= max_rad + 1e-6
    assert np.max(np.abs(pitch_tasks[:,T.pitch])) > np.deg2rad(25.)
    assert np.any(pitch_tasks[:,T.pitch] > np.deg2rad(20.))
    assert np.any(pitch_tasks[:,T.pitch] < -np.deg2rad(20.))
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
    # A frozen examiner must not depend on the PPO critic or optimiser.
    # Same random key, six actor weights and world initialization must
    # produce identical physics whether the full training params or only
    # the exported actor's 'pi' subtree is supplied.
    pi_only=trainer['evaluate_trace'](
        {'pi':params['pi']},
        env.params_array({'probe_cmd_vx':.1,'probe_cmd_vy':0.,'probe_cmd_yaw':0.}),
        mean,std,jax.random.PRNGKey(199),4)
    pi_only=jax.device_get(pi_only)
    np.testing.assert_array_equal(pi_only['position_xy'],data['position_xy'])
    np.testing.assert_array_equal(pi_only['servo_torque_nm'],data['servo_torque_nm'])
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

    # Execute the *actual protected* Pitch15 120M actor through the very
    # same CPU MJX stepping kernel. This needs neither the historical
    # training .pkl nor a synthetic stand-in for the trained weights.
    from emu.viewer_server import load_exported_actor_for_exam
    from pathlib import Path
    hero = Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    actor, exported_norm, export = load_exported_actor_for_exam(hero)
    assert export['source']['step'] == 120061952
    assert len(actor['pi']) == 3
    assert actor['pi'][0]['w'].shape == (OBS_DIM,256)
    # Two-world, four-control-frame smoke; physical performance is NOT
    # inferred from an unconditioned four-frame sanity check.
    actual = trainer['evaluate_trace'](
        actor,env.params_array({'probe_cmd_vx':.1,'probe_cmd_vy':0.,'probe_cmd_yaw':0.}),
        jnp.asarray(exported_norm['mean']),jnp.asarray(exported_norm['std']),
        jax.random.PRNGKey(199),4)
    actual=jax.device_get(actual)
    assert actual['servo_torque_nm'].shape == (4,2,6)
    assert np.isfinite(actual['position_xy']).all()
    assert np.isfinite(actual['servo_torque_nm']).all()


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


def test_protected_pitch15_full_directional_exam_cpu(tmp_path):
    """Real end-to-end 120M export -> actor inference -> MJX -> report.

    Bounded CPU CI checks software truth; it is not GPU or hardware proof.
    """
    pytest.importorskip('mujoco')
    pytest.importorskip('optax')
    pytest.importorskip('jax')
    import json
    from pathlib import Path
    from emu.evaluation import evaluate_directional_run
    champion=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    out=tmp_path/'pitch15_physical_exam'
    report_path, markdown_path=evaluate_directional_run(
        policy_file=champion, output=out, angle_deg=0.,
        direction='uphill', worlds=2, horizon=45,
        speed=.06, seed=1701, settle_frames=0, device='cpu')
    assert report_path.is_file() and markdown_path.is_file()
    report=json.loads(report_path.read_text())
    assert report['source_artifact_type']=='exported_actor'
    assert report['checkpoint_sha256'] is None
    assert len(report['source_artifact_sha256'])==64
    assert report['checkpoint_training_step']==120061952
    assert report['source_training_commit']=='a5996b8cda2c394c7c65e3e938282a3808ded121'
    assert len(report['worlds'])==2
    assert all('course_pass' in world for world in report['worlds'])
    assert all(world.get('invalid_reason') is not None or
               ('physical_evidence' in world and
                world['physical_evidence']['actuator_sample_frames'] > 0)
               for world in report['worlds'])


def test_motor_guidance_does_not_change_real_mjx_physics_or_actor_contract():
    """Two identically seeded worlds: only reward differs under motor guidance."""
    pytest.importorskip('mujoco')
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    from emu.environment import EmuMJXEnv
    from emu.contracts import ROBOT_XML,OBS_DIM
    old=EmuMJXEnv(str(ROBOT_XML),num_envs=2,seed=733,
                  attitude_profile='v5015',pitch_command_max_deg=30.,
                  motor_guidance_weight=0.,sensor_noise_scale=0.)
    new=EmuMJXEnv(str(ROBOT_XML),num_envs=2,seed=733,
                  attitude_profile='v5015',pitch_command_max_deg=30.,
                  motor_guidance_weight=.08,sensor_noise_scale=0.)
    packet={'steady_difficulty':0.5,'posture_difficulty':1.}
    state_a,obs_a=old.reset(packet)
    state_b,obs_b=new.reset(packet)
    np.testing.assert_allclose(jax.device_get(obs_a),jax.device_get(obs_b),atol=0)
    actions=jnp.full((2,6),.2,dtype=jnp.float32)
    a=old.step(state_a,actions,packet)
    b=new.step(state_b,actions,packet)
    a,b=jax.device_get((a,b))
    assert a.obs.shape==b.obs.shape==(2,OBS_DIM)
    np.testing.assert_allclose(a.obs,b.obs,atol=0)
    np.testing.assert_allclose(a.state.data.qpos,b.state.data.qpos,atol=0)
    np.testing.assert_allclose(a.metrics['reward_weighted_motor_cost'],0.,atol=0)
    fraction=np.clip(b.metrics['servo_over_rated_frac'],0.,1.)
    shaft=np.clip(b.metrics['servo_shaft_abs_power_w']/20.,0.,1.)
    expected=-.08*(.65*fraction+.35*shaft)
    np.testing.assert_allclose(b.metrics['reward_weighted_motor_cost'],expected,atol=1e-6)
    np.testing.assert_allclose(b.reward,a.reward+expected,rtol=1e-5,atol=1e-5)
