"""Independent heading, downhill-cheating and collision-course exams.

No optimizer update, training reward or curriculum feedback. The public
CLI remains available through emu.evaluation compatibility re-exports.
"""
from __future__ import annotations

# Values previously inherited from unrelated sections of evaluation.py.
from .contracts import VERSION, ROBOT_XML, CONTROL_DT
from .training_config import (
    TrainingConfig, PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST,
)
from .environment import EmuMJXEnv
from .ppo import make_trainer, configure_jax, require_backend
from .checkpoint_io import load_checkpoint
from .evaluation_commands import make_schedule
import sys
import time

# Physical signed-direction/contact-course exams and repeatable batch profiles.
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .contracts import OBS_DIM
from .directional import evaluate_trace_arrays


def exam_gravity(angle_deg, direction):
    """Exact tilted-gravity infinite-plane experiment; uphill is world-fixed."""
    if not math.isfinite(angle_deg) or not 0 <= angle_deg <= 5:
        raise ValueError('exam angle must lie in [0,5] degrees')
    uphill = dict(uphill=(1., 0.), downhill=(-1., 0.),
                  cross_left=(0., 1.), cross_right=(0., -1.)).get(direction)
    if uphill is None:
        raise ValueError('invalid slope direction')
    a = math.radians(angle_deg)
    return np.asarray((-9.81*math.sin(a)*uphill[0],
                       -9.81*math.sin(a)*uphill[1],
                       -9.81*math.cos(a)),dtype=np.float32)


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def source_commit():
    try:
        r=subprocess.run(['git','rev-parse','HEAD'],
              cwd=Path(__file__).resolve().parents[1],
              check=True,capture_output=True,text=True,timeout=5)
        return r.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return 'unknown'


def validate_request(*, run_dir, output, worlds, horizon, settle_frames,
                     angle_deg, direction, speed, lateral, yaw_rate):
    run=Path(run_dir).expanduser().resolve()
    dest=Path(output).expanduser().resolve()
    if not run.is_dir() and not (run.is_file() and run.suffix == '.emupolicy'):
        raise FileNotFoundError(run)
    if dest == run or run in dest.parents or dest in run.parents:
        raise ValueError('output directory must be separate from training directory')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError('use a new/empty exam output directory')
    if not 2 <= worlds <= 64 or not 30 <= horizon <= 700:
        raise ValueError('bounded exams require 2–64 worlds and 30–700 frames')
    if not 0 <= settle_frames < horizon - 2:
        raise ValueError('invalid settle window')
    if not all(math.isfinite(x) for x in (speed,lateral,yaw_rate)):
        raise ValueError('nonfinite request')
    if abs(speed) > .6 or abs(lateral) > .25 or abs(yaw_rate) > 1.5:
        raise ValueError('command outside bounded legacy-policy exam limits')
    g=exam_gravity(angle_deg,direction)
    return run,dest,g


def exam_request(*, angle_deg=0.,direction='uphill',worlds=8,horizon=350,
                 settle_frames=25,speed=.10,lateral=0.,yaw_rate=0.,
                 seed=1701,device='cuda',scenario='steady',turn_rate=.25,
                 gyro_bias=0.,gyro_noise=0.,gyro_delay_frames=0,
                 contact_slope=False):
    """One public exam identity, shared by run, recovery, and batch.

    Only operator-controlled settings belong here. Measured ground normals,
    actual gravity, source hashes and observed outcomes live in the result.
    """
    return dict(angle_deg=float(angle_deg),direction=direction,
                ground_mode='inclined_contact_plane' if contact_slope else 'tilted_gravity',
                worlds=int(worlds),horizon=int(horizon),
                settle_frames=int(settle_frames),
                speed=float(speed),lateral=float(lateral),yaw_rate=float(yaw_rate),
                seed=int(seed),device=device,scenario=scenario,
                turn_rate=float(turn_rate),
                gyro_bias_rad_s=float(gyro_bias),
                gyro_noise_std_rad_s=float(gyro_noise),
                gyro_delay_frames=int(gyro_delay_frames))


def evaluate_directional_run(run_dir=None, *, policy_file=None,
                 angle_deg, direction, output, worlds=8,
                 horizon=350, speed=.10, lateral=0., yaw_rate=0.,
                 seed=1701, device='cuda', settle_frames=25, heading_kp=0.0,
                 scenario='steady', turn_rate=.25,
                 gyro_bias=0., gyro_noise=0., gyro_delay_frames=0,
                 contact_slope=False):
    if not math.isfinite(heading_kp) or not 0 <= heading_kp <= 2:
        raise ValueError('heading-kp must lie in [0,2] per second')
    if not all(math.isfinite(v) for v in (gyro_bias,gyro_noise)) or (
        abs(gyro_bias) > .12 or not 0 <= gyro_noise <= .20 or
        not isinstance(gyro_delay_frames, int) or not 0 <= gyro_delay_frames <= 5):
        raise ValueError('gyro realism parameters outside bounded test limits')
    schedule=make_schedule(scenario,int(horizon),speed=speed,lateral=lateral,
                           yaw_rate=yaw_rate,turn_rate=turn_rate)
    if (run_dir is None) == (policy_file is None):
        raise ValueError('specify exactly one of full run directory or exported policy')
    run_dir,output,g=validate_request(
        run_dir=policy_file if policy_file is not None else run_dir,
        output=output,worlds=int(worlds),horizon=int(horizon),
        settle_frames=int(settle_frames),angle_deg=angle_deg,direction=direction,
        speed=speed,lateral=lateral,yaw_rate=yaw_rate)
    from .checkpoint_io import load_checkpoint
    from .environment import EmuMJXEnv
    from .contracts import ROBOT_XML
    from .ppo import make_trainer
    from .ppo import configure_jax, require_backend
    from .training_config import (
        TrainingConfig,PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST,ROLLOUT_STEPS)

    if run_dir.is_file():
        # The protected 120M Pitch15 .emupolicy is not a rewindable checkpoint.
        # Validate its physical/actor contracts and use its exported mean/std
        # through exactly the same JAX actor inference and MJX exam.
        from types import SimpleNamespace
        from .viewer_server import load_exported_actor_for_exam
        params, norm, export = load_exported_actor_for_exam(run_dir)
        cfg = SimpleNamespace(
            phase_speed_gain=float(export['control']['phase_speed_gain']),
            attitude_profile=export['training']['attitude_profile'],
            pitch_command_max_deg=float(export['training']['pitch_command_max_deg']))
        actor_std = norm['std']
        actor_schema=export['observation_schema']
        sensor_settings=export['sensor_settings']
        artifact_sha = export['artifact_sha256']
        training_step = int(export['source']['step'])
        checkpoint_sha = None
        source_type = 'exported_actor'
        source_training_commit = export['source'].get('git_commit')
        source_actor_sha = export['source'].get('actor_sha256')
    else:
        cfg=TrainingConfig.load(run_dir/'training_config.json')
        actor_schema=cfg.observation_schema
        from .training_config import training_sensor_contract
        contract=training_sensor_contract(cfg)
        sensor_settings=dict(contract['settings']) if contract else {}
        checkpoint=run_dir/'checkpoints/final.pkl'
        saved=load_checkpoint(checkpoint,validate=True)
        params=saved['state']['params']
        norm=saved['obs_norm']
        actor_std = np.sqrt(np.asarray(norm['var']))
        artifact_sha = sha256_file(checkpoint)
        training_step = int(saved['meta']['step'])
        checkpoint_sha = artifact_sha
        source_type = 'schema7_checkpoint'
        source_training_commit = (saved['meta'].get('source_provenance') or {}).get('git_commit')
        source_actor_sha = None
    from .contracts import OBS_DIM, SENSOR_OBS_DIM, OBS_SCHEMA_SENSOR, OBS_SCHEMA_LEGACY
    if actor_schema not in (OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR):
        raise ValueError('unsupported versioned directional examiner observation schema')
    expected_dim = SENSOR_OBS_DIM if actor_schema==OBS_SCHEMA_SENSOR else OBS_DIM
    if int(np.asarray(params['pi'][0]['w']).shape[0]) != expected_dim:
        raise ValueError('directional examiner actor width and versioned schema disagree')
    if np.asarray(norm['mean']).shape != (expected_dim,) or np.asarray(actor_std).shape != (expected_dim,):
        raise ValueError('directional examiner normalizer and actor schema disagree')
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=='cpu')
    uphill_axis=dict(uphill=(1.,0.),downhill=(-1.,0.),
                     cross_left=(0.,1.),cross_right=(0.,-1.))[direction]
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=int(worlds),seed=int(seed),
        contact_slope_deg=float(angle_deg) if contact_slope else 0.,
        contact_uphill_xy=uphill_axis,
        max_episode_steps=700,push_max_force_n=0.,
        vx_hi_max=.6,vx_lo_min=-.2,vy_max_mps=.25,yaw_max_rad_s=1.5,
        phase_speed_gain=cfg.phase_speed_gain,
        attitude_profile=cfg.attitude_profile,
        pitch_command_max_deg=cfg.pitch_command_max_deg,
        sensor_noise_scale=sensor_settings.get('sensor_noise_scale',0.),
        observation_schema=actor_schema,
        **{k:v for k,v in sensor_settings.items() if k!='sensor_noise_scale'},
        open_frontier=False,
        combo_mix_frac=0.,combo_probe_frac=0.,field_teach_recovery=False,
    )
    # Shared, nonrandom MJX gravity field. No robot XML modifications.
    actual_gravity=(np.asarray((0.,0.,-9.81),dtype=np.float32)
                    if contact_slope else g)
    if not contact_slope:
        env.mjx_model=env.mjx_model.tree_replace({'opt.gravity': jnp.asarray(g)})
    env.gravity_unit_world=jnp.tile(
        jnp.asarray(actual_gravity/np.linalg.norm(actual_gravity))[None,:],
        (int(worlds),1))
    trainer=make_trainer(
        env,seed=int(seed),lr=PPO_LR,rollout=ROLLOUT_STEPS,
        epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
        value_cost=PPO_VALUE_COST)
    task={
        'regime_frac_steady':1.,'regime_frac_transition':0.,
        'regime_frac_resilience':0.,'steady_difficulty':0.,
        'transition_difficulty':0.,'resilience_difficulty':0.,
        'posture_difficulty':0.,'probe_cmd_vx':float(speed),
        'probe_cmd_vy':float(lateral),'probe_cmd_yaw':float(yaw_rate),
        'probe_tilt_rad':0.,'probe_linvel_mps':0.,
        'probe_angvel_rad_s':0.,'probe_joint_rad':0.,
    }
    trace=trainer['evaluate_trace'](
        params,env.params_array(task),
        jnp.asarray(norm['mean']),jnp.asarray(actor_std),
        jax.random.PRNGKey(int(seed)),int(horizon),float(heading_kp),
        None,jnp.asarray(schedule),float(gyro_bias),float(gyro_noise),
        int(gyro_delay_frames))
    frames=jax.device_get(trace)
    evaluated=evaluate_trace_arrays(
        frames,np.tile(actual_gravity,(int(worlds),1)),dt=.02,
        settle_frames=int(settle_frames),
        ground_mode='inclined_contact_plane' if contact_slope else 'tilted_gravity')
    report={
        'schema':2,'source_commit':source_commit(),
        'source_artifact_type':source_type,
        'source_artifact_sha256':artifact_sha,
        'source_training_commit':source_training_commit,
        'actor_sha256':source_actor_sha,
        'checkpoint_sha256':checkpoint_sha,
        'checkpoint_training_step':training_step,
        'observation_schema':actor_schema,
        'exam':dict(**exam_request(
                    angle_deg=angle_deg,direction=direction,worlds=worlds,
                    horizon=horizon,settle_frames=settle_frames,
                    speed=speed,lateral=lateral,yaw_rate=yaw_rate,
                    seed=seed,device=device,scenario=scenario,
                    turn_rate=turn_rate,gyro_bias=gyro_bias,gyro_noise=gyro_noise,
                    gyro_delay_frames=gyro_delay_frames,contact_slope=contact_slope),
                    actual_gravity_world=actual_gravity.tolist(),
                    grade_reference_gravity_world=g.tolist(),
                    gravity_world=actual_gravity.tolist(),
                    measured_floor_normal_world=np.asarray(
                        frames['floor_normal_world'][0,0],dtype=float).tolist(),
                    heading_p_enabled=heading_kp>0,heading_kp=float(heading_kp),
                    heading_sensor='body gyro-Z integrated with optional bias/white noise/delay'),
        'worlds':evaluated,
    }
    output.mkdir(parents=True,exist_ok=True)
    output_json=output/'directional_exam.json'
    output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    ok=sum(bool(r.get('course_pass')) for r in evaluated)
    gait=sum(bool(r.get('walking_pass')) for r in evaluated)
    lines=[
        '# EMU v5016 directional exam','',
        'Frozen actor source ('+source_type+') SHA256: '+artifact_sha,
        'Original training source: '+str(source_training_commit),
        'Original checkpoint SHA256: '+str(checkpoint_sha or 'not included in export'),
        'Exam source: '+report['source_commit'],
        'Incline: '+str(angle_deg)+' deg, '+direction,
        'Ground mode: '+report['exam']['ground_mode']+
            ' (real plane + vertical gravity)' if contact_slope else
            'Ground mode: gravity-equivalent flat plane',
        'Course success: '+str(ok)+'/'+str(worlds),
        'Contact-pair stride candidate success: '+str(gait)+'/'+str(worlds),
        '',
        'The contact-pair gait count is an estimate, not measured ground-reaction force.',
        'Per-servo torque, continuous-rating duty, thermal exposure proxy and shaft work are in each world physical_evidence.',
        '0.638 N m is a nominal rating, not a verified safe duty envelope for the finished robot.',
        'Actuator-force times joint-speed is BEFORE separate MuJoCo joint damping/friction/springs; do not treat it as net motor output or electrical draw.',
        'Thermal exposure is uncalibrated squared torque duty; it does not predict winding temperature.',
        'Contact-pair slip is geometric and spring potential variation does not prove recovered work.',
        'Scenario: '+scenario+'; turn rate: '+str(turn_rate)+' rad/s.',
        'Gyro bias/noise/delay: '+str(gyro_bias)+' / '+
          str(gyro_noise)+' rad/s / '+str(gyro_delay_frames)+' frames.',
        'Heading P gain: '+str(heading_kp)+' /s (0 means OFF).',
        'When ON, only the yaw-rate request delivered to PPO is corrected.',
        'Exam course is ALWAYS scored against original operator commands.',
        'Yaw estimate integrates body gyro with configured sensor imperfections.',
        'Original operator command schedule is not changed by heading P.',
        'No PPO retraining, reward or policy modification was performed.',
        ('Real inclined collision plane with vertical gravity; not uneven ground.'
         if contact_slope else
         'Tilted gravity on a flat collision plane; not a physical ramp.'),
        '',
        '| World | Horizon | Course | Gait | Signed m | Stride pairs | Servo over % | Peak Nm | Drive J/m | Brake latency s |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    def numeric(value, decimals=2):
        return '—' if value is None else f'{float(value):.{decimals}f}'
    for r in evaluated:
        p=r.get('physical_evidence', {})
        maneuvers=[m for m in r.get('maneuvers', [])
                   if m.get('kind') == 'brake_to_stop']
        latency=([m['response_latency_s'] for m in maneuvers
                  if m.get('response_latency_s') is not None] or [None])[0]
        peak=max(p['servo_peak_abs_torque_nm']) if p.get('servo_peak_abs_torque_nm') else None
        over=p.get('servo_all_sample_over_continuous_fraction')
        lines.append('| '+' | '.join([
            str(r['world']),
            'yes' if r['full_horizon_survival'] else 'no',
            'yes' if r.get('course_pass') else 'no',
            'yes' if r.get('walking_pass') else 'not proven',
            numeric(r.get('signed_course_progress_m'),3),
            str(r.get('contact_stride_pairs',0)),
            numeric(over*100 if over is not None else None,1),
            numeric(peak,2),
            numeric(r.get('positive_shaft_work_per_progress_m_j_m'),1),
            numeric(latency,2),
        ])+' |')
    lines+=['','A dash means not measured or not applicable, NOT a zero-load or zero-latency result.',
            'Servo over % is time/sample fraction above nominal continuous rating, not overheating.',
            'Drive J/m is defined only for >0.01 m signed commanded-direction progress.',
            'Per-servo p95/peak/duty/work and full maneuver events are in directional_exam.json.']
    output_md=output/'directional_exam.md'
    output_md.write_text('\n'.join(lines)+'\n')
    return output_json,output_md



# G3/G4: independent, read-only terrain course. Uses make_trainer's existing
# frozen-policy evaluate_trace, never a second actor simulator or reward path.
def score_terrain_course_trace(frames: dict, *, height_m: float, speed: float,
                               settle_frames: int = 12) -> list[dict]:
    """Judge uninterrupted crossings from observed root motion and MJX contacts.

    The single reset at the start of evaluate_trace is the only allowed birth.
    Frames after a fall/timeout are discarded; restarting never earns passage.
    A posed foot or an elevated world-Z value cannot earn a crossing.
    """
    import numpy as np
    from .environment import TERRAIN_PLATFORM_HALF_LENGTH_M, TERRAIN_PLATFORM_HALF_DEPTH_M

    xy = np.asarray(frames['position_xy'], dtype=float)
    ended = np.asarray(frames['done'], dtype=bool)
    fell = np.asarray(frames['fell'], dtype=bool)
    ground = np.asarray(frames['floor_contact'], dtype=bool)
    platform = np.asarray(frames['terrain_platform_contact'], dtype=bool)
    supported = np.asarray(frames['terrain_support'], dtype=float)
    bad = np.asarray(frames['terrain_bad_contact'], dtype=float)
    vx = np.asarray(frames['terrain_vx'], dtype=float)
    watts = np.asarray(frames['servo_shaft_drive_power_w'], dtype=float)
    centers = np.asarray(frames['initial_terrain_mocap_pos'], dtype=float)
    quats = np.asarray(frames['initial_terrain_mocap_quat'], dtype=float)
    if xy.ndim != 3 or xy.shape[2] != 2 or ended.shape != xy.shape[:2]:
        raise ValueError('invalid terrain examination trace shape')
    horizon, worlds, _ = xy.shape
    rows = []
    for i in range(worlds):
        first_end = np.flatnonzero(ended[:, i])
        n = int(first_end[0]+1) if len(first_end) else horizon
        active = height_m > 1e-6 and centers[i, 2] > -.5
        actual_height_cm = (100*(centers[i,2]+TERRAIN_PLATFORM_HALF_DEPTH_M)
                            if active else 0.)
        direction = 'flat' if not active else 'step-down' if centers[i, 0] < 0 else 'step-up'
        sign = -1.0 if direction == 'step-down' else 1.0
        yaw = 2 * np.arctan2(quats[i,3], quats[i,0]) if active else 0.0
        # Dormant box yaw is randomized but physically meaningless. Both
        # original-flat and dormant-flat controls MUST grade world +X.
        axis = np.asarray((np.cos(yaw), np.sin(yaw)))
        edge = centers[i,:2] - sign*TERRAIN_PLATFORM_HALF_LENGTH_M*axis if active else np.array((.25,0.))
        progress = (xy[:n,i,:] - edge) @ axis
        lateral = (xy[:n,i,:] - edge) @ np.asarray((-axis[1],axis[0]))
        first = float(progress[0])
        relative = progress - first
        # A real raised/lowered crossing requires reaching the *far side*
        # and having real contacts with that target surface after arrival.
        far = progress > .10
        target_contacts = (platform if direction == 'step-up' else ground)[:n,i,:]
        valid = (np.arange(n) >= settle_frames) & far & (np.abs(lateral) < .27)
        valid &= (supported[:n,i] > .05) & (bad[:n,i] < .5)
        valid &= np.any(target_contacts, axis=-1)
        # Five consecutive good samples = >=100 ms real contact after the edge.
        streak = 0
        passed_at = None
        for t in range(n):
            streak = (streak+1) if valid[t] else 0
            if streak >= 5:
                passed_at = t
                break
        fully_survived = n == horizon and not bool(np.any(ended[:,i]))
        collision_pass = passed_at is not None and fully_survived
        if fully_survived:
            failure_reason = None
        else:
            flags = (
                ('height', 'terrain_fall_height'),
                ('tilt', 'terrain_fall_tilt'),
                ('roll_rate', 'terrain_fall_roll_rate'),
                ('pitch_rate', 'terrain_fall_pitch_rate'),
                ('timeout', 'terrain_timeout'),
            )
            reasons = [name for name, key in flags
                       if key in frames and bool(np.asarray(frames[key])[n-1,i] > .5)]
            failure_reason = '+'.join(reasons) if reasons else 'unknown_or_diverged'
        travel = float(relative[-1])
        tracking_mae = float(np.mean(np.abs(vx[settle_frames:n,i]-speed))) if n>settle_frames else None
        motor_work = float(np.sum(np.maximum(0,watts[:n,i]))*.02)
        # Real completion and flat progress are NOT interchangeable:
        # baseline rows report progress but have no step-crossing verdict.
        rows.append(dict(
            world=i, lesson=direction,
            maximum_requested_step_cm=round(float(height_m*100),2),
            actual_step_height_cm=round(float(actual_height_cm),2),
            frames_before_reset=n, elapsed_s=round(n*.02,2),
            first_failure_reason=failure_reason,
            height_at_end_m=(round(float(frames['terrain_height'][n-1,i]),4)
                             if 'terrain_height' in frames else None),
            full_horizon_survival=fully_survived,
            fell=bool(np.any(fell[:n,i])), reached_far_side=bool(np.any(far)),
            far_side_contact_held_100ms=passed_at is not None,
            crossing_pass=(bool(collision_pass) if active else None),
            first_pass_frame=passed_at, progress_m=round(travel,4),
            commanded_speed_m_s=float(speed),
            velocity_mae_m_s=(round(tracking_mae,4) if tracking_mae is not None else None),
            max_abs_lateral_m=round(float(np.max(np.abs(lateral))),4),
            support_fraction=round(float(np.mean(supported[:n,i]>.05)),4),
            bad_contact_fraction=round(float(np.mean(bad[:n,i]>.5)),4),
            drive_shaft_work_j=round(motor_work,4),
            drive_j_per_progress_m=(round(motor_work/travel,3) if travel>.01 else None),
        ))
    return rows


def evaluate_terrain_course(*, policy: Path, output: Path, worlds: int = 4,
                            horizon: int = 260, speed: float = .15,
                            heights_cm: tuple[float,...] = (0.,2.,4.),
                            seed: int = 1701, device: str = 'cpu') -> tuple[Path,Path]:
    """Matched flat/raised true-collision episodes, frozen actor, no training.

    This is the future terrain teacher's *independent* exam evidence; until
    4x4 actor sensing exists it cannot justify curriculum expansion.
    """
    import json
    import time
    from .contracts import OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN, ROBOT_XML
    from .viewer_server import load_exported_actor_for_exam
    from .ppo import configure_jax, require_backend

    if not 2 <= worlds <= 32 or worlds%2 or not 80 <= horizon <= 700:
        raise ValueError('terrain exam requires even 2–32 worlds and 80–700 frames')
    if not .02 <= speed <= .35:
        raise ValueError('terrain speed must be in [0.02, 0.35] m/s')
    if not heights_cm or any(not math.isfinite(x) or not 0 <= x <= 8 for x in heights_cm):
        raise ValueError('heights must be nonempty, finite centimetres within [0,8]')
    heights_cm = tuple(sorted(set(heights_cm)))
    policy = Path(policy).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix != '.emupolicy':
        raise ValueError('pass an existing verified .emupolicy file')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('terrain exam output must be new or empty')
    if output in policy.parents or output == policy.parent:
        raise ValueError('exam cannot write into policy source directory')

    params, norm, export = load_exported_actor_for_exam(policy)
    actor_schema=export['observation_schema']
    if export.get('array_only'):
        raise ValueError(
            'This 360-input actor uses two inclined collision slabs and an '
            'array-only sensor. The old single-platform terrain-exam scorer '
            'cannot certify its physical step performance. Use the mixed '
            'MJX acceptance smoke until a two-slab frozen examiner is added.'
        )
    if actor_schema not in (OBS_SCHEMA_LEGACY,OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN):
        raise ValueError('unsupported exported actor contract')
    sensor_settings=(export['sensor_settings']
        if actor_schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN) else {})
    # Select the intended JAX platform BEFORE initializing JAX/JIT.
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=='cpu')
    max_height = max(.001,max(heights_cm)/100)
    env_args = dict(
        robot_xml=str(ROBOT_XML), num_envs=worlds, seed=seed,
        attitude_profile=export['training']['attitude_profile'],
        pitch_command_max_deg=float(export['training']['pitch_command_max_deg']),
        phase_speed_gain=float(export['control']['phase_speed_gain']),
        max_episode_steps=700, push_max_force_n=0.,
        vx_hi_max=.6, vx_lo_min=-.2, vy_max_mps=.25, yaw_max_rad_s=1.5,
        sensor_noise_scale=sensor_settings.get('sensor_noise_scale',0.),
        observation_schema=actor_schema,
        **{k:v for k,v in sensor_settings.items() if k!='sensor_noise_scale'},
        combo_mix_frac=0.,combo_probe_frac=0.,
        field_teach_recovery=False,
    )
    def examiner(topology_height):
        args = dict(env_args)
        path = args.pop('robot_xml')
        env = EmuMJXEnv(path, terrain_step_max_m=topology_height,
                         paired_terrain_heights=topology_height>0, **args)
        trainer = make_trainer(
            env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
            epochs=1, minibatches=1, entropy_cost=PPO_ENTROPY_COST,
            value_cost=PPO_VALUE_COST)
        return env, trainer

    env, trainer = examiner(max_height)
    import numpy as np
    packet = dict(
        regime_frac_steady=1.,regime_frac_transition=0.,
        regime_frac_resilience=0.,steady_difficulty=0.,
        transition_difficulty=0.,resilience_difficulty=0.,
        posture_difficulty=0.,probe_cmd_vx=speed,
        probe_cmd_vy=0.,probe_cmd_yaw=0.,
        probe_tilt_rad=0.,probe_linvel_mps=0.,
        probe_angvel_rad_s=0.,probe_joint_rad=0.,
        terrain_exposure=1.,
    )
    def one_course(exam_env, exam_trainer, difficulty, label):
        task=dict(packet,terrain_height_m=difficulty/100.)
        t0=time.perf_counter()
        trace=exam_trainer['evaluate_trace'](
            params,exam_env.params_array(task),
            jnp.asarray(norm['mean']),jnp.asarray(norm['std']),
            jax.random.PRNGKey(seed),horizon,0.,(speed,0.,0.))
        frames=jax.device_get(trace)
        seconds=time.perf_counter()-t0
        if exam_env.terrain_step_mocap_id < 0:
            # The same scorer receives real dynamics, contacts and faults
            # from the original unmodified model. Only nonexistent platform
            # telemetry is explicitly marked absent, never invented as
            # physical support or success.
            frames['terrain_platform_contact']=np.zeros_like(frames['floor_contact'])
            frames['initial_terrain_mocap_pos']=np.tile(
                np.asarray([0.,0.,-1.]),(worlds,1))
            frames['initial_terrain_mocap_quat']=np.tile(
                np.asarray([1.,0.,0.,0.]),(worlds,1))
        rows=score_terrain_course_trace(frames,height_m=difficulty/100.,speed=speed)
        if difficulty>0:
            # A matched physical comparison needs equal actual severity,
            # not just an identical ceiling or PRNG seed.
            for up, down in zip(rows[::2], rows[1::2]):
                if (up['lesson']!='step-up' or down['lesson']!='step-down' or
                    abs(up['actual_step_height_cm']-down['actual_step_height_cm'])>.01):
                    raise RuntimeError('independent up/down exam height pairing failed')
        case=dict(
            case=label, max_height_cm=difficulty,
            actual_heights_cm=[x['actual_step_height_cm'] for x in rows],
            seconds=round(seconds,3),
            worlds=rows,survived=sum(x['full_horizon_survival'] for x in rows),
            crossing_pass=(None if difficulty==0 else sum(x['crossing_pass'] for x in rows)),
            mean_progress_m=round(float(np.mean([x['progress_m'] for x in rows])),4),
            mean_first_episode_s=round(float(np.mean([x['elapsed_s'] for x in rows])),3),
            failure_reasons={
                reason:sum(x['first_failure_reason']==reason for x in rows)
                for reason in sorted({x['first_failure_reason'] for x in rows
                                      if x['first_failure_reason'] is not None})
            },
        )
        print(f"{label}: survival {case['survived']}/{worlds}, "
              f"crossing {('not scored' if difficulty==0 else str(case['crossing_pass'])+'/'+str(worlds))}, "
              f"progress {case['mean_progress_m']:+.3f}m, "
              f"first episode {case['mean_first_episode_s']:.2f}s, "
              f"failures {case['failure_reasons']}, "
              f"wall {seconds:.1f}s", flush=True)
        return case

    # An independent A/B physics control: if the original exact flat
    # topology also collapses, do NOT attribute failure to the step.
    old_env, old_trainer = examiner(0.)
    cases=[one_course(old_env,old_trainer,0.,'original_flat_physics')]
    for height_cm in heights_cm:
        cases.append(one_course(env,trainer,height_cm,
                                f"terrain_topology_{height_cm:g}cm"))
    reference=cases[0]
    dormant=next((c for c in cases[1:] if c['max_height_cm']==0.),None)
    paired_diagnostics=(
        dict(
            original_flat_survived=reference['survived'],
            dormant_flat_survived=dormant['survived'],
            per_world_first_failure_frame_deltas=[
                d['frames_before_reset']-o['frames_before_reset']
                for o,d in zip(reference['worlds'],dormant['worlds'])
            ],
            per_world_progress_deltas_m=[
                round(d['progress_m']-o['progress_m'],4)
                for o,d in zip(reference['worlds'],dormant['worlds'])
            ])
        if dormant else None)

    report=dict(
        schema='emu_terrain_exam_v2_paired_up_down',source_commit=source_commit(),
        actor_artifact_sha256=export['artifact_sha256'],
        actor_schema=actor_schema,policy_step=export['source']['step'],
        evidence_grade='diagnostic_only_until_real_step_sensor_and_independent_exam',
        active_sensor_model=('physical_16cell_depth_and_valid_360' if actor_schema==OBS_SCHEMA_TERRAIN
                             else 'legacy_flat_plane_tripod_232' if actor_schema==OBS_SCHEMA_SENSOR
                             else 'legacy_220_without_terrain_preview'),
        exam=dict(seed=seed,backend=jax.default_backend(),worlds=worlds,
                  horizon=horizon,speed_m_s=speed,heights_cm=list(heights_cm),
                  model='single movable finite MJX platform, equal-height/yaw up/down pairs',
                  paired_actual_heights_and_edge_yaws=True,
                  support='actual foot-floor and foot-platform collision contacts'),
        cases=cases, flat_topology_comparison=paired_diagnostics, cautions=[
            'No PPO or curriculum update performed.',
            ('360-input actor receives real 16-zone slant depth and per-cell validity, '
             'with simulated noise/latency/foot occlusion; simulated competence '
             'is not hardware validation.' if actor_schema==OBS_SCHEMA_TERRAIN else
             '232-input actor still uses legacy tripod flat-contact-plane grade; '
             'not terrain-aware step sensing.' if actor_schema==OBS_SCHEMA_SENSOR
             else '220-input actor has no 4x4 preview, hence diagnostic only.'),
            'Original flat physics and dormant-platform flat physics are compared separately.',
            'If both flat controls fail, terrain crossing is NOT a meaningful skill estimate.',
            'Flat reference has no step-crossing score, only progress/survival.',
            'First fall/timeout truncates assessment; resets cannot earn passage.',
            'Passing requires far-side approach, >=100ms physical target support and entire-horizon survival.',
            'These are simulated servo shaft-work estimates, not measured battery energy.',
            'Paired up/down worlds share actual 30–100%-of-ceiling height and edge yaw; actual centimetres are recorded.',
            'Identical cohorts across heights; times include first-run compilation.'])
    output.mkdir(parents=True,exist_ok=True)
    json_path=output/'terrain_exam.json'
    json_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# EMU finite-terrain course · frozen-policy evidence','',
           f"Policy artifact SHA256: {export['artifact_sha256']}",
           f"Source commit: {report['source_commit']}",
           f"JAX backend: {jax.default_backend()}",
           f"Actor schema: {actor_schema}; sensor evidence: DIAGNOSTIC ONLY",'',
           '| Course | Survive | Genuine crossings | Mean progress | First episode | Wall time |',
           '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        crossings='N/A (flat)' if case['crossing_pass'] is None else f"{case['crossing_pass']}/{worlds}"
        lines.append(f"| {case['case']} | {case['survived']}/{worlds} | "
                     f"{crossings} | {case['mean_progress_m']:+.3f} m | "
                     f"{case['mean_first_episode_s']:.2f} s | {case['seconds']:.1f} s |")
    lines += ['', 'The flat row is a command-competence control, not an obstacle.',
              'A passed contact check is NOT a passed uninterrupted crossing.',
              'Original-flat vs dormant-platform control reveals physics regressions separately from policy failures.',
              'Paired up/down worlds share real height/yaw; each reports first failure, progress, actual support, effort and tracking.',
              'Exam data never enters PPO, rewards or the curriculum.']
    md_path=output/'terrain_exam.md'
    md_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return json_path,md_path


def _summarize_worlds(worlds):
    """Keep survival and course/gait obedience distinct; do not merge them."""
    def mean_optional(key):
        # Absent/unobserved events are explicitly JSON null. A missing
        # braking or transition response is NOT zero seconds, and must
        # never prevent reporting survival, actuator load or course travel.
        values=[float(w[key]) for w in worlds if w.get(key) is not None]
        return sum(values)/len(values) if values else None
    def physical_optional(key):
        values=[float(w['physical_evidence'][key]) for w in worlds
                if w.get('physical_evidence',{}).get(key) is not None]
        return sum(values)/len(values) if values else None
    count=len(worlds)
    physical=[w['physical_evidence'] for w in worlds if 'physical_evidence' in w]
    peak=max((max(p['servo_peak_abs_torque_nm']) for p in physical),default=None)
    return dict(worlds=count,
                physical_worlds=len(physical),
                positive_progress_worlds=sum(
                    w.get('positive_shaft_work_per_progress_m_j_m') is not None
                    for w in worlds),
                servo_over_continuous_frac=physical_optional('servo_all_sample_over_continuous_fraction'),
                servo_peak_abs_torque_nm=peak,
                drive_work_j=physical_optional('step_integrated_drive_shaft_work_j'),
                abs_work_j=physical_optional('step_integrated_abs_shaft_work_j'),
                spring_potential_range_j=physical_optional('spring_potential_range_j'),
                passive_shin_release_j=physical_optional('passive_shin_release_total_j'),
                passive_shin_stance_release_j=physical_optional('passive_shin_stance_release_total_j'),
                passive_shin_stance_release_fraction=physical_optional('passive_shin_stance_release_fraction'),
                passive_shin_stance_release_per_progress_j_m=physical_optional(
                    'passive_shin_stance_release_per_progress_m_j_m'),
                contact_pair_slip_speed_m_s=physical_optional('contact_pair_slip_speed_mean_m_s'),
                drive_work_per_progress_j_m=mean_optional('positive_shaft_work_per_progress_m_j_m'),
                abs_work_per_progress_j_m=mean_optional('absolute_shaft_work_per_progress_m_j_m'),
                survived_full=sum(bool(w['full_horizon_survival']) for w in worlds)/count,
                course_pass=sum(bool(w.get('course_pass')) for w in worlds)/count,
                gait_candidate_pass=sum(bool(w.get('walking_pass')) for w in worlds)/count,
                progress_m=mean_optional('signed_course_progress_m'),
                maneuver_response_s=mean_optional('maneuver_mean_response_s'),
                maneuver_unobserved_count=sum(int(w.get('maneuver_unobserved_count', 0))
                                            for w in worlds),
                braking_path_first_0p5s_m=mean_optional('braking_path_first_0p5s_m'),
                heading_mae_deg=mean_optional('heading_mae_deg'),
                yaw_rate_mae_rad_s=mean_optional('yaw_rate_mae_rad_s'),
                yaw_abs_error_1s=mean_optional('yaw_abs_error_1s'),
                yaw_abs_error_3s=mean_optional('yaw_abs_error_3s'),
                gyro_heading_error_mae_deg=mean_optional('gyro_heading_error_mae_deg'),
                mean_abs_controller_correction_rad_s=
                    mean_optional('mean_abs_controller_correction_rad_s'))


def evaluate_directional_pair(run_dir=None, *, policy_file=None,
                              output, heading_kp=0.75,
                              reuse_completed=False, **kwargs):
    """Two matched read-only exams plus one human-readable delta report.

    The same saved policy, initial seeds, command and slope are evaluated once
    with the outer heading loop OFF, once ON. Never silently select a winner.
    """
    target=Path(output).expanduser().resolve()
    if (run_dir is None) == (policy_file is None):
        raise ValueError('specify either a run directory or exported policy')
    run=Path(policy_file if policy_file is not None else run_dir).expanduser().resolve()
    if target==run or run in target.parents or target in run.parents:
        raise ValueError('comparison output must be outside original run')
    if not math.isfinite(heading_kp) or not 0 < heading_kp <= 2.0:
        raise ValueError('compare mode needs a positive heading-kp in (0,2]')
    if reuse_completed:
        off_json=target/'off'/'directional_exam.json'
        on_json=target/'on'/'directional_exam.json'
        if not off_json.is_file() or not on_json.is_file():
            raise FileNotFoundError('recovery needs BOTH finished off/on directional_exam.json reports')
        if (target/'heading_comparison.json').exists():
            raise FileExistsError('paired comparison already exists; refusing to overwrite it')
    else:
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError('comparison requires new/empty --output root')
        source_kwargs = {'policy_file': run} if policy_file is not None else {'run_dir': run}
        off_json,_=evaluate_directional_run(output=target/'off',
                                heading_kp=0.0,**source_kwargs,**kwargs)
        on_json,_=evaluate_directional_run(output=target/'on',
                               heading_kp=heading_kp,**source_kwargs,**kwargs)
    off=json.loads(off_json.read_text())
    on=json.loads(on_json.read_text())
    if reuse_completed:
        artifact=run if run.is_file() else run/'checkpoints'/'final.pkl'
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual_sha=sha256_file(artifact)
        # An exact protocol match is required before reading saved evidence.
        expected=exam_request(**kwargs)
        for mode, report, gain in (('off',off,0.),('on',on,heading_kp)):
            exam=report.get('exam',{})
            if report.get('schema')!=2 or report.get('source_artifact_sha256')!=actual_sha:
                raise ValueError('saved '+mode+' actor identity does not match source')
            if any(exam.get(k)!=v for k,v in expected.items()):
                raise ValueError('saved '+mode+' reuse protocol mismatch; do not mix examinations')
            if exam.get('heading_p_enabled')!=(gain>0) or exam.get('heading_kp')!=gain:
                raise ValueError('saved '+mode+' heading controller setting mismatched')
            if len(report.get('worlds',[]))!=expected['worlds']:
                raise ValueError('saved '+mode+' world count mismatched')
        if (off.get('source_commit') != on.get('source_commit')
            or off.get('source_artifact_type') != on.get('source_artifact_type')):
            raise ValueError('saved OFF/ON evidence came from different source revisions')
    if (off['source_artifact_sha256']!=on['source_artifact_sha256'] or
            off['checkpoint_training_step']!=on['checkpoint_training_step']):
        raise RuntimeError('paired evaluations did not use same checkpoint')
    for key in (*exam_request().keys(),
                'actual_gravity_world','measured_floor_normal_world'):
        if off['exam'].get(key)!=on['exam'].get(key):
            raise RuntimeError('paired examinations have unequal '+key)
    s_off=_summarize_worlds(off['worlds'])
    s_on=_summarize_worlds(on['worlds'])
    keys=('survived_full','course_pass','gait_candidate_pass','progress_m',
          'heading_mae_deg','yaw_rate_mae_rad_s','yaw_abs_error_1s','yaw_abs_error_3s',
          'gyro_heading_error_mae_deg',
          'mean_abs_controller_correction_rad_s',
          'maneuver_response_s','braking_path_first_0p5s_m',
          'servo_over_continuous_frac','servo_peak_abs_torque_nm',
          'drive_work_j','abs_work_j','spring_potential_range_j',
          'passive_shin_release_j','passive_shin_stance_release_j',
          'passive_shin_stance_release_fraction',
          'passive_shin_stance_release_per_progress_j_m',
          'contact_pair_slip_speed_m_s','drive_work_per_progress_j_m',
          'abs_work_per_progress_j_m')
    delta={k: (s_on[k]-s_off[k]) if s_on[k] is not None and s_off[k] is not None else None
           for k in keys}
    result=dict(schema=2,checkpoint_sha256=off['checkpoint_sha256'],
                source_artifact_sha256=off['source_artifact_sha256'],
                source_artifact_type=off['source_artifact_type'],
                examined_at_revision=off['source_commit'],
                protocol={k:off['exam'][k] for k in off['exam'] if k not in
                          ('heading_p_enabled','heading_kp')},
                heading_kp=heading_kp,analysis_revision=source_commit(),
                reused_completed_rollouts=bool(reuse_completed),
                off=s_off,on=s_on,
                on_minus_off=delta,conclusion='comparison only; no automatic promotion')
    path=target/'heading_comparison.json'
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Paired v5016 heading-hold evaluation','',
           'Source actor SHA256: '+result['source_artifact_sha256'],
           'Source type: '+result['source_artifact_type'],
           'Frozen rollout revision: '+result['examined_at_revision'],
           'Report analysis revision: '+result['analysis_revision'],
           'Reused completed rollouts: '+str(result['reused_completed_rollouts']),
           'Gain ON: '+str(heading_kp)+'/s. Both cases use same seeds and operator requests.',
           'The controller sees gyro-integrated relative yaw, not simulator true yaw.',
           'Course scoring always uses unmodified operator commands.',
           '',
           '| Measurement | OFF | ON | ON minus OFF |',
           '|---|---:|---:|---:|']
    for key in keys:
        vals=[s_off[key],s_on[key],delta[key]]
        lines.append('| '+key+' | '+' | '.join(
            '—' if x is None else f'{x:.4f}' for x in vals)+' |')
    lines+=['','Fractions are over all examined worlds. Other metrics are means',
            'of valid measured episodes, NOT conditionally matched survivor pairs.',
            'Physical summaries are per-world averages when observed, not pooled duty-weighted samples.',
            'Work per signed progress includes only worlds with >1 cm commanded-direction progress;',
            'inspect the positive_progress_worlds and physical_worlds counts before comparisons.',
            'Review individual world outcomes in off/ and on/ before any decision.',
            'A better heading MAE does not compensate for worse survival or course progress.',
            'This comparison did not retrain or alter the checkpoint.']
    md=target/'heading_comparison.md'
    md.write_text('\n'.join(lines)+'\n')
    return path,md


def directional_main(argv=None):
    """Compatibility entrypoint; the public `emu` parser owns arguments."""
    from .cli import main
    return main(['directional-exam', *(sys.argv[1:] if argv is None else argv)])
