"""Read-only v5016 real-MJX directed-course exam for saved v5015 actors.

Example:
  python -m emu.directional_exam --run-dir PATH --output NEW_DIR --angle 0.5
The output must be outside the run directory; no PPO weights are updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .contracts import OBS_DIM
from .gait_exam import evaluate_trace_arrays


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
    if not run.is_dir():
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


def evaluate_run(run_dir, *, angle_deg, direction, output, worlds=8,
                 horizon=350, speed=.10, lateral=0., yaw_rate=0.,
                 seed=1701, device='cuda', settle_frames=25):
    run_dir,output,g=validate_request(
        run_dir=run_dir,output=output,worlds=int(worlds),horizon=int(horizon),
        settle_frames=int(settle_frames),angle_deg=angle_deg,direction=direction,
        speed=speed,lateral=lateral,yaw_rate=yaw_rate)
    from .checkpoint_io import load_checkpoint
    from .environment import EmuMJXEnv
    from .paths import ROBOT_XML
    from .ppo import make_trainer
    from .runtime import configure_jax, require_backend
    from .training_config import (
        TrainingConfig,PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST,ROLLOUT_STEPS)

    cfg=TrainingConfig.load(run_dir/'training_config.json')
    checkpoint=run_dir/'checkpoints/final.pkl'
    saved=load_checkpoint(checkpoint,validate=True)
    params=saved['state']['params']
    if int(np.asarray(params['pi'][0]['w']).shape[0]) != OBS_DIM:
        raise ValueError('this examiner supports legacy 220-input actor only; no silent migration')
    norm=saved['obs_norm']
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=='cpu')
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=int(worlds),seed=int(seed),
        max_episode_steps=700,push_max_force_n=0.,
        vx_hi_max=.6,vx_lo_min=-.2,vy_max_mps=.25,yaw_max_rad_s=1.5,
        phase_speed_gain=cfg.phase_speed_gain,
        attitude_profile=cfg.attitude_profile,
        pitch_command_max_deg=cfg.pitch_command_max_deg,
        sensor_noise_scale=0.,open_frontier=False,
        combo_mix_frac=0.,combo_probe_frac=0.,field_teach_recovery=False,
    )
    # Shared, nonrandom MJX gravity field. No robot XML modifications.
    env.mjx_model=env.mjx_model.tree_replace({'opt.gravity': jnp.asarray(g)})
    env.gravity_unit_world=jnp.tile(
        jnp.asarray(g/np.linalg.norm(g))[None,:],(int(worlds),1))
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
        jnp.asarray(norm['mean']),jnp.sqrt(jnp.asarray(norm['var'])),
        jax.random.PRNGKey(int(seed)),int(horizon))
    frames=jax.device_get(trace)
    evaluated=evaluate_trace_arrays(
        frames,np.tile(g,(int(worlds),1)),dt=.02,
        settle_frames=int(settle_frames))
    report={
        'schema':1,'source_commit':source_commit(),
        'checkpoint_sha256':sha256_file(checkpoint),
        'checkpoint_training_step':int(saved['meta']['step']),
        'exam':dict(angle_deg=float(angle_deg),direction=direction,
                    gravity_world=g.tolist(),worlds=int(worlds),
                    horizon=int(horizon),settle_frames=int(settle_frames),
                    speed=float(speed),lateral=float(lateral),
                    yaw_rate=float(yaw_rate),seed=int(seed),device=device,
                    heading_p_enabled=False),
        'worlds':evaluated,
    }
    output.mkdir(parents=True,exist_ok=True)
    output_json=output/'directional_exam.json'
    output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    ok=sum(bool(r.get('course_pass')) for r in evaluated)
    gait=sum(bool(r.get('walking_pass')) for r in evaluated)
    lines=[
        '# EMU v5016 directional exam','',
        'Trained policy SHA256: '+report['checkpoint_sha256'],
        'Exam source: '+report['source_commit'],
        'Incline: '+str(angle_deg)+' deg, '+direction,
        'Course success: '+str(ok)+'/'+str(worlds),
        'Contact-pair stride candidate success: '+str(gait)+'/'+str(worlds),
        '',
        'The contact-pair gait count is an estimate, not measured ground-reaction force.',
        'No heading P controller, PPO retraining, or policy modification was used.',
        'A rotated gravity field represents infinite uniform ground; not a physical ramp.',
        '',
        '| World | Full horizon | Course pass | Gait candidate | Signed progress m | Heading MAE deg | Contact stride pairs |',
        '|---|---|---|---|---:|---:|---:|',
    ]
    for r in evaluated:
        head=r.get('heading_mae_deg')
        lines.append('| '+ ' | '.join([
            str(r['world']),
            'yes' if r['full_horizon_survival'] else 'no',
            'yes' if r.get('course_pass') else 'no',
            'yes' if r.get('walking_pass') else 'not proven',
            f"{r['signed_course_progress_m']:.3f}" if 'signed_course_progress_m' in r else '—',
            f'{head:.2f}' if head is not None else '—',
            str(r.get('contact_stride_pairs',0)),
        ])+' |')
    output_md=output/'directional_exam.md'
    output_md.write_text('\n'.join(lines)+'\n')
    return output_json,output_md


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--angle',type=float,default=0.)
    p.add_argument('--direction',choices=('uphill','downhill','cross_left','cross_right'),default='uphill')
    p.add_argument('--worlds',type=int,default=8)
    p.add_argument('--horizon',type=int,default=350)
    p.add_argument('--speed',type=float,default=.10)
    p.add_argument('--lateral',type=float,default=0.)
    p.add_argument('--yaw-rate',type=float,default=0.)
    p.add_argument('--settle-frames',type=int,default=25)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--device',choices=('cpu','cuda','gpu','auto'),default='cuda')
    a=p.parse_args(argv)
    for path in evaluate_run(a.run_dir,angle_deg=a.angle,direction=a.direction,
                  output=a.output,worlds=a.worlds,horizon=a.horizon,
                  speed=a.speed,lateral=a.lateral,yaw_rate=a.yaw_rate,
                  seed=a.seed,device=a.device,settle_frames=a.settle_frames):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
