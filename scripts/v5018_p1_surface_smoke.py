#!/usr/bin/env python3
"""Actual MJX P1 opt-in terrain reward smoke, without loading ANY brain.

This is a bounded GPU/CPU physics-contract check, not PPO training and
not evidence that EMU can climb. It proves that box-surface truth compiles
inside EmuMJXEnv, flat/box-off-course rewards are unchanged under opt-in,
and moving the model to a real obstacle activates terrain measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    args=parser.parse_args()
    if not 2<=args.worlds<=8:
        parser.error('--worlds must be 2..8')

    # Select JAX platform BEFORE importing jax: no accidental CPU fallback.
    from emu.ppo import configure_jax,require_backend
    configure_jax(args.device)
    import jax
    import jax.numpy as jnp
    import mujoco
    from emu.environment import EmuMJXEnv
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_SENSOR
    from scripts.v5018_champion_step_pilot import collidable_course
    require_backend(args.device=='cpu')

    base=dict(num_envs=args.worlds,seed=733,max_episode_steps=60,
              observation_schema=OBS_SCHEMA_SENSOR)
    report={
        'schema':1,'study':'v5018_P1_box_surface_no_actor_smoke',
        'source_robot_xml_sha256':hashlib.sha256(
            Path(ROBOT_XML).read_bytes()).hexdigest(),
        'jax':jax.__version__,'mujoco':mujoco.__version__,
        'device':[str(x) for x in jax.devices()],
        'worlds':args.worlds,'champion_loaded':False,
        'actor_observation_changed':False,
        'legacy_reward_math_changed':False,
        'constraints':['only one static yawed worldbody box',
                       'physically correct top/side pair classification is a candidate, not normal-force ground reaction',
                       'moving-leg ToF occlusion and six-servo CPU/WASM parity remain unverified']
    }
    with tempfile.TemporaryDirectory(prefix='emu_v5018_terrain_') as tmp:
        flat=collidable_course(Path(ROBOT_XML),Path(tmp)/'flat.xml',0)
        model_sha=hashlib.sha256(flat.read_bytes()).hexdigest()
        # Exactly the same physical XML and RNG. Only the opt-in
        # measurement path differs, and it must never alter 0mm controls.
        baseline=EmuMJXEnv(str(flat),terrain_contact_mode='legacy',**base)
        terrain=EmuMJXEnv(str(flat),terrain_contact_mode='box_surface',**base)
        p0=baseline.params_array({})
        p1=terrain.params_array({})
        s0,ob0=baseline.reset({})
        s1,ob1=terrain.reset({})
        obs0=np.asarray(ob0)
        obs1=np.asarray(ob1)
        if not np.array_equal(obs0,obs1):
            raise RuntimeError('box off-course changed initial deployable actor observation')
        assert baseline.obs_dim==terrain.obs_dim==232

        # Six explicit actuator outputs are identical across conditions.
        # Real MJX simulation runs in both engines for three control frames.
        reward_diffs=[]
        obs_diffs=[]
        for frame_index in range(3):
            action=jnp.zeros((args.worlds,6),dtype=jnp.float32)
            a=baseline.step_jit(s0,action,p0)
            b=terrain.step_jit(s1,action,p1)
            old_reward=np.asarray(a.reward)
            new_reward=np.asarray(b.reward)
            if not np.array_equal(old_reward,new_reward):
                raise RuntimeError(
                    f'flat 0mm legacy reward parity violated: {old_reward.tolist()} vs {new_reward.tolist()}')
            # Two separately compiled XLA kernels may round different
            # floating-point intermediates on CUDA. Diagnose differences
            # down to the *actual observation field*; never silently
            # bless a material policy-input change.
            obs_legacy=np.asarray(a.obs)
            obs_terrain=np.asarray(b.obs)
            delta=np.abs(obs_legacy.astype(np.float64)-obs_terrain.astype(np.float64))
            if not np.isfinite(delta).all():
                raise RuntimeError('off-course actor observations contain nonfinite values')
            maximum=float(delta.max(initial=0.))
            obs_diffs.append(maximum)
            exact=bool(np.array_equal(obs_legacy,obs_terrain))
            if not exact:
                wi,oi=np.unravel_index(np.argmax(delta),delta.shape)
                report.setdefault('observation_rounding',[]).append({
                    'control_frame':frame_index+1,'world':int(wi),
                    'observation_channel':int(oi),'max_abs_difference':maximum,
                    'legacy_value':float(obs_legacy[wi,oi]),
                    'terrain_value':float(obs_terrain[wi,oi]),
                    'different_scalars':int(np.count_nonzero(delta)),
                })
                print('NOTE: off-course observation difference:',
                      report['observation_rounding'][-1],flush=True)
            # Keep an extremely narrow float32 numerical allowance,
            # not an adjustable metric or relaxed reward comparator.
            if not np.allclose(obs_legacy,obs_terrain,rtol=3e-6,atol=2e-6):
                raise RuntimeError(
                    f'off-course actor observation drift exceeds float32 '
                    f'rounding budget at frame {frame_index+1}; '
                    f'max_abs_difference={maximum:.9g}; '
                    f'first offender in report/stdout')
            active=np.asarray(b.metrics.get('terrain_geometry_active'))
            if np.any(active!=0):
                raise RuntimeError('off-course control unexpectedly activated box truth')
            reward_diffs.append(float(np.max(np.abs(old_reward-new_reward))))
            s0,s1=a.state,b.state
        report['flat_legacy_reward_exact_frames']=3
        report['flat_max_reward_difference']=max(reward_diffs)
        report['off_course_box_sha256']=model_sha
        report['off_course_obs_bitwise_identical']=all(v==0 for v in obs_diffs)
        report['off_course_obs_max_abs_difference']=max(obs_diffs)
        report['off_course_obs_float32_equivalent']=True

        # Physical model contains a real 10mm box. Recompute MuJoCo/MJX
        # kinematics near the obstacle, without introducing a trained actor.
        step=collidable_course(Path(ROBOT_XML),Path(tmp)/'step.xml',10)
        challenge=EmuMJXEnv(str(step),terrain_contact_mode='box_surface',**base)
        params=challenge.params_array({})
        state,_=challenge.reset({})
        q=state.data.qpos
        # Shift base horizontally above box. This is a PROBE fixture only:
        # not a valid gait, not a reported policy movement or PPO action.
        q=q.at[:,0].set(.20)
        d=state.data._replace(qpos=q)
        d=challenge.forward_batched(challenge.mjx_model,d)
        from emu.environment import CONTROL_DT
        zeros=jnp.zeros((args.worlds,6),dtype=jnp.float32)
        m=challenge._metrics(d,state.task,zeros,state.prev_action,
                             state.phase,state.prev_foot_pos)
        if not np.all(np.isfinite(np.asarray(m['height']))):
            raise RuntimeError('active terrain height is not finite')
        active=np.asarray(m['terrain_geometry_active'])
        if not np.any(active>0):
            raise RuntimeError('physical obstacle failed to activate terrain-relative metric')
        for name in ('support','stance_slip','swing_clearance','height',
                     'terrain_top_contact_fraction','terrain_riser_contact_fraction',
                     'terrain_min_signed_toe_clearance'):
            val=np.asarray(m[name])
            if val.shape!=(args.worlds,) or not np.isfinite(val).all():
                raise RuntimeError(f'Invalid terrain metric {name}: {val}')
        report['near_box_geometry_active_worlds']=int(np.sum(active>.5))
        report['new_metrics_finite']=True
        report['terrain_box_sha256']=hashlib.sha256(step.read_bytes()).hexdigest()

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
