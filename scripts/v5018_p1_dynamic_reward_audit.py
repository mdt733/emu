#!/usr/bin/env python3
"""Read-only, genuine moving-robot P1 reward/truth comparison.

A frozen 232-input champion is executed twice, with identical seed,
commands, robot.xml+box geometry, sensor contract and network weights.
Only the opt-in terrain reward-metric implementation changes. This is the
missing transition-level test: contact onset/loss, actual top/riser normals,
sliding, terminal state and reward divergence are observed in ACTUAL MJX
frames rather than manufacturing one perfectly stationary foot contact.

No PPO updates, policy writes, new actor inputs, warm-start requirement or
showcase changes are performed. Output includes experiment limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np


def summarise_trace(trace, *, horizon, worlds):
    arrays = {key:np.asarray(val) for key,val in trace.items() if key in {
        'reward_scalar','actor_action','position_xy','foot_xyz',
        'challenge_contact','floor_contact','terrain_geometry_active',
        'terrain_support_reference_elevation_m',
        'terrain_top_contact_fraction','terrain_riser_contact_fraction',
        'terrain_floor_contact_fraction','terrain_signed_toe_gap',
        'contact_support','swing_clearance','done','fell'}}
    if arrays['position_xy'].shape!=(horizon,worlds,2):
        raise RuntimeError('physical trace shape unexpectedly changed')
    for name in ('reward_scalar','contact_support','swing_clearance',
                 'terrain_geometry_active','terrain_top_contact_fraction',
                 'terrain_riser_contact_fraction','terrain_floor_contact_fraction'):
        if (arrays[name].shape!=(horizon,worlds) or
            not np.isfinite(arrays[name]).all()):
            raise RuntimeError('Nonfinite or mis-sized physical trace: '+name)

    box=np.asarray(arrays['challenge_contact'],dtype=bool)
    prev=np.concatenate((np.zeros_like(box[:1]),box[:-1]),axis=0)
    onset=box & ~prev
    stop=np.where(np.asarray(arrays['done'])>0.5,True,False)
    records=[]
    for w in range(worlds):
        first_reset=np.flatnonzero(stop[:,w])
        valid=int(first_reset[0]+1) if first_reset.size else horizon
        br=box[:valid,w]
        active=arrays['terrain_geometry_active'][:valid,w]
        top=arrays['terrain_top_contact_fraction'][:valid,w]
        side=arrays['terrain_riser_contact_fraction'][:valid,w]
        records.append({
            'world':int(w),'frames_to_first_terminal':valid,
            'terminal_before_horizon':bool(first_reset.size),
            'actual_box_contact_world_frames':int(np.count_nonzero(np.any(br,axis=-1))),
            'actual_box_foot_contact_frames':int(np.count_nonzero(br)),
            'box_contact_onsets':int(np.count_nonzero(onset[:valid,w])),
            'terrain_active_frames':int(np.count_nonzero(active>.5)),
            'top_normal_contact_frames':int(np.count_nonzero(top>0)),
            'riser_normal_contact_frames':int(np.count_nonzero(side>0)),
            'top_candidate_max_foot_fraction':float(np.max(top,initial=0.)),
            'riser_candidate_max_foot_fraction':float(np.max(side,initial=0.)),
            'signed_forward_progress_m':float(
                arrays['position_xy'][valid-1,w,0]-
                arrays['position_xy'][0,w,0]),
            'mean_support_fraction':float(
                np.mean(arrays['contact_support'][:valid,w])),
            'mean_support_reference_elevation_m':float(
                np.mean(arrays['terrain_support_reference_elevation_m'][:valid,w])),
            'max_support_reference_elevation_m':float(
                np.max(arrays['terrain_support_reference_elevation_m'][:valid,w])),
            'mean_swing_clearance_m':float(
                np.mean(arrays['swing_clearance'][:valid,w])),
            'minimum_signed_toe_gap_m':float(
                np.min(arrays['terrain_signed_toe_gap'][:valid,w])),
        })
    return records,arrays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1701)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=240)
    parser.add_argument('--step-mm',type=int,default=10,choices=(5,10,15,20))
    parser.add_argument('--heading-kp',type=float,default=.75)
    args=parser.parse_args()
    if not 2<=args.worlds<=8 or not 120<=args.horizon<=350:
        parser.error('bounded P1 audit requires 2–8 worlds, 120–350 frames')
    if not 0<=args.heading_kp<=2:
        parser.error('heading correction 0–2')

    from emu.ppo import configure_jax,require_backend
    configure_jax('cuda')
    import jax
    import jax.numpy as jnp
    import mujoco
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_SENSOR
    from emu.environment import EmuMJXEnv
    from emu.viewer_server import load_exported_actor_for_exam
    from emu.ppo import make_trainer
    from emu.training_config import (
        PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST,ROLLOUT_STEPS)
    from scripts.v5018_champion_step_pilot import collidable_course
    require_backend(False)

    policy=Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    actor,norm,export=load_exported_actor_for_exam(policy)
    if export['observation_schema']!=OBS_SCHEMA_SENSOR or export['source']['step']!=260112384:
        raise RuntimeError('unexpected frozen 232-input reference actor')
    settings=export['sensor_contract']['settings']
    command_schedule=jnp.tile(jnp.asarray([[.10,0.,0.]],dtype=jnp.float32),
                              (args.horizon,1))
    task={
      'regime_frac_steady':1.,'regime_frac_transition':0.,
      'regime_frac_resilience':0.,'steady_difficulty':0.,
      'transition_difficulty':0.,'resilience_difficulty':0.,
      'posture_difficulty':0.,'probe_cmd_vx':.10,
      'probe_cmd_vy':0.,'probe_cmd_yaw':0.,
      'probe_tilt_rad':0.,'probe_linvel_mps':0.,
      'probe_angvel_rad_s':0.,'probe_joint_rad':0.,
    }
    report={
      'schema':1,'study':'v5018_P1_REAL_MOVING_frozen_actor_comparison',
      'policy_sha256':policy_sha,'policy_training_steps':260112384,
      'canonical_robot_sha256':hashlib.sha256(
        Path(ROBOT_XML).read_bytes()).hexdigest(),
      'seed':args.seed,'worlds':args.worlds,'horizon':args.horizon,
      'step_height_mm':args.step_mm,'command_vx_m_s':.10,
      'heading_kp':args.heading_kp,
      'jax':jax.__version__,'mujoco':mujoco.__version__,
      'device':[str(x) for x in jax.devices()],
      'actor_updates':0,'actor_observation_schema':232,
      'inference_and_action_policy_frozen':True,
      'status':'STARTED',
      'limitations':[
        'Single static box in actual MJX, not varied natural terrain',
        'Physical contact pair and normal are not solver force or energy',
        'A policy that crosses a step need not deliberately lift over it',
        'Reset/curriculum paths may diverge after reward/termination changes',
        'This observes existing sensor-232 policy; cold start remains separate',
      ]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='emu5018_p1_dynamic_') as tmp:
        xml=collidable_course(ROBOT_XML,Path(tmp)/'real_step.xml',args.step_mm)
        report['physical_xml_sha256']=hashlib.sha256(xml.read_bytes()).hexdigest()
        traces={}
        for mode in ('legacy','box_surface'):
            env=EmuMJXEnv(
                str(xml),num_envs=args.worlds,seed=args.seed,
                max_episode_steps=700,push_max_force_n=0.,
                vx_hi_max=.6,vx_lo_min=-.2,vy_max_mps=.25,yaw_max_rad_s=1.5,
                phase_speed_gain=float(export['control']['phase_speed_gain']),
                attitude_profile=export['training']['attitude_profile'],
                pitch_command_max_deg=float(
                    export['training']['pitch_command_max_deg']),
                observation_schema=OBS_SCHEMA_SENSOR,open_frontier=False,
                combo_mix_frac=0.,combo_probe_frac=0.,
                field_teach_recovery=False,terrain_contact_mode=mode,**settings)
            if env.obs_dim!=232 or env.sensor_contract!=export['sensor_contract']:
                raise RuntimeError('policy observation or sensor contract drift')
            env.challenge_geom_id=int(env.terrain_box_geom_id) if mode=='box_surface' else int(
                mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,
                                  'v5018_obstacle_step'))
            trainer=make_trainer(
                env,seed=args.seed,lr=PPO_LR,rollout=ROLLOUT_STEPS,
                epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
                value_cost=PPO_VALUE_COST)
            full=jax.device_get(trainer['evaluate_trace'](
                actor,env.params_array(task),
                jnp.asarray(norm['mean']),jnp.asarray(norm['std']),
                jax.random.PRNGKey(args.seed),args.horizon,
                float(args.heading_kp),None,command_schedule,0.,0.,0,True))
            summary,arr=summarise_trace(
                full,horizon=args.horizon,worlds=args.worlds)
            traces[mode]=arr
            report[mode]={'worlds':summary}
            print(f'P1 dynamic {mode}: contact frames',
                  [q['actual_box_contact_world_frames'] for q in summary],
                  'top',[q['top_normal_contact_frames'] for q in summary],
                  'riser',[q['riser_normal_contact_frames'] for q in summary],
                  flush=True)
            report['status']='PARTIAL_'+mode.upper()
            args.output.write_text(json.dumps(report,indent=2)+'\n')

        old,new=traces['legacy'],traces['box_surface']
        dx=np.linalg.norm(
            np.asarray(old['position_xy'])-np.asarray(new['position_xy']),
            axis=-1)
        # PPO evaluate_trace records mlp()'s UNCLIPPED actor mean.
        # The environment receives clip(mean,-1,+1), so a reported raw
        # difference >2 is NEVER a delivered actuator command difference.
        raw_action_diff=np.max(
            np.abs(old['actor_action']-new['actor_action']),axis=-1)
        delivered_old=np.clip(old['actor_action'],-1.,1.)
        delivered_new=np.clip(new['actor_action'],-1.,1.)
        clipped_action_diff=np.max(
            np.abs(delivered_old-delivered_new),axis=-1)
        rewards=np.abs(old['reward_scalar']-new['reward_scalar'])
        # Both are closed-loop rollouts, not a matched-physics counterfactual.
        # Different compiled reward graphs can produce tiny floating-point
        # perturbations, amplified by recurrent contact and feedback.
        # Without terminals, a reward number itself cannot control the actor.
        report['matched_rollout_comparison']={
            'comparison_type':'same_seed_independent_closed_loop_rollouts',
            'NOT_a_same_physical_state_reward_counterfactual':True,
            'max_world_xy_divergence_m':float(np.max(dx)),
            'max_raw_actor_mean_difference':float(np.max(raw_action_diff)),
            'max_applied_clipped_action_difference':float(np.max(clipped_action_diff)),
            'max_reward_difference':float(np.max(rewards)),
            'total_frames_with_reward_change':int(np.count_nonzero(rewards>1e-7)),
            'first_frame_xy_diverges_gt_1mm':(
                int(np.flatnonzero(np.any(dx>.001,axis=1))[0])
                if np.any(dx>.001) else None),
            'first_frame_raw_actor_mean_diff_gt_1e-4':(
                int(np.flatnonzero(np.any(raw_action_diff>1e-4,axis=1))[0])
                if np.any(raw_action_diff>1e-4) else None),
            'first_frame_applied_action_diff_gt_1e-4':(
                int(np.flatnonzero(np.any(clipped_action_diff>1e-4,axis=1))[0])
                if np.any(clipped_action_diff>1e-4) else None),
            'first_frame_reward_diff_gt_1e-6':(
                int(np.flatnonzero(np.any(rewards>1e-6,axis=1))[0])
                if np.any(rewards>1e-6) else None),
        }
        # Meaningful: at least one true contact in MOVING simulation, and
        # measurement activates as the policy actually interacts with step.
        box_contacts=sum(
            q['actual_box_contact_world_frames']
            for q in report['box_surface']['worlds'])
        active=sum(
            q['terrain_active_frames']
            for q in report['box_surface']['worlds'])
        if box_contacts==0 or active==0:
            report['status']='FAIL_NO_MOVING_CONTACT_OR_TERRAIN_ACTIVATION'
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            raise RuntimeError(
                'Moving frozen actor never reached real step contacts or '
                'terrain-sensitive frames: cannot validate P1 transitions')
        report['status']='PASS_DYNAMIC_CONTACT_TRANSITIONS_AUDITED'
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('P1 dynamic summary:',json.dumps(
            report['matched_rollout_comparison'],indent=2),flush=True)
        print('Saved:',args.output,flush=True)


if __name__=='__main__':
    main()
