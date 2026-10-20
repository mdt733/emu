"""v5018 P1 read-only frozen-champion pilot on a TRUE collidable low step.

Policy and control/sensor observations remain frozen. The box is inserted into
a copied MuJoCo worldbody, not the canonical robot model or exported policy.
Comparison starts on flat (identical extra box parked off-course), 5mm and
10mm. A world-fixed examiner rejects detours, heading reversal, timeouts,
missing foot support, and repeated threshold touches. Experimental pilot:
NO PPO training, NO terrain observation schema, NO production reward changes.

Run via: bash scripts/run_v5018_gait_and_step_pilot.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


from emu.terrain import collidable_course

def run(height_mm: int, heading_kp: float, seed: int, worlds: int,
        horizon: int, output: Path, *, step_yaw_deg: float = 0.,
        speed_m_s: float = .10, policy_file: Path | None = None):
    # configure_jax before importing jax directly
    from emu.ppo import configure_jax, require_backend
    configure_jax('cuda')
    import jax
    import jax.numpy as jnp
    import mujoco
    from emu.contracts import ROBOT_XML, OBS_SCHEMA_SENSOR
    from emu.viewer_server import load_exported_actor_for_exam
    from emu.environment import EmuMJXEnv
    from emu.ppo import make_trainer
    from emu.training_config import (
        PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST, ROLLOUT_STEPS)
    from emu.directional import examine_obstacle_crossing, ObstacleCrossingGate
    from emu.evaluation import evaluate_trace_arrays
    require_backend(False)

    default_policy = Path(__file__).resolve().parents[1]/'showcase'/'champion.emupolicy'
    policy = default_policy if policy_file is None else Path(policy_file).expanduser().resolve()
    if not policy.is_file():
        raise FileNotFoundError(f'policy file does not exist: {policy}')
    actor, normalizer, export = load_exported_actor_for_exam(policy)
    if export['observation_schema'] != OBS_SCHEMA_SENSOR:
        raise ValueError('pilot requires the immutable schema-2 232 actor')
    if policy_file is None and export['source']['step'] != 260112384:
        raise ValueError('unexpected protected policy training lineage')
    # Explicit --policy accepts independent 232-input research exports;
    # they must still prove the exact sensor contract inside this examiner.
    settings = export['sensor_contract']['settings']

    with tempfile.TemporaryDirectory(prefix='emu5018_pilot_') as tmp:
        course = collidable_course(
            ROBOT_XML,Path(tmp)/'course.xml',height_mm,
            yaw_deg=step_yaw_deg)
        course_sha = hashlib.sha256(course.read_bytes()).hexdigest()
        env = EmuMJXEnv(
            str(course),num_envs=worlds,seed=seed,
            max_episode_steps=700,push_max_force_n=0.,
            vx_hi_max=.6,vx_lo_min=-.2,vy_max_mps=.25,yaw_max_rad_s=1.5,
            phase_speed_gain=float(export['control']['phase_speed_gain']),
            attitude_profile=export['training']['attitude_profile'],
            pitch_command_max_deg=float(export['training']['pitch_command_max_deg']),
            observation_schema=OBS_SCHEMA_SENSOR,open_frontier=False,
            combo_mix_frac=0.,combo_probe_frac=0.,field_teach_recovery=False,
            **settings)
        if env.sensor_contract != export['sensor_contract']:
            raise ValueError('course sensor contract does not match champion')
        geom_id = mujoco.mj_name2id(
            env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'v5018_obstacle_step')
        if geom_id < 0:
            raise ValueError('actual obstacle collision geom missing')
        # Named contact trace sampling is consumed by the exam, not by PPO.
        env.challenge_geom_id = int(geom_id)
        trainer = make_trainer(
            env,seed=seed,lr=PPO_LR,rollout=ROLLOUT_STEPS,
            epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
            value_cost=PPO_VALUE_COST)
        task = {
            'regime_frac_steady':1.,'regime_frac_transition':0.,
            'regime_frac_resilience':0.,'steady_difficulty':0.,
            'transition_difficulty':0.,'resilience_difficulty':0.,
            'posture_difficulty':0.,'probe_cmd_vx':speed_m_s,
            'probe_cmd_vy':0.,'probe_cmd_yaw':0.,
            'probe_tilt_rad':0.,'probe_linvel_mps':0.,
            'probe_angvel_rad_s':0.,'probe_joint_rad':0.,
        }
        schedule = jnp.tile(jnp.asarray([[speed_m_s,0.,0.]],dtype=jnp.float32),
                            (horizon,1))
        trace = jax.device_get(trainer['evaluate_trace'](
            actor,env.params_array(task),
            jnp.asarray(normalizer['mean']),jnp.asarray(normalizer['std']),
            jax.random.PRNGKey(seed),horizon,float(heading_kp),
            None,schedule,0.,0.,0))
        expected_obs = env.obs_dim
        if expected_obs != 232:
            raise ValueError(f'pilot actor observation changed to {expected_obs}')

        original = evaluate_trace_arrays(
            trace,np.tile(np.asarray([0.,0.,-9.81]),(worlds,1)),
            dt=.02,settle_frames=25,ground_mode='inclined_contact_plane')
        per_world = []
        # The step's projected world-X depth changes when the front face
        # rotates relative to the requested straight-forward course.
        yaw=math.radians(step_yaw_deg)
        projected_half_x=(abs(.04*math.cos(yaw)) +
                          abs(.20*math.sin(yaw))) if height_mm else .04
        near_x=.20-projected_half_x
        far_x=.20+projected_half_x
        finish_x=max(.42,far_x+.06)
        gate=ObstacleCrossingGate(
            near_edge_m=near_x,far_edge_m=far_x,finish_m=finish_x,
            corridor_half_width_m=.12,foot_margin_m=.005,
            max_heading_error_deg=80.,max_idle_frames=60,
            min_runout_frames=8,max_translation_per_frame_m=.060)
        for w in range(worlds):
            done=np.asarray(trace['done'])[:,w]
            fell=np.asarray(trace['fell'])[:,w]
            terminal=np.flatnonzero(done>.5)
            stop=int(terminal[0]) if len(terminal) else horizon
            if terminal.size:
                final_xy = np.asarray(trace['position_xy'][stop,w])
                final_foot = np.asarray(trace['foot_xyz'][stop,w])
                final_heading = float(trace['heading_rad'][stop,w])
                last_floor = np.asarray(trace['floor_contact'][stop,w])
                last_box = np.asarray(trace['challenge_contact'][stop,w])
            else:
                final_xy = np.asarray(trace['last_position_xy'][w])
                final_foot = np.asarray(trace['last_foot_xyz'][w])
                final_heading = float(trace['last_heading_rad'][w])
                last_floor = np.asarray(trace['last_floor_contact'][w])
                last_box = np.asarray(trace['last_challenge_contact'][w])
            n=max(0,stop)
            position=np.concatenate((
                np.asarray(trace['position_xy'][:n,w]),final_xy[None,:]))
            feet=np.concatenate((
                np.asarray(trace['foot_xyz'][:n,w]),final_foot[None,:,:]))
            head=np.concatenate((
                np.asarray(trace['heading_rad'][:n,w]),[final_heading]))
            floor=np.concatenate((
                np.asarray(trace['floor_contact'][:n,w]),last_floor[None,:]))
            box=np.concatenate((
                np.asarray(trace['challenge_contact'][:n,w]),last_box[None,:]))
            support=np.asarray(floor|box,dtype=bool)
            if len(position)<3:
                per_world.append({
                    'world':w,'crossing_pass':False,
                    'rejection_reasons':['too_short_after_termination'],
                    'full_horizon_survival':False})
                continue
            flags=np.zeros(len(position),dtype=bool)
            flags[-1]=bool(len(terminal) or np.any(fell[:stop+1]>.5))
            # Observed body and ball-centre spatial exposure to the actual
            # step footprint. This is a diagnostic of over/around/side
            # contact, NOT a proxy for signed contact penetration.
            delta=feet[:,:,:2]-np.array([.20,0.])
            local_x= math.cos(yaw)*delta[:,:,0]+math.sin(yaw)*delta[:,:,1]
            local_y=-math.sin(yaw)*delta[:,:,0]+math.cos(yaw)*delta[:,:,1]
            inside=((np.abs(local_x)<.04) & (np.abs(local_y)<.20))
            ball_bottom=feet[:,:,2]-.018
            step_top=height_mm/1000.
            if height_mm:
                above_or_near=int(np.sum(inside & (ball_bottom>=step_top-.003)))
                below_top=int(np.sum(inside & (ball_bottom<step_top-.003)))
            else:
                above_or_near=below_top=0
            result=examine_obstacle_crossing(
                position,feet[:,:,:2],support,head,
                start_xy=(0.,0.),course_axis_xy=(1.,0.),
                requested_heading_rad=0.,fell=flags,gate=gate)
            result.update({
                'world':w,
                'full_horizon_survival':bool(not terminal.size),
                'physical_step_contact_world_frames':int(np.sum(np.any(box,axis=1))),
                'physical_step_contact_foot_frames':int(np.sum(box)),
                'ground_contact_foot_frames':int(np.sum(floor)),
                'foot_frames_inside_step_xy':int(np.sum(inside)) if height_mm else 0,
                'foot_frames_inside_step_xy_bottom_near_or_above_top':above_or_near,
                'foot_frames_inside_step_xy_bottom_below_top_by_3mm':below_top,
                'original_course_pass':original[w].get('course_pass',False),
                'original_contact_stride_pairs':original[w].get('contact_stride_pairs',0),
                'stride_gate_diagnostics':original[w].get('stride_gate_diagnostics'),
                'servo_above_nominal_continuous_frac':(
                    original[w].get('physical_evidence',{}).get(
                        'servo_all_sample_over_continuous_fraction')),
                'physical_servo_peak_torque_nm':(
                    max(original[w]['physical_evidence']['servo_peak_abs_torque_nm'])
                    if original[w].get('physical_evidence') else None),
            })
            per_world.append(result)
        result = {
            'schema':1, 'study':'read_only_low_step_policy_exam',
            'step_height_mm':height_mm,
            'flat_control_off_course_box':height_mm==0,
            'worlds':worlds,'horizon':horizon,'seed':seed,
            'requested_body_vx_m_s':speed_m_s,
            'step_yaw_deg':step_yaw_deg,
            'physical_obstacle_near_edge_x_m':near_x,
            'physical_obstacle_far_edge_x_m':far_x,
            'finish_line_world_x_m':finish_x,
            'requested_heading_rad':0.,
            'heading_kp':heading_kp,
            'source_commit_not_rewritten':export['source'].get('git_commit'),
            'source_policy_sha256':export['artifact_sha256'],
            'world_xml_sha256':course_sha,
            'simulator_versions':{
                'jax':jax.__version__,'mujoco':mujoco.__version__,
                'device':[str(x) for x in jax.devices()]},
            'crossing_success_worlds':sum(bool(r.get('crossing_pass'))
                                         for r in per_world),
            'step_contact_world_frames':sum(
                int(r.get('physical_step_contact_world_frames',0)) for r in per_world),
            'foot_frames_inside_step_xy':sum(
                int(r.get('foot_frames_inside_step_xy',0)) for r in per_world),
            'foot_frames_inside_step_xy_bottom_below_top_by_3mm':sum(
                int(r.get('foot_frames_inside_step_xy_bottom_below_top_by_3mm',0))
                for r in per_world),
            'mean_signed_progress_m':(
                float(np.mean([
                    r['final_signed_progress_m'] for r in per_world
                    if 'final_signed_progress_m' in r]))
                if any('final_signed_progress_m' in r for r in per_world)
                else None),
            'per_world':per_world,
            'limitations':[
                'Obstacle orientation is explicit; robot is still initially heading +X.',
                'The ball-bottom/top footprint metric is diagnostic, not a solver penetration estimate.',
                'No sample-by-sample normal force or collision impulse agreement measured.',
                'Policy has existing 232 sensor inputs, no preview profile.',
                'Actuator behaviour uses simulated MuJoCo model only.',
                'Crossing success is independent of frozen policy reward.',
                'Flat control parks identical obstacle box off the course.',
                'Pilot scores objective course result; not training readiness.',
            ],
        }
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({
            'step_height_mm':height_mm,'heading_kp':heading_kp,
            'step_yaw_deg':step_yaw_deg,'requested_speed_m_s':speed_m_s,
            'seed':seed,'pass_experiment_executed':True,
            'actual_crossing_success_worlds':result['crossing_success_worlds'],
            'worlds':worlds,
            'step_contact_world_frames':result['step_contact_world_frames'],
            'mean_signed_progress_m':result['mean_signed_progress_m'],
            'output':str(output),
        }),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--height-mm',type=int,choices=[0,5,10,15,20,30,40,60,80,100,120,150],required=True)
    p.add_argument('--policy',type=Path,default=None,
                   help='optional independent 232-input .emupolicy; default protected 260M frozen champion')
    p.add_argument('--step-yaw-deg',type=float,default=0.)
    p.add_argument('--speed',type=float,default=.10)
    p.add_argument('--heading-kp',type=float,choices=[0.,.75],required=True)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--worlds',type=int,default=4)
    p.add_argument('--horizon',type=int,default=350)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 2<=a.worlds<=8 or not 80<=a.horizon<=700:
        p.error('pilot uses 2–8 worlds and 80–700 frames')
    if not math.isfinite(a.step_yaw_deg) or abs(a.step_yaw_deg)>35:
        p.error('physical step orientation must be between -35 and +35 degrees')
    if not math.isfinite(a.speed) or not .05<=a.speed<=.18:
        p.error('physical pilot speed must be between .05 and .18 m/s')
    run(a.height_mm,a.heading_kp,a.seed,a.worlds,a.horizon,a.output,
        step_yaw_deg=a.step_yaw_deg,speed_m_s=a.speed,policy_file=a.policy)


if __name__ == '__main__':
    main()
