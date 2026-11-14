"""Frozen H40 D two-lip physics probe: same independently scored task for EVERY actor.

One program records a 64/128-world true-MJX trajectory batch and exports
small per-world evidence. This file NEVER constructs an optimizer update,
writes a checkpoint or adjusts the trained actor's success definition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from emu.d_science_exam import score_two_lip_trace
from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN, sensor_observation_contract
from emu.policy_export import load_exported_actor_for_exam
from emu.ppo import make_trainer, configure_jax, require_backend
from emu.training_config import (
    PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST)
from emu.environment import EmuMJXEnv

PROGRAMS={
    "up":dict(down=0.,command=(.18,0.,0.),phase="normal",tof="normal"),
    "mixed":dict(down=.50,command=(.18,0.,0.),phase="normal",tof="normal"),
    "down":dict(down=1.,command=(.18,0.,0.),phase="normal",tof="normal"),
    "mixed-command":dict(down=.50,command=(.20,.05,.08),
                         phase="normal",tof="normal"),
    "tof-masked":dict(down=.50,command=(.18,0.,0.),
                      phase="normal",tof="mask"),
    "phase-held":dict(down=.50,command=(.18,0.,0.),
                      phase="hold",tof="normal"),
}


def save_csv(path,records):
    if not records:
        raise ValueError(f'No physical evidence for {path}')
    with Path(path).open("w",encoding="utf-8",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(*,policy:Path,output:Path,program:str,worlds:int=128,
        horizon:int=1050,seed:int=1701,device:str="cuda"):
    if program not in PROGRAMS:
        raise ValueError(f'Unknown physical program {program}')
    if worlds<2 or worlds>512 or horizon<450 or horizon>1450:
        raise ValueError('Choose 2..512 worlds and 450..1450 control frames')
    output=output.expanduser().resolve()
    policy=policy.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite previous trial: {output}')
    pi,norm,actor=load_exported_actor_for_exam(policy)
    if actor['observation_schema']!=OBS_SCHEMA_TERRAIN or not actor['array_only']:
        raise RuntimeError('Only verified terrain_360/6 array-only policy')
    spec=PROGRAMS[program]
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=="cpu")
    train=actor['training']
    sensor=actor['sensor_settings']
    control=actor['control']
    vi=control.get('velocity_i') or {}
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=700,push_max_force_n=.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08,
        unified_capability_focus=True,
        terrain_carpet=True,terrain_carpet_prisms=True,
        solver_iterations=train.get('solver_iterations'),
        solver_ls_iterations=train.get('solver_ls_iterations'),
        academy_stream=True,academy_height_cap_m=.08,
        academy_course_episode_frames=1500,
        terrain_lesson='coherent-careful',
        attitude_profile=train['attitude_profile'],
        pitch_command_max_deg=float(train['pitch_command_max_deg']),
        phase_speed_gain=float(control['phase_speed_gain']),
        science_phase_lowcmd_scale=float(
            train.get('science_phase_lowcmd_scale',1.)),
        velocity_i_ramp_s=float(vi.get('ramp_s',.4)),
        velocity_i_multiplier=float(vi.get('multiplier',1.)),
        sensor_noise_scale=float(sensor.get('sensor_noise_scale',0.)),
        # TRUE school profiles; no simulated or invented second obstacle.
        progressive_school=True,refined_school=True,
        bidirectional_school=True,outcome_graduation=True,
        science_step_finish_on_transfer=True,
        two_lip_continuation_school=True,
        stair_tournament_arm='off',
        **{k:v for k,v in sensor.items() if k!='sensor_noise_scale'})
    assert env.sensor_contract==sensor_observation_contract(
        sensor,schema=OBS_SCHEMA_TERRAIN,array_only=True)
    trainer=make_trainer(
        env,seed=seed,lr=PPO_LR,rollout=ROLLOUT_STEPS,
        epochs=1,minibatches=1,
        entropy_cost=PPO_ENTROPY_COST,value_cost=PPO_VALUE_COST)
    teacher={
        "specialist_mix":1.0, "focus_weight_step":1.0,
        "athletic_rehearsal_frac":0.0,
        "academy_episode_probability":1.0,
        "academy_height_up_m":.05,
        "academy_height_down_m":.02,
        "terrain_height_m":.05,
        "outcome_down_share":spec['down'],
        "outcome_walk_alpha":1.,
        "regime_frac_steady":1.,
        "regime_frac_transition":0.,
        "regime_frac_resilience":0.,
        "probe_cmd_vx":spec['command'][0],
        "probe_cmd_vy":spec['command'][1],
        "probe_cmd_yaw":spec['command'][2],
        "probe_push_n":0.,
        "probe_tilt_rad":0.,
        "probe_linvel_mps":0.,
        "probe_angvel_rad_s":0.,
        "probe_joint_rad":0.,
    }
    print(f"EMU D TRUE-MJX trace program={program}, worlds={worlds}, "
          f"horizon={horizon}, seed={seed}: frozen weights",flush=True)
    started=time.monotonic()
    trace=trainer["evaluate_trace"](
        pi,env.params_array(teacher),
        jnp.asarray(norm['mean']),jnp.asarray(norm['std']),
        jax.random.PRNGKey(seed),horizon,0.,
        jnp.asarray(spec['command'],dtype=jnp.float32),
        phase_probe=spec['phase'],tof_probe=spec['tof'])
    frames=jax.device_get(trace)
    rows,examples=score_two_lip_trace(
        frames,label=program,actor_sha=actor['source']['actor_sha256'],
        seed=seed)
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'physical_worlds.csv',rows)
    if examples:save_csv(output/'trace_examples.csv',examples)
    sumfields=(
        "first_root_cross","second_root_cross",
        "first_upper_contact_before_root","second_upper_contact_before_root",
        "first_foot0_5frame","first_foot1_5frame",
        "second_foot0_5frame","second_foot1_5frame",
        "first_sequential_both","first_simultaneous_both",
        "second_sequential_both","second_simultaneous_both",
        "half_m_after_first","half_m_after_second",
        "after_second_recent_vx_ok","course_healthy_commanded")
    counts={k:sum(bool(x[k]) for x in rows) for k in sumfields}
    counts['fell']=sum(x['fell'] for x in rows)
    failures={}
    for r in rows:
        name=r['failure_stage'];failures[name]=failures.get(name,0)+1
    manifest=dict(
        schema="h40_d_two_lip_frozen_science_v1",
        actor_source=actor['source'],actor_sha=actor['source']['actor_sha256'],
        exported_policy_sha256=actor['artifact_sha256'],
        robot_xml_sha256=hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
        program=program,settings=spec,
        seed=seed,worlds=worlds,horizon=horizon,
        control_dt_s=.02,device=jax.default_backend(),
        elapsed_s=round(time.monotonic()-started,2),
        no_ppo_updates=True,first_life_only=True,
        contact="actual 8-convex-prism top normals & mesh elevation",
        first_5frame="each foot 5 consecutive frames, possibly different times",
        simultaneous="both feet 5 consecutive frames on SAME physical tile",
        corrected_exams_not_teacher_rewards=True,
        note="Up/mixed/down use real sampled independent signed risers. "
             "The masked-ToF and held-phase probes are input ablations "
             "on otherwise unchanged real physical worlds.",
        counts=counts,failures=failures,
        output_files=["physical_worlds.csv","trace_examples.csv"])
    (output/'exam.json').write_text(json.dumps(
        manifest,indent=2,allow_nan=False)+"\n",encoding='utf-8')
    print(f"D physical exam complete {program}: "
          f"first sequential {counts['first_sequential_both']}/{worlds}, "
          f"second sequential {counts['second_sequential_both']}/{worlds}; "
          f"independent course {counts['course_healthy_commanded']}/{worlds}",
          flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--program',choices=tuple(PROGRAMS),required=True)
    ap.add_argument('--worlds',type=int,default=128)
    ap.add_argument('--horizon',type=int,default=1050)
    ap.add_argument('--seed',type=int,default=1701)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=ap.parse_args(argv)
    run(**vars(args))


if __name__=="__main__":
    main()
