"""Independent frozen 360-input physical capability examiner.

Preserves strict/all-axis and physical mission as separate outcomes,
including per-direction supported step-up and step-down contact.
"""
from __future__ import annotations

from pathlib import Path
import time
import math
import numpy as np

from .environment import EmuMJXEnv
from .training_config import PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST
from .ppo import make_trainer
from .contracts import CONTROL_DT
from .evaluation_directional import source_commit
from .evaluation_commands import make_schedule

# Unified 360-actor frozen physical assessment. Reuses evaluate_trace;
# only the reporter/scorer is new, never another world or PPO implementation.
def score_unified_capability_trace(frames: dict, *, case: dict,
                                   settle_frames: int = 25) -> list[dict]:
    """Score observed two-slab contact, directed tracking and recovery.

    A successful root crossing *without five consecutive supported far-side
    frames* is a failure. Everything ends at the first reset boundary; the
    examiner is not allowed to turn a second episode into evidence.
    """
    import numpy as np
    from .contracts import CONTROL_DT
    from .environment import UNIFIED_SLAB_HALF_LENGTH_M, UNIFIED_SLAB_HALF_DEPTH_M

    xyz=np.asarray(frames['position_xy'],dtype=float)
    done=np.asarray(frames['done'],dtype=bool)
    fell=np.asarray(frames['fell'],dtype=bool)
    command=np.asarray(frames['command'],dtype=float)
    heading=np.asarray(frames['heading_rad'],dtype=float)
    normal=np.asarray(frames['initial_ground_xmat'],dtype=float)[:,:,2]
    lower=np.asarray(frames['initial_unified_ground_mocap_pos'],dtype=float)
    upper=np.asarray(frames['initial_terrain_mocap_pos'],dtype=float)
    axis=np.asarray(frames['initial_upper_xmat'],dtype=float)[:,:,0]
    across=np.asarray(frames['initial_upper_xmat'],dtype=float)[:,:,1]
    contacts_lower=np.asarray(frames['floor_contact'],dtype=bool)
    contacts_upper=np.asarray(frames['terrain_platform_contact'],dtype=bool)
    support=np.asarray(frames['terrain_support'],dtype=float)
    bad=np.asarray(frames['terrain_bad_contact'],dtype=float)
    power=np.asarray(frames['servo_shaft_drive_power_w'],dtype=float)
    pushes=np.asarray(frames['applied_push_force_n'],dtype=float)
    actions=(np.asarray(frames['actor_action'],dtype=float)
             if 'actor_action' in frames else None)
    slip=(np.asarray(frames['stance_slip_proxy'],dtype=float)
          if 'stance_slip_proxy' in frames else None)
    root_quat=np.asarray(frames['initial_root_quat'],dtype=float)
    if (xyz.ndim!=3 or xyz.shape[2]!=2 or done.shape!=xyz.shape[:2]
        or command.shape!=(*done.shape,3) or not np.isfinite(xyz).all()):
        raise ValueError('bad unified frozen trace dimensions or position values')
    horizon,worlds=done.shape
    if settle_frames>=horizon-5:
        raise ValueError('settle window leaves no meaningful evidence')
    results=[]
    for world in range(worlds):
        terminal=np.flatnonzero(done[:,world])
        stop=int(terminal[0]+1) if len(terminal) else horizon
        alive=stop==horizon and not bool(np.any(done[:,world]))
        world_fell=bool(np.any(fell[:stop,world]))
        sl=np.arange(settle_frames,max(settle_frames,stop-1))
        dx=np.diff(xyz[:stop,world],axis=0)/CONTROL_DT
        yaw=np.unwrap(heading[:stop,world])
        if stop>settle_frames+2:
            headings=yaw[sl]
            c=np.cos(headings)
            si=np.sin(headings)
            body=np.stack((c*dx[sl,0]+si*dx[sl,1],
                           -si*dx[sl,0]+c*dx[sl,1]),axis=1)
            actual_yaw=np.diff(yaw)[sl]/CONTROL_DT
            requested=command[sl,world]
            # Absolute error alone cannot tell persistent directional drift
            # from step-to-step oscillation. Keep the signed bias and standard
            # deviation from the SAME body-frame trace and settled window.
            errors=np.concatenate(
                (body-requested[:,:2],
                 (actual_yaw-requested[:,2])[:,None]),axis=1)
            mae=np.mean(np.abs(errors),axis=0)
            bias=np.mean(errors,axis=0)
            jitter=np.std(errors,axis=0)
            speed=np.linalg.norm(body,axis=1)
            achieved=float(np.sum(np.linalg.norm(np.diff(xyz[:stop,world],axis=0),axis=1)))
            commanded=float(np.sum(np.linalg.norm(requested[:,:2],axis=1))*CONTROL_DT)
        else:
            mae=np.array([float('inf')]*3)
            bias=np.full(3,float('nan'))
            jitter=np.full(3,float('nan'))
            achieved=0.
            commanded=0.
        contact_any=np.any(contacts_lower[:stop,world] | contacts_upper[:stop,world],axis=1)
        support_fraction=float(np.mean((support[:stop,world]>.05)&contact_any))
        bad_fraction=float(np.mean(bad[:stop,world]>.5))
        work=float(np.sum(np.maximum(0.,power[:stop,world]))*CONTROL_DT)
        # A valid command is tracked while *supported*. Motion toward a
        # small command is mandatory; standing still never passes.
        # A zero-motion policy must not earn a pure-yaw vertex by hiding
        # inside an overly broad absolute yaw tolerance.
        yaw_target=float(np.mean(np.abs(requested[:,2]))) if stop>settle_frames+2 else 0.
        yaw_response=float(np.mean(np.abs(actual_yaw))) if stop>settle_frames+2 else 0.
        yaw_response_ok=(yaw_target<.04 or yaw_response>=.35*yaw_target)
        tolerance=np.array([.07,.06,max(.08,.40*yaw_target)])
        tracking=bool(
            np.all(np.isfinite(mae)) and np.all(mae<=tolerance)
            and (commanded<.01 or achieved>=.30*commanded)
            and yaw_response_ok
            and support_fraction>=.35 and bad_fraction<.20 and alive)
        # A walker's alternating foot placements produce instantaneous yaw
        # oscillation even when mean heading does not drift. Preserve the
        # historical strict all-axis gate, but report a SECOND, explicitly
        # defined mission gate so supported step crossing or accurate
        # uphill progression is never buried beneath an unrelated 0.08rad/s
        # yaw-MAE cutoff. Neither result overwrites the other.
        movement_ok=bool(
            np.all(np.isfinite(mae[:2])) and
            np.all(mae[:2]<=tolerance[:2]) and
            (commanded<.01 or achieved>=.30*commanded) and
            support_fraction>=.35 and bad_fraction<.20 and alive)
        if yaw_target<.04:  # straight movement: hold mean yaw, limit wobble
            heading_ok=bool(np.isfinite(bias[2]) and np.isfinite(jitter[2])
                and abs(bias[2])<=.08 and jitter[2]<=.20)
        else:  # actual requested turn: preserve the strict yaw response gate
            heading_ok=bool(np.isfinite(mae[2]) and
                            mae[2]<=tolerance[2] and yaw_response_ok)
        mission_tracking=bool(movement_ok and heading_ok)
        grade=float(np.degrees(np.arccos(np.clip(normal[world,2],-1.,1.))))
        # Ground *direction* must match the requested task, not just the
        # absolute inclination: uphill is not evidence for downhill.
        gx=float(-normal[world,0]/max(.01,normal[world,2]))
        gy=float(-normal[world,1]/max(.01,normal[world,2]))
        desired_grade=float(case.get('slope_deg',0.))
        expected=np.tan(np.deg2rad(desired_grade))
        bearing=case.get('direction','uphill')
        expected_xy={
            'uphill':(expected,0.),'downhill':(-expected,0.),
            'cross_left':(0.,expected),'cross_right':(0.,-expected)
        }.get(bearing,(expected,0.))
        # NumPy comparisons return np.bool_, which json.dumps cannot
        # serialize. A scored exam result is a plain Python data contract.
        grade_verified=bool(abs(grade-desired_grade)<=.20
                        and abs(gx-expected_xy[0])<=np.tan(np.deg2rad(.20))
                        and abs(gy-expected_xy[1])<=np.tan(np.deg2rad(.20)))
        is_step=case.get('step_m',0.)>0.
        side=1. if world%2==0 else -1.
        line=upper[world]-side*UNIFIED_SLAB_HALF_LENGTH_M*axis[world]
        lower_lip=lower[world]+side*UNIFIED_SLAB_HALF_LENGTH_M*axis[world]+UNIFIED_SLAB_HALF_DEPTH_M*normal[world]
        upper_lip=upper[world]-side*UNIFIED_SLAB_HALF_LENGTH_M*axis[world]+UNIFIED_SLAB_HALF_DEPTH_M*normal[world]
        measured_h=float(np.dot(upper_lip-lower_lip,normal[world])) if is_step else 0.
        height_verified=bool(not is_step or abs(measured_h-case['step_m'])<.0005)
        measured_edge=float(np.degrees(np.arctan2(axis[world,1],axis[world,0])))
        requested_edge=float(case.get('edge_deg',0.))
        edge_error=(measured_edge-requested_edge+180.)%360.-180.
        edge_verified=bool(not is_step or abs(edge_error)<.35)
        displacement=xyz[:stop,world]-line[:2]
        forward=displacement@axis[world,:2]
        sideways=displacement@across[world,:2]
        far=(forward>.10)&(np.abs(sideways)<.27)
        contact_target=(contacts_upper if side>0 else contacts_lower)[:stop,world]
        sustained=((support[:stop,world]>.05)&(bad[:stop,world]<.5)
                   &np.any(contact_target,axis=1)&far)
        streak=0
        passed_at=None
        for t,ok in enumerate(sustained):
            streak=streak+1 if t>=settle_frames and ok else 0
            if streak>=5:
                passed_at=t
                break
        # The robot must start on the near side (not spawned over the edge).
        initial_near=bool(forward[0]<-.03)
        crossing=(bool(initial_near and passed_at is not None and alive
                       and grade_verified and height_verified and edge_verified)
                  if is_step else None)
        # Keep distinct qualification: can it remain supported while
        # moving on the requested actual grade, AND can it obey commands?
        # The original strict case_pass is retained for backward-compatible
        # policy ranking; stable != capable of controlled locomotion.
        slope_stability_pass=bool(
            alive and not world_fell and grade_verified and
            support_fraction>=.35 and bad_fraction<.20 and
            achieved>=max(.01,.20*commanded))
        slope_tracking_pass=bool(tracking and grade_verified)
        if case.get('kind')=='slope':
            passed=slope_tracking_pass
        elif case.get('kind') in ('step','mixed'):
            passed=bool(crossing) and tracking
        elif case.get('kind')=='recovery':
            # A perturbation is only creditable if it was physically present;
            # it must settle back into supported, command-following motion.
            requested_push=float(case.get('push_n',0.))
            seen_push=bool(np.max(pushes[:stop,world])>=.8*requested_push) if requested_push>0 else True
            q=np.clip(abs(root_quat[world,0]),0.,1.)
            measured_tilt=float(np.degrees(2*np.arccos(q)))
            requested_tilt=float(case.get('tilt_deg',0.))
            seen_tilt=(requested_tilt==0. or measured_tilt>=.75*requested_tilt)
            tail=max(settle_frames,stop-12)
            tail_supported=float(np.mean(contact_any[tail:stop])) if stop>tail else 0.
            passed=bool(tracking and seen_push and seen_tilt and tail_supported>=.60)
        else:
            passed=tracking
        kind=case.get('kind')
        if kind=='slope':
            mission_pass=bool(slope_stability_pass and mission_tracking)
        elif kind in ('step','mixed'):
            mission_pass=bool(crossing and mission_tracking)
        elif kind=='recovery':
            mission_pass=bool(passed)
        else:
            mission_pass=mission_tracking
        work_per_m=work/achieved if achieved>.01 else None
        action_delta=(float(np.mean(np.abs(np.diff(actions[:stop,world],axis=0))))
                      if actions is not None and stop>1 else None)
        slip_mean=(float(np.mean(slip[:stop,world]))
                   if slip is not None else None)
        results.append(dict(
            world=world,lesson=case['name'],
            direction=('step-up' if side>0 else 'step-down') if is_step else case.get('direction','flat'),
            first_episode_frames=stop, survived=alive, fell=world_fell,
            grade_deg=round(grade,3),grade_verified=grade_verified,
            slope_stability_pass=(slope_stability_pass if case.get('kind')=='slope' else None),
            slope_tracking_pass=(slope_tracking_pass if case.get('kind')=='slope' else None),
            physical_slope_gx=round(gx,5),physical_slope_gy=round(gy,5),
            requested_step_m=float(case.get('step_m',0.)),
            measured_step_m=round(measured_h,5),height_verified=height_verified,
            measured_edge_deg=round(measured_edge,3) if is_step else None,
            edge_verified=edge_verified if is_step else None,
            started_near_side=initial_near if is_step else None,
            reached_far_side=bool(np.any(far)) if is_step else None,
            supported_crossing_frame=passed_at,
            crossing_pass=crossing,
            tracking_mae_vx_mps=round(float(mae[0]),4) if np.isfinite(mae[0]) else None,
            tracking_mae_vy_mps=round(float(mae[1]),4) if np.isfinite(mae[1]) else None,
            tracking_mae_yaw_rad_s=round(float(mae[2]),4) if np.isfinite(mae[2]) else None,
            tracking_signed_bias_vx_mps=round(float(bias[0]),4) if np.isfinite(bias[0]) else None,
            tracking_signed_bias_vy_mps=round(float(bias[1]),4) if np.isfinite(bias[1]) else None,
            tracking_signed_bias_yaw_rad_s=round(float(bias[2]),4) if np.isfinite(bias[2]) else None,
            tracking_jitter_vx_mps=round(float(jitter[0]),4) if np.isfinite(jitter[0]) else None,
            tracking_jitter_vy_mps=round(float(jitter[1]),4) if np.isfinite(jitter[1]) else None,
            tracking_jitter_yaw_rad_s=round(float(jitter[2]),4) if np.isfinite(jitter[2]) else None,
            supported_fraction=round(support_fraction,3),
            bad_contact_fraction=round(bad_fraction,3),
            path_travel_m=round(achieved,4),
            requested_path_m=round(commanded,4),
            drive_work_j=round(work,4),
            mechanical_j_per_travel_m=round(work_per_m,2) if work_per_m is not None else None,
            mean_abs_actor_action_delta=round(action_delta,5) if action_delta is not None else None,
            mean_stance_slip_proxy=round(slip_mean,5) if slip_mean is not None else None,
            delivered_push_peak_n=round(float(np.max(pushes[:stop,world])),3),
            case_pass=bool(passed),
            mission_pass=bool(mission_pass),
            movement_tracking_pass=bool(movement_ok),
            heading_control_pass=bool(heading_ok)))
    return results


def _unified_exam_json(manifest: dict) -> str:
    """Strict JSON: preserve numeric/boolean meaning, never stringify scalars.

    JAX/NumPy scalar diagnostics may enter a frozen result despite explicit
    bool/float conversions in the scorer. Refuse arrays and nonfinite values
    rather than silently publishing invalid or misleading evidence.
    """
    import json

    def native_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported frozen-exam value: {type(value).__name__}")

    return json.dumps(manifest, indent=2, sort_keys=True,
                      allow_nan=False, default=native_scalar) + '\n'


def unified_terrain_matrix_cases(*, grades_deg=None, heights_cm=None):
    """Same absolute physical grades/heights and seeds for every tested policy.

    New optional examiner-only fixtures permit measured 2.5–3.5cm and
    8–15deg FRONTIER assessments without editing curriculum, reward, PPO,
    sensor contract or previously trained weights. Defaults are identical
    to the v5026/v5027 frozen matrix for backwards comparability.
    """
    degrees=(1.,2.,3.,5.) if grades_deg is None else tuple(grades_deg)
    heights=(.3,.5,1.,2.) if heights_cm is None else tuple(heights_cm)
    directions=('uphill','downhill','cross_left','cross_right')
    cases=[dict(name=f'{direction}_{deg:g}deg'.replace('.','p'),kind='slope',
                slope_deg=float(deg),direction=direction)
           for deg in degrees for direction in directions]
    cases.extend(
        dict(name=f'paired_steps_{cm:.1f}cm'.replace('.','p'),kind='step',
             step_m=float(cm)/100)
        for cm in heights)
    return cases


def evaluate_unified_capabilities(*, policy: Path, output: Path,
                                  worlds: int = 4, horizon: int = 300,
                                  speed: float = .10, seed: int = 1701,
                                  device: str = 'cpu', quick: bool = False,
                                  terrain_matrix: bool = False,
                                  matrix_grades_deg=None,
                                  matrix_heights_cm=None) -> tuple[Path,Path]:
    """One frozen array-only actor; identical MJX physics and seeds per case.

    No optimizer update, no teacher decisions, no automatic promotion. Uses
    real rotated slabs plus support pairs; refuses legacy 360 meaning.
    """
    import json
    from .contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN,TERRAIN_ARRAY_ONLY_REVISION
    from .viewer_server import load_exported_actor_for_exam
    from .ppo import configure_jax,require_backend
    if worlds<2 or worlds>32 or worlds%2 or horizon<100 or horizon>700:
        raise ValueError('unified examiner requires even 2..32 worlds, 100..700 frames')
    if not math.isfinite(speed) or not .04<=speed<=.30:
        raise ValueError('unified examiner speed must be 0.04..0.30 m/s')
    policy=Path(policy).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix!='.emupolicy':
        raise ValueError('unified assessment requires exported .emupolicy')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('unified exam output must be new or empty')
    if output==policy.parent or output in policy.parents or policy in output.parents:
        raise ValueError('exam output must be separate from source artifact')
    params,norm,export=load_exported_actor_for_exam(policy)
    if (export.get('observation_schema')!=OBS_SCHEMA_TERRAIN
        or not export.get('array_only')
        or export.get('sensor_model_revision',export.get('sensor_contract',{}).get('sensor_model_revision')) not in
           (None,TERRAIN_ARRAY_ONLY_REVISION)):
        raise ValueError('requires versioned array-only 360-input unified policy')
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=='cpu')
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=700,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_step_max_m=.04,
        push_max_force_n=3.2,
        attitude_profile=export['training']['attitude_profile'],
        pitch_command_max_deg=float(export['training']['pitch_command_max_deg']),
        phase_speed_gain=float(export['control']['phase_speed_gain']),
        velocity_i_ramp_s=float(
            (export['control'].get('velocity_i') or {}).get('ramp_s',.4)),
        velocity_i_multiplier=float(
            (export['control'].get('velocity_i') or {}).get('multiplier',1.)),
        sensor_noise_scale=export['sensor_settings'].get('sensor_noise_scale',0.),
        **{k:v for k,v in export['sensor_settings'].items()
           if k!='sensor_noise_scale'},
        open_frontier=True,combo_mix_frac=0.,combo_probe_frac=0.,
        field_teach_recovery=False)
    trainer=make_trainer(env,seed=seed,lr=PPO_LR,rollout=ROLLOUT_STEPS,
                        epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
                        value_cost=PPO_VALUE_COST)
    cases=[
        dict(name='flat',kind='flat'),
        dict(name='control_easy_0p08',kind='control',vx=.08),
        dict(name='uphill_2deg',kind='slope',slope_deg=2.,direction='uphill'),
        dict(name='downhill_2deg',kind='slope',slope_deg=2.,direction='downhill'),
        dict(name='cross_left_2deg',kind='slope',slope_deg=2.,direction='cross_left'),
        dict(name='cross_right_2deg',kind='slope',slope_deg=2.,direction='cross_right'),
        dict(name='paired_steps_0p8cm',kind='step',step_m=.008),
        dict(name='paired_steps_2cm',kind='step',step_m=.020),
        dict(name='mixed_incline_step',kind='mixed',step_m=.008,slope_deg=2.,edge_deg=17.),
        dict(name='reverse',kind='command',vx=-.06),
        dict(name='lateral_left',kind='command',vx=0.,vy=.05),
        dict(name='lateral_right',kind='command',vx=0.,vy=-.05),
        dict(name='yaw_left',kind='command',vx=0.,yaw=.20),
        dict(name='yaw_right',kind='command',vx=0.,yaw=-.20),
        dict(name='diamond_mixture',kind='command',vx=.06,vy=.025,yaw=.12),
        dict(name='start_from_rest',kind='transition',scenario='rest_start'),
        dict(name='brake_to_stop',kind='transition',scenario='brake_stop'),
        dict(name='push_1N',kind='recovery',push_n=1.0),
        dict(name='tilt_10deg',kind='recovery',tilt_deg=10.),
    ]
    if quick and terrain_matrix:
        raise ValueError('quick and terrain_matrix must not both be selected')
    if (matrix_grades_deg is not None or matrix_heights_cm is not None) and not terrain_matrix:
        raise ValueError('custom grade/step fixtures require --terrain-matrix')
    if terrain_matrix:
        degrees=tuple(matrix_grades_deg or (1.,2.,3.,5.))
        heights=tuple(matrix_heights_cm or (.3,.5,1.,2.))
        # This is the physical exam's validated 4cm collision model, not
        # the curriculum's unconstrained open teaching frontier. Reject
        # invalid or duplicate fixtures rather than silently clipping them.
        if (not degrees or not heights
            or any(not math.isfinite(x) or x<=0 or x>15 for x in degrees)
            or any(not math.isfinite(x) or x<=0 or x>4.0 for x in heights)
            or len(set(degrees))!=len(degrees)
            or len(set(heights))!=len(heights)
            or any(abs(x*10-round(x*10))>1e-6 for x in heights)):
            raise ValueError('exam grades must be unique 0<deg<=15; '
                'step heights unique 0<cm<=4 with 0.1cm resolution')
        # Read-only, matched physics matrix: no learning, no teacher, no
        # adaptive difficulty, no replacing the generalist frozen exam.
        cases=[dict(name='flat',kind='flat'),
               *unified_terrain_matrix_cases(grades_deg=degrees,heights_cm=heights),
               dict(name='lateral_left',kind='command',vx=0.,vy=.05),
               dict(name='lateral_right',kind='command',vx=0.,vy=-.05),
               dict(name='yaw_left',kind='command',vx=0.,yaw=.20),
               dict(name='yaw_right',kind='command',vx=0.,yaw=-.20)]
    if quick:
        cases=[c for c in cases if c['name'] in (
            'flat','control_easy_0p08','uphill_2deg','downhill_2deg','paired_steps_0p8cm',
            'mixed_incline_step','diamond_mixture','push_1N')]
    results=[]
    for case in cases:
        grade=float(case.get('slope_deg',0.))
        direction=case.get('direction','uphill')
        pitch=grade if direction!='downhill' else -grade
        roll=0.
        if direction.startswith('cross_'):
            pitch=0.
            roll=grade if direction=='cross_left' else -grade
        step=float(case.get('step_m',0.))
        vx=float(case.get('vx',speed))
        vy=float(case.get('vy',0.))
        yaw=float(case.get('yaw',0.))
        scenario=case.get('scenario','steady')
        schedule=make_schedule(scenario,horizon,speed=vx,lateral=vy,
                               yaw_rate=yaw,turn_rate=.25)
        packet=dict(regime_frac_steady=1.,regime_frac_transition=0.,
            regime_frac_resilience=0.,steady_difficulty=0.,
            transition_difficulty=0.,resilience_difficulty=0.,
            posture_difficulty=0.,specialist_mix=0.,
            probe_cmd_vx=vx,probe_cmd_vy=vy,probe_cmd_yaw=yaw,
            probe_tilt_rad=math.radians(case.get('tilt_deg',0.)),
            probe_linvel_mps=0.,probe_angvel_rad_s=0.,probe_joint_rad=0.,
            probe_push_n=float(case.get('push_n',0.)),
            probe_push_steps=5.,
            terrain_exposure=0.,terrain_height_m=step,
            slope_exposure=0.,slope_max_deg=0.,
            terrain_pitch_deg=pitch,terrain_roll_deg=roll,
            terrain_fixed_height_m=step,terrain_edge_angle_deg=float(case.get('edge_deg',0.)))
        t0=time.perf_counter()
        trace=trainer['evaluate_trace'](
            params,env.params_array(packet),
            jnp.asarray(norm['mean']),jnp.asarray(norm['std']),
            jax.random.PRNGKey(seed),horizon,0.,None,jnp.asarray(schedule))
        data=jax.device_get(trace)
        rows=score_unified_capability_trace(data,case=case)
        if step>0:
            # A real paired exam requires equal height in opposite directions.
            for up,down in zip(rows[::2],rows[1::2]):
                if (up['direction']!='step-up' or down['direction']!='step-down'
                    or abs(up['measured_step_m']-down['measured_step_m'])>.0005
                    or not up['height_verified'] or not down['height_verified']
                    or not up['edge_verified'] or not down['edge_verified']):
                    raise RuntimeError('two-slab paired real geometry did not match')
        summary=dict(case=case,worlds=rows,passed=sum(r['case_pass'] for r in rows),
                     mission_passed=sum(r['mission_pass'] for r in rows),
                     crossed=(sum(r['crossing_pass'] is True for r in rows)
                              if step>0 else None),
                     survived=sum(r['survived'] for r in rows),
                     stable=sum(bool(r['slope_stability_pass']) for r in rows),
                     wall_seconds=round(time.perf_counter()-t0,2))
        results.append(summary)
        physical=(f", supported crossing {summary['crossed']}/{worlds}"
                  if step>0 else
                  f", stable {summary['stable']}/{worlds}"
                  if case['kind']=='slope' else '')
        print(f"{case['name']}: mission {summary['mission_passed']}/{worlds}, "
              f"strict all-axis {summary['passed']}/{worlds}{physical}, "
              f"survived {summary['survived']}/{worlds}",flush=True)
    manifest=dict(
        schema='emu_unified_frozen_exam_v2_dual_scores',
        scoring_contract=dict(
            strict='historical case_pass: 3-axis mean absolute velocity/yaw tracking, support and survival',
            mission='physical task AND bounded forward/lateral MAE, supported useful movement, '
                    'plus zero-yaw mean signed yaw-rate bias <=0.08rad/s and jitter <=0.20rad/s; '
                    'actual yaw-command cases retain strict yaw response; not a heading-path guarantee',
            task='step requires five consecutive far-side supported frames and survival; '
                 'slope requires verified grade and physical moving stability',
            preserve_old_case_pass=True),
        source_commit=source_commit(),
        actor_artifact_sha256=export['artifact_sha256'],
        observation_schema=export['observation_schema'],
        sensor_revision=TERRAIN_ARRAY_ONLY_REVISION,
        source_training_step=export['source']['step'],
        protocol=dict(seed=seed,worlds=worlds,horizon=horizon,speed=speed,
                      backend=jax.default_backend(),quick=bool(quick),
                      terrain_matrix=bool(terrain_matrix),
                      matrix_grades_deg=(list(degrees) if terrain_matrix else None),
                      matrix_heights_cm=(list(heights) if terrain_matrix else None),
                      model='actual paired inclined MJX collision slabs',
                      contact='real foot contact, five supported far-side frames',
                      no_reward_or_curriculum_feedback=True,
                      recovery_azimuth_bins_sampled=min(worlds,8),
                      recovery_azimuth_bins_total=8),
        results=results,
        cautions=['A graded physical skill is not proof of real-hardware safety.',
                  'Recovery direction assessment needs at least 8 worlds to exercise all azimuth bins.',
                  'Shaft mechanical J/m is not electrical battery energy.',
                  'Newly trained unified 360 policies require source-pinned export.'])
    output.mkdir(parents=True,exist_ok=True)
    data_path=output/'unified_exam.json'
    md_path=output/'unified_exam.md'
    data_path.write_text(_unified_exam_json(manifest),encoding='utf-8')
    md=['# Unified EMU — frozen physical capability assessment','',
        'Physical mission and strict all-axis tracking are **different scores**.',
        'Walking gait yaw oscillation may fail strict yaw-rate MAE despite ',
        'real supported crossing. Mission includes bounded signed yaw bias',
        'and jitter; neither claims long-range heading precision.','',
        f"Source policy SHA256: {manifest['actor_artifact_sha256']}",
        f"Actor schema: {manifest['observation_schema']} / {manifest['sensor_revision']}",
        f"Exam source: {manifest['source_commit']}",'',
        '| Physical lesson | Mission pass | Strict all-axis | Supported crossing | Grade stability | Survival |',
        '|---|---:|---:|---:|---:|---:|']
    for item in results:
        stable=(str(item['stable'])+'/'+str(worlds)
                if item['case']['kind']=='slope' else '—')
        crossed=(str(item['crossed'])+'/'+str(worlds)
                 if item['crossed'] is not None else '—')
        md.append(f"| {item['case']['name']} | "
                  f"{item['mission_passed']}/{worlds} | {item['passed']}/{worlds} | "
                  f"{crossed} | {stable} | {item['survived']}/{worlds} |")
    md.extend(['','Per-world support, true step contact, commanded vs achieved',
               'body velocities, recovery, slip proxy, actor action changes and shaft work',
               'are in unified_exam.json. The control lesson has no fabricated',
               'electrical-energy or arbitrary energy-efficiency passing threshold.',
               'No training, policy selection or promotion was performed.'])
    md_path.write_text('\n'.join(md)+'\n')
    return data_path,md_path
