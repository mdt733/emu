"""Read-only real-H40 kinematic reach + constrained-servo lift investigation.

Separates three DIFFERENT claims:
  1. Does an in-range joint configuration place the spherical foot higher
     AND farther forward than the nominal resting foot?
  2. Can the real 0.980665 Nm-capped position servos drive toward that
     configuration in a native MuJoCo simulation without changing hardware?
  3. Does it climb an actual step? NOT TESTED HERE. This is flat-ground
     single-foot lifting, not independent proof of stair ascent.

No PPO / rewards / policy inputs / hardware changes; real H40-generated XML,
joint angles, knee-strut tendon, passive shin flex and floor contacts.

Example:
  python -m scripts.h40_lift_feasibility --height-mm 50 --forward-mm 50 \
      --output ~/emu-data/exams/h40-lift-native-s735

Both requested lengths are in millimetres. The actual MuJoCo joint ranges,
not arbitrary training curriculum caps, bound the tested search.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from emu.contracts import ACTUATED_JOINTS, ROBOT_XML
from emu.physical_variants import (
    FOOT_RADIUS_M, SERVO_RATED_CONTINUOUS_TORQUE_NM)


def _id(model, kind, name):
    n=mujoco.mj_name2id(model,kind,name)
    if n<0:
        raise RuntimeError(f"Physical H40 model lacks {name}")
    return int(n)


def physical_setup():
    """Load canonical H40 XML without modifying contact or actuator models."""
    model=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    joint=_id
    jid={name:joint(model,mujoco.mjtObj.mjOBJ_JOINT,name)
         for name in ACTUATED_JOINTS}
    address={name:int(model.jnt_qposadr[index])
             for name,index in jid.items()}
    aid={name:joint(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
         for name in ACTUATED_JOINTS}
    rg=joint(model,mujoco.mjtObj.mjOBJ_GEOM,'r_foot_ball')
    lg=joint(model,mujoco.mjtObj.mjOBJ_GEOM,'l_foot_ball')
    floor=joint(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    limits=np.asarray(model.actuator_forcerange,dtype=float)
    if (not np.all(model.actuator_forcelimited) or
            np.max(np.abs(limits))>SERVO_RATED_CONTINUOUS_TORQUE_NM+1e-5):
        raise RuntimeError("Motor force ceiling does not match hardcap H40")
    if not np.isclose(model.geom_size[rg,0],FOOT_RADIUS_M,atol=1e-6):
        raise RuntimeError("Different ball-foot hardware")
    return model,address,aid,(rg,lg,floor)


def _joint_samples(model,name,address,grid_step_deg):
    index=_id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    lo,hi=np.rad2deg(model.jnt_range[index])
    count=max(2,int(np.ceil((hi-lo)/grid_step_deg))+1)
    return np.linspace(lo,hi,count)


def reachability(model,address,geoms,*,height_mm,forward_mm,
                 grid_step_deg=5.):
    """Exhaust the TWO swing joints at fixed nominal chassis pose.

    It cannot claim a feasible loaded stance, only FK reach and desired
    displacement in actual joint-limited geometry. A candidate with a
    lifted toe and forward displacement avoids the old mistake of
    rewarding vertical clearance while the foot is stuck behind a wall.
    """
    if grid_step_deg<=0. or not np.isfinite(grid_step_deg):
        raise ValueError("Grid resolution must be a positive finite degree value")
    if not np.isfinite([height_mm,forward_mm]).all():
        raise ValueError("Requested displacement must be finite")
    if height_mm<0 or forward_mm<0:
        raise ValueError("Reachability displacement must be non-negative")
    right,_,_=geoms
    home=mujoco.MjData(model)
    mujoco.mj_forward(model,home)
    q0=home.qpos.copy()
    base=np.asarray(home.geom_xpos[right],dtype=float).copy()
    data=mujoco.MjData(model)
    hip="r_hip_pitch"; knee="r_knee"
    rows=[]
    for hip_deg in _joint_samples(model,hip,address,grid_step_deg):
        for knee_deg in _joint_samples(model,knee,address,grid_step_deg):
            data.qpos[:]=q0
            data.qpos[address[hip]]=np.deg2rad(hip_deg)
            data.qpos[address[knee]]=np.deg2rad(knee_deg)
            mujoco.mj_forward(model,data)
            pos=np.asarray(data.geom_xpos[right],dtype=float)
            height=float((pos[2]-base[2])*1000.)
            advance=float((pos[0]-base[0])*1000.)
            rows.append(dict(
                hip_pitch_deg=round(float(hip_deg),3),
                knee_deg=round(float(knee_deg),3),
                toe_lift_mm=round(height,3),
                toe_forward_mm=round(advance,3),
                meets_geometry=(height>=height_mm and advance>=forward_mm),
            ))
    candidates=[row for row in rows if row["meets_geometry"]]
    # Minimize joint travel without claiming this is an optimal motor path.
    candidates.sort(key=lambda c: (
        abs(c["hip_pitch_deg"])+abs(c["knee_deg"]),
        -c["toe_lift_mm"]))
    return rows,candidates


def _floor_support(data,geoms):
    right,left,floor=geoms
    a=np.asarray(data.contact.geom1[:data.ncon],dtype=int)
    b=np.asarray(data.contact.geom2[:data.ncon],dtype=int)
    return bool(np.any(((a==floor)&(b==left)) |
                       ((a==left)&(b==floor))))


def drive_one_candidate(model,address,aid,geoms,candidate,*,height_mm,forward_mm,settle_s=.30,
                        ramp_s=.60,hold_s=.35):
    """Unmodified native floor + active capped servo dynamics.

    An intentionally crude open-loop controller, NOT a gait optimizer.
    If it falls or a servo saturates, that does not disprove feasible
    motion with learned balance, wind-up or other joint coordination.
    """
    right,left,_=geoms
    data=mujoco.MjData(model)
    dt=float(model.opt.timestep)
    # Root free, knee springs/live contacts: no fake root welding.
    def step(controls):
        for name,v in controls.items():
            data.ctrl[aid[name]]=v
        mujoco.mj_step(model,data)
    neutral={name:0. for name in ACTUATED_JOINTS}
    for _ in range(int(settle_s/dt)):
        step(neutral)
    before=np.asarray(data.geom_xpos[right],dtype=float).copy()
    torso_before=float(data.qpos[2])
    goal=dict(neutral)
    goal['r_hip_pitch']=float(np.deg2rad(candidate['hip_pitch_deg']))
    goal['r_knee']=float(np.deg2rad(candidate['knee_deg']))
    best_rel_z_mm=-float('inf')
    best_forward_mm=-float('inf')
    best_simul=None
    first_contact_frames=0
    target_seen_frames=0
    target_streak=0
    longest_target_streak=0
    measured=0
    clipped=0
    root_min=float(data.qpos[2])
    max_actuator_force=0.
    for frame in range(int((ramp_s+hold_s)/dt)):
        t=(frame+1)*dt
        u=np.clip(t/max(ramp_s,dt),0.,1.)
        blend=u*u*(3.-2.*u)
        step({name:goal[name]*blend for name in ACTUATED_JOINTS})
        force=np.asarray(data.actuator_force,dtype=float)
        max_actuator_force=max(max_actuator_force,float(np.max(np.abs(force))))
        clipped+=int(np.any(np.abs(force)>=.98*SERVO_RATED_CONTINUOUS_TORQUE_NM))
        measured+=1
        root_min=min(root_min,float(data.qpos[2]))
        contact=_floor_support(data,geoms)
        first_contact_frames+=int(contact)
        if contact:
            # This is BOTH relative toe height above a real left-foot
            # contact AND actual forward foot translation, same instant.
            lift=(float(data.geom_xpos[right,2]) -
                  float(data.geom_xpos[left,2]))*1000.
            advance=(float(data.geom_xpos[right,0])-float(before[0]))*1000.
            if best_simul is None or (lift>=best_simul[0] and
                                      advance>=best_simul[1]):
                best_simul=(lift,advance)
            best_rel_z_mm=max(best_rel_z_mm,lift)
            best_forward_mm=max(best_forward_mm,advance)
            within_target=(lift>=height_mm and advance>=forward_mm)
            target_seen_frames+=int(within_target)
            target_streak=target_streak+1 if within_target else 0
        else:
            target_streak=0
        longest_target_streak=max(longest_target_streak,target_streak)
    return dict(
        hip_pitch_deg=candidate['hip_pitch_deg'],
        knee_deg=candidate['knee_deg'],
        nominal_toe_lift_mm=candidate['toe_lift_mm'],
        nominal_toe_forward_mm=candidate['toe_forward_mm'],
        left_foot_ground_contact_frames=first_contact_frames,
        target_held_frames=target_seen_frames,
        longest_consecutive_target_frames=longest_target_streak,
        measured_frames=measured,
        max_relative_toe_lift_during_left_contact_mm=(
            round(best_rel_z_mm,2) if first_contact_frames else None),
        max_forward_toe_motion_during_left_contact_mm=(
            round(best_forward_mm,2) if first_contact_frames else None),
        simultaneous_lift_and_advance_sample=(
            [round(float(x),2) for x in best_simul] if best_simul else None),
        minimum_torso_height_m=round(root_min,4),
        torso_height_after_settle_m=round(torso_before,4),
        peak_actuator_torque_nm=round(max_actuator_force,6),
        frames_any_servo_near_rated_limit=clipped,
        note="Single-foot open-loop flat-floor motor trial, not a step climb",
    )


def run(*,output:Path,height_mm:float=50.,forward_mm:float=50.,
        grid_step_deg:float=5.,drive_candidates:int=5):
    if drive_candidates<0 or drive_candidates>40:
        raise ValueError("Reasonable diagnostic trial count is 0..40")
    output=output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output must be a new/empty directory")
    model,address,aid,geoms=physical_setup()
    reach,passers=reachability(
        model,address,geoms,height_mm=height_mm,
        forward_mm=forward_mm,grid_step_deg=grid_step_deg)
    trials=[drive_one_candidate(model,address,aid,geoms,c,
                                height_mm=height_mm,forward_mm=forward_mm)
            for c in passers[:drive_candidates]]
    output.mkdir(parents=True,exist_ok=True)
    with (output/"lift_kinematic_grid.csv").open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=list(reach[0]))
        wr.writeheader();wr.writerows(reach)
    report=dict(
        model=str(ROBOT_XML),height_request_mm=height_mm,
        forward_request_mm=forward_mm,grid_step_deg=grid_step_deg,
        actuator_hardcap_nm=SERVO_RATED_CONTINUOUS_TORQUE_NM,
        candidate_poses=len(passers),total_poses=len(reach),
        best_kinematic_candidates=passers[:15],
        native_flat_floor_motor_trials=trials,
        warning=("Kinematic poses assume fixed nominal chassis; dynamic "
                 "tests use a genuinely free root, capped servo forces and "
                 "real contact on flat ground but only simple open-loop "
                 "joint targets. No signed step, toe-over-wall contact, "
                 "loaded upper foothold or body-rise has been demonstrated."),
    )
    (output/"lift_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# H40 real-hardware-geometry lift feasibility","",
           f"Requested toe rise={height_mm:g}mm, forward motion={forward_mm:g}mm.",
           f"Real H40 actuators: force-hardcapped at "
           f"{SERVO_RATED_CONTINUOUS_TORQUE_NM:.6f}Nm.",
           f"Joint-limit forward kinematics: {len(passers)}/{len(reach)} "
           "tested joint configurations meet both offsets.","",
           "This is an **isolated geometry + unoptimized flat-floor motor**",
           "audit, NOT proof of genuine stair climbing.","",
           "| Knee | Hip pitch | Nominal lift | Nominal advance | "
           "Max measured lift during left contact | Simultaneous target frames |",
           "|---:|---:|---:|---:|---:|---:|"]
    for row in trials:
        lift=row["max_relative_toe_lift_during_left_contact_mm"]
        lines.append(
            f"| {row['knee_deg']:+g}° | {row['hip_pitch_deg']:+g}° | "
            f"{row['nominal_toe_lift_mm']:+.1f}mm | "
            f"{row['nominal_toe_forward_mm']:+.1f}mm | "
            f"{lift:.1f}mm | {row['target_held_frames']}/{row['measured_frames']} |"
            if lift is not None else
            f"| {row['knee_deg']:+g}° | {row['hip_pitch_deg']:+g}° | "
            f"{row['nominal_toe_lift_mm']:+.1f}mm | "
            f"{row['nominal_toe_forward_mm']:+.1f}mm | "
            f"No verified left-foot contact | 0/{row['measured_frames']} |")
    if not trials:
        lines.append("| No satisfying joint targets at this grid resolution | — | — | — | — | — |")
    lines+=["","Full joint grid: lift_kinematic_grid.csv.",
             "Peak torque, limit saturation, servo feedback and chassis height:",
             "lift_summary.json.","",
             "Next: only if useful, test scripted loaded foothold and",
             "body-rise against the actual signed prism wall (not this floor)."]
    (output/"lift_summary.md").write_text("\n".join(lines)+"\n")
    print(f"H40 lift geometry: {len(passers)}/{len(reach)} candidates, "
          f"{len(trials)} motor trials; {output/'lift_summary.md'}",flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--height-mm",type=float,default=50.)
    p.add_argument("--forward-mm",type=float,default=50.)
    p.add_argument("--grid-step-deg",type=float,default=5.)
    p.add_argument("--drive-candidates",type=int,default=5)
    a=p.parse_args()
    run(output=a.output,height_mm=a.height_mm,
        forward_mm=a.forward_mm,grid_step_deg=a.grid_step_deg,
        drive_candidates=a.drive_candidates)


if __name__=="__main__":
    main()
