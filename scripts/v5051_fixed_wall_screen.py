"""v5051 actual-height fixed vertical wall CPU feasibility SCREEN.

Creates a new MuJoCo XML from *current robot.xml* with a real colliding
box ledge. One foot is commanded through a bounded servo-position sweep while
the other supports the robot. No external force, collision disable, or scripted
RL gait. It is deliberately NOT a trained actor examination: a failed open-loop
probe cannot establish mechanical impossibility.
"""
from __future__ import annotations
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from emu.contracts import ROBOT_XML, CONTROL_DT, CONTROL_SUBSTEPS


def make_wall_model(height_m):
    import mujoco
    if not .005 <= height_m <= .045:
        raise ValueError("Actual wall must be 0.5..4.5 cm")
    root=ET.parse(str(ROBOT_XML)).getroot()
    world=root.find("worldbody")
    if world is None:
        raise ValueError("Robot XML missing worldbody")
    # Solid begins at x=+.24m; box extends to +1.24m.
    # Top is exactly height_m above a zero-elevation floor.
    ET.SubElement(world,"geom",name="v5051_fixed_wall",
                  type="box",pos=f"0.74 0 {height_m/2:.8f}",
                  size=f"0.5 0.5 {height_m/2:.8f}",
                  rgba="0.9 0.65 0.1 1",friction="1.5 0.005 0.0001")
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    wall_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                             "v5051_fixed_wall")
    if wall_id < 0:
        raise RuntimeError("Wall geom absent after compilation")
    return model,wall_id


def test_swing(model,wall_id,height,side,pitch,knee,*,settle=15,frames=150):
    import mujoco
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    # Start close to a known riser, with both feet below the wall.
    data.qpos[0]=.14
    mujoco.mj_forward(model,data)
    limb="r" if side=="right" else "l"
    other="l" if side=="right" else "r"
    swing_act=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,
                               limb+"_hip_pitch")
    knee_act=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,limb+"_knee")
    swing_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                                limb+"_foot_ball")
    stance_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                                 other+"_foot_ball")
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    rad=float(model.geom_size[swing_geom,0])
    desired=np.zeros(6)
    desired[swing_act]=pitch
    desired[knee_act]=knee
    delivered=np.zeros(6)
    max_supported_lift=-math.inf
    peak_torque=0.
    riser_contact_frames=0
    upper_contact_frames=0
    top_contact_streak=0
    best_streak=0
    stance_contact_frames=0
    clear_over_riser=False
    fell=False
    for t in range(frames):
        if t>=settle:
            delivered+=np.clip(desired-delivered,
                               -4.40*CONTROL_DT,4.40*CONTROL_DT)
        data.ctrl[:6]=np.clip(delivered,
                             model.actuator_ctrlrange[:6,0],
                             model.actuator_ctrlrange[:6,1])
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model,data)
        peak_torque=max(peak_torque,
                        float(np.max(np.abs(data.actuator_force[:6]))))
        on_floor=False
        on_wall=False
        for c in data.contact:
            if c.dist>.003:
                continue
            if ({c.geom1,c.geom2}=={floor,stance_geom}):
                on_floor=True
            if ({c.geom1,c.geom2}=={wall_id,swing_geom}):
                # Contact on vertical side doesn't count as upper support.
                # MuJoCo's contact frame stores the contact normal
                # in its first three world-space components.
                normal=np.asarray(c.frame)[:3]
                if abs(float(normal[2]))>=.7:
                    on_wall=True
                else:
                    riser_contact_frames+=1
        if on_floor:
            stance_contact_frames+=1
            xyz=data.geom_xpos[swing_geom]
            clearance=float(xyz[2]-rad)
            max_supported_lift=max(max_supported_lift,clearance)
            if xyz[0] >= .24+rad and clearance>=height-.003:
                clear_over_riser=True
        if on_wall and on_floor:
            upper_contact_frames+=1
            top_contact_streak+=1
            best_streak=max(best_streak,top_contact_streak)
        else:
            top_contact_streak=0
        if (not np.isfinite(data.qpos).all() or data.qpos[2]<.145 or
            np.linalg.norm(data.qpos[:2])>.60):
            fell=True
            break
    return dict(side=side,hip_pitch_rad=pitch,knee_rad=knee,
                actual_height_cm=round(height*100,4),
                stance_floor_support_fraction=round(stance_contact_frames/(t+1),3),
                max_supported_foot_clearance_cm=(
                    round(max_supported_lift*100,4)
                    if math.isfinite(max_supported_lift) else None),
                passed_foot_centre_over_riser=bool(clear_over_riser),
                upper_contact_while_opposite_supported_frames=upper_contact_frames,
                upper_contact_streak_frames=best_streak,
                verified_upper_placement=(best_streak>=5),
                riser_side_collision_frames=riser_contact_frames,
                max_servo_torque_nm=round(peak_torque,4),fell=bool(fell))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    heights=(.01,.025,.03) if a.quick else (.01,.02,.025,.03,.035,.04)
    pitches=(-.3,0.,.3) if a.quick else (-.4,-.2,0.,.2,.4)
    knees=(.3,.6) if a.quick else (.2,.4,.6,.8)
    rows=[]
    for height in heights:
        model,wall_id=make_wall_model(height)
        for side in ("right","left"):
            for pitch in pitches:
                for knee in knees:
                    rows.append(test_swing(model,wall_id,height,side,pitch,knee))
        print(f"Fixed {height*100:g}cm: "
              f"{sum(r['verified_upper_placement'] for r in rows if r['actual_height_cm']==round(height*100,4))} "
              "verified upper-contact probes",flush=True)
    out=a.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    target=out/"fixed_wall_screen.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite fixed-wall evidence")
    result=dict(schema="emu_v5051_fixed_wall_screen_v1",cases=rows,
                definitions=dict(wall_front_x_m=.24,actual_height_list_m=list(heights),
                                 upper_placement="five consecutive real top-ish contact "
                                 "frames while opposite ball foot is floor supported"),
                caveats=[
                  "Open-loop feasibility sweep, NOT a learned climb or a full weight transfer",
                  "Thigh/shin collision is disabled in base robot.xml and therefore also here",
                  "Contact normal uses MuJoCo geometry; must be checked against precise face geometry",
                  "Failure of this limited sweep does NOT prove impossibility",
                  "No repeated-step gait or continuous post-wall motion is tested"])
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Saved {target}; upper placements="
          f"{sum(r['verified_upper_placement'] for r in rows)}/{len(rows)}")


if __name__=="__main__":
    main()
