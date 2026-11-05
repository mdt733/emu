"""v5051 CPU-only supported single-foot swing *feasibility screen*.

Uses unchanged robot.xml, real MuJoCo floor/foot collisions and force-limited
actuators. Not a PPO gait or a step-climb proof. Open-loop setpoint sweep
ramped to nominal no-load angular velocity. A supported clearance means the
opposite ball foot physically contacted the floor during that frame.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from emu.contracts import ROBOT_XML, CONTROL_DT, CONTROL_SUBSTEPS


def trial(model, *, foot, pitch, knee, seed):
    import mujoco
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    mujoco.mj_forward(model,data)
    moving="r" if foot=="right" else "l"
    stance="l" if foot=="right" else "r"
    hip_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,moving+"_hip_pitch")
    knee_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,moving+"_knee")
    swing_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,moving+"_foot_ball")
    stance_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,stance+"_foot_ball")
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    if min(hip_id,knee_id,swing_geom,stance_geom,floor)<0:
        raise RuntimeError("Robot/actuator naming changed")
    radii=float(model.geom_size[swing_geom,0])
    initial=data.geom_xpos[swing_geom].copy()
    target=np.zeros(6)
    target[hip_id]=pitch
    target[knee_id]=knee
    delivery=np.zeros(6)
    max_clear=-1.
    forward_at_max=None
    support_frames=0
    torque_peak=0.
    fell=False
    # 2s screen: initially settle, then swing; not an ideal forced hold.
    for frame in range(100):
        if frame >= 15:
            diff=np.clip(target-delivery,
                         -4.4*CONTROL_DT,4.4*CONTROL_DT)
            delivery+=diff
        data.ctrl[:6]=np.clip(delivery,
                             model.actuator_ctrlrange[:6,0],
                             model.actuator_ctrlrange[:6,1])
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model,data)
        torque_peak=max(torque_peak,float(np.max(np.abs(data.actuator_force[:6]))))
        support=False
        for con in data.contact:
            if con.dist>.003:
                continue
            if ((con.geom1==floor and con.geom2==stance_geom) or
                (con.geom2==floor and con.geom1==stance_geom)):
                support=True
                break
        if support:
            support_frames+=1
            h=float(data.geom_xpos[swing_geom,2]-radii)
            if h>max_clear:
                max_clear=h
                forward_at_max=float(data.geom_xpos[swing_geom,0]-initial[0])
        if (not np.isfinite(data.qpos).all() or data.qpos[2]<.145 or
            abs(float(data.qpos[0]-data.qpos[0]))>5.):
            fell=True
            break
    return dict(foot=foot,hip_pitch_rad=pitch,knee_rad=knee,
                opposite_foot_support_fraction=round(support_frames/(frame+1),3),
                maximum_supported_clearance_m=(round(max_clear,5) if max_clear>=0 else None),
                forward_swing_placement_at_peak_m=(round(forward_at_max,5)
                                                   if forward_at_max is not None else None),
                torque_peak_nm=round(torque_peak,4),
                fell=fell,completed_frames=frame+1,
                note="One foot's ground contact is not proof of sufficient load "
                     "support or full CoM transfer.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    pitches=(0.,-.25,.25) if a.quick else (-.4,-.2,0.,.2,.4)
    knees=(.25,.5) if a.quick else (0.,.2,.4,.6,.8)
    results=[trial(model,foot=foot,pitch=pitch,knee=knee,seed=a.seed)
             for foot in ("right","left") for pitch in pitches for knee in knees]
    out=a.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    dest=out/"swing_screen.json"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    dest.write_text(json.dumps(dict(
        schema="emu_v5051_swing_screen_v1",
        physics="robot.xml flat-plane MuJoCo feet and actuators",
        cases=results,
        note="Open-loop motor-position sweep with <=4.4rad/s target rate. "
             "Static floor, no wall and no simulated ToF. A positive value "
             "shows supported geometric lift in this screen, not a climb. "
             "A negative result is not proof of mechanical impossibility."),
        indent=2,allow_nan=False)+"\n")
    valid=[r for r in results if r["maximum_supported_clearance_m"] is not None]
    print(f"Cases {len(results)}; opposite-foot-supported cases {len(valid)}")
    for r in sorted(valid,key=lambda x:x["maximum_supported_clearance_m"],reverse=True)[:8]:
        print(r)
    print(f"Saved {dest}")


if __name__=="__main__":
    main()
