"""v5051 bounded non-learning standing feasibility *screen* on robot.xml.

This is a simplified independent controller on the robot's flat plane, not
the same randomized MJX carpet and not a proof of physical impossibility.
Failed PD gains should motivate optimisation, NOT changes to PPO rewards.

Run: python -m scripts.v5051_balance_screen --output ~/emu-data/runs/v5051-balance-s735
"""
from __future__ import annotations
import argparse
import json
import math
from collections import deque
from pathlib import Path
import numpy as np

from emu.contracts import ROBOT_XML, CONTROL_DT, CONTROL_SUBSTEPS


def run_case(model, *, gain, damping, delay_frames, noise_rad, seed, horizon_s=10.):
    import mujoco
    rng=np.random.default_rng(seed)
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    joint_names=("r_hip_roll","r_hip_pitch","r_knee",
                 "l_hip_roll","l_hip_pitch","l_knee")
    act=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
                  for n in joint_names],int)
    qids=np.array([model.jnt_qposadr[i] for i in act])
    r_foot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"r_foot_ball")
    l_foot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"l_foot_ball")
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    if min(r_foot,l_foot,floor)<0:
        raise RuntimeError("Robot flat support geometry changed")
    queue=deque(maxlen=delay_frames+1)
    mujoco.mj_forward(model,data)
    def pose():
        q=data.qpos[3:7]
        mat=np.zeros(9)
        mujoco.mju_quat2Mat(mat,q)
        r=mat.reshape((3,3))
        roll=math.atan2(r[2,1],r[2,2])
        pitch=math.atan2(-r[2,0],math.hypot(r[2,1],r[2,2]))
        return roll,pitch
    frames=int(horizon_s/CONTROL_DT)
    logged=[]
    first_failure=None
    max_torque=0.
    for frame in range(frames):
        roll,pitch=pose()
        queue.append((roll,pitch,float(data.qvel[3]),float(data.qvel[4])))
        obs=queue[0] if len(queue)==delay_frames+1 else queue[0]
        rr,pp,wx,wy=obs
        rr+=rng.normal(0.,noise_rad)
        pp+=rng.normal(0.,noise_rad)
        # Independent bounded torso attitude feedback. Servo position
        # targets, not an ideal external torso force or gravity override.
        # Negative and positive gain effects both tested by sweep.
        desired=np.zeros(6)
        desired[[0,3]]=np.clip(-gain*rr-damping*wx,-.35,.35)
        desired[[1,4]]=np.clip(-gain*pp-damping*wy,-.65,.65)
        data.ctrl[:6]=np.clip(desired,model.actuator_ctrlrange[:6,0],
                                      model.actuator_ctrlrange[:6,1])
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model,data)
        max_torque=max(max_torque,float(np.max(np.abs(data.actuator_force[:6]))))
        support={r_foot:False,l_foot:False}
        for contact in data.contact:
            if contact.dist > .003:
                continue
            if contact.geom1==floor and contact.geom2 in support:
                support[contact.geom2]=True
            if contact.geom2==floor and contact.geom1 in support:
                support[contact.geom1]=True
        roll,pitch=pose()
        healthy=(np.isfinite(data.qpos).all() and data.qpos[2]>.145
                 and abs(roll)<1.2 and abs(pitch)<1.2)
        logged.append(dict(frame=frame,root_z=float(data.qpos[2]),
                           roll_rad=roll,pitch_rad=pitch,
                           feet_supported=sum(support.values()),
                           torso_drift_xy_m=float(np.linalg.norm(data.qpos[:2])),
                           max_servo_torque_nm=max_torque))
        if not healthy:
            first_failure=frame
            break
    duration=len(logged)*CONTROL_DT
    return dict(gain=gain,damping=damping,delay_frames=delay_frames,
                noise_rad=noise_rad,seed=seed,
                survived_10_s=(first_failure is None and len(logged)==frames),
                duration_s=round(duration,3),
                max_torque_nm=round(max_torque,4),
                final_drift_m=round(logged[-1]["torso_drift_xy_m"],4),
                mean_supported_feet=round(float(np.mean(
                    [v["feet_supported"] for v in logged])),3),
                final_roll_rad=round(logged[-1]["roll_rad"],4),
                final_pitch_rad=round(logged[-1]["pitch_rad"],4))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--quick",action="store_true",help="small integration smoke")
    a=p.parse_args()
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    model.opt.timestep=.005
    if not np.allclose(np.max(np.abs(model.actuator_forcerange),axis=1),1.913):
        raise RuntimeError("Unexpected physical motor caps")
    settings=([(0.,0.,0,0.),(.5,.1,1,.005)] if a.quick else
              [(gain,damping,delay,noise)
               for gain,damping in ((0.,0.),(.25,.04),(.50,.08),
                                    (1.,.12),(2.,.2),(4.,.3))
               for delay in (0,1,2)
               for noise in (0.,.008)])
    results=[run_case(model,gain=g,damping=d,delay_frames=delay,
                      noise_rad=noise,seed=a.seed)
             for g,d,delay,noise in settings]
    dest=a.output.expanduser().resolve()
    dest.mkdir(parents=True,exist_ok=True)
    target=dest/"balance_screen.json"
    if target.exists():
        raise FileExistsError(f"Do not overwrite balance evidence: {target}")
    target.write_text(json.dumps(dict(
        schema="emu_v5051_balance_screen_v1",
        model=str(ROBOT_XML),num_cases=len(results),
        note="Flat-plane bounded PD SCREEN. Failure does not prove "
             "stationary locomotion physically impossible; does not use "
             "the full MJX prism terrain or learner noise/speed controller.",
        cases=results),indent=2,allow_nan=False)+"\n")
    for case in results:
        print(f"K={case['gain']:.2f} D={case['damping']:.2f} "
              f"delay={case['delay_frames']} noise={case['noise_rad']:.3f} "
              f"survived={case['survived_10_s']} "
              f"duration={case['duration_s']:.2f}s")
    print(f"Saved {target}")


if __name__=="__main__":
    main()
