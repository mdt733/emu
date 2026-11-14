"""Independent, READ-ONLY two-lip physical examiner for frozen H40 policies.

Uses raw MJX prism/foot contact pairs, surface corners and actual trajectories,
NEVER the curriculum's reward, milestone or academy-success flags. Pure NumPy:
no simulator, no PPO, no training, safe to exercise with synthetic traces.
"""
from __future__ import annotations

import numpy as np

from .contracts import CONTROL_DT
from .prism_exam import score_prism_crossings
from .step_autopsy import _top

MAX_TORQUE_NM = 0.980665


def _run5(values):
    """Whether five consecutive physical frames satisfy a contact predicate."""
    v = np.asarray(values, dtype=bool)
    return bool(len(v)>=5 and np.any(np.convolve(v.astype(np.int32),
                                                   np.ones(5,dtype=np.int32),
                                                   mode="valid")>=5))


def _first5(values):
    v = np.asarray(values,dtype=bool)
    streak=0
    for t,present in enumerate(v):
        streak=streak+1 if present else 0
        if streak>=5:
            return t
    return -1


def _moving_window_good(values,desired,*,width=20,tol=.10):
    """Finish frame of first truly commanded 0.4s vx window, -1 if none."""
    a=np.asarray(values,float)
    c=np.asarray(desired,float)
    if len(a)<width:return -1
    for end in range(width-1,len(a)):
        v=float(np.mean(a[end+1-width:end+1]))
        d=float(np.mean(c[end+1-width:end+1]))
        if d>=.075 and v>=.05 and abs(v-d)<=tol:
            return end
    return -1


def _physical_top_contacts(frames,w,stop,target):
    """[frame,foot] authentic upward-facing convex-prism target support."""
    z=np.asarray(frames["carpet_prism_corner_z"],float)
    pairs=np.asarray(frames["carpet_raw_contact_pairs"],bool)
    normals=np.asarray(frames["carpet_contact_vertical"],float)
    feet=np.asarray(frames["foot_xyz"],float)
    rad=np.asarray(frames["initial_foot_radii_m"],float)
    slot=target[0]%2+2*(target[1]%2)
    ok=np.zeros((stop,2),dtype=bool)
    for t in range(stop):
        for f in range(2):
            contact=np.any(pairs[t,w,2*slot:2*slot+2,f] &
                           (normals[t,w,2*slot:2*slot+2,f]>=.70))
            if not contact:continue
            top=_top(z[t,w,slot],target,feet[t,w,f,:2])
            if top is not None and abs(feet[t,w,f,2]-rad[f]-top)<=.009:
                ok[t,f]=True
    return ok


def score_two_lip_trace(frames, *, label, actor_sha, seed):
    """Per-world top-contact and motion funnel, plus illustrative trace excerpts.

    The first lip is always +X x=1m and the second +X x=2m. Both are
    generated from the source's genuine eight-prism Academy lesson and
    may be signed UP or DOWN. Report TRUE 5-frame simultaneous recovery
    separately from each foot having supported at different times.

    Every outcome is restricted to the first life. A failure on a pre-step
    MJX contact sample is never overwritten by the following reset.
    """
    crossings=score_prism_crossings(frames)
    xy=np.asarray(frames["position_xy"],float)
    yaw=np.asarray(frames["heading_rad"],float)
    feet=np.asarray(frames["foot_xyz"],float)
    radii=np.asarray(frames["initial_foot_radii_m"],float)
    corner=np.asarray(frames["carpet_prism_corner_z"],float)
    done=np.asarray(frames["done"],bool)
    fell=np.asarray(frames["fell"],bool)
    vz=np.asarray(frames["terrain_vx"],float)
    cmd=np.asarray(frames["command"],float)
    gravity=np.asarray(frames["terrain_gravity_z"],float)
    support=np.asarray(frames["terrain_support"],float)
    bad=np.asarray(frames["terrain_bad_contact"],float)
    torque=np.asarray(frames["servo_torque_nm"],float)
    power=np.asarray(frames["servo_shaft_drive_power_w"],float)
    tof=np.asarray(frames.get("tof_valid_zones",np.full(done.shape,-1)),float)
    tmax,worlds=done.shape
    if len(crossings['worlds'])!=worlds:
        raise ValueError('Independent prism scorer world count changed')
    rows=[]
    snippets=[]
    for w in range(worlds):
        term=np.flatnonzero(done[:,w])
        stop=int(term[0]+1) if len(term) else tmax
        root=xy[:stop,w]
        contacted=[]
        for target in ((1,0),(2,0)):
            ok=_physical_top_contacts(frames,w,stop,target)
            seen=tuple(_first5(ok[:,f]) for f in range(2))
            simultaneous=_first5(np.all(ok,axis=1))
            contacted.append((ok,seen,simultaneous))
        (f1,ft1,sim1),(f2,ft2,sim2)=contacted
        both1=all(t>=0 for t in ft1)
        both2=all(t>=0 for t in ft2)
        lead1=int(np.flatnonzero(np.any(f1,axis=1))[0]) if np.any(f1) else -1
        lead2=int(np.flatnonzero(np.any(f2,axis=1))[0]) if np.any(f2) else -1
        before_first=bool(np.any(np.any(f1,axis=1)&(root[:,0]<1.)))
        before_second=bool(np.any(np.any(f2,axis=1)&(root[:,0]<2.)))
        clearance=[-1.,-1.]
        penetrated=[False,False]
        # Geometry is measured at the lip, not inferred from actor rewards.
        for idx,edge in enumerate((1.,2.)):
            target=(idx+1,0)
            slot=target[0]%2
            for t in range(stop):
                if not edge-.25<=root[t,0]<=edge+.10:continue
                for foot in range(2):
                    if not (0.<=feet[t,w,foot,1]<1.):continue
                    top=_top(corner[t,w,slot],target,
                             np.array([edge,feet[t,w,foot,1]]))
                    if top is None:continue
                    toe=feet[t,w,foot,2]-radii[foot]
                    clearance[idx]=max(clearance[idx],float(toe-top))
                    if feet[t,w,foot,0]>edge+.005 and toe>top+.005:
                        penetrated[idx]=True
        first_progress=bool(both1 and np.any(root[max(ft1):,0]>=1.5))
        second_progress=bool(both2 and np.any(root[max(ft2):,0]>=2.5))
        support_recovery=False
        vx20=False
        heading_good=False
        certified=False
        if both2:
            start=max(ft2)
            # Independent physical post-second stage; can be met later,
            # not necessarily on the same frame as first progress.
            for t in range(max(start,19),stop):
                if root[t,0]<2.5:continue
                support_recovery=True
                recent=slice(t-19,t+1)
                v=float(np.mean(vz[recent,w]))
                c=float(np.mean(cmd[recent,w,0]))
                good=(c>=.075 and v>=.05 and abs(v-c)<=.10)
                vx20|=good
                turn=float(np.arctan2(
                    np.sin(yaw[t,w]-yaw[0,w]),np.cos(yaw[t,w]-yaw[0,w])))
                aligned=abs(turn)<=.35
                heading_good|=aligned
                if good and gravity[t,w]>=.8 and support[t,w]>=.5 and bad[t,w]<.5:
                    certified=True
                    break
        first_events=[e for e in crossings['worlds'][w]['independently_supported_lips']
                      if e['from_cell']==[0,0] and e['to_cell']==[1,0]]
        second_events=[e for e in crossings['worlds'][w]['independently_supported_lips']
                       if e['from_cell']==[1,0] and e['to_cell']==[2,0]]
        first_kind=(float(first_events[0]['signed_height_cm']) if first_events
                    else None)
        second_kind=(float(second_events[0]['signed_height_cm']) if second_events
                     else None)
        hit_first=int(np.flatnonzero(root[:,0]>=1.)[0]) if np.any(root[:,0]>=1.) else -1
        hit_second=int(np.flatnonzero(root[:,0]>=2.)[0]) if np.any(root[:,0]>=2.) else -1
        if not np.any(f1):failure="no_first_top_contact"
        elif not both1:failure="first_trailing_foot_not_verified"
        elif not np.any(root[:,0]>=2.):failure="never_reached_second_seam"
        elif not np.any(f2):failure="no_second_top_contact"
        elif not both2:failure="second_trailing_foot_not_verified"
        elif not second_progress:failure="no_50cm_after_second"
        elif not vx20:failure="no_valid_recent_vx"
        elif not certified:failure="poststep_unhealthy"
        else:failure="certified_by_independent_course"
        measured_progress=float(max(0.,np.max(root[:,0]-root[0,0])))
        total_work=float(np.maximum(0.,power[:stop,w]).sum()*CONTROL_DT)
        near1=(root[:,0]>=.72)&(root[:,0]<=1.10)
        near2=(root[:,0]>=1.70)&(root[:,0]<=2.10)
        valid_torque=torque[:stop,w]
        rows.append(dict(
            protocol=label,seed=seed,actor_sha=actor_sha,world=w,
            first_life_frames=stop,fell=bool(np.any(fell[:stop,w])),
            cmd_vx=float(cmd[0,w,0]),cmd_vy=float(cmd[0,w,1]),
            cmd_yaw=float(cmd[0,w,2]),birth_yaw=float(yaw[0,w]),
            first_signed_cm=first_kind,second_signed_cm=second_kind,
            first_root_cross=hit_first>=0,second_root_cross=hit_second>=0,
            first_root_frame=hit_first,second_root_frame=hit_second,
            first_lead_frame=lead1,second_lead_frame=lead2,
            first_foot0_5frame=ft1[0]>=0,first_foot1_5frame=ft1[1]>=0,
            second_foot0_5frame=ft2[0]>=0,second_foot1_5frame=ft2[1]>=0,
            first_sequential_both=both1,first_simultaneous_both=sim1>=0,
            second_sequential_both=both2,second_simultaneous_both=sim2>=0,
            first_upper_contact_before_root=before_first,
            second_upper_contact_before_root=before_second,
            first_clearance_relative_top_m=clearance[0],
            second_clearance_relative_top_m=clearance[1],
            first_toe_over_lip=penetrated[0],
            second_toe_over_lip=penetrated[1],
            half_m_after_first=first_progress,
            half_m_after_second=second_progress,
            after_second_recent_vx_ok=vx20,
            after_second_heading_aligned=heading_good,
            course_healthy_commanded=certified,
            independent_first_supported_event=bool(first_events),
            independent_second_supported_event=bool(second_events),
            max_forward_progress_m=measured_progress,
            mean_tof_valid_first=float(np.mean(tof[:stop,w][near1]))
                                 if np.any(near1) else None,
            mean_tof_valid_second=float(np.mean(tof[:stop,w][near2]))
                                  if np.any(near2) else None,
            max_abs_servo_nm=float(np.max(np.abs(valid_torque))),
            torque_over_rated_fraction=float(np.mean(
                np.abs(valid_torque)>MAX_TORQUE_NM+1e-5)),
            torque_near_rated_fraction=float(np.mean(
                np.abs(valid_torque)>.90*MAX_TORQUE_NM)),
            shaft_positive_work_j=total_work,
            shaft_j_per_forward_m=(total_work/measured_progress
                                  if measured_progress>.02 else None),
            failure_stage=failure))
        # Bounded event-level examples, not giant [1000,128,contacts] dumps.
        # The full numeric funnel remains in the per-world CSV.
        if w<3 or (certified and w<20) or (both2 and w<16):
            indices=set(range(min(12,stop)))
            for event_t in (hit_first,hit_second,lead1,lead2):
                if event_t>=0:
                    indices.update(range(max(0,event_t-8),
                                         min(stop,event_t+13)))
            for t in sorted(indices)[:100]:
                snippets.append(dict(
                    protocol=label,seed=seed,world=w,frame=t,
                    x=float(root[t,0]),y=float(root[t,1]),yaw=float(yaw[t,w]),
                    toe0_x=float(feet[t,w,0,0]),
                    toe0_z=float(feet[t,w,0,2]-radii[0]),
                    toe1_x=float(feet[t,w,1,0]),
                    toe1_z=float(feet[t,w,1,2]-radii[1]),
                    first0_top=bool(f1[t,0]),first1_top=bool(f1[t,1]),
                    second0_top=bool(f2[t,0]),second1_top=bool(f2[t,1]),
                    vx=float(vz[t,w]),cmd_vx=float(cmd[t,w,0]),
                    tof_valid_zones=float(tof[t,w])))
    return rows,snippets
