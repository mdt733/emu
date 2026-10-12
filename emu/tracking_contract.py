#!/usr/bin/env python3
"""Single source of truth for EMU locomotion competence thresholds.

The live field, held-out node exam and recovery exam may use different temporal
questions, but they share these physical tolerances and posture requirements.
"""
from __future__ import annotations
import math
import numpy as np
from .contracts import CONTROL_DT

VX_ABS_TOL=0.08; VX_REL_TOL=0.25
VY_ABS_TOL=0.07; VY_REL_TOL=0.35
YAW_ABS_TOL=0.25; YAW_REL_TOL=0.25
UPRIGHT_GRAVITY_Z_MIN=math.cos(math.radians(20.0))
SUPPORT_MIN=0.05
BAD_CONTACT_FRACTION_MAX=0.10
NODE_WINDOW_S=1.0
NODE_SETTLE_DELAY_S=0.5
NODE_STRUCTURAL_FRACTION_MIN=0.90
RECOVERY_STABLE_S=0.24
EDGE_STABLE_S=0.24

# Physical-frontier measurement is deliberately tighter than node mastery.
# These physical-frontier tolerances belong with measurement semantics,
# not as attributes on the curriculum controller that consumes the evidence.
FRONTIER_VX_ABS_TOL=0.05; FRONTIER_VX_REL_TOL=0.12
FRONTIER_VY_ABS_TOL=0.035; FRONTIER_VY_REL_TOL=0.18
FRONTIER_YAW_ABS_TOL=0.12; FRONTIER_YAW_REL_TOL=0.12
FRONTIER_EDGE_FRAC=0.72
FRONTIER_SETTLE_S=0.35


def dynamic_tolerance(cmd,abs_floor:float,rel_frac:float):
    return np.maximum(float(abs_floor),float(rel_frac)*np.abs(np.asarray(cmd)))


def tracking_tolerances(vx_cmd,vy_cmd,yaw_cmd):
    return (dynamic_tolerance(vx_cmd,VX_ABS_TOL,VX_REL_TOL),
            dynamic_tolerance(vy_cmd,VY_ABS_TOL,VY_REL_TOL),
            dynamic_tolerance(yaw_cmd,YAW_ABS_TOL,YAW_REL_TOL))


def node_window_success_numpy(cmd_vx,cmd_vy,cmd_yaw,vx,vy,yaw_rate,gravity_z,support,bad_contact,fell=None,axis=0):
    """Reference NumPy definition used by tests and offline diagnostics.

    Tracking uses mean absolute error, not abs(mean(error)); gait oscillation is
    therefore tolerated without cancellation hiding genuine error.
    """
    cvx,cvy,cy=[np.asarray(x) for x in (cmd_vx,cmd_vy,cmd_yaw)]
    vx,vy,yw=[np.asarray(x) for x in (vx,vy,yaw_rate)]
    tx,ty,tz=tracking_tolerances(cvx,cvy,cy)
    good=(np.mean(np.abs(vx-cvx),axis=axis)<=np.mean(tx,axis=axis)) & \
         (np.mean(np.abs(vy-cvy),axis=axis)<=np.mean(ty,axis=axis)) & \
         (np.mean(np.abs(yw-cy),axis=axis)<=np.mean(tz,axis=axis))
    structural=(np.mean(np.asarray(gravity_z)>UPRIGHT_GRAVITY_Z_MIN,axis=axis)>=NODE_STRUCTURAL_FRACTION_MIN) & \
               (np.mean(np.asarray(support)>SUPPORT_MIN,axis=axis)>=NODE_STRUCTURAL_FRACTION_MIN) & \
               (np.mean(np.asarray(bad_contact)>.5,axis=axis)<=BAD_CONTACT_FRACTION_MAX)
    if fell is not None:
        structural &= ~np.any(np.asarray(fell)>.5,axis=axis)
    return good & structural


def recovery_outcome_numpy(first_fall_step, first_recovery_step, *, not_before_steps: int = 0, direction_bins: int = 0):
    """Shared held-out recovery verdict/reduction.

    A recovery only counts when the recovery event occurred and the episode did
    not later fall. ``not_before_steps`` removes deliberate probe application
    time from the reported recovery latency. Optional direction bins expose the
    worst stratified nuisance direction without changing the aggregate verdict.
    """
    ff=np.asarray(first_fall_step)
    fr=np.asarray(first_recovery_step)
    if ff.shape != fr.shape:
        raise ValueError(f'fall/recovery shape mismatch: {ff.shape} vs {fr.shape}')
    survived=ff < 0
    sustained=(fr >= 0) & survived
    times=(fr[sustained]-int(not_before_steps))*CONTROL_DT
    out={
        'survival':float(survived.mean()) if survived.size else float('nan'),
        'recovery_rate':float(sustained.mean()) if sustained.size else float('nan'),
        'recovery_time_p90_s':float(np.percentile(times,90)) if times.size else -1.0,
    }
    if direction_bins > 0:
        bins=np.arange(ff.size,dtype=np.int32)%int(direction_bins)
        rates=[float(sustained[bins==b].mean()) for b in range(int(direction_bins)) if np.any(bins==b)]
        out['recovery_worst_direction']=float(np.min(rates)) if rates else float('nan')
    return out
