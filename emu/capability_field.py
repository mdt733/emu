#!/usr/bin/env python3
"""Continuous, event-based capability field for EMU.

The field keeps compact command geometry and semantic event-based evidence:

* NODE competence is a decayed probability of succeeding at a meaningful,
  sustained command window.  Continuous tracking/upright/support quality is
  recorded separately and never masquerades as a Bernoulli success.
* RECOVERY competence is one verdict per integrated resilience birth: did the
  policy return to competent locomotion and sustain it, or fail first?
* EDGE competence is one verdict per complete command maneuver: did the robot
  survive the requested transition and settle onto the new command?

The field is deliberately host-side.  PPO already transfers its rollout metric
pytree to the host once per update; this module reuses that transfer and never
adds per-death callbacks or an extra simulator pass.  Held-out evaluator
results remain quarantined and NEVER update this object.

Field schema 6 requires gait-window mean tracking error, treats one-frame
maneuvers as first-class events, and persists only evidence modern teaching or
visualization actually consumes. Incompatible maps are rejected on resume.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from .contracts import CONTROL_DT, FIELD_SHAPE, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS, FIELD_EDGE_SCALES
from .tracking_contract import (VX_ABS_TOL,VX_REL_TOL,VY_ABS_TOL,VY_REL_TOL,YAW_ABS_TOL,YAW_REL_TOL,UPRIGHT_GRAVITY_Z_MIN,SUPPORT_MIN,NODE_WINDOW_S,NODE_SETTLE_DELAY_S,NODE_STRUCTURAL_FRACTION_MIN,RECOVERY_STABLE_S,EDGE_STABLE_S)

FIELD_VERSION = 6

FIELD_BOUNDS = ((-0.65, 1.20), (-0.65, 0.65), (-5.0, 5.0))
RECOVERY_BINS = 6
DEFAULT_HALF_LIFE_STEPS = 20_000_000.0
RECOVERY_MIN_AGE_S = 0.16
SLOW_ALPHA = 0.015
MIN_EFFECTIVE_EVIDENCE = 3.0
STRONG_EFFECTIVE_EVIDENCE = 8.0
NODE_MASTER_LOWER = 0.80
RECOVERY_MASTER_LOWER = 0.70
EDGE_MASTER_LOWER = 0.70
STALE_HALF_LIVES = 1.0
NODE_ANCHORS = FIELD_NODE_ANCHORS
EDGE_ANCHORS = FIELD_EDGE_ANCHORS

# Twenty-six local directions on the 3-D command lattice.  These are direction
# sectors, not literal requirements that a maneuver end in an adjacent voxel.
EDGE_DIRS = np.asarray(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
     if not (x == 0 and y == 0 and z == 0)], dtype=np.float32)
EDGE_DIRS /= np.maximum(np.linalg.norm(EDGE_DIRS, axis=1, keepdims=True), 1e-6)

# Command normalization used only for maneuver direction/rate geometry.
EDGE_SCALES = np.asarray(FIELD_EDGE_SCALES, np.float32)



def _posterior_mean(success: np.ndarray, failure: np.ndarray) -> np.ndarray:
    """Beta(1,1) posterior mean from decayed event evidence."""
    return (1.0 + success) / np.maximum(2.0 + success + failure, 1e-6)


def _wilson_lower(success: np.ndarray, failure: np.ndarray, z: float = 1.2815515655) -> np.ndarray:
    """One-sided ~90% Wilson lower bound for possibly fractional counts.

    Exponential forgetting makes the effective counts fractional.  Wilson's
    algebra remains useful as a conservative confidence score even though the
    counts are effective evidence rather than IID integer observations.
    """
    s = np.asarray(success, np.float32)
    f = np.asarray(failure, np.float32)
    n = s + f
    p = np.divide(s, np.maximum(n, 1e-6))
    z2 = float(z * z)
    den = 1.0 + z2 / np.maximum(n, 1e-6)
    centre = p + z2 / (2.0 * np.maximum(n, 1e-6))
    rad = z * np.sqrt(np.maximum(0.0, p * (1.0 - p) / np.maximum(n, 1e-6)
                                  + z2 / (4.0 * np.maximum(n, 1e-6) ** 2)))
    out = (centre - rad) / den
    return np.where(n > 0.0, np.clip(out, 0.0, 1.0), 0.0).astype(np.float32)


class CapabilityField:
    """Dense-storage, low-resolution self-map with event-based evidence."""

    def __init__(self, half_life_steps: float = DEFAULT_HALF_LIFE_STEPS):
        self.shape = FIELD_SHAPE
        self.bounds = FIELD_BOUNDS
        self.half_life_steps = float(half_life_steps)
        self.centers = tuple(np.linspace(lo, hi, n, dtype=np.float32)
                             for (lo, hi), n in zip(self.bounds, self.shape))
        sh = self.shape

        # Clean command nodes: success/failure probability + separate quality.
        self.node_success = np.zeros(sh, np.float32)
        self.node_failure = np.zeros(sh, np.float32)
        self.node_quality_sum = np.zeros(sh, np.float32)
        self.node_slow = np.full(sh, 0.5, np.float32)
        self.node_last_step = np.full(sh, -1, np.int64)

        # Integrated recovery layer, indexed by command node + severity band.
        rsh = sh + (RECOVERY_BINS,)
        self.rec_success = np.zeros(rsh, np.float32)
        self.rec_failure = np.zeros(rsh, np.float32)

        # Maneuver graph edges: source node + one of 26 command directions.
        esh = sh + (len(EDGE_DIRS),)
        self.edge_success = np.zeros(esh, np.float32)
        self.edge_failure = np.zeros(esh, np.float32)
        self.edge_rate_sum = np.zeros(esh, np.float32)
        self.edge_slow = np.full(esh, 0.5, np.float32)

        self.total_updates = 0
        self.total_node_events = 0
        self.total_recovery_events = 0
        self.total_edge_events = 0
        self.last_step = 0
        self.last_decay = 1.0

        # Runtime event trackers are intentionally NOT serialized.  Environment
        # physics/episode buffers also restart fresh on process resume, so a
        # half-finished one-second window cannot be continued honestly anyway.
        self._runtime_n = 0
        self._clear_runtime()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _axis_index(self, x: np.ndarray, axis: int) -> np.ndarray:
        lo, hi = self.bounds[axis]
        n = self.shape[axis]
        u = (np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-6)
        return np.clip(np.rint(u * (n - 1)), 0, n - 1).astype(np.int32)

    def command_indices(self, vx, vy, yaw):
        return self._axis_index(vx, 0), self._axis_index(vy, 1), self._axis_index(yaw, 2)

    def command_for_index(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return np.asarray([self.centers[0][ix], self.centers[1][iy], self.centers[2][iz]], np.float32)

    def _flat_index(self, ix, iy, iz):
        return np.ravel_multi_index((ix, iy, iz), self.shape)

    # ------------------------------------------------------------------
    # Runtime event trackers
    # ------------------------------------------------------------------
    def _clear_runtime(self) -> None:
        self._node_active = np.zeros((0,), bool)
        self._node_flat = np.zeros((0,), np.int64)
        self._node_frames = np.zeros((0,), np.int32)
        self._node_structural_good_frames = np.zeros((0,), np.int32)
        self._node_err_sum_rt = np.zeros((0, 3), np.float32)
        self._node_tol_sum_rt = np.zeros((0, 3), np.float32)
        self._node_quality_sum_rt = np.zeros((0,), np.float32)

        self._rec_active = np.zeros((0,), bool)
        self._rec_flat = np.zeros((0,), np.int64)
        self._rec_bin = np.zeros((0,), np.int32)
        self._rec_streak = np.zeros((0,), np.int32)

        # 0=idle, 1=ramping, 2=settling target.
        self._edge_phase = np.zeros((0,), np.int8)
        self._edge_source = np.zeros((0, 3), np.float32)
        self._edge_target = np.zeros((0, 3), np.float32)
        self._edge_source_flat = np.zeros((0,), np.int64)
        self._edge_rate_cmd = np.zeros((0,), np.float32)
        self._edge_success_streak = np.zeros((0,), np.int32)

        self._prev_cmd = np.zeros((0, 3), np.float32)
        self._have_prev = np.zeros((0,), bool)

    def _ensure_runtime(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if self._runtime_n == n_envs:
            return
        self._runtime_n = n_envs
        self._node_active = np.zeros(n_envs, bool)
        self._node_flat = np.zeros(n_envs, np.int64)
        self._node_frames = np.zeros(n_envs, np.int32)
        self._node_structural_good_frames = np.zeros(n_envs, np.int32)
        self._node_err_sum_rt = np.zeros((n_envs, 3), np.float32)
        self._node_tol_sum_rt = np.zeros((n_envs, 3), np.float32)
        self._node_quality_sum_rt = np.zeros(n_envs, np.float32)

        self._rec_active = np.zeros(n_envs, bool)
        self._rec_flat = np.zeros(n_envs, np.int64)
        self._rec_bin = np.zeros(n_envs, np.int32)
        self._rec_streak = np.zeros(n_envs, np.int32)

        self._edge_phase = np.zeros(n_envs, np.int8)
        self._edge_source = np.zeros((n_envs, 3), np.float32)
        self._edge_target = np.zeros((n_envs, 3), np.float32)
        self._edge_source_flat = np.zeros(n_envs, np.int64)
        self._edge_rate_cmd = np.zeros(n_envs, np.float32)
        self._edge_success_streak = np.zeros(n_envs, np.int32)

        self._prev_cmd = np.zeros((n_envs, 3), np.float32)
        self._have_prev = np.zeros(n_envs, bool)

    # ------------------------------------------------------------------
    # Exponential forgetting
    # ------------------------------------------------------------------
    def decay_to(self, step: int) -> float:
        step = int(step)
        delta = max(0, step - int(self.last_step))
        if delta <= 0 or self.half_life_steps <= 0:
            self.last_step = step
            self.last_decay = 1.0
            return 1.0
        d = float(2.0 ** (-delta / float(self.half_life_steps)))
        for arr in (
            self.node_success, self.node_failure, self.node_quality_sum,
            self.rec_success, self.rec_failure,
            self.edge_success, self.edge_failure, self.edge_rate_sum,
        ):
            arr *= d
        self.last_step = step
        self.last_decay = d
        return d

    # ------------------------------------------------------------------
    # Basic estimates
    # ------------------------------------------------------------------
    def node_competence(self) -> np.ndarray:
        return _posterior_mean(self.node_success, self.node_failure)

    def node_lower(self) -> np.ndarray:
        return _wilson_lower(self.node_success, self.node_failure)

    def node_evidence(self) -> np.ndarray:
        return self.node_success + self.node_failure

    def node_quality(self) -> np.ndarray:
        return self.node_quality_sum / np.maximum(self.node_evidence(), 1e-6)

    def node_confidence(self) -> np.ndarray:
        return (1.0 - np.exp(-self.node_evidence() / max(STRONG_EFFECTIVE_EVIDENCE, 1e-6))).astype(np.float32)

    def recovery_competence(self) -> np.ndarray:
        return _posterior_mean(self.rec_success, self.rec_failure)

    def recovery_lower(self) -> np.ndarray:
        return _wilson_lower(self.rec_success, self.rec_failure)

    def recovery_evidence(self) -> np.ndarray:
        return self.rec_success + self.rec_failure

    def edge_competence(self) -> np.ndarray:
        return _posterior_mean(self.edge_success, self.edge_failure)

    def edge_lower(self) -> np.ndarray:
        return _wilson_lower(self.edge_success, self.edge_failure)

    def edge_evidence(self) -> np.ndarray:
        return self.edge_success + self.edge_failure

    def edge_rate(self) -> np.ndarray:
        return self.edge_rate_sum / np.maximum(self.edge_success, 1e-6)

    # ------------------------------------------------------------------
    # Event accumulation helpers
    # ------------------------------------------------------------------
    def _add_node_events(self, flats: list[np.ndarray], successes: list[np.ndarray], qualities: list[np.ndarray], step: int) -> int:
        if not flats:
            return 0
        flat = np.concatenate(flats).astype(np.int64, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        qua = np.concatenate(qualities).astype(np.float32, copy=False)
        if flat.size == 0:
            return 0
        size = int(np.prod(self.shape))
        self.node_success += np.bincount(flat, weights=suc, minlength=size).reshape(self.shape).astype(np.float32)
        self.node_failure += np.bincount(flat, weights=1.0-suc, minlength=size).reshape(self.shape).astype(np.float32)
        self.node_quality_sum += np.bincount(flat, weights=qua, minlength=size).reshape(self.shape).astype(np.float32)
        touched = np.unique(flat)
        self.node_last_step.ravel()[touched] = int(step)
        self.total_node_events += int(flat.size)
        return int(flat.size)

    def _add_recovery_events(self, flats: list[np.ndarray], bins: list[np.ndarray], successes: list[np.ndarray]) -> int:
        if not flats:
            return 0
        flat = np.concatenate(flats).astype(np.int64, copy=False)
        rb = np.concatenate(bins).astype(np.int64, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        if flat.size == 0:
            return 0
        rflat = flat * RECOVERY_BINS + rb
        rsize = int(np.prod(self.shape)) * RECOVERY_BINS
        shape = self.rec_success.shape
        self.rec_success += np.bincount(rflat, weights=suc, minlength=rsize).reshape(shape).astype(np.float32)
        self.rec_failure += np.bincount(rflat, weights=1.0-suc, minlength=rsize).reshape(shape).astype(np.float32)
        self.total_recovery_events += int(flat.size)
        return int(flat.size)

    def _edge_sector_and_rate(self, source: np.ndarray, target: np.ndarray, rate_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_n = (target - source) / EDGE_SCALES[None, :]
        mag = np.linalg.norm(delta_n, axis=1)
        unit = delta_n / np.maximum(mag[:, None], 1e-6)
        sector = np.argmax(unit @ EDGE_DIRS.T, axis=1).astype(np.int32)
        rate = np.where(rate_cmd > 1e-6, rate_cmd, mag).astype(np.float32)
        return sector, rate, mag

    def _add_edge_events(self, source_flats: list[np.ndarray], sources: list[np.ndarray], targets: list[np.ndarray],
                         rate_cmds: list[np.ndarray], successes: list[np.ndarray]) -> int:
        if not source_flats:
            return 0
        sf = np.concatenate(source_flats).astype(np.int64, copy=False)
        src = np.concatenate(sources).astype(np.float32, copy=False)
        tgt = np.concatenate(targets).astype(np.float32, copy=False)
        rc = np.concatenate(rate_cmds).astype(np.float32, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        if sf.size == 0:
            return 0
        sec, rate, mag = self._edge_sector_and_rate(src, tgt, rc)
        keep = mag >= 0.04  # ignore tiny command jitter / same-cell resamples
        if not np.any(keep):
            return 0
        sf, sec, rate, suc = (x[keep] for x in (sf, sec, rate, suc))
        eflat = sf * len(EDGE_DIRS) + sec.astype(np.int64)
        esize = int(np.prod(self.shape)) * len(EDGE_DIRS)
        shape = self.edge_success.shape
        self.edge_success += np.bincount(eflat, weights=suc, minlength=esize).reshape(shape).astype(np.float32)
        self.edge_failure += np.bincount(eflat, weights=1.0-suc, minlength=esize).reshape(shape).astype(np.float32)
        # Maneuver-rate frontier is meaningful only for successful edges.
        self.edge_rate_sum += np.bincount(eflat, weights=rate*suc, minlength=esize).reshape(shape).astype(np.float32)
        self.total_edge_events += int(eflat.size)
        return int(eflat.size)

    # ------------------------------------------------------------------
    # Main rollout ingestion
    # ------------------------------------------------------------------
    def update_from_rollout(self, metrics: dict[str, Any], *, step: int, control_dt: float = CONTROL_DT,
                            exclude_field_taught: bool = False) -> dict[str, float]:
        """Convert one PPO rollout into semantic node/recovery/maneuver events.

        ``exclude_field_taught`` is used only by the clean-evidence tournament arm:
        experience deliberately selected by the field still trains PPO, but cannot
        reinforce the field that selected it.  Untaught governor experience remains
        normal evidence.  Missing metrics are a broken trainer contract, not
        permission to continue with an inert self-map.
        """
        self.decay_to(step)
        needed = ('cmd_vx','cmd_vy','cmd_yaw','vx','vy','yaw_rate','gravity_z','support',
                  'bad_contact','episode_fell','episode_done','age_s','transition_frac','cmd_resampled',
                  'regime_resilience','birth_resilience_d','target_cmd_vx','target_cmd_vy','target_cmd_yaw',
                  'transition_ramp_speed')
        missing = [k for k in needed if k not in metrics]
        if missing:
            raise KeyError('capability field missing required rollout metrics: ' + ', '.join(missing))

        def arr(name): return np.asarray(metrics[name], np.float32)
        cvx, cvy, cyaw = arr('cmd_vx'), arr('cmd_vy'), arr('cmd_yaw')
        if cvx.ndim == 1: cvx, cvy, cyaw = cvx[None,:], cvy[None,:], cyaw[None,:]
        T,N=cvx.shape; self._ensure_runtime(N)
        vx,vy,yaw=arr('vx'),arr('vy'),arr('yaw_rate')
        gz,support,bad=arr('gravity_z'),arr('support'),arr('bad_contact')
        fell,done,age=arr('episode_fell'),arr('episode_done'),arr('age_s')
        ramp=arr('transition_frac')>.5; cmd_resampled=arr('cmd_resampled')>.5
        resilience=arr('regime_resilience')>.5; sev=np.clip(arr('birth_resilience_d'),0.0,.999999)
        tgt_vx,tgt_vy,tgt_yaw=arr('target_cmd_vx'),arr('target_cmd_vy'),arr('target_cmd_yaw')
        ramp_speed=arr('transition_ramp_speed')
        if exclude_field_taught:
            if 'field_taught' not in metrics:
                raise KeyError('capability field clean-evidence mode requires field_taught rollout metric')
            taught=arr('field_taught')>.5
        else:
            taught=np.zeros_like(cvx,dtype=bool)

        vx_tol=np.maximum(VX_ABS_TOL,VX_REL_TOL*np.abs(cvx)); vy_tol=np.maximum(VY_ABS_TOL,VY_REL_TOL*np.abs(cvy)); yaw_tol=np.maximum(YAW_ABS_TOL,YAW_REL_TOL*np.abs(cyaw))
        within=(np.abs(vx-cvx)<=vx_tol)&(np.abs(vy-cvy)<=vy_tol)&(np.abs(yaw-cyaw)<=yaw_tol)
        structural_ok=(gz>UPRIGHT_GRAVITY_Z_MIN)&(support>SUPPORT_MIN)&(bad<.5)&(fell<.5)
        core_ok=within&structural_ok
        e1=np.abs(vx-cvx)/np.maximum(vx_tol,1e-6); e2=np.abs(vy-cvy)/np.maximum(vy_tol,1e-6); e3=np.abs(yaw-cyaw)/np.maximum(yaw_tol,1e-6)
        tracking_q=np.exp(-.5*(e1*e1+e2*e2+e3*e3)/3.0)
        upright_q=np.clip((gz-.72)/(1-.72),0,1); support_q=np.clip(support/.35,0,1)
        quality=np.clip(tracking_q*(.65+.20*upright_q+.15*support_q),0,1).astype(np.float32)
        ix,iy,iz=self.command_indices(cvx,cvy,cyaw); flat_all=self._flat_index(ix.ravel(),iy.ravel(),iz.ravel()).reshape((T,N))

        node_flats=[]; node_succ=[]; node_qual=[]; rec_flats=[]; rec_bins=[]; rec_succ=[]
        edge_sf=[]; edge_src=[]; edge_tgt=[]; edge_rate=[]; edge_succ=[]
        node_target=max(1,int(round(float(NODE_WINDOW_S)/max(control_dt,1e-6))))
        rec_target=max(1,int(round(float(RECOVERY_STABLE_S)/max(control_dt,1e-6))))
        edge_target=max(1,int(round(float(EDGE_STABLE_S)/max(control_dt,1e-6))))

        def reset_node(ids):
            if len(ids)==0: return
            self._node_active[ids]=False; self._node_frames[ids]=0; self._node_structural_good_frames[ids]=0
            self._node_err_sum_rt[ids]=0.0; self._node_tol_sum_rt[ids]=0.0; self._node_quality_sum_rt[ids]=0.0

        def emit_edge(mask,success):
            ids=np.flatnonzero(mask)
            if not ids.size: return
            edge_sf.append(self._edge_source_flat[ids].copy()); edge_src.append(self._edge_source[ids].copy()); edge_tgt.append(self._edge_target[ids].copy())
            edge_rate.append(self._edge_rate_cmd[ids].copy()); edge_succ.append(np.full(ids.size,success,np.float32))
            self._edge_phase[ids]=0; self._edge_success_streak[ids]=0

        for t in range(T):
            cmd_t=np.stack([cvx[t],cvy[t],cyaw[t]],axis=1).astype(np.float32); target_t=np.stack([tgt_vx[t],tgt_vy[t],tgt_yaw[t]],axis=1).astype(np.float32)
            blocked=taught[t]
            if np.any(blocked):
                # Break every semantic event machine at the boundary of
                # field-selected practice.  Otherwise an event could begin on
                # unbiased experience and finish inside the self-selected region.
                ids=np.flatnonzero(blocked); reset_node(ids)
                self._rec_active[ids]=False; self._rec_streak[ids]=0
                self._edge_phase[ids]=0; self._edge_success_streak[ids]=0
                self._have_prev[ids]=False
            new_ep=age[t] <= (1.5*float(control_dt)+1e-6)
            node_fall=(fell[t]>.5)&(~resilience[t])&(~ramp[t])&(~blocked)&(age[t]>=float(NODE_SETTLE_DELAY_S))
            ids=np.flatnonzero(node_fall)
            if ids.size:
                ff=np.where(self._node_active[ids],self._node_flat[ids],flat_all[t,ids]); node_flats.append(ff.copy()); node_succ.append(np.zeros(ids.size,np.float32))
                q=np.where(self._node_active[ids],np.divide(self._node_quality_sum_rt[ids],np.maximum(self._node_frames[ids],1)),quality[t,ids]); node_qual.append(np.asarray(q,np.float32))
            if np.any(new_ep):
                ids=np.flatnonzero(new_ep); reset_node(ids); self._edge_phase[ids]=0; self._edge_success_streak[ids]=0
                self._rec_active[ids]=False; self._rec_streak[ids]=0; self._have_prev[ids]=False
                rids=ids[resilience[t,ids]&(~blocked[ids])]
                if rids.size:
                    self._rec_active[rids]=True; self._rec_flat[rids]=flat_all[t,rids]; self._rec_bin[rids]=np.clip(np.floor(sev[t,rids]*RECOVERY_BINS),0,RECOVERY_BINS-1).astype(np.int32)

            eligible=self._rec_active&(~blocked)&(age[t]>=float(RECOVERY_MIN_AGE_S)); self._rec_streak[eligible&core_ok[t]]+=1; self._rec_streak[eligible&(~core_ok[t])]=0
            ids=np.flatnonzero(self._rec_active&(self._rec_streak>=rec_target))
            if ids.size:
                rec_flats.append(self._rec_flat[ids].copy()); rec_bins.append(self._rec_bin[ids].copy()); rec_succ.append(np.ones(ids.size,np.float32)); self._rec_active[ids]=False; self._rec_streak[ids]=0
            ids=np.flatnonzero(self._rec_active&(done[t]>.5))
            if ids.size:
                rec_flats.append(self._rec_flat[ids].copy()); rec_bins.append(self._rec_bin[ids].copy()); rec_succ.append(np.zeros(ids.size,np.float32)); self._rec_active[ids]=False; self._rec_streak[ids]=0

            emit_edge((self._edge_phase>0)&(done[t]>.5),0.0)
            edge_signal=(ramp[t]|cmd_resampled[t])&(~blocked); emit_edge((self._edge_phase==2)&edge_signal&(done[t]<.5),0.0)
            ids=np.flatnonzero((self._edge_phase==0)&edge_signal&(done[t]<.5))
            if ids.size:
                src=np.where(self._have_prev[ids,None],self._prev_cmd[ids],cmd_t[ids]); self._edge_source[ids]=src; self._edge_target[ids]=target_t[ids]
                six,siy,siz=self.command_indices(src[:,0],src[:,1],src[:,2]); self._edge_source_flat[ids]=self._flat_index(six,siy,siz)
                delta=np.linalg.norm((target_t[ids]-src)/EDGE_SCALES[None,:],axis=1); duration=np.where(ramp_speed[t,ids]>1e-6,float(control_dt)/np.maximum(ramp_speed[t,ids],1e-6),float(control_dt)); self._edge_rate_cmd[ids]=delta/np.maximum(duration,float(control_dt))
                self._edge_phase[ids]=1; self._edge_success_streak[ids]=0
            finish=(self._edge_phase==1)&(~edge_signal)&(done[t]<.5); ids=np.flatnonzero(finish)
            if ids.size: self._edge_phase[ids]=2; self._edge_success_streak[ids]=core_ok[t,ids].astype(np.int32)
            settling=self._edge_phase==2; cont=settling&(~finish); self._edge_success_streak[cont&core_ok[t]]+=1; self._edge_success_streak[cont&(~core_ok[t])]=0; emit_edge((self._edge_phase==2)&(self._edge_success_streak>=edge_target),1.0)

            clean=(age[t]>=float(NODE_SETTLE_DELAY_S))&(~resilience[t])&(~ramp[t])&(~blocked)&(done[t]<.5); current_flat=flat_all[t]
            ids=np.flatnonzero(self._node_active&((self._node_flat!=current_flat)|(~clean))); reset_node(ids)
            ids=np.flatnonzero(clean&(~self._node_active))
            if ids.size: self._node_active[ids]=True; self._node_flat[ids]=current_flat[ids]
            active=self._node_active&clean; self._node_frames[active]+=1; self._node_structural_good_frames[active]+=structural_ok[t,active].astype(np.int32)
            self._node_err_sum_rt[active,0]+=np.abs(vx[t,active]-cvx[t,active]); self._node_err_sum_rt[active,1]+=np.abs(vy[t,active]-cvy[t,active]); self._node_err_sum_rt[active,2]+=np.abs(yaw[t,active]-cyaw[t,active])
            self._node_tol_sum_rt[active,0]+=vx_tol[t,active]; self._node_tol_sum_rt[active,1]+=vy_tol[t,active]; self._node_tol_sum_rt[active,2]+=yaw_tol[t,active]; self._node_quality_sum_rt[active]+=quality[t,active]
            ids=np.flatnonzero(active&(self._node_frames>=node_target))
            if ids.size:
                frames=np.maximum(self._node_frames[ids,None],1); ok_track=np.all((self._node_err_sum_rt[ids]/frames)<=(self._node_tol_sum_rt[ids]/frames),axis=1); ok_structure=(self._node_structural_good_frames[ids]/np.maximum(self._node_frames[ids],1))>=float(NODE_STRUCTURAL_FRACTION_MIN)
                node_flats.append(self._node_flat[ids].copy()); node_succ.append((ok_track&ok_structure).astype(np.float32)); node_qual.append((self._node_quality_sum_rt[ids]/np.maximum(self._node_frames[ids],1)).astype(np.float32)); self._node_frames[ids]=0; self._node_structural_good_frames[ids]=0; self._node_err_sum_rt[ids]=0.; self._node_tol_sum_rt[ids]=0.; self._node_quality_sum_rt[ids]=0.
            unbiased=~blocked
            self._prev_cmd[unbiased]=cmd_t[unbiased]; self._have_prev[unbiased]=True
            self._have_prev[blocked]=False

        n_node=self._add_node_events(node_flats,node_succ,node_qual,step); n_rec=self._add_recovery_events(rec_flats,rec_bins,rec_succ); n_edge=self._add_edge_events(edge_sf,edge_src,edge_tgt,edge_rate,edge_succ)
        a=float(SLOW_ALPHA); self.node_slow+=a*(self.node_competence()-self.node_slow); self.edge_slow+=a*(self.edge_competence()-self.edge_slow); self.total_updates+=1
        out=self.summary(); out.update({'field_node_events_update':float(n_node),'field_recovery_events_update':float(n_rec),'field_edge_events_update':float(n_edge)}); return out

    # ------------------------------------------------------------------
    # Curriculum packet generation
    # ------------------------------------------------------------------
    def _diverse_top(self, score: np.ndarray, count: int, *, valid: np.ndarray | None = None) -> list[tuple[int,int,int]]:
        s = np.asarray(score, np.float64).copy()
        if valid is not None:
            s = np.where(valid, s, -np.inf)
        chosen: list[tuple[int,int,int]] = []
        for _ in range(max(0, int(count))):
            flat = int(np.argmax(s))
            if not np.isfinite(s.ravel()[flat]):
                break
            idx = tuple(int(x) for x in np.unravel_index(flat, self.shape))
            chosen.append(idx)
            x,y,z = idx
            s[max(0,x-1):min(self.shape[0],x+2),
              max(0,y-1):min(self.shape[1],y+2),
              max(0,z-1):min(self.shape[2],z+2)] *= 0.25
            s[idx] = -np.inf
        return chosen

    def _recovery_teaching_severity(self, q: tuple[int,int,int]) -> float:
        p = self.recovery_competence()[q]
        lb = self.recovery_lower()[q]
        n = self.recovery_evidence()[q]
        bins = RECOVERY_BINS
        known = n >= 1.5
        if not np.any(known):
            return float(0.5 / max(1, bins))
        # Consolidate near the learnable ~60% frontier.
        score = np.where(known, np.abs(p - 0.62), np.inf)
        k = int(np.argmin(score))
        mastered = np.flatnonzero((n >= STRONG_EFFECTIVE_EVIDENCE) &
                                  (lb >= RECOVERY_MASTER_LOWER))
        if mastered.size:
            nxt = min(bins-1, int(mastered.max())+1)
            if n[nxt] < STRONG_EFFECTIVE_EVIDENCE:
                k = nxt
        return float((k + 0.5) / max(1, bins))

    def node_anchor_packet(self, count: int = NODE_ANCHORS) -> np.ndarray:
        p = self.node_competence(); lb = self.node_lower(); n = self.node_evidence()
        trend = p - self.node_slow
        known = n >= MIN_EFFECTIVE_EVIDENCE
        strong = n >= STRONG_EFFECTIVE_EVIDENCE
        mastered = strong & (lb >= NODE_MASTER_LOWER)

        # Productive frontier: learnable, not already mastered, with bonus for
        # positive learning velocity and enough evidence to be meaningful.
        frontier = np.exp(-0.5*((p-0.62)/0.19)**2)
        frontier *= (0.65 + 0.35*np.tanh(4.0*np.maximum(trend,0.0)))
        frontier *= np.clip(n / max(STRONG_EFFECTIVE_EVIDENCE,1e-6),0.0,1.0)
        frontier *= (~mastered)

        # Regression/staleness: a formerly stronger cell whose current
        # competence/evidence is fading deserves rehearsal.
        repair = np.maximum(self.node_slow-p,0.0)*3.0
        repair += np.where((p>0.75)&(n<STRONG_EFFECTIVE_EVIDENCE),0.35,0.0)

        # Unknown cells immediately adjacent to mastered territory are the
        # cleanest expansion targets. This mechanism was effectively disabled
        # because pseudo-probability alone never reaches reliable mastery.
        adjacent = np.zeros_like(mastered)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx==dy==dz==0:
                        continue
                    src = mastered[max(0,-dx):self.shape[0]-max(0,dx),
                                   max(0,-dy):self.shape[1]-max(0,dy),
                                   max(0,-dz):self.shape[2]-max(0,dz)]
                    dst = adjacent[max(0,dx):self.shape[0]-max(0,-dx),
                                   max(0,dy):self.shape[1]-max(0,-dy),
                                   max(0,dz):self.shape[2]-max(0,-dz)]
                    dst |= src
        explore = adjacent & (~known)

        # Under-covered but already encountered cells receive a modest bonus;
        # this reduces seed-history bias without forcing artificial symmetry.
        undercovered = known & (n < STRONG_EFFECTIVE_EVIDENCE) & (p > 0.45)
        coverage_score = np.where(undercovered, 1.0 - n/max(STRONG_EFFECTIVE_EVIDENCE,1e-6), 0.0)

        n_front=max(1,int(round(count*0.40)))
        n_repair=max(1,int(round(count*0.20)))
        n_explore=max(1,int(round(count*0.25)))
        n_cover=max(1,int(round(count*0.10)))
        n_master=max(0,count-n_front-n_repair-n_explore-n_cover)
        picks=[]
        picks += self._diverse_top(frontier,n_front,valid=known)
        picks += self._diverse_top(repair,n_repair,valid=(n>0.5))
        picks += self._diverse_top(np.ones(self.shape),n_explore,valid=explore)
        picks += self._diverse_top(coverage_score,n_cover,valid=undercovered)
        picks += self._diverse_top(p,n_master,valid=mastered)

        uniq=[]; seen=set()
        for q in picks:
            if q not in seen:
                uniq.append(q); seen.add(q)
        if len(uniq)<count:
            for q in self._diverse_top(n,count-len(uniq),valid=n>0):
                if q not in seen:
                    uniq.append(q); seen.add(q)

        out=np.zeros((count,5),np.float32)
        for i,q in enumerate(uniq[:count]):
            out[i,:3]=self.command_for_index(*q)
            w=0.20+float(frontier[q])+float(repair[q])+0.45*float(explore[q])+0.35*float(coverage_score[q])
            out[i,3]=max(0.05,w)
            out[i,4]=self._recovery_teaching_severity(q)
        return out

    def edge_anchor_packet(self, count: int = EDGE_ANCHORS) -> np.ndarray:
        ep=self.edge_competence(); elb=self.edge_lower(); en=self.edge_evidence(); trend=ep-self.edge_slow
        np_=self.node_competence(); nlb=self.node_lower(); nn=self.node_evidence()
        known=en>=2.0
        src_ok=(nn>=STRONG_EFFECTIVE_EVIDENCE)&(nlb>=0.65)
        mastered=(en>=STRONG_EFFECTIVE_EVIDENCE)&(elb>=EDGE_MASTER_LOWER)
        score=np.exp(-0.5*((ep-0.62)/0.20)**2)*(0.7+0.3*np.tanh(4*np.maximum(trend,0)))
        score += 1.5*np.maximum(self.edge_slow-ep,0)
        # Under-covered local directions get a small exploration bonus.
        score += np.where((en>0)&(en<STRONG_EFFECTIVE_EVIDENCE),0.25,0.0)
        score *= src_ok[...,None]
        score *= (~mastered)
        score=np.where(known,score,-np.inf)

        out=np.zeros((count,7),np.float32)
        work=score.copy()
        grid_step=np.asarray([(hi-lo)/(n-1) for (lo,hi),n in zip(self.bounds,self.shape)],np.float32)
        for i in range(count):
            flat=int(np.argmax(work))
            if not np.isfinite(work.ravel()[flat]):
                break
            ix,iy,iz,sec=np.unravel_index(flat,work.shape)
            src=self.command_for_index(ix,iy,iz)
            delta=EDGE_DIRS[sec]*grid_step*2.0
            tgt=src+delta
            for a,(lo,hi) in enumerate(self.bounds):
                tgt[a]=np.clip(tgt[a],lo,hi)
            out[i,:3]=src; out[i,3:6]=tgt; out[i,6]=max(0.05,float(work[ix,iy,iz,sec]))
            work[max(0,ix-1):min(self.shape[0],ix+2),
                 max(0,iy-1):min(self.shape[1],iy+2),
                 max(0,iz-1):min(self.shape[2],iz+2),sec] *= 0.25
            work[ix,iy,iz,sec]=-np.inf
        return out

    def influence(self, step: int, *, start: int, full: int, maximum: float, mode: str,
                  fade_start: int = 0, zero: int = 0) -> tuple[float,float]:
        """Teaching dose with optional ramp-up and fade-out.

        The default is the historical one-way ramp.  ``fade_start``/``zero``
        create the tutor-then-release schedule without adding another runtime
        subsystem: influence is simply multiplied by a linear fade to zero.
        """
        if mode in ('off','shadow') or step <= start:
            return 0.0,0.0
        x=1.0 if full<=start else float(np.clip((step-start)/(full-start),0.0,1.0))
        if fade_start > 0 and zero > fade_start and step > fade_start:
            x *= float(np.clip((zero-step)/(zero-fade_start),0.0,1.0))
        node=float(maximum)*x
        edge=node if mode=='graph' else 0.0
        return node,edge

    def teaching_packet(self, *, step: int, mode: str, start: int, full: int, maximum: float,
                        fade_start: int = 0, zero: int = 0,
                        teach_nodes: bool = True, teach_edges: bool = True,
                        node_count: int = NODE_ANCHORS, edge_count: int = EDGE_ANCHORS) -> dict[str, Any]:
        nf,ef=self.influence(step,start=start,full=full,maximum=maximum,mode=mode,
                             fade_start=fade_start,zero=zero)
        if not teach_nodes:
            nf=0.0
        if not teach_edges:
            ef=0.0
        return {
            'field_node_frac':nf,
            'field_edge_frac':ef,
            'field_node_anchors':self.node_anchor_packet(node_count),
            'field_edge_anchors':self.edge_anchor_packet(edge_count),
        }

    # ------------------------------------------------------------------
    # Held-out calibration selection (selection only; no feedback)
    # ------------------------------------------------------------------
    def predict_command(self, vx: float, vy: float, yaw: float) -> tuple[float, float]:
        ix,iy,iz=self.command_indices(np.asarray([vx]),np.asarray([vy]),np.asarray([yaw]))
        q=(int(ix[0]),int(iy[0]),int(iz[0]))
        return float(self.node_competence()[q]), float(self.node_evidence()[q])

    def calibration_commands(self, count: int) -> list[tuple[float,float,float,float,float]]:
        count=max(1,int(count))
        p=self.node_competence(); n=self.node_evidence()
        canonical=np.asarray([0.92,0.78,0.62,0.45,0.28],dtype=np.float32)
        targets=[float(canonical[j%len(canonical)]) for j in range(count)]
        candidates=[]; used=set(); known=n>=MIN_EFFECTIVE_EVIDENCE
        for t in targets:
            score=np.abs(p-t)+0.16/np.sqrt(np.maximum(n,0.05))
            score=np.where(known,score,np.inf)
            for q in used:
                x,y,z=q
                score[max(0,x-1):min(self.shape[0],x+2),
                      max(0,y-1):min(self.shape[1],y+2),
                      max(0,z-1):min(self.shape[2],z+2)]=np.inf
            flat=int(np.argmin(score))
            if not np.isfinite(score.ravel()[flat]):
                break
            q=tuple(int(x) for x in np.unravel_index(flat,self.shape)); used.add(q)
            cmd=self.command_for_index(*q)
            candidates.append((float(cmd[0]),float(cmd[1]),float(cmd[2]),float(p[q]),float(n[q])))
        if len(candidates)<count:
            work=n.copy()
            for _ in range(count-len(candidates)):
                flat=int(np.argmax(work)); q=tuple(int(x) for x in np.unravel_index(flat,self.shape))
                if work[q]<=0: break
                cmd=self.command_for_index(*q)
                candidates.append((float(cmd[0]),float(cmd[1]),float(cmd[2]),float(p[q]),float(n[q])))
                work[q]=-1
        return candidates

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------
    def summary(self) -> dict[str,float]:
        p=self.node_competence(); lb=self.node_lower(); n=self.node_evidence(); q=self.node_quality()
        known=n>=MIN_EFFECTIVE_EVIDENCE
        strong=n>=STRONG_EFFECTIVE_EVIDENCE
        mastered=strong&(lb>=NODE_MASTER_LOWER)
        frontier=known&(~mastered)&(p>=0.35)
        age=np.where(self.node_last_step>=0,np.maximum(0,int(self.last_step)-self.node_last_step),0)
        stale=mastered&(age>=float(self.half_life_steps)*float(STALE_HALF_LIVES))

        rp=self.recovery_competence(); rlb=self.recovery_lower(); rn=self.recovery_evidence()
        rknown=rn>=2.0; rmaster=(rn>=STRONG_EFFECTIVE_EVIDENCE)&(rlb>=RECOVERY_MASTER_LOWER)
        ep=self.edge_competence(); elb=self.edge_lower(); en=self.edge_evidence()
        eknown=en>=2.0; emaster=(en>=STRONG_EFFECTIVE_EVIDENCE)&(elb>=EDGE_MASTER_LOWER)
        return {
            'field_schema':float(FIELD_VERSION),
            'field_known_nodes':float(np.sum(known)),
            'field_strong_nodes':float(np.sum(strong)),
            'field_competent_nodes':float(np.sum(mastered)),
            'field_frontier_nodes':float(np.sum(frontier)),
            'field_stale_nodes':float(np.sum(stale)),
            'field_mean_competence_known':float(np.mean(p[known])) if np.any(known) else 0.5,
            'field_mean_lower_known':float(np.mean(lb[known])) if np.any(known) else 0.0,
            'field_mean_quality_known':float(np.mean(q[known])) if np.any(known) else 0.0,
            'field_learning_nodes':float(np.sum(known&((p-self.node_slow)>0.05))),
            'field_forgetting_nodes':float(np.sum(known&((self.node_slow-p)>0.05))),
            'field_known_recovery_cells':float(np.sum(rknown)),
            'field_competent_recovery_cells':float(np.sum(rmaster)),
            'field_known_edges':float(np.sum(eknown)),
            'field_competent_edges':float(np.sum(emaster)),
            'field_mean_edge_competence':float(np.mean(ep[eknown])) if np.any(eknown) else 0.5,
            'field_mean_edge_rate':float(np.mean(self.edge_rate()[eknown])) if np.any(eknown) else 0.0,
            'field_updates':float(self.total_updates),
            'field_node_events_total':float(self.total_node_events),
            'field_recovery_events_total':float(self.total_recovery_events),
            'field_edge_events_total':float(self.total_edge_events),
            'field_effective_samples':float(np.sum(n)),
            'field_last_decay':float(self.last_decay),
        }

    def state_dict(self) -> dict[str,Any]:
        return {
            'field_version':FIELD_VERSION,
            'half_life_steps':float(self.half_life_steps),
            'last_step':int(self.last_step),'total_updates':int(self.total_updates),
            'total_node_events':int(self.total_node_events),'total_recovery_events':int(self.total_recovery_events),
            'total_edge_events':int(self.total_edge_events),
            'node_success':self.node_success,'node_failure':self.node_failure,
            'node_quality_sum':self.node_quality_sum,
            'node_slow':self.node_slow,'node_last_step':self.node_last_step,
            'rec_success':self.rec_success,'rec_failure':self.rec_failure,
            'edge_success':self.edge_success,'edge_failure':self.edge_failure,
            'edge_rate_sum':self.edge_rate_sum,'edge_slow':self.edge_slow,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str,Any]) -> 'CapabilityField':
        arrays = (
            'node_success','node_failure','node_quality_sum','node_slow','node_last_step',
            'rec_success','rec_failure',
            'edge_success','edge_failure','edge_rate_sum','edge_slow')
        expected = {
            'field_version','half_life_steps','last_step','total_updates','total_node_events',
            'total_recovery_events','total_edge_events', *arrays}
        if not isinstance(d, dict) or set(d) != expected:
            got = sorted(d) if isinstance(d, dict) else type(d).__name__
            raise ValueError(f'capability field state keys mismatch: expected={sorted(expected)} got={got}')
        version=int(d['field_version'])
        if version != FIELD_VERSION:
            raise ValueError(f'capability field schema {version} cannot resume as schema {FIELD_VERSION}; start a fresh current-schema map')
        obj=cls(float(d['half_life_steps']))
        for name in arrays:
            dtype=np.int64 if name.endswith('_last_step') else np.float32
            arr=np.asarray(d[name],dtype)
            expected_shape=getattr(obj,name).shape
            if arr.shape != expected_shape:
                raise ValueError(f'capability field array shape mismatch for {name}: checkpoint={arr.shape} current={expected_shape}')
            setattr(obj,name,arr.copy())
        obj.last_step=int(d['last_step']); obj.total_updates=int(d['total_updates'])
        obj.total_node_events=int(d['total_node_events']); obj.total_recovery_events=int(d['total_recovery_events'])
        obj.total_edge_events=int(d['total_edge_events'])
        return obj

    def fingerprint(self) -> str:
        """Stable digest of all serialized field state.

        Held-out evaluation uses this as a quarantine invariant: asking the
        field which commands to examine is allowed; any mutation caused by the
        exam itself is a hard error.
        """
        h=hashlib.sha256()
        h.update(str(float(self.half_life_steps)).encode('ascii'))
        h.update(str((self.last_step,self.total_updates,self.total_node_events,self.total_recovery_events,self.total_edge_events)).encode('ascii'))
        for name in (
            'node_success','node_failure','node_quality_sum','node_slow','node_last_step',
            'rec_success','rec_failure',
            'edge_success','edge_failure','edge_rate_sum','edge_slow'):
            a=np.ascontiguousarray(getattr(self,name))
            h.update(name.encode('ascii')); h.update(str(a.shape).encode('ascii')); h.update(str(a.dtype).encode('ascii')); h.update(a.tobytes())
        return h.hexdigest()

    def save_npz(self, path: str | Path) -> None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,
            field_schema=np.asarray([FIELD_VERSION],np.int32),
            step=np.asarray([self.last_step],np.int64),half_life_steps=np.asarray([self.half_life_steps],np.float64),
            vx=self.centers[0],vy=self.centers[1],yaw=self.centers[2],
            node_competence=self.node_competence(),node_lower=self.node_lower(),node_evidence=self.node_evidence(),
            node_quality=self.node_quality(),node_confidence=self.node_confidence(),node_slow=self.node_slow,
            node_last_step=self.node_last_step,
            recovery_competence=self.recovery_competence(),recovery_lower=self.recovery_lower(),
            recovery_evidence=self.recovery_evidence(),
            edge_competence=self.edge_competence(),edge_lower=self.edge_lower(),edge_evidence=self.edge_evidence(),
            edge_rate=self.edge_rate(),edge_dirs=EDGE_DIRS)

    def write_summary(self, path: str | Path, *, step: int, mode: str, node_frac: float, edge_frac: float) -> None:
        payload={'step':int(step),'mode':str(mode),'node_influence':float(node_frac),'edge_influence':float(edge_frac),**self.summary()}
        Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
