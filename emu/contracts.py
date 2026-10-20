"""Cross-module shape and packet contracts for modern EMU.

These constants describe interfaces shared by environment, trainer and capability
field. They live here so JAX shapes and host-side teaching packets cannot drift
independently.
"""
from __future__ import annotations

from types import SimpleNamespace

SIM_DT = 0.005
CONTROL_SUBSTEPS = 4
CONTROL_DT = SIM_DT * CONTROL_SUBSTEPS

ACTION_DIM = 6
COMBO_P = 1.35
PHASE_RATE_MAX_HZ = 2.6
PITCH_CMD_MAX_RAD = 0.14  # ~8 deg operator-requested chassis pitch relative to gravity
FIELD_EDGE_SCALES = (0.90, 0.55, 3.0)  # vx, vy, yaw command geometry
SEED_VX_LO = 0.020
SEED_VX_HI = 0.060

# Shared mature recovery-birth envelope. Training and held-out "chaos" probes
# use the same physical maxima so the exam describes the learned disturbance
# family rather than a second silently drifting definition.
RECOVERY_TILT_MAX_DEG = 28.0
RECOVERY_POSITION_MAX_M = 0.025
RECOVERY_LINVEL_MAX_MPS = 0.65
RECOVERY_ANGVEL_MAX_RAD_S = 3.6
RECOVERY_JOINT_MAX_DEG = 14.0

# Exact deployable actor-observation order.  Names are deliberately explicit:
# changing this tuple is an actor-interface/schema change, not a local refactor.
ACTUATED_JOINTS = ("r_hip_roll", "r_hip_pitch", "r_knee", "l_hip_roll", "l_hip_pitch", "l_knee")
BASE_OBS_FIELDS: tuple[str, ...] = (
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_x", "gyro_y", "gyro_z",
    *(f"q_{j}" for j in ACTUATED_JOINTS),
    *(f"qv_{j}" for j in ACTUATED_JOINTS),
    *(f"prev_action_{j}" for j in ACTUATED_JOINTS),
    *(f"prev_ctrl_{j}" for j in ACTUATED_JOINTS),
    "foot_r_x", "foot_r_y", "foot_r_z", "foot_l_x", "foot_l_y", "foot_l_z",
    "vel_x", "vel_y", "tof_height", "vel_confidence",
    "cmd_vx", "cmd_vy", "cmd_yaw", "cmd_height", "cmd_pitch", "cmd_roll", "cmd_step_height",
    "phase_sin", "phase_cos",
    "shin_flex_r", "shin_flex_l", "shin_rate_r", "shin_rate_l", "shin_load_r", "shin_load_l",
)
BASE_OBS_DIM = len(BASE_OBS_FIELDS)
HISTORY_LEN = 4
OBS_DIM = BASE_OBS_DIM * HISTORY_LEN

# Explicit opt-in v5016 policy interface. Never reinterpret an archived
# 55x4 Pitch15 actor or its observation normalizer as this 58x4 interface.
OBS_SCHEMA_LEGACY = "pitch15_220"
OBS_SCHEMA_SENSOR = "sim2real_232"
SENSOR_OBS_FIELDS: tuple[str, ...] = BASE_OBS_FIELDS + (
    "ground_forward_grade_rad", "ground_sideways_grade_rad",
    "ground_visibility_confidence",
)
SENSOR_BASE_OBS_DIM = len(SENSOR_OBS_FIELDS)
SENSOR_OBS_DIM = SENSOR_BASE_OBS_DIM * HISTORY_LEN
# Observation schema identifies meanings, not merely array widths. A hash
# also covers timing and physical sensor extrinsics when using sim2real_232.
SENSOR_MODEL_REVISION = "tripod_tof_flow_async_v1"
SENSOR_SETTINGS_DEFAULTS = {
    "tof_period_frames": 2,
    "flow_period_frames": 1,
    "tof_delay_frames": 1,
    "flow_delay_frames": 1,
    "tof_dropout_prob": 0.01,
    "flow_dropout_prob": 0.02,
    "gyro_bias_std_rad_s": 0.015,
    "sensor_noise_scale": 1.0,
}


def sensor_observation_contract(settings: dict | None = None) -> dict:
    """Canonical, independently verifiable v5016 observation identity.

    Sensor mount geometry and fields are part of the meaning of a saved
    policy; a fingerprint cannot be self-certified by just rehashing an
    arbitrary serialized manifest.
    """
    import hashlib
    import json
    import math
    from .directional import (TOF_TRIPOD_MOUNTS_BODY_M, TOF_TRIPOD_BEAMS_BODY,
                              TOF_MIN_RANGE_M, TOF_MAX_RANGE_M)

    values = dict(SENSOR_SETTINGS_DEFAULTS if settings is None else settings)
    if set(values) != set(SENSOR_SETTINGS_DEFAULTS):
        raise ValueError("sensor observation settings keys mismatch")
    for key in ("tof_period_frames", "flow_period_frames"):
        if type(values[key]) is not int or not 1 <= values[key] <= 20:
            raise ValueError(f"invalid {key}")
    for key in ("tof_delay_frames", "flow_delay_frames"):
        if type(values[key]) is not int or not 0 <= values[key] < HISTORY_LEN:
            raise ValueError(f"invalid {key}")
    for key in ("tof_dropout_prob", "flow_dropout_prob"):
        x = values[key]
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError(f"invalid {key}")
    for key in ("gyro_bias_std_rad_s", "sensor_noise_scale"):
        x = values[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError(f"invalid {key}")
    body = {
        "schema": OBS_SCHEMA_SENSOR,
        "sensor_model_revision": SENSOR_MODEL_REVISION,
        "base_fields": list(SENSOR_OBS_FIELDS),
        "history_len": HISTORY_LEN,
        "action_dim": ACTION_DIM,
        "control_dt": CONTROL_DT,
        "tripod_mounts_body_m": [list(p) for p in TOF_TRIPOD_MOUNTS_BODY_M],
        "tripod_beams_body": [list(p) for p in TOF_TRIPOD_BEAMS_BODY],
        "tof_range_limits_m": [TOF_MIN_RANGE_M, TOF_MAX_RANGE_M],
        "settings": values,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**body, "sha256": digest}


OBS_FIELDS_BY_SCHEMA = {
    OBS_SCHEMA_LEGACY: BASE_OBS_FIELDS,
    OBS_SCHEMA_SENSOR: SENSOR_OBS_FIELDS,
}


def actor_observation_fields(schema: str) -> tuple[str, ...]:
    """Return the exact fields of an explicitly named actor contract."""
    try:
        return OBS_FIELDS_BY_SCHEMA[schema]
    except KeyError as exc:
        raise ValueError(f"unknown actor observation schema {schema!r}") from exc

# Preserve the current actor coordinate system exactly.  The membership here is
# intentionally explicit rather than encoded as fragile numeric slices.
OBS_NORMALIZED_FIELDS = frozenset(
    BASE_OBS_FIELDS[3:18] + BASE_OBS_FIELDS[24:44] + BASE_OBS_FIELDS[46:53]
)

FIELD_SHAPE = (21, 15, 25)
FIELD_NODE_ANCHORS = 24
FIELD_EDGE_ANCHORS = 24
FIELD_NODE_STRIDE = 5   # vx, vy, yaw, weight, recommended recovery severity
FIELD_EDGE_STRIDE = 7   # source vx/vy/yaw, target vx/vy/yaw, weight

# Base portion of the fixed-shape JAX task packet. The environment packs values
# in exactly this order; P exposes the corresponding trace-time indices.
TASK_PARAM_FIELDS: tuple[tuple[str, float], ...] = (
    ("steady_difficulty", 0.0),
    ("transition_difficulty", 0.0),
    ("resilience_difficulty", 0.0),
    ("posture_difficulty", 0.0),
    ("regime_frac_steady", 1.0),
    ("regime_frac_transition", 0.0),
    ("regime_frac_resilience", 0.0),
    ("regime_command_frac", 0.5),
    ("athletic_rehearsal_frac", 0.0),
    ("frontier_vx_hi", 0.60),
    ("frontier_vx_lo", -0.20),
    ("frontier_vy", 0.22),
    ("frontier_yaw", 1.50),
    ("frontier_push_n", 0.70),
    ("frontier_sensor_scale", 1.50),
    ("frontier_combo_radius", 1.0),
    ("frontier_transition_hold_s", 0.90),
    ("frontier_ramp_steps", 4.0),
    ("frontier_terrain_height_m", 0.003),
    ("probe_cmd_vx", float("nan")),
    ("probe_cmd_vy", float("nan")),
    ("probe_cmd_yaw", float("nan")),
    ("probe_tilt_rad", -1.0),
    ("probe_linvel_mps", -1.0),
    ("probe_angvel_rad_s", -1.0),
    ("probe_joint_rad", -1.0),
    ("probe_push_n", -1.0),
    ("probe_push_steps", 5.0),
)

_base_index = {name: i for i, (name, _) in enumerate(TASK_PARAM_FIELDS)}
FIELD_NODE_FRAC = len(TASK_PARAM_FIELDS)
FIELD_NODE_START = FIELD_NODE_FRAC + 1
FIELD_EDGE_FRAC = FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE
FIELD_EDGE_START = FIELD_EDGE_FRAC + 1
TASK_PARAM_DIM = FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE
P = SimpleNamespace(**_base_index, field_node_frac=FIELD_NODE_FRAC, field_edge_frac=FIELD_EDGE_FRAC)


def task_param_defaults() -> dict[str, float]:
    """Canonical host-side defaults for the fixed JAX task packet."""
    return dict(TASK_PARAM_FIELDS)


# Physical tracking/recovery tolerances shared by teacher and independent examiner.
import math
import numpy as np

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


def node_window_success_numpy(cmd_vx,cmd_vy,cmd_yaw,vx,vy,yaw_rate,gravity_z,support,bad_contact,fell=None,axis=0,structural_attitude_ok=None):
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
    posture=(np.asarray(structural_attitude_ok)>.5 if structural_attitude_ok is not None
             else np.asarray(gravity_z)>UPRIGHT_GRAVITY_Z_MIN)
    structural=(np.mean(posture,axis=axis)>=NODE_STRUCTURAL_FRACTION_MIN) & \
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


# Package paths and recorded trainer revision: physical/source contract.
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VERSION_FILE = PACKAGE_ROOT / "VERSION"
ROBOT_XML = PACKAGE_ROOT / "robot.xml"
VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()


def source_repo_root() -> Path | None:
    """Return the checkout root when running from source, otherwise None."""
    root = PACKAGE_ROOT.parent
    return root if (root / "pyproject.toml").exists() else None


def run_base() -> Path:
    repo = source_repo_root()
    default = repo.parent / "emu_runs" if repo is not None else Path.home() / "emu_runs"
    return Path(os.environ.get("EMU_RUN_BASE", default)).expanduser().resolve()


def results_base() -> Path:
    repo = source_repo_root()
    default = repo / "results" if repo is not None else Path.home() / "emu_results"
    return Path(os.environ.get("EMU_RESULTS_BASE", default)).expanduser().resolve()


def seed_root(seed: int) -> Path:
    return run_base() / f"{VERSION}_seed{int(seed)}"
