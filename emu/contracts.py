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
# The 55 base telemetry channels are a prefix of the ONE active actor,
# not a separately loadable historical 220-input model.
BASE_OBS_DIM = len(BASE_OBS_FIELDS)
HISTORY_LEN = 4
OBS_SCHEMA_TERRAIN = "terrain_360"
SENSOR_OBS_FIELDS = BASE_OBS_FIELDS + (
    "ground_forward_grade_rad", "ground_sideways_grade_rad",
    "ground_visibility_confidence",
)
TERRAIN_RANGE_FIELDS = tuple(
    f"tof4x4_range_{r}{c}_m" for r in range(4) for c in range(4))
TERRAIN_VALID_FIELDS = tuple(
    f"tof4x4_valid_{r}{c}" for r in range(4) for c in range(4))
TERRAIN_OBS_FIELDS = SENSOR_OBS_FIELDS + TERRAIN_RANGE_FIELDS + TERRAIN_VALID_FIELDS
TERRAIN_BASE_OBS_DIM = len(TERRAIN_OBS_FIELDS)
TERRAIN_OBS_DIM = TERRAIN_BASE_OBS_DIM * HISTORY_LEN
OBS_DIM = TERRAIN_OBS_DIM  # one 360-input actor everywhere
TERRAIN_ARRAY_ONLY_REVISION = "vl53l8cx_array_plane_square45_below30_v3"
TERRAIN_TOF_MIN_M, TERRAIN_TOF_MAX_M = 0.04, 1.5
# Physical 4x4 optical array. The shape is unchanged; its MEANING is new.
from .tof_geometry import (TOF_MOUNT_BODY_M as TERRAIN_TOF_MOUNT_BODY_M,
                           TOF_RAYS_BODY as TERRAIN_TOF_RAYS_BODY,
                           TOF_FOV_DEG, TOF_TILT_DOWN_DEG, TOF_YAW_DEG,
                           TOF_HARDWARE_ZONE_IDS)
# Observation identity covers real mounting, image-plane rays and timing.
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


def sensor_observation_contract(settings: dict | None = None, *,
                                schema: str = OBS_SCHEMA_TERRAIN,
                                array_only: bool = True) -> dict:
    """Exactly one physical 4x4 observation contract, including its SHA.

    The schema and array-only arguments are validated at the boundary so
    training/checkpoint callers cannot accidentally request an older format.
    No older sensor/actor implementation is retained here.
    """
    import hashlib
    import json
    import math

    if schema != OBS_SCHEMA_TERRAIN or array_only is not True:
        raise ValueError('Only the physical terrain_360 array actor is supported')
    values = dict(SENSOR_SETTINGS_DEFAULTS if settings is None else settings)
    if set(values) != set(SENSOR_SETTINGS_DEFAULTS):
        raise ValueError('Sensor timing/noise settings keys mismatch')
    for key in ('tof_period_frames', 'flow_period_frames'):
        if type(values[key]) is not int or not 1 <= values[key] <= 20:
            raise ValueError(f'invalid {key}')
    for key in ('tof_delay_frames', 'flow_delay_frames'):
        if type(values[key]) is not int or not 0 <= values[key] < HISTORY_LEN:
            raise ValueError(f'invalid {key}')
    for key in ('tof_dropout_prob', 'flow_dropout_prob'):
        x = values[key]
        if type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError(f'invalid {key}')
    for key in ('gyro_bias_std_rad_s', 'sensor_noise_scale'):
        x = values[key]
        if type(x) not in (int,float) or not math.isfinite(x) or x < 0:
            raise ValueError(f'invalid {key}')
    body = {
        'schema': OBS_SCHEMA_TERRAIN,
        'sensor_model_revision': TERRAIN_ARRAY_ONLY_REVISION,
        'base_fields': list(TERRAIN_OBS_FIELDS),
        'history_len': HISTORY_LEN,
        'action_dim': ACTION_DIM,
        'control_dt': CONTROL_DT,
        'settings': values,
        'tof4x4_mount_body_m': list(TERRAIN_TOF_MOUNT_BODY_M),
        'tof4x4_fov_horizontal_deg': TOF_FOV_DEG,
        'tof4x4_fov_vertical_deg': TOF_FOV_DEG,
        'tof4x4_tilt_down_deg': TOF_TILT_DOWN_DEG,
        'tof4x4_mount_yaw_deg': TOF_YAW_DEG,
        'tof4x4_rays_body': [list(v) for v in TERRAIN_TOF_RAYS_BODY],
        'tof4x4_hardware_zone_ids': list(TOF_HARDWARE_ZONE_IDS),
        'tof4x4_range_limits_m': [TERRAIN_TOF_MIN_M,TERRAIN_TOF_MAX_M],
        'tof4x4_occlusion': 'nearest ray-sphere foot hit masks terrain return',
        'tof4x4_validity': 'one bit per cell, delayed, zero on dropout',
        'array_ground_estimator': 'weighted_4x4_plane_residual_8mm_v1',
        'array_ground_max_fit_residual_m': .008,
        'array_ground_min_valid_cells': 6,
    }
    digest = hashlib.sha256(json.dumps(
        body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {**body,'sha256':digest}


def actor_observation_fields(schema: str = OBS_SCHEMA_TERRAIN) -> tuple[str, ...]:
    if schema != OBS_SCHEMA_TERRAIN:
        raise ValueError('Only physical terrain_360 observations are supported')
    return TERRAIN_OBS_FIELDS


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

# G4: the existing flat/220/232 task vector is an immutable legacy contract.
# Only the opt-in finite-terrain environment appends this v1 extension AFTER
# all field anchors. No existing P indices or actor columns move.
TERRAIN_TASK_FIELDS = (("terrain_exposure", 0.0), ("terrain_height_m", 0.0))
TERRAIN_TASK_PARAM_DIM = TASK_PARAM_DIM + len(TERRAIN_TASK_FIELDS)
# Second independent extension ONLY for the new two-solid-slab world. The
# historical 360-input single-platform model keeps its exact task packet.
# NaN means "sample a fresh value from the adaptive episode envelope".
# The optional explicit values are for exact held-out physical geometry exams.
UNIFIED_TERRAIN_FIELDS = (
    ('slope_exposure',0.0), ('slope_max_deg',0.0),
    ('terrain_pitch_deg',float('nan')), ('terrain_roll_deg',float('nan')),
    ('terrain_fixed_height_m',float('nan')),
    ('terrain_edge_angle_deg',float('nan')),
    # Teacher-only foundation→growth blend, never an actor observation.
    ('specialist_mix',0.0),
    # Grace-only pre-growth teaching. All non-control specialists remain off.
    ('foundation_control_mix',0.0),
    # Six teacher-only probabilities; zero sum keeps historical exam mapping.
    *((f'focus_weight_{name}',0.0) for name in
      ('command','resilience','slope','step','transition','control')),
    ('steering_attention',0.0),
    # Teacher-only course assignment. Added AFTER historical G4 indices.
    ('academy_episode_probability',0.0),
    # Independent supported-up and supported-down frontiers, teacher only.
    ('academy_height_up_m',0.003),
    ('academy_height_down_m',0.003),
)
UNIFIED_TASK_PARAM_DIM = TERRAIN_TASK_PARAM_DIM + len(UNIFIED_TERRAIN_FIELDS)
P = SimpleNamespace(
    **_base_index,
    field_node_frac=FIELD_NODE_FRAC,
    field_edge_frac=FIELD_EDGE_FRAC,
    terrain_exposure=TASK_PARAM_DIM,
    terrain_height_m=TASK_PARAM_DIM + 1,
    slope_exposure=TERRAIN_TASK_PARAM_DIM,
    slope_max_deg=TERRAIN_TASK_PARAM_DIM + 1,
    terrain_pitch_deg=TERRAIN_TASK_PARAM_DIM + 2,
    terrain_roll_deg=TERRAIN_TASK_PARAM_DIM + 3,
    terrain_fixed_height_m=TERRAIN_TASK_PARAM_DIM + 4,
    terrain_edge_angle_deg=TERRAIN_TASK_PARAM_DIM + 5,
    specialist_mix=TERRAIN_TASK_PARAM_DIM + 6,
    foundation_control_mix=TERRAIN_TASK_PARAM_DIM + 7,
    **{f'focus_weight_{name}':TERRAIN_TASK_PARAM_DIM+8+i
       for i,name in enumerate(
           ('command','resilience','slope','step','transition','control'))},
    steering_attention=TERRAIN_TASK_PARAM_DIM+14,
    academy_episode_probability=TERRAIN_TASK_PARAM_DIM+15,
    academy_height_up_m=TERRAIN_TASK_PARAM_DIM+16,
    academy_height_down_m=TERRAIN_TASK_PARAM_DIM+17,
)


def task_param_defaults() -> dict[str, float]:
    """Canonical host-side defaults for the fixed JAX task packet."""
    return dict(TASK_PARAM_FIELDS)


# Physical tracking/recovery tolerances shared by teacher and independent examiner.
import math

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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
from .physical_variants import robot_xml_path
ROBOT_XML = robot_xml_path(PACKAGE_ROOT)
VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()


def source_repo_root() -> Path | None:
    """Return the checkout root when running from source, otherwise None."""
    root = PACKAGE_ROOT.parent
    return root if (root / "pyproject.toml").exists() else None


def run_base() -> Path:
    """Canonical run root, independent of checkout location or terminal cwd."""
    default = Path.home() / "emu-data" / "runs"
    return Path(os.environ.get("EMU_RUN_BASE", default)).expanduser().resolve()


def results_base() -> Path:
    """Disposable research output; per-run reports belong IN each run root."""
    default = Path.home() / "emu-data" / "scratch"
    return Path(os.environ.get("EMU_RESULTS_BASE", default)).expanduser().resolve()


def seed_root(seed: int) -> Path:
    return run_base() / f"{VERSION}_seed{int(seed)}"
