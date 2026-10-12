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
P = SimpleNamespace(**_base_index, field_node_frac=FIELD_NODE_FRAC, field_edge_frac=FIELD_EDGE_FRAC)


def task_param_defaults() -> dict[str, float]:
    """Canonical host-side defaults for the fixed JAX task packet."""
    return dict(TASK_PARAM_FIELDS)
