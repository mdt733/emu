"""Modern EMU training configuration.

The public CLI deliberately exposes only experiment-level choices. Trainer internals
receive one serialized TrainingConfig, never a reconstructed forest of CLI flags.

The mature profile intentionally freezes observation-normalizer statistics at 25M
absolute environment steps. That is roughly 5M steps after the 20M foundation target:
the hypothesis is that the actor's coordinate system should stop drifting before the
mature curriculum substantially expands its command envelope. The freeze is therefore
a deliberate profile choice, not a hidden fixed default.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
from pathlib import Path

from .contracts import (COMBO_P, PHASE_RATE_MAX_HZ, RECOVERY_TILT_MAX_DEG, RECOVERY_POSITION_MAX_M,
                        RECOVERY_LINVEL_MAX_MPS, RECOVERY_ANGVEL_MAX_RAD_S, RECOVERY_JOINT_MAX_DEG)
from .ppo import PPO_DOCTRINE


# Current EMU doctrine. These are intentionally not experiment knobs: every
# supported training profile uses the same values. A future experiment should
# promote one back into TrainingConfig only when it genuinely needs to vary.
FOUNDATION_STEPS = 20_000_000
ROLLOUT_STEPS = 32
PPO_EPOCHS = 4
PPO_MINIBATCHES = 8
PPO_LR = 2e-4
PPO_ENTROPY_COST = 0.002
PPO_VALUE_COST = 0.5
LOG_EVERY_UPDATES = 10
CHECKPOINT_EVERY_UPDATES = 20
SNAPSHOT_EVERY_STEPS = 2_000_000
SNAPSHOT_KEEP = 10
PROFILE_PERFORMANCE = True
MAX_EPISODE_STEPS = 700
FRONTIER_GROWTH_FRAC = 0.0015
FRONTIER_RETREAT_FRAC = 0.0040
FRONTIER_SENSOR_SAFETY = 4.0
OBS_NORM_MODE = "continuous"
PPO_TARGET_KL = 0.05
PPO_LR_SCALE_MIN = 0.01
PPO_LR_SCALE_INIT = 0.25
PPO_LR_SCALE_MAX = 4.0
PPO_LR_ADJUST = 1.5
EVAL_EVERY_STEPS = 60_000_000
EVAL_HORIZON = 350
EVAL_SEED = 1701
EVAL_COHORTS = 1
FINAL_EVAL_COHORTS = 2
CLOUD_EVAL_EVERY_STEPS = 40_000_000
CLOUD_ENVS = 256
CLOUD_HORIZON = 250
CLOUD_BATCH_POINTS = 4
FIELD_HALF_LIFE_STEPS = 20_000_000
SENSOR_NOISE_SCALE = 1.5


DOCTRINE = {
    # Fixed PPO algorithm/network doctrine owned by ppo.py.
    **PPO_DOCTRINE,
    # PPO / rollout shape.
    "FOUNDATION_STEPS": FOUNDATION_STEPS,
    "ROLLOUT_STEPS": ROLLOUT_STEPS,
    "PPO_EPOCHS": PPO_EPOCHS,
    "PPO_MINIBATCHES": PPO_MINIBATCHES,
    "PPO_LR": PPO_LR,
    "PPO_ENTROPY_COST": PPO_ENTROPY_COST,
    "PPO_VALUE_COST": PPO_VALUE_COST,
    "PPO_TARGET_KL": PPO_TARGET_KL,
    "PPO_LR_SCALE_MIN": PPO_LR_SCALE_MIN,
    "PPO_LR_SCALE_INIT": PPO_LR_SCALE_INIT,
    "PPO_LR_SCALE_MAX": PPO_LR_SCALE_MAX,
    "PPO_LR_ADJUST": PPO_LR_ADJUST,
    # Runtime / evaluation cadence.
    "LOG_EVERY_UPDATES": LOG_EVERY_UPDATES,
    "CHECKPOINT_EVERY_UPDATES": CHECKPOINT_EVERY_UPDATES,
    "SNAPSHOT_EVERY_STEPS": SNAPSHOT_EVERY_STEPS,
    "SNAPSHOT_KEEP": SNAPSHOT_KEEP,
    "PROFILE_PERFORMANCE": PROFILE_PERFORMANCE,
    "MAX_EPISODE_STEPS": MAX_EPISODE_STEPS,
    "EVAL_EVERY_STEPS": EVAL_EVERY_STEPS,
    "EVAL_HORIZON": EVAL_HORIZON,
    "EVAL_SEED": EVAL_SEED,
    "EVAL_COHORTS": EVAL_COHORTS,
    "FINAL_EVAL_COHORTS": FINAL_EVAL_COHORTS,
    "CLOUD_EVAL_EVERY_STEPS": CLOUD_EVAL_EVERY_STEPS,
    "CLOUD_ENVS": CLOUD_ENVS,
    "CLOUD_HORIZON": CLOUD_HORIZON,
    "CLOUD_BATCH_POINTS": CLOUD_BATCH_POINTS,
    # Fixed curriculum / field doctrine.
    "FRONTIER_GROWTH_FRAC": FRONTIER_GROWTH_FRAC,
    "FRONTIER_RETREAT_FRAC": FRONTIER_RETREAT_FRAC,
    "FRONTIER_SENSOR_SAFETY": FRONTIER_SENSOR_SAFETY,
    "FIELD_HALF_LIFE_STEPS": FIELD_HALF_LIFE_STEPS,
    "SENSOR_NOISE_SCALE": SENSOR_NOISE_SCALE,
    "OBS_NORM_MODE": OBS_NORM_MODE,
    # Shared physical / geometric contracts recorded with every run.
    "COMBO_P": COMBO_P,
    "PHASE_RATE_MAX_HZ": PHASE_RATE_MAX_HZ,
    "RECOVERY_TILT_MAX_DEG": RECOVERY_TILT_MAX_DEG,
    "RECOVERY_POSITION_MAX_M": RECOVERY_POSITION_MAX_M,
    "RECOVERY_LINVEL_MAX_MPS": RECOVERY_LINVEL_MAX_MPS,
    "RECOVERY_ANGVEL_MAX_RAD_S": RECOVERY_ANGVEL_MAX_RAD_S,
    "RECOVERY_JOINT_MAX_DEG": RECOVERY_JOINT_MAX_DEG,
}


def doctrine_dict() -> dict[str, object]:
    """Stable runtime constants recorded beside each run for provenance."""
    return dict(DOCTRINE)


@dataclass(frozen=True)
class TrainingConfig:
    # Runtime / PPO shape.
    steps: int = 30_000_000
    envs: int = 2048
    seed: int = 733
    device: str = "auto"
    run_dir: str = ""
    resume: str = ""

    # Tunable PPO economics. Original doctrine values are the defaults;
    # only a ladder arm with an explicit override changes them. Rollout length,
    # network structure and optimiser algorithm remain fixed contracts.
    ppo_lr: float = PPO_LR
    ppo_entropy_cost: float = PPO_ENTROPY_COST
    ppo_value_cost: float = PPO_VALUE_COST

    # Physical model / task distribution.
    push_max_force_n: float | None = None
    vx_hi_max: float = 0.45
    vx_lo_min: float = -0.15
    vy_max_mps: float = 0.18
    yaw_max_rad_s: float = 1.20
    phase_speed_gain: float = 0.0
    combo_mix_frac: float = 0.0
    combo_probe_frac: float = 0.08

    # v5014 hidden domain envelope.  All defaults are OFF so every existing
    # v5013 profile remains mechanically identical unless a new experiment
    # explicitly enables domain randomisation.  The actor never observes these
    # radii; they only control the distribution of physical models/worlds.
    domain_randomization: bool = False
    domain_robot_radius: float = 0.0
    domain_terrain_radius: float = 0.0
    domain_hardware_radius: float = 0.0

    # v5016 opt-in motor-effort teaching hypothesis (NOT hardware truth).
    # Zero preserves the v5015 reward exactly. A small positive value pays
    # less for repeated nominal overload and gross shaft-motion effort.
    # Compare only against a matched no-guidance branch after physics exams.
    motor_guidance_weight: float = 0.0

    # Opt-in real contact incline, uniform across all training worlds.
    # The historical tilted-gravity terrain randomizer remains separate.
    contact_slope_deg: float = 0.0

    # v5015 is explicitly opt-in; v5014 defaults remain reproducible.
    attitude_profile: str = "v5014"
    pitch_command_max_deg: float = 8.0214

    # Curriculum / physical frontier.
    fall_target: float | None = None
    difficulty_base_step: float | None = None
    regime_command_frac: float | None = None
    athletic_rehearsal_frac: float = 0.0
    open_frontier: bool = False
    frontier_vx_hi_safety: float = 1.0
    frontier_vx_lo_safety: float = 0.45
    frontier_vy_safety: float = 0.45
    frontier_yaw_safety: float = 3.0
    frontier_combo_safety: float = 1.60
    frontier_push_safety_n: float = 3.0

    # Observation normalisation / PPO health.
    freeze_obs_norm_after_steps: int = 0
    allow_rollback: bool = True

    # Held-out evaluation.
    cloud_probes_per_eval: int = 10
    cloud_final_probes: int = 50

    # Capability-field teaching.  These are the deliberately narrow experiment
    # surface for v5013: dose, family, evidence feedback, and schedule.  They
    # are config (not doctrine) because the tournament intentionally varies them.
    capability_field_mode: str = "off"
    field_max_influence: float = 0.65
    field_teach_nodes: bool = True
    field_teach_edges: bool = True
    field_teach_recovery: bool = True
    field_evidence_from_taught: bool = True
    field_influence_start_steps: int = 10_000_000
    field_influence_full_steps: int = 20_000_000
    field_influence_fade_start_steps: int = 0
    field_influence_zero_steps: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {f.name for f in dataclass_fields(cls)}
        # Historical v5014 configs may omit ONLY these new optional fields.
        optional = {'attitude_profile', 'pitch_command_max_deg',
                    'ppo_lr', 'ppo_entropy_cost', 'ppo_value_cost',
                    'motor_guidance_weight', 'contact_slope_deg'}
        missing, extra = expected - set(data), set(data) - expected
        if extra or (missing - optional):
            raise ValueError(f"training config keys mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        return cls(**data)


# These are run identity / motor-contract parameters, not experiment knobs.
# Algorithm-independent aspects (physical commands, teaching, curriculum,
# normalisation) are otherwise modifiable via TrainingConfig fields.
IMMUTABLE_RUN_FIELDS = frozenset({
    'steps', 'envs', 'seed', 'device', 'run_dir', 'resume', 'attitude_profile',
})


def apply_experiment_knobs(
    config: TrainingConfig,
    knobs: tuple[tuple[str, bool | int | float | str | None], ...],
) -> TrainingConfig:
    """Apply a ladder arm's explicit TrainingConfig overrides, fail closed.

    This is the ONLY translation from a human experiment recipe to the stable
    trainer's knobs. Unknown field, duplicated field, runtime identity override
    or incompatible value raises before training ever starts.
    """
    import math
    from typing import get_type_hints

    field_types = get_type_hints(TrainingConfig)
    valid = set(field_types)
    values = {}
    for key, value in knobs:
        if key not in valid:
            raise ValueError(f'Unknown TrainingConfig experiment knob: {key}')
        if key in IMMUTABLE_RUN_FIELDS:
            raise ValueError(f'Experiment cannot override run identity / contract: {key}')
        if key in values:
            raise ValueError(f'Duplicate experiment knob: {key}')
        declared = field_types[key]
        # Follow the declared TrainingConfig type, not the current numeric
        # value: several float fields can legitimately hold integer zero.
        if declared is bool:
            valid_value = isinstance(value, bool)
        elif declared is int:
            valid_value = isinstance(value, int) and not isinstance(value, bool)
        elif declared in (float, float | None):
            valid_value = (value is None and declared == float | None) or (
                isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(float(value)))
            if value is not None and valid_value:
                value = float(value)
        elif declared is str:
            valid_value = isinstance(value, str)
        else:
            raise TypeError(f'Unsupported config knob annotation: {key} ({declared})')
        if not valid_value:
            raise ValueError(f'Invalid value for experiment knob {key}: {value!r}')
        if key == 'ppo_lr' and value <= 0:
            raise ValueError('ppo_lr must be positive')
        if key == 'contact_slope_deg' and not 0.0 <= value <= 5.0:
            raise ValueError('contact_slope_deg must be between 0 and 5 degrees')
        if key == 'motor_guidance_weight' and not 0.0 <= value <= 0.10:
            raise ValueError('motor_guidance_weight must be between 0 and 0.10')
        if key in ('ppo_entropy_cost','ppo_value_cost') and value < 0:
            raise ValueError(f'{key} cannot be negative')
        values[key] = value
    return replace(config, **values)


def _runtime(*, run_dir: Path, steps: int, seed: int, device: str, envs: int, resume: Path | None) -> dict:
    return dict(
        run_dir=str(run_dir), steps=int(steps), seed=int(seed),
        device=str(device), envs=int(envs),
        resume=str(resume) if resume is not None else "",
    )


def foundation_config(*, run_dir: Path, seed: int, device: str, envs: int, resume: Path | None = None) -> TrainingConfig:
    return replace(
        TrainingConfig(), **_runtime(run_dir=run_dir, steps=FOUNDATION_STEPS, seed=seed, device=device, envs=envs, resume=resume),
        difficulty_base_step=0.0014,
        vx_hi_max=0.40, vx_lo_min=-0.12, vy_max_mps=0.14, yaw_max_rad_s=0.90,
        phase_speed_gain=0.75, regime_command_frac=0.60,
        push_max_force_n=0.70,
        cloud_probes_per_eval=0, cloud_final_probes=0,
        capability_field_mode="shadow", allow_rollback=False,
        field_influence_start_steps=10_000_000, field_influence_full_steps=FOUNDATION_STEPS,
    )


def mature_config(mode: str, *, run_dir: Path, steps: int, seed: int, device: str, envs: int,
                  resume: Path | None) -> TrainingConfig:
    if mode not in {"shadow", "graph"}:
        raise ValueError(f"unsupported trainer mode: {mode}")
    return replace(
        TrainingConfig(), **_runtime(run_dir=run_dir, steps=steps, seed=seed, device=device, envs=envs, resume=resume),
        freeze_obs_norm_after_steps=25_000_000,
        difficulty_base_step=0.0024,
        vx_hi_max=0.60, vx_lo_min=-0.20, vy_max_mps=0.22, yaw_max_rad_s=1.50,
        phase_speed_gain=1.25, regime_command_frac=0.85,
        push_max_force_n=3.2,
        open_frontier=True,
        athletic_rehearsal_frac=0.10, combo_mix_frac=0.88, combo_probe_frac=0.12,
        frontier_vx_hi_safety=1.20, frontier_vx_lo_safety=0.60,
        frontier_vy_safety=0.60, frontier_yaw_safety=4.50,
        frontier_combo_safety=2.00, frontier_push_safety_n=3.2,
        fall_target=0.34,
        capability_field_mode=mode,
        field_influence_start_steps=FOUNDATION_STEPS, field_influence_full_steps=30_000_000,
    )

# Phase recipe and existing-run restart compatibility
from dataclasses import asdict, replace
from pathlib import Path

def _phase_config(name, target, r, t, h, *, root, seed, device, envs, resume, profile='v5014', pitch_deg=8.0214):
    run_dir = root / name
    if target <= 20_000_000:
        return replace(foundation_config(run_dir=run_dir, seed=seed, device=device, envs=envs, resume=resume),
                       attitude_profile=profile, pitch_command_max_deg=pitch_deg)
    cfg = mature_config('graph', run_dir=run_dir, steps=target, seed=seed, device=device, envs=envs, resume=resume)
    return replace(
        cfg,
        attitude_profile=profile, pitch_command_max_deg=pitch_deg,
        capability_field_mode='graph',
        field_max_influence=0.10,
        field_teach_nodes=True, field_teach_edges=True, field_teach_recovery=True,
        field_evidence_from_taught=True,
        # v5013's clearest reproducible lesson: low dose. Keep teacher and
        # physical-domain severity independent.
        field_influence_start_steps=20_000_000,
        field_influence_full_steps=30_000_000,
        domain_randomization=max(r,t,h) > 0.0,
        domain_robot_radius=r, domain_terrain_radius=t, domain_hardware_radius=h,
    )


def verify_existing_phase_config(run_dir: Path, expected) -> None:
    """Refuse a same-path continuation with a different physical/learning contract.

    A perfectly valid schema-7 checkpoint can still be from a different seed,
    env count, field teacher or hidden physical distribution. Prevent accidental
    reuse before `resolve_target_checkpoint` decides that an old phase is done.
    Hardware choice, source checkout location and resume pathname are not
    learning-contract parameters and may legitimately differ on restart.
    """
    ck = run_dir / 'checkpoints'
    if not ck.is_dir() or not any(ck.glob('*.pkl')):
        return

    cfg_path = run_dir / 'training_config.json'
    if not cfg_path.is_file():
        raise RuntimeError(f'Found checkpoints in {run_dir} without training_config.json; refusing ambiguous lineage reuse')
    actual = asdict(TrainingConfig.load(cfg_path))
    wanted = asdict(expected)
    for ignored in ('resume', 'device', 'run_dir'):
        actual.pop(ignored, None)
        wanted.pop(ignored, None)
    differing = {key: (actual.get(key), wanted[key]) for key in wanted if actual.get(key) != wanted[key]}
    if differing:
        raise RuntimeError(f'Existing phase {run_dir.name} has a different training configuration: {differing}. '
                           'Use the original options, or choose a fresh --root; do not mix checkpoint lineages.')
