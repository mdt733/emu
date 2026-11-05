"""Single physical-prism frozen-policy examiner and contact scorer.

Never read reward, academy crossing flags, or teacher attempt classifications
when determining an independently supported physical traversal.
"""
from __future__ import annotations

import numpy as np

from .terrain_carpet_mjx import TILE_M, TILE_COUNT
from .carpet_prism_mjx import PRISM_COUNT


def first_episode_mask(done):
    """Count the terminal frame, never a reset-policy frame after death.

    This safety rule now lives with the CURRENT prism examiner rather than
    importing an obsolete 9-box or three-slab physical exam.
    """
    ended=np.asarray(done,dtype=bool)
    if ended.ndim!=2:
        raise ValueError('Expected [frames, worlds] terminal array')
    earlier=np.concatenate((
        np.zeros((1,ended.shape[1]),dtype=np.int32),
        np.cumsum(ended[:-1],axis=0)),axis=0)
    return earlier==0


def score_prism_crossings(frames, *, required_frames=5):
    """Count first-life, new-cell crossings with sustained true prism top contact.

    Expected arrays are PRE-step MJX observations, consistently time-aligned:
      carpet_root_cell            [time, world, 2] (global integer cells)
      carpet_prism_corner_z       [time, world, TILE_COUNT, 4] (metres, physical XY corners)
      carpet_raw_contact_pairs    [time, world, PRISM_COUNT, 2]
      carpet_contact_vertical     [time, world, PRISM_COUNT, 2] (abs normal Z, real MJX)
      foot_xyz                    [time, world, 2, 3] (sphere centres)
      initial_foot_radii_m        [2]
      done, fell                  [time, world]

    The source heights are independently compared against the PRISM_COUNT *physical*
    convex-mesh corner positions in the prism acceptance probe; reward flags
    are never used as scoring inputs.
    """

    cells = np.asarray(frames["carpet_root_cell"], dtype=np.int32)
    tops = np.asarray(frames["carpet_prism_corner_z"], dtype=float)
    pairs = np.asarray(frames["carpet_raw_contact_pairs"], dtype=bool)
    vertical = np.asarray(frames["carpet_contact_vertical"], dtype=float)
    feet = np.asarray(frames["foot_xyz"], dtype=float)
    radii = np.asarray(frames["initial_foot_radii_m"], dtype=float)
    done = np.asarray(frames["done"], dtype=bool)
    fell = np.asarray(frames["fell"], dtype=bool)
    if cells.ndim != 3 or cells.shape[-1] != 2:
        raise ValueError("Prism trace needs [time, worlds, 2] root cells")
    time, worlds, _ = cells.shape
    shapes = {
        "carpet_prism_corner_z": (time, worlds, TILE_COUNT, 4),
        "carpet_raw_contact_pairs": (time, worlds, PRISM_COUNT, 2),
        "carpet_contact_vertical": (time, worlds, PRISM_COUNT, 2),
        "foot_xyz": (time, worlds, 2, 3),
        "done": (time, worlds),
        "fell": (time, worlds),
        "initial_foot_radii_m": (2,),
    }
    values = dict(carpet_prism_corner_z=tops,
                  carpet_raw_contact_pairs=pairs,
                  carpet_contact_vertical=vertical,
                  foot_xyz=feet, done=done, fell=fell,
                  initial_foot_radii_m=radii)
    for name, shape in shapes.items():
        if values[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {values[name].shape}")
    if not (1 <= required_frames <= 30):
        raise ValueError("required_frames must be 1..30")
    if not (np.isfinite(tops).all() and np.isfinite(vertical).all()
            and np.isfinite(feet).all() and np.isfinite(radii).all()
            and np.all((radii > 0.) & (radii < .10))):
        raise ValueError("Nonfinite or invalid physical prism trace")

    valid = first_episode_mask(done)

    def slot(cell):
        return int(cell[0] % 2 + 2 * (cell[1] % 2))

    def triangle_height(t, w, cell, xy):
        x = float(xy[0] / TILE_M - cell[0])
        y = float(xy[1] / TILE_M - cell[1])
        # A sphere centre should be over the actual physical tile footprint.
        inside = (-.001 <= x <= 1.001 and -.001 <= y <= 1.001)
        if not inside:
            return 0., False
        a, b, c, d = tops[t, w, slot(cell)]
        if x + y <= 1.:
            z = a * (1. - x - y) + b * x + c * y
        else:
            z = d * (x + y - 1.) + b * (1. - y) + c * (1. - x)
        return float(z), True

    def edge_xy(a, b):
        dx, dy = int(b[0] - a[0]), int(b[1] - a[1])
        if abs(dx) + abs(dy) != 1:
            raise ValueError("Only orthogonally adjacent tiles share a lip")
        if dx:
            x = max(a[0], b[0]) * TILE_M
            y = (a[1] + .5) * TILE_M
        else:
            x = (a[0] + .5) * TILE_M
            y = max(a[1], b[1]) * TILE_M
        return np.array([x, y], dtype=float)

    per_world = []
    for w in range(worlds):
        confirmed = tuple(int(v) for v in cells[0, w])
        visited = {confirmed}
        candidate = None
        streak = 0
        observed = []
        smooth = 0
        for t in range(time):
            if not valid[t, w]:
                break
            target = tuple(int(v) for v in cells[t, w])
            dx = target[0] - confirmed[0]
            dy = target[1] - confirmed[1]
            if abs(dx) + abs(dy) != 1 or target in visited:
                candidate, streak = None, 0
                continue
            if candidate != target:
                candidate, streak = target, 0
            target_slot = slot(target)
            supported = False
            for f in range(2):
                # Two triangles per tile, each a REAL convex MJX collider.
                contact_ok = any(
                    pairs[t, w, 2 * target_slot + tri, f] and
                    vertical[t, w, 2 * target_slot + tri, f] >= .70
                    for tri in (0, 1))
                if not contact_ok:
                    continue
                surface, inside = triangle_height(
                    t, w, target, feet[t, w, f, :2])
                if inside and abs(
                    feet[t, w, f, 2] - radii[f] - surface) <= .009:
                    supported = True
                    break
            if not supported or fell[t, w]:
                streak = 0
                continue
            streak += 1
            if streak < required_frames:
                continue
            edge = edge_xy(confirmed, target)
            old_height, ok_old = triangle_height(t, w, confirmed, edge)
            new_height, ok_new = triangle_height(t, w, target, edge)
            if not (ok_old and ok_new):
                raise AssertionError("Measured physical lip edge is missing")
            rise = new_height - old_height
            if abs(rise) > .0015:
                observed.append(dict(
                    from_cell=list(confirmed), to_cell=list(target),
                    signed_height_cm=round(rise * 100., 4), frame=t))
            else:
                smooth += 1
            confirmed = target
            visited.add(confirmed)
            candidate, streak = None, 0
        per_world.append(dict(
            world=w,
            independent_up=sum(e["signed_height_cm"] > 0 for e in observed),
            independent_down=sum(e["signed_height_cm"] < 0 for e in observed),
            smooth_grade_transitions=smooth,
            independently_supported_lips=observed))
    return dict(
        up=sum(w["independent_up"] for w in per_world),
        down=sum(w["independent_down"] for w in per_world),
        total=sum(w["independent_up"] + w["independent_down"] for w in per_world),
        worlds=per_world)


def evaluate_prism_policy(*, policy, output, worlds=4, horizon=300,
                          speed=.12, seed=1701, device="cpu", mode="mixed",
                          grade_deg=3., lip_cm=1., phase_probe="normal", tof_probe="normal"):
    """Independent frozen 360-input actor on the same physical prism topology.

    Unlike the retired two-slab matrix, terrain is procedurally sampled here:
    requested frontiers are *teacher settings*, and observed geometry/crossings
    are always reported separately. No PPO updates or lesson promotion.
    """
    import hashlib
    import json
    import math
    import time
    from pathlib import Path
    from .contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN, CONTROL_DT
    from .policy_export import load_exported_actor_for_exam
    from .ppo import configure_jax, require_backend, make_trainer
    from .training_config import (
        PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST,
    )
    from .environment import EmuMJXEnv

    policy = Path(policy).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix != ".emupolicy":
        raise ValueError("prism exam needs a verified .emupolicy file")
    if (output == policy.parent or output == policy or
            output in policy.parents or policy in output.parents):
        raise ValueError("exam results must be separate from the policy")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("prism exam output must be new or empty")
    if (type(worlds) is not int or not 2 <= worlds <= 32 or
            type(horizon) is not int or not 100 <= horizon <= 650 or
            mode not in ("flat", "slope", "step", "mixed") or
            not 0. <= speed <= 1.50 or
            not 0. <= grade_deg <= 15. or
            not 0. <= lip_cm <= 8. or
            not all(math.isfinite(x) for x in (speed, grade_deg, lip_cm))):
        raise ValueError("invalid prism exam worlds, horizon, speed or terrain frontiers")
    if mode in ("slope", "mixed") and grade_deg <= 0.:
        raise ValueError("slope/mixed exam requires a positive grade")
    if mode in ("step", "mixed") and lip_cm <= 0.:
        raise ValueError("step/mixed exam requires a positive lip")

    if phase_probe not in ("normal","hold","invert","random") or tof_probe not in ("normal","mask"):
        raise ValueError("Invalid frozen-policy observation intervention")
    params, norm, actor = load_exported_actor_for_exam(policy)
    if (actor["observation_schema"] != OBS_SCHEMA_TERRAIN or
            not actor["array_only"] or
            actor["training"].get("terrain_carpet_prisms") is not True):
        raise ValueError("prism exam requires a true-prism 360-input source actor")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device == "cpu")

    train = actor["training"]
    control = actor["control"]
    sensor = actor["sensor_settings"]
    vi = control.get("velocity_i") or {}
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=worlds, seed=seed,
        max_episode_steps=700, push_max_force_n=0.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08, unified_capability_focus=True,
        terrain_carpet=True, terrain_carpet_prisms=True,
        solver_iterations=train.get("solver_iterations"),
        solver_ls_iterations=train.get("solver_ls_iterations"),
        academy_stream=True, academy_height_cap_m=.15,
        terrain_lesson="coherent-careful",
        attitude_profile=train["attitude_profile"],
        pitch_command_max_deg=float(train["pitch_command_max_deg"]),
        phase_speed_gain=float(control["phase_speed_gain"]),
        velocity_i_ramp_s=float(vi.get("ramp_s", .4)),
        velocity_i_multiplier=float(vi.get("multiplier", 1.)),
        sensor_noise_scale=float(sensor.get("sensor_noise_scale", 0.)),
        **{k: v for k, v in sensor.items() if k != "sensor_noise_scale"},
    )
    from .contracts import sensor_observation_contract
    expected = sensor_observation_contract(
        sensor, schema=OBS_SCHEMA_TERRAIN, array_only=True)
    if env.sensor_contract != expected:
        raise RuntimeError("frozen prism examiner changed actor sensor semantics")
    trainer = make_trainer(
        env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
        epochs=1, minibatches=1,
        entropy_cost=PPO_ENTROPY_COST, value_cost=PPO_VALUE_COST,
    )
    has_grade = mode in ("slope", "mixed")
    has_step = mode in ("step", "mixed")
    lip_m = lip_cm / 100.
    # The generator samples independently signed real X/Y lips. We do NOT
    # manufacture slab fixtures to force a nominal grade or step height.
    teacher = dict(
        specialist_mix=1., athletic_rehearsal_frac=0.,
        regime_frac_steady=1., regime_frac_transition=0.,
        regime_frac_resilience=0., probe_push_n=0.,
        probe_tilt_rad=0., probe_linvel_mps=0.,
        probe_angvel_rad_s=0., probe_joint_rad=0.,
        terrain_exposure=float(has_step), terrain_height_m=(lip_m if has_step else 0.),
        slope_exposure=float(has_grade), slope_max_deg=(grade_deg if has_grade else 0.),
        academy_height_up_m=max(.003, lip_m),
        academy_height_down_m=max(.003, lip_m),
        academy_episode_probability=float(has_step),
    )
    # A specialist role is sampled only at episode birth.
    for name in ("command", "resilience", "slope", "step", "transition", "control"):
        teacher["focus_weight_" + name] = (
            .5 if mode == "mixed" and name in ("step", "slope") else
            1. if name == ("slope" if mode == "slope" else
                           "step" if mode in ("step", "mixed") else "control")
            and mode != "mixed" else 0.
        )
    if not has_grade and not has_step:
        teacher["specialist_mix"] = 0.
    task = env.params_array(teacher)

    begun = time.perf_counter()
    # A fixed operator request, never policy-generated or inferred from a
    # rewarded task. The *same* frozen actor applies in every role.
    trace = trainer["evaluate_trace"](
        params, task,
        jnp.asarray(norm["mean"]), jnp.asarray(norm["std"]),
        jax.random.PRNGKey(seed), horizon, 0.,
        jnp.asarray([speed, 0., 0.], dtype=jnp.float32),
        phase_probe=phase_probe,tof_probe=tof_probe)
    measured = jax.device_get(trace)
    independent = score_prism_crossings(measured)
    # v5051: independent physical post-contact mission, not a reward counter.
    from .step_autopsy import inspect_missions
    autopsy = inspect_missions(measured)
    xy = np.asarray(measured["position_xy"], dtype=float)
    yaw = np.unwrap(np.asarray(measured["heading_rad"], dtype=float), axis=0)
    done = np.asarray(measured["done"], dtype=bool)
    fell = np.asarray(measured["fell"], dtype=bool)
    raw_pairs = np.asarray(measured["carpet_raw_contact_pairs"], dtype=bool)
    vertical = np.asarray(measured["carpet_contact_vertical"], dtype=float)
    support_feet = (raw_pairs & (vertical >= .70)).any(axis=2)  # [T,W,2]
    support = raw_pairs.any(axis=(2, 3))
    foot_xyz = np.asarray(measured["foot_xyz"], dtype=float)
    power = np.asarray(measured["servo_shaft_drive_power_w"], dtype=float)
    normals = np.asarray(measured["floor_normal_world"], dtype=float)
    grades = np.rad2deg(np.arccos(np.clip(normals[:, :, 2], -1., 1.)))
    settle = 25
    rows = []
    for w in range(worlds):
        terminals = np.flatnonzero(done[:, w])
        stop = int(terminals[0] + 1) if len(terminals) else horizon
        start = min(settle, stop - 1)
        dt = CONTROL_DT
        dxy = np.diff(xy[start:stop, w], axis=0) / dt
        heading = yaw[start:stop-1, w]
        body_vx = dxy[:, 0] * np.cos(heading) + dxy[:, 1] * np.sin(heading)
        body_vy = -dxy[:, 0] * np.sin(heading) + dxy[:, 1] * np.cos(heading)
        omega = np.diff(yaw[start:stop, w]) / dt
        body_vx_mean = float(np.mean(body_vx)) if len(body_vx) else None
        vx_error = float(np.mean(np.abs(body_vx - speed))) if len(body_vx) else None
        delta = xy[stop-1, w]-xy[start, w]
        travelled = float(np.sum(np.linalg.norm(
            np.diff(xy[start:stop, w], axis=0), axis=1)))
        progress = float(delta[0]*np.cos(yaw[start,w]) +
                         delta[1]*np.sin(yaw[start,w]))
        work = float(np.sum(np.maximum(0., power[:stop, w])) * dt)
        # Two frames of air followed by two frames of valid TOP contact.
        # Debounce contact chatter; count footfalls, not nominal policy phase.
        # Stride is distance between successive touchdown positions of the
        # SAME foot projected onto the observed heading. This is a gait
        # diagnostic, not a target rhythm or evidence of dynamic running.
        footfalls = 0
        same_foot_strides = []
        valid_foot_contact = support_feet[start:stop, w, :]
        for foot_index in range(2):
            prior_xy = None
            for ti in range(3, len(valid_foot_contact)):
                if (bool(valid_foot_contact[ti, foot_index]) and
                    bool(valid_foot_contact[ti-1, foot_index]) and
                    not bool(valid_foot_contact[ti-2, foot_index]) and
                    not bool(valid_foot_contact[ti-3, foot_index])):
                    footfalls += 1
                    fi = start+ti
                    xy_now = foot_xyz[fi,w,foot_index,:2]
                    if prior_xy is not None:
                        along = (xy_now-prior_xy) @ np.array([
                            np.cos(yaw[fi,w]),np.sin(yaw[fi,w])])
                        if np.isfinite(along):
                            same_foot_strides.append(float(along))
                    prior_xy = xy_now
        gait_time = max(dt,(stop-start)*dt)
        own = independent["worlds"][w]
        rows.append(dict(
            world=w, survived=bool(not np.any(fell[:stop, w]) and stop == horizon),
            completed_frames=stop,
            support_fraction=float(np.mean(support[:stop, w])),
            requested_vx_mps=float(speed),
            achieved_vx_mps=body_vx_mean,
            vx_mae_mps=vx_error,
            achieved_vy_mps=float(np.mean(body_vy)) if len(body_vy) else None,
            yaw_bias_rad_s=float(np.mean(omega)) if len(omega) else None,
            yaw_jitter_rad_s=float(np.std(omega)) if len(omega) else None,
            actual_grade_mean_deg=float(np.mean(grades[:stop, w])),
            actual_grade_max_deg=float(np.max(grades[:stop, w])),
            drive_work_j=work,
            path_distance_m=travelled,
            net_xy_drift_m=float(np.linalg.norm(xy[stop-1,w]-xy[start,w])),
            physical_footfalls=footfalls,
            footfall_rate_hz=footfalls/gait_time,
            same_foot_stride_mean_m=(float(np.mean(same_foot_strides))
                                     if same_foot_strides else None),
            directed_progress_m=progress,
            shaft_drive_j_per_progress_m=(work/progress if progress > .02 else None),
            verified_up=int(own["independent_up"]),
            verified_down=int(own["independent_down"]),
            supported_lips=own["independently_supported_lips"],
        ))
    manifest = dict(
        schema="emu_prism_frozen_exam_v1",
        source_actor_sha256=actor["artifact_sha256"],
        source_training_sha=actor["source"].get("git_commit"),
        source_step=actor["source"]["step"],
        mode=mode,
        protocol=dict(
            seed=seed, worlds=worlds, horizon=horizon, device=jax.default_backend(),
            requested_vx_mps=speed,
            compiled_solver_iterations=int(env.mj_model.opt.iterations),
            compiled_solver_ls_iterations=int(env.mj_model.opt.ls_iterations),
            requested_grade_frontier_deg=grade_deg if has_grade else 0.,
            requested_lip_frontier_cm=lip_cm if has_step else 0.,
            actual_collision=f"{PRISM_COUNT} convex prism meshes / {TILE_COUNT} {TILE_M*100:g}cm source tiles",
            sensor="360-input array-only 4x4 ToF",
            no_optimizer_or_curriculum_updates=True,
            phase_probe=phase_probe,tof_probe=tof_probe,
            probe_is_input_only=True,
            deterministic_seed=True,
        ),
        measured=dict(
            survived=sum(row["survived"] for row in rows),
            signed_supported_up=independent["up"],
            signed_supported_down=independent["down"],
            total_supported_lips=independent["total"],
            note="Procedural terrain: measured grades/lips, not requested frontiers, are evidence.",
        ),
        elapsed_wall_seconds=round(time.perf_counter()-begun, 2),
        worlds=rows,
        mission_autopsy=autopsy,
    )
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "prism_exam.json"
    markdown_path = output / "prism_exam.md"
    json_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n",
                         encoding="utf-8")
    lines = [
        "# Frozen-policy true-prism examination", "",
        f"Actor SHA-256: `{actor['artifact_sha256']}`",
        f"Source training commit: `{actor['source'].get('git_commit')}`",
        f"Mode: **{mode}** | worlds: {worlds} | horizon: {horizon}",
        f"Command: +{speed:.2f} m/s | requested grade frontier: {grade_deg if has_grade else 0.}°"
        f" | requested lip frontier: {lip_cm if has_step else 0.} cm", "",
        "These are procedural **teacher targets**, NOT imposed fixed-angle slabs.",
        f"Exam solver iterations/linesearch: **{env.mj_model.opt.iterations}/{env.mj_model.opt.ls_iterations}** (export provenance).",
        "All crossings use raw physical prism-foot contacts and five supported frames.",
        "0 m/s trials report world-frame drift, not zero-cost static posture.",
        "A zero crossing count is zero demonstrated crossings, not a failed attempted fixture.", "",
        f"Survived: **{manifest['measured']['survived']}/{worlds}**",
        f"Independently supported up/down: **{independent['up']}/{independent['down']}**", "",
        f"v5051 post-contact completed missions: **{autopsy['completed_missions']}/{autopsy['eligible_supported_contacts']}** eligible supported contacts",
        "Not a fixed-height obstacle or full eight-stage climb certification; detection, riser clearance and load transfer remain unmeasured.", "",
        "| World | Survived | Mean vx | vx MAE | Actual mean grade | Up | Down | Shaft J/m* |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = row["shaft_drive_j_per_progress_m"]
        lines.append(
            f"| {row['world']} | {'yes' if row['survived'] else 'no'} | "
            f"{row['achieved_vx_mps'] if row['achieved_vx_mps'] is not None else float('nan'):.3f} | "
            f"{row['vx_mae_mps'] if row['vx_mae_mps'] is not None else float('nan'):.3f} | "
            f"{row['actual_grade_mean_deg']:.2f}° | {row['verified_up']} | {row['verified_down']} | "
            f"{value:.2f} |" if value is not None else
            f"| {row['world']} | {'yes' if row['survived'] else 'no'} | "
            f"{row['achieved_vx_mps'] if row['achieved_vx_mps'] is not None else float('nan'):.3f} | "
            f"{row['vx_mae_mps'] if row['vx_mae_mps'] is not None else float('nan'):.3f} | "
            f"{row['actual_grade_mean_deg']:.2f}° | {row['verified_up']} | {row['verified_down']} | — |"
        )
    lines += [
        "", "*Mechanical positive shaft work divided by actual forward directed progress;"
        " not battery energy, and omitted when travel is below 2 cm.",
        "Per-world net XY drift, footfall rate and same-foot stride length are in `prism_exam.json`.",
        "Footfalls require two frames airborne then two frames supported; stride requires two same-foot contacts.",
        "Per-world measured support and signed lip geometry are in `prism_exam.json`.",
        "No physical mastery, fixed-grade pass rate, or hardware safety is inferred"
        " from procedural difficulty alone.",
    ]
    markdown_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return json_path, markdown_path
