"""Read-only physical quality scorecard for one exported EMU actor.

No JAX, PPO, curriculum or weight updates. The old frozen mission gate
remains unchanged. Area-under-capability is measured only between actually
tested physical-unit severities; missing skills are unknown, not failures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median

SCHEMA = "emu_unified_frozen_exam_v2_dual_scores"
Z95 = 1.959963984540054


def wilson(k: int, n: int) -> list[float]:
    if n < 1 or not 0 <= k <= n:
        raise ValueError("successes must be between zero and attempts")
    p = k / n
    z2 = Z95 * Z95
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [round(max(0., mid - half), 6), round(min(1., mid + half), 6)]


def curve_area(points: list[tuple[float, float]]) -> dict:
    """No extrapolation and no fabricated origin; single point -> null area."""
    points = sorted(points)
    if len({x for x, _ in points}) != len(points):
        raise ValueError("duplicate severities: combine seeds before area")
    if any(not math.isfinite(x) or not math.isfinite(y) or
           not 0 <= y <= 1 for x, y in points):
        raise ValueError("invalid physical curve")
    low = points[0][0] if points else None
    high = points[-1][0] if points else None
    width = high - low if len(points) > 1 else 0.
    area = (sum((b[0]-a[0])*(a[1]+b[1])/2
                for a, b in zip(points, points[1:]))
            if width > 0 else None)
    return dict(tested_levels=len(points), minimum=low, maximum=high,
                raw_success_area=(round(area, 6) if area is not None else None),
                tested_range_fraction=(round(area / width, 6)
                                       if area is not None else None))


def commanded_world_path(xy, heading, command, *, dt=.02,
                         settle_frames=25) -> dict:
    """Measure route against *requested* heading/velocity, not actual yaw.

    Anchor commanded path at the measured pose after settling, then
    integrate requested local vx/vy/yaw in an independent reference frame.
    An unintended turn can no longer be disguised by good body-frame vx.
    Returned fields are additive diagnostic evidence, NOT a new pass gate.
    """
    import numpy as np

    xy=np.asarray(xy,dtype=float)
    heading=np.asarray(heading,dtype=float)
    command=np.asarray(command,dtype=float)
    null=dict(world_path_rms_error_m=None,
              world_path_endpoint_error_m=None,
              world_cross_track_rms_m=None,
              world_heading_error_end_rad=None,
              commanded_world_displacement_m=None,
              directed_progress_m=None,
              useful_progress_ratio=None)
    if (xy.ndim!=2 or xy.shape[1]!=2 or
        heading.shape!=(len(xy),) or command.shape!=(len(xy),3)):
        raise ValueError("World path requires aligned T×2 XY, T headings, T×3 commands")
    if not (np.isfinite(xy).all() and np.isfinite(heading).all()
            and np.isfinite(command).all() and math.isfinite(dt) and dt>0):
        raise ValueError("World path contains invalid physical data")
    if settle_frames<0:
        raise ValueError("Settling frames cannot be negative")
    if len(xy)<=settle_frames+2:
        return null
    start=settle_frames
    actual=xy[start:]
    yaw=np.unwrap(heading[start:])
    cmd=command[start:-1]
    # Commands govern each interval following the sampled settled pose.
    desired_heading=yaw[0]+np.concatenate((
        [0.],np.cumsum(cmd[:,2]*dt)))
    bearing=desired_heading[:-1]
    velocity_world=np.stack((
        np.cos(bearing)*cmd[:,0]-np.sin(bearing)*cmd[:,1],
        np.sin(bearing)*cmd[:,0]+np.cos(bearing)*cmd[:,1]),axis=1)
    planned_displacement=np.concatenate((
        np.zeros((1,2)),np.cumsum(velocity_world*dt,axis=0)),axis=0)
    relative_actual=actual-actual[0]
    error=relative_actual-planned_displacement
    # Lateral error relative to the requested (NOT actual) heading.
    lateral=-np.sin(desired_heading)*error[:,0]+np.cos(desired_heading)*error[:,1]
    planned_net=planned_displacement[-1]
    expected=float(np.linalg.norm(planned_net))
    if expected>1e-5:
        directed=float(np.dot(relative_actual[-1],planned_net)/expected)
        ratio=directed/expected
    else:
        directed=ratio=None
    # Preserve unwrapped turn error for genuine turning tasks. Do not
    # classify natural stride-frequency yaw wiggle as permanent drift.
    heading_end=float(yaw[-1]-desired_heading[-1])
    return dict(
        world_path_rms_error_m=round(float(np.sqrt(np.mean(
            np.sum(error*error,axis=1)))),5),
        world_path_endpoint_error_m=round(float(np.linalg.norm(error[-1])),5),
        world_cross_track_rms_m=round(float(np.sqrt(np.mean(lateral*lateral))),5),
        world_heading_error_end_rad=round(heading_end,5),
        commanded_world_displacement_m=round(expected,5),
        directed_progress_m=(round(directed,5) if directed is not None else None),
        useful_progress_ratio=(round(ratio,5) if ratio is not None else None))


def axis(case: dict, world: dict, protocol: dict) -> tuple:
    """Exactly one real-unit severity per signed direction; mixed stays unranked."""
    kind = case["kind"]
    if kind == "step":
        d = world.get("direction")
        if d not in ("step-up", "step-down"):
            raise ValueError("missing actual step direction")
        return "step", d, round(100 * float(case["step_m"]), 5), "cm"
    if kind == "slope":
        d = case["direction"]
        if d not in ("uphill", "downhill", "cross_left", "cross_right"):
            raise ValueError("invalid real slope direction")
        return "slope", d, float(case["slope_deg"]), "deg"
    if kind in ("flat", "command", "control"):
        vx = float(case.get("vx", protocol["speed"]))
        vy, yaw = float(case.get("vy", 0.)), float(case.get("yaw", 0.))
        if sum(abs(x) > 1e-4 for x in (vx, vy, yaw)) > 1:
            return "combined_motion", case["name"], None, None
        if abs(yaw) > 1e-4:
            return "yaw", "left" if yaw > 0 else "right", abs(yaw), "rad/s"
        if abs(vy) > 1e-4:
            return "lateral", "left" if vy > 0 else "right", abs(vy), "m/s"
        return "longitudinal", "forward" if vx >= 0 else "reverse", abs(vx), "m/s"
    if kind == "recovery":
        if case.get("push_n", 0.):
            return "recovery", "push", float(case["push_n"]), "N"
        if case.get("tilt_deg", 0.):
            return "recovery", "tilt", float(case["tilt_deg"]), "deg"
        return "recovery", case["name"], None, None
    if kind == "transition":
        return "transition", case.get("scenario", case["name"]), None, None
    if kind == "mixed":
        # This physically includes a step; a supported downhill transfer
        # does not prove the same incline-plus-step manoeuvre uphill.
        d = world.get("direction")
        if case.get("step_m", 0.) > 0 and d not in ("step-up", "step-down"):
            raise ValueError("mixed supported-step world lost its direction")
        return "mixed_terrain", case["name"] + "/" + (d or "unknown"), None, None
    raise ValueError("unrecognised case kind: " + str(kind))


def finite_values(rows, field, *, successful_only=False) -> list[float]:
    values = []
    for row in rows:
        if successful_only and not row["mission_pass"]:
            continue
        x = row.get(field)
        if x is None:
            continue
        if isinstance(x, bool):
            raise ValueError("numeric physical measurement is bool: " + field)
        f = float(x)
        if not math.isfinite(f):
            raise ValueError("nonfinite physical measurement: " + field)
        values.append(f)
    return values


def mean_or_null(rows, key):
    values = finite_values(rows, key)
    return round(mean(values), 5) if values else None


def load_exam(path: Path) -> dict:
    path = path / "unified_exam.json" if path.is_dir() else path
    if path.name != "unified_exam.json" or not path.is_file():
        raise ValueError("expected existing unified_exam.json: " + str(path))
    j = json.loads(path.read_text(encoding="utf-8"))
    if j.get("schema") != SCHEMA:
        raise ValueError("unknown frozen physical examiner protocol")
    if not j.get("protocol", {}).get("no_reward_or_curriculum_feedback"):
        raise ValueError("physical quality accepts independent exams only")
    if len(j.get("actor_artifact_sha256", "")) != 64 or not j.get("results"):
        raise ValueError("exam must identify an actor and contain physical cases")
    return j


def build_report(exams: list[dict]) -> dict:
    if not exams:
        raise ValueError("at least one physical exam required")
    def identity(x):
        return (x["actor_artifact_sha256"], x["observation_schema"],
                x["source_training_step"])
    ident = identity(exams[0])
    pools = defaultdict(list)
    seen = set()
    for exam in exams:
        if identity(exam) != ident:
            raise ValueError("cannot combine different actor policies")
        proto = exam["protocol"]
        for field in ("seed", "speed", "worlds", "horizon", "backend",
                      "physical_fixture_revision"):
            if field not in proto:
                raise ValueError("missing frozen fixture field: " + field)
        fixture = (proto["physical_fixture_revision"], proto["horizon"], proto["backend"])
        for result in exam["results"]:
            case, rows = result["case"], result["worlds"]
            if len(rows) != proto["worlds"] or not rows:
                raise ValueError("case count differs from attempted worlds")
            if {w.get("world") for w in rows} != set(range(proto["worlds"])):
                raise ValueError("physical world indices missing or duplicated")
            for required in ("mission_pass", "survived", "case_pass"):
                if any(type(w.get(required)) is not bool for w in rows):
                    raise ValueError("missing boolean physical evidence: " + required)
            if (sum(w["mission_pass"] for w in rows) != result["mission_passed"]
                or sum(w["survived"] for w in rows) != result["survived"]):
                raise ValueError("unverifiable examiner totals")
            if any(w["mission_pass"] and not w["survived"] for w in rows):
                raise ValueError("a mission cannot pass after a recorded fall/reset")
            for world in rows:
                family, direction, level, unit = axis(case, world, proto)
                # Speed must match for terrain/recovery comparisons;
                # longitudinal speed itself IS the tested severity.
                requested_speed = (float(proto["speed"]) if family in
                                   ("slope", "step", "recovery", "mixed_terrain")
                                   else None)
                key = (family, direction, level, unit, *fixture, requested_speed)
                # Duplicate physical case with another *name* but the
                # same seed/condition/world is not a fresh independent sample.
                unique = (proto["seed"], key, world["world"])
                if unique in seen:
                    raise ValueError("repeated case/seed/world would inflate the denominator")
                seen.add(unique)
                pools[key].append((proto["seed"], world))
    cases, curves = [], defaultdict(list)
    for key, items in pools.items():
        family, direction, level, unit, fixture, horizon, backend, speed = key
        rows = [w for _, w in items]
        n = len(rows)
        passed = sum(w["mission_pass"] for w in rows)
        effort = finite_values(rows, "mechanical_j_per_useful_m",
                               successful_only=True)
        # Failure reasons are NON-exclusive. A robot can miss a step
        # AND lose heading; keep both facts instead of inventing a
        # single overly tidy cause. Count only known optional fields.
        reasons = dict(
            terminated_early=sum(not w["survived"] for w in rows),
            movement_failed=sum(w.get("movement_tracking_pass") is False for w in rows),
            heading_failed=sum(w.get("heading_control_pass") is False for w in rows),
            supported_crossing_failed=(sum(w.get("crossing_pass") is False for w in rows)
                                       if family == "step" else None),
            slope_stability_failed=(sum(w.get("slope_stability_pass") is False for w in rows)
                                    if family == "slope" else None),
            not_mission_qualified=n - passed)
        row = dict(family=family, direction=direction, severity=level, unit=unit,
                   fixture=fixture, horizon_frames=horizon, backend=backend,
                   requested_terrain_speed_mps=speed,
                   seeds=sorted({seed for seed, _ in items}),
                   attempts=n, missions=passed, mission_rate=round(passed / n, 6),
                   wilson_95=wilson(passed, n),
                   nonexclusive_failure_reasons=reasons,
                   survived=sum(w["survived"] for w in rows),
                   strict_pass=sum(w["case_pass"] for w in rows),
                   movement_pass=sum(w.get("movement_tracking_pass") is True for w in rows),
                   heading_pass=sum(w.get("heading_control_pass") is True for w in rows),
                   supported_crossing=(sum(w.get("crossing_pass") is True for w in rows)
                                       if family == "step" else None),
                   measured_vx_mps=mean_or_null(rows, "achieved_mean_vx_mps"),
                   signed_yaw_bias_rad_s=mean_or_null(rows, "tracking_signed_bias_yaw_rad_s"),
                   gait_yaw_jitter_rad_s=mean_or_null(rows, "tracking_jitter_yaw_rad_s"),
                   world_path_rms_m=mean_or_null(rows, "world_path_rms_error_m"),
                   world_path_endpoint_m=mean_or_null(rows, "world_path_endpoint_error_m"),
                   directed_progress_m=mean_or_null(rows, "directed_progress_m"),
                   useful_work_j_per_m=(round(median(effort), 3) if effort else None),
                   useful_work_samples=len(effort))
        cases.append(row)
        if level is not None:
            curve = (family, direction, unit, fixture, horizon, backend, speed)
            curves[curve].append((level, passed / n))
    cases.sort(key=lambda r: (r["family"], r["direction"], r["horizon_frames"],
                              -1 if r["severity"] is None else r["severity"]))
    envelopes = []
    for key, points in sorted(curves.items(), key=lambda p: str(p[0])):
        family, direction, unit, fixture, horizon, backend, speed = key
        envelopes.append(dict(family=family, direction=direction, unit=unit,
                              fixture=fixture, horizon_frames=horizon, backend=backend,
                              requested_terrain_speed_mps=speed,
                              **curve_area(points)))
    # Coverage means BOTH signs of motion and all four slope bearings,
    # not "some slope" or "some yaw". Never hide weakest directions.
    present = {row["family"] + "/" + row["direction"] for row in cases}
    required = {
        "locomotion": {"longitudinal/forward", "longitudinal/reverse",
                       "lateral/left", "lateral/right",
                       "yaw/left", "yaw/right"},
        "control": {"yaw/left", "yaw/right",
                    "transition/rest_start", "transition/brake_stop"},
        "terrain": {"slope/uphill", "slope/downhill", "slope/cross_left",
                    "slope/cross_right", "step/step-up", "step/step-down"},
        "resilience": {"recovery/push", "recovery/tilt"},
    }
    coverage = {k: dict(tested=sorted(present & v),
                        missing=sorted(v - present),
                        complete=not bool(v - present))
                for k, v in required.items()}
    coverage["grace"] = dict(complete=False, status="diagnostic only, no validated qualification")
    coverage["efficiency"] = dict(complete=any(x["useful_work_samples"] for x in cases),
                                  status="successful useful progress only; not battery power")
    return dict(schema="emu_physical_quality_v1",
                actor_sha256=ident[0], observation_schema=ident[1],
                source_training_step=ident[2],
                seeds=sorted({seed for rows in pools.values() for seed, _ in rows}),
                cases=cases, tested_envelopes=envelopes, coverage=coverage,
                global_score=None,
                cautions=[
                    "No global scalar; absent dimensions are unknown, never zero.",
                    "Frozen historical mission gate is unchanged by new world-path diagnostics.",
                    "AUC only spans sampled severity, not an infinite robot capability ceiling.",
                    "Multiple independent seeds are needed; 8/8 alone is not proof of 100% reliability.",
                    "Mechanical shaft J/m is not electrical battery energy.",
                ])


def markdown(report: dict) -> str:
    lines = ["# EMU — physical quality report", "",
             "Actor SHA256: " + report["actor_sha256"],
             "Trained steps: " + str(report["source_training_step"]),
             "Independent seeds: " + ", ".join(map(str, report["seeds"])),
             "", "**No global score.** Missing capability is not a failure.", "",
             "| Dimension / direction | Severity | Mission | Wilson 95% | Survived | Motion | Heading | Useful shaft J/m |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["cases"]:
        severity = (f"{c['severity']:g} {c['unit']}"
                    if c["severity"] is not None else "one scenario")
        energy = (f"{c['useful_work_j_per_m']:.1f}"
                  if c["useful_work_j_per_m"] is not None else "not measured")
        lines.append(f"| {c['family']}/{c['direction']} | {severity} | "
                     f"{c['missions']}/{c['attempts']} | "
                     f"{c['wilson_95'][0]:.2f}–{c['wilson_95'][1]:.2f} | "
                     f"{c['survived']}/{c['attempts']} | "
                     f"{c['movement_pass']}/{c['attempts']} | "
                     f"{c['heading_pass']}/{c['attempts']} | {energy} |")
    lines += ["", "## Why missions missed qualification", "",
              "Failure counts overlap: e.g. a robot may both miss a foothold "
              "and drift off-heading. Missing categories are not invented.",
              "",
              "| Challenge | Missions failed | Early termination | Movement | Heading | No supported crossing | No slope stability |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in report["cases"]:
        r=c["nonexclusive_failure_reasons"]
        if not r["not_mission_qualified"]:
            continue
        def fmt(v):
            return str(v) if v is not None else "—"
        challenge=(f"{c['family']}/{c['direction']} "
                   + (f"{c['severity']:g} {c['unit']}" if c["severity"] is not None else ""))
        lines.append(f"| {challenge} | {r['not_mission_qualified']} | "
                     f"{r['terminated_early']} | {r['movement_failed']} | "
                     f"{r['heading_failed']} | "
                     f"{fmt(r['supported_crossing_failed'])} | "
                     f"{fmt(r['slope_stability_failed'])} |")
    lines += ["", "## Actual route integrity (world-frame diagnostics)", "",
              "Requested trajectory is integrated independently from command vx/vy/yaw.",
              "Null means this frozen exam predates route instrumentation; do not invent zero error.",
              "",
              "| Dimension / direction | Severity | Achieved vx | Signed yaw bias | Gait yaw jitter | Route RMS | Endpoint error | Directed progress |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def physical(value, suffix, digits=3):
        return f"{value:.{digits}f}{suffix}" if value is not None else "not measured"
    for c in report["cases"]:
        severity = (f"{c['severity']:g} {c['unit']}"
                    if c["severity"] is not None else "one scenario")
        lines.append(f"| {c['family']}/{c['direction']} | {severity} | "
                     f"{physical(c['measured_vx_mps'],' m/s')} | "
                     f"{physical(c['signed_yaw_bias_rad_s'],' rad/s')} | "
                     f"{physical(c['gait_yaw_jitter_rad_s'],' rad/s')} | "
                     f"{physical(c['world_path_rms_m'],' m')} | "
                     f"{physical(c['world_path_endpoint_m'],' m')} | "
                     f"{physical(c['directed_progress_m'],' m')} |")
    lines += ["", "## Tested-range capability curves", "",
              "| Signed capability | Range | Samples | Tested-range AUC |",
              "|---|---|---:|---:|"]
    for x in report["tested_envelopes"]:
        a = x["tested_range_fraction"]
        score = f"{a:.3f}" if a is not None else "insufficient levels"
        lines.append(f"| {x['family']}/{x['direction']} | "
                     f"{x['minimum']:g}–{x['maximum']:g} {x['unit']} | "
                     f"{x['tested_levels']} | {score} |")
    lines += ["", "## Coverage", ""]
    for name, v in report["coverage"].items():
        if "missing" in v:
            lines.append(f"- {name}: missing " + (", ".join(v["missing"]) or "none"))
        else:
            lines.append(f"- {name}: {v['status']}")
    lines += ["", "Underlying values are in physical_quality.json.",
              "No training, source changes or automatic policy selection.", ""]
    return "\n".join(lines)


def report_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exam", type=Path, required=True, action="append",
                    help="frozen unified_exam.json or its folder; repeat for different seeds")
    ap.add_argument("--output", type=Path, required=True,
                    help="new/empty output folder; inputs are read only")
    args = ap.parse_args(argv)
    src = [p.expanduser().resolve() for p in args.exam]
    report = build_report([load_exam(p) for p in src])
    dest = args.output.expanduser().resolve()
    if any(dest == p or dest in p.parents or p in dest.parents for p in src):
        raise ValueError("output must be separate from all input exam paths")
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError("report destination must be new or empty")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "physical_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (dest / "physical_quality.md").write_text(markdown(report))
    for file in ("physical_quality.json", "physical_quality.md"):
        print(dest / file)
    return 0


if __name__ == "__main__":
    raise SystemExit(report_main())
