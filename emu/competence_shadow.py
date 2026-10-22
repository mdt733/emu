"""Read-only on-policy competence diagnostics, never a curriculum controller.

This report makes existing training edge measurements interpretable without
writing to a checkpoint, adjusting a frontier, or claiming independent
certification. It can be run against an old metrics.csv on a laptop; no
GPU/JAX dependencies or model loads are required.

No score in this file is a PPO payment or a frozen-policy examination.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


# Each field is (human skill, logged task limit, optional config ceiling,
# edge precision, edge exposure). The precision is an on-policy FRAME/WINDOW
# proxy, NOT a count of completed independent episodes.
AXES = (
    ("forward", "frontier_vx_hi", "frontier_vx_hi_safety",
     "frontier_precision_vx_forward", "frontier_edge_frac_vx_forward", "m/s"),
    ("reverse", "frontier_vx_lo", "frontier_vx_lo_safety",
     "frontier_precision_vx_reverse", "frontier_edge_frac_vx_reverse", "m/s"),
    ("lateral", "frontier_vy", "frontier_vy_safety",
     "frontier_precision_vy", "frontier_edge_frac_vy", "m/s"),
    ("yaw", "frontier_yaw", "frontier_yaw_safety",
     "frontier_precision_yaw", "frontier_edge_frac_yaw", "rad/s"),
    ("combinations", "frontier_combo_radius", "frontier_combo_safety",
     "frontier_precision_combo", "frontier_edge_frac_combo", "normalised"),
)


def _finite_number(row: Mapping[str, Any], name: str) -> float | None:
    raw = row.get(name)
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric logged value {name}: {raw!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"nonfinite logged value {name}: {raw!r}")
    return v


def _certification_absent() -> dict[str, Any]:
    # Don't transform on-policy tracking into a spurious certified score.
    return {"value": None, "status": "not independently examined"}


def competence_shadow(rows: list[Mapping[str, Any]],
                      config: Mapping[str, Any] | None = None,
                      *, window: int = 12) -> dict[str, Any]:
    """Summarise sampled edge evidence without modifying curriculum state.

    A 'promising' on-policy signal is merely a future lesson candidate.
    It never changes a stage limit or overrides the proof obligations of
    the frozen examiner. Missing or sparse evidence remains 'unknown'.
    """
    if not rows:
        raise ValueError("cannot report competence from zero training rows")
    if not 1 <= window <= 1000:
        raise ValueError("invalid reporting window")
    recent = rows[-window:]
    last = rows[-1]
    cfg = config or {}
    open_frontier = cfg.get("open_frontier", None)
    if open_frontier is not None and not isinstance(open_frontier, bool):
        raise ValueError("open_frontier in config must be boolean")

    def weighted_completed() -> dict[str, float | None]:
        counts = [_finite_number(r, "episodes") for r in recent]
        if any(n is None for n in counts):
            return {"completed_episodes": None, "fall_rate": None}
        if any(n < 0 or n != int(n) for n in counts):
            raise ValueError("invalid completed-episode count")
        total = sum(counts)
        fall_rates = [_finite_number(r, "fall_rate") for r in recent]
        if any(x is not None and not 0 <= x <= 1 for x in fall_rates):
            raise ValueError("invalid episode fall rate")
        if total < 8 or any(rate is None for rate,n in zip(fall_rates,counts) if n > 0):
            return {"completed_episodes": int(total), "fall_rate": None}
        return {"completed_episodes": int(total),
                "fall_rate": sum((rate or 0.0)*n for rate,n in zip(fall_rates,counts))/total}

    cohort = weighted_completed()
    outcome = {}
    for title, limit_key, ceiling_key, precision_key, exposure_key, unit in AXES:
        limit = _finite_number(last, limit_key)
        ceiling = _finite_number(cfg, ceiling_key)
        # Configuration stores reverse safety as a positive maximum
        # magnitude; display it as a negative command boundary.
        if title == "reverse" and ceiling is not None:
            ceiling = -abs(ceiling)
        history = []
        for row in recent:
            precision = _finite_number(row, precision_key)
            exposure = _finite_number(row, exposure_key)
            if precision is None or exposure is None:
                continue
            if precision < 0:  # explicit 'no data' sentinel from existing reporter
                continue
            if not 0 <= precision <= 1 or not 0 <= exposure <= 1:
                raise ValueError(f"invalid on-policy edge measurement: {precision_key}")
            if exposure >= .005:
                history.append(precision)
        if len(history) < 3 or cohort["fall_rate"] is None:
            status = "insufficient training evidence"
        elif cohort["fall_rate"] > .35:
            status = "hold: high overall fall rate"
        elif min(history[-3:]) >= .62:
            status = "promising training signal (not certified)"
        elif max(history[-3:]) <= .28:
            status = "weak edge tracking; review before advancing"
        else:
            status = "hold: mixed training evidence"
        outcome[title] = {
            "training_frontier": limit, "safety_ceiling": ceiling,
            "unit": unit, "edge_precision_windows": len(history),
            "recent_edge_precision": None if not history else round(history[-1],5),
            "shadow_assessment": status,
            "certified_capability": _certification_absent(),
        }

    # Step-height progression already has a live training governor. This
    # report provides its source evidence, not a competing decision.
    step_frontier = _finite_number(last, "frontier_terrain_height_m")
    step_trials = [_finite_number(r, "terrain_edge_done_count") for r in recent]
    step_exposure = None if any(x is None for x in step_trials) else int(sum(step_trials))
    outcome["physical_steps"] = {
        "training_frontier": None if step_frontier is None else round(step_frontier*1000,5),
        "safety_ceiling": None if _finite_number(cfg,"terrain_height_safety_m") is None
                          else round(float(cfg["terrain_height_safety_m"])*1000,5),
        "unit": "mm",
        "completed_edge_trials_in_window": step_exposure,
        "shadow_assessment": ("insufficient training evidence" if not step_exposure
                              else "refer to existing earned-height governor"),
        "certified_capability": _certification_absent(),
    }
    return {
        "kind": "training-evidence-only",
        "policy_step": _finite_number(last,"step"),
        "command_frontier_actively_controls_training": open_frontier,
        "window_log_rows": len(recent),
        "completed_training_episodes": cohort["completed_episodes"],
        "observed_recent_fall_rate": cohort["fall_rate"],
        "skills": outcome,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only EMU training competence shadow report")
    parser.add_argument("--metrics", type=Path, required=True, help="existing metrics.csv")
    parser.add_argument("--config", type=Path, help="optional matching training_config.json")
    parser.add_argument("--window", type=int, default=12, help="last N logged rows")
    args = parser.parse_args(argv)
    with args.metrics.expanduser().open(newline="",encoding="utf-8") as fd:
        rows = list(csv.DictReader(fd))
    config = None
    if args.config is not None:
        config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    print(json.dumps(competence_shadow(rows,config,window=args.window),
                     indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
