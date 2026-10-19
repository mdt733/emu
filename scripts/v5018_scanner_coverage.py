"""EMU v5018 P0B: physically ray-cast forward ToF coverage, no oracle observations.

Uses exactly the CPU MuJoCo collision geometry for all returns. Tests two
hypothetical scan arrangements over torso height, pitch and roll. This does
not assert that a real ToF module provides these beams or data rates.
Run: python scripts/v5018_scanner_coverage.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = """<mujoco><worldbody>
<geom name="floor" type="plane" size="2 2 .1"/>
<geom name="right_step" type="box" pos=".23 -.08 .0125" size=".06 .055 .0125"/>
<geom name="left_step" type="box" pos=".34 .08 .0175" size=".06 .055 .0175"/>
<geom name="log" type="capsule" fromto=".49 -.18 .014 .49 .18 .014" size=".012"/>
</worldbody></mujoco>"""
LANES = (
    ("right", -.08, .17, "right_step"),
    ("left", +.08, .28, "left_step"),
)
MOUNT_X = .025
MOUNT_Z = -.025
RANGE_MIN, RANGE_MAX = .035, .60
ACQUISITION_HZ = 25.0
TRAVEL_SPEED_MPS = .15
GEOMETRIES = {
    "single_45deg": (45.0,),
    "three_zone_28_43_58deg": (28.0, 43.0, 58.0),
}


def attitude(pitch_deg, roll_deg):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    return ry @ rx


def cast(model, data, geom_names, root_xyz, rotation, side_y, angle):
    mount = np.array([MOUNT_X, side_y, MOUNT_Z])
    direction = np.array([math.sin(math.radians(angle)), 0.,
                          -math.cos(math.radians(angle))])
    origin = root_xyz + rotation @ mount
    world_dir = rotation @ direction
    geom = np.array([-1], dtype=np.int32)
    distance = float(mujoco.mj_ray(model, data, origin, world_dir,
                                   None, 1, -1, geom))
    if distance < RANGE_MIN or distance > RANGE_MAX:
        return {"valid": False, "hit": None, "distance_m": None,
                "point": None, "type": "unknown"}
    hit = geom_names[int(geom[0])] if int(geom[0]) >= 0 else None
    xyz = origin + distance * world_dir
    if hit == "floor":
        category = "ground"
    elif hit == "log":
        category = "rounded_hazard"
    elif hit in ("right_step", "left_step"):
        top = .025 if hit == "right_step" else .035
        category = "walkable_top" if abs(xyz[2] - top) < .001 else "step_face"
    else:
        category = "unknown"
    return {"valid": hit is not None, "hit": hit, "distance_m": distance,
            "point": xyz.tolist(), "type": category}


def sample_trial(model, data, names, lane, height, pitch, roll, beams):
    side, y, face, obstacle_name = lane
    rot = attitude(pitch, roll)
    x_values = np.arange(-.20, face - .018, TRAVEL_SPEED_MPS / ACQUISITION_HZ)
    first = None
    total_valid = 0
    obstacle_hits = 0
    category_counts = {"walkable_top": 0, "step_face": 0, "rounded_hazard": 0,
                       "ground": 0, "unknown": 0}
    first_sample = None
    for x in x_values:
        root = np.array([float(x), 0., float(height)])
        readings = [cast(model, data, names, root, rot, y, angle)
                    for angle in beams]
        total_valid += sum(p["valid"] for p in readings)
        for p in readings:
            category_counts[p["type"]] += 1
        detection = [p for p in readings if p["hit"] == obstacle_name]
        if detection:
            obstacle_hits += 1
            if first is None:
                first = float(x)
                first_sample = detection[0]
    lead = None if first is None else face - first
    return {
        "lane": side, "torso_height_m": height, "pitch_deg": pitch,
        "roll_deg": roll, "angles_deg": list(beams),
        "acquisition_hz": ACQUISITION_HZ, "speed_mps": TRAVEL_SPEED_MPS,
        "number_of_body_positions": len(x_values),
        "fraction_valid_rays": total_valid / (len(x_values) * len(beams)),
        "fraction_positions_seeing_obstacle": obstacle_hits / len(x_values),
        "first_detected_torso_x_m": first,
        "first_detection_lead_m": lead,
        "first_detection_lead_s_at_test_speed": (
            lead / TRAVEL_SPEED_MPS if lead is not None else None),
        "first_detection_kind": first_sample["type"] if first_sample else None,
        "face_hits": category_counts["step_face"],
        "top_hits": category_counts["walkable_top"],
        "counts": category_counts,
        "early_detection_at_least_50mm": bool(lead is not None and lead >= .05),
    }


def main():
    model = mujoco.MjModel.from_xml_string(ROOT)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [model.geom(i).name for i in range(model.ngeom)]
    rows = []
    for name, beams in GEOMETRIES.items():
        for height in (.21, .26, .30):
            for pitch in (-10, 0, 10):
                for roll in (-6, 0, 6):
                    for lane in LANES:
                        item = sample_trial(model, data, names, lane, height,
                                            pitch, roll, beams)
                        item["scanner"] = name
                        rows.append(item)
    summary = {}
    for name in GEOMETRIES:
        cases = [r for r in rows if r["scanner"] == name]
        leads = [r["first_detection_lead_m"] for r in cases
                 if r["first_detection_lead_m"] is not None]
        summary[name] = {
            "trials": len(cases),
            "any_detection": len(leads),
            "early_detection_50mm": sum(r["early_detection_at_least_50mm"]
                                        for r in cases),
            "first_lead_m_min": min(leads) if leads else None,
            "first_lead_m_median": float(np.median(leads)) if leads else None,
            "mean_valid_ray_fraction": float(np.mean([
                r["fraction_valid_rays"] for r in cases])),
            "face_hits": sum(r["face_hits"] for r in cases),
            "top_hits": sum(r["top_hits"] for r in cases),
        }
    payload = {
        "description": "CPU MuJoCo first-hit forward ray visibility; NOT a trained estimator",
        "geometry": "Two independent 25/35mm raised box steps plus 24mm log",
        "mount_local_m": [MOUNT_X, "+/-0.08", MOUNT_Z],
        "beam_classes": GEOMETRIES,
        "range_limits_m": [RANGE_MIN, RANGE_MAX],
        "summaries": summary,
        "samples": rows,
        "limitations": [
            "Candidate scanner angular rays are hypothetical until hardware FOV is selected.",
            "Assumes perfect pose and timing; no sensor noise, occlusion by moving legs, or dropout.",
            "Only tests forward motion at .15m/s, fixed lateral foot lanes and listed body attitudes.",
            "Face hits are hazards, NOT walkable-top height samples.",
            "First detections include case when torso is far from edge; detection time is a geometric estimate.",
            "No memory fusion/estimator, no history, no NN contract change.",
        ],
    }
    out = Path("v5018_scanner_coverage.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summary, "file": str(out)}, indent=2))
    if any(v["any_detection"] == 0 for v in summary.values()):
        raise SystemExit("No obstacle visibility for a candidate sensor; investigate")


if __name__ == "__main__":
    main()
