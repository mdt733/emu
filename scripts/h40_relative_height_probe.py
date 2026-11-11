"""Read-only H40 *real-prism* ToF height audit: no policy, training or oracle input.

Run:
  PYTHONPATH="$PWD" python -m scripts.h40_relative_height_probe \
      --device cuda --worlds 104 \
      --output ~/emu-data/exams/h40-height-s735

The probe starts 22cm before a real signed prism seam, exactly as the
heterogeneous matrix does. It asks the EXISTING 360-input observation
for the delayed 16 ToF ranges, IMU gravity and kinematic foot centres.
No privileged seam height enters the projection; requested exact height
only labels the independently checked test fixture.

A ray return is NOT necessarily a walkable top: a vertical riser can
return an endpoint below the upper surface. The two candidate stance
frames remain separate: shin-flex angle does not certify foot contact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from emu.tof_geometry import returns_relative_to_feet

DEFAULT_HEIGHTS_MM = (
    0., 3., 20., 25., 27.5, 30., 32.5, 35., 40., 50.,
    -25., -35., -50.,
)


def _zone_indices():
    from emu.contracts import TERRAIN_OBS_FIELDS
    fields = {name: i for i, name in enumerate(TERRAIN_OBS_FIELDS)}
    keys = [f"tof4x4_range_{r}{c}_m" for r in range(4) for c in range(4)]
    valid = [f"tof4x4_valid_{r}{c}" for r in range(4) for c in range(4)]
    feet = [
        [fields[f"foot_{side}_{axis}"] for axis in "xyz"]
        for side in "rl"
    ]
    return fields, [fields[k] for k in keys], [fields[k] for k in valid], feet


def analyze_packet(packet: np.ndarray, requested_mm: np.ndarray):
    """Pure host analysis of the exact sensor observation, no simulator truth."""
    from emu.contracts import TERRAIN_OBS_FIELDS
    from emu.directional import fit_forward_array_ground
    fields, ri, vi, fi = _zone_indices()
    frame = packet.reshape((packet.shape[0], -1, len(TERRAIN_OBS_FIELDS)))[:, -1, :]
    ranges = frame[:, ri]
    valid = frame[:, vi] > .5
    gravity = frame[:, [fields[f"gravity_{c}"] for c in "xyz"]]
    feet = frame[:, fi]
    projected = returns_relative_to_feet(
        np, ranges, valid, gravity, feet)
    ground = fit_forward_array_ground(np, ranges, valid, gravity)
    details = []
    for world in range(frame.shape[0]):
        for zone in range(16):
            row = dict(world=world, actual_requested_lip_mm=float(requested_mm[world]),
                       zone=zone, range_m=float(ranges[world, zone]),
                       range_valid=bool(valid[world, zone]),
                       ground_plane_valid=bool(ground["valid"][world]),
                       ground_plane_confidence=float(ground["ground_visibility_confidence"][world]),
                       ground_forward_grade_deg=float(
                           np.rad2deg(ground["ground_forward_grade_rad"][world])),
                       ground_plane_residual_mm=float(ground["fit_residual_m"][world]*1000.))
            for idx, side in enumerate(("right", "left")):
                suffix="r" if idx==0 else "l"
                # These 360-input actor values may help test stance inference.
                # The current 'shin_load' is only a flex-angle proxy.
                row.update({
                    f"{side}_shin_flex_rad":float(frame[world,fields[f"shin_flex_{suffix}"]]),
                    f"{side}_shin_rate_rad_s":float(frame[world,fields[f"shin_rate_{suffix}"]]),
                    f"{side}_shin_load_proxy":float(frame[world,fields[f"shin_load_{suffix}"]]),
                })
                row.update({
                    f"{side}_point_valid":bool(projected["return_valid"][world,idx,zone]),
                    f"{side}_hit_height_mm":float(
                        projected["return_height_over_foot_m"][world,idx,zone]*1000.),
                    f"{side}_hit_forward_mm":float(
                        projected["return_forward_from_foot_m"][world,idx,zone]*1000.),
                    f"{side}_hit_side_mm":float(
                        projected["return_sideways_from_foot_m"][world,idx,zone]*1000.),
                })
            details.append(row)
    stats = []
    for height in dict.fromkeys(requested_mm.tolist()):
        selected = np.flatnonzero(requested_mm == height)
        masks = projected["return_valid"][selected]
        rr = projected["return_height_over_foot_m"][selected]
        xx = projected["return_forward_from_foot_m"][selected]
        yy = projected["return_sideways_from_foot_m"][selected]
        # Compare same physical region in front of either foot. A highest
        # hit is not a certified upper terrace: it may be the vertical wall.
        ahead = masks & (xx > .05) & (xx < .35) & (np.abs(yy) < .25)
        max_hit_m = float(np.max(np.where(ahead, rr, -np.inf))) if np.any(ahead) else None
        present = valid[selected]
        plane = np.asarray(ground["valid"])[selected]
        stats.append(dict(requested_lip_mm=float(height),
                          sampled_worlds=len(selected),
                          raw_valid_fraction=float(np.mean(present)),
                          single_plane_valid_fraction=float(np.mean(plane)),
                          mean_plane_residual_mm=float(np.mean(
                              np.asarray(ground["fit_residual_m"])[selected])*1000.),
                          max_ray_hit_height_above_candidate_foot_mm=(
                              round(max_hit_m*1000.,2) if max_hit_m is not None else None),
                          valid_ahead_fraction=float(np.mean(ahead))))
    return details, stats


def run(*,device:str,worlds:int,output:Path,seed:int=735,
        heights_mm:tuple[float,...]=DEFAULT_HEIGHTS_MM):
    if not heights_mm or not all(np.isfinite(heights_mm)):
        raise ValueError("Need finite signed heights")
    # Input heights are MILLIMETRES. The real-prism examiner accepts
    # METRES and checks the physical mesh geometry after conversion.
    # This read-only sight probe must not inherit an arbitrary 80mm
    # TRAINING terrain frontier (or mix the units as the old check did).
    if worlds < len(heights_mm) or worlds%len(heights_mm):
        raise ValueError(f"--worlds must be a positive multiple of {len(heights_mm)} height classes")
    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Vision probe output must be new or empty")

    from emu.ppo import configure_jax, require_backend
    from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    configure_jax(device)
    require_backend(device=="cpu")
    import jax

    heights = np.tile(np.asarray(heights_mm,np.float32),worlds//len(heights_mm))
    # Explicit single unit conversion, exactly at the MJX fixture boundary.
    signed_lips_m = heights * np.float32(1.0e-3)
    print(f"H40 raw ToF / 2-foot projection: {worlds} {device} "
          f"MJX worlds, exact prism seam, seed={seed} — compiling",flush=True)
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=worlds,seed=seed,
        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_carpet=True,
        terrain_carpet_prisms=True,academy_stream=True,
        terrain_lesson="coherent-careful",
        exam_fixture_steps_m=signed_lips_m,
        exam_fixture_grades_deg=np.zeros(worlds,np.float32),
        exam_fixture_kicks_mps=np.zeros(worlds,np.float32),
    )
    state,packet=env.reset(dict(
        specialist_mix=0.,terrain_exposure=0.,slope_exposure=0.,
        academy_episode_probability=0.,probe_push_n=0.,
        regime_frac_steady=1.,regime_frac_transition=0.,
        regime_frac_resilience=0.,
    ))
    packet=np.asarray(jax.device_get(packet),dtype=np.float32)
    details,stats=analyze_packet(packet,heights)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"vision_birth.csv").open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=list(details[0]))
        wr.writeheader();wr.writerows(details)
    (output/"vision_summary.json").write_text(json.dumps(dict(
        sensor_contract=env.sensor_contract,source_kind="deployable-packet-only",
        height_fixture_mm=list(heights_mm),worlds=worlds,seed=seed,
        physical_backend=jax.default_backend(),projected_stats=stats,
        no_neural_policy_or_training=True,
        caveat=("Returns measure 3D hit points, not guaranteed top ground. "
                "Neither foot is asserted to be weight-bearing.")
    ),indent=2)+"\n")
    lines=["# H40 foot-relative 4x4 ToF — real-prism birth audit","",
           f"Backend={jax.default_backend()}, {worlds} worlds, seed={seed}.",
           "Pure conversion from the EXISTING 360-input actor packet.",
           "No NN updates, source-based step oracle, or extra hardware.","",
           "A *ray hit* may be a vertical wall rather than a ground top.",
           "Both foot-reference candidates are included in vision_birth.csv;",
           "The included shin 'load' is a flex-angle proxy, not proven foot contact.","",
           "| Real signed lip | Worlds | Raw valid | Plane valid | Mean plane RMS | Highest ahead return* |",
           "|---:|---:|---:|---:|---:|---:|"]
    for x in stats:
        best=(f"{x['max_ray_hit_height_above_candidate_foot_mm']:.1f}mm"
              if x["max_ray_hit_height_above_candidate_foot_mm"] is not None else "unknown")
        lines.append(
            f"| {x['requested_lip_mm']:+.1f}mm | {x['sampled_worlds']} | "
            f"{x['raw_valid_fraction']:.0%} | "
            f"{x['single_plane_valid_fraction']:.0%} | "
            f"{x['mean_plane_residual_mm']:.1f}mm | {best} |")
    lines.extend(["","*Highest return point within 50–350mm forward, ±250mm sideways",
                  "of either candidate support foot. It is NOT a certified",
                  "walkable step top or measured mechanical climbing limit.","",
                  "Full 16-zone per-foot X/Y/height and explicit flags:",
                  "`vision_birth.csv`.","",
                  "Next gate: inspect suspected wall-hit zones and actual stance",
                  "confidence before promoting any transformed values to an actor."])
    (output/"vision_summary.md").write_text("\n".join(lines)+"\n")
    print(f"Vision probe complete: {output/'vision_summary.md'}",flush=True)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--worlds",type=int,default=104)
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--heights-mm",default=None,
                   help="comma-separated signed physical heights, e.g. 0,25,35,50,-35")
    a=p.parse_args()
    heights=DEFAULT_HEIGHTS_MM if a.heights_mm is None else tuple(
        float(v) for v in a.heights_mm.split(","))
    run(device=a.device,worlds=a.worlds,output=a.output,seed=a.seed,
        heights_mm=heights)


if __name__=="__main__":
    main()
