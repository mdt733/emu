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


def per_world_row_contrasts(projected, requested_mm):
    """Robust *ray-hit* height differences, no stance-foot selection.

    The four physical sensor rows look successively nearer (0 = far,
    3 = near). Subtracting elevations cancels the unknown foot-reference
    height, so NO contact oracle is needed for this particular feature.
    Sharp between-row differences are evidence of nonplanar geometry,
    NOT proof of an accessible upper foothold: a vertical wall can also
    produce an abrupt return. Never treat unknown/missing rows as flat.
    """
    valid=np.asarray(projected["return_valid"])[:,0,:].reshape(-1,4,4)
    z=np.asarray(projected["return_height_over_foot_m"])[:,0,:].reshape(-1,4,4)
    x=np.asarray(projected["return_forward_from_foot_m"])[:,0,:].reshape(-1,4,4)
    out=[]
    for w in range(valid.shape[0]):
        hh=[];xx=[]
        for row in range(4):
            keep=valid[w,row]
            hh.append(float(np.median(z[w,row,keep])) if np.count_nonzero(keep)>=2 else None)
            xx.append(float(np.median(x[w,row,keep])) if np.count_nonzero(keep)>=2 else None)
        complete=all(v is not None for v in hh+xx)
        jumps=[hh[i]-hh[i+1] for i in range(3)] if complete else []
        winner=int(np.argmax(np.abs(jumps))) if complete else -1
        out.append(dict(
            world=w, requested_lip_mm=float(requested_mm[w]),
            valid_row_count=sum(a is not None for a in hh),
            far_minus_near_hit_mm=round((hh[0]-hh[3])*1000.,3) if complete else None,
            largest_signed_adjacent_row_jump_mm=(
                round(jumps[winner]*1000.,3) if complete else None),
            largest_jump_between_rows=(f"{winner}-{winner+1}" if complete else None),
            largest_jump_forward_of_foot_mm=(
                round(.5*(xx[winner]+xx[winner+1])*1000.,3)
                if complete else None),
            ray_row_hit_heights_mm=json.dumps(
                [round(v*1000.,3) if v is not None else None for v in hh]),
            ray_row_forward_mm=json.dumps(
                [round(v*1000.,3) if v is not None else None for v in xx]),
        ))
    return out


def frozen_actor_sensitivity(packet, requested_mm, policy):
    """Only perturb vision fields of a FIXED FLAT reference actor packet.

    The exact exported actor + normalizer is used; no training and no
    hidden terrain truth enters the policy. Counterfactual ToF rays are
    measured from the corresponding REAL mesh-height birth. Non-vision
    fields (joints, IMU, commands, actuation history) are held constant.
    Flat reference is matched within each repeat of the height schedule.

    This is a frozen *action-response* assay, NOT a moving-robot success
    test. Raw-only isolates 16 zone ranges/valid bits; fit-only isolates
    grade/visibility; both uses the union. If grade confidence collapses
    but raw changes are clearly visible, compare which input the actor
    actually responds to.
    """
    from emu.contracts import (
        HISTORY_LEN, OBS_SCHEMA_TERRAIN, TERRAIN_OBS_FIELDS)
    from emu.ppo import build_obs_norm_mask
    from emu.policy_export import load_exported_actor_for_exam
    params,norm,actor=load_exported_actor_for_exam(policy)
    if "pi" not in params or actor["obs_dim"]!=packet.shape[1]:
        raise ValueError("Exported 360-input actor does not match sensor packet")
    if 0. not in requested_mm:
        raise ValueError("--policy sensitivity needs a 0mm flat reference class")
    # Each supplied height appears once per repeating cycle. Reject other
    # assignments rather than silently pairing unrelated random resets.
    period=int(np.where(requested_mm==requested_mm[0])[0][1]) if (
        np.count_nonzero(requested_mm==requested_mm[0])>1) else len(requested_mm)
    if len(requested_mm)%period or not np.all(
            requested_mm.reshape(-1,period)==requested_mm[:period][None,:]):
        raise ValueError("Frozen policy comparison needs repeated height schedule")
    flat_column=int(np.where(requested_mm[:period]==0.)[0][0])
    ref_idx=(np.arange(len(requested_mm))//period)*period+flat_column
    hist=packet.reshape((-1,HISTORY_LEN,len(TERRAIN_OBS_FIELDS)))
    flat=hist[ref_idx].copy()
    fields={name:i for i,name in enumerate(TERRAIN_OBS_FIELDS)}
    ray=[fields[f"tof4x4_range_{r}{c}_m"] for r in range(4) for c in range(4)]
    bits=[fields[f"tof4x4_valid_{r}{c}"] for r in range(4) for c in range(4)]
    fit=[fields[k] for k in (
        "ground_forward_grade_rad","ground_sideways_grade_rad",
        "ground_visibility_confidence")]
    raw_idx=ray+bits
    def replace(indices):
        v=flat.copy()
        v[:,:,indices]=hist[:,:,indices]
        return v.reshape((len(packet),-1))

    mask=np.asarray(build_obs_norm_mask(OBS_SCHEMA_TERRAIN),np.float32)
    mean=np.asarray(norm["mean"],np.float32)
    std=np.asarray(norm["std"],np.float32)
    def actor_mean(v):
        v=np.asarray(v,np.float32)
        z=np.where(mask[None,:]>.5,np.clip(
            (v-mean[None,:])/std[None,:],-10.,10.),v)
        for layer in params["pi"][:-1]:
            z=np.tanh(z @ layer["w"]+layer["b"])
        last=params["pi"][-1]
        return np.clip(z @ last["w"]+last["b"],-1.,1.)

    baseline=actor_mean(flat.reshape((len(packet),-1)))
    raw=actor_mean(replace(raw_idx))
    fitted=actor_mean(replace(fit))
    both=actor_mean(replace(raw_idx+fit))
    rows=[]
    for i in range(len(packet)):
        def metrics(arr):
            change=arr[i]-baseline[i]
            return round(float(np.linalg.norm(change)),6)
        rows.append(dict(
            world=i,requested_lip_mm=float(requested_mm[i]),
            raw_range_l2_m=round(float(np.linalg.norm(
                hist[i,-1,ray]-flat[i,-1,ray])),6),
            raw_only_action_l2=metrics(raw),
            fit_only_action_l2=metrics(fitted),
            raw_and_fit_action_l2=metrics(both),
            baseline_action=json.dumps(baseline[i].round(5).tolist()),
            raw_and_fit_action=json.dumps(both[i].round(5).tolist()),
        ))
    return rows, dict(
        policy_sha256=actor["artifact_sha256"],
        actor_weights_sha256=actor["source"]["actor_sha256"],
        comparison="fixed flat actor input + measured per-height vision swaps",
        limitation=("Flat and ledge ray packets are sampled from different "
                    "reset worlds; only actor non-vision channels are held "
                    "identical. This isolates frozen policy sensitivity but "
                    "not closed-loop action or guaranteed same-pose sensor rays."),
        frozen_actor=True,optimizer_updates=0,hold_nonvision_fields=True)


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
    spatial=per_world_row_contrasts(projected,requested_mm)
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
        contrasts=[spatial[int(w)]["far_minus_near_hit_mm"] for w in selected
                   if spatial[int(w)]["far_minus_near_hit_mm"] is not None]
        jumps=[spatial[int(w)]["largest_signed_adjacent_row_jump_mm"]
               for w in selected
               if spatial[int(w)]["largest_signed_adjacent_row_jump_mm"] is not None]
        stats[-1].update(
            sampled_valid_row_contrasts=len(contrasts),
            median_far_minus_near_hit_mm=(
                round(float(np.median(contrasts)),2) if contrasts else None),
            median_largest_signed_row_jump_mm=(
                round(float(np.median(jumps)),2) if jumps else None),
        )
    return details, stats, spatial


def run(*,device:str,worlds:int,output:Path,seed:int=735,
        heights_mm:tuple[float,...]=DEFAULT_HEIGHTS_MM,
        policy:Path|None=None):
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
    details,stats,spatial=analyze_packet(packet,heights)
    # Both analyses consume the SAME measured 360 observation, with no
    # privileged step height beyond its output grouping labels.
    sensitivity=None
    policy_info=None
    if policy is not None:
        sensitivity,policy_info=frozen_actor_sensitivity(
            packet,heights,policy.expanduser().resolve())
    output.mkdir(parents=True,exist_ok=True)
    with (output/"vision_birth.csv").open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=list(details[0]))
        wr.writeheader();wr.writerows(details)
    with (output/"vision_spatial.csv").open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=list(spatial[0]))
        wr.writeheader();wr.writerows(spatial)
    if sensitivity is not None:
        with (output/"vision_policy_sensitivity.csv").open("w",newline="") as fd:
            wr=csv.DictWriter(fd,fieldnames=list(sensitivity[0]))
            wr.writeheader();wr.writerows(sensitivity)
        by_height=[]
        for height in dict.fromkeys(heights.tolist()):
            sample=[v for v in sensitivity if v["requested_lip_mm"]==height]
            by_height.append(dict(
                requested_lip_mm=float(height),trials=len(sample),
                median_raw_only_action_l2=float(np.median(
                    [v["raw_only_action_l2"] for v in sample])),
                median_fit_only_action_l2=float(np.median(
                    [v["fit_only_action_l2"] for v in sample])),
                median_raw_and_fit_action_l2=float(np.median(
                    [v["raw_and_fit_action_l2"] for v in sample])),
                median_raw_input_change_l2_m=float(np.median(
                    [v["raw_range_l2_m"] for v in sample]))))
        policy_info["by_height"]=by_height
    (output/"vision_summary.json").write_text(json.dumps(dict(
        sensor_contract=env.sensor_contract,source_kind="deployable-packet-only",
        height_fixture_mm=list(heights_mm),worlds=worlds,seed=seed,
        physical_backend=jax.default_backend(),projected_stats=stats,
        actor_counterfactual=policy_info,
        no_neural_training=True,
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
           "| Lip | Worlds | Raw valid | Plane valid | Fit RMS | Far−near hit | Largest row jump |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in stats:
        # Extremal maxima remain in JSON for lineage, but are deliberately
        # absent from the decision-facing table: a wall hit is not a top.
        contrast=x["median_far_minus_near_hit_mm"]
        jump=x["median_largest_signed_row_jump_mm"]
        lines.append(
            f"| {x['requested_lip_mm']:+.1f}mm | {x['sampled_worlds']} | "
            f"{x['raw_valid_fraction']:.0%} | "
            f"{x['single_plane_valid_fraction']:.0%} | "
            f"{x['mean_plane_residual_mm']:.1f}mm | "
            f"{contrast:+.1f}mm | {jump:+.1f}mm |"
            if contrast is not None and jump is not None else
            f"| {x['requested_lip_mm']:+.1f}mm | {x['sampled_worlds']} | "
            f"{x['raw_valid_fraction']:.0%} | "
            f"{x['single_plane_valid_fraction']:.0%} | "
            f"{x['mean_plane_residual_mm']:.1f}mm | unknown | unknown |")
    lines.extend(["","Far−near: median hit-Z of farthest of 4 rows minus nearest row.",
                  "Largest jump: biggest signed between-adjacent-row height jump",
                  "per world, then median across matched trials. These geometry",
                  "features cancel foot-reference height, but WALL hits and ramps",
                  "are NOT verified walkable upper surfaces.","",
                  "Full ray/foot coordinates: `vision_birth.csv`.",
                  "Per-world row geometry: `vision_spatial.csv`."])
    if policy_info is not None:
        lines.extend(["","## Frozen actor: vision-only counterfactual",
                      f"Actor SHA256: `{policy_info['actor_weights_sha256']}`",
                      "Each row replaces ONLY ToF ranges/valid bits, derived",
                      "ground fit, or both in an otherwise identical FLAT",
                      "actor input. Values are L2 changes across SIX normalized",
                      "servo action commands (not motor current or climbing).",
                      "All four actor history frames are swapped together.",
                      "| Lip | Raw input Δ (m L2) | Raw-only action Δ | Fit-only action Δ | Both action Δ |",
                      "|---:|---:|---:|---:|---:|"])
        for x in policy_info["by_height"]:
            lines.append(f"| {x['requested_lip_mm']:+.1f}mm | "
                         f"{x['median_raw_input_change_l2_m']:.4f} | "
                         f"{x['median_raw_only_action_l2']:.5f} | "
                         f"{x['median_fit_only_action_l2']:.5f} | "
                         f"{x['median_raw_and_fit_action_l2']:.5f} |")
        lines.extend(["","These are matched sensory counterfactuals, NOT a",
                      "closed-loop motor or stair-climbing performance test.",
                      "Individual six-action vectors and worlds:",
                      "`vision_policy_sensitivity.csv`."])
    lines.extend(["","Next gate: physical foot-lift feasibility, trusted",
                  "support inference and a controlled height-aware PPO fork."])
    (output/"vision_summary.md").write_text("\n".join(lines)+"\n")
    print(f"Vision probe complete: {output/'vision_summary.md'}",flush=True)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--worlds",type=int,default=104)
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--policy",type=Path,default=None,
                   help="Optional verified .emupolicy: frozen flat-vs-ledge NN action sensitivity")
    p.add_argument("--heights-mm",default=None,
                   help="comma-separated signed physical heights, e.g. 0,25,35,50,-35")
    a=p.parse_args()
    heights=DEFAULT_HEIGHTS_MM if a.heights_mm is None else tuple(
        float(v) for v in a.heights_mm.split(","))
    run(device=a.device,worlds=a.worlds,output=a.output,seed=a.seed,
        heights_mm=heights,policy=a.policy)


if __name__=="__main__":
    main()
