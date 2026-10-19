"""Summarise v5018 read-only gait diagnostics and discrete-step pilot JSON.

Reports data only; never promotes a policy or changes pass criteria.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean


def aggregate(root: Path):
    rows = []
    reasons = {}
    for path in sorted(root.glob('stride_*/*/directional_exam.json')):
        info=json.loads(path.read_text(encoding='utf-8'))
        label=path.parent.parent.name
        controller=path.parent.name
        worlds=info.get('worlds',[])
        diagnostics=[]
        for w in worlds:
            d=w.get('stride_gate_diagnostics')
            if d is None:
                continue
            diagnostics.append(d)
            for foot in d['per_foot']:
                tag=(label,controller,foot['foot'])
                counts=reasons.setdefault(tag,{
                    'liftoffs':0,'touchdowns':0,'qualifying_touchdowns':0,
                    'airborne_under_3_frames':0,'toe_clearance_under_4mm':0,
                    'foot_relocation_under_12mm':0,'body_motion_under_25mm':0})
                for k in ('liftoffs','touchdowns','qualifying_touchdowns'):
                    counts[k]+=foot[k]
                for k,v in foot['touchdown_rejections_nonexclusive'].items():
                    counts[k]+=v
        if not worlds:
            continue
        rows.append({
            'kind':'heading_stride_diagnosis','course':label,'mode':controller,
            'worlds':len(worlds),
            'course_pass_count':sum(bool(x.get('course_pass')) for x in worlds),
            'walking_pass_count':sum(bool(x.get('walking_pass')) for x in worlds),
            'avg_signed_progress_m':mean(
                float(x.get('signed_course_progress_m') or 0) for x in worlds),
            'avg_heading_mae_deg':mean(
                float(x.get('heading_mae_deg') or 0) for x in worlds),
            'qualified_strides_total':sum(
                int(x.get('contact_stride_pairs') or 0) for x in worlds),
            'has_stride_diagnostic':bool(diagnostics),
        })
    for path in sorted(root.glob('step_*.json')):
        info=json.loads(path.read_text(encoding='utf-8'))
        worlds=info.get('per_world',[])
        reject_counts={}
        for w in worlds:
            for reason in w.get('rejection_reasons',[]):
                reject_counts[reason]=reject_counts.get(reason,0)+1
        rows.append({
            'kind':'discrete_collidable_step',
            'course':f"{info['step_height_mm']}mm",
            'mode':'on' if info['heading_kp'] else 'off',
            'seed':info['seed'],'worlds':info['worlds'],
            'requested_speed_m_s':info.get('requested_body_vx_m_s'),
            'step_yaw_deg':info.get('step_yaw_deg',0.),
            'crossing_pass':info.get('crossing_success_worlds'),
            'step_contact_world_frames':info.get('step_contact_world_frames'),
            'foot_frames_inside_step_xy':info.get('foot_frames_inside_step_xy'),
            'foot_frames_inside_step_xy_bottom_below_top_by_3mm':
                info.get('foot_frames_inside_step_xy_bottom_below_top_by_3mm'),
            'mean_signed_progress_m':info.get('mean_signed_progress_m'),
            'rejection_counts':reject_counts,
        })
    payload={
        'schema':1,'source_folder':str(root),
        'rows':rows,
        'stride_rejection_reasons':[
            {'course':k[0],'mode':k[1],'foot':k[2],**v}
            for k,v in sorted(reasons.items())],
        'warnings':[
            'A successfully executed exam is NOT a successful crossing.',
            'The flat off-course box is a locomotion control, not obstacle success.',
            'Stride rejection counters overlap; do not sum them as independent events.',
            'No PPO terrain training or policy changes occurred.',
            'Physical actuator safety and real-world transfer remain unproven.',
        ],
    }
    out=root/'v5018_gait_and_step_summary.json'
    out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(f"=== {len(rows)} recorded reports ===")
    for row in rows:
        if row['kind']=='heading_stride_diagnosis':
            print(f"{row['course']:23} {row['mode']:3} "
                  f"course {row['course_pass_count']}/{row['worlds']} "
                  f"walking {row['walking_pass_count']}/{row['worlds']} "
                  f"strides={row['qualified_strides_total']} "
                  f"diagnostics={'yes' if row['has_stride_diagnostic'] else 'no'}")
        else:
            print(f"step={row['course']:4} yaw={row['step_yaw_deg']:+.0f}deg "
                  f"speed={row['requested_speed_m_s']:.2f}m/s "
                  f"kp={row['mode']} seed={row['seed']} "
                  f"crossing={row['crossing_pass']}/{row['worlds']} "
                  f"box_contact_frames={row['step_contact_world_frames']} "
                  f"foot_in_step_xy={row['foot_frames_inside_step_xy']} "
                  f"failure_reasons={row['rejection_counts']}")
    print(f"Saved: {out}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    aggregate(args.folder)
