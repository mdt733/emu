"""Stage A reporting: pure artifact reconstruction and exact-height physical matrix."""
import csv
import json
import runpy
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
MODULE=runpy.run_path(str(SCRIPTS/"report_step_academy.py"))

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def test_full_stage_a_report_and_bundle_from_synthetic_campaign(tmp_path):
    manifest={
        "study":"v5037-stage-a-single-lip",
        "parent_step":280_035_328,
        "parent_sha256":"c"*64,
        "source_commit":"a"*40,
        "steps_absolute":290_000_000,
        "arms":[{"index":i,"name":f"arm_{i:02d}_fixture",
                 "changes":{}} for i in range(8)]
    }
    (tmp_path/"campaign_manifest.json").write_text(json.dumps(manifest))
    for a in manifest["arms"]:
        d=tmp_path/a["name"]
        write_csv(d/"metrics.csv",[
            {"step":290_000_000,"fall_rate":.10,"vx_mean":.17,
             "cmd_vx_mean":.18,"terrain_near_attempts_up":10,
             "terrain_near_successes_up":5+a["index"]%2,
             "terrain_near_attempts_down":10,
             "terrain_near_successes_down":8,
             "terrain_easy_attempts_up":10,
             "terrain_easy_successes_up":9,
             "terrain_height_m":.029,
             "focus_alloc_step":.85},
        ])
        write_csv(d/"fixed_eval.csv",[
            {"step":290_000_000,"eval_broad_score":.83+.001*a["index"],
             "eval_core_survival":1.0,
             "eval_resilience75_survival":.92},
        ])
        (d/"run_meta.json").write_text(json.dumps(
            {"source_provenance":{"git_commit":"a"*40}}))
        (d/"checkpoints").mkdir(parents=True)
        (d/"checkpoints/final.pkl").write_bytes(b"not a real checkpoint")
    baseline=(tmp_path/"arm_00_fixture/checkpoints/final.pkl").read_bytes()
    assert MODULE["main"](["--root",str(tmp_path)])==0
    analysis=tmp_path/"analysis"
    report=(analysis/"step_academy_report.md").read_text()
    assert "Near-frontier uphill" in report
    assert "0/0" in report, "unrun physical exams should be explicitly unknown"
    assert "NOT evaluated" in report
    assert (analysis/"step_academy_report.csv").is_file()
    assert (analysis/"step_academy_report.json").is_file()
    assert (analysis/"step_academy_results.zip").is_file()
    import zipfile
    with zipfile.ZipFile(analysis/"step_academy_results.zip") as z:
        assert "campaign_manifest.json" in z.namelist()
        assert "analysis/step_academy_report.md" in z.namelist()
        assert "arm_05_fixture/metrics.csv" in z.namelist()
        assert not any(n.endswith(".pkl") for n in z.namelist())
    assert (tmp_path/"arm_00_fixture/checkpoints/final.pkl").read_bytes()==baseline

def test_paired_physical_exam_aggregation_does_not_merge_ascent_and_descent(tmp_path):
    analysis=tmp_path/"analysis"
    root=analysis/"physical"/"arm_00_fixture"/"seed_1701"
    root.mkdir(parents=True)
    rows=[]
    for height in MODULE["HEIGHTS"]:
        rows.append({
            "case":{"kind":"step","step_m":height/100},
            "worlds":[
                {"direction":"step-up","crossing_pass":True,"mission_pass":True},
                {"direction":"step-down","crossing_pass":False,"mission_pass":False},
                {"direction":"step-up","crossing_pass":False,"mission_pass":False},
                {"direction":"step-down","crossing_pass":True,"mission_pass":True},
            ]
        })
    (root/"unified_exam.json").write_text(json.dumps({
        "protocol":{"terrain_matrix":True,
                    "matrix_heights_cm":list(MODULE["HEIGHTS"])},
        "results":rows
    }))
    exams,data=MODULE["physical_matrix"](analysis,"arm_00_fixture")
    assert len(exams)==1
    assert data["3"]["up_total"]==2
    assert data["3"]["up_ok"]==1
    assert data["3"]["down_total"]==2
    assert data["3"]["down_ok"]==1
