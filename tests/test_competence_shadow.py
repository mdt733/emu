"""Offline-only visibility of skill frontiers: no GPU or learning mutation."""
import copy
import csv

import pytest

from emu.competence_shadow import competence_shadow,main


def _rows(fall=.15,precision=.82, count=20):
    return [{
        "step":str(20_000_000+i*131072),
        "episodes":"64",
        "fall_rate":str(fall),
        "frontier_vx_hi":".40",
        "frontier_vx_lo":"-.12",
        "frontier_vy":".14",
        "frontier_yaw":".90",
        "frontier_combo_radius":"1.",
        "frontier_terrain_height_m":".02",
        "terrain_edge_done_count":"3",
        "frontier_precision_vx_forward":str(precision),
        "frontier_edge_frac_vx_forward":".04",
        "frontier_precision_vx_reverse":"-1.",
        "frontier_edge_frac_vx_reverse":"0.",
        "frontier_precision_vy":str(precision),
        "frontier_edge_frac_vy":".04",
        "frontier_precision_yaw":str(precision),
        "frontier_edge_frac_yaw":".04",
        "frontier_precision_combo":"-1.",
        "frontier_edge_frac_combo":"0.",
    } for i in range(count)]


def test_good_training_evidence_is_not_certified_walking():
    data=_rows()
    before=copy.deepcopy(data)
    result=competence_shadow(data,{
        "open_frontier":False,"frontier_vx_hi_safety":1.2,
        "terrain_height_safety_m":.15},window=12)
    assert data==before
    assert result["command_frontier_actively_controls_training"] is False
    assert result["skills"]["forward"]["training_frontier"]==pytest.approx(.40)
    assert result["skills"]["forward"]["safety_ceiling"]==pytest.approx(1.2)
    assert "promising" in result["skills"]["forward"]["shadow_assessment"]
    assert result["skills"]["forward"]["certified_capability"]["value"] is None
    assert result["skills"]["reverse"]["certified_capability"]["value"] is None
    assert result["skills"]["reverse"]["shadow_assessment"]=="insufficient training evidence"
    assert result["skills"]["physical_steps"]["training_frontier"]==pytest.approx(20)
    assert result["skills"]["physical_steps"]["safety_ceiling"]==pytest.approx(150)
    assert result["skills"]["physical_steps"]["completed_edge_trials_in_window"]==36


def test_small_cohorts_and_no_edge_exposure_are_unknown():
    data=_rows()
    for row in data:
        row["episodes"]="0"
    result=competence_shadow(data,window=12)
    assert result["observed_recent_fall_rate"] is None
    assert result["skills"]["forward"]["shadow_assessment"]=="insufficient training evidence"
    for row in data:
        row["episodes"]="20"
        row["frontier_edge_frac_vx_forward"]="0"
    result=competence_shadow(data,window=12)
    assert result["skills"]["forward"]["shadow_assessment"]=="insufficient training evidence"


def test_fall_veto_prevents_false_promotion():
    data=_rows(fall=.80)
    result=competence_shadow(data,window=12)
    assert "high overall fall rate" in result["skills"]["forward"]["shadow_assessment"]
    assert result["skills"]["forward"]["certified_capability"]["value"] is None


def test_invalid_or_missing_training_records_fail_without_inventing_capability():
    with pytest.raises(ValueError,match="zero training rows"):
        competence_shadow([])
    data=_rows()
    data[-1]["fall_rate"]="nan"
    with pytest.raises(ValueError,match="nonfinite"):
        competence_shadow(data)
    data=_rows()
    data[-1]["episodes"]="-3"
    with pytest.raises(ValueError,match="invalid completed"):
        competence_shadow(data)


def test_offline_cli_reads_existing_csv_without_writing_to_run(tmp_path,capsys):
    rows=_rows(count=5)
    path=tmp_path/"metrics.csv"
    with path.open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    originals=sorted(x.name for x in tmp_path.iterdir())
    assert main(["--metrics",str(path),"--window","5"]) == 0
    assert '"not independently examined"' in capsys.readouterr().out
    assert sorted(x.name for x in tmp_path.iterdir())==originals
