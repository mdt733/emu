"""CPU-only examiner tests; do not need a checkpoint, MuJoCo or GPU."""
from __future__ import annotations

import json

import numpy as np
import pytest

from emu.physical_quality import (
    axis, build_report, commanded_world_path, curve_area, report_main, wilson,
)


def _row(index: int, passed: bool, *, direction="flat", effort=None):
    return dict(
        world=index, direction=direction,
        mission_pass=passed, case_pass=passed, survived=True,
        movement_tracking_pass=True, heading_control_pass=passed,
        crossing_pass=passed if direction.startswith("step") else None,
        achieved_mean_vx_mps=.15,
        tracking_signed_bias_yaw_rad_s=0.01,
        tracking_jitter_yaw_rad_s=.10,
        world_path_rms_error_m=.022,
        world_path_endpoint_error_m=.031,
        directed_progress_m=.9,
        mechanical_j_per_useful_m=effort,
    )


def _case(case, rows):
    return dict(case=case, worlds=rows, passed=sum(w["case_pass"] for w in rows),
                mission_passed=sum(w["mission_pass"] for w in rows),
                survived=sum(w["survived"] for w in rows))


def _exam(seed=1701, actor="a"*64, *, include_more=False):
    a = [_row(0, True), _row(1, True)]
    b = [_row(0, True), _row(1, False)]
    step = [_row(0, True, direction="step-up", effort=78.),
            _row(1, True, direction="step-down", effort=82.)]
    cases = [
        _case(dict(name="flat_vx_0p30",kind="flat",vx=.3),a),
        _case(dict(name="flat_vx_0p80",kind="flat",vx=.8),b),
        _case(dict(name="uphill_5deg",kind="slope",direction="uphill",slope_deg=5.),a),
        _case(dict(name="uphill_10deg",kind="slope",direction="uphill",slope_deg=10.),b),
        _case(dict(name="paired_steps_2p0cm",kind="step",step_m=.02),step),
    ]
    if include_more:
        cases += [
            _case(dict(name="downhill_5deg",kind="slope",direction="downhill",slope_deg=5.),a),
            _case(dict(name="lateral_left",kind="command",vx=0.,vy=.05),a),
            _case(dict(name="yaw_left",kind="command",vx=0.,yaw=.2),a),
            _case(dict(name="push_1N",kind="recovery",push_n=1.),a),
            _case(dict(name="start_from_rest",kind="transition",scenario="rest_start"),a),
        ]
    return dict(
        schema="emu_unified_frozen_exam_v2_dual_scores",
        actor_artifact_sha256=actor,
        observation_schema="terrain_360",source_training_step=260046848,
        protocol=dict(seed=seed,worlds=2,horizon=350,speed=.15,
                      backend="cpu",physical_fixture_revision="two_32m_half_length_sharp_slabs_v5029",
                      no_reward_or_curriculum_feedback=True),
        results=cases)


def test_success_probability_intervals_include_uncertainty():
    low, high=wilson(8,8)
    assert .67<low<.68 and high==1.
    low, high=wilson(0,8)
    assert low==0. and .32<high<.33
    with pytest.raises(ValueError):
        wilson(9,8)


def test_curve_area_honestly_uses_only_tested_range():
    a=curve_area([(2.,1.),(3.,0.)])
    assert a["minimum"]==2.
    assert a["maximum"]==3.
    assert a["raw_success_area"]==.5
    assert a["tested_range_fraction"]==.5
    assert curve_area([(3.,1.)])["tested_range_fraction"] is None
    assert curve_area([])["raw_success_area"] is None
    with pytest.raises(ValueError):
        curve_area([(2.,1.),(2.,1.)])


def test_world_path_matches_true_commanded_direction():
    dt=.02
    n=110
    cmd=np.zeros((n,3),dtype=float)
    cmd[:,0]=.2
    xy=np.stack((np.arange(n)*dt*.2, np.zeros(n)),axis=1)
    heading=np.zeros(n)
    result=commanded_world_path(xy,heading,cmd,dt=dt,settle_frames=25)
    assert result["world_path_endpoint_error_m"]==0.
    assert result["world_path_rms_error_m"]==0.
    assert result["world_cross_track_rms_m"]==0.
    assert result["directed_progress_m"]==pytest.approx((n-1-25)*dt*.2,abs=1e-5)
    assert result["useful_progress_ratio"]==pytest.approx(1.,abs=1e-5)


def test_body_forward_tracking_cannot_hide_world_path_drift():
    dt=.02
    n=150
    cmd=np.zeros((n,3),dtype=float);cmd[:,0]=.2
    # Robot moves 0.2m/s exactly along its OWN heading, but slowly
    # turns off-course despite a straight 0-yaw command.
    yaw=np.linspace(0.,.6,n)
    dv=.2*dt*np.stack((np.cos(yaw[:-1]),np.sin(yaw[:-1])),axis=1)
    xy=np.concatenate((np.zeros((1,2)),np.cumsum(dv,axis=0)),axis=0)
    got=commanded_world_path(xy,yaw,cmd,dt=dt,settle_frames=25)
    assert got["world_path_rms_error_m"]>.03
    assert got["world_cross_track_rms_m"]>.03
    assert got["world_heading_error_end_rad"]>.45
    assert got["world_path_endpoint_error_m"]>.08
    # Over-walking without directional progress doesn't fabricate energy
    # qualification; only mission-qualified progress is accepted elsewhere.


def test_in_place_yaw_has_no_fictitious_useful_metres():
    n=80
    cmd=np.zeros((n,3),dtype=float);cmd[:,2]=.3
    heading=np.arange(n)*.3*.02
    xy=np.zeros((n,2),dtype=float)
    got=commanded_world_path(xy,heading,cmd,settle_frames=25)
    assert got["directed_progress_m"] is None
    assert got["useful_progress_ratio"] is None
    assert got["world_path_endpoint_error_m"]==0.
    assert got["world_heading_error_end_rad"]==pytest.approx(0.,abs=1e-4)


def test_per_direction_steps_do_not_steal_each_others_success():
    s=build_report([_exam()])
    up=next(x for x in s["cases"] if x["family"]=="step" and x["direction"]=="step-up")
    down=next(x for x in s["cases"] if x["family"]=="step" and x["direction"]=="step-down")
    assert (up["missions"],up["attempts"])==(1,1)
    assert (down["missions"],down["attempts"])==(1,1)
    assert up["supported_crossing"]==1
    assert s["global_score"] is None
    assert s["coverage"]["resilience"]["complete"] is False
    assert s["coverage"]["terrain"]["complete"] is False
    assert "slope/downhill" in s["coverage"]["terrain"]["missing"]
    curves={(x["family"],x["direction"]):x for x in s["tested_envelopes"]}
    assert curves[("slope","uphill")]["tested_range_fraction"]==.75
    assert curves[("step","step-up")]["tested_range_fraction"] is None


def test_two_seeds_pool_into_same_severity_no_duplicate_case_weighting():
    r=build_report([_exam(1701),_exam(1702)])
    up=next(x for x in r["cases"] if x["family"]=="step" and x["direction"]=="step-up")
    assert up["attempts"]==2 and up["missions"]==2
    assert up["seeds"]==[1701,1702]
    assert r["seeds"]==[1701,1702]
    with pytest.raises(ValueError,match="repeated"):
        build_report([_exam(1701),_exam(1701)])


def test_multiple_actors_cannot_be_blended_into_fake_quality():
    with pytest.raises(ValueError,match="different actor"):
        build_report([_exam(actor="a"*64),_exam(actor="b"*64)])


def test_partial_quality_does_not_report_unobserved_dimensions_as_zero():
    r=build_report([_exam(include_more=True)])
    assert not r["coverage"]["control"]["complete"]
    assert "yaw/right" in r["coverage"]["control"]["missing"]
    assert "transition/brake_stop" in r["coverage"]["control"]["missing"]
    assert not r["coverage"]["resilience"]["complete"]
    assert "recovery/tilt" in r["coverage"]["resilience"]["missing"]
    assert r["coverage"]["grace"]["complete"] is False
    assert all(x["useful_work_j_per_m"] is None or
               x["useful_work_samples"]>0 for x in r["cases"])
    json.dumps(r,allow_nan=False)


def test_reporting_command_only_writes_to_new_destination(tmp_path):
    src=tmp_path/"exam"
    src.mkdir()
    (src/"unified_exam.json").write_text(json.dumps(_exam()))
    out=tmp_path/"scorecard"
    assert report_main(["--exam",str(src),"--output",str(out)])==0
    assert (out/"physical_quality.json").is_file()
    assert (out/"physical_quality.md").is_file()
    assert "No global score" in (out/"physical_quality.md").read_text()
    with pytest.raises(FileExistsError):
        report_main(["--exam",str(src),"--output",str(out)])
    with pytest.raises(ValueError,match="separate"):
        report_main(["--exam",str(src),"--output",str(src/"inside")])


def test_default_control_case_is_a_legitimate_forward_command():
    p=_exam()["protocol"]
    label=axis(dict(name="control_easy_0p08",kind="control",vx=.08),
               _row(0,True),p)
    assert label==("longitudinal","forward",.08,"m/s")


def test_mixed_incline_step_retains_true_up_and_down_evidence():
    p=_exam()["protocol"]
    case=dict(name="mixed_incline_step",kind="mixed",step_m=.008,
              slope_deg=2.,edge_deg=17.)
    up=axis(case,_row(0,True,direction="step-up"),p)
    down=axis(case,_row(1,True,direction="step-down"),p)
    assert up[1]!=down[1]
    assert up[0]==down[0]=="mixed_terrain"


def test_cli_exposes_read_only_quality_command(capsys):
    from emu.cli import main
    with pytest.raises(SystemExit) as ex:
        main(["physical-quality","--help"])
    assert ex.value.code==0
    out=capsys.readouterr().out
    assert "--exam" in out and "--output" in out


def test_failure_decomposition_is_explicit_and_nonexclusive():
    exam=_exam()
    # The fast flat case has one qualified and one unqualified world.
    # The unqualified world still survives and moves correctly; it fails
    # only steering, just like a downhill heading-control failure.
    r=build_report([exam])
    row=next(c for c in r["cases"]
             if c["family"]=="longitudinal" and c["severity"]==.8)
    causes=row["nonexclusive_failure_reasons"]
    assert causes["not_mission_qualified"]==1
    assert causes["terminated_early"]==0
    assert causes["movement_failed"]==0
    assert causes["heading_failed"]==1
    assert causes["supported_crossing_failed"] is None
    assert causes["slope_stability_failed"] is None


def test_physical_step_failures_remain_distinct_from_heading_failures():
    exam=_exam()
    step=next(c for c in exam["results"] if c["case"]["kind"]=="step")
    # An otherwise upright, heading-correct robot may fail the true lip.
    step["worlds"][0]["crossing_pass"]=False
    step["worlds"][0]["mission_pass"]=False
    step["mission_passed"]=1
    profile=build_report([exam])
    up=next(c for c in profile["cases"] if c["family"]=="step"
            and c["direction"]=="step-up")
    assert up["supported_crossing"]==0
    assert up["nonexclusive_failure_reasons"]["supported_crossing_failed"]==1
    assert up["nonexclusive_failure_reasons"]["heading_failed"]==0
    assert up["nonexclusive_failure_reasons"]["not_mission_qualified"]==1
