"""Research-only, CPU-free checks for the independent KS150 survival screen."""
from __future__ import annotations

from scripts.score_h40_ks150_heldout import (
    ACTORS, BANDS, EXPECTED_HEIGHTS, SEED, safe_ascent_blockers,
)


def _scores(*, safe_survivors: int, frontier_survivors: int) -> dict:
    return {
        "step_up_safe_4_40": {
            "survived": safe_survivors, "n": 500,
            "survival_frac": safe_survivors / 500,
        },
        "step_up_frontier_44_56": {
            "survived": frontier_survivors, "n": 200,
            "survival_frac": frontier_survivors / 200,
        },
    }


def test_actual_v1_hazard_fails_v2_hardware_safety_screen():
    # Real seed1701 matrix from frozen qualification evidence:
    # 120M 490/500 easy survival; 196/200 frontier;
    # 150M 490/500 easy survival; 107/200 frontier;
    # 160M 476/500 easy survival; 89/200 frontier.
    parent = _scores(safe_survivors=490, frontier_survivors=196)
    good_upto_4cm = _scores(safe_survivors=490, frontier_survivors=107)
    too_risky = _scores(safe_survivors=476, frontier_survivors=89)
    assert safe_ascent_blockers(parent, good_upto_4cm) == [
        "uphill 44–56mm survival below 90%"
    ]
    assert safe_ascent_blockers(parent, too_risky) == [
        "uphill 4–40mm survival regression",
        "uphill 44–56mm survival below 90%",
    ]


def test_safe_ascent_requires_separate_real_survival_not_just_crossings():
    reference = _scores(safe_survivors=490, frontier_survivors=196)
    safe = _scores(safe_survivors=480, frontier_survivors=180)
    assert safe_ascent_blockers(reference, safe) == []
    failed = _scores(safe_survivors=420, frontier_survivors=180)
    failed["step_up_safe_4_40"]["traversals"] = 500
    assert "uphill 4–40mm survival regression" in safe_ascent_blockers(
        reference, failed)


def test_exam_grid_and_actors_are_fixed_before_launch():
    assert SEED == 2701
    assert set(ACTORS) == {120, 150, 160}
    assert len(set(ACTORS.values())) == 3
    assert EXPECTED_HEIGHTS == tuple(float(i) for i in range(4, 81, 4))
    assert BANDS == (
        ("safe_4_40", 4, 40),
        ("frontier_44_56", 44, 56),
        ("stretch_44_80", 44, 80),
    )


def _academy_row(*, near_attempts=0., near_successes=0.,
                 safety_near=0., safety_near_survived=0.,
                 easy_attempts=0., easy_successes=0.,
                 safety_easy=0., safety_easy_survived=0.):
    return dict(
        academy_active_fraction=.20,
        academy_near_attempts_up=near_attempts,
        academy_near_successes_up=near_successes,
        academy_easy_attempts_up=easy_attempts,
        academy_easy_successes_up=easy_successes,
        safe_ascent_near_attempts_up=safety_near,
        safe_ascent_near_survived_up=safety_near_survived,
        safe_ascent_near_crossed_up=min(safety_near,near_successes),
        safe_ascent_easy_attempts_up=safety_easy,
        safe_ascent_easy_survived_up=safety_easy_survived,
        safe_ascent_easy_crossed_up=min(safety_easy,easy_successes),
    )


def test_v2_requires_both_supported_crossings_and_24_surviving_episodes():
    from emu.curriculum import RegimeGovernor
    school=RegimeGovernor(unified_capability_focus=True,
                          safe_ascent_enabled=True)
    school.params.academy_height_up_m=.04
    school.params.academy_height_down_m=.04
    row=_academy_row(
        near_attempts=12, near_successes=10,
        safety_near=12, safety_near_survived=12)
    event=school.advance_academy(row,max_height_m=.055,enabled=True)
    assert "up:collect_first_life" in event
    assert school.params.academy_height_up_m==.04
    event=school.advance_academy(row,max_height_m=.055,enabled=True)
    assert "up:expand_survival_and_cross" in event
    assert abs(school.params.academy_height_up_m-.0405)<1e-10
    assert school.params.academy_height_down_m==.04
    assert school.safe_ascent_evidence==school._empty_safe_ascent_evidence()


def test_v2_retreats_despite_successful_footholds_if_robot_falls():
    from emu.curriculum import RegimeGovernor
    school=RegimeGovernor(unified_capability_focus=True,
                          safe_ascent_enabled=True)
    school.params.academy_height_up_m=.045
    event=school.advance_academy(_academy_row(
        near_attempts=12,near_successes=12,safety_near=24,
        safety_near_survived=13),max_height_m=.055,enabled=True)
    assert "up:survival_retreat_near" in event
    assert abs(school.params.academy_height_up_m-.0445)<1e-10
    school=RegimeGovernor(unified_capability_focus=True,
                          safe_ascent_enabled=True)
    school.params.academy_height_up_m=.045
    event=school.advance_academy(_academy_row(
        easy_attempts=12,easy_successes=12,
        safety_easy=24,safety_easy_survived=13),max_height_m=.055,enabled=True)
    assert "up:survival_retreat_easy" in event
    assert abs(school.params.academy_height_up_m-.0445)<1e-10


def test_v2_migrates_only_explicit_v1_checkpoint_and_restores_full_survival_window():
    import pytest
    from emu.curriculum import RegimeGovernor
    v1=RegimeGovernor(unified_capability_focus=True).state_dict()
    v2=RegimeGovernor(unified_capability_focus=True,safe_ascent_enabled=True)
    with pytest.raises(ValueError,match="curriculum state keys mismatch"):
        v2.load_state_dict(v1)
    v2.load_state_dict(v1,allow_safe_ascent_parent_fork=True)
    assert v2.safe_ascent_evidence==v2._empty_safe_ascent_evidence()
    v2.safe_ascent_evidence["near_attempts"]=19.
    v2.safe_ascent_evidence["near_survived"]=18.
    saved=v2.state_dict()
    restart=RegimeGovernor(unified_capability_focus=True,safe_ascent_enabled=True)
    restart.load_state_dict(saved)
    assert restart.safe_ascent_evidence["near_attempts"]==19.
    assert restart.safe_ascent_evidence["near_survived"]==18.
    with pytest.raises(ValueError,match="V2 episode outcomes exceed births"):
        bad={**saved,"safe_ascent_evidence":{
            **saved["safe_ascent_evidence"],"near_survived":21.}}
        restart.load_state_dict(bad)
    with pytest.raises(ValueError,match="curriculum state keys mismatch"):
        RegimeGovernor(unified_capability_focus=True).load_state_dict(saved)


def test_v2_uninterrupted_milestone_boundaries():
    from types import SimpleNamespace
    from emu.training import ks150_permanent_milestones
    from emu.checkpoint_io import update_boundary_at_or_after
    args=SimpleNamespace(
        run_dir="/tmp/h40-ks150-safe-ascent-s735",
        physical_profile="h40-spring-only",
        seed=735, envs=2048, steps=170000000,
        terrain_step_max_m=.055, academy_height_cap_m=.055)
    assert ks150_permanent_milestones(args)=={
        update_boundary_at_or_after(160000000,2048):160}
    wrong=SimpleNamespace(**vars(args))
    wrong.steps=160000000
    import pytest
    with pytest.raises(RuntimeError,match="V2 milestone"):
        ks150_permanent_milestones(wrong)


def test_safe_ascent_v2_rejects_unsafe_harder_steps_even_with_more_crossings():
    from scripts.score_h40_safe_ascent_v2 import audit
    def group(traverse,survive,n):
        return dict(crossed=traverse,survived=survive,n=n)
    safe=group(230,488,500)
    hard=group(20,104,200)
    downhill=group(370,875,1000)
    deep={"stand_zero_60s":dict(
        full_survivals=8,zero_command_balance_passes=8,
        mean_horizontal_speed_mps=.034,mean_yaw_mae_rad_s=.07,
        mean_vx_mae_mps=.02,mean_vy_mae_mps=.02,
        mean_positive_motor_shaft_work_j=200.)}
    for name in ["forward_025_30s"]:
        deep[name]=dict(
            full_survivals=8,mean_vx_mae_mps=.03,
            mean_vy_mae_mps=.03,mean_yaw_mae_rad_s=.08,
            mean_positive_motor_shaft_work_j=100.)
    groups={name:dict(pass_fraction=.9)
            for name in ("command","control","transition","slope","resilience")}
    base=dict(scores={
        "step_up_easy_4_40":safe,
        "step_up_frontier_44_56":hard,
        "step_down_all_4_80":downhill},deep=deep,groups=groups)
    candidate=dict(base,scores={**base["scores"],
        "step_up_frontier_44_56":group(30,110,200)})
    blockers=audit(base,candidate)
    assert any("44–56mm uphill survival" in x for x in blockers)
