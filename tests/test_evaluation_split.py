"""Guard the stable evaluation API and independently requested physics exams."""
from __future__ import annotations

import numpy as np

from emu import evaluation as api
from emu import physical_exam as physical


def test_evaluation_api_preserves_exam_identity():
    assert api.PolicySnapshot.__module__ == "emu.evaluation"
    assert api.evaluate_directional_run is physical.evaluate_directional_run
    assert api.score_terrain_course_trace is physical.score_terrain_course_trace
    assert api.evaluate_terrain_course is physical.evaluate_terrain_course
    assert api.evaluate_directional_pair is physical.evaluate_directional_pair
    assert api.exam_request is physical.exam_request
    assert api.source_commit is physical.source_commit
    assert api.make_schedule is physical.make_schedule
    assert api.SCENARIOS is physical.SCENARIOS
    assert api.score_unified_capability_trace is physical.score_unified_capability_trace
    assert api.unified_terrain_matrix_cases is physical.unified_terrain_matrix_cases
    assert api.evaluate_unified_capabilities is physical.evaluate_unified_capabilities


def test_physical_case_grid_keeps_true_signed_step_pairs():
    cases=api.unified_terrain_matrix_cases(grades_deg=(5.,15.),
                                           heights_cm=(2.,2.5,3.,3.5))
    assert len(cases)==12
    assert [c["name"] for c in cases if c["kind"]=="step"]==[
        "paired_steps_2p0cm","paired_steps_2p5cm",
        "paired_steps_3p0cm","paired_steps_3p5cm"]
    assert {(c["direction"],c["slope_deg"]) for c in cases
            if c["kind"]=="slope"}=={
        (d,angle) for d in ("uphill","downhill","cross_left","cross_right")
        for angle in (5.,15.)}


def test_shared_operator_command_schedules_remain_deterministic():
    for name in physical.SCENARIOS:
        first=api.make_schedule(name,120)
        second=api.make_schedule(name,120)
        assert first.shape==(120,3)
        assert first.dtype==np.float32
        np.testing.assert_array_equal(first,second)
    brake=api.make_schedule("brake_stop",120,speed=.15)
    np.testing.assert_allclose(brake[:80,0],.15)
    np.testing.assert_array_equal(brake[80:],np.zeros((40,3),np.float32))


def test_v5029_speed_ladder_is_read_only_and_exposed_on_cli(capsys):
    from emu.cli import main
    try:
        main(["capability-exam","--help"])
    except SystemExit as exc:
        assert exc.code==0
    out=capsys.readouterr().out
    assert "--speed-ladder" in out
    assert "--ladder-speeds" in out
    assert api.evaluate_unified_capabilities is physical.evaluate_unified_capabilities

def test_v5029_700_frame_speed_exam_cannot_timeout_on_last_frame():
    import inspect
    code=inspect.getsource(physical.evaluate_unified_capabilities)
    assert 'max_episode_steps=max(700,horizon+50)' in code


def test_v5030_fast_flat_ladder_extends_only_its_own_schedule_envelope():
    import pytest
    for requested in (.8,1.0,1.4,1.8):
        with pytest.raises(ValueError,match='safe test envelope'):
            physical.make_schedule('steady',120,speed=requested)
        trace=physical.make_schedule('steady',120,speed=requested,
                                     lateral=0.,yaw_rate=0.,
                                     fast_flat_ladder=True)
        np.testing.assert_allclose(trace[:,0],requested,atol=1e-7)
        np.testing.assert_array_equal(trace[:,1:],np.zeros((120,2),np.float32))
    for invalid in (1.81,2.0,-1.81):
        with pytest.raises(ValueError,match='safe test envelope'):
            physical.make_schedule('steady',120,speed=invalid,
                                   lateral=0.,yaw_rate=0.,
                                   fast_flat_ladder=True)
    for kwargs in ({'lateral':.05},{'yaw_rate':.2}):
        with pytest.raises(ValueError,match='straight flat steady'):
            physical.make_schedule('steady',120,speed=.8,
                                   fast_flat_ladder=True,**kwargs)
    with pytest.raises(ValueError,match='straight flat steady'):
        physical.make_schedule('speed_step',120,speed=.8,
                               lateral=0.,yaw_rate=0.,fast_flat_ladder=True)



def test_v5030_explicit_main_sensor_contract_in_shared_physical_world():
    """There is no fake 232=>360 conversion, only shared real colliders."""
    from emu.environment import EmuMJXEnv
    from emu.contracts import OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN
    import inspect
    spec=inspect.signature(EmuMJXEnv)
    assert spec.parameters['physical_exam_legacy_actor'].default is False
    src=inspect.getsource(EmuMJXEnv.__init__)
    assert "physical_exam_legacy_actor" in src
    assert "add_unified_terrain" in src
    assert "observation_schema==OBS_SCHEMA_TERRAIN" in src
    from emu.cli import main
    import pytest
    with pytest.raises(SystemExit) as ex:
        main(['capability-exam','--help'])
    assert ex.value.code==0
    # An opt-in flag is mandatory: never silently reinterpret main weights.
    assert '--legacy-main-baseline' in inspect.getsource(main)


def test_v5030_legacy_baseline_is_strictly_nontraining():
    from emu.environment import EmuMJXEnv
    from emu.physical_exam import evaluate_unified_capabilities
    import inspect
    assert "physical_exam_legacy_actor=bool(legacy_main_baseline)" in inspect.getsource(
        evaluate_unified_capabilities)
    assert 'physical_exam_legacy_actor' not in inspect.getsource(
        __import__('emu.training',fromlist=['env_kwargs_from_args']).env_kwargs_from_args)
