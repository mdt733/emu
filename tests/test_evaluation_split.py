"""Guard the examiner refactor without invoking MJX or a GPU.

These tests assert that published CLI/caller imports retain their exact
function identity and that the shared command schedule is still operator-
defined (not inferred from the policy's movement).
"""
from __future__ import annotations

import numpy as np

from emu import evaluation as api
from emu import evaluation_atlas as atlas
from emu import evaluation_directional as directional
from emu import evaluation_commands as commands
from emu import evaluation_unified as unified


def test_evaluation_facade_preserves_callable_identity():
    assert api.PolicySnapshot.__module__ == "emu.evaluation"
    assert api.atlas_main is atlas.atlas_main
    assert api.checkpoint_training_profile is atlas.checkpoint_training_profile
    assert api.evaluate_directional_run is directional.evaluate_directional_run
    assert api.score_terrain_course_trace is directional.score_terrain_course_trace
    assert api.evaluate_terrain_course is directional.evaluate_terrain_course
    assert api.evaluate_directional_pair is directional.evaluate_directional_pair
    assert api.exam_request is directional.exam_request
    assert api.source_commit is directional.source_commit
    assert api.make_schedule is commands.make_schedule
    assert api.SCENARIOS is commands.SCENARIOS
    assert api.score_unified_capability_trace is unified.score_unified_capability_trace
    assert api.unified_terrain_matrix_cases is unified.unified_terrain_matrix_cases
    assert api.evaluate_unified_capabilities is unified.evaluate_unified_capabilities


def test_frozen_terrain_case_list_and_real_up_down_pairing():
    # Four bearings per grade, plus separate physically paired step heights.
    cases=api.unified_terrain_matrix_cases(grades_deg=(5.,15.),
                                           heights_cm=(2.,2.5,3.,3.5))
    assert len(cases)==12
    assert [c["name"] for c in cases if c["kind"]=="step"]==[
        "paired_steps_2p0cm","paired_steps_2p5cm",
        "paired_steps_3p0cm","paired_steps_3p5cm"]
    assert {(c["direction"], c["slope_deg"]) for c in cases
            if c["kind"]=="slope"}=={
        (d,deg) for d in ("uphill","downhill","cross_left","cross_right")
        for deg in (5.,15.)}


def test_shared_operator_commands_remain_deterministic():
    for scenario in commands.SCENARIOS:
        first=api.make_schedule(scenario,120)
        second=api.make_schedule(scenario,120)
        assert first.shape==(120,3)
        assert first.dtype==np.float32
        np.testing.assert_array_equal(first,second)
    brake=api.make_schedule("brake_stop",120,speed=.15)
    np.testing.assert_allclose(brake[:80,0],.15)
    np.testing.assert_array_equal(brake[80:],np.zeros((40,3),np.float32))
