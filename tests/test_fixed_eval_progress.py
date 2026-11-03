"""Fixed evaluator progress must not change held-out policy evidence.

Native JAX evaluation can spend tens of seconds compiling unseen MJX worlds.
Use a deliberately cheap fake trainer here: the real contact/physics tests
cover geometry, while these tests preserve the exam API and periodic logging.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from emu import evaluation


def test_progress_reports_every_panel_without_changing_scored_evidence():
    calls=[]
    def trainer_summary(params, packet, mean, std, key, horizon):
        calls.append(int(horizon))
        assert int(horizon)==evaluation.EVAL_HORIZON
        assert packet is not None
        return np.asarray([.75,270.,.04,.03,.06,.96,.9,.3,.8,.05,.006],
                          dtype=np.float32)

    worlds={
        name:(
            SimpleNamespace(num_envs=32,attitude_profile='v5015',
                            params_array=lambda ep: ep),
            {'evaluate_summary':trainer_summary},
        )
        for name in {spec[0] for spec in evaluation.PANELS.values()}
    }
    norm={'mean':np.zeros((2,),np.float32),
          'var':np.ones((2,),np.float32)}
    baseline=evaluation.evaluate_fixed(worlds,object(),norm,step=10_027_008,cohorts=2)
    messages=[]
    measured=evaluation.evaluate_fixed(worlds,object(),norm,step=10_027_008,
                                       cohorts=2,progress=messages.append)
    assert measured.keys()==baseline.keys()
    for key in baseline:
        assert measured[key]==baseline[key] if not key.startswith('perf_') else True
    assert len(calls)==2 * 2 * len(evaluation.PANELS)
    assert 'BEGIN' in messages[0]
    assert sum(' START ' in m for m in messages)==10
    assert sum(' DONE ' in m for m in messages)==10
    assert sum('cohort ' in m and ' running' in m for m in messages)==20
    assert 'COMPLETE' in messages[-1]
    assert measured['perf_eval_env_steps']==10*2*32*evaluation.EVAL_HORIZON


def test_heartbeat_during_slow_first_compile_stops_after_completion():
    messages=[]
    with evaluation._eval_heartbeat(messages.append,'slow frozen exam',
                                    interval_s=.005):
        time.sleep(.025)
    assert any('still working' in s and 'slow frozen exam' in s for s in messages)
    count=len(messages)
    time.sleep(.02)
    assert len(messages)==count
