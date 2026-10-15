"""v5016 command-schedule and batched exam contracts without GPU."""
import json
import zipfile

import numpy as np
import pytest

from emu.command_exams import SCENARIOS, make_schedule
from emu import v5016_batch


def test_turn_stop_schedule_has_one_true_turn_and_a_long_heading_hold():
    n=160
    left=make_schedule('turn_stop_left',n,speed=.1,turn_rate=.25)
    right=make_schedule('turn_stop_right',n,speed=.1,turn_rate=.25)
    assert left.shape==(n,3)
    assert np.all(left[:n//4,2]==0)
    assert np.all(left[n//4:2*n//3,2]==.25)
    assert np.all(left[2*n//3:,2]==0)
    np.testing.assert_allclose(right[:,2],-left[:,2])
    np.testing.assert_allclose(right[:,0],left[:,0])
    assert float(np.sum(left[:,2])*.02) > .25


def test_lateral_reverse_stop_resume_are_distinct_operator_requests():
    left=make_schedule('sidestep_left',160,lateral=.07)
    right=make_schedule('sidestep_right',160,lateral=.07)
    back=make_schedule('reverse',160)
    pause=make_schedule('stop_resume',160)
    assert np.all(left[50:70,0]==0)
    assert np.all(left[50:70,1]==.07)
    np.testing.assert_allclose(right[:,1],-left[:,1])
    assert np.all(back[50:70,0]<0)
    assert np.all(pause[50:70,0]==0)
    assert np.all(pause[:40,0]>.0)
    assert np.all(pause[120:,0]>.0)


@pytest.mark.parametrize('bad', ['unknown',None])
def test_bad_schedules_fail_closed(bad):
    with pytest.raises(ValueError):
        make_schedule(bad,160)


def test_invalid_scenario_rates_and_nonfinite_commands_fail_closed():
    with pytest.raises(ValueError):
        make_schedule('steady',20)
    with pytest.raises(ValueError):
        make_schedule('turn_stop_left',160,turn_rate=0)
    with pytest.raises(ValueError):
        make_schedule('steady',160,speed=float('nan'))
    assert 'turn_stop_left' in SCENARIOS


def test_batch_preview_is_small_staged_and_writes_nothing(tmp_path,capsys):
    source=tmp_path/'old'
    dest=tmp_path/'exam'
    assert len(v5016_batch.plan('smoke',source,dest))==2
    assert len(v5016_batch.plan('pilot',source,dest))==20
    sensor=v5016_batch.plan('sensor',source,dest)
    assert len(sensor)==4
    assert all(j['gyro_bias']==.015 and j['gyro_delay_frames']==2 for j in sensor)
    assert len(v5016_batch.plan('stretch',source,dest))==4
    jobs=v5016_batch.plan('smoke',source,dest)
    v5016_batch.execute(jobs,'a'*40)
    assert 'PLAN ONLY' in capsys.readouterr().out
    assert not dest.exists()


def test_batch_never_writes_inside_training_or_reuses_bad_evidence(tmp_path):
    run=tmp_path/'old'
    out=tmp_path/'exam'
    with pytest.raises(ValueError,match='separate'):
        v5016_batch.plan('pilot',run,run/'temp')
    j=v5016_batch.plan('smoke',run,out)[0]
    report=out/'smoke'/'baseline'/'flat'/'heading_comparison.json'
    assert not v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        'checkpoint_sha256':'b'*64,
        'examined_at_revision':'a'*40,
        'protocol':dict(angle_deg=0.,direction='uphill',scenario='steady',
                        worlds=2,horizon=40,seed=1701,gyro_bias_rad_s=0.,
                        gyro_noise_std_rad_s=0.,gyro_delay_frames=0),
        'heading_kp':.75,
    }))
    with pytest.raises(ValueError,match='incomplete'):
        v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    for mode in ('on','off'):
        path=report.parent/mode
        path.mkdir()
        (path/'directional_exam.json').write_text('{}')
    assert v5016_batch.verify_existing(report,j,'a'*40,'b'*64)
    with pytest.raises(ValueError,match='stale'):
        v5016_batch.verify_existing(report,j,'c'*40,'b'*64)


def test_results_bundle_contains_raw_paired_evidence_not_checkpoints(tmp_path):
    source=tmp_path/'fourway'
    dest=tmp_path/'out'
    jobs=v5016_batch.plan('smoke',source,dest)
    j=jobs[0]
    folder=__import__('pathlib').Path(j['output'])
    folder.mkdir(parents=True)
    comparison={
        'checkpoint_sha256':'b'*64,
        'protocol':{'scenario':'steady'},
        'off':{'survived_full':.5,'course_pass':.0,'heading_mae_deg':35.},
        'on':{'survived_full':1.,'course_pass':.5,'heading_mae_deg':10.},
        'on_minus_off':{'course_pass':.5},
    }
    (folder/'heading_comparison.json').write_text(json.dumps(comparison))
    for mode in ('on','off'):
        p=folder/mode
        p.mkdir()
        (p/'directional_exam.json').write_text(json.dumps({'worlds':[]}))
    (folder/'secret.pkl').write_bytes(b'unsharable trained checkpoint')
    profile=folder.parents[1]
    bundle,count=v5016_batch.summarize(jobs,profile,'a'*40)
    assert count==1
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert 'results.md' in names
        assert 'baseline/flat/off/directional_exam.json' in names
        assert not any(name.endswith('.pkl') for name in names)
