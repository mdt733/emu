"""v5016 matched heading P ablations: pure law and reporting contracts."""
import json
from pathlib import Path

import numpy as np
import pytest

from emu.directional import heading_rate_request
from emu.directional_exam import evaluate_pair, _summarize_worlds


def test_one_shared_controller_law_handles_wrap_sign_and_saturation():
    out, error = heading_rate_request(
        np, .0, np.deg2rad(179), np.deg2rad(-179),
        kp=1.0, max_rate=.9)
    assert float(error) == pytest.approx(-np.deg2rad(2))
    assert float(out) < 0
    out, error = heading_rate_request(
        np, np.array([.2, -.2]), np.array([.5, -.5]),
        np.zeros(2), kp=2, max_rate=.5)
    np.testing.assert_allclose(out,[.5,-.5])
    np.testing.assert_allclose(error,[.5,-.5])


def test_pair_writes_provenance_checked_comparison_without_selecting_winner(
    monkeypatch, tmp_path
):
    import emu.directional_exam as exam
    run=tmp_path/'old_training'
    run.mkdir()
    calls=[]
    def fake_eval(run_dir, *, output, heading_kp, **args):
        assert Path(run_dir)==run
        calls.append((heading_kp,args))
        out=Path(output)
        out.mkdir(parents=True)
        report={
            'checkpoint_sha256':'a'*64,
            'checkpoint_training_step':80_000_000,
            'source_commit':'exam-source',
            'exam':dict(seed=args['seed'],worlds=2,horizon=args['horizon'],
                speed=args['speed'],lateral=0.,yaw_rate=0.,
                angle_deg=.5,direction='uphill',settle_frames=0,
                scenario=args.get('scenario','steady'),
                turn_rate=args.get('turn_rate',.25),
                gyro_bias_rad_s=args.get('gyro_bias',0.),
                gyro_noise_std_rad_s=args.get('gyro_noise',0.),
                gyro_delay_frames=args.get('gyro_delay_frames',0),
                heading_kp=heading_kp,heading_p_enabled=heading_kp>0),
            'worlds':[
                dict(full_horizon_survival=True,course_pass=bool(heading_kp),
                     walking_pass=False,signed_course_progress_m=.1,
                     heading_mae_deg=10 if heading_kp else 20,
                     gyro_heading_error_mae_deg=1,
                     mean_abs_controller_correction_rad_s=heading_kp*.1),
                dict(full_horizon_survival=False,course_pass=False,
                     walking_pass=False),
            ]}
        f=out/'directional_exam.json'
        f.write_text(json.dumps(report))
        md=out/'directional_exam.md'
        md.write_text('mock')
        return f,md
    monkeypatch.setattr(exam,'evaluate_run',fake_eval)
    outfile=tmp_path/'ablation'
    paths=evaluate_pair(run,output=outfile,heading_kp=.75,
       angle_deg=.5,direction='uphill',seed=1701,worlds=2,
       speed=.1,horizon=100)
    assert len(calls)==2
    assert calls[0][0]==0 and calls[1][0]==.75
    assert calls[0][1]==calls[1][1]
    report=json.loads(paths[0].read_text())
    assert report['checkpoint_sha256']=='a'*64
    assert report['on_minus_off']['course_pass']==pytest.approx(.5)
    assert report['off']['survived_full']==report['on']['survived_full']
    assert not report['off']['gait_candidate_pass']
    assert (outfile/'heading_comparison.md').exists()
    assert 'no automatic promotion' in report['conclusion']
    with pytest.raises(FileExistsError):
        evaluate_pair(run,output=outfile,heading_kp=.75,
            angle_deg=.5,direction='uphill',seed=1701,worlds=2,
            speed=.1,horizon=100)


def test_comparison_rejects_missing_metrics_instead_of_making_up_scores():
    r=_summarize_worlds([
      dict(full_horizon_survival=False,course_pass=False,walking_pass=False),
      dict(full_horizon_survival=False,course_pass=False,walking_pass=False)
    ])
    assert r['heading_mae_deg'] is None
    assert r['progress_m'] is None
    assert r['course_pass']==0
