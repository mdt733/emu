"""Human-facing v5020 progress prints, separate from PPO learning contracts."""
import numpy as np
import pytest

from emu.results import format_line, shaft_work_per_forward_m


def test_shafts_are_measured_joules_per_real_forward_metre_not_fake_electrical_watts():
    # Three forward-command frames use 10W each; one non-forward command
    # frame draws 100W but MUST NOT contaminate the forward comparison.
    metrics={
        'servo_shaft_abs_power_w': np.array([[10.,10.,10.,100.]]),
        'terrain_forward_delta_m': np.array([[.020,.020,-.010,.020]]),
        'cmd_vx': np.array([[.2,.2,.2,-.2]])
    }
    # J = 10W*0.02s*3 = 0.6J; NET +X travel is 0.030m.
    assert shaft_work_per_forward_m(metrics)==pytest.approx(20.)
    metrics['terrain_forward_delta_m'][:]=.00001
    assert shaft_work_per_forward_m(metrics)==-1.
    metrics['cmd_vx'][:]=0.
    assert shaft_work_per_forward_m(metrics)==-1.
    assert shaft_work_per_forward_m({})==-1.


def test_progress_is_one_clear_line_and_is_honest_about_command_range_and_rating():
    row={'ep_len_mean':700.,'fall_rate':0.,
         'servo_over_rated_frac':.621,
         'servo_shaft_work_per_forward_m_j':15.}
    governor={'frontier_vx_lo':-.12,
              'frontier_vx_hi':.40,
              'resilience_difficulty':.47,
              'frontier_terrain_height_m':.000231,
              'global_retreat':False}
    line=format_line('emu_v5015',34668544,3.23,row,governor,
                     train_fps=31813,wall_fps=31044,
                     rated_terrain=True,slope_deg=0.)
    assert len(line.splitlines()) == 1
    for label in ('steps=34,668,544','rew=+3.230','len=700','fall=0.0%',
                  'fps=31,044','vx_cmd=[-0.12,+0.40]m/s',
                  'slope=0.0deg','step=0.231mm',
                  'resil=0.47','shaft=15.0J/m','over=62%'):
        assert label in line
    for discarded in ('fail(h/t/r/p)', 'cone=', 'sync=', 'train_fps=', 'wall_fps='):
        assert discarded not in line


def test_progress_never_mislabels_legacy_dormant_step_height_as_skill():
    row={'ep_len_mean':400.,'fall_rate':.125,
         'servo_shaft_work_per_forward_m_j':-1.}
    gov={'frontier_terrain_height_m':.003,
         'resilience_difficulty':.1,
         'global_retreat':True}
    out=format_line('emu_v5015',12345,1.0,row,gov,
                    train_fps=20000,
                    rated_terrain=False,variable_slope=True,
                    nan_grad_skips=2)
    assert 'step=flat' in out
    assert 'slope=var' in out
    assert 'shaft=n/a' in out
    assert 'WARNING:nan_skips=2' in out
    assert 'RETREAT' in out
