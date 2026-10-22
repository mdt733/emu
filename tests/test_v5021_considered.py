"""V5021 cold-start/servo/frontier gates (CPU only; no GPU launch)."""
from dataclasses import replace
from pathlib import Path
import subprocess
import numpy as np
import pytest

from emu.training_config import considered_config, TrainingConfig
from emu.train_cli import verify_rated_parent_continuation
from emu.rewards import training_reward

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/'scripts/run_v5021_considered.sh'


def test_considered_config_is_true_cold_start_with_expandable_command_dimensions(tmp_path):
    p=considered_config(run_dir=tmp_path/'s10',steps=10_000_000,seed=733,
                        device='cuda',envs=2048)
    assert p.resume == ''
    assert p.envs == 2048 and p.seed == 733
    assert p.open_frontier is True
    assert p.frontier_vx_hi_safety > 10.0
    assert p.frontier_vx_lo_safety > 1.0
    assert p.motor_guidance_weight > 0
    assert p.considered_reward is True
    assert p.freeze_obs_norm_after_steps == 0
    assert p.athletic_rehearsal_frac > 0
    assert p.cloud_probes_per_eval == 0


def test_cold_start_frontiers_begin_at_seed_and_can_really_expand():
    from emu.curriculum import RegimeGovernor
    v=RegimeGovernor(open_frontier=True,
        frontier_vx_hi_safety=12.0, frontier_vx_lo_safety=4.,
        frontier_vy_safety=4., frontier_yaw_safety=12.,
        cold_start_frontier=True,terrain_height_safety_m=.150,
        terrain_height_initial_m=0.,terrain_rated_elevation=True)
    assert v.params.frontier_vx_hi==pytest.approx(.060)
    assert v.params.frontier_vx_lo==pytest.approx(-.020)
    assert v.params.frontier_vy==pytest.approx(.020)
    assert v.params.frontier_yaw==pytest.approx(.150)
    assert v.params.frontier_terrain_height_m==0.
    assert v.frontier_vx_hi_safety==12.
    assert v.frontier_growth_frac==pytest.approx(.015)
    assert v.FRONTIER_FLOOR_VX_HI==.060
    historical=RegimeGovernor()
    assert historical.FRONTIER_FLOOR_VX_HI==.60
    assert historical.frontier_growth_frac==pytest.approx(.0015)


def test_considered_cli_requires_312_actor_and_real_150mm_technical_course(tmp_path,capsys):
    from emu.cli import main
    core=['train','--profile','considered','--run-dir',str(tmp_path/'s10'),
          '--steps','10000000','--envs','2048','--seed','733',
          '--device','cuda','--plan']
    with pytest.raises(ValueError,match='312'):
        main(core)
    extras=['--forward-preview','--pitch-deg','8',
            '--terrain-varied-placement','--terrain-elevation-mode','up',
            '--terrain-height-ceiling-mm','150',
            '--terrain-height-initial-mm','0',
            '--terrain-height-exposure','.40','--terrain-height-edge-fraction','.30',
            '--terrain-height-growth-mm','1','--terrain-progress-weight','.04']
    assert main(core+extras)==0
    output=capsys.readouterr().out
    assert '"considered_reward": true' in output
    assert '"observation_schema": "rolling_tof_312"' in output
    assert '"open_frontier": true' in output
    assert '"terrain_height_safety_m": 0.15' in output
    assert '"motor_guidance_weight": 0.1' in output
    assert '"resume": ""' in output
    assert not (tmp_path/'s10').exists()
    with pytest.raises(ValueError,match='150'):
        main(core+extras[:extras.index('--terrain-height-ceiling-mm')]+
             ['--terrain-height-ceiling-mm','6']+
             extras[extras.index('--terrain-height-initial-mm'):])


def metrics(overload=0., shaft=0., jerk=0., slip=0.):
    x=lambda val:np.asarray([val],dtype=np.float32)
    return dict(
        terminal=x(0.),gravity_z=x(1.),height=x(.25),
        target_height=x(.25),height_error=x(0.),
        attitude_quality=x(1.),foot_mid_cone=x(1.),
        bad_contact=x(0.),support=x(1.),stance_slip=x(slip),
        foot_impact_speed_mps=x(slip),
        swing_clearance=x(.015),target_step_height=x(.012),
        organic_step_shape=x(1.),action_delta_norm=x(jerk),
        joint_limit_pressure=x(0.),scissor_pressure=x(0.),
        stepping_activity_so_far=x(.3),age_s=x(4.),
        cmd_vx=x(.2),vx=x(.2),cmd_vy=x(0.),vy=x(0.),
        cmd_yaw=x(0.),yaw_rate=x(0.),body_pitch=x(0.),cmd_pitch=x(0.),
        ramp_frac=x(1.),servo_over_rated_frac=x(overload),
        servo_shaft_abs_power_w=x(shaft),
    )


def test_considered_motor_payment_meaningful_and_legacy_economics_preserved():
    regime=np.asarray([0.],dtype=np.float32)
    clean, clean_terms=training_reward(metrics(),regime,
        attitude_profile='v5015',xp=np,
        motor_guidance_weight=.10,considered_reward=True)
    harsh, harsh_terms=training_reward(
        metrics(overload=.8,shaft=13.,jerk=1.,slip=.25),regime,
        attitude_profile='v5015',xp=np,
        motor_guidance_weight=.10,considered_reward=True)
    _, old_terms=training_reward(
        metrics(overload=.8,shaft=13.,jerk=1.,slip=.25),regime,
        attitude_profile='v5015',xp=np,
        motor_guidance_weight=.10,considered_reward=False)
    assert clean_terms['weighted_motor_cost'][0]==pytest.approx(0.)
    assert harsh_terms['weighted_motor_cost'][0] < -.2
    assert harsh_terms['weighted_motor_cost'][0] < old_terms['weighted_motor_cost'][0]
    legacy, terms=training_reward(
        metrics(overload=.8,shaft=13.,jerk=1.,slip=.25),regime,
        attitude_profile='v5015',xp=np,considered_reward=False)
    assert terms['weighted_motor_cost'][0] == 0.
    assert np.isfinite([clean[0],harsh[0],legacy[0]]).all()


def test_source_pinned_cumulative_stages_must_not_mutate_motor_or_frontier(tmp_path):
    p=replace(considered_config(run_dir=tmp_path/'first',
               steps=10_000_000,seed=733,device='cuda',envs=2048),
              observation_schema='rolling_tof_312',
              terrain_random_placement=True,terrain_elevation_mode='up',
              terrain_height_safety_m=.150,terrain_height_initial_m=0.,
              terrain_height_exposure=.40,terrain_height_edge_fraction=.30,
              terrain_height_growth_m=.001,terrain_progress_weight=.04)
    q=replace(p,run_dir=str(tmp_path/'second'),steps=25_000_000)
    verify_rated_parent_continuation(p,q)
    with pytest.raises(RuntimeError,match='Non-continuous'):
        verify_rated_parent_continuation(p,replace(q,motor_guidance_weight=0))
    with pytest.raises(RuntimeError,match='Non-continuous'):
        verify_rated_parent_continuation(p,replace(q,frontier_vx_hi_safety=1.0))


def test_launcher_has_four_continuous_stages_and_never_uses_old_brains():
    syntax=subprocess.run(['bash','-n',str(LAUNCHER)],capture_output=True,text=True)
    assert syntax.returncode==0,syntax.stderr
    help=subprocess.run(['bash',str(LAUNCHER),'--help'],cwd=ROOT,
                        capture_output=True,text=True)
    assert help.returncode==0
    script=LAUNCHER.read_text()
    for label,target in [('s10','10000000'),('s25','25000000'),
                         ('s50','50000000'),('s100','100000000')]:
        assert f'run_stage {label} {target}' in script
    assert '--profile considered' in script
    assert '--terrain-height-ceiling-mm 150' in script
    assert 'git status --porcelain' in script
    assert 'source_commit.txt' in script
    assert 'rm -rf' not in script
    assert 'git checkout main' not in script
    assert 'approved-parent' not in script
    assert 'run_stage s10 10000000 ""' in script
