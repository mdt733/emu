"""Safety/meaning checks for true skill-rated stair heights without GPU."""
import numpy as np
import pytest


def test_rating_exploration_includes_below_near_above_and_independent_flat():
    from emu.terrain import adaptive_height_sample
    x=np.array([.10,.35,.55,.70,.80,.95,.99],dtype=float)
    exposed=np.zeros(x.shape)
    sample=adaptive_height_sample(np,.005,x,exposed,exposure=.60,
                                  probe_above_m=.002,safety_m=.006)
    assert np.min(sample[:2]) < .004
    assert .004 <= sample[3] <= .006
    assert np.max(sample[4:]) > .005
    assert np.max(sample) <= .006
    # Independent never-on-course cohort is still present at every skill.
    flat=adaptive_height_sample(np,.005,x,np.ones_like(x),
                                exposure=.60,probe_above_m=.002,safety_m=.006)
    assert np.all(flat==0.)


def test_zero_skill_rating_has_positive_microprobes_but_no_free_upgrade():
    from emu.terrain import adaptive_height_sample, rated_height_update
    x=np.linspace(0.,1.,201)
    h=adaptive_height_sample(np,0.,x,np.zeros_like(x),exposure=.95,
                             probe_above_m=.002,safety_m=.020)
    assert np.min(h)==0.
    assert np.any(h>0.)
    assert np.max(h)<=.00025+1e-10
    rating,event=rated_height_update(0.,dict(
        terrain_below_done_count=0,terrain_below_fall_count=0,
        terrain_above_done_count=6,terrain_above_cross_count=6,
        terrain_above_fall_count=0,terrain_above_contact_count=6,
        terrain_above_cross_height_sum_m=.001),
        safety_m=.020)
    assert rating==0.
    assert event.startswith('hold_')


def test_rating_promotion_retreat_and_hold_from_separate_physical_bands():
    from emu.terrain import rated_height_update
    base=dict(terrain_below_done_count=16.,terrain_below_fall_count=0.,
              terrain_above_done_count=12.,terrain_above_cross_count=6.,
              terrain_above_fall_count=2.,terrain_above_contact_count=9.,
              terrain_above_cross_height_sum_m=.036)
    h,e=rated_height_update(.005,base,safety_m=.020,growth_m=.001)
    assert h==pytest.approx(.006)
    assert e.startswith('expand_')
    weak_above=dict(base,terrain_above_cross_count=1.,
                    terrain_above_fall_count=10.)
    h,e=rated_height_update(.005,weak_above,safety_m=.020)
    assert h==.005 and e.startswith('hold_')
    weak_below=dict(base,terrain_below_fall_count=12.)
    h,e=rated_height_update(.005,weak_below,safety_m=.020)
    assert h==pytest.approx(.004) and e.startswith('retreat_')
    h,e=rated_height_update(0.,weak_below,safety_m=.020)
    assert h==0.
    capped=dict(base)
    h,e=rated_height_update(.006,capped,safety_m=.006)
    assert h==.006


def test_measured_completed_attempts_are_binned_against_birth_rating():
    from emu.scoring import terrain_edge_trials
    heights=np.array([[.003,.004,.006,.007]],dtype=float)
    rating=np.full_like(heights,.005)
    metrics=dict(
        terrain_edge_mask=np.ones_like(heights),
        terrain_edge_done=np.ones_like(heights),
        terrain_edge_crossing=np.array([[1.,1.,1.,0.]]),
        terrain_edge_contact=np.array([[1.,1.,1.,1.]]),
        terrain_edge_fell=np.array([[0.,0.,0.,1.]]),
        terrain_height_birth_m=heights,
        terrain_birth_rating_m=rating,
    )
    r=terrain_edge_trials(metrics)
    assert r['terrain_below_done_count']==2
    assert r['terrain_below_cross_count']==2
    assert r['terrain_below_fall_count']==0
    assert r['terrain_above_done_count']==2
    assert r['terrain_above_cross_count']==1
    assert r['terrain_above_fall_count']==1
    assert r['terrain_above_contact_count']==2
    assert r['terrain_above_cross_height_sum_m']==pytest.approx(.006)


def test_old_foundation_curriculum_checkpoint_can_load_without_band_counts():
    from emu.curriculum import RegimeGovernor
    old=RegimeGovernor().state_dict()
    for key in ('below_done','below_fall','above_done','above_cross',
                'above_fall','above_contact','above_cross_height_sum_m'):
        old['terrain_edge_evidence'].pop(key)
    new=RegimeGovernor(terrain_height_safety_m=.006,terrain_height_initial_m=0.,
                       terrain_rated_elevation=True)
    new.load_state_dict(old)
    assert new.params.frontier_terrain_height_m==0.
    assert new.terrain_edge_evidence['above_done']==0.


def test_bridge_plan_preserves_every_old_gait_profile_setting(tmp_path,capsys):
    import json
    from emu.cli import main
    base=['train','--forward-preview','--envs','2048','--steps','32000000',
          '--run-dir',str(tmp_path/'unused'),'--plan']
    assert main(base+['--profile','foundation'])==0
    out=capsys.readouterr().out
    foundation=json.loads(out[out.index('{'):out.rindex('}')+1])
    assert main(base+['--profile','terrain-bridge',
                      '--terrain-varied-placement','--terrain-elevation-mode','up',
                      '--terrain-height-ceiling-mm','6',
                      '--terrain-height-initial-mm','0'])==0
    out=capsys.readouterr().out
    bridge=json.loads(out[out.index('{'):out.rindex('}')+1])
    only_new={'terrain_random_placement','terrain_elevation_mode',
              'terrain_height_safety_m','terrain_height_initial_m',
              'terrain_progress_weight','terrain_height_exposure',
              'terrain_height_edge_fraction','terrain_height_growth_m',
              'approved_parent_git_commit','approved_parent_sha256'}
    for k in foundation:
        if k not in only_new:
            assert bridge[k]==foundation[k],k
    assert bridge['terrain_height_initial_m']==0.
    assert bridge['terrain_height_safety_m']==pytest.approx(.006)


def test_cli_rejects_nonzero_initial_platform_rating(tmp_path):
    from emu.cli import main
    args=['train','--profile','terrain-bridge','--forward-preview',
          '--terrain-varied-placement','--terrain-elevation-mode','up',
          '--terrain-height-ceiling-mm','6','--terrain-height-initial-mm','3',
          '--envs','2048','--steps','32000000',
          '--run-dir',str(tmp_path/'bad'),'--plan']
    with pytest.raises(ValueError,match='0mm'):
        main(args)


def test_platform_and_floor_do_not_collide_with_each_other_but_feet_do(tmp_path):
    """A buried 2m kinematic platform must not crowd out foot contact slots."""
    import mujoco
    from emu.terrain import adaptive_height_course
    from emu.contracts import ROBOT_XML
    path=adaptive_height_course(ROBOT_XML,tmp_path/'isolated.xml',
                                elevation_platform=True)
    model=mujoco.MjModel.from_xml_path(str(path))
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    platform=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                              'v5018_obstacle_step')
    foot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'r_foot_ball')
    def compatible(a,b):
        return bool((int(model.geom_contype[a])&int(model.geom_conaffinity[b])) or
                    (int(model.geom_contype[b])&int(model.geom_conaffinity[a])))
    assert not compatible(floor,platform)
    assert compatible(floor,foot)
    assert compatible(platform,foot)
