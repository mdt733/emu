"""v5041 terrain contract must be exact before replacing v5040 physics."""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from emu.terrain_carpet import (
    Limits, ROLES, TILE_M, check_neighbourhood, edge, hash01,
    neighbours, role_limits, tile,
)


def test_tile_contract_every_role():
    for role in ("standing",*ROLES):
        bounds=role_limits(role,seed=735,slope_deg=12.,
                           transition_deg=4.,step_up_m=.04,
                           step_down_m=.08)
        for x,y in ((.25,.25),(-14.27,9.61),(124.13,-58.55)):
            world=neighbours(x,y,bounds)
            report=check_neighbourhood(world,bounds)
            assert report["tiles"]==9
            assert report["triangles"]==18
            assert report["seam_error_m"] < 1e-9
            assert report["largest_step_m"]<=min(
                bounds.max_step_up_m,bounds.max_step_down_m)+1e-8
            assert len({tuple(t["index"]) for t in world["tiles"]})==9


def test_unequal_up_down_budget_is_conservative_not_conflated():
    for seed in range(64):
        bounds=role_limits("step",seed=seed,slope_deg=7.,
             transition_deg=3.,step_up_m=.02,step_down_m=.09)
        report=check_neighbourhood(neighbours(4.2,-6.5,bounds),bounds)
        assert report["largest_step_m"]<=.02+1e-9
        assert bounds.max_step_up_m==.02
        assert bounds.max_step_down_m==.09


def test_deterministic_tiles_are_global_not_episode_cache():
    limits=Limits(seed=733,max_step_up_m=.07,max_step_down_m=.07)
    before=tile(-20,15,limits)
    for i in range(-9,10):
        neighbours(i*.49,-i*.37,limits)
    assert before==tile(-20,15,limits)


def test_shared_corners_and_real_vertical_lip():
    bounds=Limits(seed=915,max_slope_deg=7,max_transition_deg=3,
                  max_step_up_m=.045,max_step_down_m=.045,step_probability=1)
    touched=0
    for i in range(-12,12):
        for j in range(-8,8):
            a=tile(i,j,bounds)
            for b in (tile(i+1,j,bounds),tile(i,j+1,bounds)):
                e=edge(a,b)
                assert e["end_gaps_m"][0]==pytest.approx(e["end_gaps_m"][1],abs=1e-9)
                touched+=abs(e["rise_m"])>1e-6
    assert touched>100


def test_zero_specialist_limits_are_physically_flat():
    bounds=role_limits("standing",seed=735,slope_deg=16,
                       transition_deg=6,step_up_m=.15,step_down_m=.12)
    for i in range(-2,2):
        for j in range(-2,2):
            assert all(v[2]==0 for v in tile(i,j,bounds)["vertices"])


def test_invalid_difficulty_rejected():
    with pytest.raises(ValueError):
        Limits(max_slope_deg=math.nan).validated()
    with pytest.raises(ValueError):
        role_limits("unknown",seed=1,slope_deg=1,
                    transition_deg=1,step_up_m=.01,step_down_m=.01)
    with pytest.raises(ValueError):
        edge(tile(0,0,Limits()),tile(2,0,Limits()))


def test_actual_viewer_live_endpoint_exposes_real_training_frontiers(tmp_path):
    from emu.viewer_server import read_terrain_training_snapshot
    root=tmp_path/"run"
    root.mkdir()
    (root/"training_config.json").write_text(json.dumps({
        "seed":735, "observation_schema":"terrain_360",
        "academy_stream":True}))
    (root/"run_meta.json").write_text(json.dumps({
        "source_provenance":{"git_commit":"0123456789abcdef"}}))
    (root/"metrics.csv").write_text(
        "step,slope_max_deg,academy_height_up_m,academy_height_down_m,"
        "academy_up_crossings,academy_down_crossings\n"
        "65000000,2,.009,.012,7,9\n"
        "90000000,8,.022,.035,22,45\n")
    data=read_terrain_training_snapshot(root)
    assert data["schema"]=="emu_terrain_live_v1"
    assert data["training_step"]==90000000
    assert data["run_name"]=="run"
    assert data["source_sha"]=="0123456789abcdef"
    assert float(data["latest_metrics"]["slope_max_deg"])==8
    assert float(data["latest_metrics"]["academy_height_up_m"])==.022
    assert float(data["latest_metrics"]["academy_height_down_m"])==.035
    assert "source training metrics" in data["physics_mode"]

def test_mjx_frontier_bands_and_directional_step_limits():
    import jax.numpy as jnp
    import numpy as np
    from emu.terrain_carpet_mjx import (
        make_profiles,tile_geometry,PROFILE_HALF)
    seeds=jnp.array([735,761,902],dtype=jnp.int32)
    up=jnp.full((3,),.02)
    down=jnp.full((3,),.04)
    xprofile,yprofile=make_profiles(jnp,seeds,up,down)
    diffs=np.asarray(jnp.diff(xprofile,axis=1))
    # The first tiles are easy, later tiles stretch beyond the earned
    # frontier, but cannot become arbitrary unbounded cliffs.
    early=np.abs(diffs[:,PROFILE_HALF+1:PROFILE_HALF+3])
    middle=np.abs(diffs[:,PROFILE_HALF+7:PROFILE_HALF+13])
    late=np.abs(diffs[:,PROFILE_HALF+16:PROFILE_HALF+20])
    assert early.max()<.65*.04
    assert middle.mean()>early.mean()
    assert late.mean()>middle.mean()
    assert diffs.max()<1.5*.02+1e-5
    assert diffs.min()>-1.5*.04-1e-5

    pos,h=tile_geometry(jnp,jnp.array([0,1,2]),jnp.array([0,0,0]),
        seeds,xprofile,yprofile,up,down,jnp.full((3,),8.0))
    top=np.asarray(h)
    gx=.5*(top[:,1]+top[:,3]-top[:,0]-top[:,2])
    gy=.5*(top[:,2]+top[:,3]-top[:,0]-top[:,1])
    grade=np.rad2deg(np.arctan(np.hypot(gx,gy)))
    # Per-tile slope is no longer a constant global plane: some 50cm
    # triangles are easier, while others probe the governor's frontier.
    # We must never exceed the requested physical grade by surprise.
    assert np.all(grade<=8.25),grade
    assert np.any(grade>3.0),grade
    # Neighbouring triangles form a continuous grade field whose
    # gradient changes with location, not a fixed pitched slab.
    p1,h1=tile_geometry(jnp,jnp.array([1,2,3]),jnp.array([0,0,0]),
        seeds,xprofile,yprofile,up,down,jnp.full((3,),8.0))
    a=np.asarray(h1)
    g0=np.stack((gx,gy),axis=1)
    g1=np.stack((
        .5*(a[:,1]+a[:,3]-a[:,0]-a[:,2]),
        .5*(a[:,2]+a[:,3]-a[:,0]-a[:,1])),axis=1)
    assert np.any(np.linalg.norm(g1-g0,axis=1)>.001)
    delta_deg=np.rad2deg(np.arctan(np.linalg.norm(g1-g0,axis=1)))
    assert np.all(delta_deg<=3.2),delta_deg


def test_cartesian_gradient_can_change_for_same_seed():
    import jax.numpy as jnp
    import numpy as np
    from emu.terrain_carpet_mjx import (
        make_profiles,tile_geometry)
    n=16
    seeds=jnp.arange(100,116,dtype=jnp.int32)
    up=jnp.zeros((n,),dtype=jnp.float32)
    down=jnp.zeros_like(up)
    px,py=make_profiles(jnp,seeds,up,down)
    prev=None;differences=[]
    for xi in range(-4,8):
        _,h=tile_geometry(jnp,jnp.full((n,),xi),
            jnp.full((n,),2),seeds,px,py,up,down,
            jnp.full((n,),8.),transition_limit_deg=3.)
        a=np.asarray(h)
        gradient=np.stack((
            .5*(a[:,1]+a[:,3]-a[:,0]-a[:,2]),
            .5*(a[:,2]+a[:,3]-a[:,0]-a[:,1])),axis=1)
        if prev is not None:
            differences.append(np.linalg.norm(gradient-prev,axis=1))
        prev=gradient
    changes=np.stack(differences,axis=1)
    # Genuine varied slope transitions across consecutive tiles without
    # dramatic >3-degree jumps in local gradient.
    assert np.any(changes>.002),changes
    angles=np.rad2deg(np.arctan(changes))
    assert np.max(angles)<3.3,angles.max()


def test_bounded_lifecycle_uses_actual_v5041_carpet_in_all_stages(tmp_path):
    """Regression: the lifecycle smoke must not secretly test old slabs."""
    from scripts.smoke_training_lifecycle import configs
    first,continued=configs(tmp_path,unified=True,carpet=True,
                            envs=8,seed=735)
    for cfg in (first,continued):
        assert cfg.terrain_carpet is True
        assert cfg.academy_stream is True
        assert cfg.terrain_lesson == 'coherent-careful'
        assert cfg.growth_recipe == 'main-mature'
        assert cfg.observation_schema == 'terrain_360'
        assert cfg.unified_capability_focus is True
        assert cfg.terrain_step_max_m == .08
        assert cfg.seed == 735 and cfg.envs == 8
    assert first.steps == 8*32
    assert continued.steps == 2*8*32
    assert continued.resume.endswith('/first/checkpoints/final.pkl')
    assert first.resume == ''
    assert first.run_dir != continued.run_dir


def test_historical_unified_lifecycle_defaults_to_old_geometry(tmp_path):
    """The new carpet option must not change previous acceptance recipes."""
    from scripts.smoke_training_lifecycle import configs
    first,_=configs(tmp_path,unified=True,envs=8)
    assert first.terrain_carpet is False
    assert first.academy_stream is False
    assert first.terrain_step_max_m == .04



def test_carpet_exam_first_episode_mask_never_counts_post_fall_resets():
    import numpy as np
    from scripts.exam_step_course import first_episode_mask
    deaths=np.asarray([[0,0],[1,0],[0,0],[0,1],[0,0]],dtype=bool)
    mask=first_episode_mask(deaths)
    assert np.array_equal(mask,[[1,1],[1,1],[0,1],[0,1],[0,0]])
    with pytest.raises(ValueError):
        first_episode_mask(np.zeros((4,),dtype=bool))


def test_frozen_carpet_trace_reports_actual_live_solid_tiles_not_old_slabs():
    """Even a two-frame untrained trace must inspect true nine-box physics."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.ppo import make_trainer
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=4,seed=735,max_episode_steps=700,
        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,academy_stream=True,
        terrain_lesson='coherent-careful',terrain_carpet=True,
        academy_height_cap_m=.15)
    assert not env.academy_geom_ids
    settings=dict(specialist_mix=1.,terrain_exposure=1.,slope_exposure=0.,
                  slope_max_deg=0.,terrain_height_m=.02,
                  academy_height_up_m=.02,academy_height_down_m=.03,
                  focus_weight_step=1.,academy_episode_probability=1.,
                  athletic_rehearsal_frac=0.,
                  regime_frac_steady=1.,regime_frac_transition=0.,
                  regime_frac_resilience=0.,probe_cmd_vx=.10,
                  probe_cmd_vy=0.,probe_cmd_yaw=0.)
    trainer=make_trainer(env,seed=735,lr=2e-4,rollout=2,epochs=1,minibatches=1)
    initial=trainer['make_initial_state'](settings)
    schedule=jnp.asarray([[.10,0.,0.],[.10,0.,0.]],dtype=jnp.float32)
    trace=jax.device_get(trainer['evaluate_trace'](
        initial.params,env.params_array(settings),
        jnp.zeros((360,),dtype=jnp.float32),
        jnp.ones((360,),dtype=jnp.float32),
        jax.random.PRNGKey(1701),2,0.,None,schedule))
    assert np.asarray(trace['floor_contact']).shape==(2,4,2)
    assert np.asarray(trace['academy_active']).shape==(2,4)
    assert np.asarray(trace['initial_carpet_solid_geom_xpos']).shape==(4,9,3)
    assert np.asarray(trace['initial_carpet_solid_geom_xmat']).shape==(4,9,3,3)
    assert np.asarray(trace['initial_carpet_cells']).shape==(4,9,2)
    assert np.asarray(trace['initial_carpet_source_hfields']).shape==(4,36)
    assert np.asarray(trace['initial_foot_radii_m']).shape==(2,)
    assert np.asarray(trace['carpet_raw_contact_pairs']).shape==(2,4,9,2)
    assert np.asarray(trace['carpet_contact_vertical']).shape==(2,4,9,2)
    assert np.asarray(trace['carpet_root_cell']).shape==(2,4,2)
    assert np.all(np.isfinite(trace['initial_carpet_solid_geom_xpos']))
    assert np.all(np.isfinite(trace['floor_normal_world']))
    # The old slab is disabled, and its fictional support must not leak
    # into the frozen policy examiner.
    assert 'academy_initial_platform_positions' not in trace
    assert 'terrain_platform_contact' not in trace



def test_independent_carpet_scorer_rejects_side_contacts_and_post_fall_resets():
    """Synthetic raw MJX trace, no STEP reward/teacher keys whatsoever."""
    import numpy as np
    from scripts.exam_step_course import independently_score_carpet_crossings
    def trace(rise=.03,vertical=1.,fall=False):
        tcount=8
        cells=np.zeros((tcount,1,2),dtype=np.int32)
        cells[1:,0,0]=1
        centres=np.zeros((tcount,1,9,3),dtype=np.float32)
        centres[:,:,:,:]=[0.,0.,-30.]
        centres[:,:,0,:]=[.25,.25,-.27]
        centres[:,:,1,:]=[.75,.25,rise-.27]
        rotations=np.broadcast_to(np.eye(3,dtype=np.float32),
                                  (tcount,1,9,3,3)).copy()
        contacts=np.zeros((tcount,1,9,2),dtype=bool)
        contacts[1:,0,1,0]=True
        normals=np.zeros_like(contacts,dtype=np.float32)
        normals[1:,0,1,0]=vertical
        feet=np.zeros((tcount,1,2,3),dtype=np.float32)
        feet[:,:,:,:]=[.75,.25,rise+.018]
        done=np.zeros((tcount,1),dtype=bool)
        fell=done.copy()
        if fall:
            done[3,0]=True
            fell[3,0]=True
        return dict(carpet_root_cell=cells,
             carpet_solid_geom_xpos=centres,
             carpet_solid_geom_xmat=rotations,
             carpet_contact_vertical=normals,
             carpet_raw_contact_pairs=contacts,
             foot_xyz=feet,
             initial_foot_radii_m=np.asarray([.018,.018]),
             initial_carpet_solid_half_sizes=np.asarray([[.258,.258,.270]]*9),
             done=done,fell=fell)
    success=independently_score_carpet_crossings(trace())
    assert success["up"]==1 and success["down"]==0
    assert success["total"]==1
    assert success["worlds"][0]["independently_supported_lips"][0][
        "signed_height_cm"]==3.0
    downhill=independently_score_carpet_crossings(trace(rise=-.03))
    assert downhill["up"]==0 and downhill["down"]==1
    side=independently_score_carpet_crossings(trace(vertical=.05))
    assert side["total"]==0
    dead=independently_score_carpet_crossings(trace(fall=True))
    assert dead["total"]==0
