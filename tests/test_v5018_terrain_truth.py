"""P1 physical box contact truth: pure fast tests independent of MuJoCo.

These assertions do NOT claim measured impulse, actuator safety, or NN learning.
The historical flat reward path is separately checked by the real MJX smoke.
"""
from __future__ import annotations

import numpy as np
import pytest

from emu.terrain import box_surface_truth


def fixture():
    # geom 0: floor; geom 7: immovable step; feet 3/4 (18mm spheres).
    foot = np.array([
        [[.20,0.,.038], [.143,.05,.018]],   # foot0 on 20mm top, foot1 at riser
        [[-.1,-.08,.018], [.30,.07,.06]],  # floor + free swing
    ],dtype=np.float32)
    prev=foot.copy()
    torso=np.array([[.20,0.,.27],[-.1,0.,.26]],dtype=np.float32)
    box=np.tile(np.array([[.20,0.,.010]],dtype=np.float32),(2,1))
    rot=np.tile(np.eye(3,dtype=np.float32)[None,:,:],(2,1,1))
    floor_normal=np.tile(np.array([[0.,0.,1.]],dtype=np.float32),(2,1))
    geom1=np.array([[7,7,-1],[0,-1,-1]],dtype=np.int32)
    geom2=np.array([[3,4,-1],[3,-1,-1]],dtype=np.int32)
    dist=np.array([[0.,0.,1.],[0.,1.,1.]],dtype=np.float32)
    # Real MuJoCo/MJX contact frame has 9 values per contact: the FIRST
    # three encode the surface normal in world axes. The two feet share
    # the box geom ID but only foot0 touches the horizontal top.
    contact_frame=np.zeros((2,3,9),dtype=np.float32)
    contact_frame[:,:,2]=1.0   # normal along +Z for top/floor by default
    contact_frame[0,1,:3]=[1.,0.,0.]   # the other foot touches a RISER
    contact_position=np.array([
        [[.20,0.,.020],[.160,.05,.010],[0.,0.,0.]],
        [[-.10,-.08,0.],[0.,0.,0.],[0.,0.,0.]],
    ],dtype=np.float32)
    return dict(
        foot_xyz=foot,previous_foot_xyz=prev,torso_xyz=torso,
        floor_foot_signed=foot[:,:,2],floor_torso_signed=torso[:,2],
        floor_normal=floor_normal,
        floor_origin=np.zeros((2,3),dtype=np.float32),
        box_center=box,box_rotation=rot,
        box_half_size=np.array([.04,.20,.01],np.float32),
        contact_geom1=geom1,contact_geom2=geom2,contact_distance=dist,
        contact_frame=contact_frame,contact_position=contact_position,
        floor_geom_id=0,box_geom_id=7,foot_geom_ids=(3,4),
        foot_radius_m=.018,control_dt=.020)


def test_box_top_contact_is_support_but_riser_contact_is_not():
    x=fixture()
    y=box_surface_truth(np,**x)
    assert y['foot_top_contact_candidate'].tolist()==[[True,False],[False,False]]
    assert y['foot_riser_contact'].tolist()==[[False,True],[False,False]]
    assert y['foot_floor_contact_candidate'].tolist()==[[False,False],[True,False]]
    assert y['foot_supported_candidate'].tolist()==[[True,False],[True,False]]
    np.testing.assert_allclose(y['support_fraction'],[.5,.5])
    # Torso directly over raised step is measured relative to its TOP.
    np.testing.assert_allclose(y['torso_height'],[.25,.26],atol=2e-6)
    # A foot centre at +38mm over +20mm step has zero ball-bottom gap.
    np.testing.assert_allclose(y['toe_clearance_signed'][0,0],0.,atol=1e-6)
    assert y['box_contact_any'].tolist()==[True,False]
    assert y['torso_over_box_footprint'].tolist()==[True,False]


def test_foot_bumping_vertical_riser_cannot_farm_top_support():
    x=fixture()
    # Place first sphere inside footprint but LOW, pushing into the box
    # side or penetrating it; the contact pair itself does NOT grant top.
    x['foot_xyz'][0,0,:]=[.18,0.,.018]
    y=box_surface_truth(np,**x)
    assert not bool(y['foot_top_contact_candidate'][0,0])
    assert bool(y['foot_riser_contact'][0,0])
    assert float(y['toe_clearance_signed'][0,0])<0
    assert float(y['support_fraction'][0])==0.
    # No contact pair means no ground support even when geometric height
    # happens to be correct (avoid treating hovering as standing).
    x=fixture()
    x['contact_geom1'][0,:]=-1
    x['contact_geom2'][0,:]=-1
    y=box_surface_truth(np,**x)
    assert float(y['support_fraction'][0])==0.


def test_true_contact_slip_uses_tangential_world_velocity():
    x=fixture()
    x['previous_foot_xyz'][0,0,0]=.18   # 20mm slide in 20ms
    y=box_surface_truth(np,**x)
    np.testing.assert_allclose(y['stance_slip'][0],1.,atol=1e-5)
    # Riser-only sliding is not credited as stance or penalised as a
    # ground-support slide; separately reported riser metric remains.
    x['foot_xyz'][0,0,:]=[.18,0.,.018]
    y=box_surface_truth(np,**x)
    assert float(y['support_fraction'][0])==0
    assert float(y['stance_slip'][0])==0


def test_yawed_obstacle_geometry_tracks_local_not_world_axes():
    x=fixture()
    a=np.deg2rad(25.)
    rot=np.array([[np.cos(a),-np.sin(a),0.],
                  [np.sin(a), np.cos(a),0.],
                  [0.,0.,1.]],dtype=np.float32)
    x['box_rotation'][0]=rot
    # Local point x=0, y=0 at centre of top irrespective of yaw.
    y=box_surface_truth(np,**x)
    assert bool(y['foot_top_contact_candidate'][0,0])
    # Rotate a foot point into a footprint-negative region: cannot
    # score a top contact based on unrotated global XY bounding box.
    x['foot_xyz'][0,0,:]=[.247, .17,.038]
    y=box_surface_truth(np,**x)
    assert not bool(y['foot_top_contact_candidate'][0,0])


def test_jax_jit_shapes_and_cpu_numerics():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    x=fixture()
    def calc(coords):
        args={name:(jnp.asarray(value) if isinstance(value,np.ndarray)
                    else value) for name,value in x.items()}
        args['foot_xyz']=coords
        return box_surface_truth(jnp,**args)
    computed=jax.jit(calc)(jnp.asarray(x['foot_xyz']))
    expected=box_surface_truth(np,**x)
    for k in ('torso_height','support_fraction','stance_slip',
              'swing_clearance','min_toe_clearance_signed'):
        np.testing.assert_allclose(np.asarray(computed[k]),expected[k],
                                   atol=2e-6,rtol=1e-6,err_msg=k)


def test_invalid_foot_radius_or_clock_cannot_produce_reward():
    x=fixture()
    x['foot_radius_m']=0.
    with pytest.raises(ValueError,match='positive'):
        box_surface_truth(np,**x)


def test_vertical_riser_contact_near_top_height_is_not_load_bearing():
    # Root cause: the earlier algorithm accepted a foot near top height
    # and a box contact pair even if the actual solver normal was
    # HORIZONTAL, meaning it was pushing sideways into an edge/riser.
    x=fixture()
    x['contact_frame'][0,0,:3]=[1.,0.,0.]
    y=box_surface_truth(np,**x)
    assert not bool(y['foot_top_contact_candidate'][0,0])
    assert bool(y['foot_riser_contact'][0,0])
    assert y['support_fraction'][0]==0.

    # Reversing contact ordering can flip the normal; a genuine top
    # contact remains top regardless of which geom is labelled geom1.
    x=fixture()
    x['contact_frame'][0,0,:3]=[0.,0.,-1.]
    y=box_surface_truth(np,**x)
    assert bool(y['foot_top_contact_candidate'][0,0])


def test_horizontal_normal_alone_does_not_prove_contact_at_top_face():
    x=fixture()
    # Contact normal points vertical but the actual contact point is
    # buried on a different face. Pair-only+height-only would mislabel.
    x['contact_position'][0,0,2]=0.005
    y=box_surface_truth(np,**x)
    assert not bool(y['foot_top_contact_candidate'][0,0])
    assert bool(y['foot_riser_contact'][0,0])

    # JAX also rejects the fake positive without changing the actor.
    import jax
    import jax.numpy as jnp
    args={k:jnp.asarray(v) if isinstance(v,np.ndarray) else v
          for k,v in x.items()}
    jit_result=jax.jit(lambda:box_surface_truth(jnp,**args))()
    assert not bool(np.asarray(jit_result['foot_top_contact_candidate'])[0,0])
    assert bool(np.asarray(jit_result['foot_riser_contact'])[0,0])


def test_close_but_positive_contact_gap_is_not_stance_support():
    # Another old false-positive: MuJoCo's collision candidate buffer can
    # report a pair within +2mm before it is actually touching.
    # With zero geom margin/gap, positive signed separation has no
    # load-bearing contact force. It must not earn support/slip credit.
    x=fixture()
    x['contact_distance'][0,0]=+.001
    x['contact_distance'][0,1]=+.001
    x['contact_distance'][1,0]=+.001
    y=box_surface_truth(np,**x)
    assert not bool(np.any(y['foot_supported_candidate']))
    assert y['support_fraction'].tolist()==[0.,0.]
    assert not bool(np.any(y['foot_top_contact_candidate']))
    assert not bool(np.any(y['foot_floor_contact_candidate']))
    # Exact geometric touching may still be a candidate (NOT force).
    x=fixture()
    assert bool(box_surface_truth(np,**x)['foot_top_contact_candidate'][0,0])


def test_torso_over_box_does_not_reduce_body_height_without_top_support():
    x=fixture()
    # Put both feet genuinely on the flat FLOOR outside obstacle while the
    # TORSO remains directly above the physical 20mm box footprint.
    x['foot_xyz'][0,:,:]=[[.10,-.12,.018],[.30,.12,.018]]
    x['contact_geom1'][0,:2]=0
    x['contact_geom2'][0,:2]=[3,4]
    x['contact_position'][0,:2,:]=[[.10,-.12,0.],[.30,.12,0.]]
    x['contact_frame'][0,:2,:3]=[0.,0.,1.]
    y=box_surface_truth(np,**x)
    assert bool(y['torso_over_box_footprint'][0])
    assert float(y['support_fraction'][0])==1.
    assert not bool(y['foot_top_contact_candidate'][0].any())
    np.testing.assert_allclose(y['support_reference_elevation_m'][0],0.)
    np.testing.assert_allclose(y['torso_height'][0],.27,atol=2e-6)


def test_torso_past_box_still_respects_supporting_foot_and_mixed_stance():
    x=fixture()
    # The RIGHT sphere is still supported atop the box even though its
    # torso has translated beyond the 40mm footprint. No reward jump.
    x['torso_xyz'][0,:2]=[.45,0.]
    y=box_surface_truth(np,**x)
    assert not bool(y['torso_over_box_footprint'][0])
    np.testing.assert_allclose(
        y['support_reference_elevation_m'][0],.020,atol=2e-6)
    np.testing.assert_allclose(y['torso_height'][0],.25,atol=2e-6)

    # One true foot on box + one supporting foot on level floor is a
    # transparent 50/50 height reference, not an XY torso switch.
    x['contact_geom1'][0,1]=0
    x['foot_xyz'][0,1,:]=[.10,.12,.018]
    x['contact_position'][0,1,:]=[.10,.12,0.]
    x['contact_frame'][0,1,:3]=[0.,0.,1.]
    y=box_surface_truth(np,**x)
    assert float(y['support_fraction'][0])==1.
    np.testing.assert_allclose(
        y['support_reference_elevation_m'][0],.010,atol=2e-6)
    np.testing.assert_allclose(y['torso_height'][0],.26,atol=2e-6)
