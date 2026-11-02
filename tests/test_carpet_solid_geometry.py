"""Independent trainer geometry exam for nine REAL solid collider tiles.

The global 2-triangle height samples are shared with every specialist
and are preserved as teacher/sensor values. Collision is 9 rigid planar
box tops, each tilted to those samples: true top support and true sides.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import mujoco

from emu.terrain_carpet_mjx import (
    TILE_COUNT,TILE_M,TILE_SIZE_Z_M,SOLID_HALF_Z_M,
    SOLID_GEOM_NAMES,SOLID_BODY_NAMES,
    add_carpet_geoms,make_profiles,generate_window,solid_box_poses,
)


def shapes(seed=735,slope=8.,up=.03,down=.05,center=(0,0)):
    seeds=jnp.asarray([seed],dtype=jnp.int32)
    u=jnp.asarray([up],dtype=jnp.float32)
    d=jnp.asarray([down],dtype=jnp.float32)
    px,py=make_profiles(jnp,seeds,u,d)
    cx=jnp.asarray([center[0]],dtype=jnp.int32)
    cy=jnp.asarray([center[1]],dtype=jnp.int32)
    cent,verts,grid=generate_window(
        jnp,cx,cy,seeds,px,py,u,d,jnp.asarray([slope]),
        transition_limit_deg=3.)
    return tuple(map(np.asarray,(cent,verts,grid,*solid_box_poses(jnp,cent,verts))))


def test_nine_real_box_collision_tiles_and_disabled_hfield_side_skirts():
    m=mujoco.MjModel.from_xml_string(
        add_carpet_geoms('<mujoco><worldbody/></mujoco>'))
    assert m.nhfield==TILE_COUNT
    for i in range(TILE_COUNT):
        box=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,SOLID_GEOM_NAMES[i])
        assert m.geom_type[box]==mujoco.mjtGeom.mjGEOM_BOX
        assert m.geom_contype[box]&4
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,SOLID_BODY_NAMES[i])
        assert m.body_mocapid[bid]>=0
        hf=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,
                              f'carpet_tile_geom_{i}')
        assert m.geom_contype[hf]==0
        assert m.geom_conaffinity[hf]==0


def test_generated_real_box_top_is_close_to_all_four_triangle_corners():
    cent,verts,_,solid_pos,quats=shapes(seed=735,slope=12.,up=.035,down=.09)
    assert cent.shape==(1,9,3)
    assert verts.shape==(1,36)
    assert solid_pos.shape==(1,9,3)
    assert quats.shape==(1,9,4)
    assert np.all(np.isfinite(solid_pos))
    assert np.allclose(np.linalg.norm(quats,axis=2),1.,atol=1e-5)
    v=verts.reshape((1,9,4))
    a,b,c,d=[v[:,:,i] for i in range(4)]
    mean=(a+b+c+d)*(.25*TILE_SIZE_Z_M)+cent[:,:,2]
    gx=.5*(b+d-a-c)*TILE_SIZE_Z_M/TILE_M
    gy=.5*(c+d-a-b)*TILE_SIZE_Z_M/TILE_M
    xyz=np.stack((gx,gy,np.ones_like(gx)),axis=2)
    normal=np.stack((-gx,-gy,np.ones_like(gx)),axis=2)
    normal=normal/np.linalg.norm(normal,axis=2,keepdims=True)
    assert np.allclose(solid_pos+SOLID_HALF_Z_M*normal,
                       np.stack((cent[:,:,0],cent[:,:,1],mean),axis=2),
                       atol=1e-6)
    prediction=np.stack((
        mean-.25*gx-.25*gy,mean+.25*gx-.25*gy,
        mean-.25*gx+.25*gy,mean+.25*gx+.25*gy),axis=2)
    true=cent[:,:,2,None]+TILE_SIZE_Z_M*v
    # Proves sensor surface and collision top cannot silently diverge,
    # even though it is a coplanar quad rather than a bent triangle pair.
    assert np.max(np.abs(prediction-true))<.008,(
        'Tile collision/sensor mismatch',np.max(np.abs(prediction-true)))


def test_flat_standing_uses_same_box_top_without_spurious_risers():
    _,verts,_,solid_pos,quat=shapes(seed=735,slope=0.,up=0.,down=0.)
    assert np.max(verts)-np.min(verts)<1e-6
    assert np.allclose(quat[:,:,:],[1,0,0,0],atol=1e-6)
    assert np.allclose(solid_pos[:,:,2],-SOLID_HALF_Z_M,atol=1e-6)


def test_global_revisit_reproduces_box_collision_identically():
    a=shapes(seed=761,center=(11,-9))
    _=shapes(seed=761,center=(30,17))
    b=shapes(seed=761,center=(11,-9))
    for x,y in zip(a,b):
        assert np.array_equal(x,y)


def test_row_shift_keeps_same_physical_slot_ids_without_recompiling():
    a=shapes(seed=735,center=(0,0))
    b=shapes(seed=735,center=(1,0))
    _,_,idx0,p0,q0=a
    _,_,idx1,p1,q1=b
    assert (idx0==idx1).all(axis=2).sum()==6
    kept=(idx0==idx1).all(axis=2)[0]
    assert np.array_equal(p0[0,kept],p1[0,kept])
    assert np.array_equal(q0[0,kept],q1[0,kept])
    assert (~kept).sum()==3
