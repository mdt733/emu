"""No reward or view-only geometry: verify all 12 actual MJX riser boxes.

Each face belongs to one of 12 stable slots inside the live 3x3 carpet.
Only discarded seam slots should move when the window shifts by one tile.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import mujoco

from emu.terrain_carpet_mjx import (
    add_carpet_geoms, WALL_COUNT, TILE_COUNT,
    WALL_BODY_NAMES, WALL_GEOM_NAMES, make_profiles,
    generate_window, generate_wall_window, wall_slot_indices)


def data_for(seed=735, slope=6., up=.03, down=.05, center=(0,0)):
    s=jnp.asarray([seed],dtype=jnp.int32)
    u=jnp.asarray([up],dtype=jnp.float32)
    d=jnp.asarray([down],dtype=jnp.float32)
    xp,yp=make_profiles(jnp,s,u,d)
    cx=jnp.asarray([center[0]],dtype=jnp.int32)
    cy=jnp.asarray([center[1]],dtype=jnp.int32)
    p,h,_=generate_window(jnp,cx,cy,s,xp,yp,u,d,
                          jnp.asarray([slope]),transition_limit_deg=3.)
    walls,quats,indices,rises=generate_wall_window(jnp,cx,cy,p,h)
    return np.asarray(walls),np.asarray(quats),np.asarray(indices),np.asarray(rises)


def test_compiles_exactly_twelve_true_movable_box_colliders():
    xml='<mujoco><worldbody/></mujoco>'
    m=mujoco.MjModel.from_xml_string(add_carpet_geoms(xml))
    assert m.nhfield==TILE_COUNT
    for i,n in enumerate(WALL_GEOM_NAMES):
        g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)
        assert g>=0 and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
        assert m.geom_contype[g]&4
        body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,WALL_BODY_NAMES[i])
        assert m.body_mocapid[body]>=0


def test_all_flat_tile_seams_have_no_ghost_vertical_box():
    w,q,i,r=data_for(slope=8.,up=0.,down=0.)
    assert w.shape==(1,WALL_COUNT,3)
    assert np.max(np.abs(r))<.0015
    assert np.all(w[0,:,2]<-25.),w
    assert np.allclose(q[0,:,0],1.,atol=1e-7)


def test_walls_follow_real_lips_and_sloped_supports():
    w,q,idx,r=data_for(seed=735,slope=8,up=.035,down=.045)
    assert idx.shape==(1,WALL_COUNT,2)
    assert w.shape==(1,WALL_COUNT,3)
    assert q.shape==(1,WALL_COUNT,4)
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(q))
    assert np.allclose(np.linalg.norm(q,axis=2),1.,atol=1e-5)
    assert np.any(np.abs(r)>.0015),(idx,r)
    live=(np.abs(r[0])>=.0015)
    assert np.all(w[0,~live,2]<-25.)
    assert np.all(w[0,live,2]>-5.)
    assert np.max(np.abs(r))<.09,r
    # Every nonburied physical wall has one of the boundary world coords.
    for k in range(WALL_COUNT):
        i,j=idx[0,k]
        if not live[k]:continue
        if k<6:
            assert abs(w[0,k,0] - i*.5)<.007
            assert abs(w[0,k,1] - (j+.5)*.5)<1e-6
        else:
            assert abs(w[0,k,0]-(i+.5)*.5)<1e-6
            assert abs(w[0,k,1]-j*.5)<.007


def test_12_wall_slots_reuse_in_place_when_center_changes_one_cell():
    old=wall_slot_indices(jnp,jnp.asarray([0]),jnp.asarray([0]))
    new=wall_slot_indices(jnp,jnp.asarray([1]),jnp.asarray([0]))
    o=np.stack([np.asarray(x)[0] for x in old],axis=1)
    n=np.stack([np.asarray(x)[0] for x in new],axis=1)
    stable=(o==n).all(axis=1)
    assert stable.sum()==7,(o,n,stable)
    assert (~stable).sum()==5
    assert len({tuple(k) for k in o})<WALL_COUNT # x and y seams can share ints
    # The (slot, orientation) pairing is unique and deterministic.
    unique=[(k<6,*map(int,o[k])) for k in range(WALL_COUNT)]
    assert len(set(unique))==WALL_COUNT


def test_global_world_revisits_regenerate_same_real_wall_geometry():
    before=data_for(seed=761,center=(5,-9))
    _=data_for(seed=761,center=(-10,11))
    after=data_for(seed=761,center=(5,-9))
    for a,b in zip(before,after):
        assert np.array_equal(a,b)
