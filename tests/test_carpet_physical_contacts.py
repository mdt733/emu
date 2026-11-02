"""Independent real MuJoCo contact exam for v5041 two-triangle carpet tiles.

No model hints or JAX estimates: this test asks the CPU contact solver whether
the feet encounter actual physical support and a real uphill vertical riser.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from emu.terrain_carpet_mjx import add_carpet_geoms, TILE_COUNT, TILE_SIZE_Z_M


def make_probe_model():
    robot='''<mujoco model="carpet_collision_exam">
      <option timestep="0.005" gravity="0 0 0"/>
      <worldbody>
        <body name="left_foot" pos="-0.17 0 0.035">
          <freejoint/>
          <geom name="left_ball" type="sphere" size=".04" mass=".1"
                contype="3" conaffinity="7"/>
        </body>
        <body name="right_foot" pos=".17 0 .035">
          <freejoint/>
          <geom name="right_ball" type="sphere" size=".04" mass=".1"
                contype="3" conaffinity="7"/>
        </body>
        <body name="riser_probe" pos="-0.015 .09 .07">
          <freejoint/>
          <geom name="riser_ball" type="sphere" size=".020" mass=".1"
                contype="3" conaffinity="7"/>
        </body>
      </worldbody>
    </mujoco>'''
    model=mujoco.MjModel.from_xml_string(add_carpet_geoms(robot))
    model.hfield_data[:]=.5
    assert model.nhfield==TILE_COUNT
    return model


def get_geoms(model):
    return {mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i):i
            for i in range(model.ngeom)}


def solve_case(rise_m, *, backend='cpu'):
    model=make_probe_model()
    data=mujoco.MjData(model)
    bodyids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,
                              f"carpet_tile_body_{i}")
             for i in range(TILE_COUNT)]
    mocaps=[int(model.body_mocapid[i]) for i in bodyids]
    for mid in mocaps:
        data.mocap_pos[mid]=[0.,0.,-30.]
    data.mocap_pos[mocaps[0]]=[-.25,0.,-TILE_SIZE_Z_M/2]
    data.mocap_pos[mocaps[1]]=[+.25,0.,rise_m-TILE_SIZE_Z_M/2]
    # Original freejoint poses match XML defaults, then place each sphere
    # just 5mm inside the corresponding real *top* collision face.
    data.qpos[0:3]=[-.17,0,.035]
    data.qpos[7:10]=[+.17,0,rise_m+.035]
    data.qpos[14:17]=[-.015,.09,rise_m*.5]
    mujoco.mj_forward(model,data)
    if backend == 'mjx':
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        jmodel=mjx.put_model(model)
        jdata=mjx.make_data(jmodel).replace(
            qpos=jnp.asarray(data.qpos),
            mocap_pos=jnp.asarray(data.mocap_pos))
        jdata=jax.jit(lambda d:mjx.forward(jmodel,d))(jdata)
        g1=np.asarray(jax.device_get(jdata.contact.geom1))
        g2=np.asarray(jax.device_get(jdata.contact.geom2))
        distances=np.asarray(jax.device_get(jdata.contact.dist))
        frames=np.asarray(jax.device_get(jdata.contact.frame))
        positions=np.asarray(jax.device_get(jdata.contact.pos))
        contacts=[]
        for n in range(len(g1)):
            if min(int(g1[n]),int(g2[n]))<0 or not np.isfinite(distances[n]):
                continue
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g))
                   for g in (g1[n],g2[n])}
            contacts.append((names,float(distances[n]),
                             frames[n].copy(),positions[n].copy()))
        return contacts
    geoms=get_geoms(model)
    contacts=[]
    for n in range(data.ncon):
        c=data.contact[n]
        a,b=int(c.geom1),int(c.geom2)
        names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
               for g in (a,b)}
        contacts.append((names,float(c.dist),
                         np.asarray(c.frame[:3]).copy(),np.asarray(c.pos).copy()))
    return contacts


@pytest.mark.parametrize("rise_m",[.0,.02,.04,.15])
def test_two_tile_foot_support_is_real_contact(rise_m):
    contacts=solve_case(rise_m)
    for foot,tile in (("left_ball","carpet_tile_geom_0"),
                      ("right_ball","carpet_tile_geom_1")):
        matches=[x for x in contacts if {foot,tile}.issubset(x[0]) and
                 x[1]<0.]
        assert matches, (rise_m,foot,tile,contacts)


@pytest.mark.parametrize("rise_m",[.04,.15])
def test_vertical_riser_is_not_a_ghost_or_ramp(rise_m):
    """A sphere above the lower top but beside the higher tile must contact

    the uphill vertical side. If this fails the model is NOT a staircase:
    never launch expensive STEP training on it.
    """
    contacts=solve_case(rise_m)
    risers=[x for x in contacts if
            {"riser_ball","carpet_tile_geom_1"}.issubset(x[0])]
    assert risers, (
        f"NO vertical lip side contact at {rise_m*100:.0f}cm: "
        f"heightfields may be top-surface only; inspect collision topology")


@pytest.mark.parametrize("rise_m",[.04,.15])
def test_mjx_and_cpu_agree_on_supported_steps_and_vertical_wall(rise_m):
    """MJX is the actual production GPU collision solver, not CPU MuJoCo.

    A CPU-only side contact passing is insufficient: the two real feet and
    the uphill wall must all create nonpositive contact distances inside
    mjx.Data.contact too.
    """
    contacts=solve_case(rise_m,backend='mjx')
    for probe,terrain in (
        ('left_ball','carpet_tile_geom_0'),
        ('right_ball','carpet_tile_geom_1'),
        ('riser_ball','carpet_tile_geom_1'),
    ):
        matching=[c for c in contacts if {probe,terrain}.issubset(c[0])
                  and c[1]<=.001]
        assert matching,(rise_m,probe,terrain,contacts)


@pytest.mark.parametrize('backend',['cpu','mjx'])
@pytest.mark.parametrize('rise_m',[.04,.15])
def test_riser_contact_normal_is_horizontal_not_false_upper_support(
        rise_m,backend):
    contacts=solve_case(rise_m,backend=backend)
    risers=[c for c in contacts if
            {'riser_ball','carpet_tile_geom_1'}.issubset(c[0]) and
            c[1]<=.003]
    assert risers,(rise_m,backend,contacts)
    normals=[np.asarray(c[2])[0] if np.asarray(c[2]).ndim==2
             else np.asarray(c[2])[:3] for c in risers]
    assert any(abs(n[2])<.50 for n in normals),(rise_m,backend,normals)


@pytest.mark.parametrize('backend',['cpu','mjx'])
@pytest.mark.parametrize('rise_m',[.02,.04,.15])
def test_foot_top_contacts_are_upward_facing(backend,rise_m):
    contacts=solve_case(rise_m,backend=backend)
    for ball,plate in (('left_ball','carpet_tile_geom_0'),
                       ('right_ball','carpet_tile_geom_1')):
        candidates=[c for c in contacts if
            {ball,plate}.issubset(c[0]) and c[1]<=.003]
        assert candidates,(backend,rise_m,ball,plate)
        normals=[np.asarray(c[2])[0] if np.asarray(c[2]).ndim==2
                 else np.asarray(c[2])[:3] for c in candidates]
        assert any(abs(n[2])>=.60 for n in normals),(backend,rise_m,normals)
