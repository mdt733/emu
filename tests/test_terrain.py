"""v5022 terrain + 4x4 front ToF. One test suite; no prototype harnesses.

Physical MJX checks are CPU tests and cannot establish policy success.
"""
import math
import numpy as np
import pytest

from emu.terrain import (
    SENSOR_MODEL, SENSOR_RESOLUTION, SENSOR_FOV_DEG, SENSOR_MOUNT_BODY_M,
    course_surfaces, surface_box_geometry, camera_rays,
    ray_boxes, ray_spheres, scan_forward, scan_tripod,
)


def pose(x=0., worlds=1):
    return (np.tile([[x,0.,.2584]],(worlds,1)),
            np.tile(np.eye(3)[None],(worlds,1,1)))


def scene(step=0.):
    # Ground at z=0 before x=.16 and z=step thereafter.
    p = np.array([[-.42,0.,-.2],[.58,0.,(step-.4)/2]])
    sz = np.array([[.58,.4,.2],[.42,.4,(step+.4)/2]])
    rot = np.tile(np.eye(3)[None],(2,1,1))
    return p,rot,sz


def test_hill_is_continuous_and_steps_remain_distinct():
    surfaces, seams = course_surfaces(-.6,0.,[.3,.3,.3,.3],
                                      [0,12,0,-12],[0,0,0])
    assert all(not seam.discontinuous for seam in seams)
    for left,right in zip(surfaces,surfaces[1:]):
        assert left.z1_m == pytest.approx(right.z0_m)
    _, discontinuities = course_surfaces(-.5,0.,[.5,.5,.5],
                                        [0,0,0],[.05,-.07])
    assert [x.jump_m for x in discontinuities] == [.05,-.07]
    assert all(x.discontinuous for x in discontinuities)


def test_solid_rotated_box_top_is_the_requested_surface():
    segments, _ = course_surfaces(-.6,0.,[.4,.4,.4],[0,10,-10],[0,0])
    for surface in segments:
        p, half, q = surface_box_geometry(surface)
        a = surface.grade_rad
        top = np.array(p) + half[2]*np.array([-math.sin(a),0,math.cos(a)])
        tangent = np.array([math.cos(a),0,math.sin(a)])
        for sign,x in [(-1,surface.x0_m),(1,surface.x1_m)]:
            point = top+sign*half[0]*tangent
            assert point[0] == pytest.approx(x)
            assert point[2] == pytest.approx(surface.height_at(x))
        assert np.linalg.norm(q) == pytest.approx(1)


def test_sensor_is_4x4_65_diagonal_variant_and_one_optical_centre():
    p,R = pose(worlds=2)
    origins,rays = camera_rays(np,p,R)
    assert SENSOR_MODEL == "VL53L8CX"
    assert SENSOR_RESOLUTION == (4,4)
    assert SENSOR_FOV_DEG == (45.,45.)
    assert SENSOR_MOUNT_BODY_M[0] > .076
    assert origins.shape == rays.shape == (2,16,5,3)
    np.testing.assert_allclose(origins,np.broadcast_to(
        origins[:,0:1,0:1,:],origins.shape))
    np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1.,atol=1e-7)
    # Four rows look at progressively nearer patches.
    assert rays[0,0,0,2] > rays[0,15,0,2]


@pytest.mark.parametrize("height", [.05,-.05])
def test_front_scan_distinguishes_real_step_geometry(height):
    p,R = pose()
    baseline,good0 = scan_forward(np,p,R,*scene(0.))
    scan,good = scan_forward(np,p,R,*scene(height))
    assert baseline.shape == good.shape == (1,16)
    both = good & good0
    assert both.any()
    assert np.any(np.abs(scan[both]-baseline[both]) > .01)


def test_chassis_pitch_changes_raw_measurements():
    p,R = pose()
    theta = np.deg2rad(12)
    tilt = R.copy()
    tilt[0] = np.array([[np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])
    a,_ = scan_forward(np,p,R,*scene(.05))
    b,_ = scan_forward(np,p,tilt,*scene(.05))
    assert not np.allclose(a,b)


def test_foot_in_front_is_visible_and_never_privileged_filtered():
    p,R = pose()
    before,valid = scan_forward(np,p,R,*scene(0))
    origins,rays = camera_rays(np,p,R)
    # Put a physical 18mm-radius ball directly on central camera zone.
    foot = (origins[0,5,0] + .12*rays[0,5,0]).reshape(1,1,3)
    after,good = scan_forward(np,p,R,*scene(0),
                              foot_centres=foot,foot_radii=.018)
    assert (good & valid).any()
    assert np.any(after[good & valid] < before[good & valid] - .02)


def test_missing_readings_are_invalid_not_zero_height():
    p,R = pose()
    out,valid = scan_forward(np,p,R,*scene(.05),
                             dropout=np.ones((1,16),dtype=bool))
    assert not valid.any() and not out.any()


@pytest.mark.parametrize("height", [.05,-.05])
def test_down_tripod_sees_split_platform_height(height):
    p,R = pose(x=.16)
    readings,valid = scan_tripod(np,p,R,*scene(height))
    assert valid.all()
    np.testing.assert_allclose(readings[0,:2],.2584-.025-height,atol=.003)
    assert readings[0,2] == pytest.approx(.2584-.025,abs=.003)


def test_jax_jit_and_world_batch_match_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    p,R = pose(worlds=2)
    centres,rot,size = scene(.05)
    np_v,np_ok = scan_forward(np,p,R,centres,rot,size)
    jax_scan = jax.jit(lambda t,m:scan_forward(
        jnp,t,m,jnp.asarray(centres),jnp.asarray(rot),jnp.asarray(size)))
    jv,jo = jax_scan(jnp.asarray(p),jnp.asarray(R))
    np.testing.assert_allclose(np.asarray(jv),np_v,atol=1e-6)
    np.testing.assert_array_equal(np.asarray(jo),np_ok)


def physical_xml(height):
    r_half=(height+.4)/2
    r_z=(height-.4)/2
    return f"""<mujoco>
    <option timestep=".005" gravity="0 0 -9.81"/>
    <asset><material name="solid" rgba=".5 .6 .4 1"/></asset>
    <worldbody>
      <geom name="approach" type="box" group="1" material="solid"
            pos="-.42 0 -.2" size=".58 .4 .2"/>
      <geom name="landing" type="box" group="1" material="solid"
            pos=".58 0 {r_z:.8f}" size=".42 .4 {r_half:.8f}"/>
      <body name="contact_probe" pos="0 0 .16">
        <freejoint/><geom type="sphere" size=".018" mass=".01"/>
      </body>
    </worldbody></mujoco>"""


@pytest.mark.parametrize("height",[.05,-.05])
def test_mjx_rays_and_real_collision_share_step_solids(height):
    mujoco=pytest.importorskip("mujoco")
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx
    cpu=mujoco.MjModel.from_xml_string(physical_xml(height))
    m=mjx.put_model(cpu)
    d=mjx.forward(m,mjx.make_data(m))
    ids=np.asarray([mujoco.mj_name2id(cpu,mujoco.mjtObj.mjOBJ_GEOM,n)
                    for n in ("approach","landing")])
    assert np.all(ids >= 0)
    p,R=pose()
    o,v=camera_rays(jnp,jnp.asarray(p),jnp.asarray(R))
    orig=o.reshape(1,80,3)[:,[1,21,51,71],:]
    vec=v.reshape(1,80,3)[:,[1,21,51,71],:]
    analytic=ray_boxes(jnp,orig,vec,d.geom_xpos[ids],
                       d.geom_xmat[ids],m.geom_size[ids])
    def trace_one(point,direction):
        return mjx.ray(m,d,point,direction,
                       geomgroup=(0,1,0,0,0,0),flg_static=True)
    hit,geom=jax.jit(jax.vmap(trace_one))(orig[0],vec[0])
    assert np.all(np.asarray(geom) >= 0)
    np.testing.assert_allclose(np.asarray(analytic)[0],np.asarray(hit),
                               atol=.003)
    # 2 different physical contact heights, in *batched* MJX data.
    first=mjx.make_data(m)
    batch=jax.tree_util.tree_map(lambda a:jnp.stack((a,a)),first)
    q=batch.qpos.at[0,:3].set(jnp.array([-.35,0.,.12]))
    q=q.at[1,:3].set(jnp.array([.35,0.,height+.12]))
    batch=batch.replace(qpos=q)
    fwd=jax.vmap(mjx.forward,in_axes=(None,0))
    step=jax.vmap(mjx.step,in_axes=(None,0))
    @jax.jit
    def settle(state):
        state=fwd(m,state)
        return jax.lax.fori_loop(0,140,lambda _,v:step(m,v),state)
    result=settle(batch)
    actual=np.asarray(result.qpos[:,2])
    np.testing.assert_allclose(actual,[.018,height+.018],atol=.022)
    assert actual[1]-actual[0] == pytest.approx(height,abs=.025)



def test_real_mjx_hillcrest_contact_and_depth_ray_parity():
    """Slope transitions are collidable support, not tilted gravity."""
    mujoco=pytest.importorskip("mujoco")
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx

    slopes, _ = course_surfaces(-.8,0.,[.4,.4,.4,.4],
                                 [0.,12.,0.,-12.],[0.,0.,0.])
    xml=['<mujoco><option timestep=".005" gravity="0 0 -9.81"/>',
         '<asset><material name="solid" rgba=".5 .5 .5 1"/></asset>',
         '<worldbody>']
    for i,segment in enumerate(slopes):
        p,size,quat = surface_box_geometry(segment)
        fmt=lambda v:' '.join(f'{x:.9f}' for x in v)
        xml.append(f'<geom name="terrain_{i}" type="box" group="1" '
                   f'material="solid" pos="{fmt(p)}" '
                   f'size="{fmt(size)}" quat="{fmt(quat)}"/>')
    xml.append('<body name="probe" pos="0 0 .16"><freejoint/>'
               '<geom type="sphere" size=".018" mass=".01"/></body>'
               '</worldbody></mujoco>')
    cpu=mujoco.MjModel.from_xml_string(''.join(xml))
    m=mjx.put_model(cpu)
    d=mjx.forward(m,mjx.make_data(m))
    ids=np.asarray([mujoco.mj_name2id(cpu,mujoco.mjtObj.mjOBJ_GEOM,
                                      f'terrain_{i}') for i in range(4)])
    assert np.all(ids>=0)
    p,R=pose(x=-.48)
    o,v=camera_rays(jnp,jnp.asarray(p),jnp.asarray(R))
    chosen=(10,35,62)
    origin=o.reshape(1,80,3)[:,chosen,:]
    rays=v.reshape(1,80,3)[:,chosen,:]
    expected=ray_boxes(jnp,origin,rays,d.geom_xpos[ids],
                       d.geom_xmat[ids],m.geom_size[ids])
    def trace(point,direction):
        return mjx.ray(m,d,point,direction,
                       geomgroup=(0,1,0,0,0,0),flg_static=True)
    hit,which=jax.jit(jax.vmap(trace))(origin[0],rays[0])
    assert np.all(np.asarray(which)>=0)
    np.testing.assert_allclose(np.asarray(expected)[0],np.asarray(hit),
                               atol=.003)
