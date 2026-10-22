"""Hard physics gate: continuous crest vs discontinuous step in real MJX.

Small spheres are objective contact probes, NOT credit for EMU walking.
Compiled once with physically supporting box volumes and measured positions.
"""
import numpy as np
import pytest
from emu.terrain_profiles import course_surfaces, surface_box_geometry


def hill_xml(slopes):
    sections, seams = course_surfaces(-.8,0.,[.4,.4,.4,.4],slopes,[0.,0.,0.])
    result = [
        '<mujoco><option timestep="0.005" gravity="0 0 -9.81"/>',
        '<asset><material name="ground" rgba=".5 .6 .5 1"/></asset>',
        '<worldbody>',
    ]
    for i, s in enumerate(sections):
        p, size, q = surface_box_geometry(s, thickness_m=.12)
        def f(x):
            return f'{x:.9f}'
        result.append(
            '<geom name="surface_%d" type="box" group="1" material="ground" '
            'pos="%s" size="%s" quat="%s"/>' % (
                i, ' '.join(map(f,p)), ' '.join(map(f,size)),
                ' '.join(map(f,q))))
    result.append(
        '<body name="probe" pos="0 0 .18"><freejoint/>'
        '<geom type="sphere" size=".018" mass=".01"/></body>'
        '</worldbody></mujoco>')
    return ''.join(result), sections


def test_mjx_two_balls_rest_on_true_different_flat_surface_heights():
    mujoco = pytest.importorskip('mujoco')
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from mujoco import mjx
    xml, s = hill_xml([0.,10.,0.,-10.])
    model = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
    initial = mjx.make_data(model)
    batch = jax.tree_util.tree_map(
        lambda a: jnp.broadcast_to(a, (2,)+a.shape), initial)
    x0, x1 = -.6, .2
    target = np.array([s[0].height_at(x0), s[2].height_at(x1)])
    q = batch.qpos
    q = q.at[0, :3].set(jnp.array([x0,0.,target[0]+.12]))
    q = q.at[1, :3].set(jnp.array([x1,0.,target[1]+.12]))
    batch = batch.replace(qpos=q)
    forward = jax.vmap(mjx.forward, in_axes=(None,0))
    step = jax.vmap(mjx.step, in_axes=(None,0))
    @jax.jit
    def settle(state):
        state = forward(model, state)
        return jax.lax.fori_loop(
            0, 140, lambda _,st: step(model,st), state)
    final = settle(batch)
    z = np.asarray(final.qpos[:,2])
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z, target + .018, atol=.025)
    assert z[1] - z[0] > .045, (z, target)


def test_actual_mjx_rays_change_before_crest_contact():
    mujoco = pytest.importorskip('mujoco')
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from mujoco import mjx
    from emu.forward_sensing import multizone_ray_geometry
    result = []
    for slopes in ([0.,0.,0.,0.],[0.,10.,0.,-10.]):
        xml, _ = hill_xml(slopes)
        m = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
        d = mjx.forward(m, mjx.make_data(m))
        t = jnp.asarray([[-.48, 0., .2584]],dtype=jnp.float32)
        R = jnp.eye(3,dtype=jnp.float32)[None]
        origins, directions = multizone_ray_geometry(jnp, t, R)
        # Aim one zone's central sub-ray; material table is nonempty.
        ray = jax.jit(lambda p, v: mjx.ray(
            m, d, p, v, geomgroup=(0,1,0,0,0,0),flg_static=True))
        distance, geom = ray(origins[0, 10, 0], directions[0, 10, 0])
        assert int(geom) >= 0
        result.append(float(distance))
    assert np.isfinite(result).all()
    assert abs(result[0] - result[1]) > .006, result



def test_oriented_box_sensor_distance_matches_real_mjx_rays_on_crest():
    """Actual geom pose and size, not an independent sloped-floor oracle."""
    mujoco = pytest.importorskip('mujoco')
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from mujoco import mjx
    from emu.forward_sensing import (
        multizone_ray_geometry, ray_oriented_box_distances)

    xml, _ = hill_xml([0., 12., 0., -12.])
    cpu = mujoco.MjModel.from_xml_string(xml)
    m = mjx.put_model(cpu)
    d = mjx.forward(m, mjx.make_data(m))
    ids = np.array([
        mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_GEOM,
                          f'surface_{i}') for i in range(4)])
    assert np.all(ids >= 0)
    t = jnp.asarray([[-.48, 0., .2584]])
    R = jnp.eye(3)[None]
    origins, vectors = multizone_ray_geometry(jnp, t, R)
    p = origins.reshape(1,80,3)
    v = vectors.reshape(1,80,3)
    # Use four of the 80 rays, spanning different vertical scanning rows.
    pick = [1, 21, 51, 71]
    ref = []
    for i in pick:
        result, geom_id = jax.jit(lambda pnt, vec: mjx.ray(
            m, d, pnt, vec, geomgroup=(0,1,0,0,0,0),
            flg_static=True))(p[0,i], v[0,i])
        assert int(geom_id) in ids
        ref.append(float(result))

    actual = ray_oriented_box_distances(
        jnp, p[:,pick,:], v[:,pick,:],
        d.geom_xpos[ids], d.geom_xmat[ids], m.geom_size[ids])
    np.testing.assert_allclose(np.asarray(actual)[0], ref, atol=.003)
    from emu.forward_sensing import scan_multizone_oriented_boxes
    result = jax.jit(lambda torso, rot: scan_multizone_oriented_boxes(
        jnp, torso, rot, d.geom_xpos[ids],
        d.geom_xmat[ids], m.geom_size[ids]))(t, R)
    assert result['ranges_m'].shape == (1,4,4)
    assert result['valid'].shape == (1,4,4)
    assert bool(np.asarray(result['valid']).any())
    assert np.isfinite(np.asarray(result['ranges_m'])).all()
    assert np.all((np.asarray(result['visibility_confidence']) >= 0) &
                  (np.asarray(result['visibility_confidence']) <= 1))
