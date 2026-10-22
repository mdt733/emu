"""v5022 real-physics spike: MJX BOX contacts, ray parity and batch support.

This is deliberately NOT part of the training scene yet. It must pass
before anyone introduces step terrain into the full EMU environment.
"""
import numpy as np
import pytest

from emu.forward_sensing import scan_box_course


def physical_scene(upper_top):
    """Two adjacent solid support volumes; no invisible infinite floor."""
    # Left surface Z=0, right surface Z=upper_top; edge at X=.16.
    # Every box extends down to Z=-.40 and has finite x/y support.
    r_half = (upper_top + .40) / 2
    r_z = (upper_top - .40) / 2
    return f"""<mujoco>
      <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
      <default><geom friction="1.5 0.005 0.0001"/></default>
      <asset><material name="opaque_terrain" rgba="0.55 0.52 0.48 1"/></asset>
      <worldbody>
        <geom name="approach" material="opaque_terrain" type="box" group="1"
              pos="-0.42 0 -0.20" size=".58 .4 .20"/>
        <geom name="landing" material="opaque_terrain" type="box" group="1"
              pos=".58 0 {r_z:.8f}" size=".42 .4 {r_half:.8f}"/>
        <body name="contact_probe" pos="0 0 .16">
          <freejoint/>
          <geom type="sphere" size=".018" mass=".01"/>
        </body>
      </worldbody>
    </mujoco>"""


@pytest.mark.parametrize("upper_top", [0.05, -0.05])
def test_mjx_rays_see_actual_stepped_geoms(upper_top):
    """A ray must hit the same exposed solid support surfaces MJX knows."""
    mujoco = pytest.importorskip("mujoco")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx
    from emu.forward_sensing import forward_ray_geometry

    cpu = mujoco.MjModel.from_xml_string(physical_scene(upper_top))
    m = mjx.put_model(cpu)
    d = mjx.forward(m, mjx.make_data(m))

    torso = np.array([[0., 0., .2584]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)[None, :, :]
    o, beams = forward_ray_geometry(jnp, jnp.asarray(torso), jnp.asarray(R))
    # Filter to group 1 (terrain), excluding the dynamic sphere in group 0.
    def ray_one(point, vec):
        distance, geom_id = mjx.ray(
            m, d, point, vec,
            geomgroup=(0, 1, 0, 0, 0, 0), flg_static=True)
        return distance, geom_id

    distances, geom_ids = jax.jit(jax.vmap(ray_one))(
        o[0], beams[0])
    hits = np.asarray(distances)
    assert np.isfinite(hits).all() and (hits > 0).all()
    assert np.all(np.asarray(geom_ids) >= 0)

    bounds_min = np.array([[-1., -.4, -.4], [.16, -.4, -.4]])
    bounds_max = np.array([[.16, .4, 0.], [1., .4, upper_top]])
    reference, good = scan_box_course(np, torso, R, bounds_min, bounds_max)
    assert good.all()
    np.testing.assert_allclose(hits, reference[0], atol=0.003)


@pytest.mark.parametrize("upper_top", [0.05, -0.05])
def test_two_mjx_worlds_really_support_different_elevations(upper_top):
    """No reward or proximity proxy: two balls settle ON real solid surfaces."""
    mujoco = pytest.importorskip("mujoco")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx

    cpu = mujoco.MjModel.from_xml_string(physical_scene(upper_top))
    m = mjx.put_model(cpu)
    first = mjx.make_data(m)
    batched = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), first)
    qpos = batched.qpos
    qpos = qpos.at[0, :3].set(jnp.array([-.35, 0., .12]))
    qpos = qpos.at[1, :3].set(jnp.array([.35, 0., upper_top + .12]))
    batched = batched.replace(qpos=qpos)
    forward_two = jax.vmap(mjx.forward, in_axes=(None, 0))
    step_two = jax.vmap(mjx.step, in_axes=(None, 0))

    @jax.jit
    def settle(d):
        d = forward_two(m, d)
        return jax.lax.fori_loop(0, 140, lambda _, s: step_two(m, s), d)

    result = settle(batched)
    actual = np.asarray(result.qpos[:, 2])
    expected = np.array([.018, upper_top + .018])
    assert np.isfinite(actual).all(), actual
    np.testing.assert_allclose(actual, expected, atol=.022)
    np.testing.assert_allclose(actual[1] - actual[0], upper_top, atol=.025)



@pytest.mark.parametrize("upper_top", [0.05, -0.05])
def test_down_tripod_observes_actual_split_surface_not_old_floor_plane(upper_top):
    """Front and rear modules can straddle a *true* discontinuity."""
    mujoco = pytest.importorskip("mujoco")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx
    from emu.forward_sensing import scan_tripod_oriented_boxes
    cpu = mujoco.MjModel.from_xml_string(physical_scene(upper_top))
    m = mjx.put_model(cpu)
    d = mjx.forward(m, mjx.make_data(m))
    ids = np.asarray([
        mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_GEOM, "approach"),
        mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_GEOM, "landing")])
    assert (ids >= 0).all()
    t = jnp.asarray([[.16,0.,.2584]],dtype=jnp.float32)
    R = jnp.eye(3,dtype=jnp.float32)[None]
    ranges, good = jax.jit(lambda pos, rot: scan_tripod_oriented_boxes(
        jnp, pos, rot, d.geom_xpos[ids], d.geom_xmat[ids],
        m.geom_size[ids]))(t, R)
    value = np.asarray(ranges)[0]
    assert bool(np.asarray(good).all())
    # Two front ToFs are over the landing, rear ToF over approach ground.
    assert value[0] == pytest.approx(.2584-.025-upper_top, abs=.003)
    assert value[1] == pytest.approx(.2584-.025-upper_top, abs=.003)
    assert value[2] == pytest.approx(.2584-.025, abs=.003)
    assert value[2] - value[0] == pytest.approx(upper_top, abs=.003)
