"""Adaptive step-height v5018: offline math + CPU collision/mocap smoke."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from emu.terrain import (
    adaptive_height_course, adaptive_height_sample,
    next_height_frontier, TERRAIN_FRONTIER_MAX_M,
    TERRAIN_BOX_HALF_HEIGHT_M,
)
from emu.contracts import ROBOT_XML
from emu.training_config import TrainingConfig


def test_continuous_birth_height_has_flat_interior_and_edge():
    u = np.array([.01, .25, .40, .7, .99])
    selected = np.array([0., .1, .3, .8, .99])
    heights = adaptive_height_sample(np, .08, u, selected, exposure=.6, edge_fraction=.35)
    assert 0 < heights[0] <= .08
    assert 0 < heights[1] <= .08
    assert heights[2] > 0
    assert heights[-1] == 0
    assert all(h <= .08 for h in heights)
    assert len(set(np.round(heights[:4], 5))) >= 3


def test_physical_frontier_never_advances_from_no_contact_or_reward():
    evidence = dict(terrain_edge_exposure=.20, terrain_edge_done_count=12,
                    terrain_edge_crossing=.6,
                    terrain_edge_fall_rate=.08, terrain_edge_contact=.01)
    upper = .08
    new, why = next_height_frontier(upper, evidence, safety_m=.15, growth_m=.001)
    assert new == pytest.approx(.081) and why == 'expand_measured_crossing'
    for bad in (
        {**evidence, 'terrain_edge_contact': 0.},
        {**evidence, 'terrain_edge_exposure': 0.},
        {**evidence, 'terrain_edge_crossing': 0.},
        {**evidence, 'terrain_edge_fall_rate': -1.},
    ):
        newer, name = next_height_frontier(.08, bad, safety_m=.15)
        assert newer <= .08 and name != 'expand_measured_crossing'
    assert next_height_frontier(.08, {**evidence, 'terrain_edge_fall_rate':.8},
                                safety_m=.15)[0] < .08
    assert next_height_frontier(.15, evidence, safety_m=.15)[0] == .15
    assert TERRAIN_FRONTIER_MAX_M == .15


def test_reproducible_adaptive_course_is_immutable(tmp_path):
    path = adaptive_height_course(ROBOT_XML, tmp_path / 'course.xml')
    original = path.read_bytes()
    tree = ET.parse(path)
    body = tree.find('.//body[@name="v5018_course_mocap"]')
    assert body is not None and body.attrib['mocap'] == 'true'
    box = body.find('geom')
    assert box is not None and box.attrib['name'] == 'v5018_obstacle_step'
    assert float(box.attrib['size'].split()[2]) == TERRAIN_BOX_HALF_HEIGHT_M
    assert adaptive_height_course(ROBOT_XML, path).read_bytes() == original
    path.write_text('bad')
    with pytest.raises(RuntimeError, match='differs'):
        adaptive_height_course(ROBOT_XML, path)


def test_adaptive_cfg_is_explicit_and_off_by_default():
    c = TrainingConfig()
    assert c.terrain_height_safety_m == 0
    assert c.terrain_height_initial_m > 0
    assert c.terrain_box_height_mm == -1


def test_real_mujoco_mocap_body_can_change_box_top(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    path = adaptive_height_course(ROBOT_XML, tmp_path / 'cpu.xml')
    model = mujoco.MjModel.from_xml_path(str(path))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'v5018_obstacle_step')
    body_id = int(model.geom_bodyid[geom_id])
    m = int(model.body_mocapid[body_id])
    assert m >= 0
    data = mujoco.MjData(model)
    for height in (0.003, .027, .08, .15):
        data.mocap_pos[m] = [.20, 0., height]
        mujoco.mj_forward(model, data)
        # World-body box centre is 75mm below its mocap top datum.
        assert float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2]) == pytest.approx(height, abs=1e-6)


def test_two_world_mjx_birth_positions_are_independent(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('jax')
    from emu.environment import EmuMJXEnv, T
    import jax
    import jax.numpy as jnp
    path = adaptive_height_course(ROBOT_XML, tmp_path / 'gpu_course.xml')
    env = EmuMJXEnv(str(path), num_envs=2, seed=733,
                    terrain_contact_mode='box_surface', terrain_adaptive=True,
                    terrain_height_exposure=.95, terrain_height_edge_fraction=.5)
    params = env.params_array({'frontier_terrain_height_m':.08})
    state = env.reset_jit(env.key0, params)
    h = np.asarray(jax.device_get(state.task[:, T.terrain_height_m]))
    pos = np.asarray(jax.device_get(state.data.mocap_pos[:, env.terrain_mocap_id, :]))
    assert np.allclose(pos[:, 2], h)
    assert np.allclose(pos[:, 0], np.where(h > 0., .20, 5.), rtol=0., atol=1e-6)
    gtop = np.asarray(jax.device_get(state.data.geom_xpos[:, env.terrain_box_geom_id, 2]))
    gtop = gtop + TERRAIN_BOX_HALF_HEIGHT_M
    assert np.allclose(gtop, h, atol=1e-5)
    assert (h <= .08 + 1e-6).all()
    next_state = env.step_jit(state, jnp.zeros((2, 6), jnp.float32), params)
    assert np.isfinite(np.asarray(jax.device_get(next_state.obs))).all()
