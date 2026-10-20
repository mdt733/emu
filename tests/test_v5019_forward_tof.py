"""Focused sensor acceptance: real ray geometry, schema and 2-world JAX wiring.

These are code contracts only; they do NOT certify a physical ToF FOV, real
footstep crossing, viewer rendering, or learned terrain skill.
"""
import numpy as np
import pytest

from emu.forward_tof import (
    first_hit_ranges, relative_height_profile,
    FORWARD_MOUNTS_BODY_M, FORWARD_DIRECTIONS_BODY)
from emu.contracts import (
    OBS_SCHEMA_SENSOR, OBS_SCHEMA_PREVIEW, OBS_DIM, SENSOR_OBS_DIM, PREVIEW_OBS_DIM,
    SENSOR_OBS_FIELDS, PREVIEW_OBS_FIELDS, sensor_observation_contract)
from emu.training_config import TrainingConfig, training_sensor_contract


def rays(x, *, height=.0, rotated=False):
    n = len(x)
    torso = np.column_stack((np.asarray(x), np.zeros(n), np.full(n,.2584)))
    rotation = np.tile(np.eye(3), (n,1,1))
    floor = np.zeros((n,3))
    normal = np.tile(np.asarray((0.,0.,1.)), (n,1))
    if height:
        box = np.tile(np.asarray((.2, 0., height/2)), (n,1))
        br = np.tile(np.eye(3), (n,1,1))
        if rotated:
            t = np.deg2rad(20.)
            br[:] = [[np.cos(t),-np.sin(t),0.],
                     [np.sin(t),np.cos(t),0.],[0.,0.,1.]]
        return first_hit_ranges(
            np, torso, rotation, floor, normal, box_center=box,
            box_rotation=br, box_half_size=np.asarray((.04,.20,height/2)))
    return first_hit_ranges(np, torso, rotation, floor, normal)


def test_measured_six_beam_shapes_and_far_obstacle_is_not_oracle():
    assert len(FORWARD_MOUNTS_BODY_M) == 6
    assert len(FORWARD_DIRECTIONS_BODY) == 6
    flat, valid = rays([0.,.04])
    assert flat.shape == valid.shape == (2,6)
    assert valid.all()
    far, far_valid = rays([0.,.04], height=.05)
    assert far_valid.all()
    assert np.allclose(far[0], flat[0], atol=1e-5)
    # World 2 has stepped forward; now the long beam strikes a 50mm top.
    assert np.any(far[1] < flat[1] - .015), (far, flat)


def test_height_is_inverted_from_ranges_not_known_box_height():
    from emu.directional import fit_tripod_ground
    flat, _ = rays([.04])
    raised, valid = rays([.04], height=.05)
    tripod = np.full((1,3), .2334)
    fit = fit_tripod_ground(np, tripod, np.asarray([[0.,0.,-1.]]))
    assert fit["valid"].all()
    flat_z, flat_valid = relative_height_profile(
        np, flat, np.ones((1,6),bool), tripod,
        fit["ground_normal_body"], fit["valid"])
    z, good = relative_height_profile(
        np, raised, valid, tripod,
        fit["ground_normal_body"], fit["valid"])
    assert flat_valid.all() and good.all()
    assert np.max(np.abs(flat_z)) < 1e-4
    assert z[0,2] > .025  # forward-right 15cm sees elevated box
    assert z[0,5] > .025  # forward-left 15cm sees elevated box


def test_invalid_ground_returns_unknown_not_flat():
    flat, valid = rays([0.])
    tripod = np.full((1,3), .2334)
    h, conf = relative_height_profile(
        np, flat, valid, tripod,
        np.asarray([[0.,0.,1.]]), np.asarray([False]))
    assert np.all(h == 0.) and np.all(conf == 0.)
    flat[:,3] = 0
    valid[:,3] = False
    h, conf = relative_height_profile(
        np, flat, valid, tripod,
        np.asarray([[0.,0.,1.]]), np.asarray([True]))
    assert conf[0,3] == 0 and h[0,3] == 0


def test_historical_232_manifest_unchanged_and_new_280_distinct():
    old = sensor_observation_contract()
    assert old["schema"] == OBS_SCHEMA_SENSOR
    assert old["base_fields"] == list(SENSOR_OBS_FIELDS)
    assert "forward_model_revision" not in old
    assert OBS_DIM == 220 and SENSOR_OBS_DIM == 232 and PREVIEW_OBS_DIM == 280
    assert len(PREVIEW_OBS_FIELDS) == 70
    assert all(x.startswith("forward_range_") for x in PREVIEW_OBS_FIELDS[58:64])
    assert all(x.startswith("forward_valid_") for x in PREVIEW_OBS_FIELDS[64:70])
    visible = sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW)
    masked = sensor_observation_contract(schema=OBS_SCHEMA_PREVIEW, preview_masked=True)
    assert visible["sha256"] != masked["sha256"] != old["sha256"]
    assert visible["forward_masked"] is False
    assert len(visible["forward_beams_body"]) == 6
    assert training_sensor_contract(
        TrainingConfig(observation_schema=OBS_SCHEMA_PREVIEW)) == visible
    assert training_sensor_contract(
        TrainingConfig(observation_schema=OBS_SCHEMA_PREVIEW,
                       forward_preview_masked=True)) == masked


def test_jax_six_ray_shape_and_two_world_actor_packets():
    jax = pytest.importorskip("jax")
    mujoco = pytest.importorskip("mujoco")
    import jax.numpy as jnp
    from emu.terrain import adaptive_height_course
    from emu.environment import EmuMJXEnv
    from emu.contracts import ROBOT_XML
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as d:
        xml = adaptive_height_course(ROBOT_XML, Path(d)/"preview.xml")
        env = EmuMJXEnv(str(xml), num_envs=2, seed=733,
                        observation_schema=OBS_SCHEMA_PREVIEW,
                        terrain_contact_mode="box_surface", terrain_adaptive=True,
                        tof_dropout_prob=0.)
        params = env.params_array({"frontier_terrain_height_m":.05})
        st, obs = env.reset(params)
        assert obs.shape == (2,280)
        # A visible preview must actually fill the input slots before PPO
        # starts; shape-only acceptance would miss twelve always-zero inputs.
        obs_host = np.asarray(jax.device_get(obs))
        assert np.any(obs_host[:, -12:-6] > 0.)
        assert np.any(obs_host[:, -6:] > 0.)
        out = env.step(st, jnp.zeros((2,6),dtype=jnp.float32), params)
        assert out.obs.shape == (2,280)
        assert np.asarray(jax.device_get(out.obs)).shape == (2,280)
        assert np.isfinite(np.asarray(jax.device_get(out.obs))).all()
        # Exact same geometry, acquisition/latency and 280 width: only
        # preview values are masked in the control arm.
        masked = EmuMJXEnv(str(xml), num_envs=2, seed=733,
                            observation_schema=OBS_SCHEMA_PREVIEW,
                            terrain_contact_mode="box_surface", terrain_adaptive=True,
                            forward_preview_masked=True, tof_dropout_prob=0.)
        _, obs2 = masked.reset(params)
        assert obs2.shape == (2,280)
        assert np.all(np.asarray(jax.device_get(obs2))[:, -12:] == 0.)
