"""v5018 P1 PPO step-course wiring: immutable physics, opt-in and no actor leaks.

Fast tests intentionally do not claim CUDA PPO learning or supported stepping.
The full human-WSL pilot is documented in docs/V5018_TERRAIN_PLAN.md.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from emu.contracts import OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR, ROBOT_XML
from emu.branch_lab import one_run_config
from emu.terrain import collidable_course
from emu.training_config import TrainingConfig


def _args(tmp_path, *, height, profile="sim2real", sensor=True,
          slope=0., terrain_radius=0.):
    return SimpleNamespace(
        run_dir=str(tmp_path / "pilot"), steps=256, envs=8,
        seed=733, device="cuda", profile=profile,
        pitch_deg=None, motor_guidance_weight=0.,
        contact_slope_deg=slope, contact_terrain_radius=terrain_radius,
        terrain_box_height_mm=height, sensor_actor=sensor,
        resume="", plan=False,
    )


@pytest.mark.parametrize("height,expected_x,expected_top", [
    (0, 5., .005),
    (5, .20, .005),
    (10, .20, .010),
])
def test_physical_course_has_declared_real_world_geom(tmp_path, height,
                                                      expected_x, expected_top):
    dest = tmp_path / f"height_{height}.xml"
    course = collidable_course(ROBOT_XML, dest, height, immutable=True)
    assert course == dest and course.is_file()
    world = ET.parse(course).getroot().find("worldbody")
    assert world is not None
    boxes = [g for g in world.findall("geom")
             if g.get("name") == "v5018_obstacle_step"]
    assert len(boxes) == 1 and boxes[0].get("type") == "box"
    p = [float(v) for v in boxes[0].get("pos").split()]
    z = [float(v) for v in boxes[0].get("size").split()]
    assert p[0] == expected_x and z[0] == .04 and z[1] == .20
    assert p[2] + z[2] == pytest.approx(expected_top)
    assert "euler" not in boxes[0].attrib

    # Files survive repeated training resumes unchanged.
    before = course.read_bytes()
    assert collidable_course(ROBOT_XML, dest, height, immutable=True).read_bytes() == before
    assert hashlib.sha256(dest.read_bytes()).hexdigest()


def test_immutable_training_course_rejects_changed_obstacle(tmp_path):
    dest = tmp_path / "training.xml"
    collidable_course(ROBOT_XML, dest, 5, immutable=True)
    old = dest.read_bytes()
    with pytest.raises(RuntimeError, match="Course file differs"):
        collidable_course(ROBOT_XML, dest, 10, immutable=True)
    assert dest.read_bytes() == old
    # Historical examiner is allowed to reuse its throwaway destination.
    collidable_course(ROBOT_XML, dest, 10)
    assert dest.read_bytes() != old


@pytest.mark.parametrize("height", [0, 5, 10])
def test_pilot_identity_and_preserved_232_input_contract(tmp_path, height):
    cfg = one_run_config(_args(tmp_path, height=height))
    assert cfg.terrain_box_height_mm == height
    assert cfg.observation_schema == OBS_SCHEMA_SENSOR
    assert cfg.resume == ""
    assert cfg.envs == 8
    assert cfg.steps == 256
    assert cfg.domain_terrain_radius == 0
    assert cfg.contact_slope_deg == 0
    assert asdict(cfg)["terrain_box_height_mm"] == height


@pytest.mark.parametrize("overrides,explanation", [
    (dict(profile="foundation"), "requires sim2real"),
    (dict(sensor=False), "requires sim2real"),
    (dict(slope=1.), "cannot mix"),
    (dict(terrain_radius=.1), "cannot mix"),
])
def test_mixed_training_worlds_fail_before_cuda(tmp_path, overrides, explanation):
    with pytest.raises(ValueError, match=explanation):
        one_run_config(_args(tmp_path, height=5, **overrides))


def test_default_train_and_old_config_load_without_new_obstacle(tmp_path):
    args = _args(tmp_path, height=None, profile="foundation", sensor=False)
    cfg = one_run_config(args)
    assert cfg.terrain_box_height_mm == -1
    assert cfg.observation_schema == OBS_SCHEMA_LEGACY
    old = asdict(cfg)
    old.pop("terrain_box_height_mm")
    path = tmp_path / "historical_training_config.json"
    path.write_text(json.dumps(old))
    assert TrainingConfig.load(path).terrain_box_height_mm == -1

    # Native model, not a viewer/demo asset, stays the default.
    from emu.training import env_kwargs_from_args
    assert "terrain_contact_mode" not in env_kwargs_from_args(cfg)


def test_real_cpu_mujoco_compile_when_installed(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    path = collidable_course(ROBOT_XML, tmp_path / "real.xml", 10, immutable=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                            "v5018_obstacle_step")
    assert gid >= 0
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_pos[gid][0] == pytest.approx(.20)
    assert 2 * model.geom_size[gid][2] == pytest.approx(.010)
    assert int(model.geom_bodyid[gid]) == 0
