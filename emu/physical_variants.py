"""v5057 source-pinned physical models: ONE skeleton, eight cold-start bodies.

No alternate terrain/actor/trainer, mutable checkout XML, or extra test files.
EMU_PHYSICAL_PROFILE selects one complete mechanical build *before importing emu*.
The generated MJCF lives in an immutable content-addressed user cache; the exact
XML SHA256 is already pinned by checkpoint_io and the .emupolicy exporter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class Profile:
    hip_height_m: float
    knee_support: float = 1.0
    hip_roll_stiffness: float = 2.0


# Paired-seed factorial-ish screen. h40 is an intentionally risky upper bound,
# not a claimed optimum. Source XML at base26 is byte-for-byte v5056.
PROFILES = {
    "base26": Profile(.2584),
    "h30": Profile(.30),
    "h34": Profile(.34),
    "h40": Profile(.40),
    "h34_k75": Profile(.34, knee_support=.75),
    "h34_k125": Profile(.34, knee_support=1.25),
    "h34_roll1": Profile(.34, hip_roll_stiffness=1.0),
    "h40_k75": Profile(.40, knee_support=.75),
}

BASE_NEUTRAL_M = .2584
BASE_HALF_LINK_M = .1202
BASE_SHIN_QUARTER_M = .0601
FOOT_RADIUS_M = .018
REFERENCE_TOTAL_MASS_KG = .48
REFERENCE_LINK_MASS_KG = .14
REFERENCE_KNEE_STIFFNESS_N_M = 750.
STRUT_HALF_LEVER_M = .0212132  # 30 mm from knee pivot along 45 deg link
STRUT_REST_M = .060
SERVO_NOLOAD_RAD_S = (60. / .222) * (3.141592653589793 / 180.)  # 4.7165 rad/s


def selected_profile() -> str:
    name = os.environ.get("EMU_PHYSICAL_PROFILE", "base26")
    if name not in PROFILES:
        raise ValueError(f"Unknown EMU_PHYSICAL_PROFILE={name!r}; allowed: {', '.join(PROFILES)}")
    return name


def neutral_hip_height(name: str | None = None) -> float:
    return PROFILES[name or selected_profile()].hip_height_m


def profile_numbers(name: str) -> dict[str, float | str]:
    p = PROFILES[name]
    scale = (p.hip_height_m - FOOT_RADIUS_M) / (
        BASE_NEUTRAL_M - FOOT_RADIUS_M)
    mass = REFERENCE_TOTAL_MASS_KG + REFERENCE_LINK_MASS_KG * (scale - 1.)
    # Same 30mm knee lever/cartridge travel across profiles. Keep neutral
    # spring support approximately matched to the *real new link mass* and
    # moment arm; do not accidentally make the 40cm model under-sprung.
    stiffness = REFERENCE_KNEE_STIFFNESS_N_M * (
        mass / REFERENCE_TOTAL_MASS_KG) * scale * p.knee_support
    return dict(profile=name, hip_height_m=p.hip_height_m,
                fully_straight_leg_m=2 * (2 ** .5) * BASE_HALF_LINK_M * scale,
                thigh_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                shin_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                mass_kg=mass, link_scale=scale, knee_strut_k_n_m=stiffness,
                spring_support_fraction=p.knee_support,
                shin_flex_stiffness_nm_rad=4.0 / scale,
                hip_roll_stiffness_nm_rad=p.hip_roll_stiffness)


def _set_xyz(element: ET.Element, key: str, xyz: tuple[float, float, float]):
    element.set(key, " ".join(f"{c:.8f}" for c in xyz))


def _get(root: ET.Element, xpath: str) -> ET.Element:
    item = root.find(xpath)
    if item is None:
        raise ValueError(f"v5057 model missing required physical link: {xpath}")
    return item


def generate_xml(source_xml: str, name: str) -> str:
    """Deterministic linked geometry + mass + passive-spring variant.

    Topology, actuator count, contact colliders, sensors, servo/phase control
    and foot sphere radius are intentionally UNCHANGED. Only experimental
    physical dimensions, rod masses, knee-strut force, hip-roll spring vary.
    """
    if name not in PROFILES:
        raise ValueError(f"Unsupported physical model: {name}")
    if name == "base26":
        return source_xml
    p = PROFILES[name]
    info = profile_numbers(name)
    s = float(info["link_scale"])
    root = ET.fromstring(source_xml)
    root.set("model", f"emu-{name}")
    _set_xyz(_get(root, ".//body[@name='torso']"), "pos",
             (0., 0., p.hip_height_m))
    for side in "rl":
        prefix = f"{side}_"
        _set_xyz(_get(root, f".//body[@name='{prefix}shin']"), "pos",
                 (-BASE_HALF_LINK_M * s, 0., -BASE_HALF_LINK_M * s))
        for body in ("shin_lower", "foot"):
            _set_xyz(_get(root, f".//body[@name='{prefix}{body}']"), "pos",
                     (BASE_SHIN_QUARTER_M * s, 0.,
                      -BASE_SHIN_QUARTER_M * s))
        thigh = _get(root, f".//geom[@name='{prefix}thigh_rod']")
        thigh.set("fromto",
                   f"0 0 0 {-BASE_HALF_LINK_M*s:.8f} 0 {-BASE_HALF_LINK_M*s:.8f}")
        thigh.set("mass", f"{.040*s:.8f}")
        for part in ("shin_upper", "shin_lower_rod"):
            geom = _get(root, f".//geom[@name='{prefix}{part}']")
            geom.set("fromto",
                     f"0 0 0 {BASE_SHIN_QUARTER_M*s:.8f} 0 {-BASE_SHIN_QUARTER_M*s:.8f}")
            geom.set("mass", f"{.015*s:.8f}")
        knee_pin = -BASE_HALF_LINK_M*s + STRUT_HALF_LEVER_M
        _set_xyz(_get(root, f".//site[@name='{prefix}knee_strut_thigh']"),
                 "pos", (knee_pin, 0., knee_pin))
        # The shin site remains 30mm from the pivot, independently of link
        # length; this is the same physical hinge spring mechanism.
        _get(root, f".//joint[@name='{prefix}hip_roll']").set(
            "stiffness", f"{p.hip_roll_stiffness:.6f}")
        # Same tube section/material, longer bending segment: equivalent
        # rotational flex stiffness scales approximately EI/L, not constant.
        _get(root, f".//joint[@name='{prefix}shin_flex']").set(
            "stiffness", f"{4.0/s:.8f}")
        _get(root, f".//spatial[@name='{prefix}knee_support']").set(
            "stiffness", f"{float(info['knee_strut_k_n_m']):.8f}")
    return ET.tostring(root, encoding="unicode")


def robot_xml_path(package_dir: Path) -> Path:
    name = selected_profile()
    original = package_dir / "robot.xml"
    if name == "base26":
        return original
    source = original.read_text(encoding="utf-8")
    xml = generate_xml(source, name)
    sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    cache = Path(os.environ.get(
        "EMU_PHYSICAL_MODEL_CACHE",
        Path.home() / ".cache" / "emu" / "physical_models")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"v5057-{name}-{sha[:20]}.xml"
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Corrupt immutable EMU model cache: {target}")
        return target
    with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache, prefix=".emu-", suffix=".tmp",
            delete=False) as temporary:
        temporary.write(xml.encode("utf-8"))
        temp = Path(temporary.name)
    os.replace(temp, target)
    return target
