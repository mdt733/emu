"""The single H40 physical robot, generated from one canonical MJCF source.

H40 is the default model for new source-pinned runs. The 26/30/34cm
factorial variants remain available at their original Git commits, not in
this current trainer. Generated H40 XML lives in a content-addressed cache;
its bytes and model SHA256 are unchanged by removing unused profile choices.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class Profile:
    hip_height_m: float
    knee_support: float = 1.0
    hip_roll_stiffness: float = 2.0


# One trained physical model. Alternatives belong in source-pinned research.
PROFILES = {"h40": Profile(.40)}

BASE_NEUTRAL_M = .2584
BASE_HALF_LINK_M = .1202
BASE_SHIN_QUARTER_M = .0601
FOOT_RADIUS_M = .018
# Provisional physical hardware budget: ALL SIX C018 servos live in the
# chassis (6*55g=330g); Pi Zero ~12g, three 14500 cells ~60g, wiring/
# regulator ~18g, structure/brackets ~40g. Weigh actual components later.
TORSO_MASS_KG = .460
BASE_TORSO_X_M = .005
HIP_MOUNT_MASS_KG = .010  # each existing hip carrier, not motor
FOOT_MASS_KG = .010       # each existing spherical foot
CARBON_DENSITY_KG_M3 = 1550.
CARBON_ROD_RADIUS_M = .001  # shin: 2mm diameter ROUND
CARBON_THIGH_SIDE_M = .002  # thigh: ONE 2mm x 2mm SQUARE rod
BASE_CARBON_THIGH_ROD_MASS_KG = (
    CARBON_DENSITY_KG_M3*CARBON_THIGH_SIDE_M**2*
    math.sqrt(2.)*BASE_HALF_LINK_M)
BASE_CARBON_SHIN_HALF_MASS_KG = (
    CARBON_DENSITY_KG_M3*math.pi*CARBON_ROD_RADIUS_M**2*
    math.sqrt(2.)*BASE_SHIN_QUARTER_M)
REFERENCE_LINK_MASS_KG = 2.*(
    BASE_CARBON_THIGH_ROD_MASS_KG + 2.*BASE_CARBON_SHIN_HALF_MASS_KG)
REFERENCE_TOTAL_MASS_KG = (TORSO_MASS_KG + 2.*HIP_MOUNT_MASS_KG +
                           2.*FOOT_MASS_KG + REFERENCE_LINK_MASS_KG)
REFERENCE_KNEE_STIFFNESS_N_M = 750.
# Preserve original source-pinned height-dependent knee spring stiffness;
# mass placement changes are isolated from passive-spring tuning.
KNEE_SPRING_HEIGHT_GAIN = .14/.48
STRUT_HALF_LEVER_M = .0212132  # 30 mm from knee pivot along 45 deg link
STRUT_REST_M = .060
SERVO_RATED_CONTINUOUS_TORQUE_NM = 0.980665  # C018 12V, hard actuator ceiling
SERVO_PEAK_STALL_TORQUE_NM = 2.941995        # only for diagnostic speed-load proxy
SERVO_NOLOAD_RAD_S = (60. / .222) * (3.141592653589793 / 180.)  # 4.7165 rad/s


def selected_profile() -> str:
    name = os.environ.get("EMU_PHYSICAL_PROFILE", "h40")
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
    stiffness = REFERENCE_KNEE_STIFFNESS_N_M * (
        1. + KNEE_SPRING_HEIGHT_GAIN*(scale-1.)) * scale * p.knee_support
    # No height-dependent chassis translation; all six servo masses
    # remain at torso, while light rods move with the legs.
    torso_x = BASE_TORSO_X_M
    rear_moment = REFERENCE_LINK_MASS_KG*scale*(BASE_HALF_LINK_M*scale*.5)
    neutral_com_x = (TORSO_MASS_KG*torso_x - rear_moment)/mass
    return dict(profile=name, hip_height_m=p.hip_height_m,
                fully_straight_leg_m=2 * (2 ** .5) * BASE_HALF_LINK_M * scale,
                thigh_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                shin_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                mass_kg=mass, link_scale=scale, knee_strut_k_n_m=stiffness,
                torso_payload_x_m=torso_x,
                neutral_com_x_m=neutral_com_x,
                torso_mass_kg=TORSO_MASS_KG,
                carbon_rod_mass_kg=REFERENCE_LINK_MASS_KG*scale,
                spring_support_fraction=p.knee_support,
                shin_flex_stiffness_nm_rad=4.0 / scale,
                hip_roll_stiffness_nm_rad=p.hip_roll_stiffness)


def _set_xyz(element: ET.Element, key: str, xyz: tuple[float, float, float]):
    element.set(key, " ".join(f"{c:.8f}" for c in xyz))


def _get(root: ET.Element, xpath: str) -> ET.Element:
    item = root.find(xpath)
    if item is None:
        raise ValueError(f"H40 model missing required physical link: {xpath}")
    return item


def generate_xml(source_xml: str, name: str) -> str:
    """Deterministic linked geometry + mass + passive-spring variant.

    Joint topology, motors, contact masks, servo torque, sensor channels and
    spring constants follow one shared model. All six servo masses stay in
    the chassis. Each thigh is one 2x2mm square carbon rod; each shin is
    one 2mm round carbon rod split at the passive flex joint.
    """
    if name not in PROFILES:
        raise ValueError(f"Unsupported physical model: {name}")
    p = PROFILES[name]
    info = profile_numbers(name)
    s = float(info["link_scale"])
    root = ET.fromstring(source_xml)
    root.set("model", f"emu-{name}")
    _set_xyz(_get(root, ".//body[@name='torso']"), "pos",
             (0., 0., p.hip_height_m))
    chassis = _get(root, ".//geom[@name='torso_box']")
    rear = float(chassis.get("pos").split()[0])-float(chassis.get("size").split()[0])
    if rear > -.010:
        raise ValueError("Chassis must physically overlap the x=0 hip joints")
    for side in "rl":
        prefix = f"{side}_"
        _set_xyz(_get(root, f".//body[@name='{prefix}shin']"), "pos",
                 (-BASE_HALF_LINK_M * s, 0., -BASE_HALF_LINK_M * s))
        for body in ("shin_lower", "foot"):
            _set_xyz(_get(root, f".//body[@name='{prefix}{body}']"), "pos",
                     (BASE_SHIN_QUARTER_M * s, 0.,
                      -BASE_SHIN_QUARTER_M * s))
        thigh = _get(root, f".//geom[@name='{prefix}thigh_rod']")
        # Local box long-axis is Z. Rotate 45deg about Y to align along
        # the folded hip-to-knee diagonal (negative X and negative Z).
        _set_xyz(thigh, "pos", (-BASE_HALF_LINK_M*s*.5, 0.,
                                -BASE_HALF_LINK_M*s*.5))
        thigh.set("size",
                  f"0.00100000 0.00100000 {BASE_HALF_LINK_M*s/math.sqrt(2.):.8f}")
        thigh.set("mass", f"{BASE_CARBON_THIGH_ROD_MASS_KG*s:.8f}")
        for part in ("shin_upper", "shin_lower_rod"):
            geom = _get(root, f".//geom[@name='{prefix}{part}']")
            geom.set("fromto",
                     f"0 0 0 {BASE_SHIN_QUARTER_M*s:.8f} 0 {-BASE_SHIN_QUARTER_M*s:.8f}")
            geom.set("mass", f"{BASE_CARBON_SHIN_HALF_MASS_KG*s:.8f}")
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
    source = original.read_text(encoding="utf-8")
    xml = generate_xml(source, name)
    sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    cache = Path(os.environ.get(
        "EMU_PHYSICAL_MODEL_CACHE",
        Path.home() / ".cache" / "emu" / "physical_models")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"h40-{sha[:20]}.xml"
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
