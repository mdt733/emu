"""v5057 source-pinned physical models: ONE skeleton, eight cold-start bodies.

No alternate terrain/actor/trainer, mutable checkout XML, or extra test files.
EMU_PHYSICAL_PROFILE selects one complete mechanical build *before importing emu*.
The generated MJCF lives in an immutable content-addressed user cache; the exact
XML SHA256 is already pinned by checkpoint_io and the .emupolicy exporter.
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


# Paired-seed factorial-ish screen. h40 is an intentionally risky upper bound,
# not a claimed optimum. Source XML is the ONE physical authority for all bodies.
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
# Provisional physical hardware budget: ALL SIX C018 servos live in the
# chassis (6*55g=330g); Pi Zero ~12g, three 14500 cells ~60g, wiring/
# regulator ~18g, structure/brackets ~40g. Weigh actual components later.
TORSO_MASS_KG = .460
BASE_TORSO_X_M = .005
HIP_MOUNT_MASS_KG = .010  # each existing hip carrier, not motor
FOOT_MASS_KG = .010       # each existing spherical foot
CARBON_DENSITY_KG_M3 = 1550.
CARBON_ROD_RADIUS_M = .001
BASE_CARBON_THIGH_ROD_MASS_KG = (
    CARBON_DENSITY_KG_M3*math.pi*CARBON_ROD_RADIUS_M**2*
    math.sqrt(2.)*BASE_HALF_LINK_M)
BASE_CARBON_SHIN_HALF_MASS_KG = (
    CARBON_DENSITY_KG_M3*math.pi*CARBON_ROD_RADIUS_M**2*
    math.sqrt(2.)*BASE_SHIN_QUARTER_M)
REFERENCE_LINK_MASS_KG = 2.*(
    2.*BASE_CARBON_THIGH_ROD_MASS_KG + 2.*BASE_CARBON_SHIN_HALF_MASS_KG)
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


def terrain_tof_pitch_deg(name: str | None = None) -> tuple[float, ...]:
    """Rotate the same rigid 4×4 beam assembly to match neutral sensor height.

    The smallest downward beam reaches ~25cm forward to level ground at
    neutral; the 2×2 carpet only guarantees 37.5cm to its nearest edge.
    Keeping original 40deg at 40cm would reach ~48cm and teach on voids.
    All rows retain fixed 20deg physical separation; no privileged input.
    """
    far = math.degrees(math.atan2(neutral_hip_height(name), .25))
    return tuple(round(far + d, 5) for d in (0., 20., 40., 60.))


def _set_xyz(element: ET.Element, key: str, xyz: tuple[float, float, float]):
    element.set(key, " ".join(f"{c:.8f}" for c in xyz))


def _get(root: ET.Element, xpath: str) -> ET.Element:
    item = root.find(xpath)
    if item is None:
        raise ValueError(f"v5057 model missing required physical link: {xpath}")
    return item


def generate_xml(source_xml: str, name: str) -> str:
    """Deterministic linked geometry + mass + passive-spring variant.

    Joint topology, motors, contact masks, servo torque, sensor channels and
    spring constants follow one shared model. All six servo masses stay in
    the chassis, while only carbon rods move with the legs.
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
        for rod,y in (("thigh_rod",-.006),("thigh_rod_b",.006)):
            thigh = _get(root, f".//geom[@name='{prefix}{rod}']")
            thigh.set("fromto",
                f"0 {y:.3f} 0 {-BASE_HALF_LINK_M*s:.8f} {y:.3f} {-BASE_HALF_LINK_M*s:.8f}")
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
