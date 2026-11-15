"""The single H40 physical robot, generated from one canonical MJCF source.

H40 is the protected mainline model. One opt-in, source-pinned research
profile changes only knee spring geometry/preload and knee joint limits.
Both generated XMLs use independent content-addressed cache identities.
The mainline H40 XML bytes and SHA256 MUST remain unchanged.
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


# One protected champion + one deliberately untrained cold-start research robot.
# Research strut: effective 150 N/m, ~0.521 Nm neutral knee assistance,
# 35/25 mm offset anchors, and -90..+95deg knee range. This spatial tendon
# springlength represents equivalent preload; a real long cartridge/rocker
# capable of this stroke has NOT been mechanically designed or validated.
RESEARCH_PROFILE = "h40-spring-v1"
PROFILES = {"h40": Profile(.40), RESEARCH_PROFILE: Profile(.40)}

# NN position-target interpretation is part of the robot contract: never
# reinterpret the frozen champion's six actions under a different profile.
def target_span_rad(name: str | None = None) -> tuple[float, ...]:
    chosen = name or selected_profile()
    if chosen not in PROFILES:
        raise ValueError(f"Unknown EMU physical profile {chosen!r}")
    if chosen == RESEARCH_PROFILE:
        return (math.radians(25.), math.radians(50.), math.radians(90.),
                math.radians(25.), math.radians(50.), math.radians(90.))
    # Preserve exactly the historical champion floats and action scaling.
    return (.38, .48, .58, .38, .48, .58)


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
    if name == RESEARCH_PROFILE:
        stiffness = 150.0  # fixed H40 research spring, not a new hardware rating
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
        spring = _get(root, f".//spatial[@name='{prefix}knee_support']")
        spring.set("stiffness", f"{float(info['knee_strut_k_n_m']):.8f}")
        if name == RESEARCH_PROFILE:
            # Offset the two anchor radii: equal 30/30mm sites coincide at
            # -90deg. Offset 35/25mm leaves 10mm separation at that angle.
            thigh_arm, shin_arm = .035, .025
            lever = thigh_arm * shin_arm / math.hypot(thigh_arm, shin_arm)
            rest = (math.hypot(thigh_arm, shin_arm) +
                    .520606 / (150.0 * lever))
            thigh_pin = -BASE_HALF_LINK_M * s + thigh_arm / math.sqrt(2.)
            _set_xyz(_get(root, f".//site[@name='{prefix}knee_strut_thigh']"),
                     "pos", (thigh_pin, 0., thigh_pin))
            _set_xyz(_get(root, f".//site[@name='{prefix}knee_strut_shin']"),
                     "pos", (shin_arm / math.sqrt(2.), 0.,
                             -shin_arm / math.sqrt(2.)))
            spring.set("springlength", f"{rest:.9f}")
            _get(root, f".//joint[@name='{prefix}knee']").set(
                "range", "-90 95")
            # Match the physical ROM on the actuator as well; the 6-output
            # policy controls +/-90deg, not a residual +/-33deg clamp.
            _get(root, f".//position[@name='{prefix}knee']").set(
                "ctrlrange", "-90 95")
            _get(root, f".//position[@name='{prefix}hip_roll']").set(
                "ctrlrange", "-25 25")
            _get(root, f".//position[@name='{prefix}hip_pitch']").set(
                "ctrlrange", "-50 50")
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
    target = cache / f"{name}-{sha[:20]}.xml"
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
