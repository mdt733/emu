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
    "h40_carbon2": Profile(.40),  # v5059: estimated 2mm load-bearing shin mechanism
}

BASE_NEUTRAL_M = .2584
BASE_HALF_LINK_M = .1202
BASE_SHIN_QUARTER_M = .0601
FOOT_RADIUS_M = .018
REFERENCE_TOTAL_MASS_KG = .48
REFERENCE_LINK_MASS_KG = .14
REFERENCE_KNEE_STIFFNESS_N_M = 750.
TORSO_MASS_KG = .30
BASE_TORSO_X_M = .026
# Two folded thighs (40g), two upper shins (15g), two lower shins (15g).
# Each link's mass grows as s and its rear COM offset as s: moment ~s².
BASE_LINK_REAR_MOMENT_KG_M = 2. * (
    .040 * .0601 + .015 * .09015 + .015 * .03005)
STRUT_HALF_LEVER_M = .0212132  # 30 mm from knee pivot along 45 deg link
STRUT_REST_M = .060
SERVO_RATED_CONTINUOUS_TORQUE_NM = 0.980665  # C018 12V, hard actuator ceiling
SERVO_PEAK_STALL_TORQUE_NM = 2.941995        # only for diagnostic speed-load proxy
SERVO_NOLOAD_RAD_S = (60. / .222) * (3.141592653589793 / 180.)  # 4.7165 rad/s

# v5059 hardware assumptions: change these documented source-pinned values
# when an actual bending, spring, backdrive or weigh-in test becomes available.
# No extra networks, training paths or dynamic environment variable overrides.
CARBON_PROFILE = "h40_carbon2"
CARBON_ROD_DIAMETER_M = .002
CARBON_FLEXURAL_MODULUS_PA = 30e9  # estimated, not measured
CARBON_DENSITY_KG_M3 = 1600.     # estimated, not weighed
CARBON_FOOT_RADIUS_M = .005      # 10mm assumed ball DIAMETER
CARBON_TAIL_LENGTH_M = .040
CARBON_KNEE_SPRING_N_M = 450.    # tension-only spring assumption
CARBON_KNEE_SPRING_SLACK_M = .031
CARBON_KNEE_THIGH_ANCHOR_M = .030
CARBON_SHIN_FLEX_DAMPING_RATIO = .15
# Lower shin half plus 3g foot+mount inertial estimate about midpoint.
CARBON_LOWER_EFFECTIVE_J_KGM2 = .00006319
CARBON_SHIN_FLEX_LIMIT_DEG = 25.
# Proposed assembly mass ledger: all six 55g servos live on the torso,
# the remaining 90g of its mass is electronics, battery and chassis.
CARBON_TORSO_MASS_KG = .420
CARBON_HIP_JOINT_MASS_KG = .010 # per hip pivot/hardware
CARBON_THIGH_SPAR_MASS_KG = .009
CARBON_KNEE_DRIVE_MASS_KG = .003 # parallel rod/fasteners proxy, not closed loop
CARBON_SHIN_UPPER_ROD_MASS_KG = .00072
CARBON_SHIN_LOWER_ROD_MASS_KG = .00072
CARBON_KNEE_CLAMP_MASS_KG = .0024
CARBON_HALL_REFERENCE_MASS_KG = .0003
CARBON_FOOT_BALL_MASS_KG = .0007
CARBON_FOOT_MOUNT_MASS_KG = .0023
CARBON_ACTUATED_DAMPING_NMS_RAD = .018   # unpowered brake reduced from .624
CARBON_ACTUATED_FRICTIONLOSS_NM = .025
CARBON_ACTUATED_ARMATURE_KGM2 = .003 # preliminary 12V gearmotor reflection

def carbon_shin_flex_parameters() -> tuple[float, float]:
    """Two-axis midpoint-hinge approximation fitted to a tip-load cantilever.

    For one midpoint hinge, tip deflection F*L²/(4k); the uniformly
    bending rod deflects F*L³/(3EI). Therefore k=3EI/(4L).
    A measured force-deflection curve supersedes these estimates.
    """
    length=(PROFILES[CARBON_PROFILE].hip_height_m - CARBON_FOOT_RADIUS_M) / math.sqrt(2.)
    second_moment=math.pi*CARBON_ROD_DIAMETER_M**4/64.
    stiffness=.75*CARBON_FLEXURAL_MODULUS_PA*second_moment/length
    damping=2.*CARBON_SHIN_FLEX_DAMPING_RATIO * math.sqrt(
        stiffness*CARBON_LOWER_EFFECTIVE_J_KGM2)
    return stiffness, damping



def selected_profile() -> str:
    name = os.environ.get("EMU_PHYSICAL_PROFILE", "base26")
    if name not in PROFILES:
        raise ValueError(f"Unknown EMU_PHYSICAL_PROFILE={name!r}; allowed: {', '.join(PROFILES)}")
    return name


def neutral_hip_height(name: str | None = None) -> float:
    return PROFILES[name or selected_profile()].hip_height_m


def profile_numbers(name: str) -> dict[str, float | str]:
    p = PROFILES[name]
    radius = CARBON_FOOT_RADIUS_M if name == CARBON_PROFILE else FOOT_RADIUS_M
    scale = (p.hip_height_m - radius) / (BASE_NEUTRAL_M - FOOT_RADIUS_M)
    if name == CARBON_PROFILE:
        # All moving-part mass is explicit in the generated XML, including
        # two 55g*3 servo assemblies per side inside the torso mass ledger.
        per_leg=(CARBON_HIP_JOINT_MASS_KG + CARBON_THIGH_SPAR_MASS_KG
                 +CARBON_KNEE_DRIVE_MASS_KG+CARBON_SHIN_UPPER_ROD_MASS_KG
                 +CARBON_SHIN_LOWER_ROD_MASS_KG+CARBON_KNEE_CLAMP_MASS_KG
                 +CARBON_HALL_REFERENCE_MASS_KG+CARBON_FOOT_BALL_MASS_KG
                 +CARBON_FOOT_MOUNT_MASS_KG)
        mass = CARBON_TORSO_MASS_KG + 2.*per_leg
        a=BASE_HALF_LINK_M*scale
        # Neutral x COM of rigid rod/inertial proxies should be above the
        # neutral ball foot. This is a proposed real torso mounting location.
        torso_x=2.*(CARBON_THIGH_SPAR_MASS_KG*a/2.
                    +CARBON_KNEE_DRIVE_MASS_KG*a/2.
                    +CARBON_SHIN_UPPER_ROD_MASS_KG*3*a/4.
                    +CARBON_SHIN_LOWER_ROD_MASS_KG*a/4.
                    +CARBON_HALL_REFERENCE_MASS_KG*a/4.
                    +CARBON_KNEE_CLAMP_MASS_KG*a)/CARBON_TORSO_MASS_KG
        flex,damping=carbon_shin_flex_parameters()
        return dict(profile=name,hip_height_m=p.hip_height_m,
                    foot_radius_m=radius,
                    fully_straight_leg_m=2*math.sqrt(2)*a,
                    thigh_length_m=math.sqrt(2)*a,
                    shin_length_m=math.sqrt(2)*a,
                    carbon_rod_diameter_m=CARBON_ROD_DIAMETER_M,
                    carbon_rod_flexural_modulus_pa=CARBON_FLEXURAL_MODULUS_PA,
                    carbon_density_kg_m3=CARBON_DENSITY_KG_M3,
                    shin_extension_m=CARBON_TAIL_LENGTH_M,
                    shin_flex_stiffness_nm_rad=flex,
                    shin_flex_damping_nms_rad=damping,
                    shin_flex_limit_deg=CARBON_SHIN_FLEX_LIMIT_DEG,
                    knee_strut_k_n_m=CARBON_KNEE_SPRING_N_M,
                    knee_spring_slack_m=CARBON_KNEE_SPRING_SLACK_M,
                    torso_payload_x_m=torso_x,mass_kg=mass,link_scale=scale,
                    spring_support_fraction=1.,
                    hip_roll_stiffness_nm_rad=p.hip_roll_stiffness)
    mass = REFERENCE_TOTAL_MASS_KG + REFERENCE_LINK_MASS_KG * (scale - 1.)
    # Same 30mm knee lever/cartridge travel across profiles. Keep neutral
    # spring support approximately matched to the *real new link mass* and
    # moment arm; do not accidentally make the 40cm model under-sprung.
    stiffness = REFERENCE_KNEE_STIFFNESS_N_M * (
        mass / REFERENCE_TOTAL_MASS_KG) * scale * p.knee_support
    # Keep the neutral horizontal COM over the same contact point, otherwise
    # increasing length quietly makes the body back-heavy. This implies
    # a real payload/chassis placement change, particularly for 40cm legs.
    torso_x = (BASE_TORSO_X_M + BASE_LINK_REAR_MOMENT_KG_M
               * (scale * scale - 1.) / TORSO_MASS_KG)
    return dict(profile=name, hip_height_m=p.hip_height_m,
                fully_straight_leg_m=2 * (2 ** .5) * BASE_HALF_LINK_M * scale,
                thigh_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                shin_length_m=(2 ** .5) * BASE_HALF_LINK_M * scale,
                mass_kg=mass, foot_radius_m=radius,
                link_scale=scale, knee_strut_k_n_m=stiffness,
                torso_payload_x_m=torso_x,
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

    Motor count, actuator channel meanings, trainer, observations and prism
    topology are constant. The explicitly named CARBON_PROFILE is a new
    experimental physical assembly with a smaller foot and lateral shin flex.
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
    _set_xyz(_get(root, ".//geom[@name='torso_box']"), "pos",
             (float(info["torso_payload_x_m"]), 0., 0.))
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
        spring = _get(root, f".//spatial[@name='{prefix}knee_support']")
        spring.set("stiffness", f"{float(info['knee_strut_k_n_m']):.8f}")
        if name == CARBON_PROFILE:
            # The true shin is a 2mm diameter flexing rod. Radius controls
            # virtual geometry; masses include separately budgeted connectors.
            thigh.set("size", f"{CARBON_ROD_DIAMETER_M/2.:.8f}")
            thigh.set("mass", f"{CARBON_THIGH_SPAR_MASS_KG:.8f}")
            thigh_body = _get(root, f".//body[@name='{prefix}thigh']")
            ET.SubElement(thigh_body,"geom",{
                "name":f"{prefix}knee_drive_rod", "type":"capsule",
                "size":f"{CARBON_ROD_DIAMETER_M/2.:.8f}",
                "fromto":f"0 0.008 0 {-BASE_HALF_LINK_M*s:.8f} 0.008 {-BASE_HALF_LINK_M*s:.8f}",
                "mass":f"{CARBON_KNEE_DRIVE_MASS_KG:.8f}",
                "contype":"0","conaffinity":"0"})
            for segment,mass_part in (("shin_upper",CARBON_SHIN_UPPER_ROD_MASS_KG),
                                      ("shin_lower_rod",CARBON_SHIN_LOWER_ROD_MASS_KG)):
                geom = _get(root, f".//geom[@name='{prefix}{segment}']")
                geom.set("size",f"{CARBON_ROD_DIAMETER_M/2.:.8f}")
                geom.set("mass",f"{mass_part:.8f}")
            torso = _get(root,".//geom[@name='torso_box']")
            torso.set("mass",f"{CARBON_TORSO_MASS_KG:.8f}")
            shin_body = _get(root, f".//body[@name='{prefix}shin']")
            ET.SubElement(shin_body,"geom",{
                "name":f"{prefix}knee_clamp", "type":"sphere","size":"0.004",
                "pos":"0 0 0", "mass":f"{CARBON_KNEE_CLAMP_MASS_KG:.8f}",
                "contype":"0","conaffinity":"0"})
            shin_lower = _get(root, f".//body[@name='{prefix}shin_lower']")
            # Unloaded Hall reference/magnet, mounted to the lower-half
            # body: its angular offset from upper shin is the Hall proxy.
            ET.SubElement(shin_lower,"geom",{
                "name":f"{prefix}hall_reference", "type":"capsule",
                "size":"0.0004",
                "fromto":f"0 0 0 {-BASE_SHIN_QUARTER_M*s+.003:.8f} 0 {BASE_SHIN_QUARTER_M*s-.003:.8f}",
                "mass":f"{CARBON_HALL_REFERENCE_MASS_KG:.8f}",
                "contype":"0","conaffinity":"0"})
            bend = _get(root, f".//joint[@name='{prefix}shin_flex']")
            bend.set("stiffness",f"{info['shin_flex_stiffness_nm_rad']:.8f}")
            bend.set("damping",f"{info['shin_flex_damping_nms_rad']:.8f}")
            bend.set("range",f"{-CARBON_SHIN_FLEX_LIMIT_DEG:.4f} {CARBON_SHIN_FLEX_LIMIT_DEG:.4f}")
            # The real round rod can bend laterally too. This passive DOF
            # does not alter the fixed 360-input actor or six motor count.
            shin_lower.insert(list(shin_lower).index(bend)+1,
                ET.Element("joint",{
                    "name":f"{prefix}shin_flex_roll","type":"hinge",
                    "axis":"1 0 0",
                    "range":f"{-CARBON_SHIN_FLEX_LIMIT_DEG:.4f} {CARBON_SHIN_FLEX_LIMIT_DEG:.4f}",
                    "stiffness":f"{info['shin_flex_stiffness_nm_rad']:.8f}",
                    "damping":f"{info['shin_flex_damping_nms_rad']:.8f}"}))
            foot_body = _get(root, f".//body[@name='{prefix}foot']")
            ball = _get(root, f".//geom[@name='{prefix}foot_ball']")
            ball.set("size",f"{CARBON_FOOT_RADIUS_M:.8f}")
            ball.set("mass",f"{CARBON_FOOT_BALL_MASS_KG:.8f}")
            ET.SubElement(foot_body,"geom",{
                "name":f"{prefix}foot_mount","type":"sphere","size":"0.003",
                "pos":"0 0 0.006", "mass":f"{CARBON_FOOT_MOUNT_MASS_KG:.8f}",
                "contype":"0","conaffinity":"0"})
            # Spring now PULLS shin tail (~40mm BEHIND knee) toward the
            # thigh anchor. springlength='0 L' ensures TENSION ONLY:
            # when slack it contributes zero force instead of pushing.
            tail=CARBON_TAIL_LENGTH_M/math.sqrt(2.)
            _set_xyz(_get(root,f".//site[@name='{prefix}knee_strut_shin']"),
                     "pos",(-tail,0.,tail))
            knee_a=-BASE_HALF_LINK_M*s+CARBON_KNEE_THIGH_ANCHOR_M/math.sqrt(2.)
            _set_xyz(_get(root,f".//site[@name='{prefix}knee_strut_thigh']"),
                     "pos",(knee_a,0.,knee_a))
            spring.set("springlength",f"0 {CARBON_KNEE_SPRING_SLACK_M:.8f}")
            # The old .624 damping was an unpowered pseudo-motor brake.
            for joint_name in ("hip_roll","hip_pitch","knee"):
                joint=_get(root,f".//joint[@name='{prefix}{joint_name}']")
                joint.set("damping",f"{CARBON_ACTUATED_DAMPING_NMS_RAD:.8f}")
                joint.set("frictionloss",f"{CARBON_ACTUATED_FRICTIONLOSS_NM:.8f}")
                joint.set("armature",f"{CARBON_ACTUATED_ARMATURE_KGM2:.8f}")
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
