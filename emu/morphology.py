"""Host-side coherent morphology variants for EMU v5014.

This module deliberately operates on MuJoCo XML *before* ``mjx.put_model``.
Changing geom sizes/poses directly inside a compiled MJX batch is attractive but
can leave collision/inertial derived values inconsistent.  Here MuJoCo gets a
complete XML variant and therefore owns recomputation of all derived physics.

A future morphology-bank runner can compile a small number of these fixed
variants once and interleave training blocks.  The actor is never told which
variant it inhabits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET


THIGH_COMPONENT_M = 0.1202
SHIN_HALF_COMPONENT_M = 0.0601
FOOT_RADIUS_M = 0.018
HIP_HALF_WIDTH_M = 0.08
TORSO_GEOM_X_M = 0.026
THIGH_MASS_KG = 0.040
SHIN_HALF_MASS_KG = 0.015
FOOT_MASS_KG = 0.010
TORSO_MASS_KG = 0.300

# Radius=1 manufacturing envelope.  These are deliberately plausible
# tolerances/alternate builds, not cartoon morphology changes.
LINK_SCALE_FRAC = 0.03
SIDE_ASYMMETRY_FRAC = 0.012
HIP_WIDTH_SCALE_FRAC = 0.03
FOOT_RADIUS_SCALE_FRAC = 0.04
TORSO_MASS_SCALE_FRAC = 0.08
TORSO_COM_X_SHIFT_M = 0.003


@dataclass(frozen=True)
class MorphologySpec:
    r_thigh_scale: float = 1.0
    l_thigh_scale: float = 1.0
    r_shin_scale: float = 1.0
    l_shin_scale: float = 1.0
    hip_width_scale: float = 1.0
    r_foot_scale: float = 1.0
    l_foot_scale: float = 1.0
    torso_mass_scale: float = 1.0
    torso_com_x_shift_m: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _sym_pair(rng: random.Random, radius: float, common_frac: float,
              asym_frac: float) -> tuple[float, float]:
    common = rng.uniform(-common_frac, common_frac) * radius
    asym = rng.uniform(-asym_frac, asym_frac) * radius
    return 1.0 + common + asym, 1.0 + common - asym


def sample_specs(count: int, seed: int, radius: float = 1.0,
                 include_nominal: bool = True) -> list[MorphologySpec]:
    """Sample deterministic, bounded morphology variants.

    Common-mode and left/right terms are sampled separately so the bank contains
    both differently-sized robots and subtly crooked real-world builds.
    """
    radius = max(0.0, min(1.0, float(radius)))
    rng = random.Random(int(seed))
    specs: list[MorphologySpec] = []
    if include_nominal and count > 0:
        specs.append(MorphologySpec())
    while len(specs) < int(count):
        rt, lt = _sym_pair(rng, radius, LINK_SCALE_FRAC, SIDE_ASYMMETRY_FRAC)
        rs, ls = _sym_pair(rng, radius, LINK_SCALE_FRAC, SIDE_ASYMMETRY_FRAC)
        rf, lf = _sym_pair(rng, radius, FOOT_RADIUS_SCALE_FRAC, SIDE_ASYMMETRY_FRAC)
        specs.append(MorphologySpec(
            r_thigh_scale=rt, l_thigh_scale=lt,
            r_shin_scale=rs, l_shin_scale=ls,
            hip_width_scale=1.0 + rng.uniform(-HIP_WIDTH_SCALE_FRAC, HIP_WIDTH_SCALE_FRAC) * radius,
            r_foot_scale=rf, l_foot_scale=lf,
            torso_mass_scale=1.0 + rng.uniform(-TORSO_MASS_SCALE_FRAC, TORSO_MASS_SCALE_FRAC) * radius,
            torso_com_x_shift_m=rng.uniform(-TORSO_COM_X_SHIFT_M, TORSO_COM_X_SHIFT_M) * radius,
        ))
    return specs


def _find(root: ET.Element, tag: str, name: str) -> ET.Element:
    elem = root.find(f".//{tag}[@name='{name}']")
    if elem is None:
        raise ValueError(f"base robot XML is missing {tag} name={name!r}")
    return elem


def _fmt(v: float) -> str:
    return f"{float(v):.8g}"


def _vec(x: float, y: float, z: float) -> str:
    return f"{_fmt(x)} {_fmt(y)} {_fmt(z)}"


def render_variant_xml(base_xml: str, spec: MorphologySpec) -> str:
    """Return one physically coherent XML variant with unchanged topology."""
    root = ET.fromstring(base_xml)

    # Torso/body placement: start high enough that the longer leg's ball just
    # touches the nominal plane.  This is a reset geometry change, not a secret
    # task advantage; both sides retain the same world floor.
    r_total = THIGH_COMPONENT_M * spec.r_thigh_scale + 2.0 * SHIN_HALF_COMPONENT_M * spec.r_shin_scale + FOOT_RADIUS_M * spec.r_foot_scale
    l_total = THIGH_COMPONENT_M * spec.l_thigh_scale + 2.0 * SHIN_HALF_COMPONENT_M * spec.l_shin_scale + FOOT_RADIUS_M * spec.l_foot_scale
    _find(root, 'body', 'torso').set('pos', _vec(0.0, 0.0, max(r_total, l_total)))

    torso = _find(root, 'geom', 'torso_box')
    torso.set('mass', _fmt(TORSO_MASS_KG * spec.torso_mass_scale))
    torso.set('pos', _vec(TORSO_GEOM_X_M + spec.torso_com_x_shift_m, 0.0, 0.0))

    _find(root, 'body', 'r_hip').set('pos', _vec(0.0, -HIP_HALF_WIDTH_M * spec.hip_width_scale, 0.0))
    _find(root, 'body', 'l_hip').set('pos', _vec(0.0, +HIP_HALF_WIDTH_M * spec.hip_width_scale, 0.0))

    for side, thigh_scale, shin_scale, foot_scale in (
        ('r', spec.r_thigh_scale, spec.r_shin_scale, spec.r_foot_scale),
        ('l', spec.l_thigh_scale, spec.l_shin_scale, spec.l_foot_scale),
    ):
        thigh = THIGH_COMPONENT_M * thigh_scale
        half_shin = SHIN_HALF_COMPONENT_M * shin_scale

        thigh_geom = _find(root, 'geom', f'{side}_thigh_rod')
        thigh_geom.set('fromto', _vec(0.0, 0.0, 0.0) + ' ' + _vec(-thigh, 0.0, -thigh))
        thigh_geom.set('mass', _fmt(THIGH_MASS_KG * thigh_scale))
        _find(root, 'body', f'{side}_shin').set('pos', _vec(-thigh, 0.0, -thigh))

        upper = _find(root, 'geom', f'{side}_shin_upper')
        upper.set('fromto', _vec(0.0, 0.0, 0.0) + ' ' + _vec(half_shin, 0.0, -half_shin))
        upper.set('mass', _fmt(SHIN_HALF_MASS_KG * shin_scale))
        _find(root, 'body', f'{side}_shin_lower').set('pos', _vec(half_shin, 0.0, -half_shin))

        lower = _find(root, 'geom', f'{side}_shin_lower_rod')
        lower.set('fromto', _vec(0.0, 0.0, 0.0) + ' ' + _vec(half_shin, 0.0, -half_shin))
        lower.set('mass', _fmt(SHIN_HALF_MASS_KG * shin_scale))
        _find(root, 'body', f'{side}_foot').set('pos', _vec(half_shin, 0.0, -half_shin))

        foot = _find(root, 'geom', f'{side}_foot_ball')
        foot.set('size', _fmt(FOOT_RADIUS_M * foot_scale))
        foot.set('mass', _fmt(FOOT_MASS_KG * foot_scale ** 3))

    return ET.tostring(root, encoding='unicode')


def write_bank(base_xml_path: Path, out_dir: Path, *, count: int = 16,
               seed: int = 5014, radius: float = 1.0) -> Path:
    """Materialize a deterministic morphology bank and JSON manifest."""
    base_xml_path = Path(base_xml_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_xml_path.read_text(encoding='utf-8')
    specs = sample_specs(count=count, seed=seed, radius=radius, include_nominal=True)
    rows = []
    for i, spec in enumerate(specs):
        name = f'emu_morph_{i:02d}.xml'
        (out_dir / name).write_text(render_variant_xml(base, spec), encoding='utf-8')
        rows.append({'index': i, 'xml': name, 'spec': spec.as_dict()})
    manifest = out_dir / 'morphology_manifest.json'
    manifest.write_text(json.dumps({
        'schema': 1, 'seed': int(seed), 'radius': float(radius), 'variants': rows,
    }, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def doctrine_dict() -> dict[str, float]:
    return {
        'MORPH_LINK_SCALE_FRAC': LINK_SCALE_FRAC,
        'MORPH_SIDE_ASYMMETRY_FRAC': SIDE_ASYMMETRY_FRAC,
        'MORPH_HIP_WIDTH_SCALE_FRAC': HIP_WIDTH_SCALE_FRAC,
        'MORPH_FOOT_RADIUS_SCALE_FRAC': FOOT_RADIUS_SCALE_FRAC,
        'MORPH_TORSO_MASS_SCALE_FRAC': TORSO_MASS_SCALE_FRAC,
        'MORPH_TORSO_COM_X_SHIFT_M': TORSO_COM_X_SHIFT_M,
    }
