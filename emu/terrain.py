"""Low-frequency reactive terrain generation for EMU v5014.

These functions intentionally generate *small, smooth nuisance terrain* rather
than stairs or obstacle courses.  EMU has no exteroceptive terrain sensor, so
features must be survivable from proprioception/contact after they are felt.

The arrays are normalized 0..1 for MuJoCo hfield_data.  Wiring an hfield into
the production vector environment is gated on a throughput benchmark because
MJX-JAX cost scales strongly with contact complexity.
"""
from __future__ import annotations

import math
import numpy as np


DEFAULT_ROWS = 33
DEFAULT_COLS = 33
ROUGHNESS_RMS_M = 0.0012
ROUGHNESS_PEAK_M = 0.0045
DISCRETE_FEATURE_MAX_M = 0.006


def _smooth2d(a: np.ndarray, passes: int = 3) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    for _ in range(int(passes)):
        p = np.pad(out, 1, mode='edge')
        out = (
            4.0 * p[1:-1, 1:-1]
            + 2.0 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
            + (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])
        ) / 16.0
    return out


def make_surface(seed: int, severity: float, *, rows: int = DEFAULT_ROWS,
                 cols: int = DEFAULT_COLS, discrete_features: int = 2) -> np.ndarray:
    """Generate a smooth, zero-mean physical height map in metres."""
    severity = max(0.0, min(1.0, float(severity)))
    rng = np.random.default_rng(int(seed))
    z = _smooth2d(rng.standard_normal((int(rows), int(cols))), passes=4)
    z -= float(z.mean())
    rms = float(np.sqrt(np.mean(z * z)))
    if rms > 1e-12:
        z *= (ROUGHNESS_RMS_M * severity) / rms

    # A few broad bumps/hollows/lips.  Their footprints are deliberately wide
    # enough to avoid tiny high-frequency contact facets under a ball foot.
    yy, xx = np.mgrid[0:rows, 0:cols]
    for _ in range(int(discrete_features)):
        cx = rng.uniform(0.15, 0.85) * (cols - 1)
        cy = rng.uniform(0.15, 0.85) * (rows - 1)
        sx = rng.uniform(0.08, 0.22) * cols
        sy = rng.uniform(0.08, 0.22) * rows
        amp = rng.uniform(-DISCRETE_FEATURE_MAX_M, DISCRETE_FEATURE_MAX_M) * severity
        z += amp * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))

    peak = max(abs(float(z.min())), abs(float(z.max())), 1e-12)
    cap = ROUGHNESS_PEAK_M + DISCRETE_FEATURE_MAX_M
    if peak > cap * max(severity, 1e-6):
        z *= (cap * severity) / peak
    z -= float(z.mean())
    return z.astype(np.float32)


def to_mujoco_hfield_data(height_m: np.ndarray, elevation_z_m: float) -> np.ndarray:
    """Convert physical heights to MuJoCo's normalized [0,1] hfield samples.

    A 0.5 sample corresponds to nominal zero height when the hfield geom is
    positioned down by ``0.5*elevation_z_m``.
    """
    elevation_z_m = float(elevation_z_m)
    if elevation_z_m <= 0.0:
        raise ValueError('elevation_z_m must be positive')
    a = 0.5 + np.asarray(height_m, dtype=np.float32) / elevation_z_m
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def doctrine_dict() -> dict[str, float | int]:
    return {
        'TERRAIN_HFIELD_ROWS': DEFAULT_ROWS,
        'TERRAIN_HFIELD_COLS': DEFAULT_COLS,
        'TERRAIN_ROUGHNESS_RMS_M': ROUGHNESS_RMS_M,
        'TERRAIN_ROUGHNESS_PEAK_M': ROUGHNESS_PEAK_M,
        'TERRAIN_DISCRETE_FEATURE_MAX_M': DISCRETE_FEATURE_MAX_M,
    }
