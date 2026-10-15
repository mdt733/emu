"""Explicit *future* v5016 actor contract, deliberately not a legacy alias.

These measurements are not silently injected into the existing trainer.
A v5016 policy needs matching training/export/viewer/hardware version tags.
Until that complete migration, the live v5015 contract remains 220→6.
"""
from __future__ import annotations

from .contracts import BASE_OBS_FIELDS, HISTORY_LEN, ACTION_DIM

V5016_NEW_FIELDS = (
    'ground_grade_forward_rad',
    'ground_grade_lateral_rad',
    'ground_plane_valid',
    'heading_error_sin',
    'heading_error_cos',
    'heading_estimate_valid',
)
V5016_BASE_FIELDS = BASE_OBS_FIELDS + V5016_NEW_FIELDS
V5016_OBS_DIM = len(V5016_BASE_FIELDS) * HISTORY_LEN
V5016_ACTION_DIM = ACTION_DIM

assert len(V5016_BASE_FIELDS) == len(set(V5016_BASE_FIELDS)) == 61
assert V5016_OBS_DIM == 244 and V5016_ACTION_DIM == 6


def declared_actor_contract(version: str):
    """Inspection only. No legacy weights are ever adapted or padded."""
    if version == 'v5015':
        return {'version': version, 'base_fields': BASE_OBS_FIELDS,
                'history': HISTORY_LEN, 'observations': len(BASE_OBS_FIELDS)*HISTORY_LEN,
                'actions': ACTION_DIM, 'runtime_ready': True}
    if version == 'v5016-candidate':
        return {'version': version, 'base_fields': V5016_BASE_FIELDS,
                'history': HISTORY_LEN, 'observations': V5016_OBS_DIM,
                'actions': V5016_ACTION_DIM, 'runtime_ready': False}
    raise ValueError('unknown actor contract version: ' + str(version))
