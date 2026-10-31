"""Shared immutable historical-policy test fixture.

Current showcase/champion.emupolicy belongs to the 360-input v5032 model.
Legacy 232-input regression tests deliberately evaluate the *archived*
v5016 artifact, not whichever actor happens to be public today.

GitHub CI fetches full history. A local non-shallow Git clone also
contains the original pre-promotion main commit. Never rewrite policy
sensor channels or silently skip these tests when Git history is absent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
V5016_ROLLBACK_SHA = "2ec9e9db0a816c658131311129ef89ec7a595a12"
V5016_POLICY_BLOB_SHA = "163a61fef6e75fcba91f254830f3858210b491a0"


@pytest.fixture(scope="session")
def archived_232_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Original source-exact v5016 232 policy; byte-validated from Git."""
    try:
        raw = subprocess.check_output(
            ["git", "show",
             V5016_ROLLBACK_SHA + ":showcase/champion.emupolicy"],
            cwd=REPO,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Historical 232 policy unavailable: use a normal Git clone "
            "with full history (CI uses checkout fetch-depth: 0). "
            f"Rollback SHA={V5016_ROLLBACK_SHA}: {exc}"
        )
    git_blob = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()
    assert git_blob == V5016_POLICY_BLOB_SHA, (
        "Archived 232 policy bytes differ from approved rollback blob"
    )
    path = tmp_path_factory.mktemp("archived_v5016") / "champion_232.emupolicy"
    path.write_bytes(raw)
    return path
