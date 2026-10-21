"""Atomic, human-readable experiment records shared by active campaigns.

Experiment bookkeeping belongs here, not in archived campaign recipes.
Checkpoint serialization and source validation remain in checkpoint_io.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path):
    """Read an existing record; missing records are an error, not a zero result."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    """Atomically persist a finite JSON record without mutating checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    try:
        temp.write_text(json.dumps(value, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def sha256_file(path: Path) -> str:
    """Content identity of one saved checkpoint or immutable scorecard."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
