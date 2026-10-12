"""Package resources and run/result path policy for EMU."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VERSION_FILE = PACKAGE_ROOT / "VERSION"
ROBOT_XML = PACKAGE_ROOT / "robot.xml"
VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()


def source_repo_root() -> Path | None:
    """Return the checkout root when running from source, otherwise None."""
    root = PACKAGE_ROOT.parent
    return root if (root / "pyproject.toml").exists() else None


def run_base() -> Path:
    repo = source_repo_root()
    default = repo.parent / "emu_runs" if repo is not None else Path.home() / "emu_runs"
    return Path(os.environ.get("EMU_RUN_BASE", default)).expanduser().resolve()


def results_base() -> Path:
    repo = source_repo_root()
    default = repo / "results" if repo is not None else Path.home() / "emu_results"
    return Path(os.environ.get("EMU_RESULTS_BASE", default)).expanduser().resolve()


def seed_root(seed: int) -> Path:
    return run_base() / f"{VERSION}_seed{int(seed)}"
