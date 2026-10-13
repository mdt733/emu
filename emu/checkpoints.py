"""Canonical checkpoint discovery for modern EMU."""
from __future__ import annotations

from pathlib import Path


def resume_candidates(run_dir: Path) -> list[Path]:
    """Canonical continuation order for one run, strongest semantic file first."""
    ck = Path(run_dir) / "checkpoints"
    # Both may exist when a process was interrupted between updating latest
    # and promoting final. The most recently committed file wins.
    out = sorted([ck / "final.pkl", ck / "latest.pkl"],
                 key=lambda p: p.stat().st_mtime_ns if p.exists() else -1,
                 reverse=True)
    out.extend(sorted(ck.glob("snapshot_step_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True))
    return [path for path in out if path.exists()]


def find_resume(run_dir: Path) -> Path | None:
    """Best checkpoint path from which to continue one run; payload is not opened."""
    candidates = resume_candidates(run_dir)
    return candidates[0] if candidates else None


def find_valid_resume(run_dir: Path) -> Path | None:
    """Best current-schema continuation checkpoint, skipping corrupt/incompatible files."""
    from .checkpoint_io import load_checkpoint

    for path in resume_candidates(run_dir):
        try:
            load_checkpoint(path, validate=True)
            return path
        except Exception:
            continue
    return None


def find_evaluation(root: Path) -> Path | None:
    """Best policy checkpoint for read-only viewing/evaluation across experiment arms."""
    candidates: list[tuple[int, float, Path]] = []
    for run in ("graph", "shadow", "foundation20"):
        ck = Path(root) / run / "checkpoints"
        for priority, name in enumerate(("final.pkl", "best.pkl", "latest.pkl")):
            path = ck / name
            if path.exists():
                candidates.append((priority, -path.stat().st_mtime, path))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None
