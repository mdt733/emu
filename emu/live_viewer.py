"""Read-only browser feed of the newest *completed* production checkpoint.

This module never interacts with a running trainer, never changes PPO state, and
only exports model snapshots that pass the strict checkpoint contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock

from .production import PHASES


@dataclass(frozen=True)
class Snapshot:
    path: Path
    signature: tuple[str, int, int]
    phase: str


def candidates(root: Path) -> list[tuple[str, Path]]:
    """Newest phase first; latest snapshot before final for active phases."""
    root = Path(root)
    result = []
    for phase, *_ in reversed(PHASES):
        ck = root / phase / 'checkpoints'
        result.extend([(phase, ck / 'latest.pkl'), (phase, ck / 'final.pkl')])
        result.extend((phase, p) for p in sorted(ck.glob('snapshot_step_*.pkl'), reverse=True))
    result.append(('complete', root / 'final.pkl'))
    # Also support a normal emu train/compare run directory.
    ck = root / 'checkpoints'
    result.extend([('training', ck / 'latest.pkl'), ('training', ck / 'final.pkl')])
    result.extend(('training', p) for p in sorted(ck.glob('snapshot_step_*.pkl'), reverse=True))
    return result


def inspect_candidate(phase: str, path: Path) -> Snapshot | None:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size < 1024:
            return None
        return Snapshot(path, (str(path.resolve()), stat.st_size, stat.st_mtime_ns), phase)
    except OSError:
        return None


class LivePolicyWatcher:
    """Cache policy JSON by completed-checkpoint signature, not by request."""
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self._lock = Lock()
        self._signature: tuple | None = None
        self._text: str | None = None
        self._revision: str | None = None
        self._phase: str | None = None
        self._step: int | None = None
        self._error = ''

    def _refresh(self) -> None:
        # Avoid pickle loads on each browser poll. Failed writes can be retried
        # next poll; last validated actor stays available throughout.
        with self._lock:
            seen = set()
            for phase, path in candidates(self.root):
                snap = inspect_candidate(phase, path)
                if snap is None or snap.signature in seen:
                    continue
                seen.add(snap.signature)
                if snap.signature == self._signature:
                    self._error = ''
                    return
                try:
                    from .policy_export import build_bundle
                    bundle = build_bundle(path)
                    # Ensure a writer didn't replace the file while exporting.
                    if inspect_candidate(phase, path) != snap:
                        self._error = 'Checkpoint changed while being read; retrying.'
                        return
                    bundle.setdefault('source', {})['phase'] = phase
                    body = json.dumps(bundle, separators=(',', ':'))
                    source = bundle['source']
                    self._signature = snap.signature
                    self._text = body
                    self._phase = phase
                    self._step = int(source['step'])
                    self._revision = f"{source['step']}:{source['actor_sha256']}"
                    self._error = ''
                    return
                except Exception as exc:
                    self._error = f'Newest candidate not usable yet ({type(exc).__name__}); retaining last valid policy.'
                    # Do not serve an older phase as new while a newer checkpoint
                    # is being written, but allow fallback at initial startup.
                    if self._text is not None:
                        return
            if self._text is None and not self._error:
                self._error = 'No validated checkpoints yet. Viewer remains ready.'

    def status(self) -> dict:
        self._refresh()
        return {'enabled': True, 'available': self._text is not None,
                'revision': self._revision, 'step': self._step,
                'phase': self._phase, 'note': self._error}

    def policy(self) -> str | None:
        self._refresh()
        return self._text
