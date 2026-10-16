"""Process/runtime support shared by training and offline evaluators."""
from __future__ import annotations

import os

from .contracts import VERSION

LOG_PREFIX = f'[emu_{VERSION}]'

class PerfBook:
    """Low-overhead wall-clock accounting for trainer performance.

    Timing is deliberately explicit and coarse.  The purpose is to identify
    whether wall time is being spent in compiled training, host-side curriculum
    bookkeeping, held-out evaluation, checkpoint device->host transfer, or disk
    I/O before attempting more invasive JAX changes.
    """
    def __init__(self, enabled: bool = False, known_names: tuple[str, ...] = ()):
        self.enabled = bool(enabled)
        self.total: dict[str, float] = {}
        self.count: dict[str, int] = {}
        self.last: dict[str, float] = {}
        self.register(*known_names)

    def register(self, *names: str) -> None:
        """Declare timer names without pretending they have fired.

        Metrics CSV has a strict schema, so event-driven timers (checkpoint,
        evaluation, cloud, log I/O) must exist as zero-valued columns from the
        first row instead of appearing only after their first event.
        """
        if not self.enabled:
            return
        for name in names:
            self.total.setdefault(str(name), 0.0)
            self.count.setdefault(str(name), 0)
            self.last.setdefault(str(name), 0.0)

    def add(self, name: str, seconds: float) -> float:
        x = float(seconds)
        if self.enabled:
            self.total[name] = self.total.get(name, 0.0) + x
            self.count[name] = self.count.get(name, 0) + 1
            self.last[name] = x
        return x

    def row(self, prefix: str = 'perf_') -> dict[str, float]:
        if not self.enabled:
            return {}
        out: dict[str, float] = {}
        for name, total in sorted(self.total.items()):
            n = max(1, self.count.get(name, 0))
            out[f'{prefix}{name}_total_s'] = float(total)
            out[f'{prefix}{name}_mean_s'] = float(total / n)
            out[f'{prefix}{name}_last_s'] = float(self.last.get(name, 0.0))
            out[f'{prefix}{name}_count'] = float(self.count.get(name, 0))
        return out


def configure_jax(device: str) -> None:
    if device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    elif device in ('cuda', 'gpu'):
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    # device=auto: don't set anything here -- but that does NOT mean
    # JAX_PLATFORM_NAME is unset. It may already be set in the calling
    # shell/environment for reasons unrelated to this process's own
    # --device flag, so the check below looks at the actual environment
    # state, not just what this function itself just set, and acts on it
    # regardless of how it got there.
    if os.environ.get('JAX_PLATFORM_NAME', '').lower() == 'gpu':
        # Forcing JAX_PLATFORM_NAME=gpu when no CUDA jaxlib is installed makes
        # jax.default_backend() raise a hard RuntimeError on first use instead
        # of gracefully falling back to cpu the way it does when nothing
        # forces a platform. And there's no recovering after the fact: jax
        # reads/caches the platform preference at *import* time, so popping
        # the env var or calling jax.clear_backends() after `import jax` has
        # already run doesn't help. So: remove the forcing *before* the
        # probe import, not after.
        forced = os.environ.pop('JAX_PLATFORM_NAME')
        import jax as _probe_jax
        real_backend = _probe_jax.default_backend()
        if real_backend == 'gpu':
            os.environ['JAX_PLATFORM_NAME'] = forced
        else:
            print(f"{LOG_PREFIX} GPU was requested (device={device!r}, or via an externally-set "
                  f"JAX_PLATFORM_NAME) but no CUDA-enabled jaxlib is installed (jax resolves to "
                  f"{real_backend!r} on its own) — continuing without forcing gpu, so it falls "
                  f"back to cpu instead of crashing. --allow-cpu-mjx still gates whether that's allowed.")


def require_backend(allow_cpu_mjx: bool) -> None:
    import jax
    backend = jax.default_backend()
    if backend != 'gpu' and not allow_cpu_mjx:
        raise SystemExit(f'{LOG_PREFIX} JAX backend is {backend!r}, not gpu. Refusing real MJX training. Use --allow-cpu-mjx only for smoke/debug.')


