"""Start a read-only ETA companion for an old, already-running controller.

The old controller serves a cached HTML document and cannot hot-reload its
Python API. Do not kill or replace it, edit its checkout, or acquire its
control/GPU lock. Only a runner inside a Studio-created detached checkout
may start this companion. Direct CLI jobs and ETA-aware controllers skip it.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys


def companion_command(checkout, env):
    checkout=Path(checkout).resolve()
    if env.get("EMU_STUDIO_UI_REV")=="eta-v1":
        return None
    if (checkout.parent.name!="checkouts" or len(checkout.name)!=40 or
            any(c not in "0123456789abcdef" for c in checkout.name)):
        return None
    state=checkout.parent.parent
    if not (state/"selection.json").is_file() or not (state/"gpu.lock").is_file():
        return None
    common=Path(subprocess.check_output(
        ["git","rev-parse","--path-format=absolute","--git-common-dir"],
        cwd=checkout,text=True).strip())
    if common.name!=".git":
        return None
    return [sys.executable,"-m","scripts.studio","--repo",str(common.parent),
            "--state",str(state),"--monitor","--port","8772"]


@contextmanager
def legacy_eta_monitor():
    proc=None
    try:
        checkout=Path(__file__).resolve().parents[1]
        cmd=companion_command(checkout,os.environ)
        if cmd:
            # close_fds prevents inheriting the runner's GPU guard.
            # Same process group: the original Stop action still works.
            proc=subprocess.Popen(cmd,cwd=checkout,close_fds=True)
            print("STUDIO ETA companion starting: http://localhost:8772/ "
                  "(read-only; original controls/results on 8770)",flush=True)
    except (OSError,subprocess.SubprocessError) as e:
        print("STUDIO ETA companion unavailable:",e,flush=True)
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except ProcessLookupError:
                pass
