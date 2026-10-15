"""CPU-only contract for the documented tiny GPU acceptance recipe."""
from dataclasses import asdict
from pathlib import Path
import importlib.util

from emu.training_config import ROLLOUT_STEPS


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_training_lifecycle.py"
    spec = importlib.util.spec_from_file_location("emu_smoke_training_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_lifecycle_uses_only_two_ppo_updates_and_a_distinct_resume_target(tmp_path):
    smoke = _module()
    first, second = smoke.configs(tmp_path / "acceptance")
    assert smoke.UPDATE_STEPS == 8 * ROLLOUT_STEPS
    assert first.envs == second.envs == 8
    assert first.steps == smoke.UPDATE_STEPS
    assert second.steps == 2 * smoke.UPDATE_STEPS
    assert first.device == second.device == "cuda"
    assert first.attitude_profile == second.attitude_profile == "v5015"
    assert first.cloud_final_probes == second.cloud_final_probes == 0
    assert first.cloud_probes_per_eval == second.cloud_probes_per_eval == 0
    assert Path(first.run_dir).parent == Path(second.run_dir).parent
    assert Path(first.run_dir) != Path(second.run_dir)
    assert Path(second.resume) == Path(first.run_dir) / "checkpoints" / "final.pkl"
    changed = {k for k, v in asdict(first).items() if asdict(second)[k] != v}
    assert changed == {"steps", "resume", "run_dir"}


def test_acceptance_root_cannot_write_inside_checkout():
    import pytest
    from emu.paths import source_repo_root
    smoke = _module()
    with pytest.raises(ValueError, match="OUTSIDE"):
        smoke.protected_root(str(source_repo_root() / "acceptance"))



def test_dirty_source_gate_explains_exact_changed_files(monkeypatch):
    import pytest
    from types import SimpleNamespace
    smoke = _module()
    monkeypatch.setattr(smoke, "git_provenance", lambda: {
        "available": True, "commit": "abc123", "dirty": True,
    })
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=" M emu/viewer_src/README.md\n"))
    with pytest.raises(RuntimeError, match="emu/viewer_src/README.md"):
        smoke.current_source()
