"""The core audit script must work without importing MJX or acting on any run."""
from pathlib import Path

from scripts.audit_core_dependencies import (
    REPO_ROOT, STAGES, inventory, local_module_imports, reachable
)
import ast


def test_static_ast_detects_relative_and_absolute_imports():
    tree = ast.parse(
        "from .terrain_carpet_mjx import make_profiles\n"
        "from emu.environment import EmuMJXEnv\n"
        "import emu.ppo\n"
        "from . import contracts\n"
    )
    assert local_module_imports(tree, {
        "terrain_carpet_mjx", "environment", "ppo", "contracts",
    }) == ["contracts", "environment", "ppo", "terrain_carpet_mjx"]


def test_reachable_handles_cycles_and_ignores_unknown_modules():
    assert reachable({"a":["b"],"b":["a","missing"],"c":[]},
                     ("a",)) == ["a","b"]


def test_inventory_preserves_actual_train_to_physics_and_actor_route():
    report = inventory(REPO_ROOT)
    assert report["backend_module_count"] >= 24
    assert report["backend_python_lines"] >= 15000
    assert report["documentation_pages"] >= 20
    assert set(STAGES) == {"run","physics","learning","exam","viewer"}
    assert "environment" in report["stage_reachable"]["learning"]
    assert "ppo" in report["stage_reachable"]["learning"]
    assert "terrain_carpet_mjx" in report["stage_reachable"]["physics"]
    assert "carpet_prism_mjx" in report["stage_reachable"]["physics"]
    assert "viewer_server" in report["stage_reachable"]["viewer"]
    assert "directional" in report["modules"]["ppo"]["imports"]
    # Static non-reachability is deliberately never called "dead".
    assert "static_unlinked_manual_review_only" in report
