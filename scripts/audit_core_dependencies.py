#!/usr/bin/env python3
"""Read-only static import inventory for EMU's five-stage core skeleton.

This is an evidence aid, NOT a delete-unused-code oracle. Dynamic imports,
CLI dispatch, JAX tracing and browser/Node imports require manual review.
Never loads a checkpoint, JAX runtime or model, and never alters files.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES: dict[str, tuple[str, ...]] = {
    "run": ("cli", "run", "training_config", "checkpoint_io"),
    "physics": ("environment", "terrain_carpet_mjx", "carpet_prism_mjx",
                "contracts", "domain_randomization", "rewards"),
    "learning": ("training", "ppo", "curriculum", "capability_focus",
                 "training_health", "capability_field"),
    "exam": ("evaluation", "physical_exam", "physical_quality",
             "capability_cloud", "results"),
    "viewer": ("viewer_server",),
}


def local_module_imports(tree: ast.AST, local_modules: set[str]) -> list[str]:
    """Find relative and absolute imports of local emu modules in Python AST."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                # Normal emu top-level package module: from .ppo import X.
                name = node.module.split(".", 1)[0]
                if name in local_modules:
                    found.add(name)
            elif node.module and node.module.startswith("emu."):
                name = node.module.split(".")[1]
                if name in local_modules:
                    found.add(name)
            elif node.module == "emu":
                found.update(a.name for a in node.names if a.name in local_modules)
            elif node.level and not node.module:
                found.update(a.name for a in node.names if a.name in local_modules)
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name.startswith("emu."):
                    name = item.name.split(".")[1]
                    if name in local_modules:
                        found.add(name)
    return sorted(found)


def reachable(graph: dict[str, list[str]], roots: tuple[str, ...]) -> list[str]:
    visited: set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current not in graph or current in visited:
            continue
        visited.add(current)
        pending.extend(graph[current])
    return sorted(visited)


def inventory(root: Path = REPO_ROOT) -> dict:
    modules = {path.stem: path for path in (root / "emu").glob("*.py")}
    local = set(modules)
    edges: dict[str, list[str]] = {}
    lines: dict[str, int] = {}
    for name, path in sorted(modules.items()):
        source = path.read_text(encoding="utf-8")
        edges[name] = [m for m in local_module_imports(
            ast.parse(source, filename=str(path)), local) if m != name]
        lines[name] = len(source.splitlines())

    script_users: dict[str, list[str]] = {name: [] for name in local}
    for directory in ("scripts", "tests"):
        for path in (root / directory).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in local_module_imports(tree, local):
                script_users[name].append(path.relative_to(root).as_posix())
    stage_reachability = {
        name: reachable(edges, roots) for name, roots in STAGES.items()
    }
    source_used = set().union(*(set(x) for x in stage_reachability.values()))
    test_or_script_used = {name for name, users in script_users.items() if users}
    # IMPORTANT: these are *review candidates*, NOT proven dead modules.
    review_only = sorted(local - source_used - test_or_script_used)
    return {
        "backend_module_count": len(modules),
        "backend_python_lines": sum(lines.values()),
        "browser_javascript_files": sum(
            path.suffix in (".js", ".mjs", ".ts", ".tsx")
            for path in (root / "emu" / "viewer_src").rglob("*") if path.is_file()
        ),
        "documentation_pages": sum(
            1 for path in (root / "docs").rglob("*.md") if path.is_file()
        ),
        "modules": {
            name: {
                "lines": lines[name],
                "imports": edges[name],
                "script_or_test_users": sorted(script_users[name]),
            } for name in sorted(modules)
        },
        "stage_reachable": stage_reachability,
        "static_unlinked_manual_review_only": review_only,
        "limitations": (
            "Static Python imports only. Dynamic CLI dispatch, JAX tracing, "
            "browser modules, package exports and historical checkpoints "
            "are NOT proof of removable code."
        ),
    }


def markdown(data: dict) -> str:
    out = [
        "# EMU static dependency inventory (not a deletion list)", "",
        f"Backend modules: {data['backend_module_count']}; "
        f"backend Python lines: {data['backend_python_lines']:,}; "
        f"browser JS/TS files: {data['browser_javascript_files']}; "
        f"docs pages: {data['documentation_pages']}.",
        "",
        "## Stages", "",
    ]
    for name, modules in data["stage_reachable"].items():
        out.append(f"- **{name}**: {len(modules)} statically connected modules "
                   f"({', '.join(modules)}).")
    out.extend(["", "## Individual backend modules", "",
                "| Module | Lines | Direct internal imports | Script/test references |",
                "|---|---:|---|---|"])
    for name, detail in data["modules"].items():
        out.append(f"| `{name}` | {detail['lines']} | "
                   f"{', '.join(detail['imports']) or '—'} | "
                   f"{len(detail['script_or_test_users'])} |")
    out.extend(["", "## Manual-review candidates only", "",
                ", ".join(data["static_unlinked_manual_review_only"]) or "None.",
                "", "**Caveat:** " + data["limitations"]])
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON for diffing")
    args = parser.parse_args(argv)
    data = inventory()
    print(json.dumps(data, sort_keys=True, indent=2) if args.json
          else markdown(data), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
