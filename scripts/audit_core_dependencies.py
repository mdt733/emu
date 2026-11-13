#!/usr/bin/env python3
"""Read-only static import and documentation-link inventory for EMU's core.

This is an evidence aid, NOT a delete-unused-code oracle. Dynamic imports,
CLI dispatch, JAX tracing and browser/Node imports require manual review.
Never loads a checkpoint, JAX runtime or model, and never alters files.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from urllib.parse import unquote
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Two dozen core test cases maximum. Keep high-value physical acceptance
# independent rather than growing a permanent test-per-experiment archive.
CORE_TEST_LIMIT = 24
# Three actively maintained docs, not historical notes per experiment.
# To create another permanent page, make an explicit ownership decision
# and update this list in review. Historical trials belong in Git.
CANONICAL_DOCS = frozenset({
    "ARCHITECTURE.md", "OPERATIONS.md", "VALIDATION.md",
})
MARKDOWN_LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')

# Keep this short on purpose: one explicit owner for each Python runtime
# module. The matching checklist is in docs/CORE_SKELETON.md.
# Adding a new engine module should prompt an ownership decision, not an
# automatic acceptance of another layer. This is not a function-level
# coverage assertion.
OWNED_BACKEND: dict[str, tuple[str, ...]] = {
    "commands": ("__init__", "cli", "run", "training_config"),
    "physics": ("contracts", "tof_geometry", "physical_variants", "environment", "terrain_carpet_mjx",
                "carpet_prism_mjx", "domain_randomization", "directional"),
    "learning": ("training", "ppo", "rewards", "curriculum",
                 "capability_focus", "capability_field", "training_health",
                 "step_goal"),
    "state_and_exams": ("checkpoint_io", "console_dashboard", "results", "evaluation", "prism_exam", "step_autopsy"),
    "delivery": ("policy_export", "viewer_server"),
    "diagnostics": ("doctor",),
}
RETIRED_BACKEND = frozenset((
    "capability_cloud", "physical_exam", "physical_quality",
    "step_academy_world", "terrain_carpet", "terrain_spike",
))

STAGES: dict[str, tuple[str, ...]] = {
    "run": ("cli", "run", "training_config", "checkpoint_io", "console_dashboard"),
    "physics": ("environment", "terrain_carpet_mjx", "carpet_prism_mjx",
                "contracts", "domain_randomization", "rewards"),
    "learning": ("training", "ppo", "curriculum", "capability_focus",
                 "training_health", "capability_field"),
    "exam": ("evaluation", "prism_exam", "results"),
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


def missing_emu_modules(tree: ast.AST, *, owner: str, available: set[str]) -> list[str]:
    """Detect explicit references to modules removed from the live EMU tree.

    This intentionally checks *module names*, not arbitrary exported symbols
    or dynamic imports. A deleted local module must never disappear from the
    inventory silently, even if it was mentioned only in a compatibility shim.
    """
    missing: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and owner == "emu":
                if node.module:
                    names.append(node.module.split(".", 1)[0])
                else:
                    names.extend(alias.name for alias in node.names)
            elif node.module and node.module.startswith("emu."):
                names.append(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[1] for alias in node.names
                         if alias.name.startswith("emu."))
        for name in names:
            if name not in available and name != "__init__":
                missing.add(name)
    return sorted(missing)


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
    doc_files = {
        p.relative_to(root / "docs").as_posix()
        for p in (root / "docs").rglob("*.md") if p.is_file()
    }
    doc_policy = {
        "max_pages": len(CANONICAL_DOCS),
        "pages": len(doc_files),
        "extra": sorted(doc_files - CANONICAL_DOCS),
        "missing": sorted(CANONICAL_DOCS - doc_files),
        "within_budget": doc_files == CANONICAL_DOCS,
    }
    modules = {path.stem: path for path in (root / "emu").glob("*.py")}
    local = set(modules)
    expected_list = [name for group in OWNED_BACKEND.values() for name in group]
    expected = set(expected_list)
    ownership_errors = {
        "unowned_backend": sorted(local - expected),
        "missing_owned_backend": sorted(expected - local),
        "retired_backend_returned": sorted(local & RETIRED_BACKEND),
        "duplicate_owners": sorted({
            name for name in expected_list if expected_list.count(name) != 1
        }),
    }
    edges: dict[str, list[str]] = {}
    lines: dict[str, int] = {}
    broken_imports: dict[str, list[str]] = {}
    for name, path in sorted(modules.items()):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        edges[name] = [m for m in local_module_imports(tree, local) if m != name]
        broken = missing_emu_modules(tree, owner="emu", available=local)
        if broken:
            broken_imports[path.relative_to(root).as_posix()] = broken
        lines[name] = len(source.splitlines())

    script_users: dict[str, list[str]] = {name: [] for name in local}
    test_cases: list[str] = []
    expanded_test_operators: list[str] = []
    for directory in ("scripts", "tests"):
        for path in (root / directory).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if directory == "tests":
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                        test_cases.append(f"{path.name}::{node.name}")
                    # A parametrized test or implicit unittest subTest
                    # would evade the plain test-method count. Fail closed
                    # until its generated cases are accounted explicitly.
                    if isinstance(node, ast.Attribute) and node.attr in ("parametrize", "subTest"):
                        expanded_test_operators.append(
                            f"{path.name}:{node.lineno}:{node.attr}")
            broken = missing_emu_modules(tree, owner=directory, available=local)
            if broken:
                broken_imports[path.relative_to(root).as_posix()] = broken
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
        "core_test_budget": {
            "limit": CORE_TEST_LIMIT,
            "cases": len(test_cases),
            "files": len(tuple((root / "tests").glob("test_*.py"))),
            "within_budget": len(test_cases) <= CORE_TEST_LIMIT and not expanded_test_operators,
            "untracked_expansions": sorted(expanded_test_operators),
            "names": sorted(test_cases),
        },
        "documentation_policy": doc_policy,
        "backend_module_count": len(modules),
        "backend_python_lines": sum(lines.values()),
        "browser_javascript_files": sum(
            path.suffix in (".js", ".mjs", ".ts", ".tsx")
            for path in (root / "emu" / "viewer_src").rglob("*") if path.is_file()
        ),
        "documentation_pages": len(doc_files),
        "modules": {
            name: {
                "lines": lines[name],
                "imports": edges[name],
                "script_or_test_users": sorted(script_users[name]),
            } for name in sorted(modules)
        },
        "stage_reachable": stage_reachability,
        "static_unlinked_manual_review_only": review_only,
        "missing_emu_module_imports": broken_imports,
        "broken_markdown_links": broken_markdown_links(root),
        "owned_backend_groups": {
            group: list(names) for group, names in OWNED_BACKEND.items()
        },
        "ownership_errors": ownership_errors,
        "limitations": (
            "Static Python imports only. Dynamic CLI dispatch, JAX tracing, "
            "browser modules, package exports and historical checkpoints "
            "are NOT proof of removable code."
        ),
    }


def broken_markdown_links(root: Path = REPO_ROOT) -> list[tuple[str,str]]:
    errors=[]
    for doc in sorted(root.rglob("*.md")):
        if ".venv" in doc.parts or "node_modules" in doc.parts or ".git" in doc.parts:
            continue
        source=doc.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(source):
            raw=match.group(1).strip().strip("<>")
            if not raw or raw.startswith("#") or re.match(
                    r"^[A-Za-z][\w+.-]*:",raw):
                continue
            target=unquote(raw.split("#",1)[0].split("?",1)[0])
            if not target:
                continue
            file=(doc.parent/target).resolve()
            if not file.is_file():
                errors.append((
                    doc.relative_to(root).as_posix(),raw))
    return errors



def markdown(data: dict) -> str:
    out = [
        "# EMU static dependency inventory (not a deletion list)", "",
        f"Backend modules: {data['backend_module_count']}; "
        f"backend Python lines: {data['backend_python_lines']:,}; "
        f"browser JS/TS files: {data['browser_javascript_files']}; "
        f"docs pages: {data['documentation_pages']}.",
        "",
        "Test budget: "
        f"{data['core_test_budget']['cases']}/{data['core_test_budget']['limit']} "
        f"cases in {data['core_test_budget']['files']} files; "
        f"untracked expanded cases: {len(data['core_test_budget']['untracked_expansions'])}.",
        "",
        "## Documentation ownership (must match the three canonical pages)", "",
        json.dumps(data["documentation_policy"], sort_keys=True),
        "", "## Stages", "",
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
    out.extend(["", "## Skeleton ownership mismatches (must all be empty)", "",
                json.dumps(data["ownership_errors"], sort_keys=True),
                "", "## Deleted-module imports (must be empty)", "",
                json.dumps(data["missing_emu_module_imports"], sort_keys=True),
                "", "## Broken local Markdown links (must be empty)", "",
                json.dumps(data["broken_markdown_links"], sort_keys=True),
                "", "## Manual-review candidates only", "",
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
    return int(bool(data["missing_emu_module_imports"]) or
               bool(data["broken_markdown_links"]) or
               any(data["ownership_errors"].values()) or
               not data["documentation_policy"]["within_budget"] or
               not data["core_test_budget"]["within_budget"])


if __name__ == "__main__":
    raise SystemExit(main())
