#!/usr/bin/env python3
"""Fail CI when a maintained Markdown link targets a deleted local file.

Old histories are intentionally in immutable Git, so references to
them must use their actual Git commit URL rather than a broken path.
Pure standard-library, no GitHub/network access, no file modifications.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote

REPO_ROOT=Path(__file__).resolve().parents[1]
MARKDOWN_LINK=re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')


def broken_links(root: Path = REPO_ROOT) -> list[tuple[str,str]]:
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


def main()->int:
    problems=broken_links()
    for doc,target in problems:
        print(f"BROKEN DOC LINK: {doc} -> {target}",file=sys.stderr)
    if problems:
        return 1
    print("PASS current Markdown links point to existing local files")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
