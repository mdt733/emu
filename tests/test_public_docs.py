"""Keep the project's public-facing Markdown navigation honest."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    'README.md', 'CONTRIBUTING.md', 'viewer/README.md',
    'docs/README.md', 'docs/STATUS.md', 'docs/INSTALL.md',
    'docs/validation.md', 'docs/training.md',
    'docs/evaluation.md', 'docs/PLAYGROUND.md', 'docs/architecture.md',
    'docs/tournament.md',
)


def test_public_document_links_resolve():
    errors = []
    for relative in PUBLIC_DOCS:
        origin = ROOT / relative
        assert origin.is_file()
        contents = origin.read_text(encoding='utf-8')
        # A link with no leading slash or URI scheme is repository-local.
        for destination in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', contents):
            url = destination.split('#', 1)[0].strip()
            if not url or url.startswith(('http://','https://','mailto:','/')):
                continue
            if '://' in url:
                continue
            target = (origin.parent / unquote(url)).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                errors.append(f'{relative}: {destination}')
    assert not errors, 'Broken public documentation links:\n' + '\n'.join(errors)
