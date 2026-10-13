"""Deploy protection must never be removed accidentally with a main-branch merge."""
from pathlib import Path
import subprocess


def test_github_pages_is_manual_and_pinned_to_protected_champion():
    root = Path(__file__).resolve().parents[1]
    workflow = (root/'.github/workflows/emu-playground.yml').read_text()
    assert 'workflow_dispatch:' in workflow
    assert '\n  push:' not in workflow
    assert '--require-champion' in workflow
    champion = root/'showcase/champion.emupolicy'
    assert champion.is_file()
    digest = subprocess.check_output(['git','hash-object',str(champion)],text=True).strip()
    assert digest == '4743d8d3cd4706edce04809fe1b61df014e57d5e'
    assert digest in workflow
