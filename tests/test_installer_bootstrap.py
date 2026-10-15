"""Regression for first-install nvm bootstrap; no real Node download is needed."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_nvm_first_install_does_not_auto_use_missing_node(tmp_path):
    nvm_dir = tmp_path / ".nvm"
    nvm_dir.mkdir()
    (nvm_dir / "nvm.sh").write_text(
        'if [[ "$1" != "--no-use" ]]; then return 3; fi\n'
        'nvm() {\n'
        '  if [[ "$1" == "install" && "$2" == "24" ]]; then\n'
        '    printf "REACHED_NODE_INSTALL\\n"\n'
        '    exit 66\n'
        '  fi\n'
        '  exit 65\n'
        '}\n',
        encoding="utf-8",
    )
    env = dict(os.environ, HOME=str(tmp_path))
    result = subprocess.run(
        ["bash", "scripts/setup_wsl.sh", "--install-node"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 66, result.stdout + result.stderr
    assert "REACHED_NODE_INSTALL" in result.stdout
