"""The overnight entry point must be safe to inspect without a GPU."""
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"overnight_v5020.sh"


def test_overnight_launcher_parses_without_expanding_a_source_checkout():
    result=subprocess.run(["bash","-n",str(SCRIPT)],
                          cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_overnight_launcher_help_does_not_launch_training_or_require_a_venv(tmp_path):
    result=subprocess.run(["bash",str(SCRIPT),"--help"],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "--check" in result.stdout
    assert "--run" in result.stdout
    assert "foundation" in result.stdout.lower()
    assert "40M" in result.stdout
    assert not list(tmp_path.iterdir())


def test_overnight_launcher_rejects_unknown_mode_without_training(tmp_path):
    result=subprocess.run(["bash",str(SCRIPT),"--launch-everything"],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2
    assert not list(tmp_path.iterdir())
