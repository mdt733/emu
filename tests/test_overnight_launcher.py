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



def test_fresh_start_requires_explicit_run_and_preserves_pinned_checkpoint(tmp_path):
    """No successful source may silently be replaced, and no --check trains."""
    text=SCRIPT.read_text(encoding="utf-8")
    assert 'if [[ -n "$EXPLICIT_FOUNDATION" ]]; then' in text
    assert 'STOP: explicitly requested checkpoint' in text
    check_at=text.index('if [[ "$MODE" == "--check" ]]; then')
    foundation_at=text.index('=== Training a new 312-input, flat-first 20M foundation')
    assert check_at < foundation_at
    assert 'emu train --profile foundation --forward-preview' in text
    assert '--steps 20000000' in text
    assert '--stop-after walk40' in text
    # The new foundation is neither a legacy 232-input warm start nor
    # the failed four-arm terrain curriculum in disguise.
    fresh_section=text[foundation_at:]
    assert 'emu campaign5020 --run' not in fresh_section
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE=false' in fresh_section
    assert 'choose_foundation([p])' in fresh_section
    assert 'FOUNDATION_PATH="$FRESH_ROOT/checkpoints/final.pkl"' in text
    assert 'git reset' not in text and 'git checkout main' not in text
