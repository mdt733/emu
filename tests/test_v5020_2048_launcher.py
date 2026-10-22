"""The full-scale launcher must remain inspectable without a GPU or training."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'run_v5020_2048.sh'


def test_2048_launcher_is_valid_shell_and_help_has_no_side_effects(tmp_path):
    syntax=subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True,text=True)
    assert syntax.returncode==0,syntax.stderr
    help_output=subprocess.run(['bash',str(SCRIPT),'--help'],
                               cwd=tmp_path,capture_output=True,text=True)
    assert help_output.returncode==0,help_output.stderr
    for needle in ('2048','20M','32M','48M','70M','up','mixed'):
        assert needle in help_output.stdout
    assert not list(tmp_path.iterdir())


def test_2048_launcher_does_not_silently_downscale_or_overwrite_main():
    source=SCRIPT.read_text()
    assert 'ENVS=2048' in source
    assert 'ROBOT' not in source  # no unexpected physical XML rewrite
    assert 'git branch --show-current' in source
    assert 'feature/v5020-coldstart-terrain' in source
    assert 'git status --porcelain' in source
    assert 'source_commit.txt' in source
    assert 'parent_sha' in source
    assert 'sha256sum' in source
    assert 'choose_fresh_foundation' in source
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE=false' in source
    assert 'emu train --run-dir' in source
    assert '--resume auto' in source or 'resume=(--resume auto)' in source
    for name in ('foundation','up','mixed12','mixed20'):
        assert f'step {name} ' in source
    assert 'rm -rf' not in source
    assert 'git reset' not in source
    assert 'git checkout main' not in source


def test_2048_launcher_rejects_bad_arguments_without_running(tmp_path):
    for argv in (['--launch'],['--run','--stop-after','nope'],
                 ['--run','--stop-after'],['--run','--stop-after','up','extra']):
        result=subprocess.run(['bash',str(SCRIPT),*argv],cwd=tmp_path,
                              capture_output=True,text=True)
        assert result.returncode != 0, (argv,result.stdout,result.stderr)
    assert not list(tmp_path.iterdir())


def test_overnight_main_gate_not_silently_removed():
    overnight=(ROOT/'scripts'/'overnight_v5020.sh').read_text()
    assert 'STOP: v5020 terrain acceptance pending' in overnight
