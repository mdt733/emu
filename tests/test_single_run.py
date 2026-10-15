"""CPU-only safety checks for the single-run CLI, not an MJX training proof."""
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import pytest

from emu import single_run as run


def options(tmp_path, **overrides):
    defaults = dict(run_dir=str(tmp_path / 'new'), steps=256, envs=8, seed=733,
                    device='cuda', profile='foundation', resume='', plan=False)
    return Namespace(**{**defaults, **overrides})


def identity():
    return {'available': True, 'dirty': False, 'commit': 'a' * 40}


def test_plan_is_pure_and_resolves_foundation_profile(tmp_path, monkeypatch, capsys):
    def never(*args, **kwargs):
        raise AssertionError('planning must not request provenance, checkpoints, GPU or training')
    monkeypatch.setattr(run, 'git_provenance', never)
    monkeypatch.setattr(run, 'resolve_target_checkpoint', never)
    monkeypatch.setattr(run, 'run_training', never)
    args = options(tmp_path, plan=True)
    assert run.train_one(args) == 0
    assert not Path(args.run_dir).exists()
    out = capsys.readouterr().out
    assert '"capability_field_mode": "shadow"' in out
    assert '"attitude_profile": "v5015"' in out
    assert '256 requested steps' in out


def test_invalid_steps_or_world_count_never_build_a_run(tmp_path):
    for invalid in (options(tmp_path, steps=0), options(tmp_path, envs=-1)):
        with pytest.raises(ValueError, match='positive'):
            run.one_run_config(invalid)


def test_rejects_dirty_or_missing_provenance_and_paths_inside_source(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'checkout')
    cfg = run.one_run_config(options(tmp_path))
    for info in ({'available': False}, {**identity(), 'dirty': True}):
        with pytest.raises(RuntimeError, match='clean Git'):
            run.inspect_run(cfg, provenance=info, resume='')
    inside = replace(cfg, run_dir=str(tmp_path / 'checkout' / 'runs' / 'bad'))
    with pytest.raises(RuntimeError, match='OUTSIDE'):
        run.inspect_run(inside, provenance=identity(), resume='')


def test_existing_files_without_config_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'repo')
    cfg = run.one_run_config(options(tmp_path))
    root = Path(cfg.run_dir)
    root.mkdir()
    (root / 'precious.pkl').write_bytes(b'keep me')
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        run.inspect_run(cfg, provenance=identity(), resume='')
    assert (root / 'precious.pkl').read_bytes() == b'keep me'


def test_existing_config_source_and_hyperparameter_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'repo')
    cfg = run.one_run_config(options(tmp_path))
    root = Path(cfg.run_dir)
    cfg.save(root / 'training_config.json')
    (root / 'run_meta.json').write_text(json.dumps({'source_provenance': {'git_commit': 'old'}}))
    with pytest.raises(RuntimeError, match='Source commit differs'):
        run.inspect_run(cfg, provenance=identity(), resume='')
    (root / 'run_meta.json').write_text(json.dumps({'source_provenance': {'git_commit': identity()['commit']}}))
    with pytest.raises(RuntimeError, match='configuration differs'):
        run.inspect_run(replace(cfg, seed=77), provenance=identity(), resume='')


def test_partial_checkpoint_needs_explicit_resume_auto(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'repo')
    cfg = run.one_run_config(options(tmp_path))
    root = Path(cfg.run_dir)
    cfg.save(root / 'training_config.json')
    (root / 'run_meta.json').write_text(json.dumps({'source_provenance': {'git_commit': identity()['commit']}}))
    partial = root / 'checkpoints/latest.pkl'
    monkeypatch.setattr(run, 'verify_existing_phase_config', lambda *_: None)
    monkeypatch.setattr(run, 'resolve_target_checkpoint', lambda *a, **kw: (None, partial, 256))
    with pytest.raises(RuntimeError, match='--resume auto'):
        run.inspect_run(cfg, provenance=identity(), resume='')
    assert run.inspect_run(cfg, provenance=identity(), resume='auto') == (partial, False)


def test_train_one_uses_existing_runner_without_any_other_training_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'git_provenance', lambda: identity())
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'repo')
    calls = []
    finished = tmp_path / 'new/checkpoints/final.pkl'
    monkeypatch.setattr(run, 'resolve_target_checkpoint', lambda *a, **kw: (
        (finished, None, 256) if calls else (None, None, 256)))
    def runner(config, *, provenance):
        calls.append((config, provenance))
    monkeypatch.setattr(run, 'run_training', runner)
    assert run.train_one(options(tmp_path)) == 0
    assert len(calls) == 1
    assert calls[0][0].steps == 256
    assert calls[0][0].resume == ''


def test_public_emu_train_command_parses_plan_without_creating_data(tmp_path, capsys):
    from emu.cli import main
    root = tmp_path / 'cli_trial'
    assert main(['train', '--run-dir', str(root), '--steps', '256',
                 '--envs', '8', '--profile', 'foundation', '--plan']) == 0
    assert not root.exists()
    assert 'Planning only' in capsys.readouterr().out


def test_mature_profile_is_explicit_and_uses_existing_graph_config(tmp_path):
    config = run.one_run_config(options(tmp_path, profile='mature', steps=30_000_000))
    assert config.capability_field_mode == 'graph'
    assert config.attitude_profile == 'v5015'
    assert config.steps == 30_000_000


def test_pre_cuda_abort_allows_safe_retry_but_not_ambiguous_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_repo_root', lambda: tmp_path / 'source')
    cfg = run.one_run_config(options(tmp_path))
    root = Path(cfg.run_dir)
    cfg.save(root / 'training_config.json')
    (root / 'console.log').write_text('GPU not found; no trainer was created')
    monkeypatch.setattr(run, 'resolve_target_checkpoint',
                        lambda *a, **kw: (None, None, 256))
    assert run.inspect_run(cfg, provenance=identity(), resume='') == (None, False)
    (root / 'orphan.pkl').write_bytes(b'preserve unknown evidence')
    with pytest.raises(RuntimeError, match='ambiguous lineage'):
        run.inspect_run(cfg, provenance=identity(), resume='')
    assert (root / 'orphan.pkl').read_bytes() == b'preserve unknown evidence'
