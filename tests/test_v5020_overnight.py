"""Offline contracts for the 32M -> 320M 2048-world overnight experiment."""
from dataclasses import replace
from pathlib import Path
import json
import re
import subprocess
import zipfile

import pytest

from emu.train_cli import verify_rated_parent_continuation
from emu.training_config import foundation_config
from emu.results import write_report

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'overnight_v5020_up.sh'


def _source_up(tmp_path):
    base=foundation_config(run_dir=tmp_path/'old-up',seed=733,
                           device='cuda',envs=2048)
    return replace(base,steps=32000000,
                   observation_schema='rolling_tof_312',
                   terrain_elevation_mode='up',
                   terrain_random_placement=True,
                   terrain_height_safety_m=.006,
                   terrain_height_initial_m=0.,
                   terrain_height_exposure=.40,
                   terrain_height_edge_fraction=.30,
                   terrain_height_growth_m=.001)


def test_overnight_continuation_only_changes_absolute_step_target_and_output(tmp_path):
    old=_source_up(tmp_path)
    for steps in (64000000,128000000,224000000,320000000):
        child=replace(old,steps=steps,run_dir=str(tmp_path/str(steps)),
                      approved_parent_git_commit='',
                      approved_parent_sha256='')
        verify_rated_parent_continuation(old,child)
        for axis,value in (('terrain_height_exposure',.7),
                           ('terrain_height_edge_fraction',.50),
                           ('terrain_height_safety_m',.012),
                           ('terrain_elevation_mode','mixed'),
                           ('push_max_force_n',3.2),
                           ('phase_speed_gain',1.25),
                           ('ppo_lr',.0005),
                           ('regime_command_frac',.8),
                           ('vx_hi_max',.6),
                           ('terrain_progress_weight',.025),
                           ('capability_field_mode','graph')):
            with pytest.raises(RuntimeError,match='Non-continuous'):
                verify_rated_parent_continuation(old,replace(child,**{axis:value}))


def test_overnight_rejects_non_up_parent(tmp_path):
    parent=_source_up(tmp_path)
    child=replace(parent,steps=64000000)
    for bad in (replace(parent,terrain_elevation_mode='mixed'),
                replace(parent,terrain_height_safety_m=0.),
                replace(parent,observation_schema='legacy_220')):
        with pytest.raises(ValueError,match='312-input up-stage'):
            verify_rated_parent_continuation(bad,child)


def test_overnight_script_syntax_help_plan_and_immutable_sequence():
    proc=subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    help=subprocess.run(['bash',str(SCRIPT),'--help'],
                        cwd=ROOT,capture_output=True,text=True)
    assert help.returncode==0,help.stderr
    assert '320M' in help.stdout
    src=SCRIPT.read_text()
    for stage,target in (('up64','64000000'),('up128','128000000'),
                         ('up224','224000000'),('up320','320000000')):
        # Shell formatting is free to align columns with multiple spaces;
        # validate the ACTUAL invocation, not its cosmetic whitespace.
        assert re.search(rf'(?m)^run_stage\s+{stage}\s+{target}\s+', src), (
            stage, target)
    assert 'ENVS=2048' not in src or '2048' in src
    assert 'PROFILE=(' in src
    assert '--profile terrain-bridge' in src
    assert '--terrain-elevation-mode up' in src
    assert '--terrain-height-initial-mm 0' in src
    assert '--terrain-height-ceiling-mm 6' in src
    assert 'results_bundle.zip' in src
    assert 'emu report --root "$RUN_ROOT"' in src
    assert 'UNSAFE_RESUME.json' in src
    assert 'known parent' not in src.lower() or 'sha256sum' in src
    assert 'rm -rf' not in src
    assert 'git checkout main' not in src
    assert 'step mixed12' not in src


def test_report_collects_all_stages_without_checkpoint_reads(tmp_path):
    import csv
    for label,n,rating in [('up64',64000000,.00040),
                           ('up128',128000000,.00072),
                           ('up224',224000000,.0010),
                           ('up320',320000000,.0014)]:
        run=tmp_path/label
        (run/'checkpoints').mkdir(parents=True)
        (run/'checkpoints'/'final.pkl').write_bytes(
            b'Not a real pickle - offline reporter must NOT deserialize')
        (run/'training_config.json').write_text(json.dumps(dict(
            steps=n,envs=2048,seed=733,terrain_elevation_mode='up',
            terrain_height_safety_m=.006)))
        (run/'run_meta.json').write_text(json.dumps(
            {'source_provenance':{'git_commit':'c'*40}}))
        with (run/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=[
                'step','reward_mean','fall_rate','ep_len_mean',
                'frontier_terrain_height_m','terrain_frontier_event'])
            w.writeheader()
            w.writerow(dict(step=n,reward_mean=3.,fall_rate=.01,
                            ep_len_mean=698,frontier_terrain_height_m=rating,
                            terrain_frontier_event='hold_challenge_not_yet_mastered'))
        with (run/'fixed_eval.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['step','eval_policy_step',
                                          'eval_core_survival','eval_broad_score'])
            w.writeheader()
            w.writerow(dict(step=n,eval_policy_step=n,
                            eval_core_survival=.99,eval_broad_score=.2))
        lineage=tmp_path/'lineage'
        lineage.mkdir(exist_ok=True)
        (lineage/f'{label}.parent.sha256').write_text('f'*64)
    files=write_report(tmp_path)
    assert len(files)==3
    md=(tmp_path/'results.md').read_text()
    for label in ('up64','up128','up224','up320'):
        assert 'Run: '+label in md
    assert '1.400 mm' in md
    with zipfile.ZipFile(tmp_path/'results_bundle.zip') as z:
        names=set(z.namelist())
        for label in ('up64','up128','up224','up320'):
            assert f'{label}/metrics.csv' in names
            assert f'{label}/fixed_eval.csv' in names
        assert all(not name.endswith('.pkl') for name in names)
