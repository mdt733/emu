"""V5020 result artifacts must be regenerable offline after a finished run."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile

from emu.results import write_report


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_v5020_campaign_report_contains_true_rating_and_shareable_bundle(tmp_path):
    run=tmp_path/'up'
    (run/'checkpoints').mkdir(parents=True)
    checkpoint=run/'checkpoints'/'final.pkl'
    checkpoint.write_bytes(b'NOT A REAL PICKLE; REPORT MUST NEVER LOAD THIS')
    _write_csv(run/'metrics.csv',[
        dict(step='20709376', reward_mean='2.969', fall_rate='0',
             ep_len_mean='304', frontier_terrain_height_m='0',
             terrain_height_birth_m='0', terrain_frontier_event='hold'),
        dict(step='32047104', reward_mean='3.180', fall_rate='.011',
             ep_len_mean='699', frontier_terrain_height_m='.000231386',
             terrain_height_birth_m='.0002',
             terrain_frontier_event='hold_collect_above_rating_trials')
    ])
    _write_csv(run/'fixed_eval.csv',[
        dict(step='32047104',eval_policy_step='32047104',
             eval_broad_score='.095',eval_core_survival='1.0',
             eval_foundation_walking_pass='.999')
    ])
    (run/'training_config.json').write_text(json.dumps(
        dict(steps=32000000,envs=2048,seed=733,terrain_elevation_mode='up',
             terrain_height_safety_m=.006,capability_field_mode='shadow')))
    (run/'run_meta.json').write_text(json.dumps(
        dict(source_provenance={'git_commit':'a'*40})))
    lineage=tmp_path/'lineage'
    lineage.mkdir()
    (lineage/'up.parent.sha256').write_text('f'*64+'\n')

    outputs=write_report(tmp_path)
    assert {p.name for p in outputs}=={
        'results.md','results_data.json','results_bundle.zip'}
    md=(tmp_path/'results.md').read_text()
    assert 'Physical elevation skill rating' in md
    assert '6.000 mm (NOT demonstrated ability)' in md
    assert '0.231 mm' in md
    assert 'hold_collect_above_rating_trials' in md
    assert 'f'*64 in md
    assert 'a'*40 in md
    assert 'up/metrics.csv' not in md  # compact summary, raw data in ZIP
    data=json.loads((tmp_path/'results_data.json').read_text())
    assert data['runs']['up']['terrain']['latest_rating_mm']==.231386
    assert data['runs']['up']['parent']['sha256']=='f'*64
    assert data['runs']['up']['checkpoint_sha256']==hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    with zipfile.ZipFile(tmp_path/'results_bundle.zip') as z:
        paths=set(z.namelist())
    assert {'results.md','results_data.json',
            'up/metrics.csv','up/fixed_eval.csv',
            'up/training_config.json'}.issubset(paths)
    assert all(not name.endswith('.pkl') for name in paths)


def test_v5020_standalone_up_dir_report_regenerates_without_campaign_root(tmp_path):
    run=tmp_path/'up'
    (run/'checkpoints').mkdir(parents=True)
    _write_csv(run/'metrics.csv',[dict(
        step='32047104',reward_mean='3.18',fall_rate='.011',
        frontier_terrain_height_m='.000231')])
    (run/'training_config.json').write_text(json.dumps(
        dict(steps=32000000,terrain_height_safety_m=.006,
             terrain_elevation_mode='up')))
    paths=write_report(run)
    assert (run/'results.md') in paths
    assert (run/'results_bundle.zip').is_file()
    assert '0.231 mm' in (run/'results.md').read_text()
