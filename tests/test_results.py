"""CPU-only results exporter contracts: complete campaign, missing data, and CLI."""
import csv
import json
import zipfile

import pytest
from emu import results


def fixture_run(root, label, parent=None, final=True):
    run = root/label
    run.mkdir(parents=True)
    (run/'checkpoints').mkdir()
    ck = run/'checkpoints/final.pkl'
    if final:
        ck.write_bytes(('test-'+label).encode('ascii'))
    target = 20_000_000 if label == 'foundation20' else 80_000_000
    (run/'training_config.json').write_text(json.dumps({
        'steps':target,'capability_field_mode':'shadow',
        'domain_terrain_radius':0.0,'pitch_command_max_deg':8.0}))
    (run/'run_meta.json').write_text(json.dumps({
        'source_provenance':{'git_commit':'a'*40}}))
    if parent:
        (run/'lab_spec.json').write_text(json.dumps({
            'parent':{'step':20_000_000,'sha256':parent}}))
    with (run/'metrics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=(
            'step','reward_mean','fall_rate','height_error_m',
            'servo_over_rated_frac','reward_weighted_alive'))
        writer.writeheader()
        for i in range(10):
            writer.writerow(dict(step=target-9+i,reward_mean=i*.1,
                fall_rate=.9-i*.08,height_error_m=-.01+i*.001,
                servo_over_rated_frac=.6,reward_weighted_alive=.58))
    (run/'fixed_eval.csv').write_text(
        'step,eval_core_survival,eval_core_vx_mae\n'+str(target)+',0.75,0.03\n')
    return run,results.digest(ck) if final else None


def test_full_fourway_report_roundtrips_into_small_shareable_zip(tmp_path):
    root = tmp_path/'campaign'
    root.mkdir()
    (root/'lab_context.json').write_text(json.dumps({
        'seed':733,'envs':2048,'git_commit':'a'*40}))
    _, parent_sha = fixture_run(root,'foundation20')
    scored = []
    for name,rank in [('baseline',.681621),('shadow',.66808),
                      ('pitch15',.653529),('slope',.636739)]:
        run,sha = fixture_run(root,'round1/'+name,parent=parent_sha)
        (run/'screen_eval.json').write_text(json.dumps({
            'screen_robust70_survival':.86,'screen_robust70_vx_mae':.1,
            'screen_growth140_survival':.7}))
        scored.append({'arm':name,'eligible':True,'score':rank,
                       'hard_survival':.87,'growth_survival':.7,
                       'pitch_mae_rad':.08,'height_shortfall_m':.003,
                       'checkpoint_sha256':sha})
    (root/'round1_scoreboard.json').write_text(json.dumps({
        'round':'round1','screen_envs':256,'screen_horizon':350,
        'screen_pitch_deg':15.0,'target':80_000_000,'results':scored}))
    outputs = results.write_report(root)
    assert {p.name for p in outputs} == {
        'results.md','results_data.json','results_bundle.zip'}
    doc=(root/'results.md').read_text()
    assert '0.682' in doc
    assert 'round1/shadow' in doc
    assert 'common held-out ranking' in doc
    assert 'servo_shaft_abs_power_w' in doc and 'spring_potential_energy_j' in doc
    assert 'not proof of walking' in doc
    data=json.loads((root/'results_data.json').read_text())
    assert len(data['runs']) == 5
    assert not any('PARENT HASH DIFFERS' in p
                   for row in data['runs'].values() for p in row['problems'])
    with zipfile.ZipFile(root/'results_bundle.zip') as z:
        assert z.testzip() is None
        names=set(z.namelist())
        assert 'round1/slope/metrics.csv' in names
        assert 'round1/baseline/screen_eval.json' in names
        assert 'foundation20/fixed_eval.csv' in names
        assert 'round1_scoreboard.json' in names
        assert all(not name.endswith(('.pkl','console.log')) for name in names)
    assert results.write_report(root)  # harmless regeneration


def test_missing_and_mismatched_evidence_warns_not_fabricated(tmp_path):
    root=tmp_path/'run'
    root.mkdir()
    run,original=fixture_run(root,'round1/baseline')
    (root/'round1_scoreboard.json').write_text(json.dumps(
        {'results':[{'arm':'baseline','checkpoint_sha256':'f'*64}]}))
    (run/'metrics.csv').unlink()
    item=results.collect(root)['runs']['round1/baseline']
    assert original != 'f'*64
    assert any('HASH MISMATCH' in x for x in item['problems'])
    assert any('metrics' in x for x in item['problems'])
    assert item['changes']['reward_mean']['late'] is None


def test_empty_root_refused_and_standalone_run_supported(tmp_path):
    with pytest.raises(ValueError,match='No campaign runs'):
        results.write_report(tmp_path)
    run,_=fixture_run(tmp_path,'solo')
    assert 'single' in results.collect(run)['runs']
    assert results.write_report(run)[0].name == 'results.md'


def test_cli_report_is_a_pure_offline_dispatch(monkeypatch,tmp_path):
    from emu import cli
    calls=[]
    monkeypatch.setattr(results,'write_report',
        lambda root,bundle=True: calls.append((root,bundle)) or [])
    assert cli.main(['report','--root',str(tmp_path),'--no-bundle']) == 0
    assert calls == [(tmp_path,False)]



def test_fixed_eval_staleness_and_source_mismatch_are_visible(tmp_path):
    root = tmp_path/'old'
    root.mkdir()
    (root/'lab_context.json').write_text(json.dumps({'git_commit':'1'*40}))
    run,_ = fixture_run(root,'foundation20')
    (run/'fixed_eval.csv').write_text('step,eval_core_survival\n100,1.0\n')
    row = results.collect(root)['runs']['foundation20']
    assert any('predates' in p for p in row['problems'])
    assert any('SOURCE COMMIT DIFFERS' in p for p in row['problems'])
