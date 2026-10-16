"""Three delivery contracts: fair promotion, evidence bundles, browser actor parity."""
from __future__ import annotations
import base64
import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pytest
from emu import branch_lab as lab
from emu import ladder_recipe as recipe
from emu import results
from emu.artifacts import append_training_observation, write_capability_field_artifacts, make_metrics_row
from emu.contracts import BASE_OBS_FIELDS, OBS_DIM, ACTION_DIM
from emu.ppo import PPO_HIDDEN_SIZES, build_obs_norm_mask
ROOT = Path(__file__).resolve().parents[1]

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
def blob(value: np.ndarray) -> dict:
    array = np.asarray(value, dtype='<f4', order='C')
    return {'dtype': 'float32-le', 'shape': list(array.shape),
            'data_b64': base64.b64encode(array.tobytes()).decode('ascii')}
def reference_action(obs: np.ndarray, layers: list[tuple], mean: np.ndarray,
                     std: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Independent NumPy oracle for the deployed deterministic actor.
    x = np.where(mask > 0.5, np.clip((obs - mean) / np.maximum(std, 1e-8),
                                    -10.0, 10.0), obs).astype(np.float32)
    for weights, bias in layers[:-1]:
        x = np.tanh(x @ weights + bias).astype(np.float32)
    weights, bias = layers[-1]
    return np.clip(x @ weights + bias, -1.0, 1.0)

def test_auto_promotion_only_accepts_eligible_sorted_arms():
    results={'round':'round1','results':[
        {'arm':'pitch15','eligible':False,'score':.99},
        {'arm':'slope','eligible':True,'score':.60},
        {'arm':'baseline','eligible':True,'score':.60},
        {'arm':'shadow','eligible':True,'score':.20},
    ]}
    assert lab.promotion_choices(results,2)==['baseline','slope']
    assert lab.promotion_choices(results,1)==['baseline']
    # The study designer (not the runner) owns round budgets and advancement.
    # For this recipe: 20M + 4*60M + 2*20M + 1*20M = 320M.
    study = lab.plan_dict(seed=733, envs=2048)
    assert study == recipe.plan_dict(seed=733, envs=2048)
    assert study['nominal_steps'] == 320_000_000
    assert study['round1_incremental_steps'] == 240_000_000
    assert study['through_round1_total_steps'] == 260_000_000
    assert list(recipe.PROMOTION_SLOTS.items()) == [('round2', 2), ('round3', 1)]
    assert study['rounds'] == recipe.ROUNDS
    assert [a['name'] for a in study['arms']] == list(recipe.BY_NAME)
    with pytest.raises(RuntimeError,match='Only 1 of 2'):
        lab.promotion_choices({'round':'round1','results':results['results'][:2]},2)

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
    # Report presentation has one owner; it cannot change eligibility or
    # select a finalist, and must write the same scorecard schema.
    scorecard = results.write_scorecard(root, {
        'round': 'round1', 'target': 80_000_000, 'screen_pitch_deg': 15.0,
        'results': [
            {'arm': 'baseline', 'eligible': True, 'score': .681621, 'reasons': []},
            {'arm': 'shadow', 'eligible': False, 'score': None,
             'reasons': ['insufficient evidence']},
        ],
    })
    scorecard_text = scorecard.read_text()
    assert '80,000,000 absolute steps' in scorecard_text
    assert '15 degree testing envelope' in scorecard_text
    assert '| baseline | True | 0.681621 |' in scorecard_text
    assert 'insufficient evidence' in scorecard_text
    outputs = results.write_report(root)
    assert {p.name for p in outputs} == {
        'results.md','results_data.json','results_bundle.zip'}
    doc=(root/'results.md').read_text()
    assert '0.682' in doc
    assert 'round1/shadow' in doc
    assert 'common held-out ranking' in doc
    assert 'servo_shaft_abs_power_w' in doc and 'spring_potential_energy_j' in doc
    assert 'not proof of walking' in doc
    assert 'Training narrative (on-policy, not a walking grade)' in doc
    assert 'On-policy teaching reward rose' in doc
    fictional = [{'step': str((i+1)*100), 'reward_mean': str(v)} for i,v in enumerate(
        [0.2, 0.3, 0.4, 0.9, 1.2, 1.3, 1.1, 0.8, 0.4, 0.25])]
    text_story = results.training_story(fictional, {
        'reward_mean': {'early': .2, 'late': .25}})
    assert any('Possible training setback' in x for x in text_story)
    assert not any('learned to walk' in x for x in text_story)
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

    # The periodic/final metrics and field exports use one owner and schema.
    class FakeField:
        def save_npz(self, path):
            path.write_bytes(b'field')
        def write_summary(self, path, **values):
            path.write_text(json.dumps(values), encoding='utf-8')

    output = tmp_path / 'training_artifacts'
    output.mkdir()
    envelope = {'field_node_frac': 0.25, 'field_edge_frac': 0.1,
                'field_large_internal': 123}
    append_training_observation(
        output, step=40, params=envelope, report={'competence': 0.7},
        metrics_row={'step': 40, 'reward_mean': 0.5})
    append_training_observation(
        output, step=80, params=envelope, report={'competence': 0.8},
        metrics_row={'step': 80, 'reward_mean': 0.6})
    with (output / 'metrics.csv').open(newline='') as stream:
        assert [int(row['step']) for row in csv.DictReader(stream)] == [40, 80]
    assert json.loads((output / 'envelope_summary.json').read_text())['params'] == {
        'field_node_frac': 0.25, 'field_edge_frac': 0.1}
    for label in ('latest', 'final'):
        write_capability_field_artifacts(
            output, FakeField(), step=80, mode='shadow',
            params=envelope, label=label)
        assert (output / f'capability_field_{label}.npz').read_bytes() == b'field'
    assert json.loads((output / 'capability_field_summary.json').read_text())['node_frac'] == 0.25
    # Reporting consumes explicit learning metadata rather than importing PPO.
    metric = make_metrics_row(
        step=80, elapsed=2.0, wall_fps=400, train_fps=500,
        reward=0.6, row={}, report={}, aux={'lr_scale': 0.5},
        nan_grad_skips=0, num_envs=8, jax_backend='cpu',
        perf_row={}, rollout_steps=32, base_lr=0.001)
    assert metric['rollout'] == 32
    assert metric['ppo_effective_lr'] == 0.0005


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js needed for browser parity')
def test_exported_actor_matches_browser_inference():
    rng = np.random.default_rng(733)
    dims = (OBS_DIM, *PPO_HIDDEN_SIZES, ACTION_DIM)
    layers = []
    for size_in, size_out in zip(dims[:-1], dims[1:]):
        weights = (rng.standard_normal((size_in, size_out)).astype(np.float32)
                   * np.float32(.6 / np.sqrt(size_in)))
        bias = (rng.standard_normal(size_out).astype(np.float32)
                * np.float32(.08))
        layers.append((weights, bias))

    mean = rng.normal(0, .15, OBS_DIM).astype(np.float32)
    std = rng.uniform(.15, 2.0, OBS_DIM).astype(np.float32)
    mask = build_obs_norm_mask()
    observations = [
        np.zeros(OBS_DIM, dtype=np.float32),
        rng.normal(0, 1, OBS_DIM).astype(np.float32),
        np.where(np.arange(OBS_DIM) % 2 == 0, 50.0, -50.0).astype(np.float32),
        np.linspace(-.7, .9, OBS_DIM, dtype=np.float32),
    ]
    bundle = {
        'format': 'emu-policy', 'schema': 1, 'robot_xml': '<mujoco/>',
        'policy': {'obs_dim': OBS_DIM, 'base_obs_dim': len(BASE_OBS_FIELDS),
                   'history_len': 4, 'action_dim': ACTION_DIM,
                   'layers': [{'w': blob(w), 'b': blob(b)} for w, b in layers]},
        'observation': {'base_fields': list(BASE_OBS_FIELDS),
                        'mean': blob(mean), 'std': blob(std),
                        'normalize_mask': blob(mask)},
        'control': {},
    }
    js = r"""
import fs from 'node:fs';
import { PolicyRunner, parsePolicy } from './emu/viewer_src/src/policy.js';
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const runner = new PolicyRunner(parsePolicy(JSON.stringify(request.bundle)));
const actions = request.observations.map(
    obs => Array.from(runner.infer(Float32Array.from(obs))));
process.stdout.write(JSON.stringify(actions));
"""
    result = subprocess.run(
        ['node', '--input-type=module', '--eval', js],
        cwd=ROOT, input=json.dumps({
            'bundle': bundle, 'observations': [o.tolist() for o in observations],
        }), text=True, capture_output=True, check=True, timeout=25,
    )
    browser_actions = np.asarray(json.loads(result.stdout), dtype=np.float32)
    assert browser_actions.shape == (len(observations), ACTION_DIM)
    for observed, browser in zip(observations, browser_actions):
        expected = reference_action(observed, layers, mean, std, mask)
        np.testing.assert_allclose(browser, expected, rtol=1e-4, atol=2e-5)
