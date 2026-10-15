"""CPU-only safeguards for the v5015 shared-foundation experiment funnel."""
import json
from dataclasses import asdict

import pytest
from emu import branch_lab as lab


def _screen(*, survival=.65, pitch=.1, height=.008):
    result={}
    for panel in ('resilience100','robust70','growth140','growthcombo140'):
        result[f'screen_{panel}_survival']=survival
        result[f'screen_{panel}_pitch_mae']=pitch
        result[f'screen_{panel}_height_shortfall_m']=height
        result[f'screen_{panel}_step_evidence']=0.54
        result[f'screen_{panel}_support']=0.47
        for axis in ('vx','vy','yaw'):
            result[f'screen_{panel}_{axis}_mae']=.08
    result['screen_gentle_slope_survival']=survival
    result['screen_gentle_slope_height_shortfall_m']=height
    return result


def test_plan_reports_shared_ancestor_four_sixty_million_continuations(tmp_path,capsys):
    assert lab.main(['--stage','plan','--root',str(tmp_path)]) == 0
    data=json.loads(capsys.readouterr().out)
    assert data['nominal_steps']==320_000_000
    assert data['foundation']==20_000_000
    assert data['through_round1_total_steps']==260_000_000
    assert data['round1_incremental_steps']==240_000_000
    assert data['rounds']=={'round1':80_000_000,'round2':100_000_000,'round3':120_000_000}
    assert {x['name'] for x in data['arms']}=={'baseline','shadow','slope','pitch15'}
    assert not list(tmp_path.iterdir())


def test_each_arm_starts_from_one_v5015_mature_recipe(tmp_path):
    base=lab.config(tmp_path/'baseline',arm=lab.BY_NAME['baseline'],target=80_000_000,
                    seed=733,envs=2048,device='cpu',resume=None)
    assert base.attitude_profile=='v5015'
    assert base.pitch_command_max_deg==8
    assert base.field_max_influence==.10
    assert base.domain_terrain_radius==0
    for name in ('shadow','slope','pitch15'):
        candidate=lab.config(tmp_path/name,arm=lab.BY_NAME[name],target=80_000_000,
                             seed=733,envs=2048,device='cpu',resume=None)
        changed={k for k,v in asdict(base).items() if asdict(candidate)[k]!=v}
        changed.discard('run_dir')
        if name=='shadow':
            assert changed=={'capability_field_mode','field_max_influence'}
        elif name=='slope':
            assert changed=={'domain_randomization','domain_terrain_radius'}
        else:
            assert changed=={'pitch_command_max_deg'}


def test_finite_metrics_and_crouch_gate():
    assert lab.score_row('baseline',_screen())['eligible']
    assert not lab.score_row('pitch15',_screen(survival=.98,height=.055))['eligible']
    assert not lab.score_row('shadow',{})['eligible']
    assert not lab.score_row('slope',_screen(pitch=.65))['eligible']


def test_selection_needs_real_scoreboard_and_resists_unsafe_promotion(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='Score'):
        lab._selection(tmp_path,'round2',['baseline','slope'],'round1',2048)
    board={'target':80_000_000,'results':[
        {'arm':'baseline','eligible':True,'checkpoint_sha256':'good'},
        {'arm':'slope','eligible':False,'checkpoint_sha256':'bad'}]}
    lab._save(tmp_path/'round1_scoreboard.json',board)
    monkeypatch.setattr(lab,'completed',lambda *a,**k:tmp_path/'checkpoint.pkl')
    monkeypatch.setattr(lab,'_hash',lambda *a:'good')
    with pytest.raises(RuntimeError,match='no eligible'):
        lab._selection(tmp_path,'round2',['baseline','slope'],'round1',2048)
    board['results'][1]['eligible']=True
    lab._save(tmp_path/'round1_scoreboard.json',board)
    with pytest.raises(RuntimeError,match='differs'):
        lab._selection(tmp_path,'round2',['baseline','slope'],'round1',2048)


def test_run_identity_cannot_mutate(tmp_path):
    lab._context(tmp_path,733,2048,{'available':True,'dirty':False,'commit':'abc'})
    with pytest.raises(RuntimeError,match='Source commit'):
        lab._context(tmp_path,733,1024,{'available':True,'dirty':False,'commit':'abc'})
    with pytest.raises(RuntimeError,match='uncommitted'):
        lab._context(tmp_path,733,2048,{'available':True,'dirty':True,'commit':'abc'})


def test_each_branch_ancestry_is_immutable(tmp_path,monkeypatch):
    parent=tmp_path/'foundation.pkl'
    parent.write_bytes(b'parent checkpoint state')
    monkeypatch.setattr(lab,'checkpoint_step',lambda *a:20_000_000)
    monkeypatch.setattr(lab,'verify_existing_phase_config',lambda *a:None)
    monkeypatch.setattr(lab,'resolve_target_checkpoint',
                        lambda run,**kw:(run/'checkpoints'/'final.pkl',None,80_019_456))
    monkeypatch.setattr(lab,'run_training',
                        lambda *a,**kw: (_ for _ in ()).throw(AssertionError('no training expected')))
    for name in ('baseline','slope'):
        run=tmp_path/'round1'/name
        (run/'checkpoints').mkdir(parents=True)
        (run/'checkpoints'/'final.pkl').write_bytes(b'checkpoint')
        lab.ensure_run(run,arm=lab.BY_NAME[name],parent=parent,target=80_000_000,
                       seed=733,envs=2048,device='cpu',provenance={})
    left=lab._read(tmp_path/'round1'/'baseline'/'lab_spec.json')
    right=lab._read(tmp_path/'round1'/'slope'/'lab_spec.json')
    assert left['parent']==right['parent']
    assert left['arm']!=right['arm']


def test_auto_promotion_only_accepts_eligible_sorted_arms():
    results={'round':'round1','results':[
        {'arm':'pitch15','eligible':False,'score':.99},
        {'arm':'slope','eligible':True,'score':.60},
        {'arm':'baseline','eligible':True,'score':.60},
        {'arm':'shadow','eligible':True,'score':.20},
    ]}
    assert lab.promotion_choices(results,2)==['baseline','slope']
    assert lab.promotion_choices(results,1)==['baseline']
    with pytest.raises(RuntimeError,match='Only 1 of 2'):
        lab.promotion_choices({'round':'round1','results':results['results'][:2]},2)


def test_autopilot_uses_foundation_all_four_arms_then_best_two_then_one(tmp_path,monkeypatch):
    events=[]
    def ensure(run,**kw):
        events.append(('train',str(run.relative_to(tmp_path))))
        return run/'checkpoints'/'final.pkl'
    def score(root,round_name,**kw):
        events.append(('score',round_name))
        values=({'baseline':.8,'shadow':.9,'slope':.7,'pitch15':.6}
                if round_name=='round1' else
                {'baseline':.8,'shadow':.9} if round_name=='round2' else
                {'shadow':.9})
        lab._save(root/f'{round_name}_scoreboard.json',
                  {'round':round_name,'target':lab.ROUNDS[round_name],
                   'results':[{'arm':n,'eligible':True,'score':v}
                              for n,v in values.items()]})
    def select(root,stage,names,previous,envs):
        events.append(('select',stage,tuple(names)))
        lab._save(root/f'{stage}_selection.json',
                  {'selected':sorted(names),'from_round':previous})
    monkeypatch.setattr(lab,'ensure_run',ensure)
    monkeypatch.setattr(lab,'completed',lambda *args,**kw:tmp_path/'source.pkl')
    monkeypatch.setattr(lab,'score_is_current',lambda *args,**kw:False)
    monkeypatch.setattr(lab,'score',score)
    monkeypatch.setattr(lab,'_selection',select)
    lab.autopilot(tmp_path,seed=733,envs=2048,device='cuda',
                  screen_envs=256,screen_horizon=350,provenance={})
    assert [e for e in events if e[0]=='train']==[
        ('train','foundation20'),('train','round1/baseline'),
        ('train','round1/shadow'),('train','round1/slope'),('train','round1/pitch15'),
        ('train','round2/shadow'),('train','round2/baseline'),
        ('train','round3/shadow')]
    assert [e for e in events if e[0]=='score']==[
        ('score','round1'),('score','round2'),('score','round3')]
    assert ('select','round2',('shadow','baseline')) in events
    assert ('select','round3',('shadow',)) in events


def test_autopilot_stops_without_two_eligible_candidates(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr(lab,'ensure_run',
                        lambda run,**kw:seen.append(('train',str(run.relative_to(tmp_path)))))
    monkeypatch.setattr(lab,'completed',lambda *a,**kw:tmp_path/'parent.pkl')
    monkeypatch.setattr(lab,'score_is_current',lambda *a,**kw:False)
    def partial_score(root,round_name,**kw):
        seen.append(('score',round_name))
        lab._save(root/f'{round_name}_scoreboard.json',
                  {'round':round_name,'target':lab.ROUNDS[round_name],
                   'results':[{'arm':'baseline','eligible':True,'score':.7}]})
    monkeypatch.setattr(lab,'score',partial_score)
    with pytest.raises(RuntimeError,match='Only 1 of 2'):
        lab.autopilot(tmp_path,seed=733,envs=2048,device='cuda',
                      screen_envs=256,screen_horizon=350,provenance={})
    assert ('score','round1') in seen
    assert not any(x[0]=='train' and x[1].startswith('round2/') for x in seen)


def test_score_cache_only_reuses_matching_checkpoint_hashes_and_cohort(tmp_path,monkeypatch):
    data={'round':'round1','target':80_000_000,'screen_pitch_deg':15.,
          'screen_envs':256,'screen_horizon':350,'results':[]}
    for name in lab.BY_NAME:
        folder=tmp_path/'round1'/name/'checkpoints'
        folder.mkdir(parents=True)
        ck=folder/'final.pkl'
        ck.write_bytes(name.encode())
        data['results'].append({'arm':name,'checkpoint_sha256':lab._hash(ck)})
    lab._save(tmp_path/'round1_scoreboard.json',data)
    (tmp_path/'round1_scorecard.md').write_text('test report')
    monkeypatch.setattr(lab,'completed',
        lambda root,stage,name,envs:root/stage/name/'checkpoints'/'final.pkl')
    assert lab.score_is_current(tmp_path,'round1',envs=2048,
                                screen_envs=256,screen_horizon=350)
    assert not lab.score_is_current(tmp_path,'round1',envs=2048,
                                    screen_envs=512,screen_horizon=350)
    (tmp_path/'round1'/'baseline'/'checkpoints'/'final.pkl').write_bytes(b'different')
    assert not lab.score_is_current(tmp_path,'round1',envs=2048,
                                    screen_envs=256,screen_horizon=350)


def test_top_level_cli_exposes_auto_and_forwards_stop_after(monkeypatch):
    from emu import cli
    got=[]
    monkeypatch.setattr(cli,'_run',
                        lambda module,*args:got.append((module,args)) or 0)
    assert cli.main(['lab5015','--stage','auto','--stop-after','round1']) == 0
    assert got and got[0][0]=='branch_lab'
    forwarded=list(got[0][1])
    assert forwarded[forwarded.index('--stage')+1]=='auto'
    assert forwarded[forwarded.index('--stop-after')+1]=='round1'

def test_recipe_is_a_separate_explicit_step_and_hypothesis_list():
    from emu import ladder_recipe
    assert ladder_recipe.FOUNDATION == 20_000_000
    assert ladder_recipe.ROUNDS == {'round1':80_000_000,'round2':100_000_000,'round3':120_000_000}
    assert tuple(ladder_recipe.BY_NAME) == ('baseline','shadow','slope','pitch15')
    assert lab.ROUNDS is ladder_recipe.ROUNDS  # runner and recipe never drift


def test_human_friendly_ladder_command_routes_to_existing_runner(monkeypatch):
    from emu import cli
    observed=[]
    monkeypatch.setattr(cli,'_run',lambda name,*args: observed.append((name,args)) or 0)
    assert cli.main(['ladder','--stage','plan']) == 0
    assert observed[0][0]=='branch_lab'
    assert observed[0][1][:2]==('--stage','plan')
    assert cli.main(['lab5015','--stage','auto']) == 0
    assert observed[1][0]=='branch_lab'


def test_ladder_supports_new_knobs_without_editing_engine(tmp_path):
    from emu.ladder_recipe import Arm
    arm=Arm('experiment','Alter two command-family knobs',(
        ('combo_mix_frac', 0.48),
        ('field_teach_edges', False),
        ('freeze_obs_norm_after_steps', 35_000_000),
    ))
    cfg=lab.config(tmp_path/'experiment',arm=arm,target=80_000_000,
                   seed=733,envs=2048,device='cpu',resume=None)
    assert cfg.combo_mix_frac == .48
    assert cfg.field_teach_edges is False
    assert cfg.freeze_obs_norm_after_steps == 35_000_000
    assert cfg.attitude_profile == 'v5015'
    assert cfg.steps == 80_000_000


@pytest.mark.parametrize('knobs,match', [
    ((('nonexistent_config', .1),), 'Unknown'),
    ((('seed', 12),), 'run identity'),
    ((('steps', 40_000_000),), 'run identity'),
    ((('run_dir', '/tmp/hijack'),), 'run identity'),
    ((('combo_mix_frac', .5),('combo_mix_frac', .2)), 'Duplicate'),
    ((('domain_randomization', .3),), 'Invalid value'),
    ((('field_max_influence', float('nan')),), 'Invalid value'),
    ((('freeze_obs_norm_after_steps', .5),), 'Invalid value'),
])
def test_invalid_ladder_knobs_fail_before_training(tmp_path, knobs, match):
    from emu.ladder_recipe import Arm
    arm=Arm('bad','Validation test',knobs)
    with pytest.raises(ValueError, match=match):
        lab.config(tmp_path/'bad',arm=arm,target=80_000_000,
                   seed=733,envs=2048,device='cpu',resume=None)


def test_bad_future_ladder_fails_at_plan_before_any_gpu_work(tmp_path,monkeypatch):
    from emu.ladder_recipe import Arm
    # A single invalid challenger must prevent the expensive shared foundation.
    arms=(Arm('baseline','control'),Arm('bad_option','mistyped',(
        ('nonexistent_knob', .5),
    )))
    monkeypatch.setattr(lab,'ARMS',arms)
    monkeypatch.setattr(lab,'BY_NAME',{a.name:a for a in arms})
    with pytest.raises(ValueError,match='Unknown TrainingConfig'):
        lab.main(['--stage','plan','--root',str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_ladder_rejects_duplicate_and_path_escape_arm_names(tmp_path,monkeypatch):
    from emu.ladder_recipe import Arm
    for arms in [
        (Arm('baseline','control'),Arm('baseline','duplicate')),
        (Arm('baseline','control'),Arm('../escape','invalid')),
    ]:
        monkeypatch.setattr(lab,'ARMS',arms)
        monkeypatch.setattr(lab,'BY_NAME',{a.name:a for a in arms})
        with pytest.raises(ValueError,match='uniquely named|Unsafe ladder'):
            lab.validate_recipe(seed=733,envs=2048,device='cpu',root=tmp_path)


def test_optimizer_knobs_are_available_without_changing_the_trainer(tmp_path):
    from emu.ladder_recipe import Arm
    from emu.training_config import PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST
    base=lab.config(tmp_path/'baseline',arm=lab.BY_NAME['baseline'],
                    target=80_000_000,seed=733,envs=128,device='cpu',resume=None)
    assert (base.ppo_lr,base.ppo_entropy_cost,base.ppo_value_cost) == (
        PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST)
    arm=Arm('half_lr','Lower learning rate with same architecture',
            (('ppo_lr',PPO_LR/2),('ppo_entropy_cost',.001),('ppo_value_cost',.4)))
    alt=lab.config(tmp_path/'half_lr',arm=arm,target=80_000_000,
                   seed=733,envs=128,device='cpu',resume=None)
    changed={name for name,value in asdict(base).items()
             if getattr(alt,name)!=value} - {'run_dir'}
    assert changed=={'ppo_lr','ppo_entropy_cost','ppo_value_cost'}


@pytest.mark.parametrize('knob', [('ppo_lr',0),('ppo_lr',-1e-4),
                                  ('ppo_entropy_cost',-.1),('ppo_value_cost',-.1)])
def test_optimizer_knob_invalid_values_rejected(tmp_path,knob):
    from emu.ladder_recipe import Arm
    with pytest.raises(ValueError):
        lab.config(tmp_path/'bad',arm=Arm('bad','unsafe', (knob,)),
                   target=80_000_000,seed=733,envs=128,device='cpu',resume=None)


def test_foundation_config_writes_inside_foundation20_not_old_00_foundation(tmp_path):
    destination = tmp_path/'foundation20'
    cfg = lab.config(destination,arm=None,target=20_000_000,seed=733,
                     envs=2048,device='cpu',resume=None)
    assert cfg.run_dir == str(destination)
    assert cfg.steps == 20_000_000
    assert cfg.attitude_profile == 'v5015'


def test_run_rejects_misrouted_checkpoint_destination_before_starting_training(tmp_path,monkeypatch):
    from dataclasses import replace
    dest=tmp_path/'foundation20'
    real_config=lab.config
    monkeypatch.setattr(lab,'config', lambda *args,**kw: replace(
        real_config(*args,**kw),
        run_dir=str(tmp_path/'00_foundation')))
    monkeypatch.setattr(lab,'verify_existing_phase_config',
                        lambda *args: (_ for _ in ()).throw(
                            AssertionError('config should be checked before resume')))
    monkeypatch.setattr(lab,'run_training',
                        lambda *args,**kw: (_ for _ in ()).throw(
                            AssertionError('training must never start')))
    with pytest.raises(RuntimeError,match='disagree on output folder'):
        lab.ensure_run(dest,arm=None,parent=None,target=20_000_000,
                       seed=733,envs=2048,device='cpu',provenance={})



def test_completed_foundation_handoff_uses_the_same_directory_as_the_trainer(tmp_path, monkeypatch):
    """CPU-only contract: simulate trainer output; use the real recipe/config wiring."""
    destination = tmp_path / 'foundation20'
    expected_steps = 20_054_016
    training_calls = []

    def synthetic_training(cfg, *, provenance):
        assert cfg.run_dir == str(destination)
        assert cfg.steps == 20_000_000
        training_calls.append(cfg.run_dir)
        final = destination / 'checkpoints' / 'final.pkl'
        final.parent.mkdir(parents=True)
        final.write_bytes(b'validated synthetic checkpoint')

    def synthetic_completion(run, *, target_steps, envs, label):
        assert run == destination
        assert target_steps == 20_000_000
        assert envs == 2048
        final = run / 'checkpoints' / 'final.pkl'
        return (final if final.exists() else None), None, expected_steps

    monkeypatch.setattr(lab, 'verify_existing_phase_config', lambda *a: None)
    monkeypatch.setattr(lab, 'resolve_target_checkpoint', synthetic_completion)
    monkeypatch.setattr(lab, 'run_training', synthetic_training)

    final = lab.ensure_run(destination, arm=None, parent=None, target=20_000_000,
                           seed=733, envs=2048, device='cpu', provenance={})
    assert final == destination / 'checkpoints' / 'final.pkl'
    assert training_calls == [str(destination)]
    assert lab._read(destination / 'lab_spec.json')['parent']['step'] == 0



def test_ladder_reports_real_stepping_telemetry_and_rejects_missing_evidence():
    row = lab.score_row('baseline', _screen())
    assert row['eligible']
    assert row['step_evidence'] == .54
    assert row['support'] == .47
    assert row['motion_mae'] == .08
    missing = _screen()
    del missing['screen_growth140_step_evidence']
    assert not lab.score_row('missing', missing)['eligible']
