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
        for axis in ('vx','vy','yaw'):
            result[f'screen_{panel}_{axis}_mae']=.08
    return result


def test_plan_is_only_72m_steps_and_never_starts_training(tmp_path,capsys):
    assert lab.main(['--stage','plan','--root',str(tmp_path)]) == 0
    data=json.loads(capsys.readouterr().out)
    assert data['nominal_steps']==72_000_000
    assert data['foundation']==20_000_000
    assert data['rounds']=={'round1':26_000_000,'round2':34_000_000,'round3':46_000_000}
    assert {x['name'] for x in data['arms']}=={'baseline','shadow','slope','pitch15'}
    assert not list(tmp_path.iterdir())


def test_each_arm_starts_from_one_v5015_mature_recipe(tmp_path):
    base=lab.config(tmp_path/'baseline',arm=lab.BY_NAME['baseline'],target=26_000_000,
                    seed=733,envs=2048,device='cpu',resume=None)
    assert base.attitude_profile=='v5015'
    assert base.pitch_command_max_deg==8
    assert base.field_max_influence==.10
    assert base.domain_terrain_radius==0
    for name in ('shadow','slope','pitch15'):
        candidate=lab.config(tmp_path/name,arm=lab.BY_NAME[name],target=26_000_000,
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
    board={'target':26_000_000,'results':[
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
                        lambda run,**kw:(run/'checkpoints'/'final.pkl',None,26_017_792))
    monkeypatch.setattr(lab,'run_training',
                        lambda *a,**kw: (_ for _ in ()).throw(AssertionError('no training expected')))
    for name in ('baseline','slope'):
        run=tmp_path/'round1'/name
        (run/'checkpoints').mkdir(parents=True)
        (run/'checkpoints'/'final.pkl').write_bytes(b'checkpoint')
        lab.ensure_run(run,arm=lab.BY_NAME[name],parent=parent,target=26_000_000,
                       seed=733,envs=2048,device='cpu',provenance={})
    left=lab._read(tmp_path/'round1'/'baseline'/'lab_spec.json')
    right=lab._read(tmp_path/'round1'/'slope'/'lab_spec.json')
    assert left['parent']==right['parent']
    assert left['arm']!=right['arm']
