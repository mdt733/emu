"""Live skill counts must reflect real event denominators, not frame counts."""
import json
import numpy as np
import pytest

from emu.live_skill_observer import LiveSkillObserver
from emu.capability_field import CapabilityField, RECOVERY_BINS, EDGE_DIRS
from emu.training_config import ppo_minibatches_for_envs


def _observer(tmp_path, worlds=4096):
    field = CapabilityField()
    return LiveSkillObserver(tmp_path, start_step=0, worlds=worlds,
                             field_shape=field.shape, field_bounds=field.bounds,
                             source_commit="a"*40, recovery_bins=RECOVERY_BINS,
                             edge_directions=EDGE_DIRS, window_steps=1)


def _rollout(worlds=4096):
    shape = (32, worlds)
    names = ["episode_done", "episode_fell", "specialist_focus", "academy_active",
             "skill_first_birth_lip_m", "academy_episode_first_lip_armed",
             "academy_episode_two_foot_ready", "skill_first_goal_completed",
             "physical_step_height_m", "physical_slope_deg"]
    names += [f"terrain_episode_{k}_{d}" for d in ("up", "down")
              for k in ("attempt", "success")]
    names += [f"slope_episode_{k}_{d}" for d in
              ("uphill", "downhill", "cross_left", "cross_right")
              for k in ("attempt", "stable", "success")]
    m = {k: np.zeros(shape, np.float32) for k in names}
    m["specialist_focus"].fill(-1)
    return m


def test_live_skill_observer_4096_terminal_evidence(tmp_path):
    obs = _observer(tmp_path)
    m = _rollout()
    # Alive, armed, bilateral and goal signals on ongoing frames must not
    # manufacture additional first-life attempts.
    m["academy_active"][:, :5] = 1
    m["specialist_focus"][:, :5] = 3
    m["skill_first_birth_lip_m"][:, :4] = .026
    m["skill_first_birth_lip_m"][:, 4] = -.026
    m["academy_episode_first_lip_armed"][:, :3] = 1
    m["academy_episode_two_foot_ready"][:, :2] = 1
    m["skill_first_goal_completed"][:, 0] = 1
    m["episode_done"][31, :5] = 1
    m["episode_fell"][31, 1:4] = 1
    m["episode_done"][31, 5:7] = 1
    m["terrain_episode_attempt_up"][31, 5:7] = 1
    m["terrain_episode_success_up"][31, 5] = 1
    m["physical_step_height_m"][31, 5:7] = .018
    m["episode_done"][31, 7] = 1
    m["physical_slope_deg"][31, 7] = 3
    m["slope_episode_attempt_uphill"][31, 7] = 1
    m["slope_episode_stable_uphill"][31, 7] = 1
    obs.observe(m, step=131072)
    obs.field_events("recovery_relative", [11, 11, 12], [1, 0, 1])
    obs.flush()
    row = json.loads(obs.path.read_text())
    c = row["cells"]
    def count(key, attempts, passes):
        assert c[key] == dict(attempts=attempts, passes=passes)
    count("academy_first_birth/up/first_lip_armed/7", 4, 3)
    count("academy_first_birth/up/bilateral_ready/7", 4, 2)
    count("academy_first_birth/up/safe_bilateral/7", 4, 1)
    count("academy_first_birth/up/goal_completed/7", 4, 1)
    count("academy_first_birth/down/survived/7", 1, 1)
    count("ordinary_first_encounter/up/safe_cross/5", 2, 1)
    count("slope_terminal_grade/uphill/stable/3", 1, 1)
    count("slope_terminal_grade/uphill/success/3", 1, 0)
    count("recovery_relative/11", 2, 1)
    assert row["worlds"] == 4096 and row["updates"] == 1
    with pytest.raises(ValueError, match="unordered"):
        obs.observe(m, step=131072)
    obs.rollback()
    obs.observe(_rollout(), step=262144)
    obs.flush(reason="final", force=True)
    rows = [json.loads(s) for s in obs.path.read_text().splitlines()]
    assert rows[-1]["policy_epoch"] == 1
    assert rows[-1]["cells"] == {}
    assert rows[-1]["start_step"] == rows[0]["end_step"]


def test_field_event_sink_does_not_change_field(tmp_path):
    a, b = CapabilityField(), CapabilityField()
    obs = _observer(tmp_path)
    flat = [np.array([1, 1, 2], np.int64)]
    success = [np.array([1, 0, 1], np.float32)]
    quality = [np.array([.8, .2, .9], np.float32)]
    bins = [np.array([0, 0, 2], np.int64)]
    src = [np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], np.float32)]
    tgt = [np.array([[.2, 0., 0.], [0., .2, 0.], [0., 0., 0.]], np.float32)]
    rate = [np.ones(3, np.float32)]
    a._add_node_events(flat, success, quality, 1)
    b._add_node_events(flat, success, quality, 1, obs.field_events)
    a._add_recovery_events(flat, bins, success)
    b._add_recovery_events(flat, bins, success, obs.field_events)
    a._add_edge_events(flat, src, tgt, rate, success)
    b._add_edge_events(flat, src, tgt, rate, success, obs.field_events)
    for key in ("node_success", "node_failure", "node_quality_sum",
                "rec_success", "rec_failure", "edge_success", "edge_failure",
                "edge_rate_sum"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    for family, n in (("command_window", 3), ("recovery_relative", 3),
                      ("transition_edge", 2)):
        assert sum(v[0] for k, v in obs.counts.items()
                   if k.startswith(family + "/")) == n
    assert ppo_minibatches_for_envs(2048) == 8
    assert ppo_minibatches_for_envs(4096) == 16
    assert 2048*32//8 == 4096*32//16


def test_reference_assignment_velocity_and_settling():
    import jax
    import jax.numpy as jnp
    from emu.live_skill_reference import (reference_assignment, reference_velocity,
        reference_progress, REFERENCE_KICKS_MPS, SETTLE_FRAMES)
    n=4096
    keys=jax.random.split(jax.random.PRNGKey(771),n)
    role=jnp.arange(n)%2==0
    athletic=jnp.arange(n)%7==0
    active=jnp.ones(n,dtype=bool)
    ids=jax.jit(lambda k: reference_assignment(jax,jnp,k,role,active,athletic,1.,.30))(keys)
    ids=np.asarray(ids)
    assert set(ids)==set(range(-1,6))
    assert not np.any((ids>=0)&(~np.asarray(role)|np.asarray(athletic)))
    assert np.all(np.asarray(reference_assignment(jax,jnp,keys,role,active,athletic,.9,.30))==-1)
    before=jnp.ones((n,12))
    after=np.asarray(jax.jit(reference_velocity,static_argnums=0)(jnp,before,jnp.asarray(ids)))
    mask=ids>=0
    np.testing.assert_array_equal(after[~mask],np.asarray(before)[~mask])
    np.testing.assert_allclose(after[mask,0]-.16,np.asarray(REFERENCE_KICKS_MPS)[ids[mask]],atol=1e-6)
    assert np.all(after[mask,1:6]==0)
    metrics=dict(vx=jnp.full(n,.16),vy=jnp.zeros(n),yaw_rate=jnp.zeros(n),
                 structural_attitude_ok=jnp.ones(n),support=jnp.ones(n),bad_contact=jnp.zeros(n))
    step=jax.jit(lambda s,d,m,f,t:reference_progress(jnp,jnp.asarray(ids),s,d,m,f,t))
    s=jnp.zeros(n,dtype=jnp.int32); d=jnp.zeros(n,dtype=bool)
    for i in range(SETTLE_FRAMES):
        s,d=step(s,d,metrics,jnp.zeros(n,dtype=bool),.02*(i+1))
    np.testing.assert_array_equal(np.asarray(d),mask)
    s,d=step(jnp.zeros(n,dtype=jnp.int32),jnp.zeros(n,dtype=bool),metrics,jnp.ones(n,dtype=bool),.02)
    assert not np.asarray(s).any() and not np.asarray(d).any()
    s,d=step(jnp.full(n,24),jnp.zeros(n,dtype=bool),metrics,jnp.zeros(n,dtype=bool),3.02)
    assert not np.asarray(d).any()


def test_reference_ledger_requires_repeated_disjoint_evidence():
    from emu.live_skill_ledger import Block, assess_reference
    good=Block(100,95)
    bad=Block(100,60)
    assert assess_reference([good])["status"]=="developing"
    assert assess_reference([good,good])["status"]=="mastered_reference"
    assert assess_reference([good,good,bad])["status"]=="regression_watch"
    assert assess_reference([good,good,bad,bad])["status"]=="repair_advised"
    assert assess_reference([good,good,bad,good])["status"]=="retained"
    assert assess_reference([good,good,Block(2,0)])["status"]=="insufficient_evidence"
    assert assess_reference([good,Block(2,2),good])["anchor"] is None
    with pytest.raises(ValueError):
        Block(5,6)


def test_practice_reallocation_preserves_fixed_reference_births():
    import jax
    import jax.numpy as jnp
    from emu.live_skill_reference import reference_assignment, reference_velocity
    from emu.live_skill_teacher import UNIFORM
    n=4096
    keys=jax.random.split(jax.random.PRNGKey(397),n)
    role=jnp.arange(n)%2==0
    active=jnp.ones(n,dtype=bool)
    athletic=jnp.arange(n)%7==0
    make=jax.jit(lambda w:reference_assignment(jax,jnp,keys,role,active,athletic,
                                                1.,.15,.15,w))
    a=np.asarray(make(jnp.asarray(UNIFORM)))
    b=np.asarray(make(jnp.asarray([7/12]+[1/12]*5)))
    fixed=(a>=0)&(a<6)
    np.testing.assert_array_equal(fixed,(b>=0)&(b<6))
    np.testing.assert_array_equal(a[fixed],b[fixed])
    np.testing.assert_array_equal(a>=6,b>=6)
    np.testing.assert_array_equal(a<0,b<0)
    assert np.count_nonzero(b==6)>np.count_nonzero(a==6)
    assert not np.any((b>=0)&(~np.asarray(role)|np.asarray(athletic)))
    from emu.live_skill_reference import REFERENCE_KICKS_MPS
    after=np.asarray(reference_velocity(jnp,jnp.zeros((n,12)),jnp.asarray(b)))
    np.testing.assert_allclose(after[b>=0,0]-.16,
        np.asarray(REFERENCE_KICKS_MPS)[b[b>=0]%6],atol=1e-6)


def test_practice_plan_uses_only_fixed_reference_not_practice():
    from emu.live_skill_teacher import practice_plan, UNIFORM, MAX_WEIGHT_CHANGE
    from emu.live_skill_reference import REFERENCE_KICKS_MPS
    rows=[]
    for i,p in enumerate((95,95,60,60)):
        rows.append(dict(schema="emu_live_skill_window_v1",
            reference_kicks_mps=list(REFERENCE_KICKS_MPS),reference_vx_mps=.16,
            reference_sensor_scale=1.,reference_step_command_m=.012,
            reference_settle_deadline_s=3.,reference_settle_frames=25,
            reference_horizon_s=3.,reference_mastery_gate='settled_and_survived',
            policy_epoch=0,start_step=i*1000000,end_step=(i+1)*1000000,
            cells={"reference_recovery/mission/0":dict(attempts=100,passes=p),
                   "practice_recovery/mission/1":dict(attempts=10000,passes=0)}))
    assert practice_plan(rows[:3])["reason"]=="uniform_maintenance"
    p=practice_plan(rows)
    assert p["reason"]=="repair" and p["selected"]==[0]
    assert p["weights"][0]>UNIFORM[0]
    assert max(abs(x-y) for x,y in zip(p["weights"],UNIFORM))<=MAX_WEIGHT_CHANGE+1e-9
    for _ in range(20):
        p=practice_plan(rows,p["weights"])
    np.testing.assert_allclose(p["weights"],[7/12]+[1/12]*5)
    assert abs(sum(p["weights"])-1)<1e-9
    clean=[dict(rows[-1],policy_epoch=1)]
    assert practice_plan(clean)["reason"]=="uniform_maintenance"


def test_reference_and_practice_exposure_and_mission_separation(tmp_path):
    obs=_observer(tmp_path)
    m=_rollout()
    shape=m["episode_done"].shape
    m["skill_reference_id"]=np.full(shape,-1,np.int32)
    m["skill_birth_vx_error"]=np.zeros(shape,np.float32)
    m["skill_birth_vy_error"]=np.zeros(shape,np.float32)
    m["skill_reference_settled"]=np.zeros(shape,np.float32)
    m["episode_length"]=np.zeros(shape,np.float32)
    # Same physical recovery task, but different cohort; a settled robot
    # that subsequently falls does not earn the mission verdict.
    m["skill_reference_id"][:,0]=0
    m["skill_reference_id"][:,1]=6
    m["skill_reference_id"][:,2]=0
    m["skill_birth_vx_error"][:,:3]=-.40
    m["specialist_focus"][:,:3]=1
    m["episode_done"][-1,:2]=1
    m["episode_length"][-1,:2]=[150,120]
    m["skill_reference_settled"][-1,:2]=1
    m["episode_fell"][-1,1]=1
    obs.observe(m,step=131072)
    row=obs.flush(force=True)
    assert row["world_frames"]==131072
    assert row["reference_frames_by_case"]==[64,0,0,0,0,0]
    assert row["practice_frames_by_case"]==[32,0,0,0,0,0]
    assert row["cells"]["reference_recovery/mission/0"]==dict(attempts=1,passes=1)
    assert row["cells"]["practice_recovery/mission/0"]==dict(attempts=1,passes=0)
    assert row["cells"]["practice_recovery/settled/0"]==dict(attempts=1,passes=1)
    m["episode_length"][-1,0]=149
    with pytest.raises(ValueError,match="horizon"):
        obs.observe(m,step=262144)


def test_checkpoint_teacher_continuation_and_rollback():
    from emu.live_skill_teacher import RecoveryTeacher
    a=RecoveryTeacher()
    batch=131072
    decisions=[]
    for i in range(36):
        counts=[[30,29 if i<16 else 15]]+[[0,0] for _ in range(5)]
        plan=a.observe(counts,batch)
        if plan is not None:
            decisions.append(plan)
    checkpoint=a.state_dict(120000000)
    b=RecoveryTeacher.from_state_dict(checkpoint,120000000)
    assert b.state_dict(120000000)==checkpoint
    for i in range(20):
        counts=[[25,12]]+[[0,0] for _ in range(5)]
        assert a.observe(counts,batch)==b.observe(counts,batch)
        assert a.state_dict(130000000)==b.state_dict(130000000)
    # Rollback discards later history and reinstates exact practice weights.
    rewind=RecoveryTeacher.from_state_dict(checkpoint,120000000)
    assert rewind.state_dict(120000000)==checkpoint
    assert rewind.clock<a.clock
    assert checkpoint["clock"]==36*batch  # detached immutable checkpoint copy
    with pytest.raises(ValueError,match="identity"):
        RecoveryTeacher.from_state_dict(checkpoint,120000001)
    with pytest.raises(ValueError,match="identity"):
        RecoveryTeacher.from_state_dict(None,120000000)
    corrupted={**checkpoint,"weights":[1.,0.,0.,0.,0.,0.]}
    with pytest.raises(ValueError,match="weights"):
        RecoveryTeacher.from_state_dict(corrupted,120000000)


def test_first_lip_physical_gates_do_not_require_torso_crossing():
    from emu.live_skill_stairs import first_lip_progress
    n=4
    active=np.array([True,True,True,False])
    lip=np.full(n,.028)
    goal=np.tile([1.6,.5],(n,1))
    xy=np.tile([.92,.5],(n,1))
    heading=np.zeros(n)
    metrics=dict(gravity_z=np.ones(n),support=np.ones(n),bad_contact=np.zeros(n))
    failed=np.zeros(n,dtype=bool)
    s=(np.zeros(n,bool),np.zeros(n,bool),np.zeros((n,2),int),
       np.zeros((n,2),bool),np.zeros(n,int),np.zeros(n,bool))
    def step(top,xy,failed=failed):
        nonlocal s
        s=first_lip_progress(np,active=active,lip=lip,goal=goal,xy=xy,
            heading=heading,metrics=metrics,top_feet=top,failed=failed,
            approach_seen=s[0],contact_seen=s[1],foot_streak=s[2],feet_seen=s[3],
            traverse_streak=s[4],traversed=s[5])
    # Genuine lead-foot contact before the body reaches the upper tile.
    lead=np.tile([True,False],(n,1))
    for _ in range(5):
        step(lead,xy)
    assert s[0].tolist()==[True,True,True,False]
    assert s[1].tolist()==[True,True,True,False]
    assert s[3][:,0].tolist()==[True,True,True,False]
    assert not np.any(s[5])
    # Second foot earns independent frames; no requirement that feet
    # contact the top simultaneously. The third robot never gets it.
    second=np.array([[False,True],[False,True],[True,False],[True,True]])
    for _ in range(5):
        step(second,xy)
    assert np.all(s[3],axis=1).tolist()==[True,True,False,False]
    beyond=np.tile([1.12,.5],(n,1))
    for _ in range(4):
        step(second,beyond)
    assert not np.any(s[5])
    step(second,beyond,np.array([False,True,False,False]))
    assert s[5].tolist()==[True,False,False,False]


def test_uphill_reference_assignment_and_once_only_credit():
    import jax
    import jax.numpy as jnp
    from emu.live_skill_uphill import uphill_assignment, assigned_height, prerequisite_payment, UP_HEIGHTS_M
    n=4096
    keys=jax.random.split(jax.random.PRNGKey(831),n)
    academy=jnp.arange(n)%2==0
    assign=jax.jit(lambda r,p:uphill_assignment(jax,jnp,keys,academy,1.,r,p),
                   static_argnums=(0,1))
    a=np.asarray(assign(.15,0.)); b=np.asarray(assign(.15,.15))
    np.testing.assert_array_equal(a>=0,(b>=0)&(b<6))
    np.testing.assert_array_equal(a[a>=0],b[a>=0])
    assert set(b)==set(range(-1,12))
    assert not np.any((b>=0)&(~np.asarray(academy)))
    h=np.asarray(assigned_height(jnp,jnp.asarray(b)))
    np.testing.assert_allclose(h[b>=0],np.asarray(UP_HEIGHTS_M)[b[b>=0]%6],atol=1e-7)
    assert np.all(h[b<0]==0.)
    failed=jnp.arange(n)%13==0
    zero=jnp.zeros((n,2),bool); first=zero.at[:,0].set(True); both=jnp.ones((n,2),bool)
    no=jnp.zeros(n,bool); yes=jnp.ones(n,bool)
    mask=(b>=6)&~np.asarray(failed)
    for stage,old,new,old_t,new_t in (('first_support',zero,first,no,no),
           ('bilateral',first,both,no,no),('traversal',both,both,no,yes)):
        pay=jax.jit(lambda p,q,v,w:prerequisite_payment(jnp,jnp.asarray(b),stage,p,q,v,w,failed))
        np.testing.assert_array_equal(np.asarray(pay(old,new,old_t,new_t)),4.*mask)
        assert not np.asarray(pay(new,new,new_t,new_t)).any()


def test_uphill_reference_review_repeated_gap():
    from emu.live_skill_uphill_review import diagnose_uphill, GATES
    def row(n,counts):
        return {g:dict(attempts=n,passes=p) for g,p in zip(GATES,counts)}
    blocks=[{4:row(100,[95,50,40,35])},{4:row(100,[90,45,40,34])}]
    r=diagnose_uphill(blocks)
    assert r['candidates'][0]['stage']=='bilateral'
    assert r['candidates'][0]['case']==4
    assert diagnose_uphill(blocks[:1])['status']=='insufficient_evidence'
    assert not diagnose_uphill([{4:row(10,[9,5,4,3])}]*2)['candidates']
    with pytest.raises(ValueError,match="Inconsistent"):
        diagnose_uphill([{4:row(100,[10,90,50,40])}]*2)


def test_uphill_fixed_reference_and_practice_ledger(tmp_path):
    obs=_observer(tmp_path)
    m=_rollout()
    shape=m["episode_done"].shape
    m["skill_uphill_id"]=np.full(shape,-1,np.int32)
    m["episode_length"]=np.zeros(shape,np.float32)
    for gate in ("approach","first_contact","first_support","each_foot_five","traversed"):
        m["skill_stair_"+gate]=np.zeros(shape,np.float32)
    # Noncontiguous done positions deliberately catch incorrect terminal
    # indexing when fixed task IDs happen to be small integers.
    cols=[44,117,3001]
    m["skill_uphill_id"][:,cols]=[4,10,-1]
    m["academy_active"][:,cols]=1
    m["skill_first_birth_lip_m"][:,cols]=.028
    m["episode_done"][-1,cols]=1
    m["episode_length"][-1,cols]=[350,100,1200]
    m["episode_fell"][-1,117]=1
    for gate in ("approach","first_contact","first_support","each_foot_five","traversed"):
        m["skill_stair_"+gate][-1,cols]=1
    obs.observe(m,step=131072)
    row=obs.flush(force=True)
    assert row["uphill_reference_frames_by_case"][4]==32
    assert row["uphill_practice_frames_by_case"][4]==32
    assert row["cells"]["reference_uphill/mission/4"]==dict(attempts=1,passes=1)
    assert row["cells"]["practice_uphill/mission/4"]==dict(attempts=1,passes=0)
    assert sum(v["attempts"] for k,v in row["cells"].items()
               if k.startswith("academy_physical_first_lip/up/traversed/"))==1


def test_adaptive_uphill_teacher_fade_stall_and_checkpoint():
    from emu.live_skill_uphill_teacher import UphillTeacher, REVIEW_STEPS
    weak=[100,96,92,40,30,25]
    good=[100,98,96,95,90,85]
    def block(t,row):
        return t.observe([[0]*6 for _ in range(4)]+[row,[0]*6], REVIEW_STEPS)
    a=UphillTeacher()
    block(a,weak); assert a.stage==[0]*6
    block(a,weak); assert a.stage[4]==2 and a.bonus[4]==4.
    assert a.weights[4]>1./6
    ck=a.state_dict(120000000)
    b=UphillTeacher.from_state_dict(ck,120000000)
    for expected in (3.,2.,1.,0.):
        for _ in range(2):
            assert block(a,good)==block(b,good)
        assert a.bonus[4]==expected
    assert a.stage[4]==0 and a.unassisted[4]
    assert a.state_dict(140000000)==b.state_dict(140000000)
    assert UphillTeacher.from_state_dict(ck,120000000).bonus[4]==4.
    s=UphillTeacher()
    block(s,weak); block(s,weak)
    for _ in range(6): block(s,weak)
    assert s.stage[4]==0 and s.cooldown[4]
    block(s,weak); block(s,weak)
    assert not s.cooldown[4] and s.stage[4]==0
    block(s,weak); block(s,weak)
    assert s.stage[4]==2
    z=UphillTeacher()
    for _ in range(10): block(z,[100,10,0,0,0,0])
    assert z.stage==[0]*6
    for _ in range(4): block(z,[0]*6)
    assert z.bonus==[0.]*6
    with pytest.raises(ValueError,match="nested"):
        block(z,[100,90,99,0,0,0])
    with pytest.raises(ValueError,match="identity"):
        UphillTeacher.from_state_dict(ck,120000001)


def test_adaptive_uphill_birth_latched_prerequisite():
    import jax
    import jax.numpy as jnp
    from emu.live_skill_uphill import uphill_assignment,birth_credit,assigned_prerequisite_payment
    keys=jax.random.split(jax.random.PRNGKey(821),4096)
    academy=jnp.arange(4096)%2==0
    assign=jax.jit(lambda w:uphill_assignment(jax,jnp,keys,academy,1.,.15,.15,w))
    a=np.asarray(assign(jnp.full(6,1./6)))
    uniform=jax.jit(lambda:uphill_assignment(jax,jnp,keys,academy,1.,.15,.15))()
    np.testing.assert_array_equal(a,np.asarray(uniform))
    b=assign(jnp.array([1./12]*4+[7./12,1./12]))
    bh=np.asarray(b)
    fixed=(a>=0)&(a<6)
    np.testing.assert_array_equal(fixed,(bh>=0)&(bh<6))
    np.testing.assert_array_equal(a[fixed],bh[fixed])
    np.testing.assert_array_equal(a>=6,bh>=6)
    s,v=birth_credit(jnp,b,jnp.array([0,1,2,3,2,0]),jnp.array([0.,4.,3.,2.,1.,0.]))
    later_s,later_v=birth_credit(jnp,b,jnp.zeros(6),jnp.zeros(6))
    assert not np.asarray(later_s).any() and not np.asarray(later_v).any()
    assert not np.asarray(v)[bh<6].any()
    old=jnp.tile(jnp.array([[True,False]]),(4096,1))
    new=jnp.ones_like(old)
    no=jnp.zeros(4096,bool)
    pay=jax.jit(lambda:assigned_prerequisite_payment(jnp,b,s,v,old,new,no,no,no))()
    np.testing.assert_array_equal(np.asarray(pay),np.where(np.asarray(s)==2,np.asarray(v),0.))
    repeat=assigned_prerequisite_payment(jnp,b,s,v,new,new,no,no,no)
    assert not np.asarray(repeat).any()


def test_neutral_ppo_reset_packet_for_combined_teachers():
    # Host-only exact production params_array, the path that failed in
    # make_trainer(env). No MJX world or GPU needed for this regression.
    from emu.environment import EmuMJXEnv
    from emu.contracts import P,UNIFIED_TASK_PARAM_DIM
    env=EmuMJXEnv.__new__(EmuMJXEnv)
    env.jnp=np
    env.sensor_noise_scale=1.
    env.terrain_step_max_m=.055
    env.academy_height_cap_m=.055
    env.exam_fixture_steps=None
    env.skill_practice_recovery_share=.15
    env.skill_adaptive_uphill=True
    packet=env.params_array({})
    assert packet.shape==(UNIFIED_TASK_PARAM_DIM,)
    np.testing.assert_allclose(packet[P.skill_practice_weight_0:P.skill_practice_weight_5+1],1/6)
    np.testing.assert_allclose(packet[P.skill_uphill_weight_0:P.skill_uphill_weight_5+1],1/6)
    assert not packet[P.skill_uphill_stage_0:P.skill_uphill_bonus_5+1].any()
    with pytest.raises(ValueError,match="Incomplete teacher"):
        env.params_array({"skill_practice_weight_0":.3})
    with pytest.raises(ValueError,match="Invalid recovery"):
        env.params_array({f"skill_practice_weight_{i}":0. for i in range(6)})
    with pytest.raises(ValueError,match="Invalid adaptive uphill"):
        env.params_array({f"skill_uphill_weight_{i}":0. for i in range(6)})
