"""CPU contracts for the general uphill engagement classroom."""
import numpy as np
import pytest
from emu.live_skill_uphill_teacher import UphillTeacher,REVIEW_STEPS
from emu.live_skill_uphill import assigned_prerequisite_payment,uphill_heights
from emu.live_skill_stairs import first_lip_progress

def test_engagement_diagnosis_and_fading():
    a=UphillTeacher("engagement")
    weak=[100,2,1,1,0,0,20]
    contact=[100,20,18,10,5,4,95]
    good=[100,98,96,95,90,85,98]
    def block(t,row): return t.observe([[0]*7]*5+[row],REVIEW_STEPS)
    assert block(a,weak)["events"]==[]
    assert block(a,weak)["events"]==[dict(case=5,action="start",stage=4)]
    b=UphillTeacher.from_state_dict(a.state_dict(120000000),120000000,"engagement")
    for expected in (3.,2.,1.,0.):
        for _ in range(2): assert block(a,contact)==block(b,contact)
        assert a.bonus[5]==expected
    for _ in range(2): block(a,contact)
    assert a.stage[5]==5
    for expected in (3.,2.,1.,0.):
        for _ in range(2): block(a,good)
        assert a.bonus[5]==expected
    assert a.unassisted[5] and a.stage[5]==0
    with pytest.raises(ValueError): UphillTeacher.from_state_dict(b.state_dict(120000000),120000000,"cliff")
    assert uphill_heights("engagement")==uphill_heights("cliff")
    # Contact may precede the torso gate; it must not be falsely rejected.
    c=UphillTeacher("engagement")
    for _ in range(2): block(c,[100,95,94,40,30,20,10])
    assert c.stage[5]==2

def test_real_controlled_approach_and_once_only_credit():
    n=6
    no=np.zeros(n,bool); yes=np.ones(n,bool)
    xy=np.tile([.9,.5],(n,1)); goal=np.tile([1.6,.5],(n,1))
    feet=np.zeros((n,2),bool); contact=feet.copy(); contact[1,0]=True
    metrics={"gravity_z":np.ones(n),"support":np.ones(n),"bad_contact":np.zeros(n)}
    # 0 moves forward, 1 plants a foot, 2 reverses, 3 turns,
    # 4 is ordinary reference, 5 fails. No reward for ordinary/fail.
    failed=np.array([0,0,0,0,0,1],bool)
    s=first_lip_progress(np,active=yes,lip=np.full(n,.03),goal=goal,xy=xy,
        heading=np.array([0,0,0,1,0,0]),metrics=metrics,top_feet=contact,failed=failed,
        approach_seen=no,contact_seen=no,foot_streak=np.zeros((n,2),int),
        feet_seen=feet,traverse_streak=np.zeros(n,int),traversed=no,
        controlled_mask=yes,world_vx=np.array([.16,0.,-.1,.16,.16,.16]))
    ids=np.array([6,7,8,9,0,10]); stages=np.array([4,5,4,4,4,4])
    def pay(a,c):
        return assigned_prerequisite_payment(np,ids,stages,np.full(n,4.),feet,feet,no,no,failed,
            previous_approach=a,next_approach=s[0],previous_contact=c,next_contact=s[1])
    np.testing.assert_array_equal(pay(no,no),[4,4,0,0,0,0])
    assert not pay(s[0],s[1]).any()
