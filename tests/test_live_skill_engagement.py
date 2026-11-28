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


def test_engagement_observer_counts_real_payments_and_rewards(tmp_path):
    from emu.live_skill_observer import LiveSkillObserver
    # Four real frame records; terminal gate evidence remains episode based.
    shape=(2,2)
    names=["episode_done","episode_fell","specialist_focus","academy_active",
           "skill_first_birth_lip_m","academy_episode_first_lip_armed",
           "academy_episode_two_foot_ready","skill_first_goal_completed",
           "physical_step_height_m","physical_slope_deg","episode_length",
           "physical_heading_rad_after_step","skill_world_vx_after_step","cmd_vx",
           "skill_uphill_birth_stage","skill_uphill_birth_bonus",
           "skill_total_reward","reward_weighted_locomotion","reward_uphill_prerequisite"]
    names += [f"terrain_episode_{k}_{d}" for d in ("up","down") for k in ("attempt","success")]
    names += [f"slope_episode_{k}_{d}" for d in ("uphill","downhill","cross_left","cross_right")
              for k in ("attempt","stable","success")]
    names += ["skill_stair_"+g for g in ("approach","first_contact","first_support","each_foot_five","traversed")]
    m={k:np.zeros(shape,np.float32) for k in names}
    m["specialist_focus"].fill(3);m["academy_active"].fill(1)
    m["skill_uphill_id"]=np.array([[5,11],[5,11]],np.int32)
    m["skill_first_birth_lip_m"].fill(.032)
    m["physical_position_xy_after_step"]=np.zeros(shape+(2,),np.float32)
    m["episode_length"][:]=[[349,1],[350,2]]
    m["episode_done"][1,0]=1
    m["skill_stair_approach"][1,0]=1
    m["cmd_vx"].fill(.16)
    m["skill_world_vx_after_step"][:,0]=[.1,-.1]
    m["skill_total_reward"][:,0]=[2.,1.]
    m["reward_weighted_locomotion"][:,0]=[1.,.25]
    m["skill_uphill_birth_stage"][:,1]=4
    m["skill_uphill_birth_bonus"][:,1]=4
    m["reward_uphill_prerequisite"][1,1]=4
    obs=LiveSkillObserver(tmp_path,start_step=0,worlds=2,field_shape=(1,),
        field_bounds=((0,1),),source_commit="test",window_steps=1,uphill_profile="engagement")
    obs.observe(m,step=4)
    row=obs.flush()
    assert row["uphill_profile"]=="engagement"
    assert obs.last_uphill_counts[5]==[1,0,0,0,0,0,1]
    assert row["cells"]["practice_uphill_stage_4_frames/5"]=={"attempts":2,"passes":1}
    assert row["cells"]["practice_uphill_stage_4_births/5"]=={"attempts":1,"passes":1}
    assert row["cells"]["practice_uphill_stage_4_bonus_4/5"]=={"attempts":1,"passes":1}
    assert row["uphill_motion_reward_sums"]=={"forward/5":[1,2.,1.],"reverse/5":[1,1.,.25]}
    assert not obs.motion_reward_sums
    m["reward_uphill_prerequisite"][0,0]=4
    with pytest.raises(ValueError,match="actual uphill bonus"):
        obs.observe(m,step=8)
