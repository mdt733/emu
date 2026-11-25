"""Teacher-only first-lip milestones, independent of legacy reward gates.

Use real destination-tile top contacts even when the torso has not crossed.
Both feet must each earn five consecutive top-contact frames on that same
tile. No claim about load transfer is inferred from contact alone.
"""
from .step_goal import DESTINATION_AFTER_LIP_M, short_climb_quality

TRAVERSE_FRAMES=5
FOOT_SUPPORT_FRAMES=5


def first_lip_progress(xp, *, active, lip, goal, xy, heading, metrics, top_feet,
                       failed, approach_seen, contact_seen, foot_streak,
                       feet_seen, traverse_streak, traversed):
    valid=active & (xp.abs(lip)>.0015) & ~failed
    edge=goal[:,0]-DESTINATION_AFTER_LIP_M
    in_approach=(xy[:,0]>=edge-.15)&(xy[:,0]<edge)&(
        xp.abs(xy[:,1]-goal[:,1])<=.25)
    approach_seen=approach_seen | (valid & in_approach)
    good_feet=top_feet & valid[:,None]
    contact_seen=contact_seen | xp.any(good_feet,axis=1)
    foot_streak=xp.where(good_feet,foot_streak+1,0)
    feet_seen=feet_seen | (foot_streak>=FOOT_SUPPORT_FRAMES)
    bilateral=xp.all(feet_seen,axis=1)
    quality=short_climb_quality(xp,goal,xy,heading,metrics['gravity_z'],
        metrics['support'],metrics['bad_contact'],bilateral,lip)
    quality=quality & valid & xp.any(good_feet,axis=1)
    traverse_streak=xp.where(quality,traverse_streak+1,0)
    traversed=traversed | (traverse_streak>=TRAVERSE_FRAMES)
    return (approach_seen,contact_seen,foot_streak,feet_seen,
            traverse_streak,traversed)
