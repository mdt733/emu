"""v5052 destination-based stair lesson: JAX/NumPy-compatible pure geometry.

Only teacher-side reward and completion checks use these fields. Nothing is
added to the 360-input deployed actor. The target is fixed at *birth* on the
opposite side of one real 1m tile boundary; it must not chase the robot.
"""
from __future__ import annotations

from .terrain_carpet_mjx import TILE_M

DESTINATION_AFTER_LIP_M = 0.60
GOAL_X_TOL_M = 0.07
GOAL_Y_TOL_M = 0.12
GOAL_HEADING_TOL_RAD = 0.35
GOAL_STABLE_FRAMES = 5


def destination_from_birth(xp, xy, distance_after_lip=DESTINATION_AFTER_LIP_M):
    """Fixed destination 60cm beyond the next +X physical tile edge."""
    edge=(xp.floor(xy[:, 0]/TILE_M)+1.)*TILE_M
    return xp.stack((edge+distance_after_lip, xy[:, 1]), axis=1)


def signed_distance_progress(xp, goal_xy, before_xy, after_xy):
    """Metres of true distance reduction. Returning towards a target is not speed."""
    before=xp.sqrt(xp.sum((goal_xy-before_xy)**2,axis=1))
    after=xp.sqrt(xp.sum((goal_xy-after_xy)**2,axis=1))
    return xp.clip(before-after,-.02,.02)



# Short nursery mission: prove the first real lip, then move on to the next
# birth. A strict 60cm park-and-hold is a separate (harder) capability.
CLIMB_AFTER_LIP_M = 0.10
CLIMB_Y_TOL_M = 0.25
CLIMB_HEADING_TOL_RAD = 0.55
CLIMB_STABLE_FRAMES = 3


def school_first_foothold(xp, goal_xy, body_xy, feet_xy,
                          real_top_contacts, birth_lip_m, academy_active,
                          already_paid, previous_lips):
    """Single first-lip physical foothold, even before the torso crosses.

    Caller must supply TRUE upward-facing goal-prism contacts, NOT foot-height
    guesses, a collision with the vertical riser, or the robot's XY alone.
    Goal tile is the fixed *birth* destination; later recycled tiles cannot
    pay. Used only in the opt-in Stair School teacher, never actor inputs.
    """
    target=xp.floor(goal_xy/TILE_M).astype(xp.int32)
    source=target-xp.asarray([1,0],dtype=xp.int32)
    body=xp.floor(body_xy/TILE_M).astype(xp.int32)
    foot=xp.floor(feet_xy/TILE_M).astype(xp.int32)
    valid_foot=real_top_contacts & xp.all(foot==target[:,None,:],axis=2)
    on_course=(xp.all(body==source,axis=1) |
               xp.all(body==target,axis=1))
    new=(academy_active & ~already_paid & (previous_lips==0) &
         (xp.abs(birth_lip_m)>.0015) & on_course &
         xp.any(valid_foot,axis=1))
    return new,valid_foot


def school_uphill_clearance(xp, goal_xy, body_xy, feet_xyz,
                            foot_bottom, real_ground_top, birth_lip_m,
                            academy_active, already_paid, first_foothold_paid):
    """One-time REAL swing clearance close to the source side of an UP lip.

    This is a *precursor* reward, not proof of step success. It needs the
    foot's sphere bottom above the local real source prism by at least the
    measured upcoming riser height, while its XY approaches the fixed lip.
    Repeated tippy taps/hovering cannot collect more than one payment.
    """
    edge=goal_xy[:,0]-DESTINATION_AFTER_LIP_M
    near=(feet_xyz[:,:,0]>=edge[:,None]-.07) & (feet_xyz[:,:,0]<edge[:,None])
    valid_y=(xp.floor(feet_xyz[:,:,1]/TILE_M)==xp.floor(goal_xy[:,None,1]/TILE_M))
    lift=(foot_bottom-real_ground_top >=
          birth_lip_m[:,None]+.006)
    source_side=(body_xy[:,0]<edge) & (
        xp.floor(body_xy[:,1]/TILE_M)==xp.floor(goal_xy[:,1]/TILE_M))
    return (academy_active & ~already_paid & ~first_foothold_paid &
            (birth_lip_m>.0015) & source_side &
            xp.any(near & valid_y & lift,axis=1))


def short_climb_quality(xp, goal_xy, xy, heading_rad, gravity_z, support,
                        bad_contact, transfer_verified, birth_lip_m,
                        distance_after_lip_m=CLIMB_AFTER_LIP_M):
    """First signed prism lip: BOTH feet verified, upright beyond the edge.

    `transfer_verified` is established by five independent real-top contact
    frames PER FOOT on the first target tile; a single lead foot cannot pass.
    The birth lip is measured from the actual two prism surfaces, never an
    operator-requested height. No stationary low-slip requirement.
    """
    edge_x=goal_xy[:,0]-DESTINATION_AFTER_LIP_M
    heading=xp.abs(xp.arctan2(xp.sin(heading_rad),xp.cos(heading_rad)))
    return (transfer_verified & (xp.abs(birth_lip_m)>.0015) &
            (xy[:,0]>=edge_x+distance_after_lip_m) &
            (xy[:,0]<edge_x+TILE_M) &
            (xp.abs(xy[:,1]-goal_xy[:,1])<=CLIMB_Y_TOL_M) &
            (heading<=CLIMB_HEADING_TOL_RAD) &
            (gravity_z>=.80) & (support>=.5) & (bad_contact<.5))

def arrival_quality(xp, goal_xy, xy, heading_rad, gravity_z, support,
                    bad_contact, slip, transfer_verified):
    """Cannot certify success from position alone; require safe top transfer.

    Transfer verification comes from independent real-prism five-frame
    two-foot upper-tile contact rather than coordinate or reward flags.
    """
    dx=xp.abs(goal_xy[:,0]-xy[:,0])
    dy=xp.abs(goal_xy[:,1]-xy[:,1])
    heading=xp.abs(xp.arctan2(xp.sin(heading_rad),xp.cos(heading_rad)))
    return (transfer_verified &
            (dx<=GOAL_X_TOL_M) & (dy<=GOAL_Y_TOL_M) &
            (heading<=GOAL_HEADING_TOL_RAD) &
            (gravity_z>=.80) & (support>=.5) &
            (bad_contact<.5) & (slip<=.08))


def stable_arrival(xp, qualified, previous_streak, already_completed):
    """Require uninterrupted stable arrival and pay once per birth."""
    streak=xp.where(qualified,previous_streak+1,0)
    newly_completed=(streak>=GOAL_STABLE_FRAMES) & ~already_completed
    return streak, newly_completed
