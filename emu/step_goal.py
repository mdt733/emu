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


def short_climb_quality(xp, goal_xy, xy, heading_rad, gravity_z, support,
                        bad_contact, transfer_verified, birth_lip_m):
    """First signed prism lip: BOTH feet verified, upright beyond the edge.

    `transfer_verified` is established by five independent real-top contact
    frames PER FOOT on the first target tile; a single lead foot cannot pass.
    The birth lip is measured from the actual two prism surfaces, never an
    operator-requested height. No stationary low-slip requirement.
    """
    edge_x=goal_xy[:,0]-DESTINATION_AFTER_LIP_M
    heading=xp.abs(xp.arctan2(xp.sin(heading_rad),xp.cos(heading_rad)))
    return (transfer_verified & (xp.abs(birth_lip_m)>.0015) &
            (xy[:,0]>=edge_x+CLIMB_AFTER_LIP_M) &
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
