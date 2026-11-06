"""v5052 destination-based stair lesson: JAX/NumPy-compatible pure geometry.

Only teacher-side reward and completion checks use these fields. Nothing is
added to the 360-input deployed actor. The target is fixed at *birth* on the
opposite side of one real 75cm tile boundary; it must not chase the robot.
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
