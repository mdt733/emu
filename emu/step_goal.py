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



def downhill_nursery_progress(xp, previous, body_xy, foot_xyz,
                              foot_bottom, edge_x, source_top, lip_m,
                              destination_contact_seen, eligible):
    """Teach controlled descending, not the uphill foot-lift recipe.

    Positive credit is one-time per episode and requires real signed
    lower-tile geometry. Destination contact is certified separately by
    the existing physical prism foot-pair detector. All six signals are
    teacher-only; no hidden state enters the 360-input deployed actor.
    """
    forward=foot_xyz[:,:,0]
    lateral=foot_xyz[:,:,1]
    row=xp.floor(body_xy[:,1])
    corridor=(lateral>row[:,None]+.06)&(lateral<row[:,None]+.94)
    near=(forward>=edge_x[:,None]-.28)&(
          forward<=edge_x[:,None]+.35)&corridor
    magnitude=xp.maximum(.012,xp.abs(lip_m[:,None]))
    destination_top=source_top+lip_m[:,None]
    approach=xp.clip((body_xy[:,0]-(edge_x-.30))/.25,0.,1.)
    # The swing foot actually descends below the high source surface.
    reach=(xp.clip((source_top-foot_bottom-.003)/magnitude,0.,1.)*
           near*(forward>=edge_x[:,None]-.012))
    # Landing is a TRUE destination top contact, not a foot entering
    # the lower slab's XY area or unrealistically passing through a wall.
    # Keep the actual contact verdict Boolean until AFTER the logical
    # support test. Casting to float first breaks both NumPy and JAX AND.
    landed=xp.asarray(destination_contact_seen,dtype=xp.bool_)
    over=(landed & (forward>=edge_x[:,None]+.018) &
          (foot_bottom>=destination_top-.015) & near).astype(foot_bottom.dtype)
    transferred=xp.clip((body_xy[:,0]-(edge_x-.02))/.18,0.,1.)*(
        xp.any(destination_contact_seen,axis=1))
    progress=xp.stack((approach,reach[:,0],reach[:,1],
                       over[:,0],over[:,1],transferred),axis=1)
    best=xp.maximum(previous,xp.where(eligible[:,None],progress,previous))
    gains=xp.maximum(0.,best-previous)
    weights=xp.asarray((2.,4.,4.,6.,6.,7.),dtype=best.dtype)
    return best,xp.sum(gains*weights[None,:],axis=1),gains

def nursery_progress(xp, previous, body_xy, foot_xyz, foot_bottom, edge_x,
                     source_top, lip_m, upper_foothold_seen, eligible):
    """Return (best [N,6], payment [N], gains [N,6]).

    source_top samples the actual source prism at x=edge-25mm for each
    foot's Y. Advancing over the edge requires physically sufficient height.
    """
    forward=foot_xyz[:,:,0]
    fy=foot_xyz[:,:,1]
    row=xp.floor(body_xy[:,1])
    corridor=((fy>row[:,None]+.06)&(fy<row[:,None]+.94))
    near=((forward>=edge_x[:,None]-.28)&
          (forward<=edge_x[:,None]+.15)&corridor)
    approach=xp.clip((body_xy[:,0]-(edge_x-.30))/.25,0.,1.)
    lift=xp.clip((foot_bottom-source_top-.003)/
                 xp.maximum(.060,lip_m[:,None]+.012),0.,1.)*near
    over=xp.clip((forward-(edge_x[:,None]-.05))/.15,0.,1.)*(
        (foot_bottom>=source_top+lip_m[:,None]+.002)&near)
    transferred=xp.clip((body_xy[:,0]-(edge_x-.02))/.18,0.,1.)*(
        xp.any(upper_foothold_seen,axis=1))
    progress=xp.stack((approach,lift[:,0],lift[:,1],
                       over[:,0],over[:,1],transferred),axis=1)
    best=xp.maximum(previous,xp.where(eligible[:,None],progress,previous))
    gains=xp.maximum(0.,best-previous)
    weights=xp.asarray((2.,6.,6.,5.,5.,7.),dtype=best.dtype)
    return best,xp.sum(gains*weights[None,:],axis=1),gains


def two_lip_continuation_success(xp, *, academy_active, supported_lips,
                                 first_lip_two_feet, half_metre_after_second,
                                 samples, mean_vx, mean_cmd_vx,
                                 gravity_z, support, bad_contact,
                                 yaw_error, terminal,
                                 second_lip_two_feet=True, required_lips=2):
    """Teacher-side C graduation: no first-lip or ballistic finish credit.

    'supported_lips' counts distinct novel MJX prism crossings.
    The second transfer is independently proven by five frames of both
    actual feet on the target top surface, not merely a body crossing.
    D retains the 50cm post-transfer milestone until the separate recent
    command window becomes valid; C used only the event frame.
    20 recent command samples span >=0.4s. A lunge cannot earn success.
    This check must never enter the deployable 360-input observation.
    """
    if required_lips not in (1,2):
        raise ValueError('Supported-lip lesson must require 1 or 2 crossings')
    return (academy_active & (supported_lips>=required_lips) &
            first_lip_two_feet & second_lip_two_feet &
            half_metre_after_second & (samples>=20) &
            (mean_cmd_vx>=.075) & (mean_vx>=.05) &
            (xp.abs(mean_vx-mean_cmd_vx)<=.10) &
            (gravity_z>=.80) & (support>=.50) &
            (bad_contact<.50) & (xp.abs(yaw_error)<.35) &
            ~terminal)


def stair_tutor_foot_progress(xp, previous, body_xy, foot_xyz,
                             foot_bottom, edge_x, source_top, lip_m,
                             foot_radius_m, eligible):
    """A *temporary* 50mm UP lesson, scored on actual geometry, never actor inputs.

    [approach, foot0 lift, foot1 lift, foot0 over, foot1 over, body escrow].
    The first five high-water increments pay ONCE. Body escrow is recorded
    here but only paid by stair_tutor_recovery after authentic foot support
    and sustained recovery. No fixed landing X: any safe top foothold works.
    This intentionally leaves the historical nursery_progress as the
    same-physics control arm for a *short* A/B trial.
    """
    forward=foot_xyz[:,:,0]
    fy=foot_xyz[:,:,1]
    row=xp.floor(body_xy[:,1])
    corridor=(fy>row[:,None]+.06)&(fy<row[:,None]+.94)
    edge=edge_x[:,None]
    near=(forward>=edge-.28)&(forward<=edge+.22)&corridor
    # The bottom of the real foot (not its centre) must clear the actual
    # upstairs prism surface. Full clearance saturation avoids high kicks.
    upper=source_top+lip_m[:,None]
    safe_x=edge+foot_radius_m[None,:]+.015
    clearance=foot_bottom>=upper+.005
    lift=(xp.clip((foot_bottom-source_top-.003)/
                  xp.maximum(.060,lip_m[:,None]+.012),0.,1.)*
          near*(forward<=safe_x+.02))
    over=(xp.clip((forward-(edge-.04))/
                  (foot_radius_m[None,:]+.055),0.,1.)*
          clearance*near)
    approach=xp.clip((body_xy[:,0]-(edge_x-.30))/.25,0.,1.)
    # Progress is measured at the *current* body pose on recovery. This
    # high-water escrow entry is for diagnostics only, never the payout.
    escrow=xp.clip((body_xy[:,0]-(edge_x-.20))/.45,0.,1.)
    progress=xp.stack((approach,lift[:,0],lift[:,1],
                       over[:,0],over[:,1],escrow),axis=1)
    best=xp.maximum(previous,xp.where(eligible[:,None],progress,previous))
    gains=xp.maximum(0.,best-previous)
    # Lift and reach make a useful manoeuvre discoverable; a flailing toe
    # cannot earn the majority of the episode's lesson reward.
    weights=xp.asarray((1.,5.,5.,4.,4.,0.),dtype=best.dtype)
    payment=xp.sum(gains*weights[None,:],axis=1)
    return best,payment,gains


def stair_tutor_recovery(xp, *, body_xy, edge_x, transfer_seen,
                         upper_top_now, pitch_rad, pitch_rate_rad_s,
                         gravity_z, support, bad_contact,
                         vx_samples, mean_vx, mean_cmd_vx,
                         previous_streak, previous_paid, eligible):
    """Escrow matures only after 0.4s of supported, moving, upright recovery.

    Both feet must have *independently* planted for >=3 frames on the
    assigned upper tile, but need not stand there simultaneously. At least
    one foot must still have real upper-top contact. Forward motion, rather
    than standing still, is required. A forward chassis lunge then fall
    yields no escrow payment. The score uses CURRENT x, never maximum x.
    """
    pose=(body_xy[:,0]>=edge_x+.10)&(body_xy[:,0]<=edge_x+.80)
    stable=(xp.abs(pitch_rad)<=.35)&(xp.abs(pitch_rate_rad_s)<=.60)&(
        gravity_z>=.82)&(support>=.5)&(bad_contact<.5)
    moving=(vx_samples>=20)&(mean_cmd_vx>=.075)&(mean_vx>=.05)&(
        xp.abs(mean_vx-mean_cmd_vx)<=.10)
    qualified=(eligible&transfer_seen&xp.any(upper_top_now,axis=1)&
               pose&stable&moving)
    streak=xp.where(qualified,previous_streak+1,0)
    released=qualified&(streak>=20)&~previous_paid
    # 20 frames AFTER both planted and genuine top support, not a single
    # lucky post-lip snapshot. Capped escrow uses current recovered x.
    progress=xp.clip((body_xy[:,0]-(edge_x-.20))/.45,0.,1.)
    payment=xp.where(released,24.*progress,0.)
    return streak,previous_paid|released,payment,released
