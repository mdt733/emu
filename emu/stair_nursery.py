"""Physical, episode-high-water stair credit. No privileged actor inputs.

Each component is bounded [0,1] and rewarded only on new episode maxima.
Touchdowns and stable completion are separate one-time MJX contact events.
"""
from __future__ import annotations


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
