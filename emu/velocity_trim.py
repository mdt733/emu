"""Bounded velocity-error accumulation on internal policy commands only.

The training reward and teacher must use the raw operator request. Neither
the simulator's true velocity nor the motor targets are read/modified here.
"""

def velocity_i_step(xp,requested,observed,confidence,previous,integral,
                    *,dt=0.02,ramp_s=0.4,multiplier=2.,
                    forward_limit=.40,reverse_limit=.12,lateral_limit=.14):
    """Vectorized controller: normalized signed shortfall ramps 0->1 in ramp_s."""
    if not (dt>0 and .1<=ramp_s<=5. and 1.<=multiplier<=3. and
            min(forward_limit,reverse_limit,lateral_limit)>0):
        raise ValueError('invalid velocity-I parameters')
    if multiplier==1.:
        return requested,xp.zeros_like(integral)
    req=xp.asarray(requested)
    est=xp.asarray(observed)
    confid=xp.asarray(confidence)
    valid=(confid>=.35) & xp.all(xp.isfinite(est),axis=1)
    active=xp.abs(req)>=.025
    changed=(req*previous<0)|(xp.abs(req-previous)>xp.maximum(.05,.5*xp.abs(previous)))
    shortfall=xp.clip((req-est)*xp.sign(req)/xp.maximum(xp.abs(req),.04),-1.,1.)
    candidate=xp.clip(integral+shortfall*(dt/ramp_s),0.,1.)
    state=xp.where(changed|~active,0.,
                   xp.where(valid[:,None],candidate,integral*.90))
    demand=req*(1.+(multiplier-1.)*state)
    demand=xp.stack((xp.clip(demand[:,0],-reverse_limit,forward_limit),
                     xp.clip(demand[:,1],-lateral_limit,lateral_limit)),axis=-1)
    return demand,state
