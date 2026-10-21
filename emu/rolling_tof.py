"""Two V-shaped 51-cell (rear -10cm to front +40cm) measured terrain tracks.

At each control step, optical-flow vx/vy and IMU yaw advect a sparse map in
the robot's gravity-levelled horizontal coordinate system. The two rays and
rear tripod ToF are real *measured* samples; no simulator height/pose/velocity
may enter the map's reconstruction. Unknown space keeps near-zero confidence.

V apex = measured ground hit of rear ToF (nominal x=-4cm, y=0).
Both paths spread to calibrated +/-5cm at nominal forward x=20cm and
continue outward to 40cm. Pitch/torso-height change ray-hit positions, not
the stored x-axis calibration. Five requested x locations (-5/0/5/10/15cm) per side feed PPO.
"""
from __future__ import annotations

GRID_SPACING_M = .01
GRID_START_M = -.10
GRID_LOOKAHEAD_M = .40
GRID_CELLS = 51                    # -10, -9, ..., 39, 40 cm
FORWARD_REFERENCE_X_M = .20        # where the two physical rays aim at neutral pose
FRONT_LANES_Y_M = (-.05, .05)      # right, left
REAR_REFERENCE_X_M = -.04
OUTPUT_SAMPLE_X_M = (-.05, 0., .05, .10, .15)
OUTPUT_SAMPLE_INDICES = (5, 10, 15, 20, 25)
CONFIDENCE_TIME_S = .60
CROSS_TRACK_SIGMA_M = .023         # width of uncertainty around each scanned V arm
SENSOR_MAP_REVISION = "rear_2d_apex_tof_v_tracks_51x2_v5"


def grid_positions(xp):
    return GRID_START_M + xp.arange(GRID_CELLS, dtype=xp.float32) * GRID_SPACING_M


def v_track_y(xp, x_m, apex_x_m, apex_y_m=None):
    """Left/right 2D path positions at body-relative, gravity-levelled x.

    apex_x may change as pitch, height and rear range change. The V is a
    straight-line *approximation*, never a substitute for a full 2D map.
    Returns [worlds,2,position_count] (or matching broadcast shape).
    """
    x = xp.asarray(x_m)
    apex = xp.asarray(apex_x_m)
    apex_y = xp.zeros_like(apex) if apex_y_m is None else xp.asarray(apex_y_m)
    targets = xp.asarray(FRONT_LANES_Y_M, dtype=xp.float32)[None,:,None]
    t = xp.maximum(x-apex,0.) / xp.maximum(FORWARD_REFERENCE_X_M-apex,.08)
    return apex_y+(targets-apex_y)*t


def empty_map(xp, worlds: int):
    z = xp.zeros((worlds, 2, GRID_CELLS), dtype=xp.float32)
    return z, z


def advance_map(xp, heights, confidences, vx_body, vy_body, yaw_rad_s,
                flow_confidence, *, dt_s: float = .02,
                old_apex_x_m=None, new_apex_x_m=None,
                old_apex_y_m=None, new_apex_y_m=None):
    """Apply full 2D rigid-body motion to a V-shaped, sparsely measured map.

    Reverse-warp target NEW-body V points into the OLD-body V. This permits
    different signed longitudinal shifts of each branch under yaw, backward
    walking, turning and sideways movement. Source candidates are weighted
    by real lateral distance from each old arm; severe departures from both
    lines invalidate memory rather than manufacturing ground measurements.
    """
    old_h=xp.asarray(heights)
    old_c=xp.clip(xp.asarray(confidences),0.,1.)
    worlds=old_h.shape[0]
    vx=xp.asarray(vx_body)
    vy=xp.asarray(vy_body)
    yaw=xp.asarray(yaw_rad_s)
    quality=xp.clip(xp.asarray(flow_confidence),0.,1.)
    reliable=quality>.10
    # Flow dropout removes translation but NOT independent gyro yaw.
    vx=xp.where(reliable,vx,0.)[:,None,None]
    vy=xp.where(reliable,vy,0.)[:,None,None]
    theta=yaw[:,None,None]*dt_s
    cs,sn=xp.cos(theta),xp.sin(theta)
    xx=grid_positions(xp)[None,None,:]
    apex_old=(xp.full((worlds,1,1),REAR_REFERENCE_X_M,dtype=xp.float32)
              if old_apex_x_m is None else xp.asarray(old_apex_x_m)[:,None,None])
    apex_new=(apex_old if new_apex_x_m is None
              else xp.asarray(new_apex_x_m)[:,None,None])
    old_apex_y=(xp.zeros_like(apex_old) if old_apex_y_m is None
                else xp.asarray(old_apex_y_m)[:,None,None])
    new_apex_y=(old_apex_y if new_apex_y_m is None
                else xp.asarray(new_apex_y_m)[:,None,None])
    yy=v_track_y(xp,xx,apex_new,new_apex_y)
    back_x=cs*xx - sn*yy + vx*dt_s
    back_y=sn*xx + cs*yy + vy*dt_s
    index=(back_x-GRID_START_M)/GRID_SPACING_M
    in_bounds=(index>=0.) & (index<=GRID_CELLS-1)
    idx=xp.clip(index,0.,GRID_CELLS-1)
    lo=xp.floor(idx).astype(xp.int32)
    hi=xp.minimum(lo+1,GRID_CELLS-1)
    frac=idx-lo

    sum_w=xp.zeros_like(back_x)
    sum_h=xp.zeros_like(back_x)
    best_w=xp.zeros_like(back_x)
    for side in range(2):          # static loop, JAX-traceable
        c0=xp.take_along_axis(old_c[:,side:side+1,:],lo,axis=2)
        c1=xp.take_along_axis(old_c[:,side:side+1,:],hi,axis=2)
        h0=xp.take_along_axis(old_h[:,side:side+1,:],lo,axis=2)
        h1=xp.take_along_axis(old_h[:,side:side+1,:],hi,axis=2)
        # Expose only evidence: no interpolation across two missing cells.
        amount=(1.-frac)*c0+frac*c1
        local_h=((1.-frac)*c0*h0+frac*c1*h1)/xp.maximum(amount,1e-8)
        fraction=(xp.maximum(back_x-apex_old,0.) /
                  xp.maximum(FORWARD_REFERENCE_X_M-apex_old,.08))
        arm_y=(old_apex_y +
               (xp.asarray(FRONT_LANES_Y_M[side],dtype=xp.float32)-old_apex_y)*fraction)
        lateral=back_y-arm_y
        distance_weight=xp.exp(-.5*(lateral/CROSS_TRACK_SIGMA_M)**2)
        weighted=xp.where(in_bounds,amount*distance_weight,0.)
        sum_w=sum_w+weighted
        sum_h=sum_h+weighted*local_h
        best_w=xp.maximum(best_w,weighted)

    # With no confident source, zero the height *and* confidence.
    good=(sum_w>1e-4)&xp.isfinite(sum_h)
    candidate=sum_h/xp.maximum(sum_w,1e-8)
    # Do not inflate confidence just because BOTH almost-identical branches
    # overlap at the rear anchor.
    support=xp.clip(best_w,0.,1.)
    base_decay=xp.exp(-dt_s/CONFIDENCE_TIME_S)
    turn_penalty=xp.clip(1.-xp.abs(yaw)[:,None,None]*dt_s,0.,1.)
    # Low flow quality reduces trust gradually, NOT by multiplication with
    # flow_confidence every 20ms (that would erase 25Hz measurements).
    flow_penalty=.96+.04*quality[:,None,None]
    uncertainty=base_decay*turn_penalty*flow_penalty
    c=xp.where(good,support*uncertainty,0.)
    return xp.where(good,candidate,0.),xp.clip(c,0.,1.)


def insert_measurement(xp, heights, confidences, sample_x_m,
                       sample_height_m, sample_valid, *,
                       sample_y_m=None, apex_x_m=None,
                       apex_y_m=None, measurement_confidence=None):
    """Insert actual gravity-levelled 2D laser hit(s) in the V-memory.

    Each lane's measured hit is deposited only onto nearby 1cm X cells,
    discounted by distance from that V track at the hit's true Y. The rear
    central ToF may be inserted into BOTH tracks at its actual -X/y=0 hit.
    No invented points between unrelated observations.
    """
    h=xp.asarray(heights)
    old=xp.clip(xp.asarray(confidences),0.,1.)
    worlds=h.shape[0]
    mx=xp.asarray(sample_x_m)
    mz=xp.asarray(sample_height_m)
    ok=(xp.asarray(sample_valid) & xp.isfinite(mx) & xp.isfinite(mz) &
        (mx>=GRID_START_M-.0005)&(mx<=GRID_LOOKAHEAD_M+.0005))
    apex=(xp.full((worlds,1,1),REAR_REFERENCE_X_M,dtype=xp.float32)
          if apex_x_m is None else xp.asarray(apex_x_m)[:,None,None])
    lane_x=xp.clip(mx,GRID_START_M,GRID_LOOKAHEAD_M)
    grid=grid_positions(xp)[None,None,:]
    spatial_weight=xp.clip(1.-xp.abs(grid-lane_x[:,:,None])/GRID_SPACING_M,0.,1.)
    if sample_y_m is not None:
        my=xp.asarray(sample_y_m)
        apex_y=(xp.zeros_like(apex) if apex_y_m is None
                else xp.asarray(apex_y_m)[:,None,None])
        expected_y=v_track_y(xp,lane_x[:,:,None],apex,apex_y)[:,:,0]
        cross=xp.exp(-.5*((my-expected_y)/CROSS_TRACK_SIGMA_M)**2)
        spatial_weight=spatial_weight*cross[:,:,None]
    confidence=xp.ones_like(mx) if measurement_confidence is None else xp.clip(
        xp.asarray(measurement_confidence),0.,1.)
    weight=xp.where(ok[:,:,None],spatial_weight*confidence[:,:,None],0.)
    # New observations replace stale ones without overcounting old evidence.
    denominator=old+weight
    combined=(h*old+mz[:,:,None]*weight)/xp.maximum(denominator,1e-8)
    fresh=xp.maximum(old,weight)
    return xp.where(fresh>0.,combined,0.),fresh


def readout(xp, heights, confidences):
    """Ten heights (-5/0/5/10/15cm per side), plus ten confidences."""
    idx=xp.asarray(OUTPUT_SAMPLE_INDICES)
    h=xp.take(heights,idx,axis=2)
    c=xp.take(confidences,idx,axis=2)
    return h.reshape((heights.shape[0],10)),xp.clip(c.reshape((heights.shape[0],10)),0.,1.)


def measured_height_and_x(xp, ranges, valid, tripod_ranges,
                          ground_normal_body, tripod_valid,
                          *, mounts_body, beams_body,
                          tripod_mounts_body, tripod_beams_body,
                          gravity_body=None, include_rear=False):
    """Project real sensor returns into gravity-levelled, body-heading X/Y.

    Slant laser hit and rear tripod hit are derived ONLY from delivered
    ranges, calibrated mounts, and IMU gravity. Body pitch/roll and changing
    height affect the projected hit coordinate automatically. The rear
    ToF is the local-zero height anchor shared by both V paths.

    include_rear=True returns x,y,z,valid for two forward hits, followed
    by rear_x,rear_y,rear_valid. Legacy four-return use is also supported.
    """
    mounts=xp.asarray(mounts_body)
    beams=xp.asarray(beams_body)
    tm=xp.asarray(tripod_mounts_body)
    tb=xp.asarray(tripod_beams_body)
    rear=tm[2][None,:]+tripod_ranges[:,2,None]*tb[2][None,:]
    point=mounts[None,:,:]+xp.asarray(ranges)[:,:,None]*beams[None,:,:]
    if gravity_body is None:
        normal=xp.asarray(ground_normal_body)
        up=normal/xp.maximum(xp.sqrt(xp.sum(normal*normal,axis=-1,keepdims=True)),1e-8)
    else:
        g=xp.asarray(gravity_body)
        up=-g/xp.maximum(xp.sqrt(xp.sum(g*g,axis=-1,keepdims=True)),1e-8)
    forward=xp.broadcast_to(xp.asarray((1.,0.,0.),dtype=point.dtype),up.shape)
    forward=forward-xp.sum(forward*up,axis=-1,keepdims=True)*up
    forward=forward/xp.maximum(xp.sqrt(xp.sum(forward*forward,axis=-1,keepdims=True)),1e-8)
    sideways=xp.cross(up,forward)
    mx=xp.sum(point*forward[:,None,:],axis=-1)
    my=xp.sum(point*sideways[:,None,:],axis=-1)
    # The local reference height uses the estimated tripod ground normal,
    # not a privileged world-Z or perfect simulator floor normal.
    height=xp.sum((point-rear[:,None,:])*ground_normal_body[:,None,:],axis=-1)
    good=(xp.asarray(valid)&xp.asarray(tripod_valid)[:,None]&
          xp.all(xp.isfinite(point),axis=-1)&xp.isfinite(height))
    height=xp.where(good,xp.clip(height,-.15,.15),0.)
    if not include_rear:
        return mx,height,good
    rx=xp.sum(rear*forward,axis=-1)
    ry=xp.sum(rear*sideways,axis=-1)
    rear_valid=(xp.asarray(tripod_valid)&xp.all(xp.isfinite(rear),axis=-1)&
                xp.isfinite(rx)&xp.isfinite(ry))
    return mx,my,height,good,rx,ry,rear_valid
