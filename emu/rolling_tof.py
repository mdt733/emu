"""Two-lane 20 cm optical-flow/IMU-compensated ToF memory.

There are 21 cells (0..20 cm inclusive) per side, but only [0,5,10,15]
are exposed to the actor. A *single* physical forward ray per side measures
the far ground; flow/gyro motion advects old measurements toward the feet.

Only sensor-estimated vx/vy/yaw and measured relative ToF heights may enter
this module. Never use simulator root qvel, actual box height, true floor
normal or world coordinates to update the actor's rolling profile.

JAX-friendly: fixed arrays, vectorised gathers, no Python loops over worlds.
"""
from __future__ import annotations

GRID_SPACING_M = .01
GRID_CELLS = 21
GRID_LOOKAHEAD_M = .20
LANES_Y_M = (-.05, .05)  # right then left
OUTPUT_SAMPLE_INDICES = (0, 5, 10, 15)
CONFIDENCE_TIME_S = .60
MAX_RELIABLE_YAW_RAD_S = 1.5
MAX_RELIABLE_LATERAL_M_S = .18
SENSOR_MAP_REVISION = "flow_advected_two_single_ray_tracks_v1"


def empty_map(xp, worlds: int):
    z = xp.zeros((worlds, 2, GRID_CELLS), dtype=xp.float32)
    return z, z


def advance_map(xp, heights, confidences, vx_body, vy_body, yaw_rad_s,
                flow_confidence, *, dt_s: float = .02):
    """Transport *measured* heights in robot-relative coordinates.

    An obstacle fixed in the environment advances from far to near by
    vx*dt in the robot's local frame. Differential yaw affects two lanes
    with opposite signs: local dx = vx - yaw*y. Side translation and yaw
    make independent straight lane memories less trustworthy, so confidence
    is discounted (but never turned into a fabricated flat height).
    Returns new values/confidence; no new measurement is inserted here.
    """
    x = xp.asarray(heights)
    conf = xp.asarray(confidences)
    vx = xp.asarray(vx_body)[:, None]
    vy = xp.asarray(vy_body)[:, None]
    yaw = xp.asarray(yaw_rad_s)[:, None]
    flow_ok = xp.asarray(flow_confidence)[:, None, None]
    lane = xp.asarray(LANES_Y_M)[None,:]
    # When optical flow is invalid, *do not pretend to know displacement*.
    delta = xp.where(flow_ok[:,:,0] > .10,
                     (vx - yaw * lane) * dt_s, 0.)
    positions = xp.arange(GRID_CELLS, dtype=xp.float32) * GRID_SPACING_M
    source_bin = (positions[None,None,:] + delta[:,:,None]) / GRID_SPACING_M
    in_grid = (source_bin >= 0.) & (source_bin <= GRID_CELLS-1)
    safe = xp.clip(source_bin, 0., GRID_CELLS-1)
    lo = xp.floor(safe).astype(xp.int32)
    hi = xp.minimum(lo+1, GRID_CELLS-1)
    w = safe - lo
    lc = xp.take_along_axis(conf, lo, axis=2)
    hc = xp.take_along_axis(conf, hi, axis=2)
    lh = xp.take_along_axis(x, lo, axis=2)
    hh = xp.take_along_axis(x, hi, axis=2)
    wl = (1.-w)*lc
    wh = w*hc
    denominator = wl+wh
    value = (wl*lh+wh*hh)/xp.maximum(denominator,1e-8)
    keep = (in_grid & (denominator > 1e-5) &
            xp.isfinite(value))
    # Bounded confidence decay protects against optical-flow drift and turns.
    base_decay = xp.exp(-dt_s/CONFIDENCE_TIME_S)
    turn_decay = xp.clip(1.- xp.abs(yaw) * dt_s * 2., 0., 1.)
    side_decay = xp.clip(1.- xp.abs(vy) * dt_s * 15., 0., 1.)
    # Measurement confidence is NOT a per-frame survival probability:
    # multiplying, e.g., 0.6 flow-confidence on all 50Hz updates would
    # erase the map after a few frames (0.6**25 ~= 0). Apply a modest
    # uncertainty penalty once per timestep instead.
    flow_decay = 0.96 + 0.04 * xp.clip(flow_ok,0.,1.)
    uncertainty = flow_decay * turn_decay[:,:,None] * side_decay[:,:,None]
    confidence = xp.where(keep, denominator*base_decay*uncertainty,0.)
    return xp.where(keep,value,0.), confidence


def insert_measurement(xp, heights, confidences, sample_x_m,
                       sample_height_m, sample_valid):
    """Add one measured height per lane at its *actual projected hit X*.

    The point lands in 0..20cm robot-relative space, which matters when
    a ray strikes a step RISER before reaching its nominal far-floor target.
    Deposit to at most the two neighbouring 1cm cells. A single valid
    hit cannot fill an unknown 10cm-wide interval.
    """
    grid_x = xp.arange(GRID_CELLS,dtype=xp.float32)*GRID_SPACING_M
    measured_x = xp.asarray(sample_x_m)
    measured_height = xp.asarray(sample_height_m)
    # Calibrated 20cm hits may round to 0.200000003 in float32;
    # accept 0.5mm tolerance, clamp the deposition, reject true outliers.
    valid = (xp.asarray(sample_valid) &
             xp.isfinite(measured_height) &
             xp.isfinite(measured_x) &
             (measured_x >= -.0005) &
             (measured_x <= GRID_LOOKAHEAD_M+.0005))
    clipped_x = xp.clip(measured_x,0.,GRID_LOOKAHEAD_M)
    weights = xp.clip(1.-xp.abs(
        grid_x[None,None,:]-clipped_x[:,:,None])/GRID_SPACING_M,0.,1.)
    weights = xp.where(valid[:,:,None], weights, 0.)
    old = xp.asarray(confidences)
    # New returns dominate old evidence at the observed location, never
    # rewrite unobserved cells as exact zero or infer values from simulator.
    total=old+weights
    blended=(heights*old+measured_height[:,:,None]*weights)/xp.maximum(total,1e-8)
    result_conf=xp.maximum(old,weights)
    return xp.where(result_conf>0.,blended,0.),result_conf


def readout(xp, heights, confidences):
    """Exactly 4 positions × 2 sides, lane-major, 8 values + 8 confidences."""
    idx=xp.asarray(OUTPUT_SAMPLE_INDICES)
    h=xp.take(heights,idx,axis=2)
    c=xp.take(confidences,idx,axis=2)
    return h.reshape((heights.shape[0],8)),xp.clip(c.reshape((heights.shape[0],8)),0.,1.)


def measured_height_and_x(xp, ranges, valid, tripod_ranges,
                          ground_normal_body, tripod_valid,
                          *, mounts_body, beams_body,
                          tripod_mounts_body, tripod_beams_body):
    """Transform measured slant returns into a *relative* ground sample.

    All inputs are sensor measurements or calibrated extrinsics. There is
    no simulator floor, obstacle or global reference anywhere in this
    inverse. If tripod ground fit is unavailable, keep the ray raw in the
    sensor queue but insert nothing in the spatial height map.
    """
    mounts = xp.asarray(mounts_body)
    beams = xp.asarray(beams_body)
    p0 = (xp.asarray(tripod_mounts_body)[0][None,:]+
          tripod_ranges[:,0,None]*xp.asarray(tripod_beams_body)[0][None,:])
    p = mounts[None,:,:]+ranges[:,:,None]*beams[None,:,:]
    dh=xp.sum((p-p0[:,None,:])*ground_normal_body[:,None,:],axis=-1)
    x=p[:,:,0]  # robot-relative forward position, not world X
    good = valid & tripod_valid[:,None] & xp.isfinite(dh)
    return x, xp.where(good,xp.clip(dh,-.15,.15),0.),good
