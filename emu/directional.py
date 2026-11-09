"""Shared 360-input MJX sensor and frozen-trace math.

Only production callers: one PPO heading-rate correction and MJX foot-contact
masks; a 4×4 array-only plane estimate and simulated/estimated optical flow.
The historical directional-course examiner and three-ray tripod forward
simulator live in immutable Git history, not in this runtime module.
"""
from __future__ import annotations

import math


def heading_rate_request(xp, requested_rate, target_heading, estimated_heading,
                         *, kp, max_rate):
    """Shared NumPy/JAX P law; caller owns heading integration and sensor validity.

    This does not act on servos. It returns a bounded *policy yaw command*
    and the shortest signed target-vs-estimate heading error.
    """
    error = xp.arctan2(xp.sin(target_heading - estimated_heading),
                      xp.cos(target_heading - estimated_heading))
    return xp.clip(requested_rate + kp * error, -max_rate, max_rate), error


def contact_pair_mask(xp, contact_geom1, contact_geom2, contact_distance,
                      floor_geom: int, foot_geoms, *, tolerance_m=0.002):
    """[worlds, contacts] -> [worlds, two feet] active floor/foot geometry pairs.

    MJX pads some contact slots; a nonnegative near-contact distance may still
    indicate a geometric pair, not confirmed impulse. Use the conservative
    near-zero distance threshold and report it as contact-pair evidence.
    """
    result = []
    for foot in foot_geoms:
        pair = (((contact_geom1 == floor_geom) & (contact_geom2 == foot)) |
                ((contact_geom2 == floor_geom) & (contact_geom1 == foot)))
        result.append(xp.any(pair & (contact_distance <= tolerance_m), axis=-1))
    return xp.stack(result, axis=-1)


# Legacy tripod *metadata only*: schema-2 policy hashes still require these
# constants, even though v5045 never simulates a tripod or trains a 232 actor.
TOF_TRIPOD_MOUNTS_BODY_M = (
    (.035, -.050, -.025),
    (.035,  .050, -.025),
    (-.040, .000, -.025),
)
TOF_TRIPOD_BEAMS_BODY = ((0.,0.,-1.),)*3
TOF_MIN_RANGE_M = .035
TOF_MAX_RANGE_M = .60


def fit_forward_array_ground(xp, ranges, valid, gravity_body):
    """Estimate grade, depth and *visibility*, solely from a 4x4 range array.

    One least-squares plane fit to the actual returned ray endpoints, with
    a residual rejection gate across all visible cells. It does not touch
    simulator normals or the old three physical ToF mounts. Mixed riser/top
    returns are NOT passed off as a smooth ground plane: low-confidence
    estimates become invalid while the original 16 raw rays remain available.

    Works in NumPy and JAX. Invalid rays cannot bias plane fitting.
    """
    from .tof_geometry import TOF_RAYS_BODY
    rays=xp.asarray(TOF_RAYS_BODY,dtype=xp.float32)
    d=xp.asarray(ranges)
    mask=xp.asarray(valid,dtype=xp.bool_) & xp.isfinite(d) & (d>=.04) & (d<=1.5)
    weight=mask.astype(xp.float32)
    points=rays[None,:,:]*xp.where(mask,d,0.)[:,:,None]
    x,y,z=points[:,:,0],points[:,:,1],points[:,:,2]
    design=xp.stack((x,y,xp.ones_like(x)),axis=-1)
    gram=xp.einsum('eki,ek,ekj->eij',design,weight,design)
    rhs=xp.einsum('eki,ek,ek->ei',design,weight,z)
    # Bias only the plane slopes to avoid singular layouts without inventing
    # an imaginary ground depth at the camera.
    reg=xp.asarray([.00005,.00005,.0000001],dtype=xp.float32)
    coefficients=xp.linalg.solve(
        gram+xp.eye(3,dtype=xp.float32)[None,:,:]*reg[None,None,:],
        rhs[...,None])[...,0]
    a,b,c=coefficients[:,0],coefficients[:,1],coefficients[:,2]
    normal=xp.stack((-a,-b,xp.ones_like(a)),axis=1)
    normal=normal/xp.maximum(xp.linalg.norm(normal,axis=1,keepdims=True),1e-7)
    gravity=xp.asarray(gravity_body)
    up=-gravity/xp.maximum(xp.linalg.norm(gravity,axis=1,keepdims=True),1e-7)
    normal=xp.where(xp.sum(normal*up,axis=1,keepdims=True)<0,-normal,normal)
    forward=xp.asarray([1.,0.,0.],dtype=xp.float32)[None,:]-up[:,0:1]*up
    forward=forward/xp.maximum(xp.linalg.norm(forward,axis=1,keepdims=True),1e-7)
    sideways=xp.cross(up,forward)
    cosine=xp.sum(normal*up,axis=1)
    count=xp.sum(weight,axis=1)
    estimated=xp.einsum('eki,ei->ek',design,coefficients)
    residual=xp.sqrt(xp.sum(weight*(estimated-z)**2,axis=1)/xp.maximum(count,1.))
    depth=-c
    # At least six non-collinear returns. Planar residual <= 8mm avoids
    # interpreting a step riser and upper platform as one smooth slope.
    good=(count>=6.) & (residual<.008) & (cosine>.25) & (
        depth>=.035) & (depth<=.60) & (
        xp.linalg.norm(gravity,axis=1)>.1)
    # A real step can invalidate the *single-plane slope*, yet optical flow
    # still needs a depth scale. Infer it from the four steep central array
    # rays (rather than returning zero velocity at precisely a step).
    middle=xp.asarray((9,10,13,14),dtype=xp.int32)
    middle_valid=xp.take(mask,middle,axis=1)
    central_count=xp.sum(middle_valid.astype(xp.float32),axis=1)
    centre_dirs=xp.take(-rays[:,2],middle,axis=0)
    central=xp.sum(xp.take(d,middle,axis=1)*centre_dirs[None,:]*
                   middle_valid.astype(xp.float32),axis=1)/xp.maximum(central_count,1.)
    optical_depth=xp.where(good,depth,central)
    depth_valid=(good | (central_count>=2)) & (
        optical_depth>=.035)&(optical_depth<=.60)
    confidence=xp.where(good,
        xp.clip(count/12.,0.,1.)*xp.clip(1.-residual/.008,0.,1.),0.)
    return dict(
        valid=good,depth_valid=depth_valid,
        ground_forward_grade_rad=xp.where(
            good,xp.arctan2(-xp.sum(normal*forward,axis=1),cosine),0.),
        ground_sideways_grade_rad=xp.where(
            good,xp.arctan2(-xp.sum(normal*sideways,axis=1),cosine),0.),
        ground_visibility_confidence=confidence,
        optical_depth_m=xp.where(depth_valid,optical_depth,0.),
        fit_residual_m=residual,
        valid_count=count,
    )


def simulate_downward_optical_flow(xp, body_velocity_mps, body_gyro_rad_s,
                                   optical_depth_m, *, flow_error_rad_s=None):
    """Angular image flow at the optical axis of a down-looking camera.

    Forward optics uses TRUE simulator body velocity, body-local gyro and
    true ray/plane intersection depth to generate noisy *measurements*.
    No true velocity enters the inverse estimator. Optical boresight is
    body -Z; image horizontal axes are body X/Y. Units rad/s approximately.

    At the boresight, yaw-induced image translation is zero to first order,
    while roll/pitch rotation must be compensated. Does not model image
    texture, pixel quantization, FOV, self-occlusion, blur or sensor latency.
    """
    v=xp.asarray(body_velocity_mps)
    w=xp.asarray(body_gyro_rad_s)
    depth=xp.asarray(optical_depth_m)
    valid=(xp.isfinite(depth)&(depth>=TOF_MIN_RANGE_M)
           &(depth<=TOF_MAX_RANGE_M)
           &xp.all(xp.isfinite(v),axis=-1)&xp.all(xp.isfinite(w),axis=-1))
    safe=xp.maximum(depth,TOF_MIN_RANGE_M)
    angular=xp.stack((-v[:,0]/safe+w[:,1],
                      -v[:,1]/safe-w[:,0]),axis=-1)
    if flow_error_rad_s is not None:
        angular=angular+xp.asarray(flow_error_rad_s)
    valid=valid&xp.all(xp.isfinite(angular),axis=-1)
    return xp.where(valid[:,None],angular,0.),valid


def estimate_flow_velocity(xp, flow_rad_s, imu_gyro_body_rad_s,
                           tof_estimated_optical_depth_m, *,
                           flow_valid=None, tof_valid=None):
    """Invert a gyro-compensated downward optical-flow measurement.

    Returns *body-frame* horizontal vxy, never world velocity. The estimator
    receives measured angular image flow, gyro and tripod-derived distance,
    NOT the MuJoCo velocity or true floor normal used by the forward model.
    Invalid sources produce zero velocity AND false validity; a policy must
    not mistake this for a confident zero-speed measurement.
    """
    flow=xp.asarray(flow_rad_s)
    gyro=xp.asarray(imu_gyro_body_rad_s)
    depth=xp.asarray(tof_estimated_optical_depth_m)
    valid=(xp.isfinite(depth)&(depth>=TOF_MIN_RANGE_M)
           &(depth<=TOF_MAX_RANGE_M)
           &xp.all(xp.isfinite(flow),axis=-1)
           &xp.all(xp.isfinite(gyro),axis=-1))
    if flow_valid is not None:
        valid=valid&xp.asarray(flow_valid)
    if tof_valid is not None:
        valid=valid&xp.asarray(tof_valid)
    scale=xp.maximum(depth,TOF_MIN_RANGE_M)
    vxy=xp.stack((-scale*(flow[:,0]-gyro[:,1]),
                  -scale*(flow[:,1]+gyro[:,0])),axis=-1)
    return dict(velocity_body_xy_mps=xp.where(valid[:,None],vxy,0.),
                velocity_valid=valid,
                velocity_confidence=xp.where(valid,1.,0.))
