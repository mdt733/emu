"""v5041 GPU trainer: nine reusable, independently mutable MJX heightfield tiles.

Every 50cm tile owns one 2x2 heightfield (two physical triangles); the
*same* nine MuJoCo collision geoms are repositioned per-world as EMU walks.

Caution: this is a new experimental training geometry contract, not a
retroactive replacement for source-pinned v5040 weights/checkpoints.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET

TILE_M=0.50
TILE_HALF_M=TILE_M/2
TILE_SIZE_Z_M=.50
TILE_BASE_M=.14
TILE_COUNT=9
PROFILE_HALF=128
PROFILE_SIZE=PROFILE_HALF*2
BODY_NAMES=tuple(f"carpet_tile_body_{i}" for i in range(TILE_COUNT))
GEOM_NAMES=tuple(f"carpet_tile_geom_{i}" for i in range(TILE_COUNT))
HFIELD_NAMES=tuple(f"carpet_tile_hfield_{i}" for i in range(TILE_COUNT))
BURIED_Z=-30.

def add_carpet_geoms(xml: str) -> str:
    root=ET.fromstring(xml)
    world=root.find("worldbody")
    if world is None:raise ValueError("Carpet requires worldbody")
    asset=root.find("asset")
    if asset is None:
        asset=ET.Element("asset")
        root.insert(0,asset)
    for i in range(TILE_COUNT):
        ET.SubElement(asset,"hfield",dict(
            name=HFIELD_NAMES[i],nrow="2",ncol="2",
            size=f"{TILE_HALF_M} {TILE_HALF_M} {TILE_SIZE_Z_M} {TILE_BASE_M}",
            elevation="0.5 0.5 0.5 0.5"))
        body=ET.SubElement(world,"body",dict(
            name=BODY_NAMES[i],mocap="true",pos=f"0 0 {-4-i}"))
        ET.SubElement(body,"geom",dict(
            name=GEOM_NAMES[i],type="hfield",hfield=HFIELD_NAMES[i],
            contype="4",conaffinity="0",friction="1.5 0.005 0.0001",
            rgba="0.39 0.65 0.39 1"))
    return ET.tostring(root,encoding="unicode")

def hash_float(xp,i,j,seed,salt):
    """JAX-numeric deterministic seeded, bounded; no Python RNG inside jit."""
    # Integer arithmetic intentionally overflows uint32, mapping every
    # global coordinate to the same tile on revisit.
    u=xp.asarray(i,dtype=xp.uint32)
    v=xp.asarray(j,dtype=xp.uint32)
    s=xp.asarray(seed,dtype=xp.uint32)
    x=(u*xp.uint32(374761393)+v*xp.uint32(668265263)
       +s*xp.uint32(1442695041)+xp.uint32(salt)*xp.uint32(1274126177))
    x=(x^(x>>xp.uint32(13)))*xp.uint32(1274126177)
    return ((x^(x>>xp.uint32(16))).astype(xp.float32)
            /xp.float32(4294967296.))

def role_envelope(xp, focus, specialist, athletic, slope_deg, step_up_m,
                  step_down_m, academy, slope_exposure, step_exposure):
    """Shared generator; one actor, six specialist-specific teacher limits.

    Foundation/athletic worlds remain essentially flat, then all specialist
    births inherit some of the demonstrated skills of the other branches.
    """
    from .capability_focus import STEP,SLOPE,RESILIENCE,TRANSITION,CONTROL,COMMAND
    f=focus.astype(xp.int32)
    slope_factor=xp.where(f==SLOPE,1.,xp.where(f==STEP,.5,
                 xp.where(f==RESILIENCE,.70,xp.where(f==TRANSITION,.65,
                 xp.where(f==CONTROL,.1,.2)))))
    step_factor=xp.where(f==STEP,1.,xp.where(f==SLOPE,.5,
                xp.where(f==RESILIENCE,.65,xp.where(f==TRANSITION,.55,
                xp.where(f==CONTROL,.1,.15)))))
    # Births with no specialist ID are mostly flat, but retain some
    # low-strength generalist exposure once either frontier is learned.
    slope_factor=xp.where(specialist,slope_factor,.08)
    step_factor=xp.where(specialist,step_factor,.06)
    slope_factor=xp.where(athletic>.5,slope_factor*.10,slope_factor)
    step_factor=xp.where(athletic>.5,step_factor*.10,step_factor)
    slope=xp.maximum(0.,slope_deg)*slope_factor*xp.clip(slope_exposure,0.,1.)
    up=xp.maximum(0.,step_up_m)*step_factor*xp.clip(step_exposure,0.,1.)
    down=xp.maximum(0.,step_down_m)*step_factor*xp.clip(step_exposure,0.,1.)
    return slope,up,down

def make_profiles(xp,seeds,up,down):
    """Exact per-world directional step limits on mostly independent strips.

    The +/-X direction independently draws UP/DOWN under an early-to-late
    0.5x→1.5x physical challenge schedule, with 90% of the directional
    frontier as the base amplitude. The +/-Y component uses 18% of the
    smaller frontier; per-tile jitter uses 5%. The final stretch may exceed
    the mastered frontier by design (but never grows without bounds).
    All global tiles remain consistent on revisit within an episode.
    """
    n=seeds.shape[0]
    idx=xp.arange(-PROFILE_HALF,PROFILE_HALF,dtype=xp.int32)[None,:]
    seed=seeds[:,None]
    hx=hash_float(xp,idx,0,seed,101)
    ax=hash_float(xp,idx,0,seed,102)
    # Intentionally start at 50% demonstrated ability near the spawn and
    # reach 150% in the farther tiles. Global coordinates make challenges
    # immutable when revisited; advancing the episode age never edits a
    # platform while the bird is standing on it.
    challenge=.5+xp.minimum(1.,xp.abs(idx.astype(xp.float32))/14.)
    signed_x=xp.where(hx<.5,.90*up[:,None],
                       -.90*down[:,None])*(.75+.25*ax)*challenge
    hy=hash_float(xp,0,idx,seed,103)
    ay=hash_float(xp,0,idx,seed,104)
    minor=xp.minimum(up,down)[:,None]
    signed_y=xp.where(hy<.5,.18*minor,-.18*minor)*(.3+.7*ay)
    xc=xp.cumsum(signed_x,axis=1)
    yc=xp.cumsum(signed_y,axis=1)
    xc=xc-xc[:,PROFILE_HALF:PROFILE_HALF+1]
    yc=yc-yc[:,PROFILE_HALF:PROFILE_HALF+1]
    return xc,yc

def tile_level(xp,ti,tj,seeds,xprofile,yprofile,up,down):
    ii=xp.clip(ti+PROFILE_HALF,0,PROFILE_SIZE-1)
    jj=xp.clip(tj+PROFILE_HALF,0,PROFILE_SIZE-1)
    xbase=xp.take_along_axis(xprofile,ii[...,None],axis=1)[...,0]
    ybase=xp.take_along_axis(yprofile,jj[...,None],axis=1)[...,0]
    noise=(hash_float(xp,ti,tj,seeds,107)-.5)*(
        xp.minimum(up,down)*.10)
    return xbase+ybase+noise

def tile_geometry(xp,ti,tj,seeds,px,py,up,down,slope_deg):
    """All four world vertex heights from one continuous grade function.

    Neighbours have identical smooth vertex height at the shared edge.
    Only the per-tile base offset creates an intentional vertical lip.
    """
    mean=tile_level(xp,ti,tj,seeds,px,py,up,down)
    x=ti.astype(xp.float32)*TILE_M
    y=tj.astype(xp.float32)*TILE_M
    phase=hash_float(xp,0,0,seeds,140)*2.*xp.pi
    # The SLOPE specialist's actual local gradient must reach its
    # near-frontier band (>=78%); using ~half the requested grade here
    # would make meaningful slope promotion mathematically impossible.
    sx=xp.tan(xp.deg2rad(slope_deg))*.82
    sy=xp.tan(xp.deg2rad(slope_deg))*.82
    def z_at(dx,dy):
        gx=x+dx
        gy=y+dy
        wave=.004*xp.clip(slope_deg/10.,0.,1.)*(
              xp.sin(gx*1.2+phase)*xp.cos(gy*.8-phase))
        return mean+sx*gx*xp.cos(phase)+sy*gy*xp.sin(phase)+wave
    heights=xp.stack((z_at(0.,0.),z_at(TILE_M,0.),
                      z_at(0.,TILE_M),z_at(TILE_M,TILE_M)),axis=1)
    centre=xp.mean(heights,axis=1)
    values=xp.clip(.5+(heights-centre[:,None])/TILE_SIZE_Z_M,.002,.998)
    position=xp.stack((x+TILE_HALF_M,y+TILE_HALF_M,
                       centre-TILE_SIZE_Z_M*.5),axis=1)
    return position,values

def slot_indices(xp,cx,cy):
    """Each slot's GLOBAL coordinates for a 3x3 window around cell (cx,cy)."""
    xs=[];ys=[]
    for i in range(TILE_COUNT):
        sx=i%3;sy=i//3
        xs.append(cx+xp.mod(sx-xp.mod(cx,3)+1,3)-1)
        ys.append(cy+xp.mod(sy-xp.mod(cy,3)+1,3)-1)
    return xp.stack(xs,axis=1),xp.stack(ys,axis=1)

def generate_window(xp,cx,cy,seeds,px,py,up,down,slope):
    xi,yi=slot_indices(xp,cx,cy)
    centers=[];heights=[]
    for i in range(TILE_COUNT):
        center,verts=tile_geometry(
            xp,xi[:,i],yi[:,i],seeds,px,py,up,down,slope)
        centers.append(center)
        heights.append(verts)
    return (xp.stack(centers,axis=1),
            xp.concatenate(heights,axis=1),
            xp.stack((xi,yi),axis=2))

def sample_surface(xp,data,mocap_ids,hfield_data,points):
    """Teacher and simulated ray physics lookup on the ACTUAL 9 tile vertices.

    No global terrain coordinates are passed into the actor. This computes
    upper support elevations from the same model hfield_data sent to MJX.
    Missing tile footprint returns -inf, not a fictitious infinite plane.
    """
    height=xp.full(points.shape[:2],-xp.inf,dtype=xp.float32)
    for i,mocap in enumerate(mocap_ids):
        p=data.mocap_pos[:,mocap,:]
        u=(points[:,:,0]-p[:,None,0]+TILE_HALF_M)/TILE_M
        v=(points[:,:,1]-p[:,None,1]+TILE_HALF_M)/TILE_M
        a,b,c,d=[hfield_data[:,i*4+k,None] for k in range(4)]
        # Two planar triangles, diagonal (right-bottom to left-top).
        near=a*(1.-u-v)+b*u+c*v
        far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
        z=p[:,None,2]+TILE_SIZE_Z_M*xp.where(u+v<=1.,near,far)
        inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
        height=xp.maximum(height,xp.where(inside,z,-xp.inf))
    return height

def ray_surface(xp,data,mocap_ids,hfield_data,origin,rays):
    """Slant range against both real triangles in every 50cm tile.

    Iterative ray-plane intersections use the same planar interpolation as
    sample_surface. Only finite ray distances inside a tile are admitted.
    The old two-slab ray never leaks its buried ground into 360 observations.
    """
    nearest=xp.full(rays.shape[:2],xp.inf,dtype=xp.float32)
    vertical=rays[:,:,2]
    safe=xp.where(xp.abs(vertical)>1e-6,vertical,-1.)
    for i,mocap in enumerate(mocap_ids):
        pos=data.mocap_pos[:,mocap,:]
        sample=hfield_data[:,i*4:i*4+4]
        middle=pos[:,2]+TILE_SIZE_Z_M*xp.mean(sample,axis=1)
        t=(middle[:,None]-origin[:,None,2])/safe
        for _ in range(3):
            u=(origin[:,None,0]+t*rays[:,:,0]-pos[:,None,0]+TILE_HALF_M)/TILE_M
            v=(origin[:,None,1]+t*rays[:,:,1]-pos[:,None,1]+TILE_HALF_M)/TILE_M
            a,b,c,d=[sample[:,k,None] for k in range(4)]
            near=a*(1.-u-v)+b*u+c*v
            far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
            height=pos[:,None,2]+TILE_SIZE_Z_M*xp.where(u+v<=1.,near,far)
            t=(height-origin[:,None,2])/safe
        inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
        good=inside&(t>0.)&(rays[:,:,2]<-.05)
        nearest=xp.minimum(nearest,xp.where(good,t,xp.inf))
    return nearest
