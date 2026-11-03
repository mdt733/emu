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
VISIT_HISTORY=64
PROFILE_HALF=128
PROFILE_SIZE=PROFILE_HALF*2
BODY_NAMES=tuple(f"carpet_tile_body_{i}" for i in range(TILE_COUNT))
GEOM_NAMES=tuple(f"carpet_tile_geom_{i}" for i in range(TILE_COUNT))
HFIELD_NAMES=tuple(f"carpet_tile_hfield_{i}" for i in range(TILE_COUNT))
# Use 9 *solid-box* collision tops and their genuine vertical side faces.
# Each plane is determined by the same four vertex samples that define
# the two triangles. The original hfields become noncontact metadata so
# sensors can sample exactly the same seeded global heightfield potential.
SOLID_BODY_NAMES=tuple(f"carpet_solid_body_{i}" for i in range(TILE_COUNT))
SOLID_GEOM_NAMES=tuple(f"carpet_solid_geom_{i}" for i in range(TILE_COUNT))
SOLID_HALF_Z_M=.270
SOLID_HALF_XY_M=.258
BURIED_Z=-30.

def add_carpet_geoms(xml: str, *, prisms: bool = False) -> str:
    root=ET.fromstring(xml)
    world=root.find("worldbody")
    if world is None:raise ValueError("Carpet requires worldbody")
    asset=root.find("asset")
    if asset is None:
        asset=ET.Element("asset")
        root.insert(0,asset)
    # v5043 opt-in: collision surfaces will be 18 dynamically sheared
    # convex triangular prisms. Legacy nine-box training remains untouched.
    # The hfields still provide nine independent *analytic* 2x2 samples.
    # The v5040 slab/Academy training geometry is not used in a carpet
    # episode. Retain its stable named IDs for legacy diagnostics, but
    # DISABLE collision pairs rather than wasting GPU broadphase work on
    # five huge invisible boxes. The carpet's nine real tiles are the only
    # ground surfaces in this opt-in training mode.
    for name in ("floor","episode_terrain_surface",
                 "academy_surface_0","academy_surface_1",
                 "academy_surface_2"):
        geom=root.find(f".//geom[@name='{name}']")
        if geom is not None:
            geom.set("contype","0")
            geom.set("conaffinity","0")
    # The historical two-slab model gives the robot bits 1 and 2.
    # Carpet tiles are bit 4, so each active contact-capable robot geom
    # must accept bit 4 before MJX compiles collision filtering.
    for name in ("torso_box","r_foot_ball","l_foot_ball"):
        geom=root.find(f".//geom[@name='{name}']")
        if geom is not None:
            affinity=int(geom.get("conaffinity","1"))
            geom.set("conaffinity",str(affinity|4))
    for i in range(TILE_COUNT):
        ET.SubElement(asset,"hfield",dict(
            name=HFIELD_NAMES[i],nrow="2",ncol="2",
            size=f"{TILE_HALF_M} {TILE_HALF_M} {TILE_SIZE_Z_M} {TILE_BASE_M}",
            elevation="0.5 0.5 0.5 0.5"))
        body=ET.SubElement(world,"body",dict(
            name=BODY_NAMES[i],mocap="true",pos=f"0 0 {-4-i}"))
        ET.SubElement(body,"geom",dict(
            name=GEOM_NAMES[i],type="hfield",hfield=HFIELD_NAMES[i],
            contype="0",conaffinity="0",friction="1.5 0.005 0.0001",
            rgba="0.39 0.65 0.39 1"))
    # Every solid is a fixed 50cm planar square with a *real* vertical
    # thickness / side face. Rotation+translation update per JAX world,
    # but no geom size or model topology is ever recompiled.
    for i in range(TILE_COUNT):
        body=ET.SubElement(world,"body",dict(
            name=SOLID_BODY_NAMES[i],mocap="true",pos=f"0 0 {-20-i}"))
        ET.SubElement(body,"geom",dict(
            name=SOLID_GEOM_NAMES[i],type="box",
            size=f"{SOLID_HALF_XY_M} {SOLID_HALF_XY_M} {SOLID_HALF_Z_M}",
            contype="0" if prisms else "4",conaffinity="0",friction="1.5 0.005 0.0001",
            rgba="0.39 0.65 0.39 1"))
    if prisms:
        from .carpet_prism_mjx import prism_mesh_xml
        prism_mesh_xml(root,active=True)
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
                  step_down_m, academy, slope_exposure, step_exposure, seeds):
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
    # Exposure is the fraction of new EPISODE births given real terrain,
    # not a multiplier on the terrain height/grade. Scaling both exposure
    # and severity made the 140M v5041 run invisible to its own teacher:
    # 0.7 deg * .15 was below the slope evidence bands; 3mm * .15
    # was below the 1.5mm lip classifier. Preserve deterministic JAX
    # generation with two independent per-world salted birth draws.
    slope_on=(hash_float(xp,0,0,seeds,151) <
              xp.clip(slope_exposure,0.,1.))
    step_on=(hash_float(xp,0,0,seeds,152) <
             xp.clip(step_exposure,0.,1.))
    # The dedicated Academy is its OWN allocated STEP birth cohort.
    # It must receive the earned lip even if ordinary step exposure
    # is low; otherwise 50% Academy occupancy buys zero usable lessons.
    step_on=step_on | academy
    slope=xp.where(slope_on,xp.maximum(0.,slope_deg)*slope_factor,0.)
    up=xp.where(step_on,xp.maximum(0.,step_up_m)*step_factor,0.)
    down=xp.where(step_on,xp.maximum(0.,step_down_m)*step_factor,0.)
    return slope,up,down

def make_profiles(xp,seeds,up,down):
    """Exact per-world directional step limits on mostly independent strips.

    The +/-X direction independently draws UP/DOWN under an early-to-late
    0.95x→1.5x physical challenge schedule, with 90% of the directional
    frontier as the base amplitude. The +/-Y component uses independent signed up/down amplitudes near
    the frontier; per-tile jitter uses 5%. The final stretch may exceed
    the mastered frontier by design (but never grows without bounds).
    All global tiles remain consistent on revisit within an episode.
    """
    n=seeds.shape[0]
    idx=xp.arange(-PROFILE_HALF,PROFILE_HALF,dtype=xp.int32)[None,:]
    seed=seeds[:,None]
    hx=hash_float(xp,idx,0,seed,101)
    ax=hash_float(xp,idx,0,seed,102)
    # Seed lips must be physically detectable: a 3mm Academy frontier
    # scaled to 50% produced sub-1.5mm encounters, which the real-contact
    # teacher correctly rejects. Start at 95% and reach 150% by tile 14.
    # Global coordinates keep revisits immutable throughout the episode.
    challenge=.95+.55*xp.minimum(1.,xp.abs(idx.astype(xp.float32))/14.)
    signed_x=xp.where(hx<.5,.90*up[:,None],
                       -.90*down[:,None])*(.75+.25*ax)*challenge
    hy=hash_float(xp,0,idx,seed,103)
    ay=hash_float(xp,0,idx,seed,104)
    # v5043 experimental carpet: a newly generated row in Y is also a
    # real, signed step encounter; it is no longer just an 18% camber.
    # Each X/Y transition gets its OWN deterministic seed-derived sign
    # and height around the independently earned UP/DOWN frontiers.
    ychallenge=.95+.25*xp.minimum(1.,xp.abs(idx.astype(xp.float32))/14.)
    signed_y=xp.where(hy<.5,.85*up[:,None],-.85*down[:,None])*\
             (.8+.2*ay)*ychallenge
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

def tile_geometry(xp,ti,tj,seeds,px,py,up,down,slope_deg,
                  transition_limit_deg=3.0):
    """All four world vertex heights from one continuous grade function.

    Neighbours have identical smooth vertex height at the shared edge.
    Only the per-tile base offset creates an intentional vertical lip.
    """
    mean=tile_level(xp,ti,tj,seeds,px,py,up,down)
    x=ti.astype(xp.float32)*TILE_M
    y=tj.astype(xp.float32)*TILE_M
    phase=hash_float(xp,0,0,seeds,140)*2.*xp.pi
    p1=hash_float(xp,0,0,seeds,141)*2.*xp.pi
    p2=hash_float(xp,0,0,seeds,142)*2.*xp.pi
    # v5043: each global vertex has seeded 2D noise; adjacent squares
    # share it exactly, while each square acquires a distinct X/Y grade.
    # The random magnitude is limited by BOTH the absolute grade frontier
    # and the maximum grade-transition frontier. No per-tile independently
    # fitted plane can introduce an unearned seam step.
    # The source of truth is exactly these four heights in MJX collision.
    grade=xp.tan(xp.deg2rad(xp.maximum(0.,slope_deg)))
    tlimit=xp.tan(xp.deg2rad(
        xp.maximum(0.,xp.minimum(transition_limit_deg,slope_deg))))
    frequency=1.15
    variable=xp.minimum(.25*grade,
        tlimit/(frequency*TILE_M*1.6))
    base=xp.maximum(0.,.65*grade-variable)
    # Seeded random vertex elevation, continuous across the entire
    # unbounded grid. The conservative amplitude budget keeps the
    # resulting per-triangle grade and neighbouring normal jumps modest.
    random_amp=xp.minimum(.28*grade*TILE_M,
                          tlimit*TILE_M/12.)
    def z_at(dx,dy):
        gx=x+dx
        gy=y+dy
        wave=(variable/frequency)*(
            xp.sin(frequency*gx+p1)*xp.cos(.85*frequency*gy+p2))
        # Weighted two-dimensional global vertex noise (1/16 kernel).
        # No dependence on which tile requested the common corner.
        vi=ti+xp.asarray(xp.rint(dx/TILE_M),dtype=xp.int32)
        vj=tj+xp.asarray(xp.rint(dy/TILE_M),dtype=xp.int32)
        corner=4.*hash_float(xp,vi,vj,seeds,201)
        for oi,oj,w in ((-1,0,2.),(1,0,2.),(0,-1,2.),(0,1,2.),
                        (-1,-1,1.),(-1,1,1.),(1,-1,1.),(1,1,1.)):
            corner=corner+w*hash_float(xp,vi+oi,vj+oj,seeds,201)
        corner=corner/16.-.5
        return (mean+base*(gx*xp.cos(phase)+gy*xp.sin(phase))+
                wave+random_amp*corner)
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

def generate_window(xp,cx,cy,seeds,px,py,up,down,slope,
                    transition_limit_deg=3.0):
    xi,yi=slot_indices(xp,cx,cy)
    centers=[];heights=[]
    for i in range(TILE_COUNT):
        center,verts=tile_geometry(
            xp,xi[:,i],yi[:,i],seeds,px,py,up,down,slope,
            transition_limit_deg=transition_limit_deg)
        centers.append(center)
        heights.append(verts)
    return (xp.stack(centers,axis=1),
            xp.concatenate(heights,axis=1),
            xp.stack((xi,yi),axis=2))

def solid_box_poses(xp,centres,vertex_samples):
    """9 collision-solid poses from the SAME 18 procedural triangles.

    A box's top is one tilted plane through the mean tile height, with
    x/y slopes from the four original triangle vertices. Its true top
    matches the triangle surface within the bounded small curvature
    differences, and its *actual* vertical sides have horizontal normals.

    The bottom is far below the walking surface; no forbidden collision
    geom resizing. The original (disabled) hfields stay at their known
    centres for ToF and teacher-side reconstruction.
    """
    v=vertex_samples.reshape((centres.shape[0],TILE_COUNT,4))
    a,b,c,d=(v[:,:,0],v[:,:,1],v[:,:,2],v[:,:,3])
    mean=(a+b+c+d)*(.25*TILE_SIZE_Z_M)+centres[:,:,2]
    gx=.5*(b+d-a-c)*TILE_SIZE_Z_M/TILE_M
    gy=.5*(c+d-a-b)*TILE_SIZE_Z_M/TILE_M
    normal=xp.stack((-gx,-gy,xp.ones_like(gx)),axis=2)
    normal=normal/xp.linalg.norm(normal,axis=2,keepdims=True)
    # Shortest quaternion rotating +Z to the box's top-face normal.
    w=xp.sqrt((1.+normal[:,:,2])*.5)
    quat=xp.stack((w,-normal[:,:,1]/(2.*w),
                   normal[:,:,0]/(2.*w),xp.zeros_like(w)),axis=2)
    top=xp.stack((centres[:,:,0],centres[:,:,1],mean),axis=2)
    box_center=top-SOLID_HALF_Z_M*normal
    return box_center,quat

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

        # A two-triangle MuJoCo hfield is a solid height volume with
        # vertical *outer side walls*. Omitting those sides would let the
        # 4x4 ToF miss precisely the uphill riser the feet collide with.
        # This matches the SAME samples and base thickness as MJX, not a
        # separate privileged ray obstacle. Two x-faces and two y-faces.
        for axis,other,edges in ((0,1,((0,2),(1,3))),
                                 (1,0,((0,1),(2,3)))):
            direction=rays[:,:,axis]
            safe_dir=xp.where(xp.abs(direction)>1e-7,direction,1.)
            other_dir=rays[:,:,other]
            for side,(lo,hi) in enumerate(edges):
                edge_pos=pos[:,axis]+(-TILE_HALF_M if side==0 else TILE_HALF_M)
                tt=(edge_pos[:,None]-origin[:,None,axis])/safe_dir
                other_at=origin[:,None,other]+tt*other_dir
                fraction=(other_at-pos[:,None,other]+TILE_HALF_M)/TILE_M
                z_at=origin[:,None,2]+tt*rays[:,:,2]
                z_top=pos[:,None,2]+TILE_SIZE_Z_M*(
                      sample[:,lo,None]*(1.-fraction)
                      +sample[:,hi,None]*fraction)
                z_bottom=pos[:,None,2]-TILE_BASE_M
                hit=(xp.abs(direction)>1e-7)&(tt>1e-7)&(
                    fraction>=0.)&(fraction<=1.)&(
                    z_at<=z_top)&(z_at>=z_bottom)
                nearest=xp.minimum(nearest,xp.where(hit,tt,xp.inf))
    return nearest

def feet_clear_of_abandoned_solids(
        xp, old_center, new_center, foot_xy, foot_radii,
        box_centres_xy, box_rotations, box_half_sizes, clearance_m=.002):
    """Conservatively refuse recycling a real solid box beneath a foot.

    Each moving tile's *actual rotated box* is projected to a world XY
    bounding rectangle. A sphere-foot must be fully clear of that rectangle
    (including its 18mm radius and a small safety margin). Using foot CELL
    indices alone incorrectly declares a foot clear while its sphere still
    overlaps the soon-to-be-recycled 8mm box overhang or tilted side face.

    The AABB projection is deliberately conservative; it never says a foot
    is clear when it could still touch the abandoned OBB in 3D. Shapes:
    old/new [E,2], feet [E,F,2], radii [F], boxes [E,9,2],
    orientations [E,9,3,3], half sizes [9,3]. Returns [E] booleans.
    """
    old_x,old_y=slot_indices(xp,old_center[:,0],old_center[:,1])
    new_x,new_y=slot_indices(xp,new_center[:,0],new_center[:,1])
    abandoned=(old_x!=new_x)|(old_y!=new_y)
    projected_half=xp.sum(
        xp.abs(box_rotations[:,:,:2,:])*box_half_sizes[None,:,None,:],
        axis=-1)
    distance=xp.abs(foot_xy[:,:,None,:]-box_centres_xy[:,None,:,:])
    edge_margin=foot_radii[None,:,None,None]+clearance_m
    may_touch=xp.all(
        distance <= projected_half[:,None,:,:]+edge_margin,axis=-1)
    return ~xp.any(abandoned & xp.any(may_touch,axis=1),axis=1)


def unseen_tile(xp,coords,visited):
    """Reward-only episode history; returning to an old tile is NOT new
    terrain competence. 64 exact (x,y) cells cover >32m of recent walking.
    """
    return ~xp.any(xp.all(
        coords[:,None,:]==visited,axis=2),axis=1)


def note_confirmed_tile(xp,coords,visited,write,confirmed):
    """Store each NEW five-frame supported GLOBAL tile exactly once.

    Even when the caller supplies a physically confirmed *revisit*, never
    consume a ring slot: otherwise repeated rocking displaces genuinely
    new history and allows an older tile to be paid again prematurely.
    Teacher-only per-world state; invisible to the actor.
    """
    novel=confirmed & unseen_tile(xp,coords,visited)
    ix=xp.arange(coords.shape[0])
    slot=xp.mod(write,VISIT_HISTORY)
    before=visited[ix,slot,:]
    out=visited.at[ix,slot,:].set(
        xp.where(novel[:,None],coords,before))
    return out,write+novel.astype(xp.int32)

def signed_height_over_support(xp,points,surface_heights):
    """Never let off-carpet -inf become +inf height/reward.

    A lost patch means unsupported and immediately fall-eligible, rather
    than printing infinite clearance and poisoning shared PPO gradients.
    """
    measured=points[:,:,2]-surface_heights
    return xp.where(xp.isfinite(surface_heights),measured,
                    xp.full_like(measured,-1.))


def supported_tile_foot_pairs(xp,geom1,geom2,distance,frame,
                              tile_geom:int,foot_geoms,
                              foot_bottom_z,terrain_top_z,
                              *,vertical_min=.60,tolerance_m=.003,
                              top_tolerance_m=.012):
    """Reject a foot grazing a VERTICAL HFIELD RISER as foothold credit.

    A mere tile-vs-foot collision pair is insufficient: require a
    near-vertical contact normal and the same sphere's physical bottom
    to be near the sampled triangle top at its XY coordinate.
    Neither signal is visible to the policy.
    """
    normals=(frame[:,:,0,:] if frame.ndim==4 else frame[:,:,:3])
    upward=xp.abs(normals[:,:,2])>=vertical_min
    result=[]
    for i,foot in enumerate(foot_geoms):
        pair=(((geom1==tile_geom)&(geom2==foot)) |
              ((geom2==tile_geom)&(geom1==foot)))
        aligned=xp.any(pair & (distance<=tolerance_m)&upward,axis=1)
        top_match=(xp.isfinite(terrain_top_z[:,i]) &
                   (xp.abs(foot_bottom_z[:,i]-terrain_top_z[:,i])
                    <=top_tolerance_m))
        result.append(aligned&top_match)
    return xp.stack(result,axis=1)

def shares_physical_edge(xp,previous,next_cell):
    """Only orthogonal neighbours have one common physical ledge.

    Corner-cutting diagonals may still be real locomotion, but they
    cannot earn a made-up single-lip reward across an edge that does
    not exist between those two tiles.
    """
    return xp.sum(xp.abs(next_cell-previous),axis=1)==1
