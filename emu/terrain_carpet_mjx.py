"""Current true-prism MJX terrain generator.

Four 75cm tiles carry four exact height samples each, as non-contact
MuJoCo hfields. Each tile's two real convex triangular-prism meshes
provide the ONLY physical terrain collisions. All six specialists use
the same generator, collider and ToF source. Legacy nine-box geometry
is available from Git history, not from this training engine.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET

TILE_M=0.75
TILE_HALF_M=TILE_M/2
TILE_SIZE_Z_M=.50
TILE_BASE_M=.14
TILE_WINDOW_SIDE=2
TILE_COUNT=TILE_WINDOW_SIDE**2
VISIT_HISTORY=64
PROFILE_HALF=128
PROFILE_SIZE=PROFILE_HALF*2
BODY_NAMES=tuple(f"carpet_tile_body_{i}" for i in range(TILE_COUNT))
GEOM_NAMES=tuple(f"carpet_tile_geom_{i}" for i in range(TILE_COUNT))
HFIELD_NAMES=tuple(f"carpet_tile_hfield_{i}" for i in range(TILE_COUNT))
def add_carpet_geoms(xml: str, *, prisms: bool = True) -> str:
    """Add only exact physical prisms and noncontact vertex sample hfields."""
    if not prisms:
        raise ValueError('This EMU version supports real prism contacts only')
    root=ET.fromstring(xml)
    world=root.find('worldbody')
    if world is None:
        raise ValueError('Prism carpet requires worldbody')
    asset=root.find('asset')
    if asset is None:
        asset=ET.Element('asset')
        root.insert(0,asset)
    floor=world.find("geom[@name='floor']")
    if floor is None:
        raise ValueError('Current prism world requires original robot floor')
    # A plane would create phantom support past the rolling 2x2
    # footprint. Keep the named floor only for structural ID consistency.
    floor.set('contype','0')
    floor.set('conaffinity','0')
    for name in ('torso_box','r_foot_ball','l_foot_ball'):
        geom=root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f'Missing physical robot contact geom {name}')
        affinity=int(geom.get('conaffinity','1'))
        geom.set('conaffinity',str(affinity|4))
    for i in range(TILE_COUNT):
        ET.SubElement(asset,'hfield',dict(
            name=HFIELD_NAMES[i],nrow='2',ncol='2',
            size=f'{TILE_HALF_M} {TILE_HALF_M} {TILE_SIZE_Z_M} {TILE_BASE_M}',
            elevation='0.5 0.5 0.5 0.5'))
        body=ET.SubElement(world,'body',dict(
            name=BODY_NAMES[i],mocap='true',pos=f'0 0 {-4-i}'))
        ET.SubElement(body,'geom',dict(
            name=GEOM_NAMES[i],type='hfield',hfield=HFIELD_NAMES[i],
            contype='0',conaffinity='0',friction='1.5 0.005 0.0001'))
    from .carpet_prism_mjx import prism_mesh_xml
    prism_mesh_xml(root,active=True)
    return ET.tostring(root,encoding='unicode')


def validate_prism_only_geoms(mujoco, model) -> None:
    """Reject retired box/Academy terrain by inspecting compiled MJ geoms.

    The old flags were always empty tuples, so they could never detect an
    accidentally reintroduced collider. Compare actual geom names instead:
    nine noncolliding source samplers and eighteen physical prisms, nothing
    else in the terrain/Academy namespace.
    """
    from .carpet_prism_mjx import PRISM_GEOM_NAMES
    expected = set(GEOM_NAMES) | set(PRISM_GEOM_NAMES)
    actual = {
        name
        for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
        and name.startswith(('carpet_', 'academy_'))
    }
    if actual != expected:
        raise RuntimeError(
            f'Prism scene terrain mismatch: missing={sorted(expected-actual)} '
            f'extra={sorted(actual-expected)}')



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
    # Keep sampled triangles near the earned slope frontier: multiplying
    # down the base slope here would recreate the near-band starvation
    # uncovered in the 140M v5041 investigation. A conservative <=15%
    # varying gradient gives random 2D orientation without systematic
    # under-exposure; transition_limit_deg bounds its spatial changes.
    variable=xp.minimum(.15*grade,
        tlimit/(frequency*TILE_M*1.6))
    base=xp.maximum(0.,.92*grade-.45*variable)
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

def update_tile_mocaps(xp, positions, mocap_ids, centers, changed=None):
    """Move the nine source-height tiles with ONE indexed JAX scatter.

    The hfield tiles are noncontact samples used by the ray synthesizer
    and the convex-prism mesh builder. Both birth (all slots) and recycle
    (changed slots only) retain exactly the old per-world positions.
    """
    if len(mocap_ids)!=TILE_COUNT or len(set(mocap_ids))!=TILE_COUNT:
        raise ValueError('Expected exactly nine distinct source tile mocaps')
    ids=xp.asarray(mocap_ids,dtype=xp.int32)
    if changed is None:
        return positions.at[:,ids,:].set(centers)
    previous=positions[:,ids,:]
    return positions.at[:,ids,:].set(
        xp.where(changed[:,:,None],centers,previous))


def midpoint_anchor(xp, root_xy):
    """Lower-left GLOBAL tile index of the 2x2 window for each robot.

    At the centre of its current 75cm square, load the next square
    in that direction (independently for X/Y). The current square
    remains present in all four quadrants. From the torso centre,
    the nearest unloaded boundary is at least TILE_M/2 = 37.5cm
    away, exceeding the normal ~30cm horizontal ToF footprint.
    This is a logical switch; the abandoned solid still needs a
    real foot-clearance check before moving.
    """
    u=root_xy/TILE_M
    cell=xp.floor(u).astype(xp.int32)
    half=u-cell.astype(xp.float32)
    return cell+xp.where(half>=.5,0,-1).astype(xp.int32)


def slot_indices(xp,cx,cy):
    """Stable modulo-two prism slots, keyed to 2x2 lower-left anchor.

    The same global tile retains the same slot while crossing any
    number of window boundaries; only a retired row/column moves.
    """
    xs=[];ys=[]
    for i in range(TILE_COUNT):
        sx=i%TILE_WINDOW_SIDE;sy=i//TILE_WINDOW_SIDE
        xs.append(cx+xp.mod(sx-xp.mod(cx,TILE_WINDOW_SIDE),
                            TILE_WINDOW_SIDE))
        ys.append(cy+xp.mod(sy-xp.mod(cy,TILE_WINDOW_SIDE),
                            TILE_WINDOW_SIDE))
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

def sample_surface(xp,data,mocap_ids,hfield_data,points):
    """All nine exact two-triangle top surfaces in ONE array calculation.

    Same physical world, sensor/teacher contract and missing-footprint
    behaviour as the former nine sequential tile checks. This reduces
    compiled graph repetition; its effect on actual GPU FPS must be measured.
    """
    if len(mocap_ids)!=TILE_COUNT:
        raise ValueError('Expected nine source tile mocap IDs')
    positions=data.mocap_pos[:,xp.asarray(mocap_ids,dtype=xp.int32),:]
    source=hfield_data.reshape((hfield_data.shape[0],TILE_COUNT,4))
    # Shapes: [world,tile,point] and [world,tile,1].
    u=(points[:,None,:,0]-positions[:,:,None,0]+TILE_HALF_M)/TILE_M
    v=(points[:,None,:,1]-positions[:,:,None,1]+TILE_HALF_M)/TILE_M
    a,b,c,d=[source[:,:,k,None] for k in range(4)]
    near=a*(1.-u-v)+b*u+c*v
    far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
    z=positions[:,:,None,2]+TILE_SIZE_Z_M*xp.where(u+v<=1.,near,far)
    inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
    return xp.max(xp.where(inside,z,-xp.inf),axis=1)

def ray_surface(xp,data,mocap_ids,hfield_data,origin,rays,*,
                prism_bottom_depth_m=None):
    """All nine physical source tiles in one batched slant-ray program.

    Array axes are [world,tile,ray]. The exact three surface iterations,
    two triangular tops and four vertical sidewalls are unchanged.
    A frozen sequential oracle tests numerical parity before GPU trials.
    """
    if len(mocap_ids)!=TILE_COUNT:
        raise ValueError('Expected nine physical source tiles')
    pos=data.mocap_pos[:,xp.asarray(mocap_ids,dtype=xp.int32),:]
    samples=hfield_data.reshape((hfield_data.shape[0],TILE_COUNT,4))
    vertical=rays[:,None,:,2]
    safe=xp.where(xp.abs(vertical)>1e-6,vertical,-1.)
    middle=pos[:,:,2]+TILE_SIZE_Z_M*xp.mean(samples,axis=-1)
    t=(middle[:,:,None]-origin[:,None,None,2])/safe
    a,b,c,d=[samples[:,:,k,None] for k in range(4)]
    for _ in range(3):
        u=(origin[:,None,None,0]+t*rays[:,None,:,0]-pos[:,:,None,0]+TILE_HALF_M)/TILE_M
        v=(origin[:,None,None,1]+t*rays[:,None,:,1]-pos[:,:,None,1]+TILE_HALF_M)/TILE_M
        near=a*(1.-u-v)+b*u+c*v
        far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
        height=pos[:,:,None,2]+TILE_SIZE_Z_M*xp.where(u+v<=1.,near,far)
        t=(height-origin[:,None,None,2])/safe
    inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
    good=inside&(t>0.)&(vertical<-.05)
    nearest=xp.where(good,t,xp.inf)
    # The actual prisms have outer risers: no upper-step blindness.
    for axis,other,edges in ((0,1,((0,2),(1,3))),
                             (1,0,((0,1),(2,3)))):
        direction=rays[:,None,:,axis]
        safe_dir=xp.where(xp.abs(direction)>1e-7,direction,1.)
        other_dir=rays[:,None,:,other]
        for side,(lo,hi) in enumerate(edges):
            edge_pos=pos[:,:,axis]+(-TILE_HALF_M if side==0 else TILE_HALF_M)
            tt=(edge_pos[:,:,None]-origin[:,None,None,axis])/safe_dir
            other_at=origin[:,None,None,other]+tt*other_dir
            fraction=(other_at-pos[:,:,None,other]+TILE_HALF_M)/TILE_M
            z_at=origin[:,None,None,2]+tt*vertical
            z_top=pos[:,:,None,2]+TILE_SIZE_Z_M*(
                  samples[:,:,lo,None]*(1.-fraction)
                  +samples[:,:,hi,None]*fraction)
            z_bottom=(z_top-prism_bottom_depth_m if
                      prism_bottom_depth_m is not None else
                      pos[:,:,None,2]-TILE_BASE_M)
            hit=(xp.abs(direction)>1e-7)&(tt>1e-7)&(
                 fraction>=0.)&(fraction<=1.)&(
                 z_at<=z_top)&(z_at>=z_bottom)
            nearest=xp.minimum(nearest,xp.where(hit,tt,xp.inf))
    return xp.min(nearest,axis=1)


def feet_clear_of_abandoned_prisms(
        xp, old_center, new_center, foot_xy, foot_radii,
        clearance_m=.004):
    """Recycle only when every sphere-foot clears abandoned REAL prisms.

    A prism's two triangles exactly cover a TILE_M square in XY; their
    shared and exterior walls are vertical, including the sheared bottom.
    The tile's projected XY AABB matches that square exactly, with
    no rotation/box overhang. Add foot radius and 4mm
    conservative safety distance. Ignore altitude deliberately:
    a full-XY-foot-clear test cannot teleport a possible support, even
    at a sharp stair corner. This guard is for PRISMS only.
    """
    old_x,old_y=slot_indices(xp,old_center[:,0],old_center[:,1])
    new_x,new_y=slot_indices(xp,new_center[:,0],new_center[:,1])
    abandoned=(old_x!=new_x)|(old_y!=new_y)
    center_xy=(xp.stack((old_x,old_y),axis=-1).astype(xp.float32)
               +.5)*TILE_M
    distance=xp.abs(foot_xy[:,:,None,:]-center_xy[:,None,:,:])
    margin=TILE_HALF_M+foot_radii[None,:,None,None]+clearance_m
    could_touch=xp.all(distance<=margin,axis=-1)
    return ~xp.any(abandoned & xp.any(could_touch,axis=1),axis=1)


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
