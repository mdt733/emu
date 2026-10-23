"""v5022 terrain + 4x4 front ToF. One test suite; no prototype harnesses.

Physical MJX checks are CPU tests and cannot establish policy success.
"""
import math
import numpy as np
import pytest

from emu.terrain import (
    SENSOR_MODEL, SENSOR_RESOLUTION, SENSOR_FOV_DEG, SENSOR_MOUNT_BODY_M,
    SENSOR_DOWN_TILT_DEG,
    course_surfaces, surface_box_geometry, camera_rays,
    ray_boxes, ray_spheres, scan_forward, scan_tripod,
)


def pose(x=0., worlds=1):
    return (np.tile([[x,0.,.2584]],(worlds,1)),
            np.tile(np.eye(3)[None],(worlds,1,1)))


def scene(step=0.):
    # Ground at z=0 before x=.16 and z=step thereafter.
    p = np.array([[-.42,0.,-.2],[.58,0.,(step-.4)/2]])
    sz = np.array([[.58,.4,.2],[.42,.4,(step+.4)/2]])
    rot = np.tile(np.eye(3)[None],(2,1,1))
    return p,rot,sz


def test_hill_is_continuous_and_steps_remain_distinct():
    surfaces, seams = course_surfaces(-.6,0.,[.3,.3,.3,.3],
                                      [0,12,0,-12],[0,0,0])
    assert all(not seam.discontinuous for seam in seams)
    for left,right in zip(surfaces,surfaces[1:]):
        assert left.z1_m == pytest.approx(right.z0_m)
    _, discontinuities = course_surfaces(-.5,0.,[.5,.5,.5],
                                        [0,0,0],[.05,-.07])
    assert [x.jump_m for x in discontinuities] == [.05,-.07]
    assert all(x.discontinuous for x in discontinuities)


def test_solid_rotated_box_top_is_the_requested_surface():
    segments, _ = course_surfaces(-.6,0.,[.4,.4,.4],[0,10,-10],[0,0])
    for surface in segments:
        p, half, q = surface_box_geometry(surface)
        a = surface.grade_rad
        top = np.array(p) + half[2]*np.array([-math.sin(a),0,math.cos(a)])
        tangent = np.array([math.cos(a),0,math.sin(a)])
        for sign,x in [(-1,surface.x0_m),(1,surface.x1_m)]:
            point = top+sign*half[0]*tangent
            assert point[0] == pytest.approx(x)
            assert point[2] == pytest.approx(surface.height_at(x))
        assert np.linalg.norm(q) == pytest.approx(1)


def test_sensor_is_4x4_65_diagonal_variant_and_one_optical_centre():
    p,R = pose(worlds=2)
    origins,rays = camera_rays(np,p,R)
    assert SENSOR_MODEL == "VL53L8CX"
    assert SENSOR_RESOLUTION == (4,4)
    assert SENSOR_FOV_DEG == (45.,45.)
    # Operator-selected candidate allows the optical centre at the torso root.
    assert SENSOR_MOUNT_BODY_M[0] == pytest.approx(0.)
    assert origins.shape == rays.shape == (2,16,5,3)
    np.testing.assert_allclose(origins,np.broadcast_to(
        origins[:,0:1,0:1,:],origins.shape))
    np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1.,atol=1e-7)
    # Four rows look at progressively nearer patches.
    assert rays[0,0,0,2] > rays[0,15,0,2]


@pytest.mark.parametrize("height", [.05,-.05])
def test_front_scan_distinguishes_real_step_geometry(height):
    p,R = pose()
    baseline,good0 = scan_forward(np,p,R,*scene(0.))
    scan,good = scan_forward(np,p,R,*scene(height))
    assert baseline.shape == good.shape == (1,16)
    both = good & good0
    assert both.any()
    assert np.any(np.abs(scan[both]-baseline[both]) > .01)


def test_chassis_pitch_changes_raw_measurements():
    p,R = pose()
    theta = np.deg2rad(12)
    tilt = R.copy()
    tilt[0] = np.array([[np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])
    a,_ = scan_forward(np,p,R,*scene(.05))
    b,_ = scan_forward(np,p,tilt,*scene(.05))
    assert not np.allclose(a,b)


def test_foot_in_front_is_visible_and_never_privileged_filtered():
    p,R = pose()
    before,valid = scan_forward(np,p,R,*scene(0))
    origins,rays = camera_rays(np,p,R)
    # Put a physical 18mm-radius ball directly on central camera zone.
    foot = (origins[0,5,0] + .12*rays[0,5,0]).reshape(1,1,3)
    after,good = scan_forward(np,p,R,*scene(0),
                              foot_centres=foot,foot_radii=.018)
    assert (good & valid).any()
    assert np.any(after[good & valid] < before[good & valid] - .02)


def test_missing_readings_are_invalid_not_zero_height():
    p,R = pose()
    out,valid = scan_forward(np,p,R,*scene(.05),
                             dropout=np.ones((1,16),dtype=bool))
    assert not valid.any() and not out.any()


@pytest.mark.parametrize("height", [.05,-.05])
def test_down_tripod_sees_split_platform_height(height):
    p,R = pose(x=.16)
    readings,valid = scan_tripod(np,p,R,*scene(height))
    assert valid.all()
    np.testing.assert_allclose(readings[0,:2],.2584-.025-height,atol=.003)
    assert readings[0,2] == pytest.approx(.2584-.025,abs=.003)


def test_jax_jit_and_world_batch_match_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    p,R = pose(worlds=2)
    centres,rot,size = scene(.05)
    np_v,np_ok = scan_forward(np,p,R,centres,rot,size)
    jax_scan = jax.jit(lambda t,m:scan_forward(
        jnp,t,m,jnp.asarray(centres),jnp.asarray(rot),jnp.asarray(size)))
    jv,jo = jax_scan(jnp.asarray(p),jnp.asarray(R))
    np.testing.assert_allclose(np.asarray(jv),np_v,atol=1e-6)
    np.testing.assert_array_equal(np.asarray(jo),np_ok)


def physical_xml(height):
    r_half=(height+.4)/2
    r_z=(height-.4)/2
    return f"""<mujoco>
    <option timestep=".005" gravity="0 0 -9.81"/>
    <asset><material name="solid" rgba=".5 .6 .4 1"/></asset>
    <worldbody>
      <geom name="approach" type="box" group="1" material="solid"
            pos="-.42 0 -.2" size=".58 .4 .2"/>
      <geom name="landing" type="box" group="1" material="solid"
            pos=".58 0 {r_z:.8f}" size=".42 .4 {r_half:.8f}"/>
      <body name="contact_probe" pos="0 0 .16">
        <freejoint/><geom type="sphere" size=".018" mass=".01"/>
      </body>
    </worldbody></mujoco>"""


@pytest.mark.parametrize("height",[.05,-.05])
def test_mjx_rays_and_real_collision_share_step_solids(height):
    mujoco=pytest.importorskip("mujoco")
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx
    cpu=mujoco.MjModel.from_xml_string(physical_xml(height))
    m=mjx.put_model(cpu)
    d=mjx.forward(m,mjx.make_data(m))
    ids=np.asarray([mujoco.mj_name2id(cpu,mujoco.mjtObj.mjOBJ_GEOM,n)
                    for n in ("approach","landing")])
    assert np.all(ids >= 0)
    p,R=pose()
    o,v=camera_rays(jnp,jnp.asarray(p),jnp.asarray(R))
    orig=o.reshape(1,80,3)[:,[1,21,51,71],:]
    vec=v.reshape(1,80,3)[:,[1,21,51,71],:]
    analytic=ray_boxes(jnp,orig,vec,d.geom_xpos[ids],
                       d.geom_xmat[ids],m.geom_size[ids])
    def trace_one(point,direction):
        return mjx.ray(m,d,point,direction,
                       geomgroup=(0,1,0,0,0,0),flg_static=True)
    hit,geom=jax.jit(jax.vmap(trace_one))(orig[0],vec[0])
    assert np.all(np.asarray(geom) >= 0)
    np.testing.assert_allclose(np.asarray(analytic)[0],np.asarray(hit),
                               atol=.003)
    # 2 different physical contact heights, in *batched* MJX data.
    first=mjx.make_data(m)
    batch=jax.tree_util.tree_map(lambda a:jnp.stack((a,a)),first)
    q=batch.qpos.at[0,:3].set(jnp.array([-.35,0.,.12]))
    q=q.at[1,:3].set(jnp.array([.35,0.,height+.12]))
    batch=batch.replace(qpos=q)
    fwd=jax.vmap(mjx.forward,in_axes=(None,0))
    step=jax.vmap(mjx.step,in_axes=(None,0))
    @jax.jit
    def settle(state):
        state=fwd(m,state)
        return jax.lax.fori_loop(0,140,lambda _,v:step(m,v),state)
    result=settle(batch)
    actual=np.asarray(result.qpos[:,2])
    np.testing.assert_allclose(actual,[.018,height+.018],atol=.022)
    assert actual[1]-actual[0] == pytest.approx(height,abs=.025)



def test_real_mjx_hillcrest_contact_and_depth_ray_parity():
    """Slope transitions are collidable support, not tilted gravity."""
    mujoco=pytest.importorskip("mujoco")
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx

    slopes, _ = course_surfaces(-.8,0.,[.4,.4,.4,.4],
                                 [0.,12.,0.,-12.],[0.,0.,0.])
    xml=['<mujoco><option timestep=".005" gravity="0 0 -9.81"/>',
         '<asset><material name="solid" rgba=".5 .5 .5 1"/></asset>',
         '<worldbody>']
    for i,segment in enumerate(slopes):
        p,size,quat = surface_box_geometry(segment)
        fmt=lambda v:' '.join(f'{x:.9f}' for x in v)
        xml.append(f'<geom name="terrain_{i}" type="box" group="1" '
                   f'material="solid" pos="{fmt(p)}" '
                   f'size="{fmt(size)}" quat="{fmt(quat)}"/>')
    xml.append('<body name="probe" pos="0 0 .16"><freejoint/>'
               '<geom type="sphere" size=".018" mass=".01"/></body>'
               '</worldbody></mujoco>')
    cpu=mujoco.MjModel.from_xml_string(''.join(xml))
    m=mjx.put_model(cpu)
    d=mjx.forward(m,mjx.make_data(m))
    ids=np.asarray([mujoco.mj_name2id(cpu,mujoco.mjtObj.mjOBJ_GEOM,
                                      f'terrain_{i}') for i in range(4)])
    assert np.all(ids>=0)
    p,R=pose(x=-.48)
    o,v=camera_rays(jnp,jnp.asarray(p),jnp.asarray(R))
    chosen=(10,35,62)
    origin=o.reshape(1,80,3)[:,chosen,:]
    rays=v.reshape(1,80,3)[:,chosen,:]
    expected=ray_boxes(jnp,origin,rays,d.geom_xpos[ids],
                       d.geom_xmat[ids],m.geom_size[ids])
    def trace(point,direction):
        return mjx.ray(m,d,point,direction,
                       geomgroup=(0,1,0,0,0,0),flg_static=True)
    hit,which=jax.jit(jax.vmap(trace))(origin[0],rays[0])
    assert np.all(np.asarray(which)>=0)
    np.testing.assert_allclose(np.asarray(expected)[0],np.asarray(hit),
                               atol=.003)


# Post-viewer acceptance gates live HERE instead of another v5022 harness.
# Optical placement is adjustable without changing the archived 232 actor.
@pytest.mark.parametrize("down_deg", [35., 50., 65.])
@pytest.mark.parametrize("forward_m", [.085, .120, .160])
def test_mount_candidates_keep_zone_order_and_physical_preview(down_deg, forward_m):
    p,R = pose()
    mount=(forward_m,0.,-.018)
    origin,rays = camera_rays(np,p,R,tilt_down_deg=down_deg,
                              mount_body_m=mount)
    assert origin.shape == rays.shape == (1,16,5,3)
    np.testing.assert_allclose(origin[0,0,0], [forward_m,0,.2404],atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1,atol=1e-7)
    assert rays[0,0,0,2] > rays[0,15,0,2]
    for subrays in (1,4,5):
        readings,valid=scan_forward(
            np,p,R,*scene(.05),mount_body_m=mount,
            tilt_down_deg=down_deg,subrays=subrays)
        assert readings.shape == valid.shape == (1,16)
        assert np.isfinite(readings).all()
        assert (readings[~valid]==0).all()
        assert np.all((readings[valid]>=.020)&(readings[valid]<=1.5))


def test_forward_mount_translation_tracks_chassis_rotation_not_world_x():
    p,R=pose()
    yaw=np.deg2rad(90)
    R[0] = [[np.cos(yaw),-np.sin(yaw),0],
            [np.sin(yaw), np.cos(yaw),0],
            [0,0,1]]
    near,_=camera_rays(np,p,R,mount_body_m=(.085,0,-.018))
    far,_=camera_rays(np,p,R,mount_body_m=(.160,0,-.018))
    np.testing.assert_allclose((far-near)[0,0,0],[0,.075,0],atol=1e-7)


def test_all_three_sampling_modes_keep_16_zone_outputs_and_invalid_semantics():
    p,R=pose()
    for subrays in (1,4,5):
        flat,valid=scan_forward(np,p,R,*scene(0.),subrays=subrays)
        assert flat.shape == valid.shape == (1,16)
        dropped,mask=scan_forward(
            np,p,R,*scene(.05),subrays=subrays,
            dropout=np.ones((1,16),dtype=bool))
        assert not mask.any()
        assert not dropped.any()
    with pytest.raises(ValueError,match="1, 4 or 5"):
        scan_forward(np,p,R,*scene(),subrays=8)


def test_jax_mount_and_sampling_modes_match_numpy_for_two_worlds():
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    p,R=pose(worlds=2)
    p[0,0] = -.1
    p[1,0] = 0.
    boxes,orient,size=scene(.05)
    for subrays in (1,4,5):
        kw=dict(tilt_down_deg=42.,mount_body_m=(.135,0.,-.018),
                subrays=subrays)
        want,mask=scan_forward(np,p,R,boxes,orient,size,**kw)
        work=jax.jit(lambda t:scan_forward(
            jnp,t,jnp.asarray(R),jnp.asarray(boxes),
            jnp.asarray(orient),jnp.asarray(size),**kw))
        got,valid=work(jnp.asarray(p))
        np.testing.assert_allclose(np.asarray(got),want,rtol=2e-5,atol=1e-5)
        np.testing.assert_array_equal(np.asarray(valid),mask)


@pytest.mark.skipif(
    __import__("os").environ.get("EMU_SENSOR_SWEEP")!="1",
    reason="opt-in post-viewer geometry sweep (EMU_SENSOR_SWEEP=1)")
def test_optional_post_viewer_mount_sweep(capsys):
    """Print evidence for the mount chosen in the viewer, not a success score.

    Override EMU_SENSOR_TILT_DEG and EMU_SENSOR_FORWARD_MM to check an exact
    user-selected mount; default sweeps 0/30/60mm x 65/70/75°.
    """
    import json
    import os
    angles=([float(os.environ["EMU_SENSOR_TILT_DEG"])]
            if os.environ.get("EMU_SENSOR_TILT_DEG") else [65.,70.,75.])
    forward=([float(os.environ["EMU_SENSOR_FORWARD_MM"])/1000.]
             if os.environ.get("EMU_SENSOR_FORWARD_MM")
             else [0.,.03,.06])
    all_results=[]
    for angle in angles:
        for xmount in forward:
            visible=[]
            for x in (-.35,-.25,-.15,-.05,0.):
                p,R=pose(x=x)
                kw=dict(tilt_down_deg=angle,
                        mount_body_m=(xmount,0.,-.018),subrays=4)
                flat,flat_ok=scan_forward(np,p,R,*scene(0.),**kw)
                up,up_ok=scan_forward(np,p,R,*scene(.05),**kw)
                down,down_ok=scan_forward(np,p,R,*scene(-.05),**kw)
                def changed(value,good):
                    return int(np.count_nonzero(
                        good & flat_ok & (np.abs(value-flat)>.012)))
                visible.append(dict(
                    torso_x_m=x,up_changed_zones=changed(up,up_ok),
                    down_changed_zones=changed(down,down_ok),
                    up_valid=int(up_ok.sum()),down_valid=int(down_ok.sum())))
            all_results.append(dict(
                tilt_down_deg=angle,mount_forward_mm=round(1000*xmount,2),
                geometry_preview=visible))
    assert len(all_results)==len(angles)*len(forward)
    print("\\nV5022_SENSOR_MOUNT_SWEEP "+json.dumps(all_results))
    # Intentional report-only: your physical design decision is not
    # determined by one invented numerical preference metric.


@pytest.mark.skipif(
    __import__("os").environ.get("EMU_SENSOR_BENCH")!="1",
    reason="opt-in GPU microbenchmark (EMU_SENSOR_BENCH=1)")
def test_optional_gpu_2048_world_ray_cost():
    """Compile vs steady-state 16/64/80 rays in the forward model ONLY.

    This is NOT an MJX/PPO FPS result; it helps choose the zone footprint
    before integrating rays into the actual batched training environment.
    """
    import time
    jax=pytest.importorskip("jax")
    import jax.numpy as jnp
    if jax.default_backend()!="gpu":
        pytest.skip("Requires a JAX GPU backend for comparable throughput")
    worlds=2048
    p=np.tile([[0.,0.,.2584]],(worlds,1)).astype(np.float32)
    p[:,0]=np.linspace(-.34,.10,worlds)
    R=np.tile(np.eye(3,dtype=np.float32)[None],(worlds,1,1))
    centres,orient,size=scene(.05)
    P=jnp.asarray(p)
    M=jnp.asarray(R)
    C,O,S=map(jnp.asarray,(centres,orient,size))
    foot=jnp.asarray(p[:,None,:])+jnp.asarray([[[.12,0.,-.13],[.02,.06,-.23]]])
    for subrays in (1,4,5):
        fn=jax.jit(lambda pts:scan_forward(
            jnp,pts,M,C,O,S,foot_centres=foot,subrays=subrays)[0])
        begin=time.perf_counter()
        output=fn(P)
        output.block_until_ready()
        compile_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        for _ in range(12):
            fn(P).block_until_ready()
        mean_ms=1000*(time.perf_counter()-begin)/12
        assert output.shape==(worlds,16)
        print(f"V5022_SENSOR_BENCH backend={jax.default_backend()} "
              f"worlds={worlds} rays={subrays*16} "
              f"compile_s={compile_seconds:.2f} mean_scan_ms={mean_ms:.3f}")



def test_browser_and_python_sensor_extrinsics_remain_identical():
    """One compact cross-language parity guard, no duplicated ray equations.

    Runs with the Node used by our existing viewer build; on Python-only
    installations it skips without preventing independent MJX tests.
    """
    import json
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("Node not installed; browser tests run separately")
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    cases=[dict(t=[-.18,0,.2584],tilt=35.,forward=.085,
                R=[[1,0,0],[0,1,0],[0,0,1]]),
           dict(t=[0,.03,.274],tilt=65.,forward=.160,
                R=[[0,-1,0],[1,0,0],[0,0,1]]),
           dict(t=[.025,0,.26],tilt=90.,forward=0.,
                R=[[1,0,0],[0,1,0],[0,0,1]])]
    js_code=(
        "import {depthProbeRays} from './emu/viewer_src/src/sensor.js';"
        f"const cases={json.dumps(cases)};"
        "console.log(JSON.stringify(cases.map(c=>depthProbeRays("
        "c.t,c.R.flat(),{tilt:c.tilt,forward:c.forward}))))"
    )
    result=subprocess.run(["node","--input-type=module","-e",js_code],
                          cwd=root,capture_output=True,text=True,check=True)
    actual=json.loads(result.stdout)
    for case,got in zip(cases,actual):
        p=np.asarray([case["t"]])
        R=np.asarray([case["R"]])
        expected_o,expected_d=camera_rays(
            np,p,R,tilt_down_deg=case["tilt"],
            mount_body_m=(case["forward"],0.,-.018))
        np.testing.assert_allclose(
            np.asarray(got["origin"]),expected_o[0,0,0],atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(got["directions"]),expected_d[0],atol=1e-7)



@pytest.mark.parametrize("variant,obstacle,features",[
    ("park","flat",True),
    ("park","flat",False),
    ("sensor","flat",False),
    ("sensor","step-up",False),
    ("sensor","step-down",False),
    ("sensor","crest",False),
    ("sensor","trough",False),
    ("sensor","log",False),
    ("sensor","angled",False),
])
def test_viewer_generated_worlds_load_in_real_mujoco(variant,obstacle,features):
    """Catch scene XML issues that syntax-only browser tests cannot detect.

    The actual archived robot_xml is used; the browser's pure terrain builder
    supplies physical geoms. No policy inference or training is performed.
    """
    import json
    import shutil
    import subprocess
    from pathlib import Path
    mujoco=pytest.importorskip("mujoco")
    if not shutil.which("node"):
        pytest.skip("Node required to inspect browser terrain XML")
    root=Path(__file__).resolve().parents[1]
    bundle=json.loads((root/"showcase/champion.emupolicy").read_text())
    src=str(bundle["robot_xml"])
    js=(
        "import {withTerrain,generateTerrainPark} from "
        "'./emu/viewer_src/src/terrain.js';"
        "import {readFileSync} from 'node:fs';"
        "const xml=readFileSync(0,'utf-8');"
        f"const kind={json.dumps(variant)},obstacle={json.dumps(obstacle)};"
        f"const features={json.dumps(features)};"
        "const park=kind==='park'?generateTerrainPark({features,trees:false}):null;"
        "console.log(withTerrain(xml,kind,{park,obstacle}));"
    )
    cmd=subprocess.run(["node","--input-type=module","-e",js],
                       input=src,text=True,capture_output=True,
                       cwd=root,check=True)
    xml=cmd.stdout.strip()
    assert '<mujoco' in xml
    model=mujoco.MjModel.from_xml_string(xml)
    assert model.ngeom>0
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    if variant=="park":
        assert floor == -1
        geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                              "park_feature_step_up")
        assert (geom>=0)==features
    elif obstacle in ("step-up","step-down","crest","trough"):
        assert floor == -1
    else:
        assert floor>=0
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert np.isfinite(data.geom_xpos).all()



def test_vertical_square_camera_projects_a_non_degenerate_rectangular_4x4():
    """Vertical VL53L8CX on a horizontal surface must give a real 2D grid.

    Old Euler-like independent pitch/yaw made the horizontal axis vanish
    at 90° down, giving V-shaped rays instead of a square detector field.
    """
    p,R=pose()
    origin,rays=camera_rays(np,p,R,tilt_down_deg=90.,
                           mount_body_m=(0.,0.,0.))
    direction=rays[0,:,0,:].reshape(4,4,3)
    assert np.all(direction[...,2]<0)
    hit=origin[0,:,0,:].reshape(4,4,3)
    hit=hit+(-hit[...,2]/direction[...,2])[...,None]*direction
    half=.2584*math.tan(math.pi/8)
    expected_x=(np.asarray([.75,.25,-.25,-.75])*half)
    expected_y=(np.asarray([-.75,-.25,.25,.75])*half)
    np.testing.assert_allclose(hit[:,:,0],
        np.broadcast_to(expected_x[:,None],(4,4)),atol=1e-7)
    np.testing.assert_allclose(hit[:,:,1],
        np.broadcast_to(expected_y[None,:],(4,4)),atol=1e-7)
    np.testing.assert_allclose(hit[:,:,2],0,atol=1e-7)
    assert np.ptp(hit[:,:,0])>.1 and np.ptp(hit[:,:,1])>.1


@pytest.mark.parametrize("tilt",[0.,25.,50.,75.,90.])
def test_camera_axes_do_not_collapse_across_mount_tilt(tilt):
    p,R=pose()
    _,rays=camera_rays(np,p,R,tilt_down_deg=tilt)
    assert rays.shape==(1,16,5,3)
    np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1,atol=1e-7)
    assert np.linalg.norm(rays[0,0,0]-rays[0,3,0])>.25
    assert np.linalg.norm(rays[0,0,0]-rays[0,12,0])>.25


def test_preferred_0mm_70deg_mount_sees_near_lateral_and_forward_ground():
    """Ground-only geometry for the viewer-selected provisional mount.

    An actual robot may block some of these zones. We deliberately test
    unoccluded rays first so feet/legs can be assessed independently.
    """
    p, R=pose()
    origin, direction=camera_rays(np,p,R)
    assert SENSOR_MOUNT_BODY_M == (0.,0.,-.018)
    assert SENSOR_DOWN_TILT_DEG == 70.
    np.testing.assert_allclose(origin[0,0,0],
                               [0.,0.,.2404],atol=1e-8)
    # Intersection on flat ground; row 0 sees ahead of rear row 3.
    centre=direction[0,:,0,:].reshape(4,4,3)
    places=origin[0,:,0,:].reshape(4,4,3)
    t=-places[:,:,2]/centre[:,:,2]
    ground=places +t[:,:,None]*centre
    assert np.isfinite(ground).all()
    # The two near outside corners sample opposite sides of the chassis.
    assert ground[3,0,1] < -.05
    assert ground[3,3,1] > .05
    assert abs(ground[3,0,0]-ground[3,3,0]) < 1e-6
    # Shallowest optical row provides a visible forward patch at 70°,
    # while near corners remain behind it and can measure lateral grade.
    assert ground[0,1,0] > .15
    assert ground[0,2,0] > .15
    assert ground[3,0,0] < .03
    assert ground[0,1,0]-ground[3,1,0] > .14


def test_preferred_mount_preserves_foot_occlusion_without_privileged_labels():
    """Two swing-foot returns may occupy near-centre zones.

    The raw sensor packet deliberately carries only range/validity;
    privileged simulator foot identity MUST NOT leak to the actor.
    """
    p,R=pose()
    world=scene(0.)
    baseline,ok=scan_forward(np,p,R,*world,subrays=5)
    origins,rays=camera_rays(np,p,R)
    inner_near=(13,14)
    foot_centres=np.asarray([
        origins[0,i,0]+.19*rays[0,i,0] for i in inner_near
    ])[None]
    crossed,valid=scan_forward(np,p,R,*world,foot_centres=foot_centres,
                              foot_radii=.018,subrays=5)
    assert crossed.shape == valid.shape == (1,16)
    assert (ok & valid).any()
    assert any(crossed[0,i]<baseline[0,i]-.015
               for i in inner_near if valid[0,i] and ok[0,i])
    # No oracle body-part masks: training will see just the measured slant
    # distance and validity, and must discover kinematic correlations.
    result=scan_forward(np,p,R,*world,foot_centres=foot_centres)
    assert len(result)==2
