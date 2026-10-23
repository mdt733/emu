import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { PolicyRunner } from './policy.js';
import { SensorObserver, depthProbeRays, readDepthProbe } from './sensor.js';
import { chasePose, smoothHeading, smoothFactor } from './chase-camera.js';
import { sampleEpisodeCadence } from './cadence.js';
import { joystickToCommands, nextYawCommand } from './controls.js';
import { measuredAttitude } from './attitude.js';
import { withTerrain, generateTerrainPark, surfaceSlope } from './terrain.js';
import { createGridTexture, createSkyTexture, createGrassTexture, createParkGridTexture, createParkSurface, createTreeCrowns, floorTileOrigin, FLOOR_SIZE_M, GroundTrail } from './appearance.js';

const TWO_PI = Math.PI * 2;
let mujocoModulePromise = null;
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function approach(x, target, maxDelta) {
  const d = target - x;
  if (Math.abs(d) <= maxDelta) return target;
  return x + Math.sign(d) * maxDelta;
}

class CapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius = 1, length = 1, capSegments = 8, radialSegments = 12) {
    super();
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);
    const lathe = new THREE.LatheGeometry(path.getPoints(capSegments), radialSegments);
    this.setIndex(lathe.getIndex());
    this.setAttribute('position', lathe.getAttribute('position'));
    this.setAttribute('normal', lathe.getAttribute('normal'));
    this.setAttribute('uv', lathe.getAttribute('uv'));
    lathe.dispose();
  }
}

export class EmuBrowserSim {
  static async create(canvas, bundle) {
    if (!mujocoModulePromise) {
      mujocoModulePromise = loadMujoco({
        locateFile: (path) => path.endsWith('.wasm') ? mujocoWasmUrl : path,
      });
    }
    const mujoco = await mujocoModulePromise;
    const sim = new EmuBrowserSim(canvas, bundle, mujoco);
    sim.loadModel();
    return sim;
  }

  constructor(canvas, bundle, mujoco) {
    this.canvas = canvas;
    this.bundle = bundle;
    this.mujoco = mujoco;
    this.policy = new PolicyRunner(bundle);
    this.model = null;
    this.data = null;
    this.mjvOption = new mujoco.MjvOption();
    this.mjvPerturb = new mujoco.MjvPerturb();
    this.mjvCamera = new mujoco.MjvCamera();
    this.mjvScene = null;
    this.maxGeoms = 2048;

    this.scene = new THREE.Scene();
    this.skyTexture = createSkyTexture(THREE);
    this.scene.background = this.skyTexture;
    this.scene.fog = new THREE.Fog(0xbad7e2, 15, 49);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.gridTexture = createGridTexture(THREE);
    this.grassTexture = createGrassTexture(THREE);
    this.parkGridTexture = createParkGridTexture(THREE);
    this.parkSurface = null;
    this.parkGridOverlay = null;
    this.treeCrowns = null;
    this.park = null;
    this.showGrid = true;
    this.showTrail = true;
    this.showShadows = true;
    this.createWorldGuides();
    this.createNavigationGuides();
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-0.72, -0.78, 0.52);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0.18);
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 11; // orbit can frame the full 4m mission arena
    this.cameraMode = 'orbit'; // orbit | chase | raw | level
    this.chaseReady = false;
    this.chaseHeading = 0;
    this.chaseTarget = new THREE.Vector3();
    this.orbitPosition = this.camera.position.clone();
    this.orbitTarget = this.controls.target.clone();
    this.levelPoseValid = false;
    this.levelQuaternion = new THREE.Quaternion();
    // A camera looks along local -Z. Here its optical axis is torso +X,
    // with image-up equal to torso +Z; the device sits ahead of torso_box.
    this.cameraMount = new THREE.Vector3(0.089, 0, 0.020);
    this.cameraBasis = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(
        new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0),
      ),
    );
    this.terrainMode = 'flat';
    this.terrainMaxDegrees = 10;
    this.sensorObstacle = 'flat';
    this.depthConfig = {enabled:true,forward:.09,tilt:50,showRays:true,subrays:'auto'};
    this.depthReading = null;
    this.depthLastMs = 0;
    this.depthRaycaster = new THREE.Raycaster();
    this.depthLinePositions = new Float32Array(16*6);
    this.depthLineColors = new Float32Array(16*6);
    const depthGeometry = new THREE.BufferGeometry();
    depthGeometry.setAttribute('position',new THREE.BufferAttribute(this.depthLinePositions,3).setUsage(THREE.DynamicDrawUsage));
    depthGeometry.setAttribute('color',new THREE.BufferAttribute(this.depthLineColors,3).setUsage(THREE.DynamicDrawUsage));
    this.depthLines = new THREE.LineSegments(depthGeometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.72,depthTest:true}));
    this.depthLines.frustumCulled=false;
    this.depthLines.visible=false;
    this.scene.add(this.depthLines);
    this.depthMountMarker = new THREE.Mesh(new THREE.SphereGeometry(.010,10,8),new THREE.MeshBasicMaterial({color:0xffcc80,depthTest:true}));
    this.depthMountMarker.visible=false;
    this.scene.add(this.depthMountMarker);
    this.parkTreesEnabled = true;
    this.headingHold = true;
    this.headingKp = 2.2; // yaw-rate [rad/s] per compass-heading error [rad]
    this.headingTarget = 0;
    this.requestedYaw = 0; // operator/demo rate, independent of heading-hold correction
    this.navPresentationActive = false; // presentation only; no alternate actor control path

    this.meshes = [];
    this.geometryCache = new Map();
    this.follow = true;
    this.followTarget = new THREE.Vector3();
    this.paused = false;
    this.autoReset = false;
    this.sensorNoise = true;
    this.sensorObserver = bundle.schema === 2
      ? new SensorObserver(bundle.observation.sensor_contract) : null;
    this.fallenFor = 0;
    this.controlFrames = 0; // visible proof of live policy inference
    this.motorTravelRad = 0; // integrated actuated joint movement (rad)

    this.commandTarget = { vx: 0, vy: 0, yaw: 0, height: 0, pitch: 0, roll: 0, step_height: 0.012 };
    this.command = { ...this.commandTarget };
    this.episodePhaseHz = 1.025;
    this.phase = 0;
    this.prevAction = new Float32Array(6);
    this.prevRawTarget = new Float32Array(6);
    this.deliveredCtrl = new Float32Array(6);
    this.baseObsDim = bundle.policy.base_obs_dim;
    this.history = Array.from({ length: 4 }, () => new Float32Array(this.baseObsDim));
    this.obs = new Float32Array(bundle.policy.obs_dim);
    this.actQpos = [];
    this.actQvel = [];
    this.shinQpos = [];
    this.shinQvel = [];
    this.torsoId = -1;
    this.footIds = [];
    this.controlAccumulator = 0;
    this.lastWall = performance.now() / 1000;

    this.addLights();
    this.resize();
  }

  addLights() {
    const hemi = new THREE.HemisphereLight(0xc9e0fb, 0x344350, 1.15);
    this.scene.add(hemi);
    const fill = new THREE.DirectionalLight(0x9bc4db, 0.5);
    fill.position.set(2, 2.5, 2.2);
    this.scene.add(fill);
    this.keyLight = new THREE.DirectionalLight(0xffefdb, 2.05);
    this.keyLight.position.set(-1.8, -2.4, 3.5);
    this.keyLight.castShadow = true;
    // 1024 rather than 2048: a quarter of the old shadow-map pixels.
    this.keyLight.shadow.mapSize.set(1024, 1024);
    this.keyLight.shadow.camera.left = -2.4;
    this.keyLight.shadow.camera.right = 2.4;
    this.keyLight.shadow.camera.top = 2.4;
    this.keyLight.shadow.camera.bottom = -2.4;
    this.keyLight.shadow.camera.near = 0.15;
    this.keyLight.shadow.camera.far = 12;
    this.keyLight.shadow.camera.updateProjectionMatrix();
    this.keyLight.shadow.bias = -0.00015;
    this.keyLight.shadow.normalBias = 0.008;
    this.keyLight.target.position.set(0, 0, 0.08);
    this.scene.add(this.keyLight, this.keyLight.target);
  }

  createWorldGuides() {
    const trail = new GroundTrail();
    this.pathTrail = trail;
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute('position', new THREE.BufferAttribute(trail.coords, 3).setUsage(THREE.DynamicDrawUsage));
    trailGeometry.setDrawRange(0, 0);
    this.trailLine = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({color: 0x80d6d2, transparent: true, opacity: 0.78, depthWrite: false}));
    this.trailLine.frustumCulled = false;
    this.scene.add(this.trailLine);

    // World X (warm), world Y (cyan). Only two simple line segments.
    const axes = new THREE.BufferGeometry();
    axes.setAttribute('position', new THREE.Float32BufferAttribute([
      -1.5, 0, 0.009, 1.5, 0, 0.009,
      0, -1.5, 0.009, 0, 1.5, 0.009,
    ], 3));
    axes.setAttribute('color', new THREE.Float32BufferAttribute([
      0.94, 0.62, 0.32, 0.94, 0.62, 0.32,
      0.35, 0.80, 0.91, 0.35, 0.80, 0.91,
    ], 3));
    this.worldAxes = new THREE.LineSegments(axes, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 0.62, depthWrite: false}));
    this.scene.add(this.worldAxes);
  }

  setAppearance({grid, trail, shadows} = {}) {
    if (typeof grid === 'boolean' && grid !== this.showGrid) {
      this.showGrid = grid;
      for (const mesh of this.meshes) {
        if (!mesh.userData.floor) continue;
        mesh.material.map = grid ? this.gridTexture : null;
        mesh.material.color.setHex(grid ? 0xffffff : 0x40515e);
        mesh.material.needsUpdate = true;
      }
      this.worldAxes.visible = grid;
      if (this.parkGridOverlay) this.parkGridOverlay.visible = grid;
    }
    if (typeof trail === 'boolean') {
      this.showTrail = trail;
      this.trailLine.visible = trail;
    }
    if (typeof shadows === 'boolean' && shadows !== this.showShadows) {
      this.showShadows = shadows;
      this.keyLight.castShadow = shadows;
      this.renderer.shadowMap.enabled = shadows;
      this.renderer.shadowMap.needsUpdate = true;
    }
  }

  disposeModel() {
    if (this.parkGridOverlay) {
      this.scene.remove(this.parkGridOverlay);
      // Deliberately share parkSurface geometry; dispose only the overlay
      // material. The park mesh below owns geometry lifecycle.
      this.parkGridOverlay.material.dispose();
      this.parkGridOverlay = null;
    }
    if (this.parkSurface) {
      this.scene.remove(this.parkSurface);
      this.parkSurface.geometry.dispose();
      this.parkSurface.material.dispose();
      this.parkSurface = null;
    }
    if (this.treeCrowns) {
      this.scene.remove(this.treeCrowns);
      this.treeCrowns.userData.sharedGeometry.dispose();
      for (const material of this.treeCrowns.userData.sharedMaterials) material.dispose();
      this.treeCrowns = null;
    }
    this.park = null;
    this.depthReading=null;
    this.depthLastMs=0;
    // Terrain geometry moved; rebuild guide heights on the next nav tick.
    this.navGoalSeen=null;this.navRouteSeen=null;
    for (const mesh of this.meshes) {
      this.scene.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    }
    this.meshes.length = 0;
    for (const geom of this.geometryCache.values()) geom.dispose();
    this.geometryCache.clear();
    if (this.mjvScene) { this.mjvScene.delete(); this.mjvScene = null; }
    if (this.data) { this.data.delete(); this.data = null; }
    if (this.model) { this.model.delete(); this.model = null; }
  }

  dispose() {
    this.disposeModel();
    this.depthLines.geometry.dispose();
    this.depthLines.material.dispose();
    this.depthMountMarker.geometry.dispose();
    this.depthMountMarker.material.dispose();
    this.routeLine.geometry.dispose();
    this.routeLine.material.dispose();
    this.targetRing.geometry.dispose();
    this.targetRing.material.dispose();
    for(const mesh of [this.targetArrow.line,this.targetArrow.cone]){
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.arenaCircle.geometry.dispose();
    this.arenaCircle.material.dispose();
    this.skyTexture.dispose();
    this.gridTexture.dispose();
    this.grassTexture.dispose();
    this.parkGridTexture.dispose();
    this.trailLine.geometry.dispose();
    this.trailLine.material.dispose();
    this.worldAxes.geometry.dispose();
    this.worldAxes.material.dispose();
    this.mjvCamera?.delete();
    this.mjvPerturb?.delete();
    this.mjvOption?.delete();
    this.controls?.dispose();
    this.renderer?.dispose();
  }

  setCameraMode(mode) {
    if (!['orbit', 'chase', 'raw', 'level'].includes(mode)) throw new Error(`Unknown camera mode: ${mode}`);
    if (mode === this.cameraMode) return;
    if (this.cameraMode === 'orbit') {
      this.orbitPosition.copy(this.camera.position);
      this.orbitTarget.copy(this.controls.target);
    }
    this.cameraMode = mode;
    this.controls.enabled = mode === 'orbit';
    this.levelPoseValid = false;
    this.chaseReady = false;
    if (mode === 'orbit') {
      this.camera.position.copy(this.orbitPosition);
      this.controls.target.copy(this.orbitTarget);
      this.camera.up.set(0, 0, 1);
      this.controls.update();
    }
  }

  setTerrainMode(mode, maximum=this.terrainMaxDegrees) {
    if (!['flat','ramp','park','sensor'].includes(mode)) throw new Error('Unknown terrain');
    const degrees = Number(maximum);
    if (!Number.isFinite(degrees) || degrees < 1 || degrees > 20) throw new Error('Terrain angle must be 1–20°');
    if (mode===this.terrainMode && degrees===this.terrainMaxDegrees) return;
    this.terrainMode=mode;
    this.terrainMaxDegrees=degrees;
    this.loadModel(); // rebuilds only browser MuJoCo contacts
  }
  setTerrainEnabled(enabled) { this.setTerrainMode(enabled ? 'ramp' : 'flat'); }
  setSensorObstacle(id) {
    if(!['flat','step-up','step-down','crest','trough','log','angled'].includes(id))throw new Error('Unknown sensor course');
    if(id===this.sensorObstacle)return;
    this.sensorObstacle=id;
    if(this.terrainMode==='sensor')this.loadModel();
  }
  setDepthInspector({enabled,forward,tilt,showRays,subrays='auto'}) {
    const values={enabled:Boolean(enabled),forward:Number(forward),tilt:Number(tilt),showRays:Boolean(showRays),subrays};
    if(!['auto',1,4,5].includes(subrays))throw new Error('Depth samples must be auto, 1, 4 or 5');
    // Validation lives with one optical mounting model, not a policy contract.
    if(values.forward<.078||values.forward>.2||values.tilt<15||values.tilt>85)throw new Error('Invalid depth mount');
    this.depthConfig=values;
    this.depthLastMs=0;
  }
  setTreesEnabled(enabled) {
    const next=Boolean(enabled);
    if (next===this.parkTreesEnabled) return;
    this.parkTreesEnabled=next;
    if(this.terrainMode==='park')this.loadModel();
  }
  setHeadingHold(enabled) {
    this.headingHold=Boolean(enabled);
    // Bumpless engage: no jump towards an old target from an earlier mode.
    this.headingTarget=this.actualHeading();
  }
  setHeadingGain(kp) {
    if (!Number.isFinite(kp) || kp < 0.3 || kp > 5)
      throw new Error('Heading P gain must be between 0.3 and 5');
    this.headingKp=kp;
  }
  manualCommandLimits() {
    return this.bundle.control.command_limits;
  }
  actualHeading() {
    if (!this.data || !this.model) return 0;
    const r=this.bodyRotation();
    return Math.atan2(r[3],r[0]); // torso +X projected into world XY
  }

  loadModel() {
    this.disposeModel();
    this.park = this.terrainMode==='park'
      ? generateTerrainPark({maxDegrees:this.terrainMaxDegrees,trees:this.parkTreesEnabled})
      : null;
    const xml = withTerrain(this.bundle.robot_xml,this.terrainMode,{park:this.park,obstacle:this.sensorObstacle});
    this.model = this.mujoco.MjModel.from_xml_string(xml);
    if (!this.model) throw new Error('MuJoCo could not load robot XML');
    this.data = new this.mujoco.MjData(this.model);
    this.mjvScene = new this.mujoco.MjvScene(this.model, this.maxGeoms);
    this.resolveIds();
    if(this.park){
      this.parkSurface=createParkSurface(THREE,this.park,this.grassTexture);
      this.scene.add(this.parkSurface);
      // Depth-tested so feet, rocks and trunks can occlude the grid, with
      // polygon offset and render order to prevent coplanar flicker. It hugs
      // every peak and depression: no floating world-Z grid on slopes.
      this.parkGridOverlay=new THREE.Mesh(this.parkSurface.geometry,
        new THREE.MeshBasicMaterial({
          map:this.parkGridTexture,transparent:true,opacity:1,
          depthWrite:false,depthTest:true,polygonOffset:true,
          polygonOffsetFactor:-1,polygonOffsetUnits:-2,
          side:THREE.DoubleSide,toneMapped:false,
        }));
      this.parkGridOverlay.renderOrder=3;
      this.parkGridOverlay.visible=this.showGrid;
      this.parkGridOverlay.frustumCulled=false;
      this.scene.add(this.parkGridOverlay);
      this.treeCrowns=createTreeCrowns(THREE,this.park.trees);
      this.scene.add(this.treeCrowns);
    }
    const d = this.bundle.control.default_commands || {};
    this.commandTarget = {
      vx: Number(d.vx ?? 0.04), vy: Number(d.vy ?? 0), yaw: Number(d.yaw ?? 0),
      height: Number(d.height ?? 0), pitch: Number(d.pitch ?? 0), roll: Number(d.roll ?? 0),
      step_height: Number(d.step_height ?? 0.012),
    };
    this.command = { ...this.commandTarget };
    this.requestedYaw = this.commandTarget.yaw;
    this.reset();
  }

  name2id(objEnum, name) {
    const id = Number(this.mujoco.mj_name2id(this.model, objEnum.value, name));
    if (id < 0) throw new Error(`robot.xml is missing ${name}`);
    return id;
  }

  resolveIds() {
    const mj = this.mujoco;
    const joints = this.bundle.control.actuated_joints;
    const jids = joints.map(n => this.name2id(mj.mjtObj.mjOBJ_JOINT, n));
    this.actQpos = jids.map(j => Number(this.model.jnt_qposadr[j]));
    this.actQvel = jids.map(j => Number(this.model.jnt_dofadr[j]));
    const sj = ['r_shin_flex', 'l_shin_flex'].map(n => this.name2id(mj.mjtObj.mjOBJ_JOINT, n));
    this.shinQpos = sj.map(j => Number(this.model.jnt_qposadr[j]));
    this.shinQvel = sj.map(j => Number(this.model.jnt_dofadr[j]));
    this.torsoId = this.name2id(mj.mjtObj.mjOBJ_BODY, 'torso');
    this.footIds = ['r_foot', 'l_foot'].map(n => this.name2id(mj.mjtObj.mjOBJ_BODY, n));
  }

  reset() {
    this.resetGeneration=(this.resetGeneration||0)+1;
    const mj = this.mujoco;
    mj.mj_resetData(this.model, this.data);
    this.data.ctrl.fill(0);
    this.data.qvel.fill(0);
    // Training births already moving near the commanded state. Do the same for
    // the interactive viewer so reset is not a special stationary task.
    this.data.qvel[0] = this.command.vx;
    this.data.qvel[1] = this.command.vy;
    this.requestedYaw = this.commandTarget.yaw;
    this.command.yaw = this.requestedYaw;
    this.data.qvel[5] = this.command.yaw;
    mj.mj_forward(this.model, this.data);
    this.headingTarget=this.actualHeading();
    // Training samples cadence at birth, then holds it through command changes.
    this.episodePhaseHz = sampleEpisodeCadence(this.command, this.bundle.control);
    this.phase = 0;
    this.prevAction.fill(0);
    this.prevRawTarget.fill(0);
    this.deliveredCtrl.fill(0);
    if (this.sensorObserver) this.sensorObserver.reset(this.sensorPose(),this.sensorNoise);
    const base = this.buildBaseObs();
    for (let i = 0; i < 4; i++) this.history[i].set(base);
    this.flattenHistory();
    this.controlAccumulator = 0;
    this.fallenFor = 0;
    this.pathTrail.clear();
    this.levelPoseValid = false;
    this.chaseReady = false;
    this.trailLine.geometry.setDrawRange(0, 0);
    this.depthLastMs=0;
    this.refreshScene();
  }

  setJoystickTargets(leftX, leftY, rightX, rightY) {
    // One mapping for manual command values and their on-screen labels.
    // Positive screen-right must map to the physically rightward strafe.
    Object.assign(this.commandTarget, joystickToCommands(
      leftX, leftY, rightX, rightY, this.manualCommandLimits(),
    ));
  }

  setDemoCommands({ vx, vy, yaw, pitch }) {
    const l = this.bundle.control.command_limits;
    this.commandTarget.vx = clamp(Number(vx), l.vx_min, l.vx_max);
    this.commandTarget.vy = clamp(Number(vy), -l.vy_abs, l.vy_abs);
    this.commandTarget.yaw = clamp(Number(yaw), -l.yaw_abs, l.yaw_abs);
    this.commandTarget.pitch = clamp(Number(pitch), -l.pitch_abs_rad, l.pitch_abs_rad);
  }

  navigationPose() {
    const o = this.torsoId * 3;
    const heading=this.actualHeading(),c=Math.cos(heading),s=Math.sin(heading);
    const wx=Number(this.data.qvel[0]),wy=Number(this.data.qvel[1]);
    // MuJoCo freejoint angular qvel is body-local; turn it into an
    // observed horizontal heading rate for navigation's yaw damping.
    const R=this.bodyRotation(),q=this.data.qvel;
    const ax=R[0]*q[3]+R[1]*q[4]+R[2]*q[5];
    const ay=R[3]*q[3]+R[4]*q[4]+R[5]*q[5];
    const az=R[6]*q[3]+R[7]*q[4]+R[8]*q[5];
    const horizontal=Math.max(.01,R[0]*R[0]+R[3]*R[3]);
    const yawRate=az-R[6]*(R[0]*ax+R[3]*ay)/horizontal;
    return {x:Number(this.data.xpos[o]),y:Number(this.data.xpos[o+1]),
      heading,vx:c*wx+s*wy,vy:-s*wx+c*wy,yawRate,
      fallen:this.isFallen()};
  }

  navigationTrees() {
    return this.terrainMode==='park' && this.parkTreesEnabled
      ? this.park?.trees || [] : [];
  }

  setNavigationPresentation(active) {
    const wasActive=this.navPresentationActive;
    this.navPresentationActive=Boolean(active);
    if(active&&!wasActive&&this.cameraMode==='orbit'){
      // Frame the route rather than only a robot-sized patch of grass.
      this.navSavedOrbit={position:this.camera.position.clone(),
                          target:this.controls.target.clone()};
      const p=this.navigationPose();
      this.controls.target.set(p.x,p.y,.12);
      this.camera.position.set(p.x-3.4,p.y-4.0,3.8);
      this.controls.update();
    } else if(!active&&wasActive&&this.navSavedOrbit){
      if(this.cameraMode==='orbit'){
        this.camera.position.copy(this.navSavedOrbit.position);
        this.controls.target.copy(this.navSavedOrbit.target);
        this.controls.update();
      }
      this.navSavedOrbit=null;
    }
    // Navigation changes camera and route overlay ONLY. It must never
    // modify neural control, heading P, current velocity, or commands.
    if(!active)this.clearNavigationOverlay();
  }

  terrainHeightAt(x,y) {
    return this.terrainMode==='park' && this.park ? this.park.sampleHeight(x,y) : 0;
  }

  createNavigationGuides() {
    const mat=new THREE.MeshBasicMaterial({color:0xfce38b,side:THREE.DoubleSide,
                                           depthWrite:false,depthTest:true});
    this.targetRing=new THREE.Mesh(new THREE.RingGeometry(.16,.205,40),mat);
    this.targetRing.renderOrder=9;
    this.scene.add(this.targetRing);
    this.targetArrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),
       new THREE.Vector3(0,0,.06),.50,0xffd96e,.15,.075);
    this.targetArrow.renderOrder=9;
    this.scene.add(this.targetArrow);
    this.routeLine=new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color:0x54f8da,depthTest:true,
                                   depthWrite:false,transparent:true,opacity:.95}));
    this.routeLine.renderOrder=8;
    this.scene.add(this.routeLine);
    this.arenaCircle=new THREE.LineLoop(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color:0x7eb8ff,transparent:true,
                                   opacity:.46,depthWrite:false}));
    this.arenaCircle.renderOrder=7;
    this.scene.add(this.arenaCircle);
    this.navGoalSeen=null;this.navRouteSeen=null;
    this.clearNavigationOverlay();
  }

  clearNavigationOverlay() {
    this.targetRing.visible=false;
    this.targetArrow.visible=false;
    this.routeLine.visible=false;
    this.arenaCircle.visible=false;
    this.navGoalSeen=null;this.navRouteSeen=null;
  }

  setNavigationOverlay(goal,route) {
    if(!goal||!route){this.clearNavigationOverlay();return;}
    if(this.navGoalSeen===goal && this.navRouteSeen===route)return;
    const z=this.terrainHeightAt(goal.x,goal.y);
    this.targetRing.position.set(goal.x,goal.y,z+.030);
    this.targetArrow.position.set(goal.x,goal.y,z+.045);
    this.targetArrow.setDirection(new THREE.Vector3(Math.cos(goal.heading),
      Math.sin(goal.heading),0));
    // Refresh the predicted route on new targets or meaningful replans.
    const pose=this.navigationPose();
    const points=[new THREE.Vector3(pose.x,pose.y,
      this.terrainHeightAt(pose.x,pose.y)+.038),
      ...route.steps.map(p=>new THREE.Vector3(p.x,p.y,this.terrainHeightAt(p.x,p.y)+.038))];
    this.routeLine.geometry.dispose();
    this.routeLine.geometry=new THREE.BufferGeometry().setFromPoints(points);
    const circle=[];
    for(let i=0;i<128;i++){
      const a=2*Math.PI*i/128,x=4*Math.cos(a),y=4*Math.sin(a);
      circle.push(new THREE.Vector3(x,y,this.terrainHeightAt(x,y)+.032));
    }
    this.arenaCircle.geometry.dispose();
    this.arenaCircle.geometry=new THREE.BufferGeometry().setFromPoints(circle);
    for(const mesh of [this.targetRing,this.targetArrow,this.routeLine,this.arenaCircle])
      mesh.visible=true;
    this.navGoalSeen=goal;this.navRouteSeen=route;
  }

  canSwapPolicy(next) {
    const old = this.bundle;
    // A browser replay can only replace the actor if physics, observations and
    // actuator interface are identical. Otherwise load an entirely new model.
    const physicsKeys = ['sim_dt','control_substeps','control_dt',
      'actuated_joints','target_span_rad','one_control_frame_actuation_delay',
      'phase_base_hz','phase_rate_max_hz','phase_speed_gain','phase_speed_reference_mps'];
    return next.robot_xml === old.robot_xml &&
      physicsKeys.every(k => JSON.stringify(next.control[k]) === JSON.stringify(old.control[k])) &&
      JSON.stringify(next.observation.base_fields) === JSON.stringify(old.observation.base_fields) &&
      JSON.stringify(next.observation.sensor_contract) === JSON.stringify(old.observation.sensor_contract) &&
      next.policy.obs_dim === old.policy.obs_dim;
  }

  replacePolicy(next) {
    if (!this.canSwapPolicy(next)) throw new Error('Model/control contract changed; reload simulation');
    const newActor = new PolicyRunner(next); // validation before changing state
    this.bundle = next;
    this.policy = newActor;
    this.reset(); // clean comparison, not a fictitious uninterrupted rollout
  }

  smoothCommands(dt) {
    this.command.vx = approach(this.command.vx, this.commandTarget.vx, 1.1 * dt);
    this.command.vy = approach(this.command.vy, this.commandTarget.vy, 0.9 * dt);
    // Yaw is smoothed separately in nextYawCommand (not from applied yaw).
    this.command.pitch = approach(this.command.pitch, this.commandTarget.pitch, 0.70 * dt);
    this.command.height = approach(this.command.height, this.commandTarget.height, 0.045 * dt);
    this.command.roll = approach(this.command.roll, this.commandTarget.roll, 0.55 * dt);
    this.command.step_height = approach(this.command.step_height, this.commandTarget.step_height, 0.050 * dt);
  }

  phaseRate() {
    return this.episodePhaseHz;
  }

  stepControl() {
    const cdt = Number(this.bundle.control.control_dt || 0.02);
    this.smoothCommands(cdt);
    // One shared yaw controller for joystick and autonomous navigation.
    // Navigation is only an upstream producer of vx/vy/yaw/pitch requests.
    const steering = nextYawCommand(this.requestedYaw,this.commandTarget.yaw,
      this.headingTarget,this.actualHeading(),cdt,this.headingHold,this.headingKp,
      Number(this.bundle.control.command_limits.yaw_abs));
    this.requestedYaw=steering.requestedYaw;
    this.headingTarget=steering.headingTarget;
    this.command.yaw=steering.appliedYaw;
    const action = this.policy.infer(this.obs);
    this.controlFrames++;
    const span = this.bundle.control.target_span_rad;

    // One control-frame delay: deliver last target, remember this action's
    // target for the next control frame. This mirrors environment.py.
    for (let i = 0; i < 6; i++) {
      this.deliveredCtrl[i] = this.prevRawTarget[i];
      this.data.ctrl[i] = this.deliveredCtrl[i];
    }

    const nsub = Number(this.bundle.control.control_substeps || 4);
    const beforeJoints=this.actQpos.map(i=>Number(this.data.qpos[i]));
    for (let k = 0; k < nsub; k++) this.mujoco.mj_step(this.model, this.data);
    for(let i=0;i<this.actQpos.length;i++)
      this.motorTravelRad+=Math.abs(Number(this.data.qpos[this.actQpos[i]])-beforeJoints[i]);

    for (let i = 0; i < 6; i++) {
      const raw = clamp(action[i] * span[i], -span[i], span[i]);
      this.prevRawTarget[i] = raw;
      this.prevAction[i] = action[i];
    }

    this.phase = (this.phase + cdt * this.phaseRate() * TWO_PI) % TWO_PI;
    if (this.sensorObserver) this.sensorObserver.advance(this.sensorPose(),this.sensorNoise);
    const base = this.buildBaseObs();
    this.history[0].set(this.history[1]);
    this.history[1].set(this.history[2]);
    this.history[2].set(this.history[3]);
    this.history[3].set(base);
    this.flattenHistory();

    if (this.isFallen()) this.fallenFor += cdt; else this.fallenFor = 0;
    if (this.autoReset && this.fallenFor > 0.8) this.reset();
  }

  flattenHistory() {
    for (let h = 0; h < 4; h++) this.obs.set(this.history[h], h * this.baseObsDim);
  }

  bodyRotation() {
    const o = this.torsoId * 9;
    const m = this.data.xmat;
    return [m[o],m[o+1],m[o+2],m[o+3],m[o+4],m[o+5],m[o+6],m[o+7],m[o+8]];
  }

  worldToBody(R, x, y, z) {
    return [
      R[0]*x + R[3]*y + R[6]*z,
      R[1]*x + R[4]*y + R[7]*z,
      R[2]*x + R[5]*y + R[8]*z,
    ];
  }

  sensorPose() {
    const R=this.bodyRotation();
    const xyz=this.torsoId*3;
    const torso=Array.from(this.data.xpos.slice(xyz,xyz+3));
    const vbody=this.worldToBody(R,this.data.qvel[0],this.data.qvel[1],this.data.qvel[2]);
    // MuJoCo freejoint angular qvel is body-local, not world angular velocity.
    const gyro=Array.from(this.data.qvel.slice(3,6));
    const heightAt=(x,y)=>{
      if(this.terrainMode==='park'&&this.park) return this.park.sampleHeight(x,y);
      if(this.terrainMode==='ramp'&&Math.abs(y)<.44){
        if(x>=.85&&x<=1.65)return .008+(x-.85)*Math.tan(Math.PI/180);
        if(x>1.65&&x<2.17)return .015;
        if(x>=2.17&&x<=2.97)return .015-(x-2.17)*Math.tan(Math.PI/180);
      }
      return 0;
    };
    return {R,torso,vbody,gyro,heightAt};
  }

  buildBaseObs() {
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const xpos = this.data.xpos;
    const R = this.bodyRotation();
    const grav = this.worldToBody(R, 0, 0, -1);
    const bodyVel = this.worldToBody(R, qvel[0], qvel[1], qvel[2]);
    const bodyAng = this.worldToBody(R, qvel[3], qvel[4], qvel[5]);
    // Training defaults to 1.5× synthetic sensor noise, not 1.0×.
    // When disabled this remains a deliberately idealised comparison mode.
    const ns = this.sensorNoise ? Number(this.bundle.control.sensor_noise_scale ?? 1.5) : 0.0;
    const n0 = Math.sin(1.73 * this.phase);
    const n1 = Math.sin(2.31 * this.phase + 0.7);
    const base = new Float32Array(this.baseObsDim);
    let k = 0;

    base[k++] = grav[0] + ns*0.008*n0;
    base[k++] = grav[1] + ns*0.008*n1;
    base[k++] = grav[2] - ns*0.004*n0;
    base[k++] = bodyAng[0] + ns*0.025*n1;
    base[k++] = bodyAng[1] + ns*0.025*n0;
    base[k++] = bodyAng[2] - ns*0.025*n1;
    const sensor = this.sensorObserver?.read([
      grav[0]+ns*.008*n0,grav[1]+ns*.008*n1,grav[2]-ns*.004*n0
    ]);
    if (sensor) {
      base[3]=sensor.gyro[0];
      base[4]=sensor.gyro[1];
      base[5]=sensor.gyro[2];
    }

    for (let i = 0; i < 6; i++) base[k++] = qpos[this.actQpos[i]] + ns*0.0015*n0;
    for (let i = 0; i < 6; i++) base[k++] = qvel[this.actQvel[i]] + ns*0.015*n1;
    for (let i = 0; i < 6; i++) base[k++] = this.prevAction[i];
    for (let i = 0; i < 6; i++) base[k++] = this.deliveredCtrl[i];

    const to = this.torsoId * 3;
    const tx = xpos[to], ty = xpos[to+1], tz = xpos[to+2];
    for (const fid of this.footIds) {
      const fo = fid * 3;
      const rel = this.worldToBody(R, xpos[fo]-tx, xpos[fo+1]-ty, xpos[fo+2]-tz);
      base[k++] = rel[0]; base[k++] = rel[1]; base[k++] = rel[2];
    }

    base[k++] = sensor ? sensor.velocity[0] : bodyVel[0] + ns*0.018*n0;
    base[k++] = sensor ? sensor.velocity[1] : bodyVel[1] + ns*0.018*n1;
    base[k++] = sensor ? sensor.depth : tz + ns*0.0025*n1;
    base[k++] = sensor ? sensor.velocityConfidence : clamp(1 - 0.18*ns, 0.2, 1.0);

    base[k++] = this.command.vx;
    base[k++] = this.command.vy;
    base[k++] = this.command.yaw;
    base[k++] = this.command.height;
    base[k++] = this.command.pitch;
    base[k++] = this.command.roll;
    base[k++] = this.command.step_height;
    base[k++] = Math.sin(this.phase);
    base[k++] = Math.cos(this.phase);

    const sf0 = qpos[this.shinQpos[0]] + ns*0.0010*n0;
    const sf1 = qpos[this.shinQpos[1]] + ns*0.0010*n1;
    const sr0 = qvel[this.shinQvel[0]] + ns*0.010*n1;
    const sr1 = qvel[this.shinQvel[1]] + ns*0.010*n0;
    base[k++] = sf0; base[k++] = sf1;
    base[k++] = sr0; base[k++] = sr1;
    base[k++] = clamp(Math.abs(sf0) / 0.10, 0, 1.5);
    base[k++] = clamp(Math.abs(sf1) / 0.10, 0, 1.5);

    if (sensor) {
      base[k++] = sensor.forwardGrade;
      base[k++] = sensor.sideGrade;
      base[k++] = sensor.groundConfidence;
    }
    if (k !== this.baseObsDim) throw new Error(`Internal observation builder produced ${k} fields instead of ${this.baseObsDim}`);
    return base;
  }

  isFallen() {
    const R = this.bodyRotation();
    const gravityZ = -this.worldToBody(R, 0, 0, -1)[2];
    return this.heightM() < 0.145 || gravityZ < 0.35;
  }

  heightM() { return Number(this.data.xpos[this.torsoId * 3 + 2]); }

  telemetry() {
    const R = this.bodyRotation();
    const vel = this.worldToBody(R, this.data.qvel[0], this.data.qvel[1], this.data.qvel[2]);
    const ang = this.worldToBody(R, this.data.qvel[3], this.data.qvel[4], this.data.qvel[5]);
    const xyz = id => [this.data.xpos[id * 3], this.data.xpos[id * 3 + 1], this.data.xpos[id * 3 + 2]];
    const pos = xyz(this.torsoId);
    const feet = this.footIds.map(xyz);
    const { chassisPitch, robotLean } = measuredAttitude(R, pos, feet[0], feet[1]);
    return {
      height: this.heightM(), vx: vel[0], vy: vel[1], yaw: ang[2],
      pitch: chassisPitch, chassisPitch, robotLean,
      cmd: { ...this.command }, requestedYaw: this.requestedYaw,
      headingHold: this.headingHold, headingActual: this.actualHeading(),
      headingTarget: this.headingTarget,
      slope: surfaceSlope(this.terrainMode, this.park,
        (feet[0][0]+feet[1][0])/2, (feet[0][1]+feet[1][1])/2, this.actualHeading()),
      fallen: this.isFallen(), phaseRate: this.phaseRate(),
    };
  }

  advance(realDt) {
    if (this.paused) return;
    const cdt = Number(this.bundle.control.control_dt || 0.02);
    this.controlAccumulator += Math.min(realDt, 0.08);
    let safety = 0;
    while (this.controlAccumulator >= cdt && safety++ < 6) {
      this.stepControl();
      this.controlAccumulator -= cdt;
    }
  }

  geometryFor(mjvGeom) {
    const mj = this.mujoco;
    const type = Number(mjvGeom.type);
    const dataId = Number(mjvGeom.dataid);
    const size = Array.from(mjvGeom.size, Number);
    const key = JSON.stringify([type, size, dataId]);
    if (this.geometryCache.has(key)) return this.geometryCache.get(key);
    let g;
    if (type === mj.mjtGeom.mjGEOM_PLANE.value) {
      g = new THREE.PlaneGeometry(FLOOR_SIZE_M, FLOOR_SIZE_M);
    } else if (type === mj.mjtGeom.mjGEOM_SPHERE.value) {
      g = new THREE.SphereGeometry(size[0], 24, 16);
    } else if (type === mj.mjtGeom.mjGEOM_CAPSULE.value) {
      g = new CapsuleGeometry(size[0], 2*size[2], 16, 12); g.rotateX(Math.PI/2);
    } else if (type === mj.mjtGeom.mjGEOM_BOX.value) {
      g = new THREE.BoxGeometry(2*size[0], 2*size[1], 2*size[2]);
    } else if (type === mj.mjtGeom.mjGEOM_CYLINDER.value) {
      g = new THREE.CylinderGeometry(size[0], size[0], 2*size[2], 20); g.rotateX(Math.PI/2);
    } else if (type === mj.mjtGeom.mjGEOM_ELLIPSOID.value) {
      g = new THREE.SphereGeometry(1, 20, 14); g.scale(size[0], size[1], size[2]);
    } else {
      g = new THREE.BufferGeometry();
    }
    this.geometryCache.set(key, g);
    return g;
  }

  refreshScene() {
    if (!this.model || !this.data) return;
    const mj = this.mujoco;
    mj.mjv_updateScene(this.model, this.data, this.mjvOption, this.mjvPerturb, this.mjvCamera, mj.mjtCatBit.mjCAT_ALL.value, this.mjvScene);
    const geoms = this.mjvScene.geoms;
    const count = Number(geoms.size());
    for (let i = 0; i < count; i++) {
      const vg = geoms.get(i);
      const rgba = Array.from(vg.rgba, Number);
      const mat = Array.from(vg.mat, Number);
      const pos = Array.from(vg.pos, Number);
      let mesh = this.meshes[i];
      if (!mesh) {
        const isFloor = Number(vg.type) === mj.mjtGeom.mjGEOM_PLANE.value;
        const isParkHeightfield = Number(vg.type) === mj.mjtGeom.mjGEOM_HFIELD.value;
        const material = isFloor
          ? new THREE.MeshStandardMaterial({map: this.showGrid ? this.gridTexture : null, roughness: 0.98, metalness: 0, color: this.showGrid ? 0xffffff : 0x40515e})
          : new THREE.MeshStandardMaterial({ roughness: 0.68, metalness: 0.06 });
        mesh = new THREE.Mesh(this.geometryFor(vg), material);
        mesh.userData.floor = isFloor;
        // Scene dataid identifies MuJoCo geometry; non-world bodies are the
        // moving robot, including feet and leg links. Do not hide them.
        mesh.userData.robotPart = dataId>=0 && dataId<Number(this.model.ngeom) &&
          Number(this.model.geom_bodyid[dataId])>0;
        mesh.userData.parkHeightfield = isParkHeightfield;
        mesh.castShadow = !isFloor;
        mesh.receiveShadow = isFloor;
        mesh.matrixAutoUpdate = false;
        this.meshes.push(mesh);
        this.scene.add(mesh);
      }
      // Heightfield collision is rendered by the seamless grass mesh above.
      mesh.visible = !mesh.userData.parkHeightfield;
      if (!mesh.userData.floor) mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      mesh.material.opacity = rgba[3];
      mesh.material.transparent = rgba[3] < 0.999;
      mesh.matrix.set(
        mat[0], mat[1], mat[2], pos[0],
        mat[3], mat[4], mat[5], pos[1],
        mat[6], mat[7], mat[8], pos[2],
        0, 0, 0, 1,
      );
      if (mesh.userData.floor) {
        // MuJoCo's contact plane is infinite; slide the *render-only* tile by
        // whole texture repeats. World-space 10-cm squares never move underfoot.
        const o = this.torsoId * 3;
        mesh.matrix.elements[12] += floorTileOrigin(Number(this.data.xpos[o]));
        mesh.matrix.elements[13] += floorTileOrigin(Number(this.data.xpos[o + 1]));
      }
      mesh.matrixWorldNeedsUpdate = true;
      vg.delete();
    }
    for (let i = count; i < this.meshes.length; i++) this.meshes[i].visible = false;
    geoms.delete();

    const root = this.torsoId * 3;
    const x = Number(this.data.xpos[root]);
    const y = Number(this.data.xpos[root + 1]);
    this.keyLight.position.set(x - 1.8, y - 2.4, 3.5);
    this.keyLight.target.position.set(x, y, 0.08);
    if (this.pathTrail.add(x, y)) {
      this.trailLine.geometry.attributes.position.needsUpdate = true;
      this.trailLine.geometry.setDrawRange(0, this.pathTrail.count);
    }

    if (this.follow && this.cameraMode === 'orbit') {
      const o = this.torsoId * 3;
      this.followTarget.set(this.data.xpos[o], this.data.xpos[o+1], Math.max(0.12, this.data.xpos[o+2]-0.04));
      const delta = this.followTarget.clone().sub(this.controls.target).multiplyScalar(0.12);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
  }

  updateChaseCamera(dt = 1 / 60) {
    // Follow the robot's *actual* world heading, not the operator's target.
    // Only heading affects the camera's bearing: body roll and pitch never
    // spin the horizon when EMU steps, leans, or catches a fall.
    const heading = this.actualHeading();
    const first = !this.chaseReady;
    const o = this.torsoId * 3;
    const torso = [Number(this.data.xpos[o]),
                   Number(this.data.xpos[o + 1]),
                   Number(this.data.xpos[o + 2])];
    this.chaseHeading = first ? heading
      : smoothHeading(this.chaseHeading, heading, smoothFactor(dt, 6));
    const terrainAt = (x, y) =>
      this.terrainMode === 'park' && this.park ? this.park.sampleHeight(x, y) : 0;
    const desired = chasePose(torso, this.chaseHeading, terrainAt);
    const position = new THREE.Vector3(...desired.position);
    const target = new THREE.Vector3(...desired.target);
    // Reset/hot-swap/teleport snaps; regular steps glide after the robot.
    const teleport = !first && this.camera.position.distanceToSquared(position) > 4;
    if (first || teleport) {
      this.camera.position.copy(position);
      this.chaseTarget.copy(target);
    } else {
      this.camera.position.lerp(position, smoothFactor(dt, 5));
      this.chaseTarget.lerp(target, smoothFactor(dt, 7));
    }
    this.chaseReady = true;
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(this.chaseTarget);
  }

  updateFirstPersonCamera() {
    const R = this.bodyRotation();
    const torsoRotation = new THREE.Matrix4().set(
      R[0], R[1], R[2], 0,
      R[3], R[4], R[5], 0,
      R[6], R[7], R[8], 0,
      0, 0, 0, 1,
    );
    const torsoQuaternion = new THREE.Quaternion().setFromRotationMatrix(torsoRotation);
    const o = this.torsoId * 3;
    this.camera.position.set(this.data.xpos[o], this.data.xpos[o+1], this.data.xpos[o+2]);
    this.camera.position.add(this.cameraMount.clone().applyQuaternion(torsoQuaternion));
    const raw = torsoQuaternion.multiply(this.cameraBasis);
    if (this.cameraMode === 'raw') {
      this.camera.quaternion.copy(raw);
    } else {
      // Render-only stabilisation: level the image's horizon, then low-pass
      // remaining camera rotation. The IMU, dynamics and NN never see this.
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(raw);
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(this.camera.position.clone().add(forward));
      if (!this.levelPoseValid) this.levelQuaternion.copy(this.camera.quaternion);
      else this.levelQuaternion.slerp(this.camera.quaternion, 0.10);
      this.camera.quaternion.copy(this.levelQuaternion);
      this.levelPoseValid = true;
    }
  }

  updateDepthInspector(nowMs) {
    // Observer only; never feeds the 220/232 policy's neural observations.
    if(!this.depthConfig.enabled || !this.data){
      this.depthLines.visible=false;
      this.depthMountMarker.visible=false;
      this.depthReading=null;
      return;
    }
    const oldCost=this.depthReading?.costMs||0;
    const interval=Math.max(75,Math.min(350,oldCost*3));
    if(this.depthLastMs && nowMs-this.depthLastMs<interval)return;
    this.depthLastMs=nowMs;
    const begin=performance.now();
    const o=this.torsoId*3;
    const torso=Array.from(this.data.xpos.slice(o,o+3));
    const pose=depthProbeRays(torso,this.bodyRotation(),this.depthConfig);
    // Shared viewer geometry (including movable robot parts) matches the
    // displayed MuJoCo scene. The park hfield uses its matching grass mesh.
    const targets=this.meshes.filter(mesh=>mesh.visible);
    if(this.parkSurface)targets.push(this.parkSurface);
    for(const mesh of targets)mesh.updateMatrixWorld(true);
    const raycaster=this.depthRaycaster;
    raycaster.near=.018;
    const origin=new THREE.Vector3(...pose.origin);
    const direction=new THREE.Vector3();
    const n=this.depthConfig.subrays;
    // Auto protects the large hfield mesh from 64 costly triangle casts.
    // Selecting 4 or 5 overrides this and shows the actual cost.
    const sampleCount=n==='auto'?(this.terrainMode==='park'?1:4):n;
    const reading=readDepthProbe(pose,(start,dir,maxRange)=>{
      raycaster.set(origin,direction.set(...dir).normalize());
      raycaster.far=maxRange;
      const hit=raycaster.intersectObjects(targets,false)[0];
      return hit?{distance:hit.distance,kind:hit.object.userData.robotPart?'self':'terrain'}:Infinity;
    },{subrays:sampleCount,noise:this.sensorNoise ? .001 : 0,dropout:0});
    const costMs=performance.now()-begin;
    this.depthReading={...reading,updatedMs:nowMs,subrays:sampleCount,costMs};
    const points=this.depthLinePositions,colors=this.depthLineColors;
    for(let i=0;i<16;i++){
      const end=reading.hitPoints[i];
      const c=reading.selfOccluded[i]?[1,.63,.24]:
        reading.mixed[i]?[1,.87,.40]:
        reading.valid[i]?[.31,.95,.65]:[1,.38,.40];
      for(let k=0;k<3;k++){
        points[i*6+k]=pose.origin[k];
        points[i*6+3+k]=end[k];
        colors[i*6+k]=colors[i*6+3+k]=c[k];
      }
    }
    this.depthLines.geometry.attributes.position.needsUpdate=true;
    this.depthLines.geometry.attributes.color.needsUpdate=true;
    this.depthLines.visible=this.depthConfig.showRays;
    this.depthMountMarker.position.copy(origin);
    this.depthMountMarker.visible=true;
  }

  render(dt = 1 / 60) {
    this.refreshScene();
    this.updateDepthInspector(performance.now());
    if (this.cameraMode === 'orbit') this.controls.update();
    else if (this.cameraMode === 'chase') this.updateChaseCamera(dt);
    else this.updateFirstPersonCamera();
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = Math.max(1, this.canvas.clientWidth || window.innerWidth);
    const h = Math.max(1, this.canvas.clientHeight || window.innerHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
