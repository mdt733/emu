import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { PolicyRunner } from './policy.js';
import { sampleEpisodeCadence } from './cadence.js';
import { joystickToCommands, nextYawCommand } from './controls.js';
import { measuredAttitude } from './attitude.js';
import { withTerrain, generateTerrainPark, surfaceSlope } from './terrain.js';
import { createGridTexture, createSkyTexture, createGrassTexture, createParkSurface, createTreeCrowns, floorTileOrigin, FLOOR_SIZE_M, GroundTrail } from './appearance.js';

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
    this.grassGridTexture = createGrassTexture(THREE, true);
    this.parkSurface = null;
    this.treeCrowns = null;
    this.park = null;
    this.showGrid = true;
    this.showTrail = true;
    this.showShadows = true;
    this.createWorldGuides();
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-0.72, -0.78, 0.52);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0.18);
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 4;
    this.cameraMode = 'orbit'; // orbit | raw | level
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
    this.parkTreesEnabled = true;
    this.joystickMode = 'split';
    this.headingHold = true;
    this.headingKp = 0.75;
    this.headingTarget = 0;
    this.requestedYaw = 0; // operator/demo rate, independent of heading-hold correction

    this.meshes = [];
    this.geometryCache = new Map();
    this.follow = true;
    this.followTarget = new THREE.Vector3();
    this.paused = false;
    this.autoReset = false;
    this.sensorNoise = false;
    this.fallenFor = 0;

    this.commandTarget = { vx: 0, vy: 0, yaw: 0, height: 0, pitch: 0, roll: 0, step_height: 0.012 };
    this.command = { ...this.commandTarget };
    this.phaseOverride = 0;
    this.episodePhaseHz = 1.025;
    this.phase = 0;
    this.prevAction = new Float32Array(6);
    this.prevRawTarget = new Float32Array(6);
    this.deliveredCtrl = new Float32Array(6);
    this.history = Array.from({ length: 4 }, () => new Float32Array(55));
    this.obs = new Float32Array(220);
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
      if (this.parkSurface) {
        this.parkSurface.material.map = grid ? this.grassGridTexture : this.grassTexture;
        this.parkSurface.material.needsUpdate = true;
      }
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
    this.skyTexture.dispose();
    this.gridTexture.dispose();
    this.grassTexture.dispose();
    this.grassGridTexture.dispose();
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
    if (!['orbit', 'raw', 'level'].includes(mode)) throw new Error(`Unknown camera mode: ${mode}`);
    if (mode === this.cameraMode) return;
    if (this.cameraMode === 'orbit') {
      this.orbitPosition.copy(this.camera.position);
      this.orbitTarget.copy(this.controls.target);
    }
    this.cameraMode = mode;
    this.controls.enabled = mode === 'orbit';
    this.levelPoseValid = false;
    if (mode === 'orbit') {
      this.camera.position.copy(this.orbitPosition);
      this.controls.target.copy(this.orbitTarget);
      this.camera.up.set(0, 0, 1);
      this.controls.update();
    }
  }

  setTerrainMode(mode, maximum=this.terrainMaxDegrees) {
    if (!['flat','ramp','park'].includes(mode)) throw new Error('Unknown terrain');
    const degrees = Number(maximum);
    if (!Number.isFinite(degrees) || degrees < 1 || degrees > 20) throw new Error('Terrain angle must be 1–20°');
    if (mode===this.terrainMode && degrees===this.terrainMaxDegrees) return;
    this.terrainMode=mode;
    this.terrainMaxDegrees=degrees;
    this.loadModel(); // rebuilds only browser MuJoCo contacts
  }
  setTerrainEnabled(enabled) { this.setTerrainMode(enabled ? 'ramp' : 'flat'); }
  setTreesEnabled(enabled) {
    const next=Boolean(enabled);
    if (next===this.parkTreesEnabled) return;
    this.parkTreesEnabled=next;
    if(this.terrainMode==='park')this.loadModel();
  }
  setJoystickMode(mode) {
    if (!['split','hybrid'].includes(mode)) throw new Error('Unknown joystick layout');
    this.joystickMode=mode;
  }
  setHeadingHold(enabled) {
    this.headingHold=Boolean(enabled);
    this.headingTarget=this.actualHeading();
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
    const xml = withTerrain(this.bundle.robot_xml,this.terrainMode,{park:this.park});
    this.model = this.mujoco.MjModel.from_xml_string(xml);
    if (!this.model) throw new Error('MuJoCo could not load robot XML');
    this.data = new this.mujoco.MjData(this.model);
    this.mjvScene = new this.mujoco.MjvScene(this.model, this.maxGeoms);
    this.resolveIds();
    if(this.park){
      this.parkSurface=createParkSurface(THREE,this.park,
        this.showGrid?this.grassGridTexture:this.grassTexture);
      this.scene.add(this.parkSurface);
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
    this.phaseOverride = Number(d.phase_rate_hz ?? 0);
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
    const base = this.buildBaseObs();
    for (let i = 0; i < 4; i++) this.history[i].set(base);
    this.flattenHistory();
    this.controlAccumulator = 0;
    this.fallenFor = 0;
    this.pathTrail.clear();
    this.levelPoseValid = false;
    this.trailLine.geometry.setDrawRange(0, 0);
    this.refreshScene();
  }

  setJoystickTargets(leftX, leftY, rightX, rightY) {
    // One mapping for manual command values and their on-screen labels.
    // Positive screen-right must map to the physically rightward strafe.
    Object.assign(this.commandTarget, joystickToCommands(
      leftX, leftY, rightX, rightY, this.bundle.control.command_limits, this.joystickMode,
    ));
  }

  setDemoCommands({ vx, vy, yaw, pitch }) {
    const l = this.bundle.control.command_limits;
    this.commandTarget.vx = clamp(Number(vx), l.vx_min, l.vx_max);
    this.commandTarget.vy = clamp(Number(vy), -l.vy_abs, l.vy_abs);
    this.commandTarget.yaw = clamp(Number(yaw), -l.yaw_abs, l.yaw_abs);
    this.commandTarget.pitch = clamp(Number(pitch), -l.pitch_abs_rad, l.pitch_abs_rad);
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
      next.policy.obs_dim === old.policy.obs_dim;
  }

  replacePolicy(next) {
    if (!this.canSwapPolicy(next)) throw new Error('Model/control contract changed; reload simulation');
    const newActor = new PolicyRunner(next); // validation before changing state
    this.bundle = next;
    this.policy = newActor;
    this.reset(); // clean comparison, not a fictitious uninterrupted rollout
  }

  setAdvanced({ height, roll, stepHeight, phaseRate }) {
    if (Number.isFinite(height)) this.commandTarget.height = height;
    if (Number.isFinite(roll)) this.commandTarget.roll = roll;
    if (Number.isFinite(stepHeight)) this.commandTarget.step_height = stepHeight;
    if (Number.isFinite(phaseRate)) this.phaseOverride = phaseRate;
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
    if (this.phaseOverride > 0.001) return this.phaseOverride;
    return this.episodePhaseHz;
  }

  stepControl() {
    const cdt = Number(this.bundle.control.control_dt || 0.02);
    this.smoothCommands(cdt);
    const steering = nextYawCommand(this.requestedYaw,this.commandTarget.yaw,
      this.headingTarget,this.actualHeading(),cdt,this.headingHold,this.headingKp,
      Number(this.bundle.control.command_limits.yaw_abs));
    this.requestedYaw=steering.requestedYaw;
    this.headingTarget=steering.headingTarget;
    this.command.yaw=steering.appliedYaw;
    const action = this.policy.infer(this.obs);
    const span = this.bundle.control.target_span_rad;

    // One control-frame delay: deliver last target, remember this action's
    // target for the next control frame. This mirrors environment.py.
    for (let i = 0; i < 6; i++) {
      this.deliveredCtrl[i] = this.prevRawTarget[i];
      this.data.ctrl[i] = this.deliveredCtrl[i];
    }

    const nsub = Number(this.bundle.control.control_substeps || 4);
    for (let k = 0; k < nsub; k++) this.mujoco.mj_step(this.model, this.data);

    for (let i = 0; i < 6; i++) {
      const raw = clamp(action[i] * span[i], -span[i], span[i]);
      this.prevRawTarget[i] = raw;
      this.prevAction[i] = action[i];
    }

    this.phase = (this.phase + cdt * this.phaseRate() * TWO_PI) % TWO_PI;
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
    for (let h = 0; h < 4; h++) this.obs.set(this.history[h], h * 55);
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
    const base = new Float32Array(55);
    let k = 0;

    base[k++] = grav[0] + ns*0.008*n0;
    base[k++] = grav[1] + ns*0.008*n1;
    base[k++] = grav[2] - ns*0.004*n0;
    base[k++] = bodyAng[0] + ns*0.025*n1;
    base[k++] = bodyAng[1] + ns*0.025*n0;
    base[k++] = bodyAng[2] - ns*0.025*n1;

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

    base[k++] = bodyVel[0] + ns*0.018*n0;
    base[k++] = bodyVel[1] + ns*0.018*n1;
    base[k++] = tz + ns*0.0025*n1;
    base[k++] = clamp(1 - 0.18*ns, 0.2, 1.0);

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

    if (k !== 55) throw new Error(`Internal observation builder produced ${k} fields instead of 55`);
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

  render() {
    this.refreshScene();
    if (this.cameraMode === 'orbit') this.controls.update();
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
