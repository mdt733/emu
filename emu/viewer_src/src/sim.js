import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { PolicyRunner } from './policy.js';
import { sampleEpisodeCadence } from './cadence.js';

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
    this.scene.background = new THREE.Color(0x0b0f14);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-0.72, -0.78, 0.52);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0.18);
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 4;

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
    const hemi = new THREE.HemisphereLight(0xdde9ff, 0x253040, 1.25);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-1.5, -1.0, 2.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    this.scene.add(key);
  }

  disposeModel() {
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
    this.mjvCamera?.delete();
    this.mjvPerturb?.delete();
    this.mjvOption?.delete();
    this.controls?.dispose();
    this.renderer?.dispose();
  }

  loadModel() {
    this.disposeModel();
    this.model = this.mujoco.MjModel.from_xml_string(this.bundle.robot_xml);
    if (!this.model) throw new Error('MuJoCo could not load robot XML');
    this.data = new this.mujoco.MjData(this.model);
    this.mjvScene = new this.mujoco.MjvScene(this.model, this.maxGeoms);
    this.resolveIds();
    const d = this.bundle.control.default_commands || {};
    this.commandTarget = {
      vx: Number(d.vx ?? 0.04), vy: Number(d.vy ?? 0), yaw: Number(d.yaw ?? 0),
      height: Number(d.height ?? 0), pitch: Number(d.pitch ?? 0), roll: Number(d.roll ?? 0),
      step_height: Number(d.step_height ?? 0.012),
    };
    this.command = { ...this.commandTarget };
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
    this.data.qvel[5] = this.command.yaw;
    mj.mj_forward(this.model, this.data);
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
    this.refreshScene();
  }

  setJoystickTargets(leftX, leftY, rightX, rightY) {
    const lim = this.bundle.control.command_limits;
    const vx = leftY >= 0 ? leftY * lim.vx_max : (-leftY) * lim.vx_min;
    this.commandTarget.vx = vx;
    this.commandTarget.vy = leftX * lim.vy_abs;
    this.commandTarget.yaw = -rightX * lim.yaw_abs;
    this.commandTarget.pitch = rightY * lim.pitch_abs_rad;
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
    this.command.yaw = approach(this.command.yaw, this.commandTarget.yaw, 4.0 * dt);
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
    const pitch = Math.asin(clamp(-R[6], -1, 1));
    return {
      height: this.heightM(), vx: vel[0], vy: vel[1], yaw: ang[2], pitch,
      cmd: { ...this.command }, fallen: this.isFallen(), phaseRate: this.phaseRate(),
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
      g = new THREE.PlaneGeometry(20, 20);
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
        const material = new THREE.MeshStandardMaterial({ roughness: 0.72, metalness: 0.02 });
        mesh = new THREE.Mesh(this.geometryFor(vg), material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.matrixAutoUpdate = false;
        this.meshes.push(mesh);
        this.scene.add(mesh);
      }
      mesh.visible = true;
      mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      mesh.material.opacity = rgba[3];
      mesh.material.transparent = rgba[3] < 0.999;
      mesh.matrix.set(
        mat[0], mat[1], mat[2], pos[0],
        mat[3], mat[4], mat[5], pos[1],
        mat[6], mat[7], mat[8], pos[2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
      vg.delete();
    }
    for (let i = count; i < this.meshes.length; i++) this.meshes[i].visible = false;
    geoms.delete();

    if (this.follow) {
      const o = this.torsoId * 3;
      this.followTarget.set(this.data.xpos[o], this.data.xpos[o+1], Math.max(0.12, this.data.xpos[o+2]-0.04));
      const delta = this.followTarget.clone().sub(this.controls.target).multiplyScalar(0.12);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
  }

  render() {
    this.refreshScene();
    this.controls.update();
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
