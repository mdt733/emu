import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { PolicyRunner } from './policy.js';
import { ArrayFlowObserver, TerrainActorObserver, depthProbeRays, depthGridRays, readDepthProbe, footSphereHit, terrainActorRays } from './sensor.js';
import { chasePose, smoothHeading, smoothFactor, sampleEpisodeCadence, measuredAttitude, sampleShove, expectedShoveRate, clearanceAboveSupport } from './sim-math.js';
import { joystickToCommands, nextYawCommand } from './controls.js';
import { withTerrain, generateTerrainPark, surfaceSlope, surfaceLabGradient, terrainSensorHeightFallback } from './terrain.js';
import { CARPET_DEFAULT, carpetRoleLimits } from './training-carpet.js';
import { terrainRoleEnvelope } from './training-solid-carpet.js';
import { createSensorLabCarpet, carpetSpawnSoleLift, carpetHeightfieldPatch, normalizedMeadowSettings, MEADOW_DEFAULTS, LAB_TILE_M, LAB_TILE_COUNT } from './sensor-lab-carpet.js';
import { createGrowingMeadowScenery } from './growing-meadow.js';
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
    this.sensorLab=Boolean(bundle.sensor_lab);
    this.policy = this.sensorLab ? null : new PolicyRunner(bundle);
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
    this.sensorLabSurface = null;
    this.parkGridOverlay = null;
    this.treeCrowns = null;
    this.park = null;
    this.parkMarkers = null;
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
    this.parkChallengeLevel = 'balanced';
    this.carpetRole = 'step';
    this.carpetDifficulty = {...CARPET_DEFAULT};
    this.meadowSettings = {...MEADOW_DEFAULTS};
    this.carpetTileOverlay = null;
    this.carpetTileCenter = null;
    this.carpetWindowCenter = null;
    this.carpetMocapIds = [];
    this.carpetGeomIds = new Set();
    this.carpetRenderedPrisms = new Set();
    this.carpetScenery = null;
    // Actual browser MuJoCo torso forces, not merely SVG decorations.
    // The real GPU trainer samples independent stochastic pushes; this
    // visibly repeatable spectator demo is NOT a replay of those RNG draws.
    this.shovePreview={enabled:false,forceLimitN:0,intensity:0,mode:'drill',
      stepsLeft:0,last:[0,0],manualPending:false,visualLeft:0};
    this.shoveArrow=new THREE.ArrowHelper(
      new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,0),
      .4,0xffbb66,.12,.075);
    this.shoveArrow.visible=false;
    this.scene.add(this.shoveArrow);
    this.sensorObstacle = 'flat';
    this.surfaceSlopeDegrees=3;
    this.surfaceStepHeightCm=2;
    this.depthConfig = {enabled:true,forward:0,lateral:0,height:-.030,tilt:70,yaw:0,showRays:true,showActorRays:true,showFootprint:true,subrays:'auto'};
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
    // The frozen training policy's rays are drawn independently from the optional
    // VL53L8CX square-FoV probe. This shows the actual serialized contract,
    // including any backwards-pointing rows. No NN inputs are modified.
    this.actorRayPositions=new Float32Array(16*6);
    const actorRayGeometry=new THREE.BufferGeometry();
    actorRayGeometry.setAttribute('position',
      new THREE.BufferAttribute(this.actorRayPositions,3).setUsage(THREE.DynamicDrawUsage));
    this.actorRayLines=new THREE.LineSegments(actorRayGeometry,
      new THREE.LineBasicMaterial({color:0xee63df,transparent:true,opacity:.82,depthTest:true}));
    this.actorRayLines.frustumCulled=false;
    this.actorRayLines.visible=false;
    this.scene.add(this.actorRayLines);
    // Optional 5x5 boundary grid maps the physical 4x4 footprint directly
    // onto the same MuJoCo-rendered ground used by the range inspector.
    const footprintGeometry=new THREE.BufferGeometry();
    this.depthFootprintPositions=new Float32Array(40*6);
    footprintGeometry.setAttribute('position',
      new THREE.BufferAttribute(this.depthFootprintPositions,3).setUsage(THREE.DynamicDrawUsage));
    footprintGeometry.setDrawRange(0,0);
    this.depthFootprint=new THREE.LineSegments(footprintGeometry,
      new THREE.LineBasicMaterial({color:0x66edda,transparent:true,opacity:.90,depthTest:true}));
    this.depthFootprint.frustumCulled=false;
    this.depthFootprint.visible=false;
    this.scene.add(this.depthFootprint);
    this.depthMountMarker = new THREE.Mesh(new THREE.SphereGeometry(.010,10,8),new THREE.MeshBasicMaterial({color:0xffcc80,depthTest:true}));
    this.depthMountMarker.visible=false;
    this.scene.add(this.depthMountMarker);
    this.parkTreesEnabled = true;
    this.parkFeaturesEnabled = true;
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
    // Policy parser has already verified schema-3 array-only semantics.
    // There is no 232 tripod observer or synthetic fallback.
    this.arraySensor=new ArrayFlowObserver(bundle.observation.sensor_contract);
    this.terrainObserver=new TerrainActorObserver(bundle.observation.sensor_contract);
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
    if (this.carpetTileOverlay) this.carpetTileOverlay.visible=grid;
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
    if(this.sensorLabSurface){
      this.scene.remove(this.sensorLabSurface);
      for(const mesh of this.sensorLabSurface.children){
        mesh.geometry.dispose();mesh.material.dispose();
      }
      this.sensorLabSurface=null;
    }
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
    if(this.parkMarkers){
      this.scene.remove(this.parkMarkers);
      for(const item of this.parkMarkers.children){
        item.material.map?.dispose();
        item.material.dispose();
      }
      this.parkMarkers=null;
    }
    this.park = null;
    this.carpetWindowCenter=null;
    this.carpetMocapIds=[];
    this.carpetGeomIds.clear();
    this.carpetRenderedPrisms.clear();
    if(this.carpetScenery){
      this.carpetScenery.dispose();
      this.carpetScenery=null;
    }
    this.shovePreview.stepsLeft=0;
    this.shovePreview.visualLeft=0;
    this.shoveArrow.visible=false;
    if(this.carpetTileOverlay){
      this.scene.remove(this.carpetTileOverlay);
      this.carpetTileOverlay.geometry.dispose();
      this.carpetTileOverlay.material.dispose();
      this.carpetTileOverlay=null;
      this.carpetTileCenter=null;
    }
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
    this.actorRayLines.geometry.dispose();
    this.actorRayLines.material.dispose();
    this.depthFootprint.geometry.dispose();
    this.depthFootprint.material.dispose();
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
    if (!['flat','ramp','park','carpet','sensor'].includes(mode)) throw new Error('Unknown terrain');
    const degrees = Number(maximum);
    if (!Number.isFinite(degrees) || degrees < 1 || degrees > 20) throw new Error('Terrain angle must be 1–20°');
    if (mode===this.terrainMode && degrees===this.terrainMaxDegrees) return;
    this.terrainMode=mode;
    this.terrainMaxDegrees=degrees;
    this.loadModel(); // rebuilds only browser MuJoCo contacts
  }
  setTerrainEnabled(enabled) { this.setTerrainMode(enabled ? 'ramp' : 'flat'); }
  setTrainingCarpet({role=this.carpetRole,limits=this.carpetDifficulty}={}) {
    const checked=carpetRoleLimits(limits,role); // refuse invalid or unknown roles
    const next={...limits,seed:checked.seed,slopeDeg:Number(limits.slopeDeg),
      transitionDeg:Number(limits.transitionDeg),stepUpM:Number(limits.stepUpM),
      stepDownM:Number(limits.stepDownM)};
    if(['seed','slopeDeg','transitionDeg','stepUpM','stepDownM']
      .some(key=>!Number.isFinite(next[key])||next[key]<0))
      throw Error('Invalid training difficulty');
    if(role===this.carpetRole&&JSON.stringify(next)===JSON.stringify(this.carpetDifficulty))return;
    this.carpetRole=role;
    this.carpetDifficulty=next;
    // Training specialist selection changes only the exercises.
    // The physical Meadow has one independent difficulty source.
  }
  setMeadowSettings(settings){
    const next=normalizedMeadowSettings(settings);
    if(JSON.stringify(next)===JSON.stringify(this.meadowSettings))return;
    this.meadowSettings=next;
    // Changing a hill beneath a foot mid-stride creates a false impulse.
    // Rebuild/reset the spectator only after an explicit slider change.
    if(this.terrainMode==='carpet')this.loadModel();
  }
  setShovePreview({enabled=false,forceLimitN=0,intensity=0,mode='drill'}={}){
    const on=Boolean(enabled);
    const limit=Math.max(0,Math.min(10,Number(forceLimitN)||0));
    const strength=Math.max(0,Math.min(1,Number(intensity)||0));
    if(!['training','inspection','drill'].includes(mode))
      throw Error('Invalid shove mode');
    const v=this.shovePreview;
    if(v.enabled===on && v.forceLimitN===limit &&
       v.intensity===strength && v.mode===mode)return;
    v.enabled=on;v.forceLimitN=limit;v.intensity=strength;
    v.mode=mode;
    v.stepsLeft=0;
    v.visualLeft=0;
    v.manualPending=false;
    v.last=[0,0];
    this.shoveArrow.visible=false;
    if(this.data&&this.torsoId>=0){
      const offset=6*this.torsoId;
      this.data.xfrc_applied[offset]=0;
      this.data.xfrc_applied[offset+1]=0;
      this.data.xfrc_applied[offset+2]=0;
    }
  }
  triggerShove(){
    if(!this.data||this.terrainMode!=='carpet'||this.shovePreview.forceLimitN<=0)
      return false;
    this.shovePreview.manualPending=true;
    return true;
  }
  applyViewerShove(dt){
    if(!this.data||this.torsoId<0)return;
    const xfrc=this.data.xfrc_applied;
    const offset=6*this.torsoId;
    xfrc[offset]=0;xfrc[offset+1]=0;xfrc[offset+2]=0;
    const v=this.shovePreview;
    if(this.terrainMode!=='carpet'||v.forceLimitN<=0){
      v.stepsLeft=0;v.visualLeft=0;v.manualPending=false;
      this.shoveArrow.visible=false;return;
    }
    if(v.stepsLeft===0){
      // North's resilience gradient only affects OPTIONAL automatic pushes.
      // Never switch shoves on behind the operator's back, and preserve the
      // manual shove button as an explicit fixed-strength diagnostic.
      const north=this.park?.physics==='streaming-heightfield'
        ?this.park.difficultyAt(Number(this.data.qpos[1])).resilience:1;
      const automatic=v.enabled&&!v.manualPending;
      const push=sampleShove({
        intensity:v.enabled?v.intensity*(.05+.95*north):0,
        mode:v.mode,
        maxForceN:v.forceLimitN*(automatic ? .12+.88*north : 1),
        manual:v.manualPending
      });
      v.manualPending=false;
      if(push){
        v.stepsLeft=push.frames;
        v.last=[push.fx,push.fy];
      }
    }
    if(v.stepsLeft>0){
      xfrc[offset]=v.last[0];
      xfrc[offset+1]=v.last[1];
      v.stepsLeft--;
      v.visualLeft=25;
      const m=Math.hypot(...v.last);
      if(m>1e-7){
        const direction=new THREE.Vector3(v.last[0]/m,v.last[1]/m,0);
        const center=this.torsoId*3;
        const pos=new THREE.Vector3(
          this.data.xpos[center]-direction.x*.36,
          this.data.xpos[center+1]-direction.y*.36,
          this.data.xpos[center+2]+.12);
        this.shoveArrow.position.copy(pos);
        this.shoveArrow.setDirection(direction);
        this.shoveArrow.setLength(.32,.13,.08);
        this.shoveArrow.setColor(0xffbb66);
        this.shoveArrow.visible=true;
      }
    }else if(v.visualLeft>0){
      v.visualLeft--;
      this.shoveArrow.setColor(0x856e57);
      this.shoveArrow.visible=true; // fading trace; no force is applied
    }else this.shoveArrow.visible=false;
  }
  shoveTelemetry(){
    const s=this.shovePreview;
    return {enabled:s.enabled,limit:s.forceLimitN,intensity:s.intensity,
      mode:s.mode,
      // Expected Bernoulli attempt frequency before on-force frames;
      // rare training impulses are per-world, not total across GPU envs.
      expectedHz:s.enabled?expectedShoveRate(s.intensity,s.mode):0,
      active:Boolean(this.data&&this.torsoId>=0&&
        Math.hypot(Number(this.data.xfrc_applied[6*this.torsoId]),
                   Number(this.data.xfrc_applied[6*this.torsoId+1]))>1e-7),
      recent:s.visualLeft>0,force:[...s.last]};
  }

  setSensorObstacle(id) {
    if(!['flat','step-up','step-up-1cm','step-up-2cm','step-down','crest','trough','log','angled',
      'lab-uphill','lab-downhill','lab-camber-left','lab-camber-right',
      'lab-step-up','lab-step-down'].includes(id))throw new Error('Unknown sensor course');
    if(id===this.sensorObstacle)return;
    this.sensorObstacle=id;
    if(this.terrainMode==='sensor')this.loadModel();
  }
  setSurfaceLab({slopeDegrees=this.surfaceSlopeDegrees,
                 stepHeightCm=this.surfaceStepHeightCm}={}) {
    if(!Number.isFinite(slopeDegrees)||slopeDegrees<0.5||slopeDegrees>12)
      throw new Error('Incline must be 0.5–12 degrees');
    if(!Number.isFinite(stepHeightCm)||stepHeightCm<0.1||stepHeightCm>6)
      throw new Error('Step must be 0.1–6 cm');
    if(slopeDegrees===this.surfaceSlopeDegrees &&
       stepHeightCm===this.surfaceStepHeightCm)return;
    this.surfaceSlopeDegrees=slopeDegrees;
    this.surfaceStepHeightCm=stepHeightCm;
    if(this.terrainMode==='sensor')this.loadModel();
  }
  setDepthInspector({enabled,forward,lateral=0,tilt,yaw=0,height=-.040,showRays,showActorRays=true,showFootprint=true,subrays='auto'}) {
    const values={enabled:Boolean(enabled),forward:Number(forward),lateral:Number(lateral),height:Number(height),tilt:Number(tilt),yaw:Number(yaw),
      showRays:Boolean(showRays),showActorRays:Boolean(showActorRays),showFootprint:Boolean(showFootprint),subrays};
    if(!['auto',1,4,5].includes(subrays))throw new Error('Depth samples must be auto, 1, 4 or 5');
    // Validation lives with one optical mounting model, not a policy contract.
    if(!Number.isFinite(values.height)||values.height<-.12||values.height>.10||
       !Number.isFinite(values.lateral)||Math.abs(values.lateral)>.12||
       !Number.isFinite(values.yaw)||Math.abs(values.yaw)>30||
       values.forward<0||values.forward>.2||values.tilt<0||values.tilt>90)
      throw new Error('Invalid depth mount');
    this.depthConfig=values;
    this.depthLastMs=0;
  }
  setParkChallengeLevel(level) {
    if(!['gentle','balanced','stretch'].includes(level))
      throw new Error('Unknown park challenge level');
    if(level===this.parkChallengeLevel)return;
    this.parkChallengeLevel=level;
    // Only browser collision geometry changes. Never touch policy weights.
    if(this.terrainMode==='park')this.loadModel();
  }
  setParkFeaturesEnabled(enabled) {
    const next=Boolean(enabled);
    if(next===this.parkFeaturesEnabled)return;
    this.parkFeaturesEnabled=next;
    if(this.terrainMode==='park')this.loadModel();
  }
  setTreesEnabled(enabled) {
    const next=Boolean(enabled);
    if (next===this.parkTreesEnabled) return;
    this.parkTreesEnabled=next;
    if(['park','carpet'].includes(this.terrainMode))this.loadModel();
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
      ? generateTerrainPark({maxDegrees:this.terrainMaxDegrees,
                             trees:this.parkTreesEnabled,features:this.parkFeaturesEnabled,
                             challengeLevel:this.parkChallengeLevel})
      : this.terrainMode==='carpet'
        ? createSensorLabCarpet(this.meadowSettings)
        : null;
    this.scene.fog=this.park?.physics==='streaming-heightfield'
      ?new THREE.Fog(0xb1c8c1,7,22):new THREE.Fog(0xbad7e2,15,49);
    const xml = withTerrain(this.bundle.robot_xml,this.terrainMode,{park:this.park,obstacle:this.sensorObstacle,
      slopeDegrees:this.surfaceSlopeDegrees,stepHeightCm:this.surfaceStepHeightCm});
    this.model = this.mujoco.MjModel.from_xml_string(xml);
    if (!this.model) throw new Error('MuJoCo could not load robot XML');
    this.data = new this.mujoco.MjData(this.model);
    this.mjvScene = new this.mujoco.MjvScene(this.model, this.maxGeoms);
    this.resolveIds();
    if(this.park?.physics==='streaming-heightfield'){
      // Real CPU collision hfields and render triangles share the SAME four
      // corner samples. Outer 2.5m scenery is deliberately visual-only.
      this.makeCarpetTileOverlay();
      this.makeSensorLabSurface();
      this.carpetScenery=createGrowingMeadowScenery(THREE,this.scene,this.park);
    }else if(this.park){
      this.parkSurface=createParkSurface(THREE,this.park,this.grassTexture);
      this.scene.add(this.parkSurface);
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
      this.createParkMarkers();
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

  makeSensorLabSurface(){
    // Mesh triangles are reconstructed only for the spectator renderer.
    // Physical contacts are from the same mutable MuJoCo 2x2 heightfields.
    const group=new THREE.Group();
    this.sensorLabSurface=group;
    const material=new THREE.MeshStandardMaterial({
      map:this.grassTexture,color:0xffffff,vertexColors:true,
      roughness:.98,side:THREE.DoubleSide,flatShading:true});
    for(const tile of this.park.tiles){
      const mesh=new THREE.Mesh(new THREE.BufferGeometry(),material);
      mesh.userData.tileSlot=tile.slot;
      mesh.userData.sensorLabPrism=true; // hide duplicate MuJoCo view geom
      mesh.receiveShadow=true;
      group.add(mesh);
    }
    this.carpetSurfaceMaterial=material;
    this.scene.add(group);
    this.syncGrowingCarpetVisuals();
  }

  updateTerrainTileVisual(tile){
    if(!this.sensorLabSurface)return;
    const mesh=this.sensorLabSurface.children[tile.slot];
    const x=tile.i,y=tile.j,[a,b,c,d]=tile.corners,down=.5;
    const v=[[x,y,a],[x+1,y,b],[x,y+1,c],[x+1,y+1,d]];
    const faces=[[0,1,2],[3,2,1]];
    const points=[],uv=[],colors=[];
    function add(tri,earth=false){
      for(const xyz of tri){
        points.push(...xyz);uv.push(xyz[0],xyz[1]);
        colors.push(...(earth?[.48,.42,.31]:[.96,1,.91]));
      }
    }
    for(const face of faces)add(face.map(i=>v[i]));
    for(const [q,r] of [[0,1],[1,3],[3,2],[2,0]]){
      const p=v[q],w=v[r],p0=[p[0],p[1],p[2]-down],
            w0=[w[0],w[1],w[2]-down];
      add([p,w,p0],true);add([w,w0,p0],true);
    }
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
    g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));
    g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    g.computeVertexNormals();
    mesh.geometry.dispose();mesh.geometry=g;
  }

  syncGrowingCarpetVisuals(changed=this.park?.tiles||[]){
    if(!this.sensorLabSurface||!this.park)return;
    for(const tile of changed)this.updateTerrainTileVisual(tile);
    this.carpetScenery?.sync();
    this.carpetTileCenter=null;
  }

  applyCarpetTilePhysics(changed){
    // MuJoCo explicitly permits modifying hfield_data in a compiled
    // model. Mesh vertex edits would be unsafe for collision BVHs.
    const samples=this.model.hfield_data,pos=this.data.mocap_pos;
    for(const tile of changed){
      const off=Number(this.model.hfield_adr[tile.slot]);
      const data=carpetHeightfieldPatch(tile);
      for(let k=0;k<4;k++)samples[off+k]=data[k];
      const id=this.carpetMocapIds[tile.slot];
      pos[3*id]=tile.x;pos[3*id+1]=tile.y;pos[3*id+2]=tile.base;
    }
    this.mujoco.mj_forward(this.model,this.data);
  }

  growCarpetNearRobot(){
    if(this.park?.physics!=='streaming-heightfield'||!this.data)return;
    const feet=this.footGeomIds.map(id=>({
      x:Number(this.data.geom_xpos[3*id]),
      y:Number(this.data.geom_xpos[3*id+1]),
      radius:Number(this.model.geom_size[3*id])
    }));
    const changes=this.park.growAround(Number(this.data.qpos[0]),
                                      Number(this.data.qpos[1]),feet);
    if(!changes.length)return;
    this.applyCarpetTilePhysics(changes);
    this.syncGrowingCarpetVisuals(changes);
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
    this.footGeomIds = ['r_foot_ball','l_foot_ball'].map(n =>
      this.name2id(mj.mjtObj.mjOBJ_GEOM,n));
    if(this.park?.physics==='streaming-heightfield'){
      this.carpetMocapIds=Array.from({length:LAB_TILE_COUNT},(_,i)=>{
        const body=this.name2id(mj.mjtObj.mjOBJ_BODY,'sensor_lab_tile_'+i);
        const id=Number(this.model.body_mocapid[body]);
        if(id<0)throw Error('Streaming Carpet requires 225 mocap bodies');
        return id;
      });
      if(new Set(this.carpetMocapIds).size!==LAB_TILE_COUNT)
        throw Error('Each streaming hfield needs a distinct mocap');
      this.carpetGeomIds=new Set(Array.from({length:LAB_TILE_COUNT},(_,i)=>
        this.name2id(mj.mjtObj.mjOBJ_GEOM,'sensor_lab_ground_'+i)));
    }

  }

  createParkMarkers() {
    if(!this.park?.features.length)return;
    const group=new THREE.Group();
    this.park.features.forEach((feature,i)=>{
      // 2D signs are NON-COLLIDING presentation only, never an actor input.
      const canvas=document.createElement('canvas');
      canvas.width=384;canvas.height=80;
      const ctx=canvas.getContext('2d');
      if(!ctx)return;
      ctx.fillStyle='rgba(12,29,34,0.86)';
      ctx.beginPath();ctx.roundRect(4,5,376,70,15);ctx.fill();
      ctx.strokeStyle='rgba(110,231,206,0.78)';
      ctx.lineWidth=3;ctx.stroke();
      ctx.fillStyle='#e9fff7';
      ctx.font='600 28px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText((i+1)+'. '+feature.label,192,40,360);
      const tex=new THREE.CanvasTexture(canvas);
      tex.colorSpace=THREE.SRGBColorSpace;
      const mat=new THREE.SpriteMaterial({map:tex,transparent:true,
        depthTest:true,depthWrite:false});
      const sign=new THREE.Sprite(mat);
      sign.position.set(feature.x,feature.y,
        this.park.sampleHeight(feature.x,feature.y)+0.20);
      sign.scale.set(.90,.19,1);
      group.add(sign);
    });
    this.parkMarkers=group;
    this.scene.add(group);
  }

  reset() {
    this.resetGeneration=(this.resetGeneration||0)+1;
    const mj = this.mujoco;
    mj.mj_resetData(this.model, this.data);
    if(this.park?.physics==='streaming-heightfield'){
      const restored=this.park.resetGrowth();
      this.applyCarpetTilePhysics(restored);
      this.syncGrowingCarpetVisuals(restored);
    }
    this.data.ctrl.fill(0);
    this.data.qvel.fill(0);
    // Training births already moving near the commanded state. Do the same for
    // the interactive viewer so reset is not a special stationary task.
    this.data.qvel[0] = this.sensorLab ? 0 : this.command.vx;
    this.data.qvel[1] = this.sensorLab ? 0 : this.command.vy;
    this.requestedYaw = this.commandTarget.yaw;
    this.command.yaw = this.requestedYaw;
    this.data.qvel[5] = this.command.yaw;
    if(this.park?.physics==='streaming-heightfield'){
      this.data.qpos[0]=LAB_TILE_M*.5;
      this.data.qpos[1]=LAB_TILE_M*.5;
      this.carpetWindowCenter=[0,0];
    }
    mj.mj_forward(this.model, this.data);
    if(this.park?.physics==='streaming-heightfield'){
      // qpos0 is neutral on the original z=0 floor, but this tile may be
      // raised by a slope or a signed step. Align actual 18mm foot spheres
      // against the true CPU MuJoCo prism surface BEFORE starting inference
      // and sensor histories. Never alter the policy or terrain geometry.
      this.data.qpos[2] += carpetSpawnSoleLift(
        this.park,this.data.geom_xpos,this.model.geom_size,this.footGeomIds);
      mj.mj_forward(this.model,this.data);
    }
    this.headingTarget=this.actualHeading();
    // Training samples cadence at birth, then holds it through command changes.
    this.episodePhaseHz = sampleEpisodeCadence(this.command, this.bundle.control);
    this.phase = 0;
    this.prevAction.fill(0);
    this.prevRawTarget.fill(0);
    this.deliveredCtrl.fill(0);
    this.terrainObserver.reset(
      this.terrainActorPose(),this.model,this.data,this.footGeomIds,this.sensorNoise);
    this.arraySensor.reset(
      this.sensorPose(),this.model,this.data,this.sensorNoise);
    const base = this.buildBaseObs();
    for (let i = 0; i < 4; i++) this.history[i].set(base);
    this.flattenHistory();
    this.controlAccumulator = 0;
    this.fallenFor = 0;
    this.shovePreview.stepsLeft=0;
    this.shovePreview.visualLeft=0;
    this.shovePreview.manualPending=false;
    this.shoveArrow.visible=false;
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
    if(!['park','carpet'].includes(this.terrainMode)||!this.park)return [];
    // The existing pose-lattice planner assumes a 62cm circular clearance
    // around each supplied obstacle. Feature waypoints make autonomous
    // navigation avoid untrained steps/logs, but manual joystick can inspect.
    return [
      ...(this.parkTreesEnabled?this.park.trees:[]),
      ...(this.terrainMode==='park'&&this.parkFeaturesEnabled ? this.park.features.flatMap(f=>{
        // Planner is intentionally conservative. Include the *physical*
        // obstacle footprint, not just one centre point; human joystick
        // driving remains unrestricted for actual challenge attempts.
        const spread=Math.max(0,f.radius-.34);
        return [{x:f.x,y:f.y},{x:f.x+spread,y:f.y},
          {x:f.x-spread,y:f.y},{x:f.x,y:f.y+spread},
          {x:f.x,y:f.y-spread}];
      }) : []),
    ];
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
    return ['park','carpet'].includes(this.terrainMode) && this.park ? this.park.sampleHeight(x,y) : 0;
  }
  makeCarpetTileOverlay(){
    this.carpetTileOverlay=new THREE.LineSegments(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color:0xf2d59a,transparent:true,
        opacity:.74,depthTest:true,depthWrite:false}));
    this.carpetTileOverlay.renderOrder=4;
    this.carpetTileOverlay.frustumCulled=false;
    this.scene.add(this.carpetTileOverlay);
    this.carpetTileCenter=null;
  }
  updateCarpetTileOverlay(){
    if(!this.carpetTileOverlay||!this.park||!this.data)return;
    const key=this.park.revision;
    if(key===this.carpetTileCenter)return;
    this.carpetTileCenter=key;
    const size=this.park.tileSizeM;
    const positions=[],add=(a,b)=>positions.push(...a,...b);
    for(const t of this.park.window()){
      const x=t.i*size,y=t.j*size,[a,b,c,d]=t.corners;
      const ll=[x,y,a+.008],lr=[x+size,y,b+.008],
        ul=[x,y+size,c+.008],ur=[x+size,y+size,d+.008];
      add(ll,lr);add(lr,ur);add(ur,ul);add(ul,ll);
      add(lr,ul); // same real two-prism diagonal, not a fake square.
    }
    this.carpetTileOverlay.geometry.dispose();
    this.carpetTileOverlay.geometry=new THREE.BufferGeometry();
    this.carpetTileOverlay.geometry.setAttribute('position',
      new THREE.Float32BufferAttribute(positions,3));
    this.carpetTileOverlay.visible=this.showGrid;
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
    const action = this.sensorLab ? new Float32Array(6) : this.policy.infer(this.obs);
    this.controlFrames++;
    const span = this.bundle.control.target_span_rad;

    // One control-frame delay: deliver last target, remember this action's
    // target for the next control frame. This mirrors environment.py.
    for (let i = 0; i < 6; i++) {
      this.deliveredCtrl[i] = this.prevRawTarget[i];
      this.data.ctrl[i] = this.deliveredCtrl[i];
    }

    // One control-frame force is held across all MuJoCo substeps, just
    // like the real trainer's xfrc_applied on its torso body.
    this.applyViewerShove(cdt);
    const nsub = Number(this.bundle.control.control_substeps || 4);
    const beforeJoints=this.actQpos.map(i=>Number(this.data.qpos[i]));
    for (let k = 0; k < nsub; k++) this.mujoco.mj_step(this.model, this.data);
    this.growCarpetNearRobot();
    for(let i=0;i<this.actQpos.length;i++)
      this.motorTravelRad+=Math.abs(Number(this.data.qpos[this.actQpos[i]])-beforeJoints[i]);

    for (let i = 0; i < 6; i++) {
      const raw = clamp(action[i] * span[i], -span[i], span[i]);
      this.prevRawTarget[i] = raw;
      this.prevAction[i] = action[i];
    }

    this.phase = (this.phase + cdt * this.phaseRate() * TWO_PI) % TWO_PI;
    this.terrainObserver.advance(
      this.terrainActorPose(),this.model,this.data,this.footGeomIds,this.sensorNoise);
    this.arraySensor.advance(
      this.sensorPose(),this.model,this.data,this.sensorNoise);
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
      if(['park','carpet'].includes(this.terrainMode)&&this.park) return this.park.sampleHeight(x,y);
      if(this.terrainMode==='ramp'&&Math.abs(y)<.44){
        if(x>=.85&&x<=1.65)return .008+(x-.85)*Math.tan(Math.PI/180);
        if(x>1.65&&x<2.17)return .015;
        if(x>=2.17&&x<=2.97)return .015-(x-2.17)*Math.tan(Math.PI/180);
      }
      return 0;
    };
    // Neither the park heightfield nor the Carpet's eight convex meshes is
    // a supported plane/box in our pure 360-observation raycaster. Supply
    // the ACTUAL world surface to optical flow; never fake z=0 ground.
    const terrainHeightAt=terrainSensorHeightFallback(this.terrainMode,this.park);
    return {R,torso,vbody,gyro,heightAt,terrainHeightAt};
  }

  terrainActorPose() {
    // Same torso-origin physical rays as MJX. On the park hfield, sample its
    // shared height surface rather than pretending it is a flat plane.
    const R=this.bodyRotation(),offset=this.torsoId*3;
    return {R,torso:Array.from(this.data.xpos.slice(offset,offset+3)),
      // Convex mesh geoms (Carpet) are not recognized by the fast JS
      // raycaster. The exact CPU prism-height surface must supply the
      // physical ToF return, just as the park heightfield already does.
      terrainHeightAt:terrainSensorHeightFallback(this.terrainMode,this.park)};
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
    const noisyGravity=[
      grav[0]+ns*.008*n0,grav[1]+ns*.008*n1,grav[2]-ns*.004*n0
    ];
    const sensor=this.arraySensor.read(
      noisyGravity,this.terrainObserver.read());
    base[3]=sensor.gyro[0];
    base[4]=sensor.gyro[1];
    base[5]=sensor.gyro[2];

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

    base[k++] = sensor.velocity[0];
    base[k++] = sensor.velocity[1];
    base[k++] = sensor.depth;
    base[k++] = sensor.velocityConfidence;

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

    base[k++] = sensor.forwardGrade;
    base[k++] = sensor.sideGrade;
    base[k++] = sensor.groundConfidence;
    const grid=this.terrainObserver.read();
    for(const range of grid.ranges) base[k++]=range;
    for(const valid of grid.valid) base[k++]=valid?1:0;
    if (k !== this.baseObsDim) throw new Error(`Internal observation builder produced ${k} fields instead of ${this.baseObsDim}`);
    return base;
  }

  isFallen() {
    const R = this.bodyRotation();
    const gravityZ = -this.worldToBody(R, 0, 0, -1)[2];
    // On a downward/sideways diagnostic plane, absolute world-Z falls
    // as the robot walks. Falling means body clearance ABOVE SUPPORT is
    // too low; do not misclassify successful downhill walking as a fall.
    const grade=this.terrainMode==='sensor' ?
      surfaceLabGradient(this.sensorObstacle,this.surfaceSlopeDegrees) : null;
    const root=this.data.qpos;
    const supportZ=grade ? grade.gx*root[0]+grade.gy*root[1] :
      ['park','carpet'].includes(this.terrainMode)&&this.park
        ? this.park.sampleHeight(root[0],root[1]) : 0;
    return clearanceAboveSupport(this.heightM(),supportZ) < 0.145 ||
      gravityZ < 0.35;
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
        (feet[0][0]+feet[1][0])/2, (feet[0][1]+feet[1][1])/2, this.actualHeading(),
        {obstacle:this.sensorObstacle,slopeDegrees:this.surfaceSlopeDegrees}),
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
        // Two-sided raycasting is essential when the configurable optical
        // origin sits INSIDE the torso at 0mm, and for floor/support normals.
        // Otherwise Three.js ignores the exit face and sees through the body.
        const material = isFloor
          ? new THREE.MeshStandardMaterial({map: this.showGrid ? this.gridTexture : null, roughness: 0.98, metalness: 0, color: this.showGrid ? 0xffffff : 0x40515e, side: THREE.DoubleSide})
          : new THREE.MeshStandardMaterial({ roughness: 0.68, metalness: 0.06, side: THREE.DoubleSide });
        mesh = new THREE.Mesh(this.geometryFor(vg), material);
        mesh.userData.floor = isFloor;
        // mjvGeom.objid is the actual MuJoCo geom ID. The unrelated
        // dataid identifies a mesh/heightfield asset, NOT a physical geom.
        // Recomputed for each rendered geom below, as display order may vary.
        mesh.userData.parkHeightfield = isParkHeightfield;
        mesh.castShadow = !isFloor;
        mesh.receiveShadow = isFloor;
        mesh.matrixAutoUpdate = false;
        this.meshes.push(mesh);
        this.scene.add(mesh);
      }
      // mjv_updateScene may reorder geoms. Only MuJoCo GEOM objects whose
      // parent body is non-world are visible self-occluders (legs/feet).
      // A contact-visualization or world terrain mesh is not robot geometry.
      const geomId = Number(vg.objid);
      mesh.userData.footGeom = this.footGeomIds.includes(geomId);
      mesh.userData.sensorLabPrism = this.carpetGeomIds.has(geomId);
      mesh.userData.robotPart = Number(vg.objtype) === Number(mj.mjtObj.mjOBJ_GEOM.value)
        && Number.isInteger(geomId) && geomId >= 0
        && geomId < Number(this.model.ngeom)
        && Number(this.model.geom_bodyid[geomId]) > 0
        && !this.carpetGeomIds.has(geomId); // mocap tiles are TERRAIN, not self-occlusion
      // Heightfield collision is rendered by the seamless grass mesh above.
      // MuJoCo WASM returns a visual record for each mesh geom, but the
      // JS geometryFor fallback cannot reconstruct its triangles. The
      // sensor-lab's eight exact authored triangles are drawn separately.
      mesh.visible = !mesh.userData.parkHeightfield &&
        !mesh.userData.sensorLabPrism;
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
      ['park','carpet'].includes(this.terrainMode) && this.park ? this.park.sampleHeight(x, y) : 0;
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
    // Optional visual inspector is independent of actor measurements; schema-9
    // 4x4 actor rays use their own truthful, timed physical observation ring.
    if(!this.depthConfig.enabled || !this.data){
      this.depthLines.visible=false;
      this.actorRayLines.visible=false;
      this.depthFootprint.visible=false;
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
    const rotation=this.bodyRotation();
    const pose=depthProbeRays(torso,rotation,this.depthConfig);
    // Shared viewer geometry (including movable robot parts) matches the
    // displayed MuJoCo scene. The park hfield uses its matching grass mesh.
    const targets=this.meshes.filter(mesh=>mesh.visible &&
      !mesh.userData.sensorLabPrism);
    if(this.parkSurface)targets.push(this.parkSurface);
    if(this.sensorLabSurface)targets.push(...this.sensorLabSurface.children);
    for(const mesh of targets)mesh.updateMatrixWorld(true);
    const raycaster=this.depthRaycaster;
    raycaster.near=.018;
    const origin=new THREE.Vector3(...pose.origin);
    const direction=new THREE.Vector3();
    const n=this.depthConfig.subrays;
    const footCentres=this.footGeomIds.map(id=>({
      centre:Array.from(this.data.geom_xpos.slice(3*id,3*id+3)),
      radius:Number(this.model.geom_size[3*id])}));
    // Auto protects the large hfield mesh from 64 costly triangle casts.
    // Selecting 4 or 5 overrides this and shows the actual cost.
    const sampleCount=n==='auto'?(['park','carpet'].includes(this.terrainMode)?1:4):n;
    const reading=readDepthProbe(pose,(start,dir,maxRange)=>{
      raycaster.set(origin,direction.set(...dir).normalize());
      raycaster.far=maxRange;
      const hit=raycaster.intersectObjects(targets,false)[0];
      let foot=Infinity;
      for(const part of footCentres)
        foot=Math.min(foot,footSphereHit(start,dir,part.centre,part.radius));
      if(foot<Math.min(hit?.distance??Infinity,maxRange)+1e-4)
        return {distance:foot,kind:'foot',surface:'physical 36mm foot sphere'};
      return hit?{distance:hit.distance,
        kind:hit.object.userData.footGeom?'foot':
          hit.object.userData.robotPart?'self':'terrain',
        surface:hit.object.uuid}:Infinity;
    },{subrays:sampleCount,noise:this.sensorNoise ? .001 : 0,dropout:0});
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
    if(this.depthConfig.showActorRays){
      const contract=this.bundle.observation.sensor_contract;
      const actorRays=terrainActorRays(rotation,contract.tof4x4_rays_body);
      const positions=this.actorRayPositions;
      const bodyMount=contract.tof4x4_mount_body_m;
      const actorOrigin=new THREE.Vector3(
        ...torso.map((v,i)=>v+
          rotation[3*i]*bodyMount[0]+rotation[3*i+1]*bodyMount[1]+
          rotation[3*i+2]*bodyMount[2]));
      for(let i=0;i<actorRays.length;i++){
        raycaster.set(actorOrigin,direction.set(...actorRays[i]).normalize());
        raycaster.far=1.5;
        const hit=raycaster.intersectObjects(targets,false)[0];
        const endpoint=hit?.point || actorOrigin.clone().addScaledVector(direction,.65);
        for(let k=0;k<3;k++){
          positions[i*6+k]=actorOrigin.getComponent(k);
          positions[i*6+3+k]=endpoint.getComponent(k);
        }
      }
      this.actorRayLines.geometry.attributes.position.needsUpdate=true;
      this.actorRayLines.visible=true;
    }else this.actorRayLines.visible=false;
    if(this.depthConfig.showFootprint){
      // Footprint boundary casts are purely for visualisation and never
      // become privileged actor observations. They stop at the first
      // physical surface; missing, body-blocked edges remain open.
      const rays=depthGridRays(rotation,this.depthConfig);
      const hits=rays.map(ray=>{
        raycaster.set(origin,direction.set(...ray).normalize());
        raycaster.far=1.5;
        const hit=raycaster.intersectObjects(targets,false)[0];
        return hit && !hit.object.userData.robotPart?hit.point:null;
      });
      let written=0;
      const addEdge=(a,b)=>{
        if(!hits[a]||!hits[b])return;
        for(const p of [hits[a],hits[b]]){
          this.depthFootprintPositions[written++]=p.x;
          this.depthFootprintPositions[written++]=p.y;
          this.depthFootprintPositions[written++]=p.z+.004;
        }
      };
      for(let row=0;row<5;row++)for(let col=0;col<5;col++){
        const i=row*5+col;
        if(col<4)addEdge(i,i+1);
        if(row<4)addEdge(i,i+5);
      }
      this.depthFootprint.geometry.attributes.position.needsUpdate=true;
      this.depthFootprint.geometry.setDrawRange(0,written/3);
      this.depthFootprint.visible=written>0;
    }else this.depthFootprint.visible=false;
    const bodyForward=[rotation[0],rotation[3],rotation[6]];
    const forwardHits=reading.hitPoints.map((p,i)=>
      reading.valid[i]&&!reading.selfOccluded[i]
        ?p.reduce((sum,n,k)=>sum+(n-pose.origin[k])*bodyForward[k],0):NaN)
      .filter(Number.isFinite);
    this.depthReading={...reading,updatedMs:nowMs,subrays:sampleCount,
      reachNearM:forwardHits.length?Math.min(...forwardHits):null,
      reachFarM:forwardHits.length?Math.max(...forwardHits):null,
      footprintRays:this.depthConfig.showFootprint?25:0,
      costMs:performance.now()-begin};
    this.depthMountMarker.position.copy(origin);
    this.depthMountMarker.visible=true;
  }

  render(dt = 1 / 60) {
    this.refreshScene();
    this.updateCarpetTileOverlay();
    this.updateDepthInspector(performance.now());
    if (this.cameraMode === 'orbit') this.controls.update();
    else if (this.cameraMode === 'chase') this.updateChaseCamera(dt);
    else this.updateFirstPersonCamera();
    this.carpetScenery?.update(this.camera.position);
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
