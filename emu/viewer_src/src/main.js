import './style.css';
import { parsePolicy } from './policy.js';
import { VirtualStick } from './joystick.js';
import { joystickToCommands } from './controls.js';
import { EmuBrowserSim } from './sim.js';
import { DemoCommander } from './live-demo.js';
import { GoalNavigator } from './goal-navigation.js';
import { GoalPlannerWorker } from './goal-planner-client.js';
import { showcasePolicyUrl, isJsonResponse, isViewerConfig, readShowcaseManifest, assertShowcasePolicyResponse } from './showcase.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const statusEl = $('status');
const policyFile = $('policyFile');
const policySelect = $('policySelect');
const pauseBtn = $('pauseBtn');
const resetBtn = $('resetBtn');
const followToggle = $('followToggle');
const autoResetToggle = $('autoResetToggle');
const noiseToggle = $('noiseToggle');
const gridToggle = $('gridToggle');
const trailToggle = $('trailToggle');
const shadowToggle = $('shadowToggle');
const terrainMode = $('terrainMode');
const terrainMax = $('terrainMax');
const treesToggle = $('treesToggle');
const joystickMode = $('joystickMode');
const headingHoldToggle = $('headingHoldToggle');
const headingGain = $('headingGain');
const pitchRange = $('pitchRange');
const cameraBtn = $('cameraBtn');
const cameraModes = ['orbit', 'chase', 'raw', 'level'];
let cameraMode = 'orbit';
function cameraLabel(mode) {
  return {orbit:'View: orbit', chase:'View: chase',
          raw:'View: POV', level:'View: level POV'}[mode];
}
function cameraDescription(mode) {
  return {orbit:'orbit camera', chase:'third-person chase camera behind and above robot',
          raw:'raw chassis POV', level:'visually levelled POV'}[mode];
}
const pitchNotice = $('pitchNotice');
const dropOverlay = $('dropOverlay');
const leftValues = $('leftValues');
const rightValues = $('rightValues');
const livePanel = $('livePanel');
const demo = new DemoCommander();
const navigator = new GoalNavigator();
let navWorker;
let workerUnavailable=null;
try{navWorker=new GoalPlannerWorker();}
catch(error){workerUnavailable=String(error?.message||error);console.error('Goal worker:',error);}
navigator.useBackgroundPlanner(request=>{
  if(!navWorker)throw new Error(workerUnavailable||'Background navigation unavailable');
  return navWorker.request(request);
});
const demoMode = $('demoMode'), navHud = $('navHud');
const navPlanner = $('navPlanner');
let lastResetGeneration = -1;
let navCommandClock = Infinity;
let navCommandUpdates = 0;
let navLastCommand = {vx:0,vy:0,yaw:0};
let demoActive = false;
const goalDemoActive = () => demoActive && demoMode.value === 'goals';
function syncGoalMode() {
  const enabled=Boolean(sim && goalDemoActive());
  navigator.active=enabled;
  navigator.plannerMode=navPlanner.value;
  navigator.reset();
  navCommandClock=Infinity;
  navCommandUpdates=0;
  navLastCommand={vx:0,vy:0,yaw:0};
  if(sim) {
    // Only the presentation changes between demo modes. Both demos must
    // share the EXISTING neural actor command and control pipeline.
    sim.setNavigationPresentation(enabled);
    sim.autoReset=autoResetToggle.checked||demoActive;
  }
  navHud.hidden=!enabled;
  $('navNext').disabled=!enabled;
}
// One proven command delivery path for every autonomous input source.
function applyAutonomousCommand(command) {
  sim.setDemoCommands(command);
}
function navigationUpdate(dt) {
  const pose=sim.navigationPose();
  if(sim.resetGeneration!==lastResetGeneration) {
    lastResetGeneration=sim.resetGeneration;
    navigator.reset();
  }
  let plannedCommand=null;
  navigator.tick(dt,pose,bundle.control.command_limits,
    sim.navigationTrees(),command=>{plannedCommand=command;});
  // Both modes still use the same tested setDemoCommands() pipeline.
  // Higher cruising speed needs tighter steering corrections than the
  // initial 650ms zero-order hold (which travelled ~0.6m at full speed).
  // Deliver the latest planned velocity at 4Hz in cruise and 5Hz near
  // the final pose; PPO inference continues at its normal 50Hz.
  const remaining=navigator.goal?Math.hypot(
    pose.x-navigator.goal.x,pose.y-navigator.goal.y):Infinity;
  const commandPeriod=remaining>1.0?.25:.20;
  navCommandClock+=Math.min(Math.max(dt,0),.1);
  if(plannedCommand && navCommandClock>=commandPeriod){
    applyAutonomousCommand(plannedCommand);
    navLastCommand={...plannedCommand};
    navCommandClock=0;
    navCommandUpdates++;
  }
  const state=navigator.status(pose);
  sim.setNavigationOverlay(state.goal,state.route);
  $('demoKind').textContent=state.reason;
  if(!state.goal){
    $('navHudTitle').textContent=state.reason;
    $('navHudInfo').textContent='Walking while the planner finds a route';
    $('navEstimate').textContent=state.plannerError
      ? 'Planner error: '+state.plannerError
      : `Actor frames ${sim.controlFrames||0} · joints ${sim.motorTravelRad.toFixed(3)} rad · cmds ${navCommandUpdates}`;
    return;
  }
  const degrees=n=>Math.round(n*180/Math.PI);
  $('navHudTitle').textContent=`TARGET ${state.arrivals+1} · ${state.reason}`;
  $('navHudInfo').textContent=`${state.distance.toFixed(2)} m · heading ${degrees(state.headingError)}° · ${state.arrivals} parked${state.failures?' · '+state.failures+' timed out':''}`;
  $('navHudInfo').textContent+=` · vx ${navLastCommand.vx.toFixed(2)} commanded / ${pose.vx.toFixed(2)} observed`;
  $('navEstimate').textContent=state.plannerError
    ? 'Background planner error: '+state.plannerError
    : state.route?.solved
      ? `${state.route.planner||'pose lattice'} · ~${state.route.estimatedSeconds.toFixed(1)} s · actor frames ${sim.controlFrames||0} · joints ${sim.motorTravelRad.toFixed(3)} rad · cmd ${navCommandUpdates}`
      : `Searching · actor frames ${sim.controlFrames||0} · joints ${sim.motorTravelRad.toFixed(3)} rad · cmd ${navCommandUpdates}`;
}
let liveRevision = null;
let livePolling = false;
let showcaseManifest = null;

const settingsDrawer = $('settingsDrawer');
const drawerBackdrop = $('drawerBackdrop');
const settingsOpen = $('settingsOpen');
const settingsClose = $('settingsClose');
function setSettingsOpen(open) {
  settingsDrawer.hidden = !open;
  drawerBackdrop.hidden = !open;
  settingsDrawer.setAttribute('aria-hidden', String(!open));
  settingsOpen.setAttribute('aria-expanded', String(open));
  if (open) settingsClose.focus();
  else settingsOpen.focus();
}
settingsOpen.addEventListener('click', () => setSettingsOpen(true));
settingsClose.addEventListener('click', () => setSettingsOpen(false));
drawerBackdrop.addEventListener('click', () => setSettingsOpen(false));
window.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !settingsDrawer.hidden) { setSettingsOpen(false); e.preventDefault(); }
});

let sim = null;
let bundle = null;
let policyLibrary = [];
let activePolicyIndex = -1;
let left = {x: 0, y: 0};
let right = {x: 0, y: 0};
let lastFrame = performance.now() / 1000;
let fpsEma = 0;
let lastTelemetry = 0;
let loading = false;

function setStatus(text, kind='idle') {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function deg(rad) { return rad * 180 / Math.PI; }
function fmtSigned(x, n=2) { return `${x >= 0 ? '+' : ''}${x.toFixed(n)}`; }

function limits() {
  const trained=bundle?.control?.command_limits ||
    {vx_min:-0.2,vx_max:0.6,vy_abs:0.22,yaw_abs:1.5,pitch_abs_rad:0.08};
  const manualPitch=pitchRange.value==='trained' ? trained.pitch_abs_rad
    : Number(pitchRange.value)*Math.PI/180;
  return {...trained, pitch_abs_rad:manualPitch};
}
function applyViewerPitchRange() {
  const trained=bundle?.control?.command_limits?.pitch_abs_rad;
  if (Number.isFinite(trained)) {
    pitchRange.options[0].textContent=`Trained ±${deg(trained).toFixed(0)}°`;
  }
  const extended=pitchRange.value!=='trained';
  $('pitchRangeNote').hidden=!extended;
  sim?.setViewerPitchLimitDeg(extended?Number(pitchRange.value):null);
  updateStickLabels();
}
function applyHeadingGain() {
  const kp=Number(headingGain.value);
  $('headingGainOut').textContent=kp.toFixed(1);
  sim?.setHeadingGain(kp);
}

function updateStickLabels() {
  const {vx, vy, yaw, pitch} = joystickToCommands(left.x, left.y, right.x, right.y, limits(), joystickMode.value);
  leftValues.textContent = `vx ${fmtSigned(vx)} · vy ${fmtSigned(vy)}`;
  rightValues.textContent = `yaw ${fmtSigned(yaw)} · pitch ${fmtSigned(deg(pitch),1)}°`;
  sim?.setJoystickTargets(left.x, left.y, right.x, right.y);
}

const leftStick = new VirtualStick($('leftStick'), (x,y) => { if (demoActive && (x || y)) stopDemo(); left={x,y}; updateStickLabels(); });
const rightStick = new VirtualStick($('rightStick'), (x,y) => { if (demoActive && (x || y)) stopDemo(); right={x,y}; updateStickLabels(); });

function applyAdvanced() {
  const h = Number($('heightCmd').value) / 1000;
  const r = Number($('rollCmd').value) * Math.PI / 180;
  const s = Number($('stepCmd').value) / 1000;
  const p = Number($('phaseCmd').value);
  $('heightCmdOut').textContent = `${Math.round(h*1000)} mm`;
  $('rollCmdOut').textContent = `${deg(r).toFixed(1)}°`;
  $('stepCmdOut').textContent = `${Math.round(s*1000)} mm`;
  $('phaseCmdOut').textContent = p <= 0.001 ? 'auto' : `${p.toFixed(2)} Hz`;
  sim?.setAdvanced({height:h, roll:r, stepHeight:s, phaseRate:p});
}
for (const id of ['heightCmd','rollCmd','stepCmd','phaseCmd']) $(id).addEventListener('input', applyAdvanced);

function metricText(x) {
  const n = Number(x);
  return Number.isFinite(n) && n >= 0 ? n.toFixed(3) : '—';
}

function updateResultCard() {
  const card = $('resultCard');
  const ev = bundle?.evaluation;
  if (!ev) { card.hidden = true; return; }
  card.hidden = false;
  $('resultRole').textContent = ev.role || ev.summary_name || bundle?.source?.run_name || 'held-out';
  $('resultBroad').textContent = metricText(ev.broad);
  $('resultHard').textContent = metricText(ev.screen_hard_recovery);
  $('resultGrowth').textContent = metricText(ev.screen_growth_headroom);
  $('resultRecovery').textContent = metricText(ev.recovery);
  $('resultWorst').textContent = metricText(ev.worst_direction);
  $('resultCal').textContent = metricText(ev.calibration_mae);
}

function refreshPolicySelect() {
  policySelect.innerHTML = '';
  policyLibrary.forEach((entry, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    const src = entry.bundle.source || {};
    const seed = entry.bundle.evaluation?.seed;
    opt.textContent = `${seed != null ? `s${seed} · ` : ''}${src.run_name || entry.fileName}`;
    policySelect.appendChild(opt);
  });
  policySelect.hidden = policyLibrary.length < 2;
  if (activePolicyIndex >= 0) policySelect.value = String(activePolicyIndex);
}

async function activatePolicy(index) {
  const entry = policyLibrary[index];
  if (!entry || loading) return;
  loading = true;
  activePolicyIndex = index;
  setStatus(`Loading ${entry.fileName}…`, 'warn');
  pauseBtn.disabled = true;
  resetBtn.disabled = true;
  try {
    const hotSwap = sim && sim.canSwapPolicy(entry.bundle);
    if (hotSwap) sim.replacePolicy(entry.bundle);
    else {
      const nextSim = await EmuBrowserSim.create(canvas, entry.bundle);
      if (sim) sim.dispose();
      sim = nextSim;
    }
    bundle = entry.bundle;
    sim.setCameraMode(cameraMode);
    sim.setJoystickMode(joystickMode.value);
    sim.setHeadingHold(headingHoldToggle.checked);
    applyHeadingGain();
    applyViewerPitchRange();
    sim.setTreesEnabled(treesToggle.checked);
    sim.setTerrainMode(terrainMode.value,Number(terrainMax.value));
    sim.follow = followToggle.checked;
    sim.autoReset = autoResetToggle.checked || demoActive;
    sim.sensorNoise = noiseToggle.checked;
    sim.setAppearance({grid: gridToggle.checked, trail: trailToggle.checked, shadows: shadowToggle.checked});
    applyAdvanced();
    updateStickLabels();

    const src = bundle.source || {};
    $('policyName').textContent = src.run_name || entry.fileName;
    $('policyStep').textContent = Number(src.step || 0).toLocaleString();
    if (showcaseManifest) {
      $('showcaseTitle').textContent = entry.isShowcase
        ? (showcaseManifest.trained ? 'CHAMPION POLICY' : 'UNTRAINED PREVIEW')
        : 'LOCAL POLICY';
      $('showcaseSubtitle').textContent = entry.isShowcase
        ? (showcaseManifest.trained ? 'Browser-local physics and inference · no account needed' : 'Viewer preview only. This is not a trained walking policy.')
        : 'Your own dropped policy — not the published model';
    }
    pitchNotice.hidden = bundle.features?.pitch_tracking_trained === true;
    updateResultCard();
    refreshPolicySelect();
    pauseBtn.disabled = false;
    resetBtn.disabled = false;
    pauseBtn.textContent = 'Pause';
    setStatus(`Local MuJoCo/WASM · actor ${String(src.actor_sha256 || '').slice(0,10) || 'loaded'}`, 'ok');
    dragDepth = 0;
    dropOverlay.hidden = true;
    sim.resize();
    // An explicitly opened local .emupolicy also needs demo controls:
    // /api/initial-policy is not a live watcher or static Pages manifest.
    if (livePanel.hidden && !showcaseManifest) {
      livePanel.hidden = false;
      $('demoPanelTitle').textContent = 'DEMO CONTROLS';
      $('liveUpdate').closest('label').hidden = true;
      $('livePhase').textContent = 'Local policy';
      $('liveStep').textContent = Number(src.step || 0).toLocaleString();
      $('liveNote').textContent = 'Browser-only navigation: no training server involved';
      demoActive = $('liveDemo').checked;
    }
    demo.reset();
    lastResetGeneration=-1;
    syncGoalMode();
    return true;
  } catch (err) {
    console.error(err);
    setStatus(err?.message || String(err), 'error');
    return false;
  } finally {
    loading = false;
  }
}

async function addPolicyFiles(files) {
  const incoming = [...files];
  if (!incoming.length) return;
  setStatus(`Reading ${incoming.length} polic${incoming.length === 1 ? 'y' : 'ies'}…`, 'warn');
  let firstNew = -1;
  for (const file of incoming) {
    try {
      const parsed = parsePolicy(await file.text());
      const actor = String(parsed.source?.actor_sha256 || '');
      const duplicate = actor && policyLibrary.some(x => String(x.bundle.source?.actor_sha256 || '') === actor);
      if (duplicate) continue;
      if (firstNew < 0) firstNew = policyLibrary.length;
      policyLibrary.push({ fileName: file.name, bundle: parsed });
    } catch (err) {
      console.error(err);
      setStatus(`${file.name}: ${err?.message || String(err)}`, 'error');
    }
  }
  refreshPolicySelect();
  policyFile.value = '';
  if (firstNew >= 0) await activatePolicy(firstNew);
}

policyFile.addEventListener('change', () => {
  if (policyFile.files?.length) addPolicyFiles(policyFile.files);
});
policySelect.addEventListener('change', () => activatePolicy(Number(policySelect.value)));

pauseBtn.addEventListener('click', () => {
  if (!sim) return;
  sim.paused = !sim.paused;
  pauseBtn.textContent = sim.paused ? 'Resume' : 'Pause';
});
resetBtn.addEventListener('click', () => sim?.reset());
cameraBtn.addEventListener('click', () => {
  cameraMode = cameraModes[(cameraModes.indexOf(cameraMode) + 1) % cameraModes.length];
  sim?.setCameraMode(cameraMode);
  cameraBtn.textContent = cameraLabel(cameraMode);
  cameraBtn.setAttribute('aria-label', `Camera: ${cameraDescription(cameraMode)}`);
});
function updateTerrain() {
  $('terrainMaxOut').textContent = `${terrainMax.value}°`;
  treesToggle.disabled = terrainMode.value!=='park';
  if (!sim) return;
  try { sim.setTerrainMode(terrainMode.value,Number(terrainMax.value)); if(goalDemoActive())navigator.reset(); applyAdvanced(); updateStickLabels(); }
  catch(err) { terrainMode.value='flat';sim.setTerrainMode('flat');setStatus(`Terrain unavailable: ${err.message}`,'error'); }
}
terrainMode.addEventListener('change',updateTerrain);
terrainMax.addEventListener('change',updateTerrain);
treesToggle.addEventListener('change',()=>{if(sim)try{sim.setTreesEnabled(treesToggle.checked);if(goalDemoActive())navigator.reset();}catch(err){setStatus(`Tree collision unavailable: ${err.message}`,'error');}});
joystickMode.addEventListener('change',()=>{
  $('moveStickMode').textContent = joystickMode.value==='hybrid'?'vx / vy + turn':'vx / vy';
  sim?.setJoystickMode(joystickMode.value);
  updateStickLabels();
});
headingHoldToggle.addEventListener('change',()=>sim?.setHeadingHold(headingHoldToggle.checked));
headingGain.addEventListener('input',applyHeadingGain);
pitchRange.addEventListener('change',applyViewerPitchRange);
followToggle.addEventListener('change', () => { if (sim) sim.follow = followToggle.checked; });
autoResetToggle.addEventListener('change', () => { if (sim) sim.autoReset = autoResetToggle.checked || demoActive; });
noiseToggle.addEventListener('change', () => { if (sim) sim.sensorNoise = noiseToggle.checked; });
const applyAppearance = () => {
  sim?.setAppearance({grid: gridToggle.checked, trail: trailToggle.checked, shadows: shadowToggle.checked});
};
for (const control of [gridToggle, trailToggle, shadowToggle]) control.addEventListener('change', applyAppearance);


window.addEventListener('resize', () => sim?.resize());
window.addEventListener('beforeunload', () => {navWorker?.dispose();sim?.dispose();});

let dragDepth = 0;
window.addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; dropOverlay.hidden = false; });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('dragleave', e => { e.preventDefault(); dragDepth = Math.max(0, dragDepth-1); if (!dragDepth) dropOverlay.hidden = true; });
window.addEventListener('drop', e => {
  e.preventDefault(); dragDepth = 0; dropOverlay.hidden = true;
  const files = [...(e.dataTransfer?.files || [])];
  if (files.length) addPolicyFiles(files);
});

// Desktop fallback: WASD = vx/vy, Q/E = yaw, R/F = pitch.
const held = new Set();
function updateKeyboard() {
  let lx = 0, ly = 0, rx = 0, ry = 0;
  if (held.has('KeyA')) lx -= 1;
  if (held.has('KeyD')) lx += 1;
  if (held.has('KeyW')) ly += 1;
  if (held.has('KeyS')) ly -= 1;
  if (held.has('KeyQ')) rx -= 1;
  if (held.has('KeyE')) rx += 1;
  if (held.has('KeyR')) ry += 1;
  if (held.has('KeyF')) ry -= 1;
  if (lx || ly) leftStick.set(lx, ly);
  else if (leftStick.pointerId === null && [...held].some(k => ['KeyA','KeyD','KeyW','KeyS'].includes(k)) === false) leftStick.set(0,0);
  if (rx || ry) rightStick.set(rx, ry);
  else if (rightStick.pointerId === null && [...held].some(k => ['KeyQ','KeyE','KeyR','KeyF'].includes(k)) === false) rightStick.set(0,0);
}
window.addEventListener('keydown', e => {
  if (!settingsDrawer.hidden || ['INPUT','BUTTON','SUMMARY','SELECT'].includes(document.activeElement?.tagName)) return;
  if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','KeyR','KeyF'].includes(e.code)) {
    if (demoActive) stopDemo();
    held.add(e.code); updateKeyboard(); e.preventDefault();
  } else if (e.code === 'Space' && sim) {
    sim.paused = !sim.paused; pauseBtn.textContent = sim.paused ? 'Resume' : 'Pause'; e.preventDefault();
  } else if (e.code === 'Backspace' && sim) { sim.reset(); e.preventDefault(); }
});
window.addEventListener('keyup', e => { held.delete(e.code); updateKeyboard(); });

function updateTelemetry(now, dt) {
  if (!sim || now - lastTelemetry < 0.10) return;
  lastTelemetry = now;
  const t = sim.telemetry();
  $('fps').textContent = fpsEma ? fpsEma.toFixed(0) : '—';
  $('height').textContent = `${(t.height*1000).toFixed(0)} mm${t.fallen ? ' · FALL' : ''}`;
  $('vxRead').textContent = `${fmtSigned(t.vx)} / ${fmtSigned(t.cmd.vx)}`;
  $('vyRead').textContent = `${fmtSigned(t.vy)} / ${fmtSigned(t.cmd.vy)}`;
  $('yawRead').textContent = `${fmtSigned(t.yaw)} / ${fmtSigned(t.cmd.yaw)}`;
  $('yawRequestRead').textContent = `${fmtSigned(t.requestedYaw)} ${demoActive ? '· demo' : '· manual'}`;
  $('headingRead').textContent=t.headingHold
    ? `${fmtSigned(deg(t.headingTarget),0)}° / ${fmtSigned(deg(t.headingActual),0)}°`
    : 'direct yaw-rate';
  const s = t.slope;
  $('slopeRead').textContent = !s.supported ? 'No terrain underfoot'
    : s.flat ? '0.0° · level' : `${s.degrees.toFixed(1)}°`;
  $('slopeArrow').hidden = !s.supported || s.flat;
  $('slopeArrow').style.transform = `rotate(${s.arrowDegrees.toFixed(1)}deg)`;
  $('leanRead').textContent = `${fmtSigned(deg(t.robotLean),1)}°`;
  $('pitchRead').textContent = `${fmtSigned(deg(t.pitch),1)}° / ${fmtSigned(deg(t.cmd.pitch),1)}°`;
}

function frame(ms) {
  // Schedule the NEXT browser frame before doing anything that can throw.
  // An exception in the experimental navigator may never freeze the actor.
  requestAnimationFrame(frame);
  try{
  const now = ms / 1000;
  const dt = Math.max(0, Math.min(0.1, now - lastFrame));
  lastFrame = now;
  if (dt > 0) {
    const fps = 1/dt;
    fpsEma = fpsEma ? 0.92*fpsEma + 0.08*fps : fps;
  }
  if (sim) {
    if (demoActive && !sim.paused) {
      if(goalDemoActive()){
        try{navigationUpdate(dt);}
        catch(error){
          // Browser navigation is optional; on failure immediately restore
          // known-working random commands without touching actor/simulation.
          console.error('Goal navigation disabled:',error);
          demoMode.value='commands';
          syncGoalMode();
          setStatus('Navigation error: '+String(error?.message||error), 'error');
          demo.tick(dt, limits(), applyAutonomousCommand,
            bundle?.demo?.node_candidates||[]);
        }
      }else{
        demo.tick(dt, limits(), applyAutonomousCommand, bundle?.demo?.node_candidates || []);
        $('demoKind').textContent = demo.kind;
      }
    }
    sim.advance(dt);
    sim.render(dt);
    updateTelemetry(now, dt);
  }
  }catch(error){
    console.error('Viewer frame error:',error);
    // Never secretly pause a working actor because a new optional
    // navigation/HUD/render calculation threw. Prefer the established demo
    // command stream, leaving the next animation frame scheduled.
    if(goalDemoActive()){
      demoMode.value='commands';
      syncGoalMode();
    }
    setStatus('Viewer frame error: '+String(error?.message||error),'error');
  }
}

function stopDemo() {
  demoActive = false;
  $('liveDemo').checked = false;
  demo.reset();
  syncGoalMode();
  $('demoKind').textContent = 'manual control';
  if (sim) {
    sim.autoReset = autoResetToggle.checked;
    sim.setJoystickTargets(left.x, left.y, right.x, right.y);
  }
}

$('liveDemo').addEventListener('change', () => {
  demoActive = $('liveDemo').checked;
  demo.reset();
  syncGoalMode();
  if (sim && !demoActive) sim.setJoystickTargets(left.x, left.y, right.x, right.y);
  $('demoKind').textContent = demoActive ? 'choosing command…' : 'manual control';
});
demoMode.addEventListener('change',()=>{
  demo.reset();
  syncGoalMode();
  $('demoKind').textContent=goalDemoActive()?'Planning goal route…':'random in-envelope commands';
});
navPlanner.addEventListener('change',()=>{
  // Cancel any route built for the previous motion model. Async replies
  // from an old Worker request cannot apply after reset generation changes.
  navigator.plannerMode=navPlanner.value;
  navigator.reset();
  if(sim)sim.clearNavigationOverlay();
  $('demoKind').textContent='Planning with '+navPlanner.selectedOptions[0].textContent;
});
$('navNext').addEventListener('click',()=>{
  if(!sim)return;
  if(!goalDemoActive()){
    demoMode.value='goals';demoActive=true;$('liveDemo').checked=true;
    syncGoalMode();
  }
  navigator.nextGoal(sim.navigationPose(),bundle.control.command_limits,sim.navigationTrees());
  const state=navigator.status(sim.navigationPose());
  sim.setNavigationOverlay(state.goal,state.route);
});

async function pollLive() {
  if (livePolling) return;
  livePolling = true;
  try {
    const response = await fetch('/api/live/status', {cache:'no-store'});
    if (!response.ok) throw new Error(`Status HTTP ${response.status}`);
    const state = await response.json();
    $('livePhase').textContent = state.phase || 'Awaiting checkpoint';
    $('liveStep').textContent = state.step == null ? '—' : Number(state.step).toLocaleString();
    $('liveNote').textContent = state.note || (state.available ? 'Latest validated checkpoint ready' : 'Waiting for first checkpoint');
    if (!state.available || !state.revision || state.revision === liveRevision ||
        !$('liveUpdate').checked || loading) return;
    const policyResponse = await fetch('/api/live/policy', {cache:'no-store'});
    if (!policyResponse.ok) throw new Error(`Policy HTTP ${policyResponse.status}`);
    const parsed = parsePolicy(await policyResponse.text());
    const actualRevision = `${parsed.source.step}:${parsed.source.actor_sha256}`;
    // A policy may advance between the two requests; only accept its own id.
    const idx = policyLibrary.findIndex(e => e.isLive);
    const entry = {fileName:'LIVE · latest checkpoint', bundle:parsed, isLive:true};
    let target = idx;
    if (idx < 0) { target = policyLibrary.length; policyLibrary.push(entry); }
    else policyLibrary[idx] = entry;
    const ok = await activatePolicy(target);
    if (ok) liveRevision = actualRevision;
  } catch (err) {
    $('liveNote').textContent = `Live feed: ${err?.message || String(err)}`;
    console.error(err);
  } finally {
    livePolling = false;
  }
}

async function startShowcaseFeed() {
  // Relative URLs intentionally support GitHub Pages project subpaths.
  const manifestResponse = await fetch(new URL('./showcase.json', document.baseURI), {cache:'no-store'});
  const manifest = await readShowcaseManifest(manifestResponse);
  showcaseManifest = manifest;
  $('modeLabel').textContent = 'public playground';
  $('showcaseBanner').hidden = false;
  $('showcaseTitle').textContent = manifest.trained ? 'CHAMPION POLICY' : 'UNTRAINED PREVIEW';
  $('showcaseSubtitle').textContent = manifest.trained
    ? 'Browser-local physics and inference · no account needed'
    : 'Viewer preview only. This is not a trained walking policy.';
  $('showcaseNote').hidden = false;
  const buildLabel = manifest.source_commit ? ` · Viewer build ${manifest.source_commit.slice(0, 8)}` : '';
  $('showcaseNote').textContent = (manifest.trained
    ? 'Autonomous commands are demonstrations, not held-out evaluation.'
    : 'Preview weights are untrained. A trained champion has not been published yet.') + buildLabel;
  if (manifest.source_url) {
    $('sourceLink').href = manifest.source_url;
    $('sourceLink').hidden = false;
  }
  setStatus('Fetching bundled policy…', 'warn');
  const file = await fetch(showcasePolicyUrl(manifest, document.baseURI), {cache:'no-store'});
  assertShowcasePolicyResponse(file);
  const parsed = parsePolicy(await file.text());
  policyLibrary.push({fileName:manifest.label, bundle:parsed, isShowcase:true});
  const ok = await activatePolicy(policyLibrary.length - 1);
  if (!ok) throw new Error('Showcase policy could not be started');
  demoActive = manifest.autoplay;
  $('liveDemo').checked = demoActive;
  // The public playground gets the same demo controls as the live watcher.
  livePanel.hidden = false;
  $('demoPanelTitle').textContent = 'DEMO CONTROLS';
  $('liveUpdate').closest('label').hidden = true;
  $('livePhase').textContent = manifest.trained ? 'Published champion' : 'Untrained fixture';
  $('liveStep').textContent = Number(parsed.source?.step || 0).toLocaleString();
  $('liveNote').textContent = manifest.trained
    ? 'Static, published policy — no training server required'
    : 'Demo fixture: explore the controls, not the walking performance';
  $('demoKind').textContent = demoActive ? 'choosing command…' : 'manual control';
  sim.autoReset = autoResetToggle.checked || demoActive;
  syncGoalMode();
  return true;
}

async function startViewerFeed() {
  try {
    // Local Python server takes precedence, including when a previously built
    // static showcase manifest happens to be present in webviewer_dist.
    const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
    if (loopback) {
      const response = await fetch('/api/viewer-config', {cache:'no-store'});
      // The Vite preview SPA fallback responds with index.html and HTTP 200
      // for /api/viewer-config. Only actual JSON can identify our Python API.
      if (isJsonResponse(response)) {
        const config = await response.json();
        if (!isViewerConfig(config)) {
          throw new Error('Invalid EMU local viewer API configuration');
        }
        if (config.initial_policy && !config.live) {
          const file = await fetch('/api/initial-policy', {cache:'no-store'});
          if (file.ok) {
            const parsed = parsePolicy(await file.text());
            policyLibrary.push({fileName:'Preloaded policy',bundle:parsed});
            await activatePolicy(policyLibrary.length-1);
          }
        }
        if (config.live) {
          livePanel.hidden = false;
          demoActive = $('liveDemo').checked;
          syncGoalMode();
          await pollLive();
          setInterval(pollLive, 12000);
        }
        return;
      }
    }
    await startShowcaseFeed();
  } catch (err) {
    console.error('Viewer startup:', err);
    setStatus(`Startup: ${err?.message || String(err)}`, 'error');
  }
}

applyAdvanced();
applyHeadingGain();
applyViewerPitchRange();
startViewerFeed();
requestAnimationFrame(frame);
