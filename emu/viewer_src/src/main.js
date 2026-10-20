import './style.css';
import { parsePolicy } from './policy.js';
import { VirtualStick } from './joystick.js';
import { joystickToCommands } from './controls.js';
import { EmuBrowserSim } from './sim.js';
import { DemoHandoff, commandStickPositions } from './demo-handoff.js';
import { GoalNavigator } from './goal-navigation.js';
import { GoalPlannerWorker } from './goal-planner-client.js';
import { visibleChallengeLabel } from './challenge-course.js';
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
const challengeTier = $('challengeTier');
const challengeLogs = $('challengeLogs');
const demoMode = $('demoMode');
const headingHoldToggle = $('headingHoldToggle');
const headingGain = $('headingGain');
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
const dropOverlay = $('dropOverlay');
const leftValues = $('leftValues');
const rightValues = $('rightValues');
const livePanel = $('livePanel');
const navigator = new GoalNavigator();
let navWorker;
let workerUnavailable=null;
try{navWorker=new GoalPlannerWorker();}
catch(error){workerUnavailable=String(error?.message||error);console.error('Goal worker:',error);}
navigator.useBackgroundPlanner(request=>{
  if(!navWorker)throw new Error(workerUnavailable||'Background navigation unavailable');
  return navWorker.request(request);
});
const navHud = $('navHud');
let lastResetGeneration = -1;
let navCommandClock = Infinity;
let navCommandUpdates = 0;
let navLastCommand = {vx:0,vy:0,yaw:0};
let demoActive = true;
let navigationFault = false;
const handoff = new DemoHandoff();
const goalDemoActive = () => demoActive;
function syncGoalMode() {
  const enabled=Boolean(sim && goalDemoActive());
  navigator.active=enabled;
  // Always reset the target sequence after terrain change, actor swap,
  // fall/reset or manual handoff. No stale route may cross removed terrain.
  const circuit=demoMode.value==='circuit'
    ? sim?.navigationChallengeCourse()?.waypoints||null : null;
  navigator.setCircuitWaypoints(circuit);
  $('demoRouteNote').textContent = circuit
    ? 'Approach → cross each obstacle in order. Route progress is a DEMO, not a certified crossing score.'
    : 'Random pose targets within the arena. Steering uses the existing neural policy.';
  $('navNext').textContent=circuit?'Next waypoint':'New destination';
  navCommandClock=Infinity;
  navCommandUpdates=0;
  navLastCommand={vx:0,vy:0,yaw:0};
  if(sim) {
    // Autonomy and manual use the same policy command/observation path.
    sim.setNavigationPresentation(enabled);
    sim.autoReset=autoResetToggle.checked||demoActive;
  }
  navHud.hidden=!sim;
  $('navNext').disabled=!sim;
}
// One proven command delivery path for every autonomous input source.
function applyAutonomousCommand(command) {
  sim.setDemoCommands(command);
  navLastCommand={...command};
  updateStickLabels();
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
  // Route feedback up to 10 Hz through the same command path.
  // Neural actor inference remains at its normal 50 Hz.
  const commandPeriod=.10;
  navCommandClock+=Math.min(Math.max(dt,0),.1);
  if(plannedCommand && navCommandClock>=commandPeriod){
    applyAutonomousCommand(plannedCommand);
    navCommandClock=0;
    navCommandUpdates++;
  }
  const state=navigator.status(pose);
  sim.setNavigationOverlay(state.goal,state.route);
  $('demoKind').textContent=state.reason;
  if(!state.goal){
    $('navHudTitle').textContent=state.reason;
    $('navHudInfo').textContent='Planning a route…';
    $('navEstimate').textContent=state.plannerError
      ? 'Planner error: '+state.plannerError
      : `Actor frames ${sim.controlFrames||0} · joints ${sim.motorTravelRad.toFixed(3)} rad · cmds ${navCommandUpdates}`;
    return;
  }
  const degrees=n=>Math.round(n*180/Math.PI);
  $('navHudTitle').textContent=state.circuit && state.goal?.label
    ? `CIRCUIT · ${state.goal.label}`
    : `TARGET ${state.arrivals+1} · ${state.reason}`;
  $('navHudInfo').textContent=`${state.distance.toFixed(2)} m · heading ${degrees(state.headingError)}° · ${state.arrivals} parked${state.failures?' · '+state.failures+' timed out':''}`;
  $('navHudInfo').textContent+=` · line ${state.crossTrack>=0?'+':''}${state.crossTrack.toFixed(2)} m (avg ${state.meanTrackError.toFixed(2)}) · crosses ${state.crossings}`;
  $('navHudInfo').textContent+=` · profile speed ${state.referenceSpeed.toFixed(2)} / observed ${state.observedSpeed.toFixed(2)} m/s`;
  $('navEstimate').textContent=state.plannerError
    ? 'Background planner error: '+state.plannerError
    : state.route?.solved
      ? `${state.route.planner||'pose lattice'} · ~${state.route.estimatedSeconds.toFixed(1)} s plan / ${state.elapsed.toFixed(1)} s actual · mean heading Δ ${degrees(state.meanHeadingError)}° · mean speed Δ ${state.meanSpeedError.toFixed(2)} m/s · frames ${sim.controlFrames||0}`
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
  return bundle?.control?.command_limits ||
    {vx_min:-.2,vx_max:.6,vy_abs:.22,yaw_abs:1.5,pitch_abs_rad:.08};
}
function applyHeadingGain() {
  const kp=Number(headingGain.value);
  $('headingGainOut').textContent=kp.toFixed(1);
  sim?.setHeadingGain(kp);
}

function updateStickLabels() {
  if(demoActive){
    const sticks=commandStickPositions(navLastCommand,limits());
    leftStick.setDisplay(sticks.left.x,sticks.left.y);
    rightStick.setDisplay(sticks.right.x,sticks.right.y);
    leftValues.textContent=`vx ${fmtSigned(navLastCommand.vx)} · vy ${fmtSigned(navLastCommand.vy)}`;
    rightValues.textContent=`yaw ${fmtSigned(navLastCommand.yaw)} · pitch ${fmtSigned(deg(navLastCommand.pitch||0),1)}°`;
    return;
  }
  const {vx,vy,yaw,pitch}=joystickToCommands(left.x,left.y,right.x,right.y,limits());
  leftValues.textContent=`vx ${fmtSigned(vx)} · vy ${fmtSigned(vy)}`;
  rightValues.textContent=`yaw ${fmtSigned(yaw)} · pitch ${fmtSigned(deg(pitch),1)}°`;
  sim?.setJoystickTargets(left.x,left.y,right.x,right.y);
}
function humanStickChange(side,x,y) {
  if(demoActive && (Math.abs(x)>.06||Math.abs(y)>.06))stopDemo();
  if(!demoActive){
    handoff.takeControl(performance.now()/1000);
    if(side==='left')left={x,y};else right={x,y};
    updateStickLabels();
  }
}
const leftStick=new VirtualStick($('leftStick'),(x,y)=>humanStickChange('left',x,y));
const rightStick=new VirtualStick($('rightStick'),(x,y)=>humanStickChange('right',x,y));

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
    sim.setHeadingHold(headingHoldToggle.checked);
    applyHeadingGain();
    sim.setTreesEnabled(treesToggle.checked);
    sim.setChallengeOptions(Number(challengeTier.value),challengeLogs.checked);
    sim.setTerrainMode(terrainMode.value,Number(terrainMax.value));
    displayCourseControls();
    sim.follow = followToggle.checked;
    sim.autoReset = autoResetToggle.checked || demoActive;
    sim.sensorNoise = noiseToggle.checked;
    sim.setAppearance({grid: gridToggle.checked, trail: trailToggle.checked, shadows: shadowToggle.checked});
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
      $('liveNote').textContent = 'Runs entirely in your browser';
      demoActive = $('liveDemo').checked;
    }
    lastResetGeneration=-1;
    navigationFault=false;
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
function displayCourseControls() {
  const isCourse=terrainMode.value==='course';
  $('challengeControls').hidden=!isCourse;
  $('parkSlopeControls').hidden=terrainMode.value!=='park';
  treesToggle.closest('label').hidden=terrainMode.value!=='park';
  if(isCourse){
    $('challengeDescription').textContent=
      visibleChallengeLabel(sim?.navigationChallengeCourse()||{
        tierMm:Number(challengeTier.value),logs:challengeLogs.checked})+
      ' · Coloured MuJoCo collision geoms, not decoration. Pilot evidence is not a universal gait guarantee.';
  }
}
function updateTerrain() {
  $('terrainMaxOut').textContent = `${terrainMax.value}°`;
  treesToggle.disabled = terrainMode.value!=='park';
  if(terrainMode.value!=='course' && demoMode.value==='circuit')
    demoMode.value='random'; // circuit has physical checkpoints only in course
  displayCourseControls();
  if (!sim) return;
  try {
    sim.setTerrainMode(terrainMode.value,Number(terrainMax.value));
    syncGoalMode();
    updateStickLabels();
  } catch(err) {
    terrainMode.value='flat';
    sim.setTerrainMode('flat');
    demoMode.value='random';
    displayCourseControls();
    syncGoalMode();
    setStatus(`Terrain unavailable: ${err.message}`,'error');
  }
}
terrainMode.addEventListener('change',updateTerrain);
terrainMax.addEventListener('change',updateTerrain);
challengeTier.addEventListener('change',()=>{
  if(sim)try{
    sim.setChallengeOptions(Number(challengeTier.value),challengeLogs.checked);
    syncGoalMode(); displayCourseControls();
  }catch(err){setStatus(`Challenge unavailable: ${err.message}`,'error');}
  else displayCourseControls();
});
challengeLogs.addEventListener('change',()=>{
  if(sim)try{
    sim.setChallengeOptions(Number(challengeTier.value),challengeLogs.checked);
    syncGoalMode(); displayCourseControls();
  }catch(err){setStatus(`Challenge unavailable: ${err.message}`,'error');}
  else displayCourseControls();
});
demoMode.addEventListener('change',()=>{
  if(demoMode.value==='circuit'&&terrainMode.value!=='course'){
    terrainMode.value='course';
    updateTerrain();
  }else syncGoalMode();
});
treesToggle.addEventListener('change',()=>{if(sim)try{sim.setTreesEnabled(treesToggle.checked);syncGoalMode();}catch(err){setStatus(`Tree collision unavailable: ${err.message}`,'error');}});
headingHoldToggle.addEventListener('change',()=>sim?.setHeadingHold(headingHoldToggle.checked));
headingGain.addEventListener('input',applyHeadingGain);
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
  if(demoActive)return; // Never let idle key-up repaint autonomous sticks.
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
  $('yawRequestRead').textContent = `${fmtSigned(t.requestedYaw)} ${demoActive ? '· auto' : '· manual'}`;
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
    if (!demoActive && !navigationFault && handoff.shouldResume(now,
       held.size>0 || leftStick.pointerId!==null || rightStick.pointerId!==null ||
       Math.hypot(left.x,left.y)>.06 || Math.hypot(right.x,right.y)>.06)) {
      startDemo();
    }
    if (demoActive && !sim.paused) {
      try{navigationUpdate(dt);}
      catch(error){
        console.error('Goal navigation:',error);
        navigationFault=true;
        stopDemo();
        sim.setDemoCommands({vx:0,vy:0,yaw:0,pitch:0});
        setStatus('Navigation error: '+String(error?.message||error), 'error');
      }
    }else if(!demoActive) {
      const leftTime=Math.max(0,30-(now-handoff.lastHumanInput));
      $('navHudTitle').textContent='MANUAL CONTROL';
      $('navHudInfo').textContent=navigationFault?'Navigation requires a restart':
        `Autonomous navigation resumes in ${Math.ceil(leftTime)}s`;
      $('navEstimate').textContent='Move either joystick or use WASD · Q/E to steer';
    }
    sim.advance(dt);
    sim.render(dt);
    updateTelemetry(now, dt);
  }
  }catch(error){
    console.error('Viewer frame error:',error);
    // Keep the next browser animation frame alive for recovery.
    navigationFault=true;
    if(goalDemoActive())stopDemo();
    setStatus('Viewer frame error: '+String(error?.message||error),'error');
  }
}

function stopDemo(){
  demoActive=false;
  handoff.takeControl(performance.now()/1000);
  $('liveDemo').checked=false;
  syncGoalMode();
  // Reset the heading setpoint to the ACTUAL chassis heading. Otherwise the
  // P controller can keep completing the discarded autonomous turn.
  sim?.setHeadingHold(headingHoldToggle.checked);
  $('demoKind').textContent='Manual control';
  leftStick.setDisplay(left.x,left.y);
  rightStick.setDisplay(right.x,right.y);
  updateStickLabels();
}
function startDemo(){
  if(!sim||navigationFault)return;
  demoActive=true;
  handoff.resume();
  $('liveDemo').checked=true;
  left={x:0,y:0};right={x:0,y:0};
  syncGoalMode();
  sim.setHeadingHold(headingHoldToggle.checked);
  $('demoKind').textContent='Finding next destination…';
  updateStickLabels();
}
$('liveDemo').addEventListener('change',()=>{
  if($('liveDemo').checked)startDemo();else stopDemo();
});
$('navNext').addEventListener('click',()=>{
  if(!sim)return;
  if(!demoActive)startDemo();
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
  demoActive = manifest.trained;
  handoff.resume();
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
  $('demoKind').textContent = demoActive ? 'Finding next destination…' : 'Manual control';
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

applyHeadingGain();
displayCourseControls();
startViewerFeed();
requestAnimationFrame(frame);
