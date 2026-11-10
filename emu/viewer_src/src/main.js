import './style.css';
import { parsePolicy } from './policy.js';
import { VirtualStick } from './joystick.js';
import { joystickToCommands, DemoHandoff, commandStickPositions } from './controls.js';
import { EmuBrowserSim } from './sim.js';
import { terrainHeightOffsetsCm } from './sensor.js';
import { CARPET_DEFAULT, parseCarpetMetrics, trainingLimits } from './training-carpet.js';
import { MEADOW_DEFAULTS, normalizedMeadowSettings } from './sensor-lab-carpet.js';
import { carpetLearningStage, carpetRepresentativeRole, terrainRoleEnvelope } from './training-solid-carpet.js';
import { specialistLesson, friendlyFrontiers, SLIDER_FIELDS, roleSliders, customRoleOverride, carpetGeometrySignature } from './training-inspector.js';
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
const carpetRole=$('carpetRole'),carpetPanel=$('carpetPanel'),carpetImport=$('carpetImport');
const carpetValues=$('carpetValues'),carpetStatus=$('carpetStatus');
const carpetLive=$('carpetLive'),carpetFetch=$('carpetFetch'),carpetApply=$('carpetApply');
const carpetAutoApply=$('carpetAutoApply');
const carpetMatch=$('carpetMatch'),carpetDrive=$('carpetDrive');
const carpetShoves=$('carpetShoves'),carpetTestShove=$('carpetTestShove');
const carpetPushRate=$('carpetPushRate'),carpetResetLimits=$('carpetResetLimits');
const carpetDifficultyMode=$('carpetDifficultyMode'),carpetStepTrack=$('carpetStepTrack');
const worldField=$('worldField'),worldCarpet=$('worldCarpet'),fieldPanel=$('fieldPanel');
const carpetEdits=new Map();
const tuningInputs=Object.fromEntries(SLIDER_FIELDS.map(([key,id])=>[key,$(id)]));
let carpetExerciseClock=0;
let carpetLastExercise='';

let carpetDifficulty={...CARPET_DEFAULT};
// The sole physical authority for Meadow geology, independent of specialists.
let meadowSettings={...MEADOW_DEFAULTS};
let carpetSource='Illustrative defaults';
let carpetFetching=false;
let carpetLastAppliedSignature='';
let discoverCarpetFromLive=false;
const depthOn=$('depthOn'),depthTilt=$('depthTilt'),depthActorRays=$('depthActorRays');
const depthMount=$('depthMount'),depthHeight=$('depthHeight'),
      depthLateral=$('depthLateral'),depthYaw=$('depthYaw'),depthRays=$('depthRays'),depthSamples=$('depthSamples'),depthFootprint=$('depthFootprint');
const depthView=$('depthView');
const depthCells=Array.from({length:16},()=>{
  const cell=document.createElement('span');
  cell.className='depth-cell invalid';cell.textContent='—';
  $('depthGrid').appendChild(cell);return cell;
});
const treesToggle = $('treesToggle');
const parkFeaturesToggle = $('parkFeaturesToggle');
const parkChallengeLevel = $('parkChallengeLevel');
const parkCourseList = $('parkCourseList');
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
  const enabled=Boolean(sim && goalDemoActive() && terrainMode.value==='park');
  navigator.active=enabled;
  navigator.reset();
  navCommandClock=Infinity;
  navCommandUpdates=0;
  navLastCommand={vx:0,vy:0,yaw:0};
  if(sim) {
    // Autonomy and manual use the same policy command/observation path.
    sim.setNavigationPresentation(enabled);
    sim.autoReset=autoResetToggle.checked||demoActive;
  }
  // Field navigation is never visible in Carpet, even during a
  // manual-control handoff or when the 30s timer reaches zero.
  navHud.hidden=!enabled;
  $('navNext').disabled=!enabled;
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
  $('navHudTitle').textContent=`TARGET ${state.arrivals+1} · ${state.reason}`;
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
  if(terrainMode.value==='carpet'&&carpetDrive.value==='lesson'&&
     (Math.abs(x)>.06||Math.abs(y)>.06)){
    carpetDrive.value='manual';syncCarpetExercise();
  }
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
    sim.setParkFeaturesEnabled(parkFeaturesToggle.checked);
    sim.setParkChallengeLevel(parkChallengeLevel.value);
    sim.setTrainingCarpet({role:carpetRepresentativeRole(carpetEffectiveLimits(),carpetRole.value),
                           limits:carpetEffectiveLimits()});
    sim.setMeadowSettings(meadowSettings);
    sim.setTerrainMode(terrainMode.value,Number(terrainMax.value));
    updateParkCourseInfo();
    updateDepthSettings();
    // Retired sensor worlds are no longer reachable from the UI.
    if(terrainMode.value==='carpet'){
      demoActive=false;
      $('liveDemo').checked=false;
      carpetExerciseClock=0;
    }
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
    syncCarpetExercise();
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
function updateParkCourseInfo() {
  parkCourseList.closest('.park-course-info').hidden=terrainMode.value!=='park';
  parkCourseList.replaceChildren();
  if(terrainMode.value!=='park' || !parkFeaturesToggle.checked ||
     !sim?.park?.features?.length){
    const item=document.createElement('li');
    item.textContent='Select the terrain park and enable the course.';
    parkCourseList.appendChild(item);
    return;
  }
  for(const feature of sim.park.features){
    const item=document.createElement('li');
    const title=document.createElement('strong');
    title.textContent=feature.label+' — ';
    item.appendChild(title);
    item.appendChild(document.createTextNode(feature.challenge));
    parkCourseList.appendChild(item);
  }
}
function setWorld(world){
  if(!['park','carpet'].includes(world))throw Error('Unknown world');
  terrainMode.value=world;
  discoverCarpetFromLive=false;
  updateTerrain();
}
worldField.addEventListener('click',()=>setWorld('park'));
worldCarpet.addEventListener('click',()=>setWorld('carpet'));
function updateWorldTabs(){
  const carpet=terrainMode.value==='carpet';
  for(const [btn,on] of [[worldField,!carpet],[worldCarpet,carpet]]){
    btn.classList.toggle('active',on);
    btn.setAttribute('aria-pressed',String(on));
  }
  fieldPanel.hidden=carpet;
  carpetPanel.hidden=!carpet;
  $('liveDemo').disabled=carpet;
  navHud.hidden=carpet||!sim||!demoActive;
  document.body.dataset.world=carpet?'carpet':'field';
  $('carpetHudRow').hidden=!carpet;
  $('carpetGrowthRow').hidden=!carpet;
  $('viewerExercise').textContent=carpet
    ?currentCarpetLesson().heading:'Field exploration';
}
function carpetSourceLimits(){
  const role=carpetRepresentativeRole(carpetDifficulty,carpetRole.value);
  const academy=carpetStepTrack.value==='academy'&&role==='step';
  return {...carpetDifficulty,
    stepTrack:academy?'academy':'ordinary',
    stepUpM:academy
      ?carpetDifficulty.academyStepUpM??carpetDifficulty.stepUpM
      :carpetDifficulty.ordinaryStepUpM??carpetDifficulty.stepUpM,
    stepDownM:academy
      ?carpetDifficulty.academyStepDownM??carpetDifficulty.stepDownM
      :carpetDifficulty.ordinaryStepDownM??carpetDifficulty.stepDownM};
}
function carpetEffectiveLimits(){
  const source=carpetSourceLimits();
  const role=carpetRepresentativeRole(source,carpetRole.value);
  const custom=carpetMatch.checked?null:carpetEdits.get(role);
  if(custom)return {...source,previewOverride:custom};
  if(carpetDifficultyMode.value==='frontier'){
    // An explicit VIEWER challenge at the global source ceilings.
    // Do not confuse this unscaled geometry with MJX role birth values.
    return {...source,previewOverride:{terrain:{
      slopeDeg:source.slopeDeg,
      transitionDeg:source.transitionDeg,
      stepUpM:source.stepUpM,
      stepDownM:source.stepDownM}}};
  }
  return source;
}
function currentCarpetLesson(){
  return specialistLesson(carpetEffectiveLimits(),carpetRole.value);
}
function carpetTerrainSignature(){
  return carpetGeometrySignature(currentCarpetLesson(),carpetStepTrack.value);
}
function syncCarpetExercise(){
  if(!sim||terrainMode.value!=='carpet'){
    sim?.setShovePreview({enabled:false,forceLimitN:0});
    return;
  }
  const lesson=currentCarpetLesson();
  sim.autoReset=carpetDrive.value==='lesson'||autoResetToggle.checked;
  // Optional physical shoves obey the unified Meadow settings, NOT
  // training specialist settings or an invisible teacher assignment.
  const strength=Number($('meadowShove').value)/100;
  sim.setShovePreview({
    enabled:carpetShoves.checked&&strength>0,
    forceLimitN:carpetShoves.checked?4*strength:0,
    intensity:strength,
    mode:carpetPushRate.value
  });
  if(carpetDrive.value==='lesson'&&demoActive)stopDemo();
  if(carpetDrive.value==='manual'){
    $('lessonCommand').textContent='Viewer command: manual joystick / keyboard';
    sim.setDemoCommands({vx:0,vy:0,yaw:0,pitch:0});
  }
}
function renderCarpetValues(){
  const base=carpetEffectiveLimits();
  const lesson=currentCarpetLesson(),f=friendlyFrontiers(lesson);
  const measured=carpetMatch.checked&&
    carpetDifficultyMode.value==='assignment'&&carpetDifficulty.source==='training';
  const stage=carpetLearningStage(carpetDifficulty);
  const phase={
    foundation:'flat foundation',specialists:'six specialists',
    'multi-lip':'multi-lip Academy',illustrative:'illustrative'
  }[stage];
  $('carpetContext').textContent=`${lesson.heading}: ${lesson.description}.
    ${measured?'Training targets (not achieved skill).':'Editable viewer scenario; NOT live training difficulty.'}
    Infinite procedural Meadow: terrain increases in difficulty to the NORTH (+Y),
    becomes easier to the south and varies at steady difficulty east/west.
    EMU begins facing EAST (+X): turn left to explore harder northern terrain.
    15×15m moving real collision window, deterministic return visits, misty visual fallaway.`;
  if(terrainMode.value==='carpet')$('viewerExercise').textContent=lesson.heading;
  const values=roleSliders(base,carpetRole.value);
  for(const [key,id,unit,,digits] of SLIDER_FIELDS){
    const el=tuningInputs[key];
    el.value=String(values[key]);
    $(id+'Out').textContent=values[key].toFixed(digits)+unit;
  }
  const original=roleSliders(carpetSourceLimits(),carpetRole.value);
  const sRatio=carpetDifficulty.slopeDeg>0
    ?Math.round(original.slope/carpetDifficulty.slopeDeg*100):0;
  const tRatio=carpetDifficulty.stepUpM>0
    ?Math.round(original.up/(carpetSourceLimits().stepUpM*1000)*100):0;
  const frontier=carpetSourceLimits();
  const sourceRole=roleSliders(frontier,carpetRole.value);
  const slopeExposure=Number(frontier.slopeExposure??1);
  const stepExposure=Number(frontier.stepExposure??1);
  const fmt=(n,d=2)=>Number(n).toFixed(d);
  const roleIsStep=lesson.role==='step';
  $('carpetStepTrackRow').hidden=!roleIsStep;
  const activeStepTrack=roleIsStep?carpetStepTrack.value:'ordinary';
  $('carpetHudTerrain').textContent=
    `Meadow at start: ${meadowSettings.hillPeakDeg.toFixed(1)}° hill peaks · +${(meadowSettings.stepUpM*1000).toFixed(0)}/−${(meadowSettings.stepDownM*1000).toFixed(0)} mm max`;
  $('carpetPhysicalEvidence').textContent=[
    `GLOBAL FRONTIER · grade ≤${fmt(frontier.slopeDeg)}° · ${activeStepTrack==='academy'?'Academy signed':'Ordinary signed'} lips ≤${fmt(frontier.stepUpM*1000,3)}/${fmt(frontier.stepDownM*1000,3)} mm`,
    `BIRTH EXPOSURE · slope ${fmt(slopeExposure*100,1)}% of ordinary births · steps ${fmt(stepExposure*100,1)}% of ordinary births; Academy has its own allocation`,
    `SELECTED ${lesson.role.toUpperCase()} · source birth ${fmt(sourceRole.slope)}° · +${fmt(sourceRole.up,3)}/−${fmt(sourceRole.down,3)} mm`,
    `INSPECTOR TARGET (NOT VIEWER COLLIDERS) · ${fmt(lesson.terrain.slopeDeg)}° · +${fmt(lesson.terrain.stepUpM*1000,3)}/−${fmt(lesson.terrain.stepDownM*1000,3)} mm`,
    activeStepTrack==='academy'&&!frontier.academyEligible
      ?'Academy births are NOT currently allocated in the source; this is an illustrative Academy preview.'
      :carpetDifficultyMode.value==='frontier'
      ?'FRONTIER DEMO — unscaled source ceilings, NOT the sampled GPU physical assignment.'
      :carpetMatch.checked
        ?'Representative EXPOSED birth; source RNG chooses which real worlds get terrain. Tile heights still vary by direction and seed.'
        :'CUSTOM VIEWER ONLY — this does not change the running trainer.'
  ].join('\n');
  $('tuneOrigin').textContent=!carpetMatch.checked
    ?`CUSTOM ${lesson.role.toUpperCase()} · Reset sliders restores selected source preview.`
    :carpetDifficultyMode.value==='frontier'
      ?'Unscaled source maximum; this is a diagnostic challenge, NOT the GPU sampling distribution.'
      :`Exposed ${lesson.role.toUpperCase()} birth: slope ${sRatio}% · upward lip ${tRatio}% of source frontier. Unexposed births are flat; exposure does not reduce height.`;
  carpetValues.textContent=[
    `Source: ${carpetDifficulty.source==='training'?carpetDifficulty.run:'Illustrative defaults'}`,
    `Age ${Number(carpetDifficulty.step||0).toLocaleString()} steps · phase: ${phase}`,
    (carpetDifficulty.trainingCarpetPrisms
      ? 'SOURCE: 4 recycled convex-prism MJX tiles. VIEWER: 225 runtime-updated heightfield tiles.'
      : 'VIEWER: infinite coordinates, 225 moving CPU heightfields; not an exact MJX prism exam.'),
    `Training frontier slope ${Number(carpetDifficulty.slopeDeg||0).toFixed(2)}°; specialist target ${lesson.terrain.slopeDeg.toFixed(3)}° (INSPECTOR ONLY)`,
    `Inspector target up/down ${(lesson.terrain.stepUpM*1000).toFixed(3)} / ${(lesson.terrain.stepDownM*1000).toFixed(3)} mm (NOT Meadow terrain)`,
    `Role velocity ${f.speed} · strafe ${f.lateral} · yaw ${f.yaw}`,
    `Push cap ${f.shove} · effective push chance fraction ${lesson.intensity.toFixed(3)}`,
    `Recovery at spawn (different quantity): ${lesson.birthRecovery.toFixed(2)}`,
    'All exercises share ONE trained actor. Browser previews are not',
    'exact samples from the GPU trainer and never change training.'
  ].join('\n');
}
function acceptCarpetSnapshot(data,label){
  carpetDifficulty=trainingLimits(data);
  carpetSource=label;
  renderCarpetValues();
  if(terrainMode.value!=='carpet')return;
  if(carpetAutoApply.checked&&carpetMatch.checked)applyCarpetTerrain(true);
  else carpetStatus.textContent='Read '+label+'; custom preview preserved. Reset sliders to follow the trainer.';
}
async function refreshCarpetLive(manual=false){
  if(carpetFetching||(!manual&&(!carpetLive.checked||
      (terrainMode.value!=='carpet'&&!discoverCarpetFromLive))))return;
  carpetFetching=true;
  try{
    const response=await fetch('/api/live/terrain',{cache:'no-store'});
    if(!response.ok)throw Error(`Live training difficulty HTTP ${response.status}`);
    const data=await response.json();
    if(data.schema!=='emu_terrain_live_v1')throw Error('Unrecognized terrain telemetry format');
    const prev=`${carpetDifficulty.run}:${carpetDifficulty.step}`;
    const incoming=trainingLimits(data);
    if(discoverCarpetFromLive){
      discoverCarpetFromLive=false;
      if(incoming.trainingCarpet){
        terrainMode.value='carpet';updateTerrain();
      }
    }
    if(prev!==`${incoming.run}:${incoming.step}`)acceptCarpetSnapshot(data,'local EMU training viewer');
    else if(manual)carpetStatus.textContent='Live source unchanged: '+prev;
  }catch(err){
    if(manual)carpetStatus.textContent='No local training feed: '+err.message+' · import a CSV/JSON instead.';
  }finally{carpetFetching=false;}
}
function applyCarpetTerrain(automatic=false){
  renderCarpetValues();
  const signature=carpetTerrainSignature();
  if(automatic&&signature===carpetLastAppliedSignature){
    syncCarpetExercise();return;
  }
  if(!sim){carpetStatus.textContent='Saved preview for when a policy loads.';return;}
  try{
    const limits=carpetEffectiveLimits();
    sim.setTrainingCarpet({
      role:carpetRepresentativeRole(limits,carpetRole.value),limits});
    if(terrainMode.value!=='carpet'){
      terrainMode.value='carpet';updateTerrain();
    }
    carpetExerciseClock=0;
    carpetLastAppliedSignature=signature;
    syncCarpetExercise();
    carpetStatus.textContent=(automatic?'Auto-followed':'Applied')+
      ' this specialist exercise. Meadow terrain remains controlled above.';
  }catch(err){setStatus('Training terrain: '+err.message,'error');}
}
function updateTerrain(){
  $('terrainMaxOut').textContent=`${terrainMax.value}°`;
  const isPark=terrainMode.value==='park';
  updateWorldTabs();
  treesToggle.disabled=!isPark;
  parkFeaturesToggle.disabled=!isPark;
  parkChallengeLevel.disabled=!isPark;
  terrainMax.disabled=!isPark;
  renderCarpetValues();
  if(!isPark&&demoActive)stopDemo();
  if(!sim)return;
  try{
    const limits=carpetEffectiveLimits();
    if(!isPark)sim.setTrainingCarpet({
      role:carpetRepresentativeRole(limits,carpetRole.value),limits});
    sim.setTerrainMode(terrainMode.value,Number(terrainMax.value));
    if(isPark&&demoActive)navigator.reset();
    if(!isPark)carpetLastAppliedSignature=carpetTerrainSignature();
    carpetExerciseClock=0;
    syncCarpetExercise();
    updateStickLabels();
    updateParkCourseInfo();
  }catch(err){
    terrainMode.value='park';
    sim.setTerrainMode('park');
    updateWorldTabs();
    setStatus(`Terrain unavailable: ${err.message}`,'error');
  }
}
function runCarpetLesson(dt){
  if(!sim||sim.paused||terrainMode.value!=='carpet'||carpetDrive.value!=='lesson')return;
  const lesson=currentCarpetLesson();
  carpetExerciseClock+=Math.max(0,Math.min(.10,dt));
  const command=lesson.commandAt(carpetExerciseClock);
  sim.setDemoCommands(command);
  const delivered=sim.commandTarget;
  const descriptor=`vx ${fmtSigned(delivered.vx)} · vy ${fmtSigned(delivered.vy)} · yaw ${fmtSigned(delivered.yaw)}`;
  if(descriptor!==carpetLastExercise){
    carpetLastExercise=descriptor;
    $('lessonCommand').textContent='Viewer command: '+descriptor;
  }
}
function updateCarpetShoveStatus(){
  if(!sim||terrainMode.value!=='carpet')return;
  const shove=sim.shoveTelemetry();
  const hz=shove.expectedHz;
  const expected=hz>0?`~${hz.toFixed(hz<.1?3:2)} shove starts/s (1 per ${(1/hz).toFixed(1)}s on average)`:'disabled';
  $('shoveRateNote').textContent=`Stochastic per-robot rate: ${expected}. ${shove.mode==='training'
    ?'Based on the actual GPU per-frame probability and estimated birth-task multiplier.'
    :shove.mode==='inspection'?'Viewer inspection multiplies GPU event odds by 8.'
    :'HIGH-FREQUENCY VIEWER DRILL — not the actual training frequency.'}`;
  $('carpetShoveState').textContent=shove.active
    ? `FORCE APPLIED · ${shove.force[0].toFixed(2)}N X, ${shove.force[1].toFixed(2)}N Y · amber arrow`
    :shove.recent?'Recent shove · muted arrow (force ended)'
    :`Random pushes: ${shove.enabled?'on':'off'} · cap ${shove.limit.toFixed(2)} N · task intensity ${shove.intensity.toFixed(3)}`;
}
carpetRole.addEventListener('change',()=>{
  carpetExerciseClock=0;
  applyCarpetTerrain(false);
});
for(const control of [carpetDifficultyMode,carpetStepTrack]){
  control.addEventListener('change',()=>{
    carpetMatch.checked=true;
    carpetExerciseClock=0;
    applyCarpetTerrain(false);
  });
}
function readSliderOverride(){
  const values=Object.fromEntries(SLIDER_FIELDS.map(([key])=>[key,
    Number(tuningInputs[key].value)]));
  return customRoleOverride(values);
}
for(const [key,id,unit,,digits] of SLIDER_FIELDS){
  const input=tuningInputs[key];
  input.addEventListener('input',()=>{
    $(id+'Out').textContent=Number(input.value).toFixed(digits)+unit;
  });
  input.addEventListener('change',()=>{
    const role=carpetRepresentativeRole(carpetDifficulty,carpetRole.value);
    carpetEdits.set(role,readSliderOverride());
    carpetMatch.checked=false;
    applyCarpetTerrain(false);
  });
}
carpetMatch.addEventListener('change',()=>{
  if(!carpetMatch.checked){
    const role=carpetRepresentativeRole(carpetDifficulty,carpetRole.value);
    if(!carpetEdits.has(role))
      carpetEdits.set(role,customRoleOverride(roleSliders(carpetDifficulty,role)));
  }
  applyCarpetTerrain(false);
});
carpetResetLimits.addEventListener('click',()=>{
  carpetEdits.delete(carpetRepresentativeRole(carpetDifficulty,carpetRole.value));
  carpetMatch.checked=true;
  applyCarpetTerrain(false);
});
carpetDrive.addEventListener('change',()=>{
  carpetExerciseClock=0;
  syncCarpetExercise();
});
carpetShoves.addEventListener('change',syncCarpetExercise);
const meadowPresets=Object.freeze({
  gentle:{hillPeakDeg:2,hillScaleM:12,stepUpM:0,stepDownM:0,
    stepFrequency:0,northGainPerM:.015},
  rolling:{...MEADOW_DEFAULTS},
  rugged:{hillPeakDeg:10,hillScaleM:8.5,stepUpM:.02,stepDownM:.02,
    stepFrequency:.28,northGainPerM:.022},
  extreme:{hillPeakDeg:15,hillScaleM:6.5,stepUpM:.04,stepDownM:.04,
    stepFrequency:.45,northGainPerM:.03}
});
const meadowControls=[
  ['hillPeakDeg','meadowHill',v=>v.toFixed(1)+'°'],
  ['hillScaleM','meadowSize',v=>v.toFixed(1)+' m'],
  ['stepUpM','meadowUp',v=>v.toFixed(1)+' cm',.01],
  ['stepDownM','meadowDown',v=>v.toFixed(1)+' cm',.01],
  ['stepFrequency','meadowSteps',v=>v.toFixed(0)+'%',.01],
  ['northGainPerM','meadowNorth',v=>v.toFixed(1)+'% / m',.01]
];
function renderMeadowControls(){
  for(const [key,id,fmt,conversion=1] of meadowControls){
    $(id).value=String(meadowSettings[key]/conversion);
    $(id+'Out').textContent=fmt(Number($(id).value));
  }
  $('meadowDifficultyInfo').textContent=
    'Natural hills regularly approach '+meadowSettings.hillPeakDeg.toFixed(1)+
    '° at the starting latitude, alternating from gentle valleys to ridges.'+
    ' North difficulty rises '+(meadowSettings.northGainPerM*100).toFixed(1)+
    '% per metre; south is easier; east and west vary the shapes at the same level.';
}
function applyMeadowControls(){
  const next={...meadowSettings};
  for(const [key,id,,conversion=1] of meadowControls)
    next[key]=Number($(id).value)*conversion;
  meadowSettings=normalizedMeadowSettings(next);
  $('meadowPreset').value='custom';
  renderMeadowControls();
  try{
    sim?.setMeadowSettings(meadowSettings);
    if(sim?.terrainMode==='carpet')
      carpetStatus.textContent='New hills applied; viewer robot reset to protect foot contacts.';
    syncCarpetExercise();
  }catch(err){setStatus('Meadow terrain: '+String(err?.message||err),'error');}
}
for(const [key,id,fmt] of meadowControls){
  $(id).addEventListener('input',()=>{$(id+'Out').textContent=fmt(Number($(id).value));});
  $(id).addEventListener('change',applyMeadowControls);
}
$('meadowPreset').addEventListener('change',()=>{
  const preset=meadowPresets[$('meadowPreset').value];
  if(!preset)return; // 'custom' keeps the current physical settings
  meadowSettings=normalizedMeadowSettings({...MEADOW_DEFAULTS,...preset});
  renderMeadowControls();
  try{sim?.setMeadowSettings(meadowSettings);}
  catch(err){setStatus('Meadow terrain: '+String(err?.message||err),'error');}
  syncCarpetExercise();
});
$('meadowReset').addEventListener('click',()=>{
  meadowSettings={...MEADOW_DEFAULTS};
  $('meadowPreset').value='rolling';
  renderMeadowControls();
  sim?.setMeadowSettings(meadowSettings);
  syncCarpetExercise();
});
renderMeadowControls();
$('meadowShove').addEventListener('input',()=>{
  $('meadowShoveOut').textContent=$('meadowShove').value+'%';
});
$('meadowShove').addEventListener('change',syncCarpetExercise);
carpetPushRate.addEventListener('change',syncCarpetExercise);
carpetTestShove.addEventListener('click',()=>{
  if(sim&&terrainMode.value==='carpet'&&!sim.triggerShove())
    $('carpetShoveState').textContent='Increase Maximum shove above 0 N to test.';
});
carpetAutoApply.addEventListener('change',()=>{
  if(carpetAutoApply.checked&&terrainMode.value==='carpet'&&carpetMatch.checked)
    applyCarpetTerrain(true);
});
carpetApply.addEventListener('click',()=>applyCarpetTerrain(false));
carpetFetch.addEventListener('click',()=>refreshCarpetLive(true));
carpetImport.addEventListener('change',async()=>{
  const file=carpetImport.files?.[0];if(!file)return;
  try{
    const text=await file.text();
    const obj=file.name.toLowerCase().endsWith('.csv')?parseCarpetMetrics(text):JSON.parse(text);
    acceptCarpetSnapshot(obj,file.name);
  }catch(error){carpetStatus.textContent='Import failed: '+error.message;}
  carpetImport.value='';
});
setInterval(()=>refreshCarpetLive(false),5000);
renderCarpetValues();

function updateDepthSettings(){
  const c=bundle?.observation?.sensor_contract;
  $('actorDepthSource').textContent=c?.tof4x4_rays_body?.length===16
    ? 'Policy rays: physical 45°×45° VL53L8CX, 70° down and -30mm below torso (magenta). Green: adjustable diagnostic preview.'
    : 'Load the current h40 policy or sensor fixture to inspect its 16 physical rays.';
  $('depthTiltOut').textContent=depthTilt.value+'°';
  $('depthMountOut').textContent=depthMount.value+' mm';
  $('depthHeightOut').textContent=(Number(depthHeight.value)>=0?'+':'')+depthHeight.value+' mm';
  $('depthLateralOut').textContent=(Number(depthLateral.value)>=0?'+':'')+depthLateral.value+' mm';
  $('depthYawOut').textContent=(Number(depthYaw.value)>=0?'+':'')+depthYaw.value+'°';
  if(sim)sim.setDepthInspector({
    enabled:depthOn.checked,tilt:Number(depthTilt.value),
    forward:Number(depthMount.value)/1000,lateral:Number(depthLateral.value)/1000,
    height:Number(depthHeight.value)/1000,yaw:Number(depthYaw.value),
    showRays:depthRays.checked,
    showActorRays:depthActorRays.checked,
    showFootprint:depthFootprint.checked,
    subrays:depthSamples.value==='auto'?'auto':Number(depthSamples.value),
  });
}
for(const input of [depthOn,depthTilt,depthMount,depthHeight,depthLateral,depthYaw,depthRays,depthActorRays,depthFootprint,depthSamples])
  input.addEventListener('input',updateDepthSettings);
$('depthFootPreset').addEventListener('click',()=>{
  depthTilt.value='70';depthMount.value='0';depthHeight.value='-30';
  depthLateral.value='0';depthYaw.value='0';
  depthSamples.value='5';
  depthOn.checked=true;
  updateDepthSettings();
});
depthView.addEventListener('change',updateDepthGrid);
$('depthCopy').addEventListener('click',async()=>{
  const reading=sim?.depthReading;
  if(!reading){$('depthNote').textContent='No scan to copy yet';return;}
  const terrainHeightCm=terrainHeightOffsetsCm(reading);
  const info={
    sensor:'VL53L8CX 4x4 geometric preview; not hardware capture',
    policy:bundle?.source?.run_name||'unknown',
    terrain:terrainMode.value,
    obstacle:'current viewer terrain',
    tilt_down_deg:Number(depthTilt.value),
    mount_forward_mm:Number(depthMount.value),
    mount_vertical_mm:Number(depthHeight.value),
    mount_lateral_mm:Number(depthLateral.value),
    mount_yaw_deg:Number(depthYaw.value),
    valid_forward_reach_cm:reading.reachNearM==null?null:[
      Number((reading.reachNearM*100).toFixed(1)), Number((reading.reachFarM*100).toFixed(1))],
    depth_sample_rays:reading.subrays*16,
    trained_actor_pitch_rows_deg:bundle?.observation?.sensor_contract?.tof4x4_pitch_from_horizontal_deg,
    trained_actor_mount_body_m:bundle?.observation?.sensor_contract?.tof4x4_mount_body_m,
    trained_actor_rays_body:bundle?.observation?.sensor_contract?.tof4x4_rays_body,
    trained_actor_yaw_columns_deg:bundle?.observation?.sensor_contract?.tof4x4_yaw_deg,
    trained_actor_rays_visible:depthActorRays.checked,
    visual_footprint_rays:reading.footprintRays||0,
    scan_time_ms:Number(reading.costMs.toFixed(2)),
    displayed_as:depthView.value==='height'?'signed terrain elevation (cm)':'slant range (cm)',
    height_reference:'horizontal world plane at z=0 (visual diagnostic, not actor input)',
    height_colours_saturate_at_cm:15,
    terrain_height_cm:Array.from({length:4},(_,row)=>
      Array.from({length:4},(_,col)=>{
        const z=terrainHeightCm[row*4+col];
        return z===null?null:Number(z.toFixed(1));
      })),
    ranges_cm:Array.from({length:4},(_,row)=>
      Array.from({length:4},(_,col)=>{
        const i=row*4+col;
        return reading.valid[i]?Number((100*reading.ranges[i]).toFixed(1)):null;
      })),
    self_occluded_zones:reading.selfOccluded.map((v,i)=>v?i:null).filter(v=>v!==null),
    foot_occluded_zones:reading.footOccluded?.map((v,i)=>v?i:null).filter(v=>v!==null),
    bottom_row_foot_hits:reading.footHitCounts?.slice(12,16),
    mixed_edge_zones:reading.mixed.map((v,i)=>v?i:null).filter(v=>v!==null),
  };
  try {
    if(!window.navigator.clipboard?.writeText)throw new Error('Clipboard unavailable');
    await window.navigator.clipboard.writeText(JSON.stringify(info,null,2));
    $('depthNote').textContent='Copied sensor reading · paste it into chat';
  }catch(error){
    $('depthNote').textContent='Clipboard blocked · take a screenshot instead';
  }
});

function signedHeightText(cm){
  if(Math.abs(cm)<.05)return '0.0';
  return (cm>0?'+':'')+cm.toFixed(1);
}
function heightColor(cm){
  // Fixed +/-15cm colour scale. Actual height values remain unclipped.
  const strength=Math.min(1,Math.abs(cm)/15);
  if(Math.abs(cm)<.3)return 'hsl(170 32% 28%)'; // expected flat ground
  if(cm>0)return 'hsl('+(34-20*strength).toFixed(0)+' 68% '+(30+9*strength).toFixed(0)+'%)';
  return 'hsl('+(205+12*strength).toFixed(0)+' 65% '+(30+10*strength).toFixed(0)+'%)';
}
function updateDepthGrid(){
  const data=sim?.depthReading;
  const mode=depthView.value;
  const heights=data?terrainHeightOffsetsCm(data):null;
  let hits=0;
  depthCells.forEach((cell,i)=>{
    const valid=Boolean(data?.valid[i]);
    const blocked=valid&&data.selfOccluded[i];
    const footBlocked=Boolean(data?.footOccluded?.[i]);
    const height=heights?.[i]??null;
    cell.className=valid?'depth-cell':'depth-cell invalid';
    cell.removeAttribute('title');
    if(blocked){
      // Seeing the robot is not evidence of an elevated piece of terrain.
      cell.className+=' occluded';
      cell.textContent=footBlocked?'FOOT':'SELF';
      cell.style.background=footBlocked?'#865c19':'#855524';
      cell.title=footBlocked
        ? 'Foot blocks '+(data.footHitCounts?.[i]||0)+' of '+data.subrays+
          ' rays in this zone; do not interpret it as ground'
        : 'Robot part occludes this zone · no terrain height';
    }else if(valid){
      hits++;
      const cm=data.ranges[i]*100;
      cell.textContent=mode==='height'
        ?(height===null?'—':signedHeightText(height)):cm.toFixed(1);
      cell.title=(height===null?'Terrain unknown':signedHeightText(height)+' cm above flat z=0')+
        ' · raw '+cm.toFixed(1)+' cm';
      if(data.mixed[i]){
        // A yellow outline warns of a discontinuity without hiding the
        // sign of the measured height inside the cell.
        cell.className+=' mixed';
        cell.title+=' · mixed edge, interpret cautiously';
      }
      if(mode==='height'){
        cell.style.background=height===null?'':heightColor(height);
      }else{
        // Raw physical slant range remains available and unmodified.
        const fraction=Math.max(0,Math.min(1,(cm-5)/145));
        cell.style.background='hsl('+(174+fraction*52).toFixed(0)+' 42% '+(29+fraction*11).toFixed(0)+'%)';
      }
    }else{
      cell.textContent='—';
      cell.style.background='';
      cell.title='No valid terrain return';
    }
  });
  $('depthGrid').setAttribute('aria-label',mode==='height'
    ? 'Live 4 by 4 signed centimetres relative to horizontal flat ground z equals zero'
    : 'Live 4 by 4 raw line-of-sight range in centimetres');
  $('depthNote').textContent=!depthOn.checked
    ? 'Diagnostic off · champion unchanged'
    : !data?'Waiting for robot · no scan yet'
    : hits+'/16 visible terrain zones · '+(data.subrays*16)+' depth rays'+
      (data.footprintRays?' + '+data.footprintRays+' grid':'')+' · '+data.costMs.toFixed(1)+
      ' ms · '+(data.reachNearM==null?'no ground reach':
      'forward hit footprint '+(100*data.reachNearM).toFixed(0)+'–'+
      (100*data.reachFarM).toFixed(0)+' cm')+
      ' · '+(data.footOccluded
        ?(data.footOccluded.slice(12,16).join(',')==='false,true,true,false'
          ? '✓ bottom row: ground / FOOT / FOOT / ground'
          : 'bottom row foot masks: '+data.footOccluded.slice(12,16).map(v=>v?'F':'·').join(''))
        :'')+
      ' · '+(mode==='height'?'signed cm from flat z=0 (colour ±15 cm)':'raw slant cm');
}

// World switches are the two explicit Field / Carpet buttons above.
terrainMax.addEventListener('change',updateTerrain);
treesToggle.addEventListener('change',()=>{if(sim)try{sim.setTreesEnabled(treesToggle.checked);if(goalDemoActive())navigator.reset();}catch(err){setStatus(`Tree collision unavailable: ${err.message}`,'error');}});
parkFeaturesToggle.addEventListener('change',()=>{if(sim)try{
  sim.setParkFeaturesEnabled(parkFeaturesToggle.checked);
  if(goalDemoActive())navigator.reset();
  updateParkCourseInfo();
}catch(err){setStatus(`Park course unavailable: ${err.message}`,'error');}});
parkChallengeLevel.addEventListener('change',()=>{if(sim)try{
  sim.setParkChallengeLevel(parkChallengeLevel.value);
  if(goalDemoActive())navigator.reset();
  updateParkCourseInfo();
}catch(err){setStatus(`Park difficulty unavailable: ${err.message}`,'error');}});
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
    if(terrainMode.value==='carpet'&&carpetDrive.value==='lesson'){
      carpetDrive.value='manual';syncCarpetExercise();
    }
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
  const worldY=Number(sim.data?.qpos?.[1]??0);
  const local=sim.park?.physics==='streaming-heightfield'
    ?sim.park.difficultyAt(worldY):null;
  $('carpetGrowth').textContent=local
    ? Math.round(100*local.difficulty)+'% difficulty · '+
      local.slopeDeg.toFixed(1)+'° hill peaks · '+
      Math.abs(worldY).toFixed(1)+'m '+(worldY>=0?'N':'S')
    :'—';
  if(local)$('carpetHudTerrain').textContent=
    local.slopeDeg.toFixed(1)+'° natural hill peaks · +'+
    Math.round(local.stepUpM*1000)+'/−'+Math.round(local.stepDownM*1000)+
    ' mm max';
  $('slopeArrow').hidden = !s.supported || s.flat;
  $('slopeArrow').style.transform = `rotate(${s.arrowDegrees.toFixed(1)}deg)`;
  $('leanRead').textContent = `${fmtSigned(deg(t.robotLean),1)}°`;
  $('pitchRead').textContent = `${fmtSigned(deg(t.pitch),1)}° / ${fmtSigned(deg(t.cmd.pitch),1)}°`;
  updateDepthGrid();
  updateCarpetShoveStatus();
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
    if (!demoActive && terrainMode.value==='park' && !navigationFault && handoff.shouldResume(now,
       held.size>0 || leftStick.pointerId!==null || rightStick.pointerId!==null ||
       Math.hypot(left.x,left.y)>.06 || Math.hypot(right.x,right.y)>.06)) {
      startDemo();
    }
    if (demoActive && terrainMode.value==='park' && !sim.paused) {
      try{navigationUpdate(dt);}
      catch(error){
        console.error('Goal navigation:',error);
        navigationFault=true;
        stopDemo();
        sim.setDemoCommands({vx:0,vy:0,yaw:0,pitch:0});
        setStatus('Navigation error: '+String(error?.message||error), 'error');
      }
    }
    runCarpetLesson(dt);
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
  if(terrainMode.value==='carpet'){
    $('liveDemo').checked=false;
    setStatus('Autonomous navigation is for Field mode; Carpet uses specialist exercises.','warn');
    return;
  }
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
          if(config.sensor_lab){
            // Standalone CPU calibration. The model is h40 and the actor
            // inference is DISABLED; do not mislabel this a trained gait.
            terrainMode.value='carpet';
            carpetRole.value='standing';
            carpetDifficulty={...CARPET_DEFAULT,slopeDeg:0,transitionDeg:0,
              stepUpM:0,stepDownM:0};
            demoActive=false;
            $('liveDemo').checked=false;
            autoResetToggle.checked=true;
            carpetDrive.value='manual';
            carpetShoves.checked=false;
            depthOn.checked=true;depthTilt.value='70';
            depthMount.value='0';depthHeight.value='-30';
            depthLateral.value='0';depthYaw.value='0';
            depthSamples.value='5';
            depthActorRays.checked=true;
            depthRays.checked=true;
            depthFootprint.checked=true;
            $('sensorDetails').open=true;
          }
          const file = await fetch('/api/initial-policy', {cache:'no-store'});
          if (file.ok) {
            const parsed = parsePolicy(await file.text());
            policyLibrary.push({fileName:config.sensor_lab?'H40 SENSOR LAB — UNTRAINED':'Preloaded policy',bundle:parsed});
            await activatePolicy(policyLibrary.length-1);
            if(config.sensor_lab){
              // Start in neutral calibration pose. User can click Run
              // physics; there is no h40-trained policy on this computer.
              sim.paused=true;
              pauseBtn.textContent='Run physics';
              livePanel.hidden=true;
              updateWorldTabs();
              updateDepthSettings();
              $('modeLabel').textContent='40 cm CPU SENSOR LAB — no trained gait';
              setStatus('40cm h40 · 4×1m / 8 real CPU prisms · neural actor OFF','ok');
              $('depthNote').textContent=
                'Mount: 70° down, 0mm forward, -30mm below torso, zero yaw/sideways. Scan actual feet and terrain to confirm your zone pattern.';
            }
          }
        }
        if (config.live) {
          discoverCarpetFromLive=true;
          // Current real-prism sources auto-enter the infinite-coordinate CPU Meadow.
          // Its recycled heightfields are NOT the MJX convex-prism teacher.
          // The frozen champion and older runs keep the
          // original Terrain Park until the user explicitly selects it.
          await refreshCarpetLive(true);
          if(carpetDifficulty.trainingCarpet){
            terrainMode.value='carpet';
            updateTerrain();
          }
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
startViewerFeed();
requestAnimationFrame(frame);
