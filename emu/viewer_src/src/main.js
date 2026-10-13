import './style.css';
import { parsePolicy } from './policy.js';
import { VirtualStick } from './joystick.js';
import { EmuBrowserSim } from './sim.js';
import { DemoCommander } from './live-demo.js';

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
const pitchNotice = $('pitchNotice');
const dropOverlay = $('dropOverlay');
const leftValues = $('leftValues');
const rightValues = $('rightValues');
const livePanel = $('livePanel');
const demo = new DemoCommander();
let demoActive = false;
let liveRevision = null;
let livePolling = false;

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
  return bundle?.control?.command_limits || {vx_min:-0.2,vx_max:0.6,vy_abs:0.22,yaw_abs:1.5,pitch_abs_rad:0.08};
}

function updateStickLabels() {
  const l = limits();
  const vx = left.y >= 0 ? left.y*l.vx_max : (-left.y)*l.vx_min;
  const vy = left.x*l.vy_abs;
  const yaw = -right.x*l.yaw_abs;
  const pitch = right.y*l.pitch_abs_rad;
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
    sim.follow = followToggle.checked;
    sim.autoReset = autoResetToggle.checked || demoActive;
    sim.sensorNoise = noiseToggle.checked;
    applyAdvanced();
    updateStickLabels();

    const src = bundle.source || {};
    $('policyName').textContent = src.run_name || entry.fileName;
    $('policyStep').textContent = Number(src.step || 0).toLocaleString();
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
    demo.reset();
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
followToggle.addEventListener('change', () => { if (sim) sim.follow = followToggle.checked; });
autoResetToggle.addEventListener('change', () => { if (sim) sim.autoReset = autoResetToggle.checked || demoActive; });
noiseToggle.addEventListener('change', () => { if (sim) sim.sensorNoise = noiseToggle.checked; });

window.addEventListener('resize', () => sim?.resize());
window.addEventListener('beforeunload', () => sim?.dispose());

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
  if (['INPUT','BUTTON','SUMMARY','SELECT'].includes(document.activeElement?.tagName)) return;
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
  $('pitchRead').textContent = `${fmtSigned(deg(t.pitch),1)}° / ${fmtSigned(deg(t.cmd.pitch),1)}°`;
}

function frame(ms) {
  const now = ms / 1000;
  const dt = Math.max(0, Math.min(0.1, now - lastFrame));
  lastFrame = now;
  if (dt > 0) {
    const fps = 1/dt;
    fpsEma = fpsEma ? 0.92*fpsEma + 0.08*fps : fps;
  }
  if (sim) {
    if (demoActive && !sim.paused) {
      demo.tick(dt, limits(), cmd => sim.setDemoCommands(cmd), bundle?.demo?.node_candidates || []);
      $('demoKind').textContent = demo.kind;
    }
    sim.advance(dt);
    sim.render();
    updateTelemetry(now, dt);
  }
  requestAnimationFrame(frame);
}

function stopDemo() {
  demoActive = false;
  $('liveDemo').checked = false;
  demo.reset();
  $('demoKind').textContent = 'manual control';
  if (sim) {
    sim.autoReset = autoResetToggle.checked;
    sim.setJoystickTargets(left.x, left.y, right.x, right.y);
  }
}

$('liveDemo').addEventListener('change', () => {
  demoActive = $('liveDemo').checked;
  demo.reset();
  if (sim) {
    sim.autoReset = autoResetToggle.checked || demoActive;
    if (!demoActive) sim.setJoystickTargets(left.x, left.y, right.x, right.y);
  }
  $('demoKind').textContent = demoActive ? 'choosing command…' : 'manual control';
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

async function startViewerFeed() {
  try {
    const response = await fetch('/api/viewer-config', {cache:'no-store'});
    if (!response.ok) return; // Works as a plain standalone Vite viewer.
    const config = await response.json();
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
      await pollLive();
      setInterval(pollLive, 12000);
    }
  } catch (err) { console.warn('Viewer feed not configured:', err); }
}

applyAdvanced();
updateStickLabels();
startViewerFeed();
requestAnimationFrame(frame);
