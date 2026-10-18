// Real Chromium/MuJoCo/WASM smoke, not a mocked navigator state machine.
// Run after preparing the packaged v5016 policy and vite build:
//   npm install --no-save --package-lock=false --ignore-scripts playwright-core
//   node scripts/e2e-goal-smoke.mjs
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {existsSync} from 'node:fs';
import {chromium} from 'playwright-core';

const executable=[
  process.env.CHROME_BIN,'/usr/bin/google-chrome','/usr/bin/chromium',
  '/usr/bin/google-chrome-stable','/opt/google/chrome/chrome'
].find(x=>x&&existsSync(x));
if(!executable)throw new Error('Chrome/Chromium executable is required for the real viewer smoke test');
const port=8767;
const vite=spawn('node',['node_modules/vite/bin/vite.js','preview',
  '--host','127.0.0.1','--port',String(port),'--strictPort'],
  {cwd:process.cwd(),stdio:['ignore','pipe','pipe']});
let output='';
for(const stream of [vite.stdout,vite.stderr])stream.on('data',d=>{output+=d.toString();});
let browser;
function delay(n){return new Promise(r=>setTimeout(r,n));}
async function waitServer(){
  for(let i=0;i<100;i++){
    if(vite.exitCode!==null)throw new Error('Vite exited: '+output.slice(-3000));
    try{
      const r=await fetch('http://127.0.0.1:'+port+'/');
      if(r.ok)return;
    }catch{}
    await delay(160);
  }
  throw new Error('Vite preview failed to start: '+output.slice(-3000));
}
function frames(text){
  const m=/actor frames?\s+(\d+)/i.exec(text);
  return m?Number(m[1]):NaN;
}
function jointTravel(text){
  const m=/joints\s+([\d.]+)\s+rad/i.exec(text);
  return m?Number(m[1]):NaN;
}
try {
  await waitServer();
  browser=await chromium.launch({executablePath:executable,headless:true,
    args:['--no-sandbox','--disable-dev-shm-usage',
      '--use-gl=angle','--use-angle=swiftshader',
      '--enable-webgl','--ignore-gpu-blocklist']});
  const page=await browser.newPage({viewport:{width:1280,height:820}});
  const errors=[];
  page.on('pageerror',e=>errors.push('page: '+e.message));
  page.on('console',msg=>{if(msg.type()==='error')errors.push('console: '+msg.text());});
  await page.goto('http://127.0.0.1:'+port+'/',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.querySelector('#policyStep')?.textContent.includes('260,112,384'),
    null,{timeout:85000});
  // Use public UI only: no private globals or test-only changes to model.
  const select=page.locator('#demoMode');
  assert.equal(await select.inputValue(),'commands','known-good random mode first');
  await page.waitForTimeout(2200);
  assert.equal(await page.locator('#status').getAttribute('data-kind'),'ok',
    'champion must load without viewer errors');
  await page.locator('#settingsOpen').click();
  await page.locator('#demoMode').waitFor({state:'visible'});
  await select.selectOption('goals');
  await page.waitForTimeout(2000);
  assert.equal(await select.inputValue(),'goals',
    'goal selection must not silently fall back after error');
  const a=await page.locator('#navEstimate').textContent();
  await page.waitForTimeout(3800);
  const b=await page.locator('#navEstimate').textContent();
  const f0=frames(a),f1=frames(b);
  assert.ok(Number.isFinite(f0)&&Number.isFinite(f1),
    'actor inference counter missing from goal mission HUD: '+a+' / '+b);
  assert.ok(f1-f0>10,'the model stopped evaluating in goal mode: '+f0+' -> '+f1);
  const j0=jointTravel(a),j1=jointTravel(b);
  assert.ok(Number.isFinite(j0)&&Number.isFinite(j1),
    'joint movement instrumentation missing: '+a+' / '+b);
  assert.ok(j1-j0>.05,'the ACTUAL servo joints did not move in goal mode: '+j0+' -> '+j1);
  assert.equal(await page.locator('#status').getAttribute('data-kind'),'ok',
    'goal mode entered an error state: '+await page.locator('#status').textContent());
  assert.equal(await select.inputValue(),'goals','goal mode crashed back to random');
  const alternate=[];
  // Prove BOTH experimental backends can be selected through the real UI
  // without interrupting the SAME 260M PPO actor and MuJoCo motor motion.
  for(const planner of ['curves','arcs']){
    await page.locator('#navPlanner').selectOption(planner);
    await page.waitForTimeout(1600);
    const before=await page.locator('#navEstimate').textContent();
    await page.waitForTimeout(3400);
    const after=await page.locator('#navEstimate').textContent();
    const v0=frames(before),v1=frames(after);
    const r0=jointTravel(before),r1=jointTravel(after);
    assert.ok(v1-v0>5,planner+' froze neural inference: '+before+' -> '+after);
    assert.ok(r1-r0>.03,planner+' froze servo joints: '+before+' -> '+after);
    assert.doesNotMatch(after,/planner error/i,
      planner+' had a background planner exception: '+after);
    assert.equal(await select.inputValue(),'goals',
      planner+' unexpectedly switched off goal mode');
    alternate.push({planner,frames:v1-v0,jointsRad:r1-r0,status:after});
  }
  const report={source:'actual Chromium WebGL+WASM',
    actorFramesBefore:f0,actorFramesAfter:f1,delta:f1-f0,
    jointsBefore:j0,jointsAfter:j1,jointMovementRad:j1-j0,
    alternate,navBefore:a,navAfter:b,errors};
  console.log('EMU GOAL NAVIGATION E2E: '+JSON.stringify(report));
  if(errors.length)console.warn('Browser logged errors: '+errors.join('; '));
  // Do not call a successful route "completed": this verifies that the
  // trained actor is alive when mission planner/visualisation is active.
}finally{
  if(browser)await browser.close();
  vite.kill('SIGTERM');
  if(vite.exitCode===null)await Promise.race([once(vite,'exit'),delay(1500)]);
}
