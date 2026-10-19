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
  // The champion should begin in pose-lattice autonomous navigation.
  assert.equal(await page.locator('#liveDemo').isChecked(),true);
  assert.equal(await page.locator('#demoMode').count(),0);
  assert.equal(await page.locator('#navPlanner').count(),0);
  await page.waitForTimeout(2300);
  assert.equal(await page.locator('#status').getAttribute('data-kind'),'ok');
  const a=await page.locator('#navEstimate').textContent();
  await page.waitForTimeout(3800);
  const b=await page.locator('#navEstimate').textContent();
  const f0=frames(a),f1=frames(b),j0=jointTravel(a),j1=jointTravel(b);
  assert.ok(Number.isFinite(f0)&&Number.isFinite(f1)&&f1-f0>10,
    'PPO actor must continue under navigation: '+a+' / '+b);
  assert.ok(Number.isFinite(j0)&&Number.isFinite(j1)&&j1-j0>.05,
    'MuJoCo motors must move in navigation mode: '+a+' / '+b);
  const automaticStick=await page.locator('#leftStick .nub').evaluate(
    node=>node.style.transform);
  assert.ok(automaticStick.includes('translate'));
  // Real keyboard control takes over without another mode selector.
  await page.keyboard.down('w');
  await page.waitForTimeout(500);
  assert.equal(await page.locator('#liveDemo').isChecked(),false);
  assert.match(await page.locator('#navHudTitle').textContent(),/MANUAL/);
  await page.keyboard.up('w');
  const report={source:'actual Chromium WebGL+WASM',
    actorFramesBefore:f0,actorFramesAfter:f1,delta:f1-f0,
    jointsBefore:j0,jointsAfter:j1,jointMovementRad:j1-j0,
    automaticStick,manualTakeover:true,navBefore:a,navAfter:b,errors};
  console.log('EMU GOAL NAVIGATION E2E: '+JSON.stringify(report));
  if(errors.length)console.warn('Browser logged errors: '+errors.join('; '));
  // Does not claim full mission completion or test the 30-second idle timeout.
}finally{
  if(browser)await browser.close();
  vite.kill('SIGTERM');
  if(vite.exitCode===null)await Promise.race([once(vite,'exit'),delay(1500)]);
}
