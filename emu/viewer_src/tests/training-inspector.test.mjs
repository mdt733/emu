import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {trainingLimits} from '../src/training-carpet.js';
import {specialistLesson,friendlyFrontiers} from '../src/training-inspector.js';

const snapshot=(step=34_000_000)=>trainingLimits({
  training_step:step,run_name:'test-live',
  latest_metrics:{step,frontier_vx_hi:'.28',frontier_vx_lo:'-.07',
    frontier_vy:'.12',frontier_yaw:'0.7',frontier_push_n:'.65',
    resilience_difficulty:'.36',slope_max_deg:'.7',
    terrain_height_m:'.004',academy_height_up_m:'.003',
    academy_height_down_m:'.002',
    slope_exposure:'1',terrain_exposure:'1'},
  training_config:{terrain_carpet:true,academy_stream:true,
    seed:735,foundation_steps:20000000,
    academy_intro_steps:60000000,
    push_max_force_n:0.7}
});

test('measured speed, shove, slope and both lip ceilings reach every exercise',()=>{
  const data=snapshot();
  assert.equal(data.vxHi,.28);
  assert.equal(data.vxLo,-.07);
  assert.equal(data.vyMax,.12);
  assert.equal(data.yawMax,.7);
  assert.equal(data.pushN,.65);
  assert.equal(data.pushMaxForceN,.7);
  const step=specialistLesson(data,'step');
  assert.equal(step.role,'step');
  assert.equal(step.terrain.slopeDeg,.35);
  assert.equal(step.terrain.stepUpM,.004); // Ordinary max(terrain, Academy), before Academy births
  assert.equal(friendlyFrontiers(step).speed,'[-0.07, +0.28] m/s');
  assert.ok(step.commandAt(1).vx>0);
  assert.ok(step.commandAt(1).vx<=.22);
});

test('roles have different representative conditions and command tasks',()=>{
  const data=snapshot();
  const roles=['command','resilience','slope','step','transition','control'];
  const lessons=roles.map(role=>specialistLesson(data,role));
  assert.deepEqual(lessons.map(x=>x.role),roles);
  assert.equal(lessons.find(x=>x.role==='slope').terrain.slopeDeg,.7);
  assert.ok(lessons.find(x=>x.role==='control').terrain.slopeDeg<
            lessons.find(x=>x.role==='slope').terrain.slopeDeg);
  assert.ok(lessons.find(x=>x.role==='command').commandAt(21).vy>0);
  assert.ok(lessons.find(x=>x.role==='transition').commandAt(21).vx<0);
  assert.equal(lessons.find(x=>x.role==='resilience').caps.push,.65);
});

test('flat foundation stays flat while command and push preview remain informative',()=>{
  const lesson=specialistLesson(snapshot(9_500_000),'step');
  assert.equal(lesson.role,'step');
  assert.equal(lesson.terrain.slopeDeg,0);
  assert.equal(lesson.terrain.stepUpM,0);
  assert.equal(lesson.terrain.stepDownM,0);
});

test('one world switch; sensor lab is not a floating second settings UI',()=>{
  const ui=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const css=readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  assert.match(ui,/id="worldField"/);
  assert.match(ui,/id="worldCarpet"/);
  assert.match(ui,/id="sensorDetails"/);
  assert.ok(ui.indexOf('id="settingsDrawer"')<ui.indexOf('id="depthPanel"'));
  assert.match(ui,/id="carpetRole"/);
  assert.match(ui,/id="carpetMatch"/);
  assert.match(ui,/id="carpetShoves"/);
  assert.match(ui,/id="carpetDrive"/);
  assert.match(main,/runCarpetLesson\(dt\)/);
  assert.match(main,/carpetDrive\.value==='manual'/);
  assert.match(sim,/const xfrc=this\.data\.xfrc_applied/);
  assert.match(sim,/xfrc\[offset\]=v\.last\[0\]/);
  assert.match(sim,/this\.shoveArrow\.setDirection\(direction\)/);
  assert.match(css,/\.settings-drawer \.depth-panel/);
});

test('settings begin with specialist, then editable effective role weights',()=>{
  const ui=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.ok(ui.indexOf('id="carpetPanel"')<ui.indexOf('id="viewPreferences"'));
  assert.ok(ui.indexOf('id="carpetRole"')<ui.indexOf('id="tuneSlope"'));
  for(const id of ['tuneSlope','tuneTransition','tuneUp','tuneDown',
                   'tuneForward','tuneReverse','tuneLateral','tuneYaw',
                   'tuneResilience','tunePush']){
    assert.match(ui,new RegExp('id="'+id+'"'));
  }
  assert.doesNotMatch(ui,/id="carpetQuick"/);
  assert.doesNotMatch(ui,/id="depthCourse"/);
  assert.doesNotMatch(ui,/id="surfaceSlope"/);
  assert.doesNotMatch(ui,/id="surfaceStep"/);
  assert.doesNotMatch(ui,/v5023 surface laboratory/i);
  assert.match(main,/carpetEdits\.set\(role,readSliderOverride\(\)\)/);
  assert.match(main,/carpetMatch\.checked=false/);
  assert.match(main,/carpetResetLimits\.addEventListener/);
  assert.match(main,/navHud\.hidden=!enabled/);
  assert.match(main,/goalDemoActive\(\) && terrainMode\.value==='park'/);
});
