import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {withTerrain} from '../src/terrain.js';
import {carpetRoleLimits,parseCarpetMetrics,trainingLimits} from '../src/training-carpet.js';

test('real EMU CSV exporter events containing commas survive input parsing',()=>{
  const data=parseCarpetMetrics(
    'step,academy_event,slope_max_deg,academy_height_up_m,academy_height_down_m\n'+
    '62000000,"up:collect,down:collect",5,.018,.027\n'+
    '85000000,"up:expand,down:collect",9,.024,.038\n');
  const frontier=trainingLimits(data);
  assert.equal(frontier.step,85000000);
  assert.equal(frontier.slopeDeg,9);
  assert.equal(frontier.stepUpM,.024);
  assert.equal(frontier.stepDownM,.038);
  assert.equal(frontier.transitionMeasured,false);
  assert.equal(data.latest_metrics.academy_event,'up:expand,down:collect');
});
test('specialist roles use shared generated collision but distinct limits',()=>{
  const b={seed:735,slopeDeg:8,transitionDeg:3,stepUpM:.02,stepDownM:.04};
  const up=carpetRoleLimits(b,'slope'),steps=carpetRoleLimits(b,'step');
  assert.equal(steps.slopeDeg,4);
  assert.equal(up.stepUpM,.01);
  assert.equal(steps.stepUpM,.02);
  assert.equal(steps.stepDownM,.04);
  assert.throws(()=>carpetRoleLimits(b,'unknown'),/Unknown/);

});
test('viewer integrates carpet into real MuJoCo, not detached lab',()=>{
  const source=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  const ui=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(source,/createSolidCarpet\(this\.carpetDifficulty,this\.carpetRole/);
  assert.match(source,/this\.parkSurface=createParkSurface/); // unchanged Terrain Park
  assert.match(source,/this\.recycleCarpetWindow\(\)/);
  assert.match(source,/this\.writeCarpetMocap\(/);
  assert.match(source,/this\.park\.sampleHeight\(x,y\)/);
  assert.match(source,/updateCarpetTileOverlay\(\)/);
  assert.match(ui,/value="carpet"/);
  assert.match(ui,/id="carpetImport"/);
  assert.match(ui,/id="carpetApply"/);
  assert.match(main,/\/api\/live\/terrain/);
});
test('retired heightfield carpet fails closed: only 9 real browser contact solids',()=>{
  const xml='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="50 50 .1"/></worldbody></mujoco>';
  assert.throws(()=>withTerrain(xml,'carpet',{park:{mode:'carpet',
     physics:'hfield',heights:new Float32Array(9)}}),/Carpet requires explicit real browser solid-box contacts/);
});
