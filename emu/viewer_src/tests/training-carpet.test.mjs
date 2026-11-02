import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {generateTerrainPark,withTerrain,surfaceSlope} from '../src/terrain.js';
import {CARPET_DEFAULT,carpetRoleLimits,generateTrainingCarpet,parseCarpetMetrics,trainingLimits} from '../src/training-carpet.js';

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
test('standing uses unchanged physical engine without accidental obstacles',()=>{
  const carpet=generateTrainingCarpet(CARPET_DEFAULT,'standing',{trees:false});
  assert.equal(carpet.features.length,0);
  assert.ok(carpet.heights.every(v=>v===0));
  assert.equal(carpet.sampleHeight(0,0),0);
  assert.ok(carpet.preview);
});
test('new carpet is separate from original eight-feature park',()=>{
  const carpet=generateTrainingCarpet({...CARPET_DEFAULT,seed:761},'step',{trees:true});
  const original=generateTerrainPark({features:true,trees:true});
  assert.equal(original.features.length,8);
  assert.equal(carpet.features.length,0);
  assert.ok(carpet.trees.length>0);
  assert.equal(carpet.resolution,177);
  assert.equal(carpet.tileSizeM,.5);
  assert.ok(carpet.heights.some(x=>x>0));
  assert.ok(Math.abs(carpet.sampleHeight(0,0))<.02);
  const xml='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="50 50 .1"/></worldbody></mujoco>';
  const ground=withTerrain(xml,'carpet',{park:carpet});
  assert.match(ground,/name="viewer_park_ground" type="hfield"/);
  assert.match(ground,/name="viewer_tree_trunk_/);
  assert.doesNotMatch(ground,/park_feature_crest/);
  assert.match(ground,/name="viewer_park_hfield"/);
  const uphill=surfaceSlope('carpet',carpet,1.1,.5,0);
  assert.equal(uphill.supported,true);
});
test('specialist roles use shared generated collision but distinct limits',()=>{
  const b={seed:735,slopeDeg:8,transitionDeg:3,stepUpM:.02,stepDownM:.04};
  const up=carpetRoleLimits(b,'slope'),steps=carpetRoleLimits(b,'step');
  assert.equal(steps.slopeDeg,4);
  assert.equal(up.stepUpM,.01);
  assert.equal(steps.stepUpM,.02);
  assert.equal(steps.stepDownM,.04);
  assert.throws(()=>carpetRoleLimits(b,'unknown'),/Unknown/);
  const a=generateTrainingCarpet(b,'slope',{trees:false}),
        c=generateTrainingCarpet(b,'step',{trees:false});
  assert.notDeepEqual(Array.from(a.heights.slice(5000,5200)),Array.from(c.heights.slice(5000,5200)));
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