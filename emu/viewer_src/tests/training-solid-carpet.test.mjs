import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {
  createSolidCarpet,carpetLearningStage,carpetRepresentativeRole,
  terrainRoleEnvelope,safeCarpetShift
} from '../src/training-solid-carpet.js';
import {solidCarpetXml,withTerrain} from '../src/terrain.js';
import {actorTerrainDistance} from '../src/sensor.js';
import {trainingLimits} from '../src/training-carpet.js';

const measured=(step,more={})=>({
  source:'training',seed:735,step,
  foundationSteps:20_000_000,academyIntroSteps:60_000_000,
  academyStream:true,slopeDeg:12,transitionDeg:3,
  stepUpM:.03,stepDownM:.04,...more
});

test('real cold foundation is a flat 50cm nine-solid world, regardless of selected role',()=>{
  for(const age of [0,131072,5_000_000,19_999_999]){
    const b=measured(age);
    assert.equal(carpetLearningStage(b),'foundation');
    assert.equal(carpetRepresentativeRole(b,'auto'),'standing');
    for(const role of ['auto','step','slope','resilience']){
      const p=terrainRoleEnvelope(b,role);
      assert.equal(p.slopeDeg,0);
      assert.equal(p.stepUpM,0);
      assert.equal(p.stepDownM,0);
      const w=createSolidCarpet(b,role);
      assert.equal(w.tiles.length,9);
      assert.ok(w.tiles.every(t=>Math.abs(t.top)<1e-9));
      assert.ok(w.tiles.every(t=>Math.abs(t.center[2]+.270)<1e-9));
      assert.ok(w.tiles.every(t=>t.normal[2]===1));
    }
  }
});

test('auto stage moves from flat foundation to specialist slopes to Academy steps',()=>{
  assert.equal(carpetRepresentativeRole(measured(20_000_000),'auto'),'slope');
  assert.equal(carpetLearningStage(measured(59_999_999)),'specialists');
  assert.equal(carpetRepresentativeRole(measured(60_000_000),'auto'),'step');
  assert.equal(carpetLearningStage(measured(140_000_000)),'multi-lip');
  assert.equal(carpetRepresentativeRole(measured(90_000_000),'control'),'control');
});

test('nine box slots use the same modulo-three cell mapping as the GPU trainer',()=>{
  const b=createSolidCarpet(measured(90_000_000));
  const tiles=b.tiles;
  assert.deepEqual(tiles.map(t=>t.slot),[0,1,2,3,4,5,6,7,8]);
  assert.equal(new Set(tiles.map(t=>t.i+','+t.j)).size,9);
  for(const t of tiles){
    assert.equal(t.slot,((t.i%3+3)%3)+3*((t.j%3+3)%3));
    assert.ok(Math.abs(Math.hypot(...t.normal)-1)<1e-10);
    assert.ok(Math.abs(Math.hypot(...t.quat)-1)<1e-10);
    assert.ok(Math.abs(b.sampleHeight(t.x,t.y)-t.top)<1e-9);
  }
  const advance=b.window(1,0);
  assert.deepEqual(advance.map(t=>t.slot),[0,1,2,3,4,5,6,7,8]);
  assert.equal(new Set(advance.map(t=>t.i+','+t.j)).size,9);
  const kept=tiles.filter(t=>advance.some(a=>a.i===t.i&&a.j===t.j));
  assert.equal(kept.length,6);
  for(const t of kept){
    const v=advance.find(a=>a.i===t.i&&a.j===t.j);
    assert.deepEqual(v,t);
  }
});

test('unearned terrain flat, slope-only continuous grades, steps real signed lips',()=>{
  const flat=createSolidCarpet(measured(30_000_000,{slopeDeg:0,stepUpM:0,stepDownM:0}),'step');
  assert.ok(flat.tiles.every(t=>t.top===0));
  const graded=createSolidCarpet(measured(30_000_000,{slopeDeg:10,stepUpM:0,stepDownM:0}),'slope');
  assert.ok(graded.tiles.some(t=>Math.hypot(t.normal[0],t.normal[1])>.005));
  // A source-sampled continuous smooth potential can differ at shared
  // planes by millimetres; sharp multi-cm lips must come from STEP offsets.
  const steps=createSolidCarpet(measured(90_000_000,{slopeDeg:0,stepUpM:.03,stepDownM:.05}),'step');
  const differences=[];
  for(let i=-7;i<7;i++){
    const a=steps.at(i,0),b=steps.at(i+1,0);
    differences.push(b.top-a.top);
  }
  assert.ok(differences.some(d=>d>.002),'must contain actual uphill lips');
  assert.ok(differences.some(d=>d<-.002),'must contain actual downhill lips');
  assert.ok(differences.every(d=>Math.abs(d)<.1),'bounded source profiled lips');
});

test('live exposure means chance of full-height terrain, not scaling its severity',()=>{
  const b=measured(90_000_000,{
    slopeDeg:.7,stepUpM:.004,stepDownM:.004,
    academyStepUpM:.003,academyStepDownM:.003,
    slopeExposure:.15,stepExposure:.15
  });
  const slope=terrainRoleEnvelope(b,'slope');
  assert.equal(slope.slopeDeg,.7);
  assert.equal(slope.slopeExposureRate,.15);
  const ordinary=terrainRoleEnvelope({...b,academyStream:false},'step');
  assert.equal(ordinary.stepUpM,.004);
  assert.equal(ordinary.stepDownM,.004);
  const academy=terrainRoleEnvelope(b,'step');
  assert.equal(academy.stepUpM,.003);
  assert.equal(academy.stepDownM,.003);
  const suppressed=terrainRoleEnvelope({...b,slopeExposure:0,stepExposure:0,academyStream:false},'slope');
  assert.equal(suppressed.slopeDeg,0);
  assert.equal(suppressed.stepUpM,0);
  const stillAcademy=terrainRoleEnvelope({...b,stepExposure:0},'step');
  assert.equal(stillAcademy.stepUpM,.003);
  const carpet=createSolidCarpet(b,'step');
  assert.ok(carpet.at(1,0).top!==carpet.at(0,0).top);
});

test('never recycle a box while a sphere foot overlaps its abandoned edge',()=>{
  const carpet=createSolidCarpet(measured(90_000_000));
  const before=carpet.window(0,0),after=carpet.window(1,0);
  assert.equal(safeCarpetShift(before,after,[[-.25,.25],[.25,.25]]),false);
  assert.equal(safeCarpetShift(before,after,[[.40,.25],[.35,.20]]),true);
});

test('browser MuJoCo sees nine literal contact boxes, no hfield or buried floor',()=>{
  const carpet=createSolidCarpet(measured(90_000_000));
  const raw='<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="5 5 0.05"/></worldbody></mujoco>';
  const xml=withTerrain(raw,'carpet',{park:carpet});
  assert.equal((xml.match(/mocap="true"/g)||[]).length,9);
  assert.equal((xml.match(/viewer_carpet_solid_\d+" type="box"/g)||[]).length,9);
  assert.doesNotMatch(xml,/name="floor"/);
  assert.doesNotMatch(xml,/type="hfield"/);
  assert.doesNotMatch(xml,/viewer_park_hfield/);
  assert.equal((solidCarpetXml(carpet).match(/size="0.258 0.258 0.270"/g)||[]).length,9);
});

test('viewer follows trainer source config without calling 20M a skill milestone',()=>{
  const telemetry=trainingLimits({
    training_step:5_000_000,
    latest_metrics:{step:5_000_000,slope_max_deg:'0',academy_height_up_m:'0',academy_height_down_m:'0'},
    training_config:{seed:735,terrain_carpet:true,academy_stream:true,
      foundation_steps:20_000_000,academy_intro_steps:60_000_000,
      carpet_transition_cap_deg:3},
    physics_mode:'nine-solid-box trainer'
  });
  assert.equal(telemetry.trainingCarpet,true);
  assert.equal(telemetry.foundationSteps,20_000_000);
  assert.equal(telemetry.academyIntroSteps,60_000_000);
  assert.equal(telemetry.transitionConfigured,true);
  assert.equal(carpetLearningStage(telemetry),'foundation');
  assert.equal(telemetry.slopeDeg,0);
});

test('the integrated viewer keeps the real physical contacts separate from the robot sensor self-occlusion',()=>{
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const ui=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  assert.match(sim,/this\.data\.mocap_pos/);
  assert.match(sim,/this\.data\.mocap_quat/);
  assert.match(sim,/safeCarpetShift/);
  assert.match(sim,/!this\.carpetGeomIds\.has\(geomId\)/);
  assert.match(main,/carpetAutoApply/);
  assert.match(main,/carpetEffectiveLimits/);
  assert.match(ui,/value="auto" selected/);
});


test('actor 4x4 optical ray sees a MOVING MuJoCo solid support and its vertical side, never robot',()=>{
  const model={
    ngeom:1,
    geom_bodyid:[1],
    body_mocapid:[-1,0],
    geom_type:[6], // MuJoCo mjGEOM_BOX
    geom_size:[.258,.258,.270]
  };
  const data={geom_xpos:[0,0,-.270],
    geom_xmat:[1,0,0,0,1,0,0,0,1]};
  assert.ok(Math.abs(
    actorTerrainDistance([0,0,.30],[0,0,-1],model,data,[],null)-.30)<1e-9);
  assert.ok(Math.abs(
    actorTerrainDistance([-.45,0,-.10],[1,0,0],model,data,[],null)-.192)<1e-9);
  // Same box geometry on a normal ARTICULATED robot body is NEVER ground.
  model.body_mocapid[1]=-1;
  assert.equal(actorTerrainDistance([0,0,.30],[0,0,-1],model,data,[],null),0);
});

test('no synthetic future-tile height is fed into carpet actor or optical-flow sensors',()=>{
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  assert.match(sim,/terrainHeightAt:this\.terrainMode==='park'/);
  assert.match(sim,/const terrainHeightAt=this\.terrainMode==='park'/);
});
