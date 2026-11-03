import test from 'node:test';
import assert from 'node:assert/strict';
import {trainingLimits} from '../src/training-carpet.js';
import {specialistLesson} from '../src/training-inspector.js';
import {roleSliders} from '../src/training-slider-model.js';
import {createSolidCarpet} from '../src/training-solid-carpet.js';
import {carpetGeometrySignature} from '../src/carpet-refresh.js';

const near=(got,want,tol=1e-9)=>assert.ok(Math.abs(got-want)<=tol,
  `Expected ${want} ± ${tol}; got ${got}`);
const row=(stepHeight=.004)=>trainingLimits({
  schema:'emu_terrain_live_v1',
  training_step:110_297_088,
  run_name:'v5041_carpet_s735_140m_fork10m',
  latest_metrics:{
    step:110_297_088,
    slope_max_deg:.7,
    slope_exposure:.03,
    terrain_height_m:stepHeight,
    terrain_exposure:.15,
    academy_height_up_m:.001,
    academy_height_down_m:.002,
    terrain_slope_transition_frontier_deg:1.0,
    frontier_vx_hi:.56
  },
  training_config:{
    terrain_carpet:true,academy_stream:true,seed:735,
    foundation_steps:20_000_000,academy_intro_steps:60_000_000,
    carpet_transition_cap_deg:3
  }
});

test('110M console frontier 4mm at 15% exposure is 0.6mm REAL STEP birth',()=>{
  const b=row(),l=specialistLesson(b,'step');
  assert.equal(b.ordinaryStepUpM,.004);
  assert.equal(b.academyStepUpM,.001);
  near(l.terrain.stepUpM,.004*.15);
  near(l.terrain.stepDownM,.004*.15);
  near(l.terrain.slopeDeg,.7*.5*.03);
  near(roleSliders(b,'step').up,.6);
  near(roleSliders(b,'step').slope,.0105);
  const actual=createSolidCarpet(b,'step');
  near(actual.limits.stepUpM,.0006);
  near(actual.limits.slopeDeg,.0105);
  assert.equal(actual.tiles.length,9);
});

test('Academy is an episode assignment, not automatic for ALL STEP worlds at 110M',()=>{
  const b=row();
  assert.equal(b.academyEligible,true);
  near(b.stepUpM,.004);
  near(b.ordinaryStepUpM,.004);
  near(b.academyStepUpM,.001);
  const academy={...b,stepUpM:b.academyStepUpM,stepDownM:b.academyStepDownM};
  near(specialistLesson(academy,'step').terrain.stepUpM,.00015);
  near(specialistLesson(academy,'step').terrain.stepDownM,.0003);
});

test('viewer frontier challenge bypasses exposure ONLY when explicitly selected',()=>{
  const b=row();
  const challenge={
    ...b,previewOverride:{terrain:{
      slopeDeg:b.slopeDeg,transitionDeg:b.transitionDeg,
      stepUpM:b.stepUpM,stepDownM:b.stepDownM
    }}
  };
  const lesson=specialistLesson(challenge,'step');
  near(lesson.terrain.stepUpM,.004);
  near(lesson.terrain.slopeDeg,.7);
  near(specialistLesson(b,'step').terrain.stepUpM,.0006);
});

test('auto-refresh sees sub-mm step growth and hundredth-degree slope growth',()=>{
  const a=specialistLesson(row(.004),'step');
  const b=specialistLesson(row(.005),'step');
  assert.notEqual(carpetGeometrySignature(a,'ordinary'),
                  carpetGeometrySignature(b,'ordinary'),
                  'Old 2mm quantisation hid 0.6 vs 0.75mm role physics');
  const steep=specialistLesson({...row(),slopeExposure:.05},'step');
  assert.notEqual(carpetGeometrySignature(a,'ordinary'),
                  carpetGeometrySignature(steep,'ordinary'));
  assert.notEqual(carpetGeometrySignature(a,'ordinary'),
                  carpetGeometrySignature(a,'academy'),
                  'Academy versus ordinary birthplace must force a reset');
  assert.equal(carpetGeometrySignature(a,'ordinary'),
               carpetGeometrySignature(specialistLesson(row(),'step'),'ordinary'));
});

test('precise viewer controls display hundredths of mm and do not contain obsolete 2mm snap',()=>{
  const fs=await import('node:fs');
  const ui=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=fs.readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(ui,/id="carpetDifficultyMode"/);
  assert.match(ui,/id="carpetStepTrack"/);
  assert.match(ui,/id="carpetPhysicalEvidence"/);
  assert.match(ui,/id="carpetHudTerrain"/);
  assert.match(ui,/id="tuneUp"[^>]*step="\.025"/);
  assert.match(main,/carpetGeometrySignature\(currentCarpetLesson\(\)/);
  assert.doesNotMatch(main,/Math\.round\(p\.stepUpM\*500\)/);
});
