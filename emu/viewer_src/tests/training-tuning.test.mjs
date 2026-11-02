import test from 'node:test';
import assert from 'node:assert/strict';
import {trainingLimits} from '../src/training-carpet.js';
import {specialistLesson} from '../src/training-inspector.js';
import {roleSliders,customRoleOverride} from '../src/training-slider-model.js';
import {createSolidCarpet} from '../src/training-solid-carpet.js';
import {sampleShove,shoveChance} from '../src/training-shoves.js';

const data=step=>trainingLimits({
  training_step:step,
  latest_metrics:{step,frontier_vx_hi:'.28',frontier_vx_lo:'-.07',
    frontier_vy:'.12',frontier_yaw:'.7',frontier_push_n:'.65',
    resilience_difficulty:'.36',slope_max_deg:'.7',
    terrain_height_m:'.004',academy_height_up_m:'.003',
    academy_height_down_m:'.002',slope_exposure:'.8',terrain_exposure:'.5'},
  training_config:{terrain_carpet:true,academy_stream:true,seed:735,
    foundation_steps:20000000,academy_intro_steps:60000000,
    push_max_force_n:0.7}
});

test('the 10 sliders expose effective specialist weights, not unscaled global frontiers',()=>{
  const b=data(34000000);
  const slope=roleSliders(b,'slope'),steps=roleSliders(b,'step');
  assert.ok(Math.abs(slope.slope-.56)<1e-9);
  assert.ok(Math.abs(steps.slope-.28)<1e-9);
  assert.equal(steps.up,2);
  assert.equal(steps.down,2);
  assert.equal(steps.forward,.28);
  assert.equal(steps.reverse,.07);
  assert.equal(steps.push,.65);
  assert.equal(roleSliders(b,'resilience').resilience,.36);
  assert.equal(roleSliders(b,'command').resilience,.36*.15);
});

test('custom sliders override exactly the requested role terrain and commands',()=>{
  const b=data(34000000);
  const original=roleSliders(b,'step');
  const custom=customRoleOverride({...original,
    slope:4,transition:2,up:28,down:15,
    forward:.2,reverse:.08,lateral:.04,yaw:.3,
    resilience:1,push:1.2});
  const l=specialistLesson({...b,previewOverride:custom},'step');
  assert.equal(l.terrain.slopeDeg,4);
  assert.equal(l.terrain.stepUpM,.028);
  assert.equal(l.terrain.stepDownM,.015);
  assert.equal(l.terrain.transitionDeg,2);
  assert.equal(l.caps.forward,.2);
  assert.equal(l.caps.push,1.2);
  assert.equal(l.intensity,1);
  const solid=createSolidCarpet({...b,previewOverride:custom},'step');
  assert.equal(solid.limits.stepUpM,.028);
  assert.equal(solid.limits.stepDownM,.015);
  assert.equal(solid.tiles.length,9);
  // Original trainer curriculum data was never modified by the preview.
  assert.equal(b.slopeDeg,.7);
  assert.ok(Math.abs(roleSliders(b,'step').slope-.28)<1e-9);
});

test('manual challenge can be shown at a flat training foundation age',()=>{
  const b=data(5000000);
  assert.equal(roleSliders(b,'step').up,0);
  const o=customRoleOverride({...roleSliders(b,'step'),up:24,down:12});
  const p=specialistLesson({...b,previewOverride:o},'step');
  assert.equal(p.terrain.stepUpM,.024);
  assert.equal(p.terrain.stepDownM,.012);
});

test('random shoves use teacher per-frame odds, selectable inspection rate and 2–8 frames',()=>{
  assert.equal(shoveChance(0,8),0);
  assert.equal(shoveChance(.36,1),.0015*.36);
  assert.equal(shoveChance(.36,8),.0015*.36*8);
  // Event chance, random direction, random magnitude, random duration.
  const draws=[0,.25,.5,.5];
  const r=sampleShove({intensity:.36,multiplier:8,maxForceN:.7,
    rng:()=>draws.shift()});
  assert.ok(r);
  assert.ok(Math.abs(r.fx)<1e-9);
  assert.ok(Math.abs(r.fy-.35)<1e-9);
  assert.equal(r.frames,5);
  assert.equal(sampleShove({intensity:1,maxForceN:1,active:true,manual:true}),null);
  assert.equal(sampleShove({intensity:0,maxForceN:1,manual:false,rng:()=>0}),null);
  const manual=sampleShove({intensity:0,maxForceN:1,manual:true,
    rng:()=>.9});
  assert.ok(manual);
  assert.equal(manual.frames,8);
});
