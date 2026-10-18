import test from 'node:test';
import assert from 'node:assert/strict';
import {integratePose,wrapAngle} from '../src/goal-navigation.js';
import {planCurvatureRoute} from '../src/curvature-planner.js';
import {
 ALL_JOYSTICK_OPTIONS,ARC_YAW_FRACTIONS,ARC_TRANSLATION_MODES,
 joystickPrimitives,arcPrimitives,reversePrimitive,bidirectionalMotionRoute
} from '../src/bidirectional-lattice.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.30,yaw_abs:1.6};
const origin={x:0,y:0,heading:0};
const angle=(a,b)=>Math.abs(wrapAngle(a-b));
const separation=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

test('variable-radius lattice has 9 yaw rates for each of forward, reverse and in-place turning',()=>{
  const options=arcPrimitives(limits);
  assert.deepEqual(joystickPrimitives(limits),options);
  assert.equal(ARC_YAW_FRACTIONS.length,9);
  assert.deepEqual(ARC_TRANSLATION_MODES,['forward','reverse','pivot']);
  assert.equal(ALL_JOYSTICK_OPTIONS,27);
  assert.equal(options.length,27);
  assert.equal(options.filter(p=>p.executable).length,26,
    'EMU can pivot; only idle (no movement, no rotation) is a no-op');
  assert.equal(new Set(options.map(p=>p.id)).size,27);
  assert.ok(options.every(p=>p.cost>=p.dt));
  assert.ok(options.every(p=>Math.abs(p.vx)<=limits.vx_max+.01 &&
    p.vy===0 && Math.abs(p.yaw)<=limits.yaw_abs+.01));
  const fwd=options.filter(x=>x.mode==='forward');
  const rev=options.filter(x=>x.mode==='reverse');
  const pivots=options.filter(x=>x.mode==='pivot');
  assert.equal(fwd.length,9);assert.equal(rev.length,9);assert.equal(pivots.length,9);
  assert.ok(fwd.every(x=>x.vx>.8));
  assert.ok(rev.every(x=>x.vx<-.3));
  assert.ok(pivots.filter(x=>x.executable).every(x=>x.vx===0&&Math.abs(x.yaw)>.1));
  assert.equal(pivots.find(x=>x.yaw===0).executable,false);
  // Distinct angular speeds imply distinct arc radii at fixed forward speed.
  assert.equal(new Set(fwd.filter(x=>x.yaw!==0)
    .map(x=>x.radius.toFixed(3))).size,4);
  assert.ok(fwd.find(x=>x.yaw===0).radius===Infinity);
  assert.ok(fwd.find(x=>x.yaw>0).curvature>0);
  assert.ok(fwd.find(x=>x.yaw<0).curvature<0);
  assert.equal(new Set(fwd.map(x=>x.vx)).size,1);
  assert.equal(new Set(rev.map(x=>x.vx)).size,1);
});

test('reverse frontier inverts forward/reverse arcs AND actual in-place turns',()=>{
  const options=arcPrimitives(limits).filter(p=>p.executable);
  const starts=[origin,{x:1.42,y:-.85,heading:2.61},
    {x:-.21,y:1.53,heading:-3.09}];
  for(const pose of starts){
    for(const primitive of options){
      const next=integratePose(pose,primitive,primitive.dt);
      const previous=integratePose(next,reversePrimitive(primitive),primitive.dt);
      assert.ok(separation(pose,previous)<1e-9,primitive.id+' x/y mismatch');
      assert.ok(angle(pose.heading,previous.heading)<1e-9,
        primitive.id+' heading mismatch');
    }
  }
});

test('continuous curve planner can connect exact heading and position with no grid hops',()=>{
  for(const dest of [
    {x:2.1,y:0,heading:0},
    {x:1.7,y:1.0,heading:Math.PI/2},
    {x:-1.1,y:0,heading:0},
  ]){
    const path=planCurvatureRoute(origin,dest,limits,[],{noDetour:true});
    assert.ok(path?.solved,'no curve for '+JSON.stringify(dest));
    assert.ok(path.steps.length>=2);
    assert.ok(separation(path.steps.at(-1),dest)<.026);
    assert.ok(angle(path.steps.at(-1).heading,dest.heading)<.03);
  }
});

test('two bounded search trees join at compatible complete poses without an XY-only teleport',()=>{
  const target={x:1.0,y:0,heading:0};
  const route=bidirectionalMotionRoute(origin,target,limits,[],{maxNodes:1800});
  assert.ok(route?.solved,'forward/backward motion searches did not meet');
  assert.equal(route.planner,'bidirectional-motion-lattice');
  assert.ok(route.expanded>0 && route.expanded<=1800);
  assert.ok(route.connectionMetres<1.7);
  assert.ok(separation(route.steps.at(-1),target)<.025);
  assert.ok(angle(route.steps.at(-1).heading,target.heading)<.03);
  assert.equal(route.steps.filter(x=>x.final).length,1);
  assert.ok(route.estimatedSeconds>0);
});

test('bidirectional planning rejects blocked targets and impossible command envelopes',()=>{
  assert.equal(bidirectionalMotionRoute(origin,{x:.7,y:0,heading:0},
    limits,[{x:.7,y:0}],{maxNodes:200}),null);
  assert.throws(()=>bidirectionalMotionRoute(origin,{x:NaN,y:0,heading:0},
    limits),/Nonfinite/);
  assert.throws(()=>joystickPrimitives({...limits,vx_max:0}),/insufficient/);
});

test('new motion-primitives remain experimental; live browser keeps known-good controller',async()=>{
  const {readFileSync}=await import('node:fs');
  const {fileURLToPath}=await import('node:url');
  const read=p=>readFileSync(fileURLToPath(new URL(p,import.meta.url)),'utf8');
  const main=read('../src/main.js'),worker=read('../src/goal-worker.js');
  assert.match(worker,/planPoseRoute/);
  assert.match(main,/applyAutonomousCommand\(plannedCommand\)/);
  assert.doesNotMatch(main,/bidirectionalMotionRoute/);
});
