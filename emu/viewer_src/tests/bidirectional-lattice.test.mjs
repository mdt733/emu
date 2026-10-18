import test from 'node:test';
import assert from 'node:assert/strict';
import {integratePose,wrapAngle} from '../src/goal-navigation.js';
import {planCurvatureRoute} from '../src/curvature-planner.js';
import {
 ALL_JOYSTICK_OPTIONS,joystickPrimitives,reversePrimitive,bidirectionalMotionRoute
} from '../src/bidirectional-lattice.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.30,yaw_abs:1.6};
const origin={x:0,y:0,heading:0};
const angle=(a,b)=>Math.abs(wrapAngle(a-b));
const separation=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

test('actual 9 translation directions × 3 yaw choices produce 27 discrete joystick candidates',()=>{
  const options=joystickPrimitives(limits);
  assert.equal(ALL_JOYSTICK_OPTIONS,27);
  assert.equal(options.length,27);
  assert.equal(options.filter(p=>p.executable).length,24,
    'centre command is invalid for dynamical balance (standing/pivots)');
  assert.equal(new Set(options.map(p=>p.id)).size,27);
  assert.ok(options.every(p=>p.cost>=p.dt));
  assert.ok(options.every(p=>Math.abs(p.vx)<=limits.vx_max+.01 &&
    Math.abs(p.vy)<=limits.vy_abs+.01 &&
    Math.abs(p.yaw)<=limits.yaw_abs+.01));
  const forward=options.find(x=>x.joystick.join(',')==='1,0,0');
  const strafe=options.find(x=>x.joystick.join(',')==='0,1,0');
  const backward=options.find(x=>x.joystick.join(',')==='-1,0,0');
  assert.ok(forward.vx>.8);
  assert.ok(strafe.vy>0 && strafe.vx===0);
  assert.ok(backward.vx<0);
  assert.ok(forward.vx>Math.abs(strafe.vy)*4,
    'time estimator must distinguish fast forward and expensive sideways travel');
});

test('reverse frontier uses mathematically inverse SE2 twist, preserving original forward control',()=>{
  const options=joystickPrimitives(limits).filter(p=>p.executable);
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
