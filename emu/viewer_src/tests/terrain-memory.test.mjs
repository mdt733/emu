import test from 'node:test';
import assert from 'node:assert/strict';
import {groundRayHit, RollingTerrainDiagnostic, V_CELLS, V_SAMPLES} from '../src/terrain-memory.js';

const near=(a,b,tol=1e-3)=>assert.ok(Math.abs(a-b)<=tol,[a,b]);

test('first-hit sensor sees a real raised step before the flat floor',()=>{
  const origin=[0,0,.2334], dx=.165/.2334;
  const length=Math.hypot(dx,1),ray=[dx/length,0,-1/length];
  const flat=groundRayHit(origin,ray,()=>0);
  const step=groundRayHit(origin,ray,(x)=>x>=.12&&x<=.24?.05:0);
  assert.ok(flat&&step);
  assert.ok(flat.range-step.range>.02,{flat,step});
});

test('six directions are NOT fabricated: two front rays and three downward',()=>{
  const mem=new RollingTerrainDiagnostic();
  const pose={R:[1,0,0,0,1,0,0,0,1],torso:[0,0,.2584],
    vbody:[.1,0,0],gyro:[0,0,0]};
  const hit=(start,dir,max)=>groundRayHit(start,dir,()=>0,max);
  for(let i=0;i<3;i++)mem.update(pose,false,hit);
  assert.equal(mem.beams.length,5);
  assert.equal(mem.beams.filter(b=>b.valid).length,5);
  assert.equal(mem.heights.length,2);
  assert.equal(mem.heights[0].length,V_CELLS);
  assert.deepEqual(V_SAMPLES,[5,10,15,20,25]);
  assert.equal(mem.values().actorHeights.length,10);
  assert.equal(mem.values().actorConfidence.length,10);
  // The two forward lookahead rays have a slant range > direct down-rays.
  assert.ok(mem.beams[0].range>mem.beams[4].range);
});

test('rolling memory preserves sparse measurements, moves forward hits back toward feet',()=>{
  const m=new RollingTerrainDiagnostic();
  m.insert(0,.20,-.05,.05,true);
  m.insert(1,.20,+.05,.02,true);
  assert.equal(m.values().actorConfidence[4],0);
  for(let i=0;i<25;i++)m.transport(.02,.1,0,0,1);
  const d=m.values();
  assert.ok(d.actorConfidence[4]>.005);
  assert.ok(d.actorConfidence[9]>.005);
  near(d.actorHeights[4],.05,.02);
  near(d.actorHeights[9],.02,.02);
  assert.equal(d.actorConfidence[0],0);
});

test('turning and reverse travel advect measured patches without creating full map',()=>{
  const m=new RollingTerrainDiagnostic();
  m.insert(0,.10,-.029,.03,true);
  m.insert(1,.10,+.029,.07,true);
  m.transport(.20,-.10,0,0,1);
  const index=m.confidence[0].findIndex(c=>c>0.1);
  assert.ok(index>=0);
  const totalBefore=m.confidence.flatMap(a=>[...a]).filter(v=>v>.001).length;
  m.transport(.10,0,0,1,1);
  const totalAfter=m.confidence.flatMap(a=>[...a]).filter(v=>v>.001).length;
  assert.ok(totalBefore<100&&totalAfter<100);
  assert.ok(m.confidence.every(a=>[...a].every(v=>v>=0&&v<=1)));
});

test('a frozen memory keeps its 2x51 values while rays continue measuring',()=>{
  const m=new RollingTerrainDiagnostic();
  m.insert(0,.15,-.04,.03,true);
  const old=[...m.heights[0]];
  const pose={R:[1,0,0,0,1,0,0,0,1],torso:[0,0,.2584],
    vbody:[.3,0,0],gyro:[0,0,.2]};
  const hit=(start,dir,max)=>groundRayHit(start,dir,()=>0,max);
  for(let i=0;i<5;i++)m.update(pose,false,hit,{freeze:true});
  assert.deepEqual([...m.heights[0]],old);
  assert.equal(m.beams.length,5);
});
