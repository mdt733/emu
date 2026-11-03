import test from 'node:test';
import assert from 'node:assert/strict';
import {buildPrismTerrain,normalizePrismSettings,TILE_SIZE_M} from '../src/prism-terrain.js';
const near=(a,b,eps=1e-11)=>assert.ok(Math.abs(a-b)<eps,`mismatch ${a} vs ${b}`);
test('3x3 gives eighteen real closed triangular prisms, 5x5 gives fifty',()=>{
  for (const n of [3,5,7,9]) {
    const m=buildPrismTerrain({tiles:n});
    assert.equal(m.prisms.length,n*n*2);
    assert.equal(m.tiles.length,n*n);
    for(const p of m.prisms) {
      assert.ok(p.normal[2]>0);
      for(const v of p.upper) assert.ok(v[2]>m.baseZ);
      for(const v of p.lower) near(v[2],m.baseZ);
    }
  }
});
test('smooth slopes, crests, saddles and transitions have zero edge gaps',()=>{
  for(const surface of ['rolling','crest','valley','cross','saddle','flat'])for(const diagonal of ['slash','backslash','checker']){
    const m=buildPrismTerrain({surface,diagonal,steps:'none',slopeDeg:15,transitionDeg:12});
    assert.equal(m.stats.intentionalEdges,0);
    assert.equal(m.stats.joinedEdges,40);
    near(m.stats.accidentalMaxMm,0);
    for(const e of m.edges){near(e.a[0][2],e.b[0][2]);near(e.a[1][2],e.b[1][2]);}
  }
});
test('only declared steps separate shared edges; full explicit risers cover them',()=>{
  for(const steps of ['up','down','up-down','staircase','checker']){
    const m=buildPrismTerrain({steps,slopeDeg:7,transitionDeg:5,stepUpMm:28,stepDownMm:14});
    assert.ok(m.stats.intentionalEdges>0);
    near(m.stats.accidentalMaxMm,0);
    for(const r of m.risers){
      near(r.high[0][0],r.low[0][0]);near(r.high[0][1],r.low[0][1]);
      near(r.high[1][0],r.low[1][0]);near(r.high[1][1],r.low[1][1]);
      assert.ok(r.high[0][2]>r.low[0][2]);
      near(r.high[0][2]-r.low[0][2],r.heightM);
    }
  }
});
test('height queries interpolate the intended triangular top, not a fitted box plane',()=>{
  function planeHeight(tri,x,y){
    const [a,b,c]=tri;
    const u=b[0]-a[0],v=b[1]-a[1],w=c[0]-a[0],z=c[1]-a[1];
    const det=u*z-v*w;
    const dx=x-a[0],dy=y-a[1];
    const t=(dx*z-dy*w)/det,q=(dy*u-dx*v)/det;
    return a[2]+t*(b[2]-a[2])+q*(c[2]-a[2]);
  }
  for(const diagonal of ['slash','backslash','checker']){
    const m=buildPrismTerrain({diagonal});
    for(const t of m.tiles){
      const x=t.i*TILE_SIZE_M,y=t.j*TILE_SIZE_M;
      for(const [u,v] of [[.18,.15],[.77,.16],[.17,.79],[.68,.67]]){
        const px=x+u*TILE_SIZE_M,py=y+v*TILE_SIZE_M;
        const tri=t.tops.find(p=>{
          const [a,b,c]=p;
          const area=(v,w,q)=>(w[0]-v[0])*(q[1]-v[1])-(w[1]-v[1])*(q[0]-v[0]);
          return area(a,b,[px,py])>=-1e-12&&area(b,c,[px,py])>=-1e-12&&area(c,a,[px,py])>=-1e-12;
        });
        assert.ok(tri,`no covering triangle at ${px},${py}`);
        near(m.sampleHeight(px,py),planeHeight(tri,px,py));
      }
    }
  }
});
test('same seed/settings regenerate bit-identical, seed changes rolling shape',()=>{
  const a=buildPrismTerrain({seed:5}),b=buildPrismTerrain({seed:5}),c=buildPrismTerrain({seed:6});
  assert.deepEqual(a.tiles,b.tiles);
  assert.notDeepEqual(a.tiles,c.tiles);
});
test('invalid numeric inputs fail closed; allowed setting values clamp',()=>{
  assert.throws(()=>normalizePrismSettings({slopeDeg:'not a number'}));
  assert.equal(normalizePrismSettings({tiles:99,slopeDeg:500,stepUpMm:-2}).tiles,5);
  assert.equal(normalizePrismSettings({slopeDeg:500,stepUpMm:-2}).slopeDeg,20);
});