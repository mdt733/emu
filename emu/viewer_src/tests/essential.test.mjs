import test from 'node:test';
import assert from 'node:assert/strict';
import { joystickToCommands, headingHoldRate } from '../src/controls.js';
import { parseShowcaseManifest, assertShowcasePolicyResponse } from '../src/showcase.js';
import { withGentleCourse, withTerrain, generateTerrainPark, localSlopeLimitDeg, GENTLE_SLOPE_DEGREES } from '../src/terrain.js';
const limits = {vx_min:-0.2,vx_max:0.6,vy_abs:0.22,yaw_abs:1.5,pitch_abs_rad:0.08};
const base = {schema:1,mode:'showcase',policy:'policies/champion.emupolicy',trained:true,autoplay:true,label:'EMU'};
const xml = '<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>';

test('manual joystick right/left are reversed from old vy mapping, yaw stays unchanged', () => {
  const right = joystickToCommands(1, 0, 0, 0, limits);
  const left = joystickToCommands(-1, 0, 0, 0, limits);
  assert.equal(right.vy, -0.22);
  assert.equal(left.vy, 0.22);
  assert.equal(Math.abs(right.yaw), 0);
  assert.equal(joystickToCommands(0, 0, 1, 0, limits).yaw, -1.5);
});

test('rejects path traversal, external URLs, absolute assets and malformed manifest', () => {
  for (const p of ['../secret', '/policies/champion.emupolicy', 'https://evil/anything', 'policies/../../secret', 'policies/foo.txt']) {
    assert.throws(() => parseShowcaseManifest({...base, policy:p}));
  }
  assert.throws(() => parseShowcaseManifest({...base, schema:2}));
});

test('Missing static .emupolicy SPA fallback yields actionable error', () => {
  const html = new Response('<!doctype html>', {
    status:200, headers:{'Content-Type':'text/html'},
  });
  assert.throws(() => assertShowcasePolicyResponse(html), /Bundled \.emupolicy file is missing/);
});

test('gentle terrain is real additional MuJoCo contact geometry', () => {
  const out = withGentleCourse(xml);
  assert.equal(GENTLE_SLOPE_DEGREES, 1);
  for (const name of ['viewer_ramp_up', 'viewer_ramp_top', 'viewer_ramp_down']) {
    assert.match(out, new RegExp(`name="${name}"`));
  }
  assert.match(out, /euler="0 -1 0"/);
  assert.match(out, /euler="0 1 0"/);
  assert.equal(xml.includes('viewer_ramp'), false);
  assert.throws(() => withGentleCourse(out), /already inserted/);
});

test('inverted pitch, blended diagonal steering, and side-only strafing',()=>{
  assert.equal(joystickToCommands(0,0,0,1,limits).pitch,-limits.pitch_abs_rad);
  const side=joystickToCommands(1,0,0,0,limits,'hybrid');
  assert.equal(side.vy,-limits.vy_abs);assert.equal(side.yaw,0);
  const diagonal=joystickToCommands(0.6,0.8,0,0,limits,'hybrid');
  assert.ok(diagonal.vx>0 && diagonal.yaw<0 && diagonal.vy<0);
  assert.ok(Math.abs(diagonal.vy)<Math.abs(side.vy));
});
test('heading-hold P responds to drift with saturating correction',()=>{
  assert.ok(headingHoldRate(0,0,0.1,0.75,1.5)<0);
  assert.ok(headingHoldRate(0,0,-0.1,0.75,1.5)>0);
  assert.equal(headingHoldRate(1.5,1.0,0,0.75,1.5),1.5);
});
test('continuous physical heightfield agrees with visual sample grid',()=>{
  const park=generateTerrainPark({maxDegrees:10});
  assert.equal(park.heights.length,park.resolution**2);
  assert.ok(park.heights.some(h=>h>0.05));
  assert.ok(park.trees.length>=10);
  assert.ok(park.trees.every(t=>Math.hypot(t.x,t.y)>3.4));
  assert.ok(park.trees.every(t=>Math.abs(park.sampleHeight(t.x,t.y)-t.z)<1e-7));
  const half=park.radius,cell=park.cell,n=park.resolution;
  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const x=-half+i*cell,y=-half+j*cell,k=j*n+i;
    if(Math.hypot(x,y)<=2)assert.equal(park.heights[k],0);
    if(i+1<n){
      const nx=x+cell;
      const maxAngle=Math.min(localSlopeLimitDeg(Math.hypot(x,y),10),localSlopeLimitDeg(Math.hypot(nx,y),10));
      const limit=Math.tan(maxAngle*Math.PI/180)*cell*0.56;
      assert.ok(Math.abs(park.heights[k]-park.heights[k+1])<limit+1e-5);
    }
  }
  assert.equal(localSlopeLimitDeg(2,10),1);
  assert.equal(localSlopeLimitDeg(3,10),2);
  assert.ok(localSlopeLimitDeg(12,10)>=10);
});
test('park XML uses exactly one joined heightfield and real tree collisions',()=>{
  const out=withTerrain(xml,'park',{maxDegrees:10});
  assert.match(out,/<hfield name="viewer_park_hfield"/);
  assert.match(out,/<geom name="viewer_park_ground" type="hfield"/);
  assert.match(out,/viewer_tree_trunk_0/);
  assert.doesNotMatch(out,/viewer_park_0" type="box"/);
  assert.doesNotMatch(out,/name="floor" type="plane"/);
  assert.equal(out,withTerrain(xml,'park',{maxDegrees:10}));
  assert.throws(()=>withTerrain(out,'park'),/already inserted/);
  const noTrees=withTerrain(xml,'park',{trees:false});
  assert.doesNotMatch(noTrees,/viewer_tree_trunk_0/);
});
