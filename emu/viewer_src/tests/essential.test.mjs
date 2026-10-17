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
test('radial terrain is physically represented, flat at spawn, deterministic and locally grade-limited',()=>{
  const park=generateTerrainPark({maxDegrees:10});
  assert.ok(park.tiles.length>400);
  assert.ok(park.tiles.some(t=>localSlopeLimitDeg(Math.hypot(t.x,t.y),10)>=10));
  assert.ok(park.tiles.every(t=>Math.hypot(t.x,t.y)>2));
  assert.ok(park.tiles.every(t=>t.grade<=t.limit+1e-9));
  assert.equal(localSlopeLimitDeg(2,10),1);
  assert.equal(localSlopeLimitDeg(3,10),2);
  const generated=withTerrain(xml,'park',{maxDegrees:10});
  assert.match(generated,/viewer_park_0/);
  assert.match(generated,/type="box"/);
  assert.equal(generated,withTerrain(xml,'park',{maxDegrees:10}));
  assert.throws(()=>withTerrain(generated,'park'),/already inserted/);
});
