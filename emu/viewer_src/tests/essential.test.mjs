import test from 'node:test';
import assert from 'node:assert/strict';
import { joystickToCommands } from '../src/controls.js';
import { parseShowcaseManifest, assertShowcasePolicyResponse } from '../src/showcase.js';
import { withGentleCourse, GENTLE_SLOPE_DEGREES } from '../src/terrain.js';
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
