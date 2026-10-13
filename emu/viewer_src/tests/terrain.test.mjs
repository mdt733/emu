import test from 'node:test';
import assert from 'node:assert/strict';
import { withGentleCourse, GENTLE_SLOPE_DEGREES } from '../src/terrain.js';

const xml = '<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>';
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

test('invalid XML fails closed instead of substituting visual terrain', () => {
  assert.throws(() => withGentleCourse('<mujoco/>'), /worldbody/);
});
