import test from 'node:test';
import assert from 'node:assert/strict';
import { measuredAttitude } from '../src/attitude.js';

const I = [1,0,0, 0,1,0, 0,0,1];
const r = d => d * Math.PI / 180;

test('neutral stance yields zero robot lean and chassis pitch', () => {
  const x = measuredAttitude(I, [0,0,.25], [0,-.08,0], [0,.08,0]);
  assert.ok(Math.abs(x.robotLean) < 1e-10);
  assert.ok(Math.abs(x.chassisPitch) < 1e-10);
});

test('torso can pitch 45 degrees while feet midpoint remains below it', () => {
  const a = r(45), c = Math.cos(a), s = Math.sin(a);
  const R = [c,0,s, 0,1,0, -s,0,c];
  const x = measuredAttitude(R, [0,0,.25], [0,-.08,0], [0,.08,0]);
  assert.ok(Math.abs(x.chassisPitch - a) < 1e-10);
  assert.ok(Math.abs(x.robotLean) < 1e-10);
});

test('lean comes from feet midpoint, regardless of chassis pitch', () => {
  const x = measuredAttitude(I, [.07,0,.25], [0,-.08,0], [0,.08,0]);
  assert.ok(Math.abs(x.robotLean - Math.atan2(.07,.25)) < 1e-10);
});

test('robot lean follows heading, not fixed world X', () => {
  const yaw90 = [0,-1,0, 1,0,0, 0,0,1];
  const x = measuredAttitude(yaw90, [0,.1,.25], [0,-.08,0], [0,.08,0]);
  assert.ok(Math.abs(x.robotLean - Math.atan2(.1,.25)) < 1e-10);
});
