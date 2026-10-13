import test from 'node:test';
import assert from 'node:assert/strict';
import { joystickToCommands } from '../src/controls.js';

const limits = {vx_min: -0.2, vx_max: 0.6, vy_abs: 0.22, yaw_abs: 1.5, pitch_abs_rad: 0.08};

test('manual joystick right/left are reversed from old vy mapping, yaw stays unchanged', () => {
  const right = joystickToCommands(1, 0, 0, 0, limits);
  const left = joystickToCommands(-1, 0, 0, 0, limits);
  assert.equal(right.vy, -0.22);
  assert.equal(left.vy, 0.22);
  assert.equal(Math.abs(right.yaw), 0);
  assert.equal(joystickToCommands(0, 0, 1, 0, limits).yaw, -1.5);
});

test('forward/back and pitch mappings are unchanged; all axes neutralize', () => {
  assert.equal(joystickToCommands(0, 1, 0, 0, limits).vx, 0.6);
  assert.equal(joystickToCommands(0, -1, 0, 0, limits).vx, -0.2);
  assert.equal(joystickToCommands(0, 0, 0, 1, limits).pitch, 0.08);
  assert.deepEqual(joystickToCommands(0, 0, 0, 0, limits), {vx: 0, vy: -0, yaw: -0, pitch: 0});
});
