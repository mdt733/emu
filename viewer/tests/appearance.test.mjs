import test from 'node:test';
import assert from 'node:assert/strict';
import { GRID_STEP_M, GRID_MAJOR_M, FLOOR_SIZE_M, floorTileOrigin, GroundTrail, createGridTexture, createSkyTexture } from '../src/appearance.js';

test('grid is genuinely 10 cm and major lines repeat at exactly 1 m', () => {
  assert.equal(GRID_STEP_M, 0.1);
  assert.equal(GRID_MAJOR_M, 1);
  assert.equal(FLOOR_SIZE_M / GRID_MAJOR_M, 40);
  for (const x of [0, 0.12, 1.4, 12.33, -0.3, -11.5]) {
    const shift = floorTileOrigin(x);
    assert.ok(shift % GRID_MAJOR_M === 0);
    assert.ok(Math.abs(x - shift) <= 0.5);
  }
  assert.throws(() => floorTileOrigin(Infinity), /invalid/);
});

test('path history is bounded, world-anchored and cleared on reset', () => {
  const trail = new GroundTrail(3, 0.035);
  assert.equal(trail.add(0, 0), true);
  assert.equal(trail.add(0.01, 0), false);
  assert.equal(trail.add(0.04, 0), true);
  assert.equal(trail.add(0.08, 0), true);
  assert.equal(trail.add(0.12, 0), true);
  assert.equal(trail.count, 3);
  assert.ok(Math.abs(trail.coords[0] - 0.04) < 1e-6);
  assert.ok(Math.abs(trail.coords[6] - 0.12) < 1e-6);
  assert.equal(trail.add(NaN, 0), false);
  trail.clear();
  assert.equal(trail.count, 0);
  assert.equal(trail.add(-1, 1), true);
  assert.equal(trail.coords[0], -1);
});


test('floor and sky assets are generated once, without external images or meshes', () => {
  const old = globalThis.document;
  let strokes = 0;
  const ctx = {fillRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){strokes++;},
    createLinearGradient(){return {addColorStop(){}};}};
  globalThis.document = {createElement(type) {assert.equal(type, 'canvas'); return {width:0,height:0,getContext: () => ctx};}};
  class Texture {constructor(canvas){this.image = canvas;this.repeat = {set(x,y){this.x=x;this.y=y;}};}}
  const THREE = {CanvasTexture:Texture, SRGBColorSpace:'srgb', RepeatWrapping:'repeat'};
  try {
    const floor = createGridTexture(THREE);
    assert.equal(floor.image.width, 512);
    assert.equal(floor.image.height, 512);
    assert.equal(floor.repeat.x, 40);
    assert.equal(floor.repeat.y, 40);
    assert.equal(floor.wrapS, 'repeat');
    assert.equal(strokes, 11);
    const sky = createSkyTexture(THREE);
    assert.equal(sky.image.height, 256);
    assert.equal(sky.colorSpace, 'srgb');
  } finally {globalThis.document = old;}
});
