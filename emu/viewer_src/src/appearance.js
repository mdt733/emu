// Pure spatial helpers and one-time visual assets. No MuJoCo or PPO coupling.
export const GRID_STEP_M = 0.10;
export const GRID_MAJOR_M = 1.0;
export const FLOOR_SIZE_M = 40;
export const TRAIL_CAPACITY = 180;

// A visually infinite plane snaps by complete 1-m texture tiles. This keeps
// world-space grid lines stationary when the camera follows a walking robot.
export function floorTileOrigin(metres) {
  if (!Number.isFinite(metres)) throw new Error('invalid floor position');
  return Math.round(metres / GRID_MAJOR_M) * GRID_MAJOR_M || 0;
}

export class GroundTrail {
  constructor(maxPoints = TRAIL_CAPACITY, spacing = 0.035) {
    if (!Number.isInteger(maxPoints) || maxPoints < 2 || !(spacing > 0)) throw new Error('invalid trail configuration');
    this.maxPoints = maxPoints;
    this.spacingSq = spacing * spacing;
    this.coords = new Float32Array(maxPoints * 3);
    this.count = 0;
  }
  clear() { this.count = 0; }
  add(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    if (this.count) {
      const offset = (this.count - 1) * 3;
      const dx = x - this.coords[offset];
      const dy = y - this.coords[offset + 1];
      if (dx * dx + dy * dy < this.spacingSq) return false;
    }
    if (this.count === this.maxPoints) {
      this.coords.copyWithin(0, 3);
      this.count--;
    }
    const offset = this.count * 3;
    this.coords[offset] = x;
    this.coords[offset + 1] = y;
    this.coords[offset + 2] = 0.006;
    this.count++;
    return true;
  }
}

export function createGridTexture(THREE) {
  // One 1-m tile drawn once and wrapped across the floor. Not 40,000 meshes.
  const side = 512;
  const canvas = document.createElement('canvas');
  canvas.width = side; canvas.height = side;
  const g = canvas.getContext('2d');
  if (!g) throw new Error('2D canvas unavailable for floor texture');
  g.fillStyle = '#354451';
  g.fillRect(0, 0, side, side);
  for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) {
    if ((x + y) % 2 === 0) {
      g.fillStyle = 'rgba(255,255,255,0.018)';
      g.fillRect(x * side / 10, y * side / 10, side / 10, side / 10);
    }
  }
  for (let i = 0; i <= 10; i++) {
    const isMajor = i === 0 || i === 10;
    const mid = i === 5;
    g.beginPath();
    g.strokeStyle = isMajor ? 'rgba(177,202,215,0.57)' : mid ? 'rgba(170,194,211,0.38)' : 'rgba(143,174,194,0.24)';
    g.lineWidth = isMajor ? 2.8 : mid ? 1.7 : 1.0;
    const d = i * side / 10;
    g.moveTo(d, 0); g.lineTo(d, side);
    g.moveTo(0, d); g.lineTo(side, d);
    g.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(FLOOR_SIZE_M / GRID_MAJOR_M, FLOOR_SIZE_M / GRID_MAJOR_M);
  texture.anisotropy = 4;
  return texture;
}

export function createSkyTexture(THREE) {
  // A cheap screen-space atmosphere instead of an HDR cubemap or sky renderer.
  const canvas = document.createElement('canvas');
  canvas.width = 8; canvas.height = 256;
  const g = canvas.getContext('2d');
  if (!g) throw new Error('2D canvas unavailable for sky texture');
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, '#101b2b');
  grad.addColorStop(0.52, '#1d3342');
  grad.addColorStop(0.76, '#2b414a');
  grad.addColorStop(1, '#253640');
  g.fillStyle = grad;
  g.fillRect(0, 0, 8, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
