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


// Procedural grass: one tiny repeated texture instead of thousands of blades
// or downloaded assets. One texture tile covers exactly one world metre.
export function createGrassTexture(THREE, grid=false) {
  const side=512, canvas=document.createElement('canvas');
  canvas.width=canvas.height=side;
  const g=canvas.getContext('2d');
  if(!g)throw new Error('No 2D canvas for grass');
  g.fillStyle='#729b50';g.fillRect(0,0,side,side);
  let state=733;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)|0;return(state>>>0)/4294967296;};
  for(let i=0;i<5700;i++){
    const x=random()*side,y=random()*side,len=1+random()*5;
    const light=random()>0.5;
    g.strokeStyle=light?'rgba(188,214,123,0.22)':'rgba(29,91,27,0.15)';
    g.lineWidth=0.6+random()*0.8;
    g.beginPath();g.moveTo(x,y);g.lineTo(x+(random()-0.5)*3,y-len);g.stroke();
  }
  if(grid){
    for(let i=0;i<=10;i++){
      const p=i*side/10;
      g.strokeStyle=i===0||i===10?'rgba(245,255,225,0.35)':'rgba(236,251,203,0.16)';
      g.lineWidth=i===0||i===10?1.7:0.8;
      g.beginPath();g.moveTo(p,0);g.lineTo(p,side);g.moveTo(0,p);g.lineTo(side,p);g.stroke();
    }
  }
  const texture=new THREE.CanvasTexture(canvas);
  texture.colorSpace=THREE.SRGBColorSpace;
  texture.wrapS=texture.wrapT=THREE.RepeatWrapping;
  texture.anisotropy=4;
  return texture;
}

// Separate transparent overlay: the grass texture stays readable below the
// grid. Unlike baking faint lines into shaded green pixels, this is rendered
// after the underlying park, on exactly the same collision-derived geometry.
// A 1-m repeat contains ten 10-cm cells and a strong 1-m boundary.
export function createSkyTexture(THREE) {
  const canvas=document.createElement('canvas');
  canvas.width=512;canvas.height=256;
  const g=canvas.getContext('2d');
  if(!g)throw new Error('2D canvas unavailable for sky');
  const sky=g.createLinearGradient(0,0,0,256);
  sky.addColorStop(0,'#3888d5');
  sky.addColorStop(0.46,'#7db9e6');
  sky.addColorStop(0.76,'#bfdfee');
  sky.addColorStop(1,'#d9e7d0');
  g.fillStyle=sky;g.fillRect(0,0,512,256);
  // Wispy clouds are painted once, not generated as volumetric objects.
  for(const [x,y,s] of [[75,65,1],[295,91,0.8],[440,47,1.3],[194,135,0.6]]){
    const cloud=g.createRadialGradient(x,y,2,x,y,55*s);
    cloud.addColorStop(0,'rgba(255,255,255,0.38)');
    cloud.addColorStop(0.47,'rgba(255,255,255,0.21)');
    cloud.addColorStop(1,'rgba(255,255,255,0)');
    g.save();g.translate(x,y);g.scale(1.7,0.43);g.translate(-x,-y);
    g.fillStyle=cloud;g.fillRect(x-60*s,y-60*s,120*s,120*s);
    g.restore();
  }
  const texture=new THREE.CanvasTexture(canvas);
  texture.colorSpace=THREE.SRGBColorSpace;
  return texture;
}
