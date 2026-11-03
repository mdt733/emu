// Pure, deterministic geometry for the EMU viewer-only triangular-prism lab.
// Model coordinates: [x, y, elevation]. No physics or THREE dependency.
export const TILE_SIZE_M = 0.5;
export const PRESETS = Object.freeze(['rolling', 'crest', 'valley', 'cross', 'saddle', 'flat']);
export const STEP_PATTERNS = Object.freeze(['none', 'up', 'down', 'up-down', 'staircase', 'checker']);
export const DEFAULT_PRISM_SETTINGS = Object.freeze({
  tiles: 5, seed: 735, surface: 'rolling', steps: 'up-down',
  slopeDeg: 8, transitionDeg: 5, stepUpMm: 25, stepDownMm: 18,
  diagonal: 'checker', exaggeration: 1,
});
const rad = deg => deg * Math.PI / 180;
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
const positiveMod = (n, m) => (n % m + m) % m;
function hash(a, b, seed, salt = 0) {
  let x = (Math.imul(a, 374761393) + Math.imul(b, 668265263) +
    Math.imul(seed, 1442695041) + Math.imul(salt, 1274126177)) | 0;
  x = Math.imul(x ^ (x >>> 13), 1274126177);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}
export function normalizePrismSettings(raw = {}) {
  const o = {...DEFAULT_PRISM_SETTINGS};
  for (const key of ['tiles', 'seed', 'slopeDeg', 'transitionDeg', 'stepUpMm', 'stepDownMm', 'exaggeration']) {
    if (raw[key] === undefined) continue;
    if (!Number.isFinite(Number(raw[key]))) throw Error(`Invalid prism setting: ${key}`);
    o[key] = Number(raw[key]);
  }
  o.tiles = [3, 5, 7, 9].includes(o.tiles) ? o.tiles : 5;
  o.seed = clamp(Math.floor(o.seed), 0, 2147483647);
  o.slopeDeg = clamp(o.slopeDeg, 0, 20);
  o.transitionDeg = clamp(o.transitionDeg, 0, 15);
  o.stepUpMm = clamp(o.stepUpMm, 0, 150);
  o.stepDownMm = clamp(o.stepDownMm, 0, 150);
  o.exaggeration = clamp(o.exaggeration, 1, 10);
  if (PRESETS.includes(raw.surface)) o.surface = raw.surface;
  if (STEP_PATTERNS.includes(raw.steps)) o.steps = raw.steps;
  if (['checker', 'slash', 'backslash'].includes(raw.diagonal)) o.diagonal = raw.diagonal;
  return o;
}
function tileOffset(i, j, s) {
  const up = s.stepUpMm / 1000;
  const down = s.stepDownMm / 1000;
  switch (s.steps) {
    case 'up': return i >= 0 ? up : 0;
    case 'down': return i >= 0 ? -down : 0;
    case 'up-down': return (i >= 0 ? up : 0) - (j >= 1 ? down : 0);
    case 'staircase': return Math.floor(i / 2) * up - Math.floor(j / 2) * down;
    case 'checker': return positiveMod(i + j, 2) ? up : -down;
    default: return 0;
  }
}
function continuousHeight(x, y, s) {
  const slope = Math.tan(rad(s.slopeDeg));
  const change = Math.tan(rad(s.transitionDeg));
  const phase = 2 * Math.PI * hash(0, 0, s.seed, 143);
  const freq = 1.35; // radians per metre; grade transition is a derivative, not another tile lip.
  switch (s.surface) {
    case 'flat': return 0;
    case 'crest': return slope * (-Math.sqrt(x*x + 0.10*0.10) + 0.10) + change / freq * Math.sin(freq * y + phase);
    case 'valley': return slope * (Math.sqrt(x*x + 0.10*0.10) - 0.10) + change / freq * Math.sin(freq * y + phase);
    case 'cross': return slope * y + change / freq * Math.sin(freq*x + phase);
    case 'saddle': return slope * 0.45 * x + change * (x*y) / 2.0;
    default: return slope * (0.87 * x + 0.28 * y) +
      change / freq * Math.sin(freq*x + phase) * Math.cos(freq*y - phase);
  }
}
function topTriangleNormals(tri) {
  const [a,b,c] = tri;
  const u = [b[0]-a[0],b[1]-a[1],b[2]-a[2]];
  const v = [c[0]-a[0],c[1]-a[1],c[2]-a[2]];
  const n = [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const l = Math.hypot(...n);
  if (l < 1e-12 || n[2] <= 0) throw Error('Degenerate or inverted top triangle');
  return n.map(x=>x/l);
}
function facePrism(tri, baseZ, tile, index) {
  const upper = tri;
  const lower = tri.map(p=>[p[0],p[1],baseZ]);
  const normal = topTriangleNormals(upper);
  return {tile:[tile.i,tile.j], index, upper, lower, normal,
    angleDeg: Math.atan2(Math.hypot(normal[0],normal[1]),normal[2]) * 180 / Math.PI};
}
function vertex(i, j, ox, oy, offset, s, heightScale) {
  const x = (i + ox) * TILE_SIZE_M, y = (j + oy) * TILE_SIZE_M;
  return [x,y,(continuousHeight(x,y,s) + offset)*heightScale];
}
export function buildPrismTerrain(raw = {}) {
  const settings = normalizePrismSettings(raw);
  const s = settings, radius = Math.floor(s.tiles / 2);
  const minI = -radius, maxI = radius, scale = s.exaggeration;
  const tileMap = new Map(), tiles = [], prisms = [], risers = [];
  const key = (i,j) => `${i},${j}`;
  let minimum = Infinity, maximum = -Infinity;
  for(let j=minI;j<=maxI;j++) for(let i=minI;i<=maxI;i++) {
    const offset = tileOffset(i,j,s);
    // Each tile gets independent offsets ONLY at explicitly stepped boundaries.
    // Without a step, adjoining tiles compute their vertices from the SAME world coordinate.
    const corners = [vertex(i,j,0,0,offset,s,scale),vertex(i,j,1,0,offset,s,scale),
      vertex(i,j,0,1,offset,s,scale),vertex(i,j,1,1,offset,s,scale)];
    for (const p of corners) {minimum=Math.min(minimum,p[2]);maximum=Math.max(maximum,p[2]);}
    const diagonal = s.diagonal === 'checker'
      ? (positiveMod(i+j,2) ? 'slash' : 'backslash') : s.diagonal;
    const [a,b,c,d] = corners;
    const tops = diagonal === 'slash' ? [[a,b,c],[b,d,c]] : [[a,b,d],[a,d,c]];
    const tile = {i,j,offset,diagonal,corners,tops};
    tiles.push(tile);tileMap.set(key(i,j),tile);
  }
  // Deep enough that all prisms remain convex solids even at extreme slopes and exaggeration.
  const baseZ = minimum - Math.max(0.20*scale, maximum-minimum + 0.03);
  for (const t of tiles) t.tops.forEach((tri,k)=>prisms.push(facePrism(tri,baseZ,t,k)));
  let accidentalMaxM = 0, intendedMaxM = 0, joinedEdges = 0;
  const edges = [];
  for (const t of tiles) for (const [di,dj] of [[1,0],[0,1]]) {
    const next=tileMap.get(key(t.i+di,t.j+dj));
    if(!next)continue;
    const left = di ? [t.corners[1],t.corners[3]] : [t.corners[2],t.corners[3]];
    const right= di ? [next.corners[0],next.corners[2]] : [next.corners[0],next.corners[1]];
    const deltaA=right[0][2]-left[0][2],deltaB=right[1][2]-left[1][2];
    const deliberate = Math.abs(next.offset-t.offset)>1e-12;
    // This records exactly the geometric seam: same XY edge on BOTH sides.
    const gap=Math.max(Math.abs(deltaA),Math.abs(deltaB));
    if (!deliberate) { joinedEdges++;accidentalMaxM=Math.max(accidentalMaxM,gap); }
    else {
      intendedMaxM=Math.max(intendedMaxM,gap);
      const high=deltaA>=0 ? right:left, low=deltaA>=0 ? left:right;
      risers.push({from:[t.i,t.j],to:[next.i,next.j],high,low,heightM:gap});
    }
    edges.push({a:left,b:right,deliberate,heightM:gap});
  }
  function sampleHeight(x,y) {
    const i=Math.floor(x/TILE_SIZE_M),j=Math.floor(y/TILE_SIZE_M);
    const t=tileMap.get(key(i,j));if(!t)return null;
    const u=x/TILE_SIZE_M-i,v=y/TILE_SIZE_M-j;
    // Barycentric interpolation of one of the two physically planar triangle tops.
    const [a,b,c,d]=t.corners;
    if(t.diagonal==='slash') return u+v<=1
      ? a[2]+u*(b[2]-a[2])+v*(c[2]-a[2])
      : d[2]+(1-u)*(c[2]-d[2])+(1-v)*(b[2]-d[2]);
    return u>=v
      ? a[2]+u*(b[2]-a[2])+v*(d[2]-b[2])
      : a[2]+v*(c[2]-a[2])+u*(d[2]-c[2]);
  }
  return {settings,tiles,prisms,risers,edges,baseZ,minimum,maximum,sampleHeight,
    stats:{tiles:tiles.length,prisms:prisms.length,joinedEdges,intentionalEdges:risers.length,
      accidentalMaxMm:accidentalMaxM*1000,intendedMaxMm:intendedMaxM*1000,
      minMm:minimum*1000,maxMm:maximum*1000}};
}