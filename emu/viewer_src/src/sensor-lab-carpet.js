// Shared CPU MuJoCo geometry for trained actor preview AND sensor calibration.
// Fixed 2x2 metre fixture: four 1m tiles and eight real triangular prisms.
// The GPU MJX implementation recycles its four tiles; this fixture does not.
// Source of the prism topology: emu/carpet_prism_mjx.py.
export const LAB_TILE_M = 1;
export const LAB_TILE_COUNT = 4;
export const LAB_PRISM_COUNT = 8;
const DEPTH = .65;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const num=(v,def=0)=>Number.isFinite(Number(v))?Number(v):def;
const triangles=[
  [[-.5,-.5],[.5,-.5],[-.5,.5]],
  [[.5,.5],[-.5,.5],[.5,-.5]]
];
const faces=[[0,1,2],[5,4,3],[0,3,4],[0,4,1],
  [1,4,5],[1,5,2],[2,5,3],[2,3,0]];
const f=x=>Number(x).toFixed(7);
const heightsFor=(i,j,config)=>{
  const slope=Math.tan(clamp(num(config.slopeDeg),0,15)*Math.PI/180);
  const rise=clamp(num(config.stepUpM),0,.15);
  const drop=clamp(num(config.stepDownM),0,.15);
  // Exact grade: on level transitions the four tiles form ONE continuous
  // x-aligned slope. Any deliberate rise/drop is a real vertical seam.
  // No multiplier that turns a requested 15-degree slope into 10 degrees.
  const base=slope*(i+.5)+(i?rise:0)-(j?drop:0);
  const gx=slope,gy=0;
  return [base-gx*.5-gy*.5,base+gx*.5-gy*.5,
          base-gx*.5+gy*.5,base+gx*.5+gy*.5];
};
export function createSensorLabCarpet(config={}) {
  const tiles=[];
  for(let j=0;j<2;j++)for(let i=0;i<2;i++){
    tiles.push({i,j,slot:i+2*j,corners:heightsFor(i,j,config),
      x:i+.5,y:j+.5});
  }
  const at=(i,j)=>tiles.find(t=>t.i===i&&t.j===j)||null;
  const sampleHeight=(x,y)=>{
    const i=Math.floor(x),j=Math.floor(y),t=at(i,j);
    if(!t)return NaN; // OUTSIDE fixture is void, never invisible floor.
    const u=x-i,v=y-j,[a,b,c,d]=t.corners;
    return u+v<=1?a+(b-a)*u+(c-a)*v:
      d+(c-d)*(1-u)+(b-d)*(1-v);
  };
  return {mode:'carpet',physics:'eight-1m-prisms',tiles,seed:735,
    tileSizeM:LAB_TILE_M,cell:LAB_TILE_M,radius:2,
    features:[],trees:[],limits:config,flatRadius:0,
    at,sampleHeight,window:()=>tiles};
}
// Render and MuJoCo assets consume identical vertices: no second terrain
// geometry in the browser's ray-caster.
export function sensorLabPrismVertices(park) {
  if(park?.physics!=='eight-1m-prisms'||park.tiles?.length!==4)
    throw Error('Sensor lab expects four 1m physical tiles');
  const prism=[];
  for(const tile of park.tiles)for(let triangle=0;triangle<2;triangle++){
    const vertex=triangles[triangle].map(([x,y])=>{
      const u=x+.5,v=y+.5,[a,b,c,d]=tile.corners;
      const z=triangle===0?a+(b-a)*u+(c-a)*v:
        d+(c-d)*(1-u)+(b-d)*(1-v);
      return [x,y,z];
    });
    const local=[...vertex,...vertex.map(([x,y,z])=>[x,y,z-DEPTH])];
    prism.push({tile,triangle,k:tile.slot*2+triangle,local,
      world:local.map(([x,y,z])=>[tile.x+x,tile.y+y,z])});
  }
  return prism;
}
export const LAB_PRISM_FACES = Object.freeze(faces.map(f=>Object.freeze([...f])));

export function sensorLabPrismXml(park){
  if(park?.physics!=='eight-1m-prisms'||park.tiles?.length!==4)
    throw Error('Sensor lab expects four 1m physical tiles');
  const assets=[],bodies=[];
  for(const {tile,k,local:vert} of sensorLabPrismVertices(park)){
    assets.push('<mesh name="sensor_lab_mesh_'+k+'" vertex="'+
      vert.flat().map(f).join(' ')+'" face="'+faces.flat().join(' ')+'"/>');
    bodies.push('<body name="sensor_lab_tile_'+k+
      '" mocap="true" pos="'+[tile.x,tile.y,0].map(f).join(' ')+'">'+
      '<geom name="sensor_lab_prism_'+k+'" type="mesh" mesh="sensor_lab_mesh_'+k+
      '" contype="1" conaffinity="0" friction="1.5 0.005 0.0001" '+
      'rgba="0.39 0.65 0.39 1"/></body>');
  }
  return {assets:assets.join('\n'),bodies:bodies.join('\n')};
}
