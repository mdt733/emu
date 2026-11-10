// The H40 browser's growing Meadow-style Carpet, still using the trained
// prism topologies, signed steps and sensor collision source.
// CPU MuJoCo compiles its geoms once. We preallocate a bounded 15x15 reserve,
// keep unused bodies far below the world, and reveal NEW tiles as EMU walks.
// Unlike the MJX recycler, walked-on terrain is NEVER taken away.
export const LAB_TILE_M=1;
export const LAB_FIELD_SIDE=15;
export const LAB_FIELD_HALF=Math.floor(LAB_FIELD_SIDE/2);
export const LAB_TILE_COUNT=LAB_FIELD_SIDE*LAB_FIELD_SIDE;
export const LAB_PRISM_COUNT=LAB_TILE_COUNT*2;
const DEPTH=.65;
const HIDDEN_Z=-40;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const num=(v,def=0)=>Number.isFinite(Number(v))?Number(v):def;
const wrap=(n,k)=>((n%k)+k)%k;
const triangles=[
  [[-.5,-.5],[.5,-.5],[-.5,.5]],
  [[.5,.5],[-.5,.5],[.5,-.5]]
];
const faces=[[0,1,2],[5,4,3],[0,3,4],[0,4,1],
  [1,4,5],[1,5,2],[2,5,3],[2,3,0]];
const f=n=>Number(n).toFixed(7);

// The original four birth tiles are exact. After them, periodic signed lips
// provide further physically real challenges on the SAME full-square prisms.
// Setting slope/up/down to zero really gives flat, continuous ground.
export function carpetTileCorners(i,j,config={}){
  const slope=Math.tan(clamp(num(config.slopeDeg),0,15)*Math.PI/180);
  const rise=clamp(num(config.stepUpM),0,.15);
  const drop=clamp(num(config.stepDownM),0,.15);
  const base=slope*(i+.5)+(wrap(i,4)===1?rise:0)-
    (wrap(j,5)===1?drop:0);
  return [base-slope*.5,base+slope*.5,
          base-slope*.5,base+slope*.5];
}

export function createSensorLabCarpet(config={}){
  const tiles=[],index=new Map();
  const key=(i,j)=>i+','+j;
  for(let j=-LAB_FIELD_HALF;j<=LAB_FIELD_HALF;j++)
    for(let i=-LAB_FIELD_HALF;i<=LAB_FIELD_HALF;i++){
      const slot=tiles.length,active=i>=0&&i<2&&j>=0&&j<2;
      const tile={i,j,slot,x:i+.5,y:j+.5,
        corners:carpetTileCorners(i,j,config),active};
      tiles.push(tile);index.set(key(i,j),tile);
    }
  const at=(i,j)=>{
    const t=index.get(key(i,j));return t?.active?t:null;
  };
  const sampleHeight=(x,y)=>{
    if(!Number.isFinite(x)||!Number.isFinite(y))return NaN;
    const i=Math.floor(x),j=Math.floor(y),tile=at(i,j);
    if(!tile)return NaN; // Unbuilt land is VOID, never an invisible plane.
    const u=x-i,v=y-j,[a,b,c,d]=tile.corners;
    return u+v<=1?a+(b-a)*u+(c-a)*v:
      d+(c-d)*(1-u)+(b-d)*(1-v);
  };
  const park={mode:'carpet',physics:'growing-prism-field',tiles,
    seed:Math.floor(num(config.seed,735)),limits:config,
    tileSizeM:LAB_TILE_M,cell:LAB_TILE_M,radius:LAB_FIELD_HALF,
    features:[],trees:[],flatRadius:0,revision:0,
    at,sampleHeight,window:()=>tiles.filter(t=>t.active),
    resetGrowth(){
      for(const t of tiles)t.active=t.i>=0&&t.i<2&&t.j>=0&&t.j<2;
      this.revision++;
    },
    growAround(x,y){
      if(![x,y].every(Number.isFinite))return [];
      const ix=Math.floor(x),iy=Math.floor(y),newTiles=[];
      // 2m lookahead on all sides keeps the 30cm H40 ToF footprint
      // grounded well before EMU reaches the actual fall-away edge.
      for(let j=iy-2;j<=iy+2;j++)for(let i=ix-2;i<=ix+2;i++){
        const t=index.get(key(i,j));
        if(!t||t.active)continue;
        t.active=true;newTiles.push(t);
      }
      if(newTiles.length)this.revision++;
      return newTiles;
    },
    count(){return tiles.reduce((n,t)=>n+(t.active?1:0),0);}
  };
  return park;
}

export function sensorLabPrismVertices(park,{activeOnly=false}={}){
  if(park?.physics!=='growing-prism-field'||park.tiles?.length!==LAB_TILE_COUNT)
    throw Error('Growing Carpet requires the 15x15 reserved prism field');
  const result=[];
  for(const tile of park.tiles){
    if(activeOnly&&!tile.active)continue;
    for(let triangle=0;triangle<2;triangle++){
      const vertex=triangles[triangle].map(([x,y])=>{
        const u=x+.5,v=y+.5,[a,b,c,d]=tile.corners;
        return [x,y,triangle===0?a+(b-a)*u+(c-a)*v:
          d+(c-d)*(1-u)+(b-d)*(1-v)];
      });
      const local=[...vertex,...vertex.map(([x,y,z])=>[x,y,z-DEPTH])];
      result.push({tile,triangle,k:tile.slot*2+triangle,local,
        world:local.map(([x,y,z])=>[tile.x+x,tile.y+y,z])});
    }
  }
  return result;
}
export const LAB_PRISM_FACES=Object.freeze(faces.map(a=>Object.freeze([...a])));

// Actual H40 MuJoCo ball feet, NOT body-height or decorative sky mesh.
export function carpetSpawnSoleLift(park,geomXpos,geomSize,footGeomIds){
  if(park?.physics!=='growing-prism-field'||footGeomIds?.length!==2)
    throw Error('Carpet spawn requires the two H40 foot spheres');
  const slope=Math.tan(clamp(num(park.limits.slopeDeg),0,15)*Math.PI/180);
  return Math.max(...footGeomIds.map(id=>{
    const o=3*id,x=Number(geomXpos[o]),y=Number(geomXpos[o+1]);
    const radius=Number(geomSize[o]),z=Number(geomXpos[o+2]);
    const ground=park.sampleHeight(x,y);
    if(![x,y,radius,z,ground].every(Number.isFinite)||radius<=0)
      throw Error('Carpet feet require valid prism support');
    return ground+radius*Math.hypot(1,slope)-z;
  }));
}

export function sensorLabPrismXml(park){
  if(park?.physics!=='growing-prism-field'||park.tiles?.length!==LAB_TILE_COUNT)
    throw Error('Growing Carpet requires 15x15 physical 1m tiles');
  const assets=[],bodies=[];
  for(const {tile,k,local:vert} of sensorLabPrismVertices(park)){
    assets.push('<mesh name="sensor_lab_mesh_'+k+'" vertex="'+
      vert.flat().map(f).join(' ')+'" face="'+faces.flat().join(' ')+'"/>');
    bodies.push('<body name="sensor_lab_tile_'+k+
      '" mocap="true" pos="'+[tile.x,tile.y,tile.active?0:HIDDEN_Z].map(f).join(' ')+'">'+
      '<geom name="sensor_lab_prism_'+k+'" type="mesh" mesh="sensor_lab_mesh_'+k+
      '" contype="1" conaffinity="0" friction="1.5 0.005 0.0001" '+
      'rgba="0.39 0.65 0.39 1"/></body>');
  }
  return {assets:assets.join('\n'),bodies:bodies.join('\n')};
}
