// Infinite-coordinate H40 procedural terrain, bounded physical CPU window.
// The model contains 225 recycled MuJoCo 2x2 heightfields, not new bodies.
// Changing model.hfield_data is supported by MuJoCo at runtime, whereas
// editing compiled collision-mesh vertices would be unsafe. This is viewer
// geometry (two top triangles, vertical sides), not an MJX-prism replay.
export const LAB_TILE_M=1;
export const LAB_FIELD_SIDE=15; // 7m physical margin around the player
export const LAB_FIELD_HALF=7;
export const LAB_TILE_COUNT=LAB_FIELD_SIDE*LAB_FIELD_SIDE;
export const LAB_HFIELD_COUNT=LAB_TILE_COUNT;
export const LAB_HFIELD_Z=.5;
export const LAB_HIDDEN_Z=-40;
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const number=(x,d=0)=>Number.isFinite(Number(x))?Number(x):d;
const mod=(x,n)=>((x%n)+n)%n;
const hash=(i,j,seed)=>{let a=(Math.imul(i|0,374761393)^
  Math.imul(j|0,668265263)^(seed|0))|0;
  a=Math.imul(a^(a>>>13),1274126177);return((a^(a>>>16))>>>0)/4294967296;};
const smooth=t=>t*t*(3-2*t);
function noise(x,y,seed){
  const i=Math.floor(x),j=Math.floor(y),u=smooth(x-i),v=smooth(y-j);
  const a=hash(i,j,seed),b=hash(i+1,j,seed),c=hash(i,j+1,seed),d=hash(i+1,j+1,seed);
  return ((a+(b-a)*u)*(1-v)+(c+(d-c)*u)*v)*2-1;
}

// World +Y is NORTH; X is east/west. Latitude governs the envelope, while
// the global seed/coordinates make scenery and geology repeat on any revisit.
// It reaches a cap rather than demanding physically impossible stairs.
export function carpetLatitudeDifficulty(y){
  return clamp((y+6)/42,0,1);
}
export function carpetLocalLimits(y,config={}){
  const d=carpetLatitudeDifficulty(y);
  return {difficulty:d,
    slopeDeg:clamp(number(config.slopeDeg,10),0,15)*(.08+.92*d),
    stepUpM:clamp(number(config.stepUpM,.025),0,.15)*d,
    stepDownM:clamp(number(config.stepDownM,.025),0,.15)*d,
    resilience:d};
}
function continuousRelief(x,y,config){
  const seed=Math.floor(number(config.seed,735));
  const caps=carpetLocalLimits(y,config);
  const slope=Math.tan(caps.slopeDeg*Math.PI/180);
  // Smooth global vertex noise. Max derivatives are bounded by the user
  // slope envelope; no strong north-facing ramp accumulates without limit.
  // Two wavelengths give long walkable grades and small foothold
  // undulations; at the northern cap they approach (but should not
  // routinely exceed) the configured slope ceiling.
  return (noise(x/3.4,y/3.4,seed)*.48+
    noise(x/1.9+.3,y/1.9-.7,seed+17)*.17)*slope;
}
export function carpetTileCorners(i,j,config={}){
  const seed=Math.floor(number(config.seed,735));
  const l=carpetLocalLimits(j+.5,config);
  const h=hash(i,j,seed+91);
  // Independent per-tile signed lips, no seam on flat/easy south when
  // requested up/down both zero; same x/y tile is bit-deterministic.
  const signed=(h>=.5?(h-.5)*2*l.stepUpM:-(.5-h)*2*l.stepDownM);
  return [continuousRelief(i,j,config)+signed,
    continuousRelief(i+1,j,config)+signed,
    continuousRelief(i,j+1,config)+signed,
    continuousRelief(i+1,j+1,config)+signed];
}
export function fieldSlot(i,j){
  return mod(j,LAB_FIELD_SIDE)*LAB_FIELD_SIDE+mod(i,LAB_FIELD_SIDE);
}
function worldForSlot(anchor,slot){
  const sx=slot%LAB_FIELD_SIDE,sy=Math.floor(slot/LAB_FIELD_SIDE);
  return [anchor[0]+mod(sx-mod(anchor[0],LAB_FIELD_SIDE),LAB_FIELD_SIDE),
    anchor[1]+mod(sy-mod(anchor[1],LAB_FIELD_SIDE),LAB_FIELD_SIDE)];
}
export function createSensorLabCarpet(config={}){
  const tiles=Array.from({length:LAB_TILE_COUNT},(_,slot)=>({
    slot,i:0,j:0,x:0,y:0,active:false,corners:[0,0,0,0],base:0
  }));
  const initial=[-LAB_FIELD_HALF,-LAB_FIELD_HALF];
  let anchor=initial.slice();
  function fill(tile,i,j){
    tile.i=i;tile.j=j;tile.x=i+.5;tile.y=j+.5;
    tile.corners=carpetTileCorners(i,j,config);
    tile.base=Math.min(...tile.corners);
    tile.active=true;
  }
  for(const t of tiles){const [i,j]=worldForSlot(anchor,t.slot);fill(t,i,j);}
  const at=(i,j)=>{const t=tiles[fieldSlot(i,j)];
    return t.i===i&&t.j===j&&t.active?t:null;};
  const sampleHeight=(x,y)=>{
    if(!Number.isFinite(x)||!Number.isFinite(y))return NaN;
    const i=Math.floor(x),j=Math.floor(y),t=at(i,j);
    if(!t)return NaN;
    const u=x-i,v=y-j,[a,b,c,d]=t.corners;
    return u+v<=1?a+(b-a)*u+(c-a)*v:d+(c-d)*(1-u)+(b-d)*(1-v);
  };
  const park={mode:'carpet',physics:'streaming-heightfield',tiles,
    seed:Math.floor(number(config.seed,735)),
    tileSizeM:LAB_TILE_M,cell:LAB_TILE_M,radius:LAB_FIELD_HALF,
    limits:config,features:[],trees:[],flatRadius:0,
    revision:0,at,sampleHeight,window:()=>tiles,
    difficultyAt:y=>carpetLocalLimits(y,config),
    resetGrowth(){
      anchor=initial.slice();
      this.anchor=anchor.slice();
      for(const tile of tiles){const [i,j]=worldForSlot(anchor,tile.slot);fill(tile,i,j);}
      this.revision++;
      return tiles;
    },
    // Move ONLY the exhausted edge slots, never a foot's support.
    // The 7m half-width means a healthy foot is >5m from evicted ground.
    // A full-window jump (e.g. user teleport) is rejected, not simulated.
    growAround(x,y,feet=[]){
      if(![x,y].every(Number.isFinite))return [];
      const cx=Math.floor(x),cy=Math.floor(y);
      const desired=[cx-LAB_FIELD_HALF,cy-LAB_FIELD_HALF];
      if(desired[0]===anchor[0]&&desired[1]===anchor[1])return [];
      if(desired.some((n,k)=>Math.abs(n-anchor[k])>1))return [];
      const changes=[];
      for(const tile of tiles){
        const [i,j]=worldForSlot(desired,tile.slot);
        if(i===tile.i&&j===tile.j)continue;
        const margin=.5+ .004;
        if(feet.some(p=>Math.abs(p.x-tile.x)<=margin+number(p.radius) &&
                        Math.abs(p.y-tile.y)<=margin+number(p.radius)))return [];
        changes.push([tile,i,j]);
      }
      for(const [tile,i,j] of changes)fill(tile,i,j);
      anchor=desired;this.anchor=desired.slice();this.revision++;
      return changes.map(([tile])=>tile);
    },
    count(){return LAB_TILE_COUNT;},
    // Decorative sampling never provides physical contact or ToF range.
    previewHeight:(x,y)=>{const i=Math.floor(x),j=Math.floor(y);
      const c=carpetTileCorners(i,j,config),u=x-i,v=y-j;
      return u+v<=1?c[0]+(c[1]-c[0])*u+(c[2]-c[0])*v:
        c[3]+(c[2]-c[3])*(1-u)+(c[1]-c[3])*(1-v);}
  };
  park.anchor=anchor.slice();
  return park;
}
export function carpetHeightfieldPatch(tile){
  return tile.corners.map(h=>clamp((h-tile.base)/LAB_HFIELD_Z,0,1));
}
export function carpetSpawnSoleLift(park,geomXpos,geomSize,footGeomIds){
  if(park?.physics!=='streaming-heightfield'||footGeomIds?.length!==2)
    throw Error('Carpet spawn requires two H40 foot spheres');
  return Math.max(...footGeomIds.map(id=>{
    const p=3*id,x=Number(geomXpos[p]),y=Number(geomXpos[p+1]),
      radius=Number(geomSize[p]),z=Number(geomXpos[p+2]),
      ground=park.sampleHeight(x,y);
    if(![x,y,radius,z,ground].every(Number.isFinite)||radius<=0)
      throw Error('Carpet feet require valid prism support');
    const corners=park.at(Math.floor(x),Math.floor(y)).corners;
    const gx=(corners[1]-corners[0]),gy=(corners[2]-corners[0]);
    return ground+radius*Math.hypot(1,gx,gy)-z;
  }));
}
const fmt=n=>Number(n).toFixed(7);
export function streamingCarpetXml(park){
  if(park?.physics!=='streaming-heightfield'||park.tiles.length!==LAB_TILE_COUNT)
    throw Error('Streaming Carpet requires exactly 225 source tiles');
  const assets=[],bodies=[];
  for(const t of park.tiles){
    assets.push('<hfield name="sensor_lab_hfield_'+t.slot+
      '" nrow="2" ncol="2" size="0.5 0.5 '+LAB_HFIELD_Z+' 0.5" elevation="'+
      carpetHeightfieldPatch(t).map(fmt).join(' ')+'"/>');
    bodies.push('<body name="sensor_lab_tile_'+t.slot+
      '" mocap="true" pos="'+[t.x,t.y,t.base].map(fmt).join(' ')+'">'+
      '<geom name="sensor_lab_ground_'+t.slot+
      '" type="hfield" hfield="sensor_lab_hfield_'+t.slot+
      '" contype="1" conaffinity="0" friction="1.5 0.005 0.0001" '+
      'rgba="0.39 0.65 0.39 1"/></body>');
  }
  return {assets:assets.join('\n'),bodies:bodies.join('\n')};
}
