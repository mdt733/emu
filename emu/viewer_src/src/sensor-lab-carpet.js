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
export const LAB_HFIELD_Z=1.0;
export const LAB_HIDDEN_Z=-40;
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const number=(x,d=0)=>Number.isFinite(Number(x))?Number(x):d;
const mod=(x,n)=>((x%n)+n)%n;
const hash=(i,j,seed)=>{let a=(Math.imul(i|0,374761393)^
  Math.imul(j|0,668265263)^(seed|0))|0;
  a=Math.imul(a^(a>>>13),1274126177);return((a^(a>>>16))>>>0)/4294967296;};
// Gradient noise is spatially continuous and has NO repeating sine
// wavelengths. Hash random lattice *directions*, not heights; quintic
// interpolation gives smooth terrain derivatives even across noise cells.
// Precompute 16 directions once, never use trig during tile generation.
const TAU=Math.PI*2;
const GRADIENTS=Array.from({length:16},(_,k)=>[
  Math.cos(k*TAU/16),Math.sin(k*TAU/16)
]);
const fade=t=>t*t*t*(t*(t*6-15)+10);
function gradientNoise(x,y,seed){
  const i=Math.floor(x),j=Math.floor(y),u=x-i,v=y-j;
  const dot=(a,b,dx,dy)=>{
    const g=GRADIENTS[Math.floor(hash(a,b,seed)*16)];
    return g[0]*dx+g[1]*dy;
  };
  const a=dot(i,j,u,v),b=dot(i+1,j,u-1,v),
        c=dot(i,j+1,u,v-1),d=dot(i+1,j+1,u-1,v-1);
  const sx=fade(u),sy=fade(v);
  return (a+(b-a)*sx)*(1-sy)+(c+(d-c)*sx)*sy;
}

// One authoritative viewer difficulty contract. Unlike the legacy specialist
// inspector, these settings directly control terrain at the START latitude.
// North (+Y) scales them smoothly; east/west has new features at equal grade.
export const MEADOW_DEFAULTS=Object.freeze({
  // Start on a walkable rolling meadow; users can choose Rugged 10°.
  seed:735,hillPeakDeg:6,hillScaleM:9.5,
  stepUpM:.007,stepDownM:.007,stepFrequency:.14,
  northGainPerM:.02
});
export function normalizedMeadowSettings(changes={}){
  const c={...MEADOW_DEFAULTS,...changes};
  const value=(v,fallback,lo,hi)=>
    clamp(Number.isFinite(Number(v))?Number(v):fallback,lo,hi);
  return {
    seed:Math.floor(value(c.seed,735,0,4294967295)),
    hillPeakDeg:value(c.hillPeakDeg,6,0,20),
    hillScaleM:value(c.hillScaleM,9.5,3,18),
    stepUpM:value(c.stepUpM,.007,0,.1),
    stepDownM:value(c.stepDownM,.007,0,.1),
    stepFrequency:value(c.stepFrequency,.14,0,.75),
    northGainPerM:value(c.northGainPerM,.02,0,.06)
  };
}
export function carpetLatitudeDifficulty(y,config=MEADOW_DEFAULTS){
  const settings=normalizedMeadowSettings(config);
  return clamp(1+settings.northGainPerM*y,.08,1.8);
}
export function carpetLocalLimits(y,config=MEADOW_DEFAULTS){
  const c=normalizedMeadowSettings(config);
  const multiple=carpetLatitudeDifficulty(y,c);
  return {difficulty:multiple,
    slopeDeg:clamp(c.hillPeakDeg*multiple,0,20),
    stepUpM:clamp(c.stepUpM*multiple,0,.12),
    stepDownM:clamp(c.stepDownM*multiple,0,.12),
    stepFrequency:clamp(c.stepFrequency*multiple,0,.9),
    resilience:clamp(multiple/1.8,0,1)};
}
function continuousRelief(x,y,config){
  const c=normalizedMeadowSettings(config);
  const peak=Math.tan(carpetLocalLimits(y,c).slopeDeg*Math.PI/180);
  if(peak===0)return 0;
  const scale=c.hillScaleM/9.5,seed=c.seed;
  const u=x/scale,v=y/scale;

  // Gentle, large-scale domain warp breaks the straight contours and tidy
  // parallel ridges of the old three-sine-wave terrain. Independent warp
  // vectors wind the hill crests and valleys across *global* world space;
  // adjacent squares always call this same continuous height function.
  const wx=5.8*gradientNoise(u/19+.31,v/18-.82,seed+11)+
           1.15*gradientNoise(u/8-.73,v/9+.42,seed+13);
  const wy=5.8*gradientNoise(u/20-.67,v/17+.38,seed+17)+
           1.15*gradientNoise(u/9+.59,v/8+.13,seed+19);
  const X=u+wx,Y=v+wy;

  // Hierarchical landforms: broad watersheds (~13 m), irregular rolling
  // hills (~6 m), small weathered undulations (~4 m), and a subdued
  // ridged/valley signal. Octaves use different orientations and seeds,
  // so slopes bend, crest, flatten and change compass direction naturally.
  // This is cheap procedural geomorphology, not simulated water erosion.
  const broad=13*1.20*gradientNoise(X/13+.2,Y/12-.5,seed+31);
  const rolling=6.4*.77*gradientNoise(
    (.88*X-.48*Y)/6.4,(.48*X+.88*Y)/6.4,seed+37);
  const detail=3.7*.25*gradientNoise(
    (.68*X+.73*Y)/3.7,(-.73*X+.68*Y)/3.7,seed+41);
  const basin=gradientNoise(X/11-.4,Y/10+.2,seed+47);
  const ridges=11*.34*(.24-Math.abs(basin));

  // Calibrated against ACTUAL 1m MuJoCo triangle slopes: a selected
  // 10° frontier corresponds to ~10° near the 90th percentile, not a
  // uniform ten-degree hillside. Most terrain is considerably gentler.
  // Increasing hill size expands the landforms without changing typical
  // grades. The +Y latitude envelope remains the sole difficulty control.
  return peak*scale*.49*(broad+rolling+detail+ridges);
}
export function carpetTileCorners(i,j,config=MEADOW_DEFAULTS){
  const c=normalizedMeadowSettings(config),l=carpetLocalLimits(j+.5,c);
  const h=hash(i,j,c.seed+91),event=hash(i,j,c.seed+73);
  const eligible=event<l.stepFrequency;
  const signed=!eligible?0:(h>=.5
    ?(h-.5)*2*l.stepUpM:-(.5-h)*2*l.stepDownM);
  return [
    continuousRelief(i,j,c)+signed,
    continuousRelief(i+1,j,c)+signed,
    continuousRelief(i,j+1,c)+signed,
    continuousRelief(i+1,j+1,c)+signed
  ];
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
  const values=tile.corners.map(h=>(h-tile.base)/LAB_HFIELD_Z);
  // No invisible clipping: slider grades must be physically representable
  // by the compiled 1m vertical range or the scene is invalid.
  if(values.some(v=>!Number.isFinite(v)||v<-.000001||v>1.000001))
    throw Error('Meadow hill exceeds compiled heightfield vertical range');
  return values.map(v=>clamp(v,0,1));
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
    // IMPORTANT MuJoCo normalizes XML elevation to [0,1] at compile
    // time. Supplying actual 0.1m differences here would incorrectly
    // expand them to a 1m cliff. Allocate ZERO data in XML and write the
    // correct metric samples directly to model.hfield_data at runtime.
    assets.push('<hfield name="sensor_lab_hfield_'+t.slot+
      '" nrow="2" ncol="2" size="0.5 0.5 '+LAB_HFIELD_Z+' 0.5"/>');
    bodies.push('<body name="sensor_lab_tile_'+t.slot+
      '" mocap="true" pos="'+[t.x,t.y,t.base].map(fmt).join(' ')+'">'+
      '<geom name="sensor_lab_ground_'+t.slot+
      '" type="hfield" hfield="sensor_lab_hfield_'+t.slot+
      '" contype="1" conaffinity="0" friction="1.5 0.005 0.0001" '+
      'rgba="0.39 0.65 0.39 1"/></body>');
  }
  return {assets:assets.join('\n'),bodies:bodies.join('\n')};
}
