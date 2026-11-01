// Read-only, planner-side terrain costs. Never promote curriculum exposure to
// a claim of physical traversability. Heights come from the viewer's actual
// physical park mesh; known steps/logs remain explicit keep-out obstacles.
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const NAV_BOUNDS_M=5.6;
export const NAV_CLEARANCE_M=.62;

export function buildNavigationTerrain(heightAt,{
    bound=NAV_BOUNDS_M,cell=.28,cautionDegrees=5,avoidDegrees=12}={}){
  if(typeof heightAt!=='function')return null;
  if(!(avoidDegrees>cautionDegrees&&cautionDegrees>=0))
    throw new Error('Invalid navigation terrain thresholds');
  const cells=Math.ceil(2*bound/cell)+1;
  const spacing=2*bound/(cells-1);
  const risk=new Array(cells*cells);
  const h=Math.max(.08,spacing*.48);
  for(let j=0;j<cells;j++)for(let i=0;i<cells;i++){
    const x=-bound+i*spacing,y=-bound+j*spacing;
    const dx=(heightAt(x+h,y)-heightAt(x-h,y))/(2*h);
    const dy=(heightAt(x,y+h)-heightAt(x,y-h))/(2*h);
    const grade=Math.atan(Math.hypot(dx,dy))*180/Math.PI;
    // Infinity is a prohibited slope, values 0..1 represent caution.
    risk[j*cells+i]=grade>avoidDegrees ? -1 :
      clamp((grade-cautionDegrees)/(avoidDegrees-cautionDegrees),0,1);
  }
  return {bound,spacing,cells,risk,cautionDegrees,avoidDegrees};
}

export function terrainRiskAt(terrain,x,y){
  if(!terrain)return 0;
  const {bound,spacing,cells,risk}=terrain;
  if(!Number.isFinite(x+y)||Math.abs(x)>bound||Math.abs(y)>bound)return Infinity;
  const i=clamp(Math.round((x+bound)/spacing),0,cells-1);
  const j=clamp(Math.round((y+bound)/spacing),0,cells-1);
  const value=risk[j*cells+i];
  if(!Number.isFinite(value)||value<0)return Infinity;
  return value;
}

export function terrainCostAt(terrain,x,y){
  const r=terrainRiskAt(terrain,x,y);
  return Number.isFinite(r)?1+1.75*r*r:Infinity;
}

export function sweptTerrainCost(terrain,a,b,step=.13){
  if(!terrain)return 1;
  const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/step));
  let total=0;
  for(let i=0;i<=n;i++){
    const t=i/n,cost=terrainCostAt(terrain,
      a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t);
    if(!Number.isFinite(cost))return Infinity;
    total+=cost;
  }
  return total/(n+1);
}
