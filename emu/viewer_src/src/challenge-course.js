// EMU v5018 obstacle demonstrations: a REAL MuJoCo collision course, not a
// texture or fake hologram. The learning/exam environment remains independent.
// The existing park's physical grass heightfield remains. A gently flat
// annular lane centres each 5–20mm obstacle at the correct ground-relative
// height; trees are kept elsewhere but excluded from the route corridor.
//
// Historical evidence: 5/10/15/20 mm shallow BOX courses had some frozen
// 260M-policy crossings. Capsule-log WALKING has not been established.
// Selected tier is a viewer demonstration request, NOT an automatically
// measured level unlocked by training.
export const CHALLENGE_RADIUS_M = 4; // exactly the 4m existing navigation circle
export const CHALLENGE_COUNT = 8;
export const CHALLENGE_TIERS_MM = Object.freeze([0,5,10,15,20]);
export const CHALLENGE_PALETTE = Object.freeze({
  5:{label:'5 mm step',hex:0x4cc78b,rgba:'0.20 0.76 0.46 1'},
  10:{label:'10 mm step',hex:0x64a8f7,rgba:'0.30 0.59 0.98 1'},
  15:{label:'15 mm step',hex:0xe6b45b,rgba:'0.94 0.66 0.24 1'},
  20:{label:'20 mm step',hex:0xd986d1,rgba:'0.84 0.37 0.71 1'},
  log:{label:'Capsule log · experimental',hex:0xf28e56,
       rgba:'0.94 0.41 0.20 1'},
});
const fmt=n=>Number(n.toFixed(6));
const numberRange=(value,min,max,name)=>{
  if(!Number.isFinite(value)||value<min||value>max)
    throw new Error(name+' must be '+min+'–'+max);
};
export function generateChallengeCourse({tierMm=10,logs=false,
                                         radius=CHALLENGE_RADIUS_M,
                                         count=CHALLENGE_COUNT}={}){
  if(!CHALLENGE_TIERS_MM.includes(tierMm))
    throw new Error('Challenge difficulty must be 0, 5, 10, 15 or 20 mm');
  numberRange(radius,2.7,4.2,'Challenge ring radius');
  if(!Number.isInteger(count)||count<4||count>12)
    throw new Error('Challenge station count must be 4–12');
  const unlocked=CHALLENGE_TIERS_MM.filter(h=>h>0&&h<=tierMm);
  const stations=[];
  for(let i=0;i<count;i++){
    const a=2*Math.PI*i/count;
    const tangent=a+Math.PI/2; // advance anticlockwise
    const h=unlocked.length?unlocked[i%unlocked.length]:0;
    // One experimental log every fourth station, never implicitly
    // claiming that the frozen policy learned to traverse logs.
    const isLog=Boolean(logs&&tierMm>0&&i%4===3);
    const palette=isLog?CHALLENGE_PALETTE.log:CHALLENGE_PALETTE[h];
    const cx=radius*Math.cos(a),cy=radius*Math.sin(a);
    const ux=Math.cos(tangent),uy=Math.sin(tangent);
    const name=isLog?'Experimental log':palette?.label||'Open lane';
    stations.push({
      index:i,kind:h===0?'open':isLog?'log':'step',
      heightMm:isLog?20:h,label:name,
      color:isLog?CHALLENGE_PALETTE.log.hex:(palette?.hex??0x87a1aa),
      rgba:isLog?CHALLENGE_PALETTE.log.rgba:(palette?.rgba??'0.5 0.6 0.65 1'),
      x:fmt(cx),y:fmt(cy),yawRad:tangent,
      // Each challenge faces the forward tangent. The trial path
      // explicitly goes from entry, over the obstacle, to exit.
      entry:{x:fmt(cx-ux*0.47),y:fmt(cy-uy*0.47),heading:tangent,
             label:`Approach station ${i+1}: ${name}`,station:i,phase:'entry'},
      exit:{x:fmt(cx+ux*0.47),y:fmt(cy+uy*0.47),heading:tangent,
            label:`Cross station ${i+1}: ${name}`,station:i,phase:'cross'},
      // Keep inter-station travel around the CIRCUMFERENCE rather than
      // allowing a single shortest-path chord to cut through the park.
      arc:{
        x:fmt(radius*Math.cos(a+Math.PI/count)),
        y:fmt(radius*Math.sin(a+Math.PI/count)),
        heading:tangent+Math.PI/count,
        label:`Follow ring towards station ${((i+1)%count)+1}`,
        station:i,phase:'ring',
      },
    });
  }
  // The actual AUTONOMOUS GOALS belong halfway between each pair of
  // obstacles, not on an obstacle, just before it, or on its far side.
  // Keep the original anticlockwise station order and tangent headings.
  // A via pose at each obstacle is a ROUTE SHAPING CONTROL POINT only:
  // the navigator still stops/advances solely at these 8 gap waypoints.
  const waypoints=stations.map((station,i)=>{
    const gapAngle=2*Math.PI*(i+.5)/count;
    const next=(i+1)%count;
    return {
      x:fmt(radius*Math.cos(gapAngle)),
      y:fmt(radius*Math.sin(gapAngle)),
      heading:gapAngle+Math.PI/2,
      station:i,phase:'between',
      label:`Between obstacles ${i+1} and ${next+1}`,
      // The previous gap -> this gap crosses obstacle i. Its centre
      // and original CCW heading guide the normal lattice planner;
      // it is NOT an intermediate parking obligation.
      via:{x:station.x,y:station.y,heading:station.yawRad,
           station:i,kind:station.kind},
    };
  });
  return {tierMm,logs:Boolean(logs),radius,count,stations,waypoints};
}
export function challengeCourseXml(course,{groundHeightAt=()=>0}={}){
  if(!course||!Array.isArray(course.stations))
    throw new Error('Missing physical challenge stations');
  if(typeof groundHeightAt!=='function')
    throw new Error('Challenge ground height sampler must be a function');
  const chunks=[];
  for(const s of course.stations){
    if(s.kind==='open')continue;
    const terrainZ=groundHeightAt(s.x,s.y);
    if(!Number.isFinite(terrainZ))
      throw new Error('Challenge ground sampler returned nonfinite height');
    if(s.kind==='step'){
      const halfHeight=s.heightMm/2000;
      chunks.push(`<geom name="viewer_challenge_step_${s.index}" type="box"
        pos="${s.x} ${s.y} ${fmt(terrainZ+halfHeight)}"
        size="0.04 0.20 ${fmt(halfHeight)}"
        euler="0 0 ${fmt(s.yawRad*180/Math.PI)}"
        friction="1.5 0.005 0.0001" rgba="${s.rgba}" />`);
    }else{
      const a=s.yawRad, ux=-Math.sin(a),uy=Math.cos(a); // across the path
      // The flat challenge corridor shares an actual heightfield with
      // the grassy park; position the capsule at terrain elevation.
      const p1=[fmt(s.x-ux*0.20),fmt(s.y-uy*0.20),
                fmt(groundHeightAt(s.x-ux*0.20,s.y-uy*0.20)+.010)];
      const p2=[fmt(s.x+ux*0.20),fmt(s.y+uy*0.20),
                fmt(groundHeightAt(s.x+ux*0.20,s.y+uy*0.20)+.010)];
      chunks.push(`<geom name="viewer_challenge_log_${s.index}"
        type="capsule" fromto="${p1.join(' ')} ${p2.join(' ')}"
        size="0.010" friction="1.5 0.005 0.0001"
        rgba="${s.rgba}" />`);
    }
  }
  return chunks.join('\n');
}
export function visibleChallengeLabel(course){
  if(!course||course.tierMm===0)return 'Open course · no obstacles';
  return `Up to ${course.tierMm} mm steps`+
    (course.logs?' + unproven logs':'')+' · physical collision';
}
