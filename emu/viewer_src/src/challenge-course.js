// EMU v5018 viewer-only physical obstacle course.
// A real, non-flat MuJoCo park heightfield provides the ground. Obstacle
// heights are relative to the local sampled surface, not world Z=0.
// These are demonstrations, NEVER evidence of a trained terrain policy.
export const CHALLENGE_RADIUS_M = 4;
export const CHALLENGE_COUNT = 8;
export const CHALLENGE_TIERS_MM = Object.freeze([0,5,10,15,20]);
export const CHALLENGE_WIDTH_M = .80; // doubled from the old 0.40m full width
export const CHALLENGE_YAW_OFFSETS_DEG = Object.freeze([0,15,-15,25,-25,10,-10,20]);
export const CHALLENGE_PALETTE = Object.freeze({
  5:{label:'5 mm step',hex:0x4cc78b,rgba:'0.20 0.76 0.46 1'},
  10:{label:'10 mm step',hex:0x64a8f7,rgba:'0.30 0.59 0.98 1'},
  15:{label:'15 mm step',hex:0xe6b45b,rgba:'0.94 0.66 0.24 1'},
  20:{label:'20 mm step',hex:0xd986d1,rgba:'0.84 0.37 0.71 1'},
  log:{label:'Half-buried log · experimental',hex:0xf28e56,
       rgba:'0.94 0.41 0.20 1'},
});
const fmt=n=>Number(n.toFixed(6));
const rad=d=>d*Math.PI/180;
function checkFinite(v,name){
  if(!Number.isFinite(v))throw new Error(name+' must be finite');
  return v;
}
function worldPoint(station,forward=0,lateral=0){
  const a=station.yawRad,c=Math.cos(a),s=Math.sin(a);
  return {x:station.x+c*forward-s*lateral,
          y:station.y+s*forward+c*lateral};
}
export function generateChallengeCourse({tierMm=10,logs=false,
                                         radius=CHALLENGE_RADIUS_M,
                                         count=CHALLENGE_COUNT}={}){
  if(!CHALLENGE_TIERS_MM.includes(tierMm))
    throw new Error('Challenge difficulty must be 0, 5, 10, 15 or 20 mm');
  if(!Number.isFinite(radius)||radius<2.7||radius>4.2)
    throw new Error('Challenge ring radius must be 2.7–4.2m');
  if(!Number.isInteger(count)||count<4||count>12)
    throw new Error('Challenge station count must be 4–12');
  const available=CHALLENGE_TIERS_MM.filter(h=>h>0&&h<=tierMm);
  const stepKinds=['step','step_up','step_down','log'];
  const stations=Array.from({length:count},(_,i)=>{
    const a=2*Math.PI*i/count;
    const heading=a+Math.PI/2; // fixed CCW NAVIGATION regardless of obstacle yaw
    const yawOffsetDeg=CHALLENGE_YAW_OFFSETS_DEG[i%CHALLENGE_YAW_OFFSETS_DEG.length];
    // Lead with the selected frontier height, then interleave smaller
    // previously tested heights instead of making EVERY station maximal.
    const heightMm=available.length
      ?available[available.length-1-(i%available.length)]:0;
    const family=stepKinds[i%stepKinds.length];
    const kind=heightMm===0?'open':
      family==='log'&&!logs?'step':family;
    const paint=kind==='log'?CHALLENGE_PALETTE.log:CHALLENGE_PALETTE[heightMm];
    const familyName={step:'Step',step_up:'Step up · gentle ramp down',
      step_down:'Gentle ramp up · step down',log:'Half-buried log · experimental',
      open:'Open lane'}[kind];
    return {
      index:i,kind,heightMm,
      label:kind==='open'?familyName:`${familyName} · ${heightMm} mm`,
      color:paint?.hex??0x87a1aa,
      rgba:paint?.rgba??'0.5 0.6 0.65 1',
      x:fmt(radius*Math.cos(a)),
      y:fmt(radius*Math.sin(a)),
      headingRad:heading, // fixed tangential CCW walker heading
      yawOffsetDeg,yawRad:heading+rad(yawOffsetDeg),
      widthM:CHALLENGE_WIDTH_M,
    };
  });
  // IMPORTANT: ONE navigator goal per GAP, exactly halfway between the
  // centres of the neighbouring obstacles. Ordered CCW indefinitely.
  // The obstacle centre is a path-planner VIA, NOT a stop/arrival goal.
  const waypoints=stations.map((station,i)=>{
    const gapAngle=2*Math.PI*(i+.5)/count;
    return {
      x:fmt(radius*Math.cos(gapAngle)),
      y:fmt(radius*Math.sin(gapAngle)),
      heading:gapAngle+Math.PI/2,
      station:i,phase:'between',
      label:`Between obstacles ${i+1} and ${(i+1)%count+1}`,
      via:{x:station.x,y:station.y,
           heading:station.headingRad,station:i,kind:station.kind},
    };
  });
  return {tierMm,logs:Boolean(logs),radius,count,stations,waypoints,
          direction:'counterclockwise'};
}

// Pure testable probe coordinates for the actual WASM collider. The
// centre of each composite module and its gently sloped ramp are physical
// target surfaces; the sphere will be dropped onto these points.
export function challengeProbe(station,groundHeightAt=()=>0,part='top'){
  if(typeof groundHeightAt!=='function')throw new Error('Missing terrain sampler');
  let forward=0,topOffset=station.heightMm/1000;
  if(station.kind==='step_up'){
    if(part==='ramp') {forward=.30;topOffset=station.heightMm/1000*(.68-.30)/(.68+.06);}
    else forward=-.20;
  }else if(station.kind==='step_down'){
    if(part==='ramp') {forward=-.30;topOffset=station.heightMm/1000*(-.30+.68)/(.68+.06);}
    else forward=.20;
  }else if(station.kind==='log'){
    topOffset=station.heightMm/1000; // cylinder centre is buried at ground Z
  }else if(station.kind==='open')topOffset=0;
  const p=worldPoint(station,forward);
  return {x:fmt(p.x),y:fmt(p.y),
          surfaceZ:groundHeightAt(p.x,p.y)+topOffset};
}
function boxXml(name,at,halfX,halfY,halfZ,topZ,yawDeg,color){
  const z=topZ-halfZ;
  return `<geom name="${name}" type="box"
    pos="${fmt(at.x)} ${fmt(at.y)} ${fmt(z)}"
    size="${fmt(halfX)} ${fmt(halfY)} ${fmt(halfZ)}"
    euler="0 0 ${fmt(yawDeg)}"
    friction="1.5 0.005 0.0001" rgba="${color}" />`;
}
// A thin, REAL tilted MuJoCo box connects two ground-relative top heights.
// q = yaw around world Z * pitch about local Y; local +X descends for
// positive pitch. The tilted surface runs in the SAME station yaw frame.
function rampXml(station,groundHeightAt,name,startX,endX,startAbove,endAbove){
  const from=worldPoint(station,startX),to=worldPoint(station,endX);
  const top0=checkFinite(groundHeightAt(from.x,from.y),'ramp start surface')+startAbove;
  const top1=checkFinite(groundHeightAt(to.x,to.y),'ramp end surface')+endAbove;
  const run=endX-startX,dz=top1-top0;
  const pitch=-Math.atan2(dz,run),yaw=station.yawRad;
  const a=Math.cos(yaw/2),b=Math.sin(yaw/2);
  const c=Math.cos(pitch/2),d=Math.sin(pitch/2);
  const quat=[a*c,-b*d,a*d,b*c].map(fmt);
  const midpoint=worldPoint(station,(startX+endX)/2);
  const halfThickness=.005;
  return `<geom name="${name}" type="box"
    pos="${fmt(midpoint.x)} ${fmt(midpoint.y)} ${fmt((top0+top1)/2-halfThickness)}"
    size="${fmt(Math.hypot(run,dz)/2)} ${fmt(station.widthM/2)} ${fmt(halfThickness)}"
    quat="${quat.join(' ')}"
    friction="1.5 0.005 0.0001" rgba="${station.rgba}" />`;
}
export function challengeCourseXml(course,{groundHeightAt=()=>0}={}){
  if(!course||!Array.isArray(course.stations))
    throw new Error('Missing physical challenge stations');
  if(typeof groundHeightAt!=='function')
    throw new Error('Challenge ground height sampler must be a function');
  const chunks=[];
  for(const station of course.stations){
    if(station.kind==='open')continue;
    const ground=checkFinite(groundHeightAt(station.x,station.y),'challenge ground height');
    const h=station.heightMm/1000;
    const name=`viewer_challenge_${station.kind}_${station.index}`;
    const yawDeg=station.yawRad*180/Math.PI;
    if(station.kind==='step'){
      chunks.push(boxXml(name,station,.04,station.widthM/2,h/2,
                         ground+h,yawDeg,station.rgba));
    }else if(station.kind==='step_up'||station.kind==='step_down'){
      // A "step down" is the reverse traversal of an up-step module:
      // approach by a gentle ramp UP, then take a sharp step DOWN.
      // No fake hole is cut into MuJoCo's solid heightfield.
      const rise=station.kind==='step_up';
      const plateauX=rise?-.20:.20;
      const top=worldPoint(station,plateauX);
      const plateauZ=checkFinite(
        groundHeightAt(top.x,top.y),'step plateau ground height')+h;
      chunks.push(boxXml(name+'_plateau',top,.14,station.widthM/2,h/2,
                         plateauZ,yawDeg,station.rgba));
      chunks.push(rampXml(station,groundHeightAt,name+'_return_ramp',
        rise?-.06:-.68,rise?.68:.06,rise?h:0,rise?0:h));
    }else if(station.kind==='log'){
      const lateral=station.widthM/2;
      const left=worldPoint(station,0,-lateral);
      const right=worldPoint(station,0,lateral);
      // Centreline sits AT the sampled terrain elevation (not +radius):
      // exactly the lower half of the 2r cylinder is buried in earth.
      const ends=[left,right].map(p=>[
        fmt(p.x),fmt(p.y),
        fmt(checkFinite(groundHeightAt(p.x,p.y),'log-end ground height')),
      ]);
      chunks.push(`<geom name="${name}" type="capsule"
        fromto="${ends[0].join(' ')} ${ends[1].join(' ')}"
        size="${fmt(h)}" friction="1.5 0.005 0.0001"
        rgba="${station.rgba}" />`);
    }
  }
  return chunks.join('\n');
}
export function visibleChallengeLabel(course){
  if(!course||course.tierMm===0)return 'Open course · no obstacles';
  return `Mixed steps, step-ups and step-downs up to ${course.tierMm} mm`+
    (course.logs?' + half-buried experimental logs':'')+
    ' · angled real collisions';
}
