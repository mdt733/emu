import { streamingCarpetXml } from './sensor-lab-carpet.js';

// Infinite Meadow is the sole public simulation world. The historical Field
// heightfield, challenge boxes, trees and destination routes were retired.
// Dedicated CPU sensor geometry remains below only for internal calibration.
const rounded=n=>Number(n.toFixed(6));

// Depth inspection courses use real MuJoCo support geometry, never painted
// obstacles. These courses are DIAGNOSTIC ONLY: the champion is not retrained.
export const DEPTH_COURSES = Object.freeze([
  ['flat','Flat reference'], ['step-up-1cm','1 cm training step up'],
  ['step-up-2cm','2 cm training step up'], ['step-up','5 cm step up'],
  ['step-down','5 cm step down'], ['crest','Hill crest'],
  ['trough','Hill trough'], ['log','Half-round log'],
  ['angled','Angled ridge'],
  ['lab-uphill','Custom uphill'],
  ['lab-downhill','Custom downhill'],
  ['lab-camber-left','Custom left camber'],
  ['lab-camber-right','Custom right camber'],
  ['lab-step-up','Custom step up'],
  ['lab-step-down','Custom step down'],
]);

const LAB_SLOPES=new Set(['lab-uphill','lab-downhill','lab-camber-left','lab-camber-right']);
export function surfaceLabGradient(kind,degrees=3) {
  if(!LAB_SLOPES.has(kind))return null;
  if(!Number.isFinite(degrees)||degrees<0.5||degrees>12)
    throw new Error('Surface lab incline must be 0.5–12 degrees');
  const grade=Math.tan(degrees*Math.PI/180);
  return {
    gx:kind==='lab-uphill'?grade:kind==='lab-downhill'?-grade:0,
    gy:kind==='lab-camber-left'?grade:kind==='lab-camber-right'?-grade:0,
  };
}
function labSlopeXml(kind,degrees) {
  const {gx,gy}=surfaceLabGradient(kind,degrees);
  // One REAL MuJoCo sloped box. Its upper plane passes through (0,0,0)
  // so the robot spawns directly on a smooth continuous incline.
  const t=.25,rad=degrees*Math.PI/180,sign=gx+gy<0?-1:1;
  const pitch=gx!==0;
  const cx=pitch?t*Math.sin(rad)*sign:0;
  const cy=pitch?0:t*Math.sin(rad)*sign;
  const cz=-t*Math.cos(rad);
  const euler=pitch?'0 '+(-sign*degrees)+' 0':(sign*degrees)+' 0 0';
  return '<geom name="surface_lab_plane" type="box" pos="'+
    [cx,cy,cz].map(rounded).join(' ')+'" size="6 2.5 '+t+
    '" euler="'+euler+'" friction="1.5 .005 .0001" rgba=".30 .56 .50 1"/>';
}
function depthCourseXml(kind,{slopeDegrees=3,stepHeightCm=2}={}) {
  if(!DEPTH_COURSES.some(([id])=>id===kind))
    throw new Error('Unknown depth lab obstacle');
  const surface=(name,x0,x1,z0,z1)=>{
    const angle=Math.atan2(z1-z0,x1-x0);
    const thickness=.25, midX=(x0+x1)/2, midZ=(z0+z1)/2;
    const x=midX+thickness*Math.sin(angle);
    const z=midZ-thickness*Math.cos(angle);
    return '<geom name="depth_'+name+'" type="box" pos="'+x+' 0 '+z+
      '" size="'+((x1-x0)/(2*Math.cos(angle)))+' 1.3 '+thickness+
      '" euler="0 '+(-angle*180/Math.PI)+' 0" friction="1.5 0.005 0.0001" rgba="0.35 0.58 0.51 1"/>';
  };
  if(LAB_SLOPES.has(kind))
    return {removeFloor:true,geoms:labSlopeXml(kind,slopeDegrees)};
  if(['lab-step-up','lab-step-down'].includes(kind)){
    const height=Number(stepHeightCm);
    if(!Number.isFinite(height)||height<0.1||height>6)
      throw new Error('Surface lab step must be 0.1–6 cm');
    const z=(kind==='lab-step-up'?1:-1)*height/100;
    return {removeFloor:true,geoms:[
      surface('approach',-4,.65,0,0),
      surface('landing',.65,5,z,z),
    ].join('\n')};
  }
  if(kind==='flat')return {removeFloor:false,geoms:''};
  if(kind==='log')return {
    removeFloor:false,geoms:'<geom name="depth_log" type="cylinder" pos=".83 0 .025" size=".055 .52" euler="90 0 0" friction="1.5 .005 .0001" rgba=".61 .43 .29 1"/>',
  };
  if(kind==='angled')return {
    removeFloor:false,geoms:'<geom name="depth_angled_ridge" type="box" pos=".85 0 .03" size=".052 .52 .03" euler="0 0 22" friction="1.5 .005 .0001" rgba=".67 .50 .29 1"/>',
  };
  if(['step-up','step-down','step-up-1cm','step-up-2cm'].includes(kind)){
    const z=kind==='step-up-1cm'?.01:kind==='step-up-2cm'?.02:
      kind==='step-up'?.05:-.05;
    const edge=kind==='step-up-1cm'||kind==='step-up-2cm'?.25:.65;
    return {removeFloor:true,geoms: [
      surface('approach',-4,edge,0,0),
      surface('landing',edge,5,z,z),
    ].join('\n')};
  }
  const z=kind==='crest'?.065:-.065;
  return {removeFloor:true,geoms: [
    surface('approach',-4,.65,0,0),
    surface('rise',.65,1.20,0,z),
    surface('top',1.20,1.55,z,z),
    surface('exit',1.55,2.1,z,0),
    surface('after',2.1,5,0,0),
  ].join('\n')};
}

export function withTerrain(robotXml,mode='carpet',options={}){
  if(typeof robotXml!=='string'||!robotXml.includes('</worldbody>'))
    throw Error('Cannot locate worldbody in policy robot XML');
  if(/sensor_lab_hfield_|sensor_lab_ground_|viewer_park_hfield|surface_lab_plane|depth_(?:approach|log|rise|landing|angled_ridge)/.test(robotXml))
    throw Error('Viewer terrain already inserted');
  // Internal baseline fixture. Not a public Field option.
  if(mode==='flat')return robotXml;
  if(mode==='sensor'){
    const course=depthCourseXml(options.obstacle||'flat',options);
    let xml=robotXml;
    if(course.removeFloor){
      xml=xml.replace(/<geom\s+name="floor"[^>]*\/>/,'');
      if(xml===robotXml)throw Error('Sensor lab requires named floor');
    }
    return xml.replace('</worldbody>',course.geoms+'\n</worldbody>');
  }
  if(mode!=='carpet')throw Error('Only Infinite Meadow is supported');
  const park=options.park;
  if(park?.physics!=='streaming-heightfield')
    throw Error('Infinite Meadow needs its real streaming heightfield');
  const geom=streamingCarpetXml(park);
  let xml=robotXml.replace(/<geom\s+name="floor"[^>]*\/>/,'');
  if(xml===robotXml)throw Error('Streaming Meadow requires the source floor');
  if(xml.includes('</asset>'))
    xml=xml.replace('</asset>',geom.assets+'\n</asset>');
  else xml=xml.replace('<worldbody>','<asset>'+geom.assets+'</asset>\n<worldbody>');
  return xml.replace('</worldbody>',geom.bodies+'\n</worldbody>');
}

// Browser JS raycaster cannot intersect MuJoCo heightfield geoms directly.
// Sensors must read the same exact active physical terrain as MuJoCo;
// unloaded world locations return NaN, never invisible ground.
export function terrainSensorHeightFallback(mode,park){
  return mode==='carpet'&&park?.physics==='streaming-heightfield'
    ?(x,y)=>park.sampleHeight(x,y):null;
}

// Ground angle follows whichever of the tile's TWO physical triangles
// supports the feet. A step lip is not a continuous grade.
export function surfaceSlope(mode,park,x,y,robotHeading=0,lab=null){
  if(![x,y,robotHeading].every(Number.isFinite))
    return {supported:false,degrees:0,arrowDegrees:0};
  let gx=0,gy=0;
  if(mode==='carpet'){
    if(!park||!Number.isFinite(park.sampleHeight(x,y)))
      return {supported:false,degrees:0,arrowDegrees:0};
    const size=park.tileSizeM||1,i=Math.floor(x/size),j=Math.floor(y/size);
    const [a,b,c,d]=park.at(i,j).corners;
    const u=x/size-i,v=y/size-j;
    gx=(u+v<=1?b-a:d-c)/size;
    gy=(u+v<=1?c-a:d-b)/size;
  }else if(mode==='sensor'){
    const gradient=lab&&surfaceLabGradient(lab.obstacle,lab.slopeDegrees);
    if(!gradient)
      return {supported:false,degrees:0,arrowDegrees:0,flat:false};
    gx=gradient.gx;gy=gradient.gy;
  }else if(mode!=='flat'){
    throw Error('Only Infinite Meadow is supported');
  }
  const degrees=Math.atan(Math.hypot(gx,gy))*180/Math.PI;
  const relative=Math.atan2(Math.sin(Math.atan2(gy,gx)-robotHeading),
                            Math.cos(Math.atan2(gy,gx)-robotHeading));
  return {supported:true,degrees,arrowDegrees:-relative*180/Math.PI,
    flat:degrees<.05};
}
