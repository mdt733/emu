import {streamingCarpetXml} from './sensor-lab-carpet.js';

// One browser physics topology: 225 mutable, colliding CPU heightfields.
// Both the Infinite Meadow and the specialist exercises use these SAME
// colliders; the untrained sensor calibration lab does too. Historical
// diagnostic logs, ramps, Field/Park boxes and alternate surface-model
// modes remain available in Git, not this live viewer.
export function withTerrain(robotXml,mode='carpet',options={}){
  if(typeof robotXml!=='string'||!robotXml.includes('</worldbody>'))
    throw Error('Cannot locate worldbody in policy robot XML');
  if(mode!=='carpet')
    throw Error('Only physical streaming Meadow/specialist terrain is supported');
  if(/sensor_lab_hfield_|sensor_lab_ground_|viewer_park_hfield/.test(robotXml))
    throw Error('Viewer terrain already inserted');
  const park=options.park;
  if(park?.physics!=='streaming-heightfield')
    throw Error('Viewer requires its real streaming heightfield');
  const geom=streamingCarpetXml(park);
  let xml=robotXml.replace(/<geom\s+name="floor"[^>]*\/>/,'');
  if(xml===robotXml)throw Error('Streaming physical terrain requires the source floor');
  if(xml.includes('</asset>'))
    xml=xml.replace('</asset>',geom.assets+'\n</asset>');
  else xml=xml.replace('<worldbody>','<asset>'+geom.assets+'</asset>\n<worldbody>');
  return xml.replace('</worldbody>',geom.bodies+'\n</worldbody>');
}

// Raycasters cannot intersect MuJoCo hfields. Sensors MUST sample the
// exact same active physical tiles; unloaded regions remain void (NaN).
export function terrainSensorHeightFallback(mode,park){
  return mode==='carpet'&&park?.physics==='streaming-heightfield'
    ?(x,y)=>park.sampleHeight(x,y):null;
}

// Foot-level terrain grade is derived from the SAME two planar physical
// triangles, not an alternate teaching slope or visual mesh.
export function surfaceSlope(mode,park,x,y,robotHeading=0){
  if(![x,y,robotHeading].every(Number.isFinite))
    return {supported:false,degrees:0,arrowDegrees:0};
  if(mode!=='carpet'||!park||!Number.isFinite(park.sampleHeight(x,y)))
    return {supported:false,degrees:0,arrowDegrees:0};
  const size=park.tileSizeM||1,i=Math.floor(x/size),j=Math.floor(y/size);
  const tile=park.at(i,j);
  if(!tile)return {supported:false,degrees:0,arrowDegrees:0};
  const [a,b,c,d]=tile.corners;
  const u=x/size-i,v=y/size-j;
  const gx=(u+v<=1?b-a:d-c)/size;
  const gy=(u+v<=1?c-a:d-b)/size;
  const degrees=Math.atan(Math.hypot(gx,gy))*180/Math.PI;
  const relative=Math.atan2(Math.sin(Math.atan2(gy,gx)-robotHeading),
                            Math.cos(Math.atan2(gy,gx)-robotHeading));
  return {supported:true,degrees,arrowDegrees:-relative*180/Math.PI,
    flat:degrees<.05};
}
