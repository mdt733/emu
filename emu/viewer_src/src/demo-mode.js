// One unambiguous source of demo routing semantics. Terrain selection is
// a user-facing action: choosing Challenge ring should SHOW a circuit,
// not silently keep the old random-goal demonstration. Explicitly changing
// the demo dropdown afterward remains under the user's control.
export function defaultDemoModeForTerrain(terrainMode) {
  if(!['flat','ramp','park','course'].includes(terrainMode))
    throw new Error('Unknown demo terrain '+terrainMode);
  return terrainMode==='course'?'circuit':'random';
}

export function waypointsForDemoMode(mode,course) {
  if(mode==='random')return null;
  if(mode!=='circuit')throw new Error('Unknown autonomous demo mode '+mode);
  if(!course || !Array.isArray(course.waypoints) ||
     course.waypoints.length<2 ||
     course.waypoints.some(w=>![w.x,w.y,w.heading].every(Number.isFinite)))
    throw new Error('Challenge circuit requested without a valid physical course');
  return course.waypoints;
}
