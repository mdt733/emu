// Optional browser-only physics. This does NOT modify exported policies or MJX.
// One 1-degree up/flat/down course. Its ramps are real MuJoCo contact boxes,
// not painted scenery; long slopes are gentler than a step or a sharp curb.
// Beginning of the course is at x=0.85m, forward of the spawn position.
export const GENTLE_SLOPE_DEGREES = 1;
export const COURSE_START_X = 0.85;

const course = `
    <!-- Browser-only physical course: 1 degree up, level, 1 degree down. -->
    <geom name="viewer_ramp_up" type="box" pos="1.25 0 0.0005"
          size="0.4 0.44 0.008" euler="0 -1 0"
          friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
    <geom name="viewer_ramp_top" type="box" pos="1.91 0 0.0075"
          size="0.26 0.44 0.008"
          friction="1.5 0.005 0.0001" rgba="0.28 0.56 0.51 1" />
    <geom name="viewer_ramp_down" type="box" pos="2.57 0 0.0005"
          size="0.4 0.44 0.008" euler="0 1 0"
          friction="1.5 0.005 0.0001" rgba="0.23 0.50 0.47 1" />
`;

export function withGentleCourse(robotXml) {
  if (typeof robotXml !== 'string' || !robotXml.includes('</worldbody>')) {
    throw new Error('Cannot locate worldbody in policy robot XML');
  }
  if (robotXml.includes('viewer_ramp_up')) throw new Error('Viewer terrain already inserted');
  return robotXml.replace('</worldbody>', `${course}  </worldbody>`);
}
