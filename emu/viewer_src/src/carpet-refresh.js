// Compare PHYSICAL role-envelope difficulty, not large rounded global
// frontiers. At 110M a 4mm global step * 15% exposure is only 0.6mm;
// the retired 2mm quantisation silently skipped the whole curriculum.
export function carpetGeometrySignature(lesson,stepProgram='ordinary'){
  if(!lesson||!lesson.terrain)throw Error('Missing carpet lesson');
  const p=lesson.terrain;
  const finite=(value,label)=>{
    const n=Number(value);
    if(!Number.isFinite(n))throw Error('Invalid carpet '+label);
    return n;
  };
  return JSON.stringify([
    lesson.stage,lesson.role,String(stepProgram),finite(p.seed,'seed'),
    // 0.01 degree for slope and transition, 0.025mm for steps.
    Math.round(finite(p.slopeDeg,'slope')*100),
    Math.round(finite(p.transitionDeg,'transition')*100),
    Math.round(finite(p.stepUpM,'rise')*40000),
    Math.round(finite(p.stepDownM,'drop')*40000)
  ]);
}
