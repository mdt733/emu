// Stochastic spectator shove sampler. Same baseline event probability
// (per 50Hz control frame) and force/duration envelopes as MJX; the
// OPTIONAL inspection multiplier is explicitly NOT training-equivalent.
export const TRAINING_PUSH_PROB_PER_FRAME=0.0015;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function shoveChance(intensity=0,multiplier=1){
  return clamp(TRAINING_PUSH_PROB_PER_FRAME*
    clamp(Number(intensity)||0,0,1)*clamp(Number(multiplier)||1,1,20),0,1);
}
export function sampleShove({intensity=0,multiplier=1,maxForceN=0,
                             active=false,manual=false,rng=Math.random}={}){
  if(active||!Number.isFinite(maxForceN)||maxForceN<=0)return null;
  const chance=shoveChance(intensity,multiplier);
  if(!manual && !(chance>0&&rng()<chance))return null;
  // Trainer draws one angle and a magnitude uniformly inside [0, cap];
  // each shove lasts an integer 2..8 control frames.
  const angle=rng()*Math.PI*2;
  const magnitude=rng()*maxForceN;
  const frames=2+Math.floor(rng()*7);
  return {fx:magnitude*Math.cos(angle),fy:magnitude*Math.sin(angle),
    magnitude,angle,frames};
}
