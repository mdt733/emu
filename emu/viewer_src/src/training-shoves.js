// Browser spectator force sampling. The 50Hz trainer starts pushes with
// p=.0015*T.resilience per frame, and also samples disturbed RESET POSES.
// "training" preserves that rare steady-state event probability. "drill"
// intentionally DOES NOT represent the GPU curriculum: it stresses the
// recovery response with up to 2 events/s on a single viewer robot.
export const TRAINING_PUSH_PROB_PER_FRAME=0.0015;
export const CONTROL_HZ=50;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,Number(x)||0));
export const SHOVE_MODES=Object.freeze(['training','inspection','drill']);

export function shoveChance(intensity=0,multiplier=1){
  return clamp(TRAINING_PUSH_PROB_PER_FRAME*
    clamp(intensity,0,1)*clamp(multiplier,1,20),0,1);
}
export function shoveChanceForMode(intensity=0,mode='training'){
  const d=clamp(intensity,0,1);
  if(mode==='training')return shoveChance(d,1);
  if(mode==='inspection')return shoveChance(d,8);
  if(mode==='drill'){
    // A visible, frequent diagnostic recovery workout: above zero, about
    // 1..2 attempts/s as the resilience intensity grows. No impulses at 0.
    return d===0?0:clamp((1+d)/CONTROL_HZ,0,.08);
  }
  throw Error('Unknown shove preview mode: '+String(mode));
}
export function expectedShoveRate(intensity=0,mode='training'){
  return CONTROL_HZ*shoveChanceForMode(intensity,mode);
}
export function sampleShove({intensity=0,multiplier=1,mode=null,maxForceN=0,
                             active=false,manual=false,rng=Math.random}={}){
  if(active||!Number.isFinite(maxForceN)||maxForceN<=0)return null;
  const chance=mode===null?shoveChance(intensity,multiplier)
    :shoveChanceForMode(intensity,mode);
  if(!manual && !(chance>0&&rng()<chance))return null;
  // Trainer draws one random direction and a magnitude in [0, cap];
  // force is applied through genuine MuJoCo torso xfrc_applied.
  const angle=rng()*Math.PI*2;
  const magnitude=rng()*maxForceN;
  const frames=2+Math.floor(rng()*7);
  return {fx:magnitude*Math.cos(angle),fy:magnitude*Math.sin(angle),
    magnitude,angle,frames};
}
