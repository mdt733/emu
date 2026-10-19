// Small, testable boundary between autonomous commands and human inputs.
// Viewer UI and the MuJoCo actor both use the same split-stick command axes.
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const scale=(a,b)=>b>0?clamp(a/b,-1,1):0;
export const MANUAL_IDLE_SECONDS=30;
export function commandStickPositions({vx=0,vy=0,yaw=0,pitch=0},limits){
  const forward=vx>=0?limits.vx_max:-limits.vx_min;
  return {left:{x:scale(-vy,limits.vy_abs),y:scale(vx,forward)},
    right:{x:scale(-yaw,limits.yaw_abs),
      y:scale(-pitch,limits.pitch_abs_rad)}};
}
export class DemoHandoff {
  constructor(idle=MANUAL_IDLE_SECONDS){
    this.idle=idle;
    this.lastHumanInput=0;
    this.autonomous=true;
  }
  takeControl(now){
    this.autonomous=false;
    this.lastHumanInput=now;
  }
  resume(){
    this.autonomous=true;
  }
  shouldResume(now,inputHeld=false){
    return !this.autonomous&&!inputHeld&&now-this.lastHumanInput>=this.idle;
  }
}
