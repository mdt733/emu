// One predictable split-stick mapping for manual and autonomous input:
// left = forward / sideways, right = yaw / chassis pitch.
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
export function joystickToCommands(leftX,leftY,rightX,rightY,limits){
  const vx=leftY>=0?leftY*limits.vx_max:(-leftY)*limits.vx_min;
  const vy=-leftX*limits.vy_abs;
  const yaw=-clamp(rightX,-1,1)*limits.yaw_abs;
  const pitch=-rightY*limits.pitch_abs_rad;
  return {vx,vy,yaw,pitch};
}

export const wrapRadians = radians => Math.atan2(Math.sin(radians), Math.cos(radians));

// A heading setpoint is an angle in world coordinates; the robot's neural
// actor expects a YAW RATE command in radians/s. A pure outer-loop P controller
// translates heading error to that rate, then clamps to the actor's taught
// yaw envelope. No direct-stick feedforward and no integral wind-up.
export const MAX_HEADING_ERROR_RAD = 1.0; // cap target lead at ~57° during a fall

export function headingHoldRate(requestedRate, headingTarget, actualHeading, kp, maxRate) {
  if (!Number.isFinite(actualHeading) || !Number.isFinite(headingTarget))
    return clamp(requestedRate, -maxRate, maxRate);
  const error = clamp(wrapRadians(headingTarget - actualHeading),
                      -MAX_HEADING_ERROR_RAD, MAX_HEADING_ERROR_RAD);
  return clamp(kp * error, -maxRate, maxRate);
}

/**
 * Integrate the operator's requested rotation into a world heading SETPOINT,
 * never the P-corrected yaw output. Releasing the stick holds the target.
 * With P disabled, the original direct yaw-rate steering is unchanged.
 */
export function nextYawCommand(requestedYaw, targetYaw, headingTarget,
                               actualHeading, dt, headingHold, kp, maxRate) {
  const maximumStep=Math.max(0,4*Math.min(Math.max(dt,0),.1));
  const difference=targetYaw-requestedYaw;
  // A heading *setpoint* must stop moving immediately when the stick is
  // released. Only direct-rate steering needs the old rate slew limiter.
  const request=headingHold
    ? clamp(targetYaw,-maxRate,maxRate)
    : clamp(requestedYaw+clamp(difference,-maximumStep,maximumStep),
            -maxRate,maxRate);
  if(!headingHold || !Number.isFinite(actualHeading)) {
    return {requestedYaw:request,appliedYaw:request,
            headingTarget:Number.isFinite(actualHeading)?wrapRadians(actualHeading):headingTarget};
  }
  // Keep the desired compass heading near the actual robot when it cannot
  // turn quickly enough; otherwise commands bank up an impossible full turn.
  const desired=wrapRadians(headingTarget+request*Math.max(0,dt));
  const boundedError=clamp(wrapRadians(desired-actualHeading),
                           -MAX_HEADING_ERROR_RAD,MAX_HEADING_ERROR_RAD);
  const target=wrapRadians(actualHeading+boundedError);
  return {requestedYaw:request,headingTarget:target,
          appliedYaw:headingHoldRate(request,target,actualHeading,kp,maxRate)};
}


// Shared mapping between autonomous commands and manual sticks.
// Small, testable boundary between autonomous commands and human inputs.
// Viewer UI and the MuJoCo actor both use the same split-stick command axes.
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
