import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {CHALLENGE_RADIUS_M,CHALLENGE_PALETTE,
  generateChallengeCourse,challengeCourseXml,visibleChallengeLabel
} from '../src/challenge-course.js';
import {withTerrain,surfaceSlope} from '../src/terrain.js';
import {ARENA_RADIUS_M,GoalNavigator} from '../src/goal-navigation.js';

const xml='<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.05" /></worldbody></mujoco>';
const near=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<=tol, `${a} != ${b}`);

test('obstacles sit ON the original 4m circle, entry and exit straddle the actual collisions',()=>{
  assert.equal(CHALLENGE_RADIUS_M,ARENA_RADIUS_M);
  const c=generateChallengeCourse({tierMm:20});
  assert.equal(c.stations.length,8);
  assert.equal(c.waypoints.length,16);
  for(const station of c.stations){
    near(Math.hypot(station.x,station.y),4);
    const dx=station.exit.x-station.entry.x;
    const dy=station.exit.y-station.entry.y;
    near(Math.hypot(dx,dy),.94,2e-6);
    // World step centre is the midpoint of its approach and crossing legs.
    near((station.entry.x+station.exit.x)/2,station.x,2e-6);
    near((station.entry.y+station.exit.y)/2,station.y,2e-6);
    near(Math.atan2(Math.sin(Math.atan2(dy,dx)-station.yawRad),
                    Math.cos(Math.atan2(dy,dx)-station.yawRad)),0,1e-6);
    assert.equal(station.entry.phase,'entry');
    assert.equal(station.exit.phase,'cross');
  }
});
test('viewer difficulty is an explicit selection, not a fictitious trained/unlocked skill',()=>{
  const flat=generateChallengeCourse({tierMm:0});
  assert.equal(challengeCourseXml(flat),'');
  assert.match(visibleChallengeLabel(flat),/no obstacles/);
  for(const tierMm of [5,10,15,20]){
    const c=generateChallengeCourse({tierMm});
    assert.ok(c.stations.every(s=>s.heightMm<=tierMm));
    assert.ok(c.stations.every(s=>s.kind==='step'));
    assert.ok(c.stations.some(s=>s.heightMm===tierMm));
    for(const st of c.stations)
      assert.ok(st.color===CHALLENGE_PALETTE[st.heightMm].hex);
  }
  assert.throws(()=>generateChallengeCourse({tierMm:30}),/0, 5, 10, 15 or 20/);
});
test('step boxes and experimental capsule logs are physical worldbody MuJoCo geoms',()=>{
  const course=generateChallengeCourse({tierMm:20,logs:true});
  const out=withTerrain(xml,'course',{course});
  assert.equal((out.match(/name="viewer_challenge_step_/g)||[]).length,6);
  assert.equal((out.match(/name="viewer_challenge_log_/g)||[]).length,2);
  assert.match(out,/type="box"/);
  assert.match(out,/type="capsule"/);
  assert.match(out,/fromto="[^"]+"\s+size="0\.010"/);
  assert.match(visibleChallengeLabel(course),/unproven logs/);
  assert.match(out,/friction="1\.5/);
  // Original physical floor is still present, unlike the park heightfield.
  assert.match(out,/name="floor"/);
  assert.throws(()=>withTerrain(out,'course',{course}),/already inserted/);
  // Old modes still work unchanged.
  assert.equal(withTerrain(xml,'flat'),xml);
  assert.match(withTerrain(xml,'ramp'),/viewer_ramp_up/);
  assert.equal(surfaceSlope('course',null,0,0).flat,true);
});
test('circuit demo requests fixed targets from worker, without random substitutions',async()=>{
  const navigator=new GoalNavigator();
  const course=generateChallengeCourse({tierMm:10});
  const posted=[];
  navigator.useBackgroundPlanner(async req=>{
    posted.push(req);
    return {goal:req.goal,route:null};
  });
  navigator.active=true;
  navigator.setCircuitWaypoints(course.waypoints);
  await navigator.nextGoal({x:0,y:0,heading:0},
    {vx_min:-.2,vx_max:.6,vy_abs:.2,yaw_abs:1.5},[]);
  await Promise.resolve();
  assert.equal(posted.length,1);
  assert.equal(posted[0].kind,'target');
  assert.equal(posted[0].goal.label,course.waypoints[0].label);
  assert.equal(navigator.circuitNextIndex,0); // failed plans do NOT silently skip obstacles
  assert.equal(navigator.status({x:0,y:0,heading:0}).circuit,true);
  navigator.setCircuitWaypoints(null);
  assert.equal(navigator.status({x:0,y:0,heading:0}).circuit,false);
});
test('public and live viewer share discoverable challenge controls',()=>{
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  for(const id of ['challengeTier','challengeLogs','demoMode','challengeControls',
                   'challengeDescription','demoRouteNote']){
    assert.match(html,new RegExp(`id="${id}"`));
    assert.ok(main.includes("$('"+id+"')"), 'missing viewer binding for '+id);
  }
  assert.match(main,/sim\.navigationChallengeCourse\(\)/);
  assert.match(main,/setChallengeOptions/);
});
