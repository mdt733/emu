import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {CHALLENGE_RADIUS_M,CHALLENGE_PALETTE,
  generateChallengeCourse,challengeCourseXml,visibleChallengeLabel
} from '../src/challenge-course.js';
import {withTerrain,surfaceSlope,generateTerrainPark} from '../src/terrain.js';
import {defaultDemoModeForTerrain,waypointsForDemoMode} from '../src/demo-mode.js';
import {ARENA_RADIUS_M,GoalNavigator,planPoseRoute} from '../src/goal-navigation.js';

const xml='<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.05" /></worldbody></mujoco>';
const near=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<=tol, `${a} != ${b}`);

test('obstacles sit ON the original 4m circle, entry and exit straddle the actual collisions',()=>{
  assert.equal(CHALLENGE_RADIUS_M,ARENA_RADIUS_M);
  const c=generateChallengeCourse({tierMm:20});
  assert.equal(c.stations.length,8);
  assert.equal(c.waypoints.length,24);
  assert.deepEqual(c.waypoints.slice(0,3).map(w=>w.phase),
    ['entry','cross','ring']);
  for(const station of c.stations){
    near(Math.hypot(station.x,station.y),4);
    const dx=station.exit.x-station.entry.x;
    const dy=station.exit.y-station.entry.y;
    near(Math.hypot(dx,dy),.94,2e-6);
    // World step centre is the midpoint of its approach and crossing legs.
    near((station.entry.x+station.exit.x)/2,station.x,2e-6);
    near((station.entry.y+station.exit.y)/2,station.y,2e-6);
    near(Math.atan2(Math.sin(Math.atan2(dy,dx)-station.yawRad),
                    Math.cos(Math.atan2(dy,dx)-station.yawRad)),0,3e-6);
    assert.equal(station.entry.phase,'entry');
    assert.equal(station.exit.phase,'cross');
    assert.equal(station.arc.phase,'ring');
    near(Math.hypot(station.arc.x,station.arc.y),4,1e-6);
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
  // CHALLENGE MUST KEEP all physical park terrain, scenery and trees.
  // The old viewer mode removed these and left a bare flat plane.
  assert.match(out,/<geom name="viewer_park_ground" type="hfield"/);
  assert.match(out,/viewer_park_hfield/);
  assert.match(out,/viewer_tree_trunk_0/);
  assert.doesNotMatch(out,/name="floor"/);
  assert.throws(()=>withTerrain(out,'course',{course}),/already inserted/);
  // Old modes still work unchanged.
  assert.equal(withTerrain(xml,'flat'),xml);
  assert.match(withTerrain(xml,'ramp'),/viewer_ramp_up/);
  assert.equal(surfaceSlope('course',null,0,0).supported,false);
  const park=generateTerrainPark({maxDegrees:10,challengeRing:true});
  assert.equal(surfaceSlope('course',park,0,0).degrees,0);
  for(const obstacle of course.stations){
    near(park.sampleHeight(obstacle.x,obstacle.y),0,1e-6);
    assert.ok(park.trees.every(tree=>
      Math.abs(Math.hypot(tree.x,tree.y)-4)>=1.08),
      'tree trunk is planted in the challenge corridor');
  }
  assert.ok(park.trees.length>=5,'challenge ring must retain a forest');
  assert.ok(park.heights.some(h=>h>.025),
    'hills beyond the flat ring must remain actual physics');
  const parkXml=withTerrain(xml,'park',{park:generateTerrainPark({maxDegrees:10})});
  assert.match(parkXml,/viewer_park_hfield/);
});
test('real pose lattice routes actually cut across each collidable challenge footprint',()=>{
  const course=generateChallengeCourse({tierMm:20,logs:true});
  const limits={vx_min:-.2,vx_max:.6,vy_abs:.2,yaw_abs:1.5};
  for(const s of course.stations){
    const route=planPoseRoute(s.entry,s.exit,limits,[],{maxNodes:6500});
    assert.ok(route?.solved,'unsolved circuit crossing for station '+s.index);
    const points=[s.entry,...route.steps];
    let touchesFootprint=false;
    for(let i=1;i<points.length;i++){
      for(let sub=0;sub<=20;sub++){
        const fraction=sub/20;
        const x=points[i-1].x+(points[i].x-points[i-1].x)*fraction;
        const y=points[i-1].y+(points[i].y-points[i-1].y)*fraction;
        const dx=x-s.x,dy=y-s.y;
        const localX=Math.cos(s.yawRad)*dx+Math.sin(s.yawRad)*dy;
        const localY=-Math.sin(s.yawRad)*dx+Math.cos(s.yawRad)*dy;
        if(Math.abs(localX)<.039&&Math.abs(localY)<.195)touchesFootprint=true;
      }
    }
    assert.ok(touchesFootprint,'planner detoured AROUND rather than OVER station '+s.index);
  }
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
  assert.match(main,/navigationChallengeCourse\(\)/);
  assert.match(main,/setChallengeOptions/);
});

test('terrain switch defaults to actual circuit, but manual random mode remains selectable',()=>{
  assert.equal(defaultDemoModeForTerrain('park'),'random');
  assert.equal(defaultDemoModeForTerrain('course'),'circuit');
  assert.equal(defaultDemoModeForTerrain('flat'),'random');
  const course=generateChallengeCourse({tierMm:10});
  assert.equal(waypointsForDemoMode('circuit',course).length,24);
  assert.equal(waypointsForDemoMode('random',course),null);
  assert.throws(()=>waypointsForDemoMode('circuit',null),/without a valid physical course/);
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(main,/demoMode\.value=defaultDemoModeForTerrain\(terrainMode\.value\)/);
  assert.match(main,/navigator\.setCircuitWaypoints\(circuit\)/);
  assert.match(main,/sim\?\.setChallengeCircuitOverlay\(circuit\)/);
});

test('forest and grass are loaded in both park and physical challenge modes',()=>{
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  assert.match(sim,/this\.terrainMode==='park'\|\|this\.terrainMode==='course'/);
  assert.match(sim,/challengeRing:this\.terrainMode==='course'/);
  assert.match(sim,/createParkSurface\(THREE,this\.park,this\.grassTexture\)/);
  assert.match(sim,/createTreeCrowns\(THREE,this\.park\.trees\)/);
  assert.match(sim,/setChallengeCircuitOverlay\(waypoints=null\)/);
});

