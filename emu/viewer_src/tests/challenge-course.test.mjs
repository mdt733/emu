import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {
  CHALLENGE_RADIUS_M,CHALLENGE_WIDTH_M,CHALLENGE_YAW_OFFSETS_DEG,
  CHALLENGE_PALETTE,generateChallengeCourse,challengeCourseXml,
  challengeProbe,visibleChallengeLabel,
} from '../src/challenge-course.js';
import {withTerrain,generateTerrainPark,surfaceSlope} from '../src/terrain.js';
import {defaultDemoModeForTerrain,waypointsForDemoMode} from '../src/demo-mode.js';
import {ARENA_RADIUS_M,GoalNavigator,planPoseRoute,planCircuitLeg}
  from '../src/goal-navigation.js';

const xml='<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.05" /></worldbody></mujoco>';
const near=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5};
const midAngle=(i,count=8)=>2*Math.PI*(i+.5)/count;

test('EXACTLY eight real goals halfway between obstacles, heading CCW without reversal',()=>{
  assert.equal(CHALLENGE_RADIUS_M,ARENA_RADIUS_M);
  const course=generateChallengeCourse({tierMm:20,logs:true});
  assert.equal(course.direction,'counterclockwise');
  assert.equal(course.stations.length,8);
  assert.equal(course.waypoints.length,8);
  for(let i=0;i<course.stations.length;i++){
    const station=course.stations[i],goal=course.waypoints[i],a=midAngle(i);
    near(Math.hypot(station.x,station.y),4,1e-6);
    near(goal.x,4*Math.cos(a),1e-6);
    near(goal.y,4*Math.sin(a),1e-6);
    near(Math.hypot(goal.x,goal.y),4,1e-6);
    near(goal.heading,a+Math.PI/2,1e-12);
    assert.equal(goal.phase,'between');
    assert.equal(goal.station,i);
    assert.equal(goal.via.station,i);
    near(goal.via.x,station.x);
    near(goal.via.y,station.y);
    near(goal.via.heading,station.headingRad);
    assert.ok(!('entry' in goal)&&!('exit' in goal)&&!('arc' in goal));
    if(i>0)near(goal.heading-course.waypoints[i-1].heading,
                2*Math.PI/8,1e-12);
  }
  // Wrap from last gap at 337.5° to first at 22.5° must remain CCW.
  near(wrap(course.waypoints[0].heading-course.waypoints[7].heading),
       2*Math.PI/8,1e-12);
});

test('twice-wide physical obstacle geometry, angle variation and all families',()=>{
  const course=generateChallengeCourse({tierMm:20,logs:true});
  assert.equal(CHALLENGE_WIDTH_M,.80);
  assert.deepEqual(course.stations.map(s=>s.kind),[
    'step','step_up','step_down','log',
    'step','step_up','step_down','log']);
  assert.ok(course.stations.some(s=>s.heightMm===20));
  assert.ok(course.stations.some(s=>s.heightMm===5));
  assert.deepEqual(course.stations.map(s=>s.yawOffsetDeg),
                   CHALLENGE_YAW_OFFSETS_DEG);
  for(const station of course.stations){
    near(wrap(station.yawRad-station.headingRad),
         wrap(station.yawOffsetDeg*Math.PI/180),1e-12);
    near(station.widthM,.80);
  }
  const out=challengeCourseXml(course);
  assert.equal((out.match(/name="viewer_challenge_step_[04]"/g)||[]).length,2);
  assert.equal((out.match(/name="viewer_challenge_step_up_[15]_plateau"/g)||[]).length,2);
  assert.equal((out.match(/name="viewer_challenge_step_up_[15]_return_ramp"/g)||[]).length,2);
  assert.equal((out.match(/name="viewer_challenge_step_down_[26]_plateau"/g)||[]).length,2);
  assert.equal((out.match(/name="viewer_challenge_step_down_[26]_return_ramp"/g)||[]).length,2);
  assert.equal((out.match(/name="viewer_challenge_log_[37]"/g)||[]).length,2);
  assert.match(out,/type="capsule"/);
  assert.match(out,/size="0\.04 0\.4 [^"]+"/);
  assert.match(out,/quat="[^"]+"/); // real oriented down/up ramps
  assert.match(out,/euler="0 0 [^"]+"/);
  assert.match(visibleChallengeLabel(course),/half-buried experimental logs/);
  const noLogs=generateChallengeCourse({tierMm:20,logs:false});
  assert.equal(noLogs.stations.filter(s=>s.kind==='log').length,0);
  const empty=generateChallengeCourse({tierMm:0});
  assert.equal(challengeCourseXml(empty),'');
  assert.match(visibleChallengeLabel(empty),/no obstacles/);
  assert.throws(()=>generateChallengeCourse({tierMm:30}),/0, 5, 10, 15 or 20/);
});

test('uneven physical hfield beneath actual ring, local obstacle Z and half-buried logs',()=>{
  const park=generateTerrainPark({maxDegrees:10,trees:true,challengeRing:true});
  const course=generateChallengeCourse({tierMm:20,logs:true});
  const heights=course.stations.map(s=>park.sampleHeight(s.x,s.y));
  assert.ok(heights.every(h=>h>0.001),'the obstacle ring must be OUTSIDE the flat spawn patch');
  assert.ok(Math.max(...heights)-Math.min(...heights)>.005,
    'stations must see varied real ground heights; no flattened ring');
  assert.equal(surfaceSlope('course',park,0,0).degrees,0);
  assert.ok(park.trees.length>=5,'challenge course has real forest');
  assert.ok(park.trees.some(t=>Math.hypot(t.x,t.y)<3),'trees visible inside arena');
  assert.ok(park.trees.some(t=>Math.hypot(t.x,t.y)>5),'outer forest retained');
  for(const tree of park.trees)
    assert.ok(Math.abs(Math.hypot(tree.x,tree.y)-4)>=1.08,
      'no collidable trunk in the crossing lane');
  const out=withTerrain(xml,'course',{course,park});
  assert.match(out,/<geom name="viewer_park_ground" type="hfield"/);
  assert.match(out,/viewer_park_hfield/);
  assert.match(out,/viewer_tree_trunk_0/);
  assert.doesNotMatch(out,/name="floor"/);
  assert.equal((out.match(/name="viewer_challenge_/g)||[]).length,12);
  assert.throws(()=>withTerrain(out,'course',{course,park}),/already inserted/);
  // A literal step top must be raised from *sampled local ground*.
  const st=course.stations[0],top=park.sampleHeight(st.x,st.y)+st.heightMm/1000;
  const box=out.match(/name="viewer_challenge_step_0" type="box"\s+pos="([^"]+)"\s+size="([^"]+)"/);
  assert.ok(box);
  const pos=box[1].split(' ').map(Number),size=box[2].split(' ').map(Number);
  near(pos[2]+size[2],top,1e-6);
  near(size[1],.4);
  // Capsule centreline is ON local ground, not at ground plus radius.
  const log=course.stations.find(s=>s.kind==='log');
  const hit=out.match(new RegExp(`name="viewer_challenge_log_${log.index}"[^>]*?fromto="([^"]+)"`));
  assert.ok(hit);
  const endpoints=hit[1].split(' ').map(Number);
  near(endpoints[2],park.sampleHeight(endpoints[0],endpoints[1]),2e-6);
  near(endpoints[5],park.sampleHeight(endpoints[3],endpoints[4]),2e-6);
  // The inverted family is the ramp-up followed by a sharp step-down;
  // both up/down modules have physical top and slanted ramp geoms.
  const up=course.stations.find(s=>s.kind==='step_up');
  const down=course.stations.find(s=>s.kind==='step_down');
  near(challengeProbe(up,park.sampleHeight).surfaceZ,
       park.sampleHeight(challengeProbe(up).x,challengeProbe(up).y)+up.heightMm/1000,1e-5);
  near(challengeProbe(down,park.sampleHeight).surfaceZ,
       park.sampleHeight(challengeProbe(down).x,challengeProbe(down).y)+down.heightMm/1000,1e-5);
  assert.ok(Number.isFinite(challengeProbe(up,park.sampleHeight,'ramp').surfaceZ));
  assert.ok(Number.isFinite(challengeProbe(down,park.sampleHeight,'ramp').surfaceZ));
  const flatMode=withTerrain(xml,'flat');
  assert.equal(flatMode,xml);
  assert.match(withTerrain(xml,'ramp'),/viewer_ramp_up/);
  assert.match(withTerrain(xml,'park',{park}),/viewer_park_hfield/);
});

test('one midpoint target per leg, internal A* crosses each obstacle and never stops on it',()=>{
  const course=generateChallengeCourse({tierMm:20,logs:true});
  const park=generateTerrainPark({maxDegrees:10,trees:true,challengeRing:true});
  let start={x:0,y:0,heading:0};
  // Initial move finds first between-obstacle waypoint without any via.
  let first={...course.waypoints[0]};delete first.via;
  const into=planCircuitLeg(start,first,limits,park.trees);
  assert.ok(into?.solved);
  assert.equal(into.steps.some(s=>s.viaCrossing),false);
  start=first;
  // Then eight consecutive crossings including the 360° wraparound.
  for(let step=1;step<=8;step++){
    const target=course.waypoints[step%8];
    const route=planCircuitLeg(start,target,limits,park.trees,{maxNodes:6500});
    assert.ok(route?.solved,`no path from gap ${start.station} to ${target.station}`);
    assert.equal(route.crossingStation,target.station);
    const crossings=route.steps.filter(s=>s.viaCrossing);
    assert.equal(crossings.length,1,'a crossing is a route shape, not another goal');
    near(crossings[0].x,target.via.x);
    near(crossings[0].y,target.via.y);
    assert.ok(crossings[0].cmd.vx>0,'no reverse or stop at the obstacle via point');
    assert.equal(crossings[0].final,undefined);
    assert.equal(route.steps.at(-1).final,true);
    near(route.steps.at(-1).x,target.x);
    near(route.steps.at(-1).y,target.y);
    start=target;
  }
  // Replans AFTER the obstacle must not send EMU BACK over the obstacle.
  const target=course.waypoints[2];
  const obstacle=target.via,dir=obstacle.heading;
  const after={x:obstacle.x+.25*Math.cos(dir),
               y:obstacle.y+.25*Math.sin(dir),heading:dir};
  const finish=planCircuitLeg(after,target,limits,park.trees);
  assert.ok(finish?.solved);
  assert.equal(finish.steps.some(s=>s.viaCrossing),false);
});

test('worker advances ordered midpoint destinations; failure never samples random',async()=>{
  const navigator=new GoalNavigator();
  const course=generateChallengeCourse({tierMm:10,logs:true});
  const posted=[];
  navigator.useBackgroundPlanner(async req=>{
    posted.push(req);
    return {goal:req.goal,route:planCircuitLeg(req.pose,req.goal,req.limits,req.trees)};
  });
  navigator.active=true;
  navigator.setCircuitWaypoints(course.waypoints);
  navigator.nextGoal({x:0,y:0,heading:0},limits,[]);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(posted[0].kind,'target');
  assert.equal(posted[0].goal.via,undefined);
  assert.equal(navigator.goal.label,course.waypoints[0].label);
  assert.equal(navigator.circuitNextIndex,1);
  navigator.nextGoal(course.waypoints[0],limits,[]);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(posted[1].kind,'target');
  assert.equal(posted[1].goal.via.station,1);
  assert.equal(navigator.goal.label,course.waypoints[1].label);
  assert.equal(navigator.circuitNextIndex,2);
  assert.deepEqual(posted.map(p=>p.kind),['target','target']);
  assert.equal(navigator.status(course.waypoints[1]).circuitDirection,'counterclockwise');

  const broken=new GoalNavigator();
  const missing=[];
  broken.useBackgroundPlanner(async req=>{missing.push(req);return {goal:req.goal,route:null};});
  broken.active=true;
  broken.setCircuitWaypoints(course.waypoints);
  broken.nextGoal({x:0,y:0,heading:0},limits,[]);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(missing.length,1);
  assert.equal(missing[0].kind,'target');
  assert.equal(broken.circuitNextIndex,0);
  assert.equal(broken.goal,null);
});

test('public/live UI uses the shared CCW gap circuit and remains manually switchable',()=>{
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  for(const id of ['challengeTier','challengeLogs','demoMode',
    'challengeControls','challengeDescription','demoRouteNote']){
    assert.match(html,new RegExp(`id="${id}"`));
    assert.ok(main.includes("$('"+id+"')"),'missing main binding '+id);
  }
  assert.equal(defaultDemoModeForTerrain('course'),'circuit');
  assert.equal(defaultDemoModeForTerrain('park'),'random');
  const course=generateChallengeCourse({tierMm:10});
  assert.equal(waypointsForDemoMode('circuit',course).length,8);
  assert.equal(waypointsForDemoMode('random',course),null);
  assert.throws(()=>waypointsForDemoMode('circuit',null),/without a valid physical course/);
  assert.match(main,/demoMode\.value=defaultDemoModeForTerrain\(terrainMode\.value\)/);
  assert.match(main,/navigator\.setCircuitWaypoints\(circuit\)/);
  assert.match(main,/setChallengeCircuitOverlay\(circuit\)/);
  const sim=readFileSync(new URL('../src/sim.js',import.meta.url),'utf8');
  assert.match(sim,/this\.terrainMode==='park'\|\|this\.terrainMode==='course'/);
  assert.match(sim,/createParkSurface\(THREE,this\.park,this\.grassTexture\)/);
  assert.match(sim,/createTreeCrowns\(THREE,this\.park\.trees\)/);
  assert.match(sim,/setChallengeCircuitOverlay\(waypoints=null\)/);
});
