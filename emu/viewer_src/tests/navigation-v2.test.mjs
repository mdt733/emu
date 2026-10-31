import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {buildNavigationTerrain,terrainRiskAt,sweptTerrainCost} from '../src/navigation-terrain.js';
import {OnlineGaitResponse,PredictivePathController,vectorFieldGuidance,
  clampNavigationCommand} from '../src/predictive-navigation.js';
import {buildRoutePath,trackRoutePath} from '../src/path-follower.js';
import {planPoseRoute,integratePose} from '../src/goal-navigation.js';

const limits={vx_min:-.53,vx_max:1.04,vy_abs:.22,yaw_abs:1.5};
const pose={x:0,y:0,heading:0,vx:0,vy:0,yawRate:0};
const text=path=>readFileSync(fileURLToPath(new URL(path,import.meta.url)),'utf8');

test('4x4 depth laboratory lives only inside Settings and starts inactive',()=>{
  const html=text('../index.html'),css=text('../src/style.css');
  assert.ok(html.indexOf('id="settingsDrawer"')<html.indexOf('id="depthPanel"'));
  assert.ok(html.indexOf('id="depthPanel"')<html.indexOf('id="resultCard"'));
  assert.match(html,/<details class="drawer-group" id="depthDetails">/);
  assert.match(html,/<input id="depthOn" type="checkbox" \/>/);
  assert.match(css,/\.settings-drawer \.depth-panel \{\s*position:static/);
  assert.doesNotMatch(css,/\.depth-panel \{[\s\S]*?position:absolute/);
});

test('terrain grade cost is finite on gentle ground and blocks known severe slopes',()=>{
  const grade=(x,y)=>x<0?0:x*.32;
  const terrain=buildNavigationTerrain(grade,{cell:.20,cautionDegrees:4,avoidDegrees:12});
  assert.equal(terrainRiskAt(null,1,1),0);
  assert.equal(terrainRiskAt(terrain,-1,0),0);
  assert.equal(terrainRiskAt(terrain,1,0),Infinity);
  assert.equal(sweptTerrainCost(terrain,{x:-1,y:0},{x:1,y:0}),Infinity);
  const flat=buildNavigationTerrain(()=>0);
  assert.equal(terrainRiskAt(flat,1,0),0);
  assert.equal(sweptTerrainCost(flat,{x:0,y:0},{x:1,y:0}),1);
  assert.equal(planPoseRoute(pose,{x:1,y:0,heading:0},limits,[],{terrain}),null);
  assert.ok(planPoseRoute(pose,{x:1,y:0,heading:0},limits,[],{terrain:flat})?.solved);
});

test('vector-field guidance corrects signed off-track drift using vy',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.60,vy:0,yaw:0,profiled:true}},
    {x:2,y:0,heading:0,cmd:{vx:.60,vy:0,yaw:0,profiled:true}}]});
  const straight=path.runs[0];
  const on=vectorFieldGuidance({...pose,x:.3},straight,0,limits);
  const off=vectorFieldGuidance({...pose,x:.3,y:.25},straight,0,limits);
  assert.ok(Math.abs(on.command.vy)<1e-4);
  assert.ok(off.command.vy<-.02,'positive y deviation requests rightward correction');
  assert.ok(off.command.vx>0);
});

test('classic route follower preserves diagonal planner vx and vy',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:.18,heading:0,cmd:{vx:.6,vy:.11,yaw:0,profiled:true}},
    {x:2,y:.36,heading:0,cmd:{vx:.6,vy:.11,yaw:0,profiled:true}}]});
  const guided=trackRoutePath({...pose,x:.06,y:.01,vx:.2},
    path,0,0,limits);
  assert.ok(guided.command.vx>.15);
  assert.ok(guided.command.vy>.01,'sideways component should not be thrown away');
});

test('response model predicts inertia and learns safely bounded gains',()=>{
  const model=new OnlineGaitResponse();
  const first=model.rollout(pose,{vx:.7,vy:.1,yaw:.3},.2);
  assert.ok(first.vx>0&&first.vx<.7);
  assert.ok(first.yawRate>0&&first.yawRate<.3);
  assert.ok(first.x>0);
  let actual={...pose};
  for(let i=0;i<45;i++){
    model.issued={vx:.5,vy:.1,yaw:.3};
    model.observe(actual,.1);
    actual=model.rollout(actual,model.issued,.1);
  }
  assert.ok(model.report().samples>30);
  for(const key of ['vx','vy','yaw'])
    assert.ok(Number.isFinite(model.gain[key])&&model.tau[key]>.08);
});

test('predictive rollout returns bounded commands and a finite future trace',()=>{
  const path=buildRoutePath({start:pose,steps:[
    {x:1,y:0,heading:0,cmd:{vx:.65,vy:0,yaw:0,profiled:true}},
    {x:2,y:0,heading:0,cmd:{vx:.65,vy:0,yaw:0,profiled:true}}]});
  const predictive=new PredictivePathController();
  predictive.observe(pose,.1);
  const baseline={vx:.5,vy:0,yaw:0,pitch:0};
  const out=predictive.command(pose,path,0,0,limits,[],null,baseline);
  assert.equal(out.prediction.length,5);
  assert.ok(Number.isFinite(out.cost));
  assert.deepEqual(clampNavigationCommand(out.command,limits),out.command);
  assert.ok(out.prediction.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
  const blank=predictive.command(pose,{runs:[]},0,0,limits,[],null,baseline);
  assert.deepEqual(blank.command,baseline);
});

test('global planner and predictive module stay above the neural actuator contract',()=>{
  const nav=text('../src/goal-navigation.js');
  const main=text('../src/main.js');
  const worker=text('../src/goal-worker.js');
  assert.match(main,/navControllerMode/);
  assert.match(main,/sim\.setHeadingHold\(false\)/);
  assert.match(worker,/terrain/);
  assert.match(nav,/PredictivePathController/);
  assert.match(main,/sim\.setNavigationPrediction/);
  assert.doesNotMatch(text('../src/predictive-navigation.js'),/data\.ctrl|policy\.infer|mj_step/);
});
