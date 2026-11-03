// EMU Prism Lab: visual model only. No MuJoCo collision or GPU claims.
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {buildPrismTerrain,DEFAULT_PRISM_SETTINGS} from './prism-terrain.js';

const $=id=>document.getElementById(id);
const fields=['tiles','seed','surface','steps','slopeDeg','transitionDeg','stepUpMm','stepDownMm','diagonal','exaggeration'];
const switches=['edges','risers','prismEdges','foot','transparent'];
const formatters={slopeDeg:v=>`${v}°`,transitionDeg:v=>`${v}°`,stepUpMm:v=>`${v} mm`,stepDownMm:v=>`${v} mm`,exaggeration:v=>`×${v}`};
const param=new URLSearchParams(location.search);
let incoming={};for(const key of fields)if(param.has(key))incoming[key]=param.get(key);
let settings;
try{settings=buildPrismTerrain({...DEFAULT_PRISM_SETTINGS,...incoming}).settings;}catch{settings={...DEFAULT_PRISM_SETTINGS};}
for(const key of fields)$(key).value=settings[key];
for(const key of Object.keys(formatters))$(key+'Out').textContent=formatters[key](settings[key]);

const scene=new THREE.Scene();scene.background=new THREE.Color(0xadcbd0);scene.fog=new THREE.Fog(0xadcbd0,9,23);
const renderer=new THREE.WebGLRenderer({canvas:$('prismCanvas'),antialias:true,preserveDrawingBuffer:true,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputColorSpace=THREE.SRGBColorSpace;
const camera=new THREE.PerspectiveCamera(43,1,.025,60);
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=.09;
controls.minDistance=.75;controls.maxDistance=18;
controls.maxPolarAngle=Math.PI*.495;
scene.add(new THREE.HemisphereLight(0xffffff,0x64816d,2.5));
const sun=new THREE.DirectionalLight(0xfff1cf,3.0);
sun.position.set(-2.5,6,4);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.left=-6;sun.shadow.camera.right=6;sun.shadow.camera.top=6;sun.shadow.camera.bottom=-6;
sun.shadow.bias=-.0001;scene.add(sun);
const group=new THREE.Group();scene.add(group);
const topMaterial=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.90,metalness:0,side:THREE.DoubleSide});
const stepMaterial=new THREE.MeshBasicMaterial({color:0xffad54,opacity:.78,transparent:true,depthWrite:false,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:-3});
const seamMaterial=new THREE.LineBasicMaterial({color:0x133c3f,transparent:true,opacity:.75});
const innerMaterial=new THREE.LineBasicMaterial({color:0xe9f7b2,transparent:true,opacity:.26});
const riseMaterial=new THREE.LineBasicMaterial({color:0xffc676,linewidth:2});
const footGeometry=new THREE.SphereGeometry(.018,18,12);
const footMaterial=new THREE.MeshStandardMaterial({color:0xffe4ad,metalness:.1,roughness:.35});
let model=null,grid=null,referenceFoot=null;
const to3=p=>[p[0],p[2],p[1]]; // world [x,y,z] elevation -> THREE [x,z,y]
function renderPrism(p){
  const vertices=[],colors=[];
  const topTint=new THREE.Color().setHSL(.29-Math.min(.08,p.angleDeg/120),.29,.46);
  const sideTint=new THREE.Color(0x4e6754);
  const bottomTint=new THREE.Color(0x33463b);
  function tri(a,b,c,tint){
    for(const xyz of [a,b,c]){vertices.push(...to3(xyz));colors.push(tint.r,tint.g,tint.b);}
  }
  // Inverting the winding compensates for the Z-up -> Y-up handedness switch.
  tri(p.upper[0],p.upper[2],p.upper[1],topTint);
  tri(p.lower[0],p.lower[1],p.lower[2],bottomTint);
  for(let k=0;k<3;k++){
    const n=(k+1)%3;
    tri(p.upper[k],p.lower[n],p.lower[k],sideTint);
    tri(p.upper[k],p.upper[n],p.lower[n],sideTint);
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
  geo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  geo.computeVertexNormals();
  const mesh=new THREE.Mesh(geo,topMaterial);mesh.receiveShadow=true;mesh.castShadow=true;
  return mesh;
}
function line(points,material){
  if(!points.length)return;
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(points.flatMap(to3),3));
  const obj=new THREE.LineSegments(g,material);group.add(obj);
}
function makeModel(){
  while(group.children.length){const obj=group.children[0];group.remove(obj);
    if(obj.geometry!==footGeometry)obj.geometry?.dispose();}
  if(grid){scene.remove(grid);grid.geometry.dispose();grid.material.dispose();}
  const preview=buildPrismTerrain(settings);model=preview;
  const {stats,prisms,tiles,risers,baseZ}=preview;
  topMaterial.transparent=$('transparent').checked;
  topMaterial.opacity=topMaterial.transparent?.60:1;
  topMaterial.depthWrite=!topMaterial.transparent;
  for(const prism of prisms)group.add(renderPrism(prism));
  const raised=(p,delta=.001)=>[p[0],p[1],p[2]+delta];
  if($('edges').checked){
    const points=[];
    for(const tile of tiles)for(const t of tile.tops)for(const [a,b] of [[0,1],[1,2],[2,0]]){
      points.push(raised(t[a]),raised(t[b]));
    }
    line(points,seamMaterial);
  }
  if($('prismEdges').checked){
    const pts=[];
    for(const p of prisms)for(let k=0;k<3;k++){
      const next=(k+1)%3;
      pts.push(raised(p.upper[k],.0015),raised(p.lower[k],.0015));
      pts.push(raised(p.lower[k],.0015),raised(p.lower[next],.0015));
    }
    line(pts,innerMaterial);
  }
  if($('risers').checked){
    const orange=[];
    for(const r of risers){
      const [a,b]=r.high,[c,d]=r.low;
      const arr=[a,c,b,b,c,d];
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',
        new THREE.Float32BufferAttribute(arr.flatMap(to3),3));geo.computeVertexNormals();
      const m=new THREE.Mesh(geo,stepMaterial);group.add(m);
      orange.push(raised(a,.002),raised(b,.002));
    }
    line(orange,riseMaterial);
  }
  grid=new THREE.GridHelper(Math.max(4,settings.tiles*.5+1),Math.max(8,settings.tiles*2),0x56736d,0x829d93);
  grid.position.y=baseZ-.006;scene.add(grid);
  const footPos=preview.sampleHeight(.13,.13);
  if($('foot').checked&&footPos!==null){
    const a=new THREE.Mesh(footGeometry,footMaterial),b=new THREE.Mesh(footGeometry,footMaterial);
    a.position.set(.13,footPos+.018,.13);
    const bx=.23,by=.13,bh=preview.sampleHeight(bx,by);
    b.position.set(bx,(bh??footPos)+.018,by);
    group.add(a,b);
  }
  const physicalStep=(stats.intendedMaxMm/settings.exaggeration).toFixed(2);
  $('diagnostics').textContent=`${stats.tiles} squares / ${stats.prisms} triangular prisms\n`+
    `${stats.joinedEdges} continuous shared edges\n`+
    `${stats.intentionalEdges} deliberately stepped edges\n`+
    `ACCIDENTAL SEAM GAP: ${stats.accidentalMaxMm.toFixed(8)} mm\n`+
    `LARGEST INTENTIONAL LIP: ${physicalStep} mm\n`+
    `Display height: ×${settings.exaggeration}\n`+
    `Full geometry: independent 3D prism solids`;
  return preview;
}
function cameraReset(){
  const size=settings.tiles*.5;
  controls.target.set(.25,0,.25);
  camera.position.set(size*1.28,size*1.15+1.3,size*1.50);
  controls.update();
}
function writeUrl(){
  const url=new URL(location.href);for(const f of fields)url.searchParams.set(f,String(settings[f]));
  history.replaceState(null,'',url);
}
function syncInputs(){
  const raw={};for(const f of fields)raw[f]=$(f).value;
  settings=buildPrismTerrain(raw).settings;
  for(const key of Object.keys(formatters))$(key+'Out').textContent=formatters[key](settings[key]);
  makeModel();writeUrl();
}
for(const key of fields)$(key).addEventListener(key==='seed'?'change':'input',syncInputs);
for(const key of switches)$(key).addEventListener('change',makeModel);
$('reroll').addEventListener('click',()=>{$('seed').value=(Number($('seed').value)+1)%2147483647;syncInputs();});
$('resetView').addEventListener('click',cameraReset);
$('copySettings').addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(location.href);$('actionStatus').textContent='Settings link copied. Share it to recreate this exact terrain.';}
  catch{$('actionStatus').textContent='Copy unavailable; copy the browser address to share these settings.';}
});
$('saveImage').addEventListener('click',()=>{
  renderer.render(scene,camera);
  const a=document.createElement('a');a.download=`emu-prism-lab-s${settings.seed}.png`;
  a.href=renderer.domElement.toDataURL('image/png');a.click();
  $('actionStatus').textContent='Screenshot saved. Send it back with feedback on the joins.';
});
$('togglePanel').addEventListener('click',()=>{
  $('labPanel').hidden=!$('labPanel').hidden;
  $('togglePanel').setAttribute('aria-expanded',String(!$('labPanel').hidden));
});
function resize(){const w=innerWidth,h=innerHeight;if(renderer.domElement.width!==Math.round(w*renderer.getPixelRatio())||
  renderer.domElement.height!==Math.round(h*renderer.getPixelRatio()))renderer.setSize(w,h,false);
  camera.aspect=w/h;camera.updateProjectionMatrix();}
window.addEventListener('resize',resize);
makeModel();cameraReset();resize();
function tick(){requestAnimationFrame(tick);controls.update();renderer.render(scene,camera);}
tick();