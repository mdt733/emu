// Meadow-inspired ambience for H40's growing physical prism field.
// Pure Three.js decoration: never used by MuJoCo, ToF or policy state.
// Landscapes use cached mesh geometry, and deterministic tiles keep their trees
// when revisited. Inspired by github.com/mdt733/meadow/assets/meadow.js.
const TAU=Math.PI*2;
const hash=(a,b,seed=735)=>{
  let n=Math.imul(a|0,374761393)^Math.imul(b|0,668265263)^(seed|0);
  n=Math.imul(n^(n>>>13),1274126177);
  return ((n^(n>>>16))>>>0)/4294967296;
};

export function meadowTreeForTile(tile,seed=735){
  const a=hash(tile.i,tile.j,seed);
  // Plant sparse copses, not an obstacle right on the H40 starting line.
  if(a>.18||(tile.i>=-1&&tile.i<=4&&tile.j>=-1&&tile.j<=2))
    return null;
  const x=tile.i+.18+.58*hash(tile.i+3,tile.j-7,seed);
  const y=tile.j+.18+.58*hash(tile.i-5,tile.j+11,seed);
  const tall=.48+.38*hash(tile.i+31,tile.j+3,seed);
  return {x,y,tall,branch:hash(tile.i+23,tile.j+15,seed)};
}

export function createGrowingMeadowScenery(THREE,scene,park){
  const root=new THREE.Group();
  root.name='Meadow visual scenery (no collision)';
  scene.add(root);
  const sky=new THREE.Mesh(
    new THREE.SphereGeometry(46,64,24),
    new THREE.ShaderMaterial({
      side:THREE.BackSide,depthWrite:false,depthTest:false,fog:false,
      vertexShader:`
        varying vec3 vDirection;
        void main(){
          vDirection=position;
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
        }`,
      fragmentShader:`
        varying vec3 vDirection;
        void main(){
          vec3 d=normalize(vDirection);
          float a=atan(d.y,d.x);
          float horizon=exp(-pow((d.z+.045)*4.1,2.));
          vec3 sky=mix(vec3(.92,.75,.63),vec3(.30,.55,.83),
            smoothstep(-.22,.77,d.z));
          sky=mix(sky,vec3(1.,.82,.65),horizon*.34);
          float w=sin(11.*a+1.3*sin(3.*a)+22.*d.z)*
                  sin(19.*a-13.*d.z);
          float cloud=smoothstep(.57,.95,w)*
            smoothstep(.04,.30,d.z)*(1.-smoothstep(.57,.83,d.z));
          sky=mix(sky,vec3(.99,.96,.88),cloud*.28);
          float sun=pow(max(0.,dot(d,normalize(vec3(-.64,.20,.74)))),48.);
          sky+=vec3(1.,.77,.48)*sun*.24;
          gl_FragColor=vec4(sky,1.);
        }`
    })
  );
  sky.name='Meadow procedural golden-hour skybox';
  sky.renderOrder=-1000;
  sky.frustumCulled=false;
  root.add(sky);

  // Three closed mountain horizons with far-to-near atmospheric colour.
  // These rings are visual-only and live well beyond the actual Carpet edge.
  const mountainLayers=[
    {radius:43,base:-5,mean:1.25,relief:2.6,color:0xa9bdc7,opacity:.55},
    {radius:36,base:-5,mean:1.05,relief:2.3,color:0x839fac,opacity:.78},
    {radius:29,base:-5,mean:.45,relief:1.65,color:0x678b83,opacity:.80}
  ];
  const mountainGeo=(layer,k)=>{
    const points=[],segments=120;
    for(let n=0;n<segments;n++){
      const crest=i=>{
        const angle=i*TAU/segments;
        const jag=.45*Math.sin(angle*(5+k*2)+k*.9)+
          .25*Math.sin(angle*(13+k*3)-k*1.2)+
          .16*Math.sin(angle*(29+k*2)+k*2.3);
        const peak=layer.mean+layer.relief*(.49+jag);
        return [Math.cos(angle)*layer.radius,
                Math.sin(angle)*layer.radius,peak];
      };
      const a=crest(n),b=crest(n+1);
      const lowA=[a[0],a[1],layer.base],
            lowB=[b[0],b[1],layer.base];
      points.push(...a,...lowA,...b,...b,...lowA,...lowB);
    }
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
    g.computeVertexNormals();
    return g;
  };
  mountainLayers.forEach((layer,k)=>{
    const mesh=new THREE.Mesh(mountainGeo(layer,k),
      new THREE.MeshBasicMaterial({color:layer.color,
        transparent:true,opacity:layer.opacity,
        fog:false,depthWrite:false,side:THREE.DoubleSide}));
    mesh.renderOrder=-900+k*5;
    root.add(mesh);
  });

  const trunkGeo=new THREE.CylinderGeometry(.032,.065,1,6);
  trunkGeo.rotateX(Math.PI/2);
  const crownGeo=new THREE.IcosahedronGeometry(1,1);
  const trunkMat=new THREE.MeshStandardMaterial({
    color:0x70513b,roughness:1,flatShading:true});
  const crowns=[0x466d40,0x678b49,0x56824f,0x728e55]
    .map(c=>new THREE.MeshStandardMaterial({
      color:c,roughness:1,flatShading:true}));
  const trees=new Map();

  // Flower colour points: scenery only, small enough not to obscure legs.
  const flowerGeometry=new THREE.BufferGeometry(),flowers=[];
  flowerGeometry.setAttribute('position',new THREE.Float32BufferAttribute([],3));
  const flowerCloud=new THREE.Points(flowerGeometry,new THREE.PointsMaterial({
    color:0xf0d6a2,size:.027,sizeAttenuation:true,transparent:true,
    opacity:.76,depthWrite:false}));
  root.add(flowerCloud);

  function createTree(tile){
    const location=meadowTreeForTile(tile,park.seed);
    if(!location)return;
    const ground=park.sampleHeight(location.x,location.y);
    if(!Number.isFinite(ground))return;
    const group=new THREE.Group();
    group.position.set(location.x,location.y,ground);
    const trunk=new THREE.Mesh(trunkGeo,trunkMat);
    trunk.scale.z=location.tall*.54;
    trunk.position.z=location.tall*.27;
    const foliage=new THREE.Mesh(crownGeo,
      crowns[Math.floor(location.branch*crowns.length)]);
    foliage.scale.set(.18+location.branch*.11,
      .17+location.branch*.1,location.tall*.34);
    foliage.position.set(0,0,location.tall*.78);
    foliage.rotation.z=location.branch*TAU;
    group.add(trunk,foliage);
    group.userData.treeTile=tile.slot;
    root.add(group);
    trees.set(tile.slot,group);
  }

  function sync(){
    // No tree gets regenerated when a tile is revisited; reset hides the old
    // decorative patches just as it hides the old physical terrain.
    for(const [slot,mesh] of trees)
      mesh.visible=Boolean(park.tiles[slot]?.active);
    for(const tile of park.window())
      if(!trees.has(tile.slot))createTree(tile);
    // Rebuild a tiny points-only tuft set. The set is bounded by 225 tiles.
    flowers.length=0;
    for(const tile of park.window())for(let k=0;k<2;k++){
      const x=tile.i+.18+.65*hash(tile.i+k+13,tile.j-31,park.seed);
      const y=tile.j+.16+.68*hash(tile.i-17,tile.j+k+41,park.seed);
      const z=park.sampleHeight(x,y);
      if(Number.isFinite(z))flowers.push(x,y,z+.015);
    }
    flowerCloud.geometry.dispose();
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(flowers,3));
    flowerCloud.geometry=g;
  }
  sync();

  return {
    sync,
    update(cameraPosition){
      // Skybox follows only the camera: never changes terrain or sensors.
      sky.position.copy(cameraPosition);
    },
    dispose(){
      root.parent?.remove(root);
      for(const child of root.children){
        if(child.geometry&&child.geometry!==trunkGeo&&child.geometry!==crownGeo)
          child.geometry.dispose();
        if(child.material&&child.material!==trunkMat&&!crowns.includes(child.material))
          child.material.dispose();
      }
      trunkGeo.dispose();crownGeo.dispose();trunkMat.dispose();
      crowns.forEach(m=>m.dispose());
    }
  };
}
