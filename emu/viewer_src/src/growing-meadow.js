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
  // Original user-owned Meadow skybox.png is the preferred panorama.
  // This viewer is not a copy of Meadow's WebGL engine; sample the SAME
  // 360° equirectangular image directly in a lightweight Three.js sphere.
  // Until the image downloads (or offline), retain a procedural fallback.
  const skyFallback=new THREE.DataTexture(
    new Uint8Array([126,178,222,255]),1,1,THREE.RGBAFormat);
  skyFallback.needsUpdate=true;
  let skyDisposed=false,loadedSky=null;
  const sky=new THREE.Mesh(
    new THREE.SphereGeometry(46,64,24),
    new THREE.ShaderMaterial({
      side:THREE.BackSide,depthWrite:false,depthTest:false,fog:false,
      uniforms:{uSky:{value:skyFallback},uUseSkyImage:{value:0}},
      vertexShader:`
        varying vec3 vDirection;
        void main(){
          vDirection=position;
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
        }`,
      fragmentShader:`
        varying vec3 vDirection;
        uniform sampler2D uSky;
        uniform float uUseSkyImage;
        void main(){
          vec3 d=normalize(vDirection);
          // Original Meadow skyFS uses Y-up; viewer is Z-up. Keep its
          // latitude/azimuth panorama coordinates after this axis swap.
          float a=atan(d.y,d.x),lat=asin(clamp(d.z,-1.,1.));
          float u=a/6.2831853+.5;
          float v=clamp(.5-lat/3.14159265-.035,.001,.999);
          vec3 sky;
          if(uUseSkyImage>.5){
            sky=texture2D(uSky,vec2(u,v)).rgb;
            const float seam=.0025;
            if(u<seam)sky=mix(texture2D(uSky,
              vec2(1.-seam+u,v)).rgb,sky,smoothstep(0.,seam,u));
            else if(u>1.-seam)sky=mix(sky,
              texture2D(uSky,vec2(u-1.+seam,v)).rgb,
              smoothstep(1.-seam,1.,u));
          }else{
            // Meadow's original procedural fallback, adapted to Z-up.
            float horizon=exp(-pow((lat+.055)*3.9,2.));
            sky=mix(vec3(.68,.79,.89),vec3(.29,.52,.77),
              smoothstep(-.24,.91,d.z));
            sky=mix(sky,vec3(1.,.78,.59),horizon*.51);
            float wisps=sin(11.*a+1.3*sin(3.*a)+24.*lat)*
                        sin(19.*a-14.*lat);
            float clouds=smoothstep(.47,.91,wisps)*
              smoothstep(.07,.28,d.z)*(1.-smoothstep(.56,.76,d.z));
            sky=mix(sky,vec3(.97,.94,.85),clouds*.21);
            float glow=pow(max(0.,dot(d,
              normalize(vec3(-.65,.74,.17)))),28.);
            sky+=vec3(1.,.73,.43)*glow*.22;
          }
          gl_FragColor=vec4(sky,1.);
        }`
    })
  );
  sky.name='Original Meadow panorama with procedural offline fallback';
  // Download the exact panorama already committed in the user's Meadow repo.
  // CORS on raw.githubusercontent.com permits use as a WebGL texture.
  const originalSkyUrl='https://raw.githubusercontent.com/mdt733/meadow/main/skybox.png';
  new THREE.TextureLoader().load(originalSkyUrl,texture=>{
    if(skyDisposed){texture.dispose();return;}
    texture.flipY=false;
    texture.wrapS=THREE.RepeatWrapping;
    texture.wrapT=THREE.ClampToEdgeWrapping;
    texture.colorSpace=THREE.SRGBColorSpace;
    texture.needsUpdate=true;
    loadedSky=texture;
    sky.material.uniforms.uSky.value=texture;
    sky.material.uniforms.uUseSkyImage.value=1;
  },undefined,()=>{
    // Original panorama unavailable (e.g. offline). Shader fallback remains.
    console.info('Meadow panorama unavailable; using procedural sky');
  });
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
  const mountainGroup=new THREE.Group();root.add(mountainGroup);
  mountainLayers.forEach((layer,k)=>{
    const mesh=new THREE.Mesh(mountainGeo(layer,k),
      new THREE.MeshBasicMaterial({color:layer.color,
        transparent:true,opacity:layer.opacity,
        fog:false,depthWrite:false,side:THREE.DoubleSide}));
    mesh.renderOrder=-900+k*5;
    mountainGroup.add(mesh);
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
    if(!location)return null;
    const ground=park.sampleHeight(location.x,location.y);
    if(!Number.isFinite(ground))return null;
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
    return group;
  }

  const apronMaterial=new THREE.MeshBasicMaterial({
    color:0x789c75,transparent:true,opacity:.65,depthWrite:false,
    side:THREE.DoubleSide,fog:false,vertexColors:true});
  let apron=null;
  function drawFallingEdge(){
    if(apron){root.remove(apron);apron.geometry.dispose();}
    const minX=park.anchor[0],minY=park.anchor[1];
    const maxX=minX+15,maxY=minY+15;
    const pos=[],colors=[];
    const push=(x,y,z,t)=>{
      pos.push(x,y,z);
      const pale=[.56,.66,.60],dark=[.18,.31,.28];
      colors.push(...pale.map((v,k)=>v*(1-t)+dark[k]*t));
    };
    function quad(a,b,c,d){
      for(const p of [a,b,c,a,c,d])push(...p);
    }
    // Only decorative terrain OUTSIDE the actual 225 collision tiles;
    // the 2-m falling apron has no MuJoCo contacts or ToF support.
    for(let k=0;k<15;k++){
      for(const side of [0,1,2,3]){
        const points=side===0?[[minX+k,minY],[minX+k+1,minY]]:
          side===1?[[minX+k,maxY],[minX+k+1,maxY]]:
          side===2?[[minX,minY+k],[minX,minY+k+1]]:
                   [[maxX,minY+k],[maxX,minY+k+1]];
        const [a,b]=points;
        const height=p=>park.previewHeight(p[0]===maxX?p[0]-.0001:p[0],
           p[1]===maxY?p[1]-.0001:p[1]);
        const out=p=>side===0?[p[0],p[1]-2.5]:
          side===1?[p[0],p[1]+2.5]:
          side===2?[p[0]-2.5,p[1]]:[p[0]+2.5,p[1]];
        const aa=out(a),bb=out(b);
        quad([a[0],a[1],height(a),0],[b[0],b[1],height(b),0],
             [bb[0],bb[1],height(b)-2.5,1],
             [aa[0],aa[1],height(a)-2.5,1]);
      }
    }
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
    g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    g.computeVertexNormals();
    apron=new THREE.Mesh(g,apronMaterial);
    apron.frustumCulled=false;
    root.add(apron);
  }

  let lastRevision=-1;
  function sync(){
    if(lastRevision===park.revision)return;
    lastRevision=park.revision;
    // Same slot can represent new global coordinates. Dispose its old
    // tree group instead of leaving static trees behind the robot.
    for(const tile of park.window()){
      const previous=trees.get(tile.slot),key=tile.i+','+tile.j;
      if(previous?.key===key)continue;
      if(previous?.group)root.remove(previous.group);
      trees.set(tile.slot,{key,group:createTree(tile)});
    }
    const flowers=[];
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
    drawFallingEdge();
  }
  sync();

  return {
    sync,
    update(cameraPosition){
      // Skybox follows only the camera: never changes terrain or sensors.
      sky.position.copy(cameraPosition);
      mountainGroup.position.set(cameraPosition.x,cameraPosition.y,0);
    },
    dispose(){
      skyDisposed=true;
      loadedSky?.dispose();
      skyFallback.dispose();
      root.parent?.remove(root);
      if(apron)apron.geometry.dispose();
      apronMaterial.dispose();
      for(const child of mountainGroup.children){
        child.geometry.dispose();child.material.dispose();
      }
      sky.geometry.dispose();
      for(const child of root.children){
        if(child.geometry&&child.geometry!==trunkGeo&&child.geometry!==crownGeo&&child!==sky)
          child.geometry.dispose();
        if(child.material&&child.material!==trunkMat&&!crowns.includes(child.material))
          child.material.dispose();
      }
      trunkGeo.dispose();crownGeo.dispose();trunkMat.dispose();
      crowns.forEach(m=>m.dispose());
    }
  };
}
