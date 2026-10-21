// Optional viewer-only V terrain memory overlay. It never affects controls.
import { V_SAMPLES, V_CELLS } from './terrain-memory.js';
const $=id=>document.getElementById(id);
export class TerrainMemoryPanel {
  constructor(){
    this.shell=$('terrainMemoryPanel');
    this.toggle=$('terrainMemoryToggle');
    this.close=$('terrainMemoryClose');
    this.rays=$('terrainRaysToggle');
    this.freeze=$('terrainMemoryFreeze');
    this.charts=[$('terrainRightCanvas'),$('terrainLeftCanvas')];
    this.status=$('terrainMemoryStatus');
    this.samples=$('terrainMemorySamples');
    this.lastDraw=0;
    this.toggle.addEventListener('click',()=>this.open(this.shell.hidden));
    this.close.addEventListener('click',()=>this.open(false));
    this.open(false);
  }
  open(on){
    this.shell.hidden=!on;
    this.toggle.setAttribute('aria-expanded',String(on));
    this.toggle.textContent=on?'Hide terrain scan':'Terrain scan';
  }
  get enabled(){return !this.shell.hidden;}
  draw(sim,timeMs){
    if(!this.enabled||timeMs-this.lastDraw<90)return;
    this.lastDraw=timeMs;
    const d=sim?.terrainMemory?.values();
    if(!d){this.status.textContent='No simulator loaded';return;}
    for(let side=0;side<2;side++)this.drawChart(
      this.charts[side],d.heights[side],d.confidence[side],side);
    const mm=x=>Number.isFinite(x)?Math.round(x*1000)+' mm':'—';
    const actor=d.actorHeights.map((h,i)=>d.actorConfidence[i]>.05?mm(h):'·');
    this.samples.textContent=
      'Right −5/0/5/10/15: '+actor.slice(0,5).join('  ')+'\n'+
      'Left  −5/0/5/10/15: '+actor.slice(5).join('  ');
    const hits=d.beams.filter(q=>q.valid).length;
    this.status.textContent=
      'Hits '+hits+'/5  ·  vx '+d.motion.vx.toFixed(2)+
      '  vy '+d.motion.vy.toFixed(2)+' m/s · yaw '+
      d.motion.yaw.toFixed(2)+' · active '+d.activeSamples+'/102'+
      (this.freeze.checked?' · FROZEN':'');
  }
  drawChart(canvas,heights,confidence,side) {
    const w=canvas.clientWidth||300,h=canvas.clientHeight||108;
    const scale=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(w*scale)),height=Math.max(1,Math.round(h*scale));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    const g=canvas.getContext('2d');if(!g)return;
    g.setTransform(scale,0,0,scale,0,0);
    g.clearRect(0,0,w,h);
    g.fillStyle='rgba(13,24,34,.68)';g.fillRect(0,0,w,h);
    const l=28,r=w-7,top=10,bottom=h-22,zero=top+(bottom-top)*.64;
    g.strokeStyle='rgba(167,190,207,.38)';
    g.beginPath();g.moveTo(l,zero+.5);g.lineTo(r,zero+.5);g.stroke();
    for(let i=0;i<V_CELLS;i++){
      const x=l+(r-l)*(i+.5)/V_CELLS,px=(r-l)/V_CELLS;
      const c=confidence[i]||0,value=heights[i]||0;
      const level=Math.min(44,Math.abs(value)*360);
      if(c<=.01){
        g.fillStyle='rgba(117,138,159,.20)';
        g.fillRect(x-px*.4,zero-1,Math.max(1,px*.78),2);
      }else{
        g.fillStyle=side===0?'rgba(103,211,235,'+Math.max(.18,c)+')':
                             'rgba(248,177,116,'+Math.max(.18,c)+')';
        g.fillRect(x-px*.43,value>=0?zero-level:zero,
                   Math.max(1,px*.85),Math.max(1,level));
      }
    }
    for(let i=0;i<V_SAMPLES.length;i++){
      const x=l+(r-l)*(V_SAMPLES[i]+.5)/V_CELLS;
      g.strokeStyle='rgba(253,236,143,.88)';
      g.beginPath();g.moveTo(x,top);g.lineTo(x,bottom);g.stroke();
      g.fillStyle='#dce7f1';g.font='9px system-ui';g.textAlign='center';
      g.fillText(['−5','0','5','10','15'][i],x,h-7);
    }
    g.fillStyle='#8599ac';g.textAlign='left';g.font='9px system-ui';
    g.fillText('−10',2,h-7);
    g.textAlign='right';g.fillText('+40',w-2,h-7);
  }
}
