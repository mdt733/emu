#!/usr/bin/env python3
"""Read-only live viewer for EMU capability fields.

Run with:
    emu view

or from a source checkout:
    python -m emu.field_viewer

The viewer auto-follows the newest capability_field_latest.npz under the current
version's run root. It understands the current field schema only and never writes
to or communicates with the trainer.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from .paths import seed_root

from .capability_field import FIELD_VERSION
from .paths import VERSION

HTML = r'''<!doctype html><meta charset="utf-8"><title>EMU capability field</title>
<style>
:root{color-scheme:dark}body{font-family:system-ui,sans-serif;margin:20px;background:#111;color:#ddd;max-width:1180px}.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}label{font-size:14px}select,input{font-size:16px}canvas{margin-top:14px;border:1px solid #555;max-width:100%;image-rendering:pixelated;background:#000;cursor:crosshair}.muted{color:#999;font-size:13px}.stats{font-family:ui-monospace,monospace;white-space:pre-wrap;margin-top:10px;line-height:1.45}.run{margin:8px 0;color:#bbb;font-family:ui-monospace,monospace;font-size:13px}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:18px}@media(max-width:800px){.grid{grid-template-columns:1fr}}.legend{display:flex;gap:10px;align-items:center;font-size:12px;color:#aaa;margin-top:8px}.sw{width:80px;height:10px;background:linear-gradient(90deg,#7a1414,#d8a52c,#32b86b)}
</style>
<h2>EMU live capability field</h2>
<div class="run" id="run">waiting for active field…</div>
<div class="bar">
<label>Layer <select id="layer">
<option value="node_competence">command competence</option>
<option value="node_lower">conservative competence</option>
<option value="node_confidence">confidence</option>
<option value="node_quality">command quality</option>
<option value="node_learning">learning / forgetting</option>
<option value="node_staleness">staleness</option>
<option value="node_evidence">effective trials</option>
<option value="recovery_frontier">integrated recovery frontier</option>
<option value="edge_coverage">maneuver coverage</option>
<option value="edge_competence">maneuver competence</option>
<option value="edge_rate">maneuver rate</option>
</select></label>
<label>Yaw slice <input id="yaw" type="range" min="0" max="24" value="12"><span id="yawLabel"></span></label>
<label><input id="auto" type="checkbox" checked> auto refresh</label>
</div>
<div class="legend"><span>low</span><span class="sw"></span><span>high</span><span>dark = unknown</span></div>
<div class="grid"><div><canvas id="c" width="840" height="600"></canvas></div><div><div class="stats" id="picked">click a voxel for details</div></div></div>
<div class="stats" id="stats"></div>
<div class="muted">x = vx, y = vy, slider = yaw rate. Held-out evaluator outcomes are deliberately not painted back into this training-derived map.</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d'),layer=document.getElementById('layer'),yaw=document.getElementById('yaw'),yl=document.getElementById('yawLabel'),stats=document.getElementById('stats'),picked=document.getElementById('picked'),runEl=document.getElementById('run');let D=null,SEL=null;
function clamp(v,a=0,b=1){return Math.max(a,Math.min(b,v))}
function heat(v,known=true){if(!known)return '#070707';v=clamp(v);let r,g,b;if(v<.5){r=170+Math.round(120*v);g=35+Math.round(330*v);b=35}else{r=290-Math.round(420*(v-.5));g=200+Math.round(80*(v-.5));b=55+Math.round(60*(v-.5))}return `rgb(${clamp(r,0,255)},${clamp(g,0,255)},${clamp(b,0,255)})`}
function diverge(v){v=clamp((v+1)/2);return heat(v,true)}
function at(a,ix,iy,iz){const ny=D.vy.length,nz=D.yaw.length;return a[(ix*ny+iy)*nz+iz]}
function draw(){if(!D||D.waiting)return;const zi=clamp(+yaw.value,0,D.yaw.length-1),nx=D.vx.length,ny=D.vy.length,w=c.width/nx,h=c.height/ny,L=layer.value;yl.textContent=` ${D.yaw[zi].toFixed(2)} rad/s`;ctx.clearRect(0,0,c.width,c.height);let max=Math.max(1,D.max_node_evidence||1);if(L==='edge_rate')max=Math.max(.01,D.max_edge_rate||.01);for(let ix=0;ix<nx;ix++)for(let iy=0;iy<ny;iy++){let v=0,known=true,arr=null;if(L==='node_competence'){v=at(D.node_competence,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>=D.min_evidence}else if(L==='node_lower'){v=at(D.node_lower,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>=D.min_evidence}else if(L==='node_confidence'){v=at(D.node_confidence,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_quality'){v=at(D.node_quality,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_learning'){v=at(D.node_learning,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_staleness'){v=at(D.node_staleness,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_evidence'){v=Math.log1p(at(D.node_evidence,ix,iy,zi))/Math.log1p(max);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='recovery_frontier'){v=at(D.recovery_frontier,ix,iy,zi);known=at(D.recovery_known,ix,iy,zi)>0}else if(L==='edge_coverage'){v=at(D.edge_coverage,ix,iy,zi)/26;known=at(D.edge_coverage,ix,iy,zi)>0}else if(L==='edge_competence'){v=at(D.edge_competence_node,ix,iy,zi);known=at(D.edge_coverage,ix,iy,zi)>0}else{v=at(D.edge_rate_node,ix,iy,zi)/max;known=at(D.edge_coverage,ix,iy,zi)>0}ctx.fillStyle=(L==='node_learning'&&known)?diverge(clamp(v,-1,1)):heat(v,known);ctx.fillRect(ix*w,c.height-(iy+1)*h,Math.ceil(w),Math.ceil(h));if(SEL&&SEL.ix===ix&&SEL.iy===iy&&SEL.iz===zi){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(ix*w+1,c.height-(iy+1)*h+1,Math.max(1,w-2),Math.max(1,h-2));}}ctx.fillStyle='#ddd';ctx.font='13px system-ui';ctx.fillText(`vx ${D.vx[0].toFixed(2)} → ${D.vx[nx-1].toFixed(2)} m/s`,10,18);ctx.save();ctx.translate(15,c.height-10);ctx.rotate(-Math.PI/2);ctx.fillText(`vy ${D.vy[0].toFixed(2)} → ${D.vy[ny-1].toFixed(2)} m/s`,0,0);ctx.restore();runEl.textContent=`active: ${D.run_name}   field schema ${D.field_schema}`;renderPicked();stats.textContent=`updated: ${D.mtime_text}\nknown/mastered/frontier nodes: ${D.known_nodes} / ${D.competent_nodes} / ${D.frontier_nodes}\nknown/mastered recovery cells: ${D.known_recovery} / ${D.competent_recovery}\nknown/mastered maneuver edges: ${D.known_edges} / ${D.competent_edges}\nevents node/recovery/edge: ${D.node_events} / ${D.recovery_events} / ${D.edge_events}`}
function renderPicked(){if(!D||!SEL){picked.textContent='click a voxel for details';return}const {ix,iy,iz}=SEL;const e=at(D.node_evidence,ix,iy,iz);picked.textContent=`vx    ${D.vx[ix].toFixed(3)} m/s\nvy    ${D.vy[iy].toFixed(3)} m/s\nyaw   ${D.yaw[iz].toFixed(3)} rad/s\n\ncompetence   ${(100*at(D.node_competence,ix,iy,iz)).toFixed(1)}%\nconservative ${(100*at(D.node_lower,ix,iy,iz)).toFixed(1)}%\nconfidence   ${(100*at(D.node_confidence,ix,iy,iz)).toFixed(1)}%\nquality      ${(100*at(D.node_quality,ix,iy,iz)).toFixed(1)}%\neff trials   ${e.toFixed(1)}\ntrend        ${(100*at(D.node_learning,ix,iy,iz)).toFixed(1)} pp\nstaleness    ${(100*at(D.node_staleness,ix,iy,iz)).toFixed(1)}%\n\nrecovery frontier ${(100*at(D.recovery_frontier,ix,iy,iz)).toFixed(0)}% severity\nmaneuver dirs     ${at(D.edge_coverage,ix,iy,iz).toFixed(0)} / 26\nmaneuver competence ${(100*at(D.edge_competence_node,ix,iy,iz)).toFixed(1)}%\nmaneuver rate     ${at(D.edge_rate_node,ix,iy,iz).toFixed(2)} norm/s`}
c.addEventListener('click',ev=>{if(!D||D.waiting)return;const r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*c.width/r.width,y=(ev.clientY-r.top)*c.height/r.height;SEL={ix:clamp(Math.floor(x/(c.width/D.vx.length)),0,D.vx.length-1),iy:clamp(D.vy.length-1-Math.floor(y/(c.height/D.vy.length)),0,D.vy.length-1),iz:+yaw.value};draw()});
async function load(){try{const r=await fetch('/data',{cache:'no-store'}),ct=r.headers.get('content-type')||'';if(!r.ok){throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0,160)}`)}if(!ct.includes('application/json'))throw new Error(`expected JSON, got ${ct}`);const d=await r.json();if(d.waiting){D=d;runEl.textContent=d.message||'waiting…';stats.textContent=`searching: ${d.search_root||''}`;return}D=d;yaw.max=D.yaw.length-1;if(+yaw.value>D.yaw.length-1)yaw.value=Math.floor(D.yaw.length/2);if(SEL)SEL.iz=+yaw.value;draw()}catch(e){stats.textContent='capability viewer error: '+e.message}}
layer.addEventListener('change',draw);yaw.addEventListener('input',()=>{if(SEL)SEL.iz=+yaw.value;draw()});load();setInterval(()=>{if(document.getElementById('auto').checked)load()},2000);
</script>'''


def payload(path: Path):
    last = None
    for _ in range(3):
        try:
            with np.load(path) as d:
                vx, vy, yaw = [np.asarray(d[k], float) for k in ('vx', 'vy', 'yaw')]
                nc = np.asarray(d['node_competence'], float)
                ne = np.asarray(d['node_evidence'], float)
                schema = int(np.asarray(d['field_schema']).ravel()[0])
                if schema != FIELD_VERSION:
                    raise ValueError(f'field schema {schema} is not current schema {FIELD_VERSION}')
                nl = np.asarray(d['node_lower'], float)
                nq = np.asarray(d['node_quality'], float)
                conf = np.asarray(d['node_confidence'], float)
                slow = np.asarray(d['node_slow'], float)
                step = int(np.asarray(d['step']).ravel()[0])
                half = float(np.asarray(d['half_life_steps']).ravel()[0])
                last_node = np.asarray(d['node_last_step'], float)
                rc = np.asarray(d['recovery_competence'], float)
                re = np.asarray(d['recovery_evidence'], float)
                rl = np.asarray(d['recovery_lower'], float)
                ec = np.asarray(d['edge_competence'], float)
                ee = np.asarray(d['edge_evidence'], float)
                el = np.asarray(d['edge_lower'], float)
                er = np.asarray(d['edge_rate'], float)
            break
        except Exception as exc:
            last = exc
            time.sleep(0.04)
    else:
        raise last

    min_evidence = 3.0
    strong = 8.0
    node_master = (ne >= strong) & (nl >= .80)
    frontier = (ne >= min_evidence) & (~node_master) & (nc >= .35)
    node_learning = np.clip(nc - slow, -1.0, 1.0)
    node_stale = np.clip(np.maximum(0.0, step-last_node) / max(half, 1.0), 0.0, 2.0) / 2.0

    rec_ok = (re >= strong) & (rl >= .70)
    bins = (np.arange(rc.shape[-1], dtype=float) + .5) / max(1, rc.shape[-1])
    recovery_frontier = np.max(np.where(rec_ok, bins, 0.0), axis=-1)
    recovery_known = np.sum(re >= 2.0, axis=-1)

    edge_known_mask = ee >= 2.0
    edge_master_mask = (ee >= strong) & (el >= .70)
    edge_coverage = np.sum(edge_known_mask, axis=-1)
    edge_comp_node = np.sum(np.where(edge_known_mask, ec, 0.0), axis=-1) / np.maximum(edge_coverage, 1)
    edge_rate_node = np.max(np.where(edge_known_mask, er, 0.0), axis=-1)

    mt = path.stat().st_mtime
    summary_path = path.parent / 'capability_field_summary.json'
    summ = {}
    try:
        if summary_path.exists(): summ = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {
        'waiting': False, 'field_schema': schema,
        'run_name': path.parent.name, 'path': str(path),
        'vx': vx.tolist(), 'vy': vy.tolist(), 'yaw': yaw.tolist(),
        'node_competence': nc.ravel().tolist(), 'node_lower': nl.ravel().tolist(),
        'node_evidence': ne.ravel().tolist(), 'node_quality': nq.ravel().tolist(),
        'node_confidence': conf.ravel().tolist(), 'node_learning': node_learning.ravel().tolist(),
        'node_staleness': node_stale.ravel().tolist(),
        'recovery_frontier': recovery_frontier.ravel().tolist(), 'recovery_known': recovery_known.ravel().tolist(),
        'edge_coverage': edge_coverage.ravel().tolist(), 'edge_competence_node': edge_comp_node.ravel().tolist(),
        'edge_rate_node': edge_rate_node.ravel().tolist(),
        'max_node_evidence': float(np.max(ne)) if ne.size else 1.0,
        'max_edge_rate': float(np.max(edge_rate_node)) if edge_rate_node.size else 0.0,
        'min_evidence': min_evidence,
        'known_nodes': int(np.sum(ne >= min_evidence)), 'competent_nodes': int(np.sum(node_master)),
        'frontier_nodes': int(np.sum(frontier)),
        'known_recovery': int(np.sum(re >= 2.0)),
        'competent_recovery': int(np.sum(rec_ok)),
        'known_edges': int(np.sum(edge_known_mask)), 'competent_edges': int(np.sum(edge_master_mask)),
        'node_events': int(float(summ.get('field_node_events_total', 0))),
        'recovery_events': int(float(summ.get('field_recovery_events_total', 0))),
        'edge_events': int(float(summ.get('field_edge_events_total', 0))),
        'mtime': mt, 'mtime_text': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=f'EMU {VERSION} live capability-field viewer')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--run-dir', default='', help='watch one exact run directory')
    group.add_argument('--run-root', default='', help='auto-follow newest field under <run-root>')
    ap.add_argument('--seed', type=int, default=733)
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args(argv)

    exact = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    root = Path(args.run_root).expanduser().resolve() if args.run_root else seed_root(args.seed)

    def find_npz():
        if exact is not None:
            p = exact / 'capability_field_latest.npz'
            if not p.exists(): p = exact / 'capability_field_final.npz'
            return p if p.exists() else None
        if not root.exists(): return None
        candidates = [p for pat in ('*/capability_field_latest.npz','*/capability_field_final.npz') for p in root.glob(pat) if p.is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send_bytes(self, status, content_type, body):
            self.send_response(status); self.send_header('Content-Type', content_type); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def send_json(self, obj, status=200): self.send_bytes(status,'application/json; charset=utf-8',json.dumps(obj,separators=(',',':')).encode())
        def do_GET(self):
            req=urlparse(self.path).path
            if req=='/': return self.send_bytes(200,'text/html; charset=utf-8',HTML.encode())
            if req=='/data':
                npz=find_npz()
                if npz is None:
                    where=str(exact if exact is not None else root)
                    return self.send_json({'waiting':True,'message':'waiting for first capability-field snapshot…','search_root':where})
                try: return self.send_json(payload(npz))
                except Exception as exc: return self.send_json({'waiting':True,'message':f'field exists but is temporarily unreadable: {exc}','search_root':str(npz)},503)
            return self.send_json({'error':'not found','path':req},404)

    srv=ThreadingHTTPServer(('127.0.0.1',args.port),H)
    mode=f'exact run {exact}' if exact is not None else f'auto-follow newest field under {root}'
    print(f'EMU {VERSION} capability viewer: http://127.0.0.1:{args.port}/\n{mode}')
    srv.serve_forever()

if __name__=='__main__':
    main()
