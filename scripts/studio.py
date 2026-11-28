"""Loopback-only EMU Studio. No third-party server dependencies.

The trusted Git branch publishes launch.json. Updates fetch and inspect
one commit; starting executes that exact commit in an isolated worktree.
Never interpret a manifest as shell code or update an active checkout.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import fcntl
import http.server
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from scripts.studio_progress import progress, partial_archive
from scripts.studio_eta import forecast

BRANCH = "research/ks150-live-skill-school-v1"
ORIGINS = {"https://github.com/mdt733/emu", "https://github.com/mdt733/emu.git",
           "git@github.com:mdt733/emu.git", "ssh://git@github.com/mdt733/emu.git"}
SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
MODULE = re.compile(r"scripts\.[a-zA-Z_][a-zA-Z0-9_]*\Z")
FIELDS = {"schema","id","title","description","module","args","parent_study",
          "result","worlds","students","parent_step","target_step","profile"}


def manifest(raw):
    m=json.loads(raw)
    if not isinstance(m,dict) or set(m)!=FIELDS or m["schema"]!="emu.launch.v1":
        raise ValueError("Unsupported launch manifest schema")
    for k in ("id","parent_study"):
        if not isinstance(m[k],str) or not SLUG.fullmatch(m[k]):
            raise ValueError("Invalid manifest study identifier")
    if not isinstance(m["module"],str) or not MODULE.fullmatch(m["module"]):
        raise ValueError("Launch module must be a scripts.* module")
    for k,limit in (("title",120),("description",1000)):
        if not isinstance(m[k],str) or not 0<len(m[k])<=limit:
            raise ValueError("Invalid manifest text")
    if (not isinstance(m["result"],str) or not m["result"].endswith(".zip")
            or not SLUG.fullmatch(m["result"][:-4])):
        raise ValueError("Invalid result filename")
    if (not isinstance(m["args"],list) or len(m["args"])>64 or
            any(not isinstance(a,str) or len(a)>250 or "\x00" in a or
                ("{" in a or "}" in a) and a not in ("{study_root}","{parent_root}")
                for a in m["args"]) or m["args"].count("{study_root}")!=1
                or m["args"].count("{parent_root}")!=1):
        raise ValueError("Invalid launch argument array")
    if (type(m["worlds"]) is not int or not 1<=m["worlds"]<=16384 or
            type(m["parent_step"]) is not int or type(m["target_step"]) is not int
            or not 0<=m["parent_step"]<m["target_step"] or
            m["profile"]!="h40-spring-only"):
        raise ValueError("Invalid launch physics or budget")
    students=m["students"]
    if (not isinstance(students,list) or not 1<=len(students)<=48 or
            any(not isinstance(x,str) or not SLUG.fullmatch(x) for x in students)
            or len(set(students))!=len(students)):
        raise ValueError("Invalid student list")
    return m


def git(repo,*args):
    r=subprocess.run(["git","-C",str(repo),*args],text=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip()[-4000:] or "Git failed")
    return r.stdout.strip()


def atomic_json(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def tail(path,limit=48000):
    try:
        with path.open("rb") as f:
            f.seek(0,2); size=f.tell()
            f.seek(max(0,size-limit))
            s=f.read().decode("utf-8",errors="replace")
            return ("[earlier output omitted]\n" if size>limit else "")+s
    except OSError:
        return ""


def number(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):
        return None


def recent_jsonl(path):
    # Read a bounded tail. Discard a partially read first or last record.
    try:
        with path.open("rb") as f:
            f.seek(0,2); size=f.tell(); start=max(0,size-262144)
            f.seek(start)
            if start: f.readline()
            lines=f.read().splitlines()
        for line in reversed(lines):
            try: return json.loads(line)
            except (ValueError,UnicodeError): continue
    except OSError:
        pass
    return None


def student_data(root,name,parent,target):
    folder=root/"runs"/name
    keys=("step","reward_mean","fall_rate","vx_mean","cmd_vx_mean",
          "train_steps_per_sec","steps_per_sec","elapsed_s","core_competence",
          "academy_height_up_m","academy_height_down_m","rollback_count")
    history=deque(maxlen=96)
    try:
        with (folder/"metrics.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r={k:number(row.get(k)) for k in keys}
                if r["step"] is not None: history.append(r)
    except (OSError,csv.Error,UnicodeError):
        pass
    last=history[-1] if history else {}
    def latest(pattern):
        paths=list((folder/"live_skills").glob(pattern))
        try:
            paths.sort(key=lambda p:p.stat().st_mtime_ns,reverse=True)
        except OSError:
            return None
        for p in paths:
            row=recent_jsonl(p)
            if isinstance(row,dict): return row
        return None
    raw=latest("segment-*.jsonl")
    lesson=None
    if raw:
        cells=raw.get("cells",{})
        lesson={k:raw.get(k) for k in ("start_step","end_step","policy_epoch","world_frames",
              "reference_frames_by_case","practice_frames_by_case",
              "uphill_reference_frames_by_case","uphill_practice_frames_by_case",
              "reference_kicks_mps","uphill_heights_m")}
        lesson["cells"]={k:v for k,v in cells.items() if
                         k.startswith(("reference_recovery/","practice_recovery/",
                                       "reference_uphill/","practice_uphill/",
                                       "reference_uphill_before_approach_frame/"))}
    endpoint=(root/"evidence"/name/"endpoint.json").is_file()
    final=(folder/"checkpoints"/"final.pkl").is_file()
    return dict(name=name,history=list(history),last=last,lesson=lesson,
                recovery_teacher=latest("teacher-segment-*.jsonl"),
                uphill_teacher=latest("uphill-teacher-segment-*.jsonl"),
                final_checkpoint=final,endpoint=endpoint,
                progress=1. if endpoint else max(0.,min(1.,
                    ((last.get("step") or parent)-parent)/(target-parent))))


class Studio:
    def __init__(self,repo,state_dir,python,monitor=False):
        self.monitor=bool(monitor)
        self.repo=repo.resolve()
        self.state=state_dir.expanduser().resolve()
        self.state.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.python=Path(python).absolute()
        if not self.python.is_file():
            raise ValueError("Existing EMU Python environment not found")
        if git(self.repo,"remote","get-url","origin") not in ORIGINS:
            raise ValueError("Origin must be the trusted mdt733/emu repository")
        self.lock=threading.RLock()
        self.busy=None
        self.closing=False
        self.error=""
        self.selected=None
        self.job=None
        self.proc=None
        self.viewer=None
        self.gpu_lock=self.state/"gpu.lock"
        self.study_base=Path.home()/"emu-data"/"studies"
        self.token=secrets.token_urlsafe(32)
        if (self.state/"selection.json").is_file():
            try:
                saved=json.loads((self.state/"selection.json").read_text())
                self.selected=self.read_commit(saved["commit"])
            except Exception as e:
                self.error="Saved selection unavailable: "+str(e)
        if (self.state/"job.json").is_file():
            try: self.job=json.loads((self.state/"job.json").read_text())
            except (ValueError,OSError): pass

    def read_commit(self,commit):
        if not isinstance(commit,str) or not SHA.fullmatch(commit):
            raise ValueError("Invalid Git commit")
        m=manifest(git(self.repo,"show",commit+":launch.json"))
        module=m["module"].replace(".","/")+".py"
        git(self.repo,"cat-file","-e",commit+":"+module)
        root=self.study_base/(m["id"]+"-"+commit[:12])
        return dict(commit=commit,manifest=m,root=str(root),
                    parent=str(self.study_base/m["parent_study"]))

    def refresh(self):
        git(self.repo,"fetch","--no-tags","origin","refs/heads/"+BRANCH)
        commit=git(self.repo,"rev-parse","FETCH_HEAD^{commit}")
        selected=self.read_commit(commit)
        atomic_json(self.state/"selection.json",{"commit":commit})
        with self.lock: self.selected=selected

    def occupied(self):
        # Child inherits the lock, so a launcher crash cannot silently start
        # another GPU campaign while the registered runner still exists.
        with self.gpu_lock.open("a") as f:
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True

    def worktree(self,selected):
        sha=selected["commit"]
        folder=self.state/"checkouts"/sha
        folder.parent.mkdir(exist_ok=True)
        if not folder.exists():
            git(self.repo,"worktree","add","--detach",str(folder),sha)
        if git(folder,"rev-parse","HEAD")!=sha or git(folder,"status","--porcelain"):
            raise RuntimeError("Pinned worktree is not clean or has wrong source")
        if manifest((folder/"launch.json").read_text())!=selected["manifest"]:
            raise RuntimeError("Worktree manifest differs from selected commit")
        return folder

    def launch(self,resume=False):
        with self.lock:
            selected=self.job if resume else self.selected
        if not selected:
            raise RuntimeError("Fetch a launch manifest first")
        # Re-read the immutable Git manifest, even for a persisted resume.
        # A local status file is not a command or path authority.
        selected=self.read_commit(selected["commit"])
        if self.occupied():
            raise RuntimeError("A Studio-managed GPU campaign is already active")
        folder=self.worktree(selected)
        m=selected["manifest"]
        args=[selected["root"] if a=="{study_root}" else
              selected["parent"] if a=="{parent_root}" else a for a in m["args"]]
        if not Path(selected["parent"]).is_dir():
            raise RuntimeError("Registered parent study is missing: "+selected["parent"])
        env=os.environ.copy()
        for k in ("JAX_PLATFORMS","EMU_UNIFIED_ACCEPTANCE","EMU_MISSION_REFRESHER_SHA256"):
            env.pop(k,None)
        env.update(PYTHONPATH=str(folder),PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",
                   EMU_PHYSICAL_PROFILE=m["profile"],XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env.setdefault("JAX_COMPILATION_CACHE_DIR",str(Path.home()/".cache/emu/jax-live-school"))
        log=self.state/(m["id"]+"-"+selected["commit"][:12]+".log")
        with self.gpu_lock.open("a") as guard,log.open("ab",buffering=0) as out:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            command=[str(self.python),"-m",m["module"],*args]
            out.write(("\n--- "+time.strftime("%Y-%m-%d %H:%M:%S")+
                       " | "+selected["commit"]+" ---\n").encode())
            with self.lock:
                if self.closing: raise RuntimeError("Studio is closing")
                proc=subprocess.Popen(command,cwd=folder,env=env,stdout=out,
                     stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(guard.fileno(),))
                self.proc=proc
                self.job=dict(**selected,status="running",pid=proc.pid,
                              log=log.name,command=command,returncode=None)
                atomic_json(self.state/"job.json",self.job)

    def open_viewer(self):
        # Separate CPU viewer, never the GPU physics process. Build may
        # require the repository's existing npm toolchain.
        with self.lock:
            selected=self.job or self.selected
            if self.viewer is not None and self.viewer.poll() is None:
                return
        if not selected:
            raise RuntimeError("Fetch a revision before opening the viewer")
        folder=self.worktree(selected)
        root=Path(selected["root"])/"runs"
        args=[str(self.python),"-m","emu.cli","viewer",str(folder/"showcase/champion.emupolicy"),
              "--port","8771","--no-open"]
        args+=["--live","--root",str(root)]
        env={**os.environ,"PYTHONPATH":str(folder),"JAX_PLATFORMS":"cpu",
             "XLA_PYTHON_CLIENT_PREALLOCATE":"false","PYTHONUNBUFFERED":"1"}
        with self.lock:
            if self.closing: raise RuntimeError("Studio is closing")
            with (self.state/"viewer.log").open("wb",buffering=0) as out:
                self.viewer=subprocess.Popen(args,cwd=folder,env=env,stdout=out,
                                stderr=subprocess.STDOUT,start_new_session=True)

    def poll(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.job["returncode"]=self.proc.returncode
            self.job["status"]="finished" if self.proc.returncode==0 else "stopped / failed"
            atomic_json(self.state/"job.json",self.job)
            self.proc=None

    def start_action(self,action):
        with self.lock:
            self.poll()
            if self.busy: raise RuntimeError("Another Studio operation is in progress")
            if self.closing: raise RuntimeError("Studio is closing")
            if action in ("run","update-run","resume") and self.occupied():
                raise RuntimeError("A Studio-managed GPU campaign is already active")
            self.error=""
            self.busy=action
        def work():
            try:
                if action in ("update","update-run"): self.refresh()
                if action in ("run","update-run","resume"): self.launch(resume=action=="resume")
                if action=="viewer": self.open_viewer()
            except Exception as e:
                with self.lock: self.error=str(e)
            finally:
                with self.lock: self.busy=None
        threading.Thread(target=work,daemon=True).start()

    def stop(self):
        with self.lock:
            self.poll()
            if self.proc is None:
                raise RuntimeError("No GPU process owned by this Studio session")
            os.killpg(self.proc.pid,signal.SIGINT)

    def snapshot(self):
        with self.lock:
            self.poll()
            if self.monitor:
                # Observe the original controller's atomic state receipts;
                # never acquire its control lock or mutate its job.
                for attr,name in (('job','job.json'),('selected','selection.json')):
                    try:
                        data=json.loads((self.state/name).read_text())
                        old=getattr(self,attr)
                        if old is None or old['commit']!=data['commit']:
                            selected=self.read_commit(data['commit'])
                        else:
                            selected={k:old[k] for k in ('commit','manifest','root','parent')}
                        if attr=='job':
                            if data.get('root')!=selected['root']:
                                raise ValueError('Stored study root mismatch')
                            expected_log=selected['manifest']['id']+'-'+selected['commit'][:12]+'.log'
                            if data.get('log')!=expected_log:
                                raise ValueError('Stored log identity mismatch')
                            selected.update({k:data.get(k) for k in ('status','pid','log','command','returncode')})
                        setattr(self,attr,selected)
                    except FileNotFoundError:
                        pass
            selected=self.selected
            job=dict(self.job) if self.job else None
            busy,error=self.busy,self.error
            viewer=self.viewer
        active=self.occupied()
        if job and job["status"]=="running" and not active and self.proc is None:
            job["status"]="previous session ended; resume available"
        focus=job or selected
        rows=[]
        result=False
        if focus:
            m=focus["manifest"]; root=Path(focus["root"])
            rows=[student_data(root,s,m["parent_step"],m["target_step"]) for s in m["students"]]
            result=(root/m["result"]).is_file()
        log_path=self.state/job["log"] if job else None
        log=tail(log_path) if log_path else ""
        try: age=max(0,time.time()-log_path.stat().st_mtime) if log_path else None
        except OSError: age=None
        status=progress(focus,rows,log,active,age) if focus else None
        if status:
            status["eta"]=forecast(focus,rows,status,log,self.state)
            parent,target=m["parent_step"],m["target_step"]
            for row in rows:
                row["progress"]=(status["student_steps"][row["name"]]-parent)/(target-parent)
        return dict(selected=selected,job=job,busy=busy,error=error,active=active,
                    monitor=self.monitor,progress=status,partial_ready=bool(job),
                    can_stop=not self.monitor and self.proc is not None and self.proc.poll() is None,
                    students=rows,result_ready=result,
                    log=log,
                    viewer=dict(running=viewer is not None and viewer.poll() is None,
                                log=tail(self.state/"viewer.log",6000)),
                    branch=BRANCH)

    def close(self):
        # Leave no deliberate detached GPU jobs after normal shutdown.
        with self.lock:
            self.closing=True
            for p in (self.proc,self.viewer):
                if p is not None and p.poll() is None:
                    try: os.killpg(p.pid,signal.SIGINT)
                    except ProcessLookupError: pass


def handler(studio,port):
    allowed={f"127.0.0.1:{port}",f"localhost:{port}"}
    origins={"http://"+h for h in allowed}
    html=(Path(__file__).parent/"studio.html").read_text(encoding="utf-8")
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,status,data,kind="application/json; charset=utf-8",extra=None):
            self.send_response(status)
            self.send_header("Content-Type",kind)
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Referrer-Policy","no-referrer")
            self.send_header("Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-src http://127.0.0.1:8771 http://localhost:8771; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            for k,v in (extra or {}).items(): self.send_header(k,v)
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError): pass
        def json(self,status,obj):
            self.send(status,json.dumps(obj,allow_nan=False).encode())
        def trusted(self): return self.headers.get("Host") in allowed
        def do_GET(self):
            if not self.trusted(): return self.json(403,{"error":"Local host required"})
            path=urlsplit(self.path).path
            if path=="/":
                return self.send(200,html.replace("__STUDIO_TOKEN__",studio.token).encode(),"text/html; charset=utf-8")
            if path=="/api/status":
                if self.headers.get("X-EMU-Token")!=studio.token:
                    return self.json(403,{"error":"Session token required"})
                try: return self.json(200,studio.snapshot())
                except Exception as e: return self.json(500,{"error":str(e)})
            if path=="/partial-results":
                try:
                    status=studio.snapshot()
                    job=status["job"]
                    if not job: return self.json(404,{"error":"No job to export"})
                    archive,size=partial_archive(job,studio.state/job["log"],status["progress"])
                    with archive:
                        self.send_response(200)
                        self.send_header("Content-Type","application/zip")
                        self.send_header("Content-Disposition",'attachment; filename="emu-partial-results.zip"')
                        self.send_header("Content-Length",str(size))
                        self.send_header("Cache-Control","no-store")
                        self.end_headers()
                        while chunk:=archive.read(65536): self.wfile.write(chunk)
                except (BrokenPipeError,ConnectionResetError): pass
                except Exception as e: return self.json(500,{"error":str(e)})
                return
            if path=="/results":
                # No path parameter: only the current job's registered ZIP.
                if studio.monitor: studio.snapshot()
                with studio.lock: focus=studio.job or studio.selected
                if not focus: return self.json(404,{"error":"No study selected"})
                name=focus["manifest"]["result"]; p=Path(focus["root"])/name
                if not p.is_file(): return self.json(404,{"error":"Results not ready"})
                self.send_response(200)
                self.send_header("Content-Type","application/zip")
                self.send_header("Content-Disposition",'attachment; filename="'+name+'"')
                self.send_header("Content-Length",str(p.stat().st_size))
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                try:
                    with p.open("rb") as f:
                        while chunk:=f.read(65536): self.wfile.write(chunk)
                except (OSError,ConnectionResetError): pass
                return
            return self.json(404,{"error":"Not found"})
        def do_POST(self):
            if (not self.trusted() or self.headers.get("Origin") not in origins
                    or self.headers.get("X-EMU-Token")!=studio.token
                    or self.headers.get("Content-Length","0")!="0"):
                return self.json(403,{"error":"Local same-origin session required"})
            if studio.monitor: return self.json(403,{"error":"Read-only monitor; use original Studio for controls"})
            action=urlsplit(self.path).path.removeprefix("/api/")
            try:
                if action=="stop": studio.stop()
                elif action in ("update","run","update-run","resume","viewer"): studio.start_action(action)
                else: return self.json(404,{"error":"Unknown action"})
                return self.json(202,{"accepted":True})
            except Exception as e: return self.json(409,{"error":str(e)})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--port",type=int,default=8770)
    p.add_argument("--state",type=Path,default=Path.home()/".local/share/emu-studio")
    p.add_argument("--python",type=Path,default=Path(sys.executable))
    p.add_argument("--monitor",action="store_true",help="read-only monitor of an existing Studio; no job control")
    a=p.parse_args()
    studio=Studio(a.repo,a.state,a.python,monitor=a.monitor)
    if a.monitor:
        with http.server.ThreadingHTTPServer(("127.0.0.1",a.port),handler(studio,a.port)) as server:
            print(f"EMU Studio read-only monitor: http://127.0.0.1:{a.port}/",flush=True)
            try: server.serve_forever()
            except KeyboardInterrupt: pass
        return
    with (studio.state/"studio.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with http.server.ThreadingHTTPServer(("127.0.0.1",a.port),handler(studio,a.port)) as server:
            print(f"EMU Studio: http://127.0.0.1:{a.port}/",flush=True)
            print("Loopback only. Ctrl-C closes Studio and requests a stop of its child processes.",flush=True)
            try: server.serve_forever()
            except KeyboardInterrupt: pass
            finally: studio.close()


if __name__=="__main__":
    main()
