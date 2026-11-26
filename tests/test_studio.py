"""Studio tests use temporary Git repositories and a harmless subprocess.

No network, CUDA, training or permanent study changes.
"""
import http.client
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer

from scripts import studio

MANIFEST=Path(studio.__file__).resolve().parents[1]/"launch.json"


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.repo=self.base/"repo"
        self.repo.mkdir()
        self.git("init","-q")
        self.git("config","user.name","Studio Test")
        self.git("config","user.email","studio@example.invalid")
        self.git("remote","add","origin","https://github.com/mdt733/emu.git")
        self.m=studio.manifest(MANIFEST.read_text())
        self.m["module"]="scripts.dummy"
        (self.repo/"scripts").mkdir()
        (self.repo/"scripts/dummy.py").write_text(
            "import pathlib,sys,time\n"
            "root=pathlib.Path(sys.argv[sys.argv.index('--root')+1])\n"
            "root.mkdir(parents=True,exist_ok=True)\n"
            "print('pinned dummy start',flush=True)\n"
            "time.sleep(.6)\n"
            "(root/'finished.txt').write_text('ok')\n")
        (self.repo/"launch.json").write_text(json.dumps(self.m))
        self.git("add",".")
        self.git("commit","-qm","first")
        self.sha=self.git("rev-parse","HEAD")
        self.s=studio.Studio(self.repo,self.base/"state",sys.executable)
        self.s.study_base=self.base/"studies"
        (self.s.study_base/self.m["parent_study"]).mkdir(parents=True)
        self.s.selected=self.s.read_commit(self.sha)
        self.addCleanup(self.s.close)

    def git(self,*a):
        return subprocess.check_output(["git","-C",str(self.repo),*a],
                                       text=True).strip()

    def finish(self):
        self.s.proc.wait(timeout=10)
        d=self.s.snapshot()
        self.assertFalse(d["active"])
        self.assertEqual(d["job"]["returncode"],0)
        return d

    def test_manifest_rejects_unrecognised_schema_module_path_and_tokens(self):
        for change in (
            {"schema":"unknown"},{"module":"os"},{"id":"../escape"},
            {"parent_study":"a/b"},{"result":"../unsafe.zip"},
            {"args":["--root","{unknown}","{parent_root}"]},
            {"students":["../outside"]},{"worlds":True}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    studio.manifest(json.dumps({**self.m,**change}))

    def test_pinned_launch_resume_and_clean_original(self):
        self.s.launch()
        self.assertTrue(self.s.occupied())
        with self.assertRaises(RuntimeError): self.s.launch()
        d=self.finish()
        self.assertIn("pinned dummy start",d["log"])
        self.assertEqual(d["job"]["commit"],self.sha)
        self.assertEqual(self.git("rev-parse","HEAD"),self.sha)
        self.assertEqual(self.git("status","--porcelain"),"")
        root=Path(d["job"]["root"])
        self.assertTrue((root/"finished.txt").is_file())
        # New selection does not mutate or overwrite the previous source.
        (self.repo/"launch.json").write_text(json.dumps({**self.m,"title":"New"}))
        self.git("add",".");self.git("commit","-qm","second")
        second=self.git("rev-parse","HEAD")
        self.s.selected=self.s.read_commit(second)
        self.assertNotEqual(self.s.selected["root"],str(root))
        self.s.launch(resume=True)
        d=self.finish()
        self.assertEqual(d["job"]["commit"],self.sha)
        self.assertEqual(d["selected"]["commit"],second)
        self.assertEqual(d["job"]["root"],str(root))
        self.assertEqual(self.git("rev-parse","HEAD"),second)

    def test_loopback_token_and_origin(self):
        server=ThreadingHTTPServer(("127.0.0.1",0),studio.handler(self.s,0))
        port=server.server_port
        server.RequestHandlerClass=studio.handler(self.s,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        def get(method,path,headers):
            c=http.client.HTTPConnection("127.0.0.1",port,timeout=5)
            try:
                c.request(method,path,headers=headers)
                r=c.getresponse();return r.status,r.read()
            finally: c.close()
        try:
            self.assertEqual(get("GET","/api/status",{})[0],403)
            h={"X-EMU-Token":self.s.token}
            self.assertEqual(get("GET","/api/status",h)[0],200)
            self.assertEqual(get("GET","/",{"Host":"evil.example"})[0],403)
            self.assertEqual(get("POST","/api/run",{**h,"Origin":"http://evil.example"})[0],403)
            local={**h,"Origin":f"http://127.0.0.1:{port}"}
            self.assertEqual(get("POST","/api/unknown",local)[0],404)
            self.assertEqual(get("GET","/",{})[0],200)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=="__main__":
    unittest.main()
