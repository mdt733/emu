"""Pure stdlib tests, no CUDA and no running Studio needed."""
import tempfile
import unittest
from pathlib import Path
from scripts.studio_eta import forecast, recent_rates, read_timings, estimate

class EtaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.names=["A-s1","B-s1"]
        self.focus=dict(root=str(self.root),commit="a"*40,manifest=dict(
            id="test",module="scripts.live_skill_pilot",profile="h40-spring-only",
            worlds=4096,students=self.names,parent_step=1000,target_step=4001000))
        self.rows=[dict(name=n,history=[],last={},endpoint=False,final_checkpoint=False) for n in self.names]
        for p in ("acceptance/combined.json","evidence/parent-s1/endpoint.json"):
            self.touch(p)
        self.status=dict(phase="Student training",active=True,student=self.names[0],
                         student_steps={n:1000 for n in self.names},done=0,total=4000000,log_age_seconds=0)
    def tearDown(self): self.tmp.cleanup()
    def touch(self,p):
        x=self.root/p;x.parent.mkdir(parents=True,exist_ok=True);x.touch()
    def calc(self,log=""):
        return forecast(self.focus,self.rows,self.status,log,self.root,now=2_000_000_000)
    def test_all_future_work_and_compile(self):
        r=self.calc()
        self.assertEqual([c["kind"] for c in r["components"]],
            ["startup","training","pulse","matrix","seal","startup","training","pulse","matrix","seal","package"])
        self.assertGreater(r["remaining_seconds"],2*4000000/39700)
        self.assertLess(r["low_seconds"],r["remaining_seconds"])
        self.assertGreater(r["high_seconds"],r["remaining_seconds"])
        self.assertGreater(r["current_seconds"],4000000/39700)
    def test_live_wall_rate_and_completed_work(self):
        self.rows[0]["history"]=[dict(step=1000+i*400000,elapsed_s=100+i*10) for i in range(5)]
        self.status["student_steps"][self.names[0]]=1601000
        r=self.calc("STUDENT A-s1:\nSTUDIO TRAIN A-s1 1601000\n")
        self.assertEqual(r["rate"],40000)
        self.assertEqual(r["current_seconds"],60)
        self.rows[0]["final_checkpoint"]=True
        self.status["phase"]="Student examinations"
        self.touch("evidence/A-s1/pulse.json")
        r=self.calc()
        self.assertFalse(any(c["kind"]=="pulse" and c["name"]=="A-s1" for c in r["components"]))
        self.assertTrue(any(c["kind"]=="matrix" and c["name"]=="A-s1" for c in r["components"]))
    def test_resume_no_fresh_update(self):
        self.rows[0]["history"]=[dict(step=1000+i*400000,elapsed_s=100+i*10) for i in range(5)]
        self.status["student_steps"][self.names[0]]=1601000
        r=self.calc("STUDENT A-s1:\nSTUDIO TIME 1999999990 train A-s1 start\n")
        self.assertTrue(any(c["kind"]=="startup" and c["current"] for c in r["components"]))
    def test_retry_and_stopped(self):
        samples,starts=read_timings("STUDIO TIME 1 pulse A start\n--- retry ---\nSTUDIO TIME 2 pulse A end\nSTUDIO TIME 3 pulse B start\nSTUDIO TIME 53 pulse B end")
        self.assertEqual(samples,{"pulse":[50.]})
        self.assertEqual(starts,{})
        self.assertEqual(recent_rates([dict(step=1,elapsed_s=10),dict(step=400001,elapsed_s=20),dict(step=400001,elapsed_s=1)]),[])
        self.status["active"]=False
        self.assertIsNone(self.calc())
        self.status["phase"]="Complete"
        self.assertEqual(self.calc()["remaining_seconds"],0)

if __name__=="__main__": unittest.main()
