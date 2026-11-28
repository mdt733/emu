"""Pure stdlib checks of Studio progress and bounded partial archive."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.studio_progress import progress, partial_archive


class ProgressTests(unittest.TestCase):
    def test_observed_preflight_training_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); root=p/"study"; root.mkdir()
            log=p/"job.log"
            log.write_text("PRECHECK: both mature lesson families, 12 updates\n"
                           "real batched PPO training update PASS; world_steps=131,072\n")
            names=["U-s9501","R-s9501","H-s9501","H-s9502","R-s9502","U-s9502"]
            parent=120061952; target=125304832
            focus=dict(root=str(root),commit="a"*40,manifest=dict(
                students=names,parent_step=parent,target_step=target,result="results.zip"))
            rows=[dict(name=n,last={},lesson=None,endpoint=False,final_checkpoint=False)
                  for n in names]
            s=progress(focus,rows,log.read_text(),True,7*3600)
            self.assertEqual((s["phase"],s["done"],s["total"],s["lower_bound"],s["training_done"]),
                             ("GPU acceptance",1,12,True,0))
            self.assertEqual(s["training_total"],31457280)
            self.assertEqual(progress(focus,rows,log.read_text()+"STUDIO PREFLIGHT 7/12\n",True)["done"],7)
            self.assertEqual(progress(focus,rows,log.read_text()+"\n--- retry ---\n",True)["phase"],"Preflight setup")
            (root/"acceptance").mkdir()
            (root/"acceptance/combined.json").write_text("{}")
            self.assertEqual(progress(focus,rows,"",True)["phase"],"Parent examinations")
            for n in ("9501","9502"):
                x=root/"evidence"/("parent-s"+n); x.mkdir(parents=True)
                (x/"endpoint.json").write_text("{}")
            rows[0]["last"]={"step":parent+131072}
            r=progress(focus,rows,"STUDENT U-s9501:\nSTUDIO TRAIN U-s9501 "+str(parent+2*131072)+"\n",True)
            self.assertEqual((r["phase"],r["done"],r["training_done"]),
                             ("Student training",262144,262144))
            rows[0]["final_checkpoint"]=True
            self.assertEqual(progress(focus,rows,"STUDENT U-s9501:\n",True)["phase"],"Student examinations")
            student=root/"runs"/names[0]
            (student/"checkpoints").mkdir(parents=True)
            (student/"checkpoints/final.pkl").write_bytes(b"not exported")
            (student/"metrics.csv").write_text("step,reward_mean\n120193024,1.0\n")
            (root/"study.json").write_text("{}")
            (student/"live_skills").mkdir()
            (student/"live_skills/segment-a.jsonl").write_text('{"broken":')
            archive,size=partial_archive(focus,log,s)
            with archive,zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("studio.log",z.namelist())
                self.assertFalse(any(x.endswith(".pkl") for x in z.namelist()))
                self.assertFalse(json.loads(z.read("snapshot.json"))["complete"])
                self.assertEqual(z.read("runs/U-s9501/live_skills/segment-a.jsonl"),b'{"broken":')


if __name__=="__main__":
    unittest.main()
