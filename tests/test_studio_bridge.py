import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.studio_bridge import companion_command

class BridgeTests(unittest.TestCase):
    def test_direct_and_new_controller_skip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertIsNone(companion_command(root,{}))
            self.assertIsNone(companion_command(root/"checkouts"/("a"*40),
                                                 {"EMU_STUDIO_UI_REV":"eta-v1"}))
    def test_legacy_checkout_command(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            folder=root/"checkouts"/("a"*40)
            folder.mkdir(parents=True)
            (root/"selection.json").touch()
            (root/"gpu.lock").touch()
            with patch("scripts.studio_bridge.subprocess.check_output",
                       return_value=str(root/"repo"/".git")+"\n"):
                cmd=companion_command(folder,{})
            self.assertEqual(cmd[1:3],["-m","scripts.studio"])
            self.assertEqual(cmd[cmd.index("--state")+1],str(root))
            self.assertEqual(cmd[cmd.index("--repo")+1],str(root/"repo"))
            self.assertEqual(cmd[-3:],["--monitor","--port","8772"])

if __name__=="__main__": unittest.main()
