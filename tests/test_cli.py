import tempfile
import unittest
from pathlib import Path

import emu_cli


class CliTests(unittest.TestCase):
    def test_checkpoint_priority_beats_mtime(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); c1=root/'rung_runs'/'a'/'checkpoints'; c2=root/'rung_runs'/'b'/'checkpoints'
            c1.mkdir(parents=True); c2.mkdir(parents=True)
            final=c1/'final.pkl'; latest=c2/'latest.pkl'
            final.write_bytes(b'final'); latest.write_bytes(b'latest')
            latest.touch()
            self.assertEqual(emu_cli._find_checkpoint(root), final)

    def test_seed_run_root_isolated(self):
        self.assertEqual(emu_cli._run_root_for_seed(733), emu_cli.DEFAULT_RUN_ROOT)
        self.assertIn('seed1742', str(emu_cli._run_root_for_seed(1742)))

    def test_resume_without_checkpoint_stops(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                emu_cli.main(['resume','--run-root',td,'--steps','20000000'])


if __name__=='__main__':
    unittest.main()
