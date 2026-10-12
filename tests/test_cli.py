import tempfile
import unittest
from pathlib import Path
from emu import cli as emu_cli

class CliTests(unittest.TestCase):
    def test_checkpoint_priority_beats_mtime(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); a=root/'graph'/'checkpoints'; b=root/'shadow'/'checkpoints'
            a.mkdir(parents=True); b.mkdir(parents=True)
            final=a/'final.pkl'; latest=b/'latest.pkl'
            final.write_bytes(b'final'); latest.write_bytes(b'latest'); latest.touch()
            self.assertEqual(emu_cli.find_evaluation(root),final)

    def test_default_root_is_versioned_and_seeded(self):
        self.assertIn(emu_cli.VERSION,str(emu_cli.seed_root(733)))
        self.assertIn('seed733',str(emu_cli.seed_root(733)))
        self.assertNotEqual(emu_cli.seed_root(733),emu_cli.seed_root(1742))

    def test_public_commands_are_small(self):
        # Parsing --help exits successfully; the public surface should remain intentionally tiny.
        with self.assertRaises(SystemExit) as ctx:
            emu_cli.main(['--help'])
        self.assertEqual(ctx.exception.code,0)

if __name__=='__main__': unittest.main()
