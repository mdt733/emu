import unittest
from capability_cloud import make_common_recovery_panel, FAMILIES

class CommonRecoveryPanelTests(unittest.TestCase):
    def test_matched_bundles(self):
        rows=make_common_recovery_panel(50)
        self.assertEqual(len(rows),50)
        bundles={}
        for r in rows: bundles.setdefault(r['bundle_id'],[]).append(r)
        self.assertEqual(len(bundles),10)
        for group in bundles.values():
            self.assertEqual({r['family'] for r in group},set(FAMILIES))
            cmds={(r['cmd_vx'],r['cmd_vy'],r['cmd_yaw']) for r in group}
            self.assertEqual(len(cmds),1)

if __name__=='__main__': unittest.main()
