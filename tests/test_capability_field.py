import unittest
import numpy as np

from capability_field import CapabilityField, FIELD_VERSION


def base_metrics(T=96,N=8,cmd=0.10):
    z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32)
    age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+z
    c=z+cmd
    return {'cmd_vx':c.copy(),'cmd_vy':z.copy(),'cmd_yaw':z.copy(),'target_cmd_vx':c.copy(),'target_cmd_vy':z.copy(),'target_cmd_yaw':z.copy(),
            'transition_ramp_speed':z.copy(),'vx':c.copy(),'vy':z.copy(),'yaw_rate':z.copy(),'gravity_z':o.copy(),'support':o*.30,'bad_contact':z.copy(),
            'episode_fell':z.copy(),'episode_done':z.copy(),'age_s':age.copy(),'regime_resilience':z.copy(),'birth_resilience_d':z.copy(),'transition_frac':z.copy()}


class CapabilityFieldTests(unittest.TestCase):
    def test_schema_and_mastered_node(self):
        f=CapabilityField(); d=f.update_from_rollout(base_metrics(),step=65536)
        self.assertEqual(FIELD_VERSION,2)
        self.assertGreater(d['field_node_events_update'],0)
        self.assertGreaterEqual(d['field_competent_nodes'],1)
        self.assertGreater(d['field_mean_quality_known'],.8)

    def test_bad_tracking_is_failure_not_low_quality_success(self):
        m=base_metrics(); m['vx'][:]=-.5
        f=CapabilityField(); f.update_from_rollout(m,step=65536)
        known=f.node_evidence()>=3
        self.assertTrue(np.any(known))
        self.assertLess(float(np.mean(f.node_competence()[known])),.5)

    def test_one_recovery_event_per_birth(self):
        T,N=40,4; m=base_metrics(T,N,.2); m['regime_resilience'][:]=1; m['birth_resilience_d'][:]=.5
        m['vx'][:10]=0.0
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertEqual(int(d['field_recovery_events_update']),N)

    def test_complete_maneuver_event(self):
        T,N=35,4; m=base_metrics(T,N,0.0)
        cmd=m['cmd_vx']; target=m['target_cmd_vx']; ramp=m['transition_frac']; rs=m['transition_ramp_speed']
        for t in range(T):
            if 5<=t<15:
                ramp[t]=1; cmd[t]=.3*((t-5)/10); target[t]=.3; rs[t]=.1
            elif t>=15:
                cmd[t]=.3; target[t]=.3
        m['vx']=cmd.copy()
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertEqual(int(d['field_edge_events_update']),N)
        self.assertGreater(d['field_mean_edge_rate'],0)


    def test_repeated_recovery_events_can_be_mastered(self):
        T,N=40,8; m=base_metrics(T,N,.2); m['regime_resilience'][:]=1; m['birth_resilience_d'][:]=.5
        m['vx'][:10]=0.0
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertEqual(int(d['field_recovery_events_update']),N)
        self.assertGreaterEqual(int(d['field_competent_recovery_cells']),1)

    def test_repeated_maneuver_events_can_be_mastered(self):
        T,N=35,8; m=base_metrics(T,N,0.0)
        cmd=m['cmd_vx']; target=m['target_cmd_vx']; ramp=m['transition_frac']; rs=m['transition_ramp_speed']
        for t in range(T):
            if 5<=t<15:
                ramp[t]=1; cmd[t]=.3*((t-5)/10); target[t]=.3; rs[t]=.1
            elif t>=15:
                cmd[t]=.3; target[t]=.3
        m['vx']=cmd.copy()
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertEqual(int(d['field_edge_events_update']),N)
        self.assertGreaterEqual(int(d['field_competent_edges']),1)

    def test_roundtrip_and_packet_contract(self):
        f=CapabilityField(); f.update_from_rollout(base_metrics(),step=65536)
        g=CapabilityField.from_state_dict(f.state_dict())
        self.assertEqual(g.summary()['field_node_events_total'],f.summary()['field_node_events_total'])
        p=g.teaching_packet(step=30_000_000,mode='graph',start=20_000_000,full=30_000_000,maximum=.65)
        self.assertEqual(p['field_node_anchors'].shape,(24,5))
        self.assertEqual(p['field_edge_anchors'].shape,(24,7))

    def test_schema1_cannot_silently_resume(self):
        with self.assertRaises(ValueError):
            CapabilityField.from_state_dict({'field_version':1})


if __name__=='__main__': unittest.main()
