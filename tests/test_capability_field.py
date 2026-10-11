import unittest
import numpy as np

from capability_field import CapabilityField, FIELD_VERSION, EDGE_SCALES


def base_metrics(T=96,N=8,cmd=0.10):
    z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32)
    age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+z
    c=z+cmd
    return {'cmd_vx':c.copy(),'cmd_vy':z.copy(),'cmd_yaw':z.copy(),'target_cmd_vx':c.copy(),'target_cmd_vy':z.copy(),'target_cmd_yaw':z.copy(),
            'transition_ramp_speed':z.copy(),'vx':c.copy(),'vy':z.copy(),'yaw_rate':z.copy(),'gravity_z':o.copy(),'support':o*.30,'bad_contact':z.copy(),
            'episode_fell':z.copy(),'episode_done':z.copy(),'age_s':age.copy(),'regime_resilience':z.copy(),'birth_resilience_d':z.copy(),'transition_frac':z.copy(),'cmd_resampled':z.copy()}


class CapabilityFieldTests(unittest.TestCase):
    def test_schema_and_mastered_node(self):
        f=CapabilityField(); d=f.update_from_rollout(base_metrics(),step=65536)
        self.assertEqual(FIELD_VERSION,3)
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

    def test_window_mean_tracking_accepts_gait_oscillation(self):
        m=base_metrics(T=96,N=8,cmd=.10)
        # Only half the frames satisfy the old instantaneous 0.08 m/s gate,
        # but the one-second mean absolute error is 0.06 m/s and should pass.
        err=np.where((np.arange(96)[:,None]%2)==0,.10,.02).astype(np.float32)
        m['vx']=m['cmd_vx']+err
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertGreaterEqual(int(d['field_competent_nodes']),1)

    def test_fast_post_settle_fall_is_failure(self):
        m=base_metrics(T=40,N=4,cmd=.20)
        m['episode_fell'][25]=1; m['episode_done'][25]=1
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertGreaterEqual(int(d['field_node_events_update']),4)
        known=f.node_evidence()>0
        self.assertLess(float(np.mean(f.node_competence()[known])),.5)

    def test_one_frame_jump_registers_edge_and_rate(self):
        T,N=30,8; m=base_metrics(T,N,0.0)
        m['cmd_vx'][5:]=.3; m['target_cmd_vx'][5:]=.3; m['vx'][5:]=.3
        m['transition_frac'][5]=1; m['cmd_resampled'][5]=1; m['transition_ramp_speed'][5]=1.0
        f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
        self.assertEqual(int(d['field_edge_events_update']),N)
        rates=f.edge_rate()[f.edge_evidence()>0]
        self.assertTrue(np.allclose(rates, (0.3/0.9)/0.02, rtol=.03))

    def test_signed_vy_yaw_and_reverse_one_frame_maneuvers(self):
        cases=[(-.30,0.0,0.0),(0.0,.20,0.0),(0.0,-.20,0.0),(0.0,0.0,1.0),(0.0,0.0,-1.0)]
        for target in cases:
            with self.subTest(target=target):
                T,N=30,4; m=base_metrics(T,N,0.0)
                tvx,tvy,tyaw=target
                for key,val in [('cmd_vx',tvx),('target_cmd_vx',tvx),('vx',tvx),
                                ('cmd_vy',tvy),('target_cmd_vy',tvy),('vy',tvy),
                                ('cmd_yaw',tyaw),('target_cmd_yaw',tyaw),('yaw_rate',tyaw)]:
                    m[key][5:]=val
                m['transition_frac'][5]=1; m['cmd_resampled'][5]=1; m['transition_ramp_speed'][5]=1.0
                f=CapabilityField(); d=f.update_from_rollout(m,step=65536)
                self.assertEqual(int(d['field_edge_events_update']),N)
                rates=f.edge_rate()[f.edge_evidence()>0]
                expected=float(np.linalg.norm(np.asarray(target,np.float32)/EDGE_SCALES)/.02)
                self.assertTrue(np.allclose(rates,expected,rtol=.03))

    def test_missing_metric_fails_loudly(self):
        m=base_metrics(); del m['cmd_resampled']
        with self.assertRaises(KeyError): CapabilityField().update_from_rollout(m,step=65536)

    def test_state_shape_mismatch_fails_loudly(self):
        f=CapabilityField(); f.update_from_rollout(base_metrics(),step=65536)
        d=f.state_dict(); d['node_success']=np.zeros((1,),np.float32)
        with self.assertRaises(ValueError): CapabilityField.from_state_dict(d)

    def test_unknown_config_keys_are_ignored(self):
        f=CapabilityField(); d=f.state_dict(); d['cfg']['future_optional_knob']=123
        g=CapabilityField.from_state_dict(d)
        self.assertEqual(g.shape,f.shape)

    def test_read_only_queries_do_not_mutate_field(self):
        f=CapabilityField(); f.update_from_rollout(base_metrics(),step=65536)
        before=f.fingerprint()
        f.predict_command(.1,0.0,0.0)
        f.calibration_commands(4)
        f.summary()
        self.assertEqual(before,f.fingerprint())

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
