import pickle
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple, Any

import numpy as np

from capability_field import CapabilityField
from training import save_checkpoint, validate_resume_payload


class FakeState(NamedTuple):
    params: Any
    opt_state: Any


class CheckpointResumeTests(unittest.TestCase):
    def test_exact_resume_roundtrip_contract(self):
        field=CapabilityField()
        state=FakeState(
            params={'actor':np.arange(12,dtype=np.float32).reshape(3,4),
                    'critic':np.asarray([1.25,-.5],np.float32)},
            opt_state=(np.asarray([3.0,4.0],np.float32), {'count':np.asarray(17,np.int32)}),
        )
        norm={'mean':np.asarray([.1,.2,.3],np.float32),
              'var':np.asarray([1.1,.9,1.3],np.float32),'count':1234.0}
        governor={'params':{'steady_difficulty':.25},'marker':[1,2,3],
                  '_capability_field':field.state_dict()}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'roundtrip.pkl'
            save_checkpoint(path,state,{'steady_difficulty':.25},{'step':123456},norm,governor_meta=governor)
            with path.open('rb') as f: loaded=pickle.load(f)
            validate_resume_payload(loaded,path)

        np.testing.assert_array_equal(loaded['state'].params['actor'],state.params['actor'])
        np.testing.assert_array_equal(loaded['state'].params['critic'],state.params['critic'])
        np.testing.assert_array_equal(loaded['state'].opt_state[0],state.opt_state[0])
        self.assertEqual(int(loaded['state'].opt_state[1]['count']),17)
        np.testing.assert_array_equal(loaded['obs_norm']['mean'],norm['mean'])
        np.testing.assert_array_equal(loaded['obs_norm']['var'],norm['var'])
        self.assertEqual(float(loaded['obs_norm']['count']),norm['count'])
        self.assertEqual(loaded['governor_meta']['marker'],governor['marker'])
        restored=CapabilityField.from_state_dict(loaded['governor_meta']['_capability_field'])
        self.assertEqual(restored.fingerprint(),field.fingerprint())

    def test_resume_requires_meta(self):
        with self.assertRaises(ValueError):
            validate_resume_payload({'state':FakeState(params={},opt_state=())})


if __name__=='__main__':
    unittest.main()
