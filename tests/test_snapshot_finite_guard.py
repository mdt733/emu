"""Failed snapshots cannot evict a valid rewind point."""
from types import SimpleNamespace
import numpy as np

from emu.checkpoint_io import save_rotating_snapshot


def test_nonfinite_rotating_snapshot_retains_older_file(tmp_path):
    norm = {'mean':np.zeros(220, dtype=np.float32),
            'var':np.ones(220, dtype=np.float32),'count':1.0}
    meta = {'step':12,'lr_scale':1.0}
    args = dict(curriculum_state={'params':{}},health_state={},field_state={})
    ok_state = SimpleNamespace(params={'x':np.array([0.1], dtype=np.float32)},
                               opt_state={'y':np.array([0.2], dtype=np.float32)})
    first = save_rotating_snapshot(tmp_path,ok_state,meta,norm,step=12,keep=1,**args)
    assert first['saved'] == 1.0
    previous = next(tmp_path.glob('snapshot_step_*.pkl'))
    before = previous.read_bytes()
    bad_state = SimpleNamespace(params={'x':np.array([np.nan], dtype=np.float32)},
                                opt_state={'y':np.array([0.2], dtype=np.float32)})
    rejected = save_rotating_snapshot(tmp_path,bad_state,dict(meta,step=24),norm,step=24,keep=1,**args)
    assert rejected['saved'] == 0.0
    assert list(tmp_path.glob('snapshot_step_*.pkl')) == [previous]
    assert previous.read_bytes() == before
