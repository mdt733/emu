"""One true cross-language test: deployed 360 actor is numerically the Python actor."""
from __future__ import annotations
import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from emu.contracts import (
    ACTION_DIM, OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM, TERRAIN_OBS_FIELDS,
    SENSOR_SETTINGS_DEFAULTS, sensor_observation_contract)
from emu.ppo import PPO_HIDDEN_SIZES, build_obs_norm_mask

ROOT=Path(__file__).resolve().parents[1]


def _float_blob(value):
    array=np.asarray(value,dtype='<f4',order='C')
    return {'dtype':'float32-le','shape':list(array.shape),
            'data_b64':base64.b64encode(array.tobytes()).decode('ascii')}


def test_exported_360_actor_matches_browser_inference():
    """The normalizer and six joint targets must agree across Python and JS."""
    if shutil.which('node') is None:
        pytest.skip('Node.js needed for independent browser policy parity')
    rng=np.random.default_rng(735)
    dims=(TERRAIN_OBS_DIM,*PPO_HIDDEN_SIZES,ACTION_DIM)
    layers=[]
    for n_in,n_out in zip(dims[:-1],dims[1:]):
        w=(rng.standard_normal((n_in,n_out)).astype(np.float32)
           *np.float32(.6/np.sqrt(n_in)))
        b=(rng.standard_normal(n_out).astype(np.float32)*np.float32(.08))
        layers.append((w,b))
    mean=rng.normal(0,.15,TERRAIN_OBS_DIM).astype(np.float32)
    std=rng.uniform(.15,2.,TERRAIN_OBS_DIM).astype(np.float32)
    mask=build_obs_norm_mask(OBS_SCHEMA_TERRAIN)
    contract=sensor_observation_contract(
        dict(SENSOR_SETTINGS_DEFAULTS),schema=OBS_SCHEMA_TERRAIN,array_only=True)
    bundle={'format':'emu-policy','schema':3,'robot_xml':'<mujoco/>',
      'source':{'checkpoint_schema':9},
      'policy':{'obs_dim':TERRAIN_OBS_DIM,'base_obs_dim':90,
                'history_len':4,'action_dim':ACTION_DIM,
                'hidden_sizes':list(PPO_HIDDEN_SIZES),
                'layers':[{'w':_float_blob(w),'b':_float_blob(b)}
                          for w,b in layers]},
      'observation':{'base_fields':list(TERRAIN_OBS_FIELDS),
                     'sensor_contract':contract,'mean':_float_blob(mean),
                     'std':_float_blob(std),'normalize_mask':_float_blob(mask)},
      'control':{}}

    # Public-release approval must independently verify the new physical
    # 16-ray geometry and ST's two-axis zone inversion, not just a hash.
    # This is inside the existing delivery test; no new permanent test.
    from scripts.verify_showcase_release import validate_browser_bundle
    assert validate_browser_bundle(bundle,require_trained=False)['obs_dim']==360
    tampered=json.loads(json.dumps(bundle))
    zone=tampered['observation']['sensor_contract']
    zone['tof4x4_hardware_zone_ids'][0]=12
    unsigned={k:v for k,v in zone.items() if k!='sha256'}
    zone['sha256']=hashlib.sha256(json.dumps(
        unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError,match='zone order'):
        validate_browser_bundle(tampered,require_trained=False)
    observations=[np.zeros(TERRAIN_OBS_DIM,dtype=np.float32),
        rng.normal(0,1,TERRAIN_OBS_DIM).astype(np.float32),
        np.where(np.arange(TERRAIN_OBS_DIM)%2==0,50.,-50.).astype(np.float32)]
    javascript="""
import fs from 'node:fs';
import {PolicyRunner,parsePolicy} from './emu/viewer_src/src/policy.js';
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const runner=new PolicyRunner(parsePolicy(JSON.stringify(data.bundle)));
process.stdout.write(JSON.stringify(data.observations.map(
    obs=>Array.from(runner.infer(Float32Array.from(obs))))));
"""
    process=subprocess.run(['node','--input-type=module','--eval',javascript],
        cwd=ROOT,input=json.dumps({'bundle':bundle,
             'observations':[x.tolist() for x in observations]}),
        text=True,capture_output=True,check=True,timeout=30)
    browser=np.asarray(json.loads(process.stdout),dtype=np.float32)
    assert browser.shape==(len(observations),ACTION_DIM)
    for obs,actual in zip(observations,browser):
        x=np.where(mask>.5,np.clip((obs-mean)/np.maximum(std,1e-8),-10,10),obs)
        for w,b in layers[:-1]:
            x=np.tanh(x@w+b).astype(np.float32)
        w,b=layers[-1]
        expected=np.clip(x@w+b,-1.,1.)
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=3e-5)
