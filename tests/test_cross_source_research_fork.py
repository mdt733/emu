"""Strict, opt-in source-to-source research continuation contract.

No GPU / MuJoCo needed for this test: independent of full 2048 CUDA smoke.
"""
import hashlib
from pathlib import Path

import pytest

from emu.run import _validate_cross_source_research_parent, inspect_run
from emu.training_config import TrainingConfig

OLD = "a7be11b587f0912e642e100b22ec6d617f548f5b"
NEW = "93cb5cf5a541416ab4366140b2205aca6bfb89a7"


def _sample(tmp_path):
    path=tmp_path/"B_halfsecond25"/"checkpoints"/"final.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"known immutable parent checkpoint bytes")
    sig=hashlib.sha256(path.read_bytes()).hexdigest()
    payload={"meta":{"source_provenance":{"git_commit":OLD}}}
    prov={"commit":NEW,"available":True,"dirty":False}
    return path,sig,payload,prov


def test_explicit_research_fork_requires_exact_checkpoint_digest(tmp_path):
    parent,digest,payload,prov=_sample(tmp_path)
    _validate_cross_source_research_parent(
        parent,payload=payload,expected_source=OLD,
        expected_sha256=digest,provenance=prov)
    with pytest.raises(RuntimeError,match="checkpoint SHA256 changed"):
        _validate_cross_source_research_parent(
            parent,payload=payload,expected_source=OLD,
            expected_sha256="0"*64,provenance=prov)
    with pytest.raises(RuntimeError,match="completed final.pkl"):
        _validate_cross_source_research_parent(
            parent,payload=payload,expected_source=NEW,
            expected_sha256=digest,provenance=prov)


def test_cross_source_research_refuses_nonfinal_and_incomplete_sha(tmp_path):
    parent,digest,payload,prov=_sample(tmp_path)
    with pytest.raises(RuntimeError,match="completed final.pkl"):
        _validate_cross_source_research_parent(
            parent.with_name("snapshot.pkl"),payload=payload,
            expected_source=OLD,expected_sha256=digest,provenance=prov)
    with pytest.raises(RuntimeError,match="full lower-case"):
        _validate_cross_source_research_parent(
            parent,payload=payload,expected_source=OLD[:8],
            expected_sha256=digest,provenance=prov)


def test_research_resume_requires_both_hashes_and_explicit_parent(tmp_path):
    config=TrainingConfig(run_dir=str(tmp_path/"fresh"))
    prov={"commit":NEW,"available":True,"dirty":False}
    with pytest.raises(RuntimeError,match="BOTH"):
        inspect_run(config,provenance=prov,resume="/outside/checkpoints/final.pkl",
                    research_parent_source=OLD)
    with pytest.raises(RuntimeError,match="explicit external"):
        inspect_run(config,provenance=prov,resume="auto",
                    research_parent_source=OLD,research_parent_sha256="a"*64)
    # Without explicit fork flags, the historical source guard remains closed.
    assert inspect_run(config,provenance=prov,resume="") == (None,False)
