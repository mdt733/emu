"""Prevent silent drift between flagship launcher and trainer authorisation.

CPU-only: import/inspect the real pinned command and launch environment,
then exercise training.unified_launch_mode with the same settings.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

from train_pinnacle import (training_environment, command, manifest,
                           DEFAULT_ROOT, REPO, TARGET_STEPS, ENVIRONMENTS,
                           FOUNDATION, SEED)
from emu.training import unified_launch_mode
from emu.contracts import OBS_SCHEMA_TERRAIN


def test_cold_start_launcher_authorises_real_training_gate(monkeypatch, tmp_path):
    """The broken v5032 attempt used EMU_UNIFIED_CONTINUATION, ignored by trainer."""
    monkeypatch.delenv("EMU_UNIFIED_CONTINUOUS",raising=False)
    env=training_environment()
    assert env["EMU_UNIFIED_CONTINUOUS"]=="cold-growth-approved"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"]=="false"
    assert env["PYTHONPATH"].split(os.pathsep)[0]==str(REPO)
    monkeypatch.setenv("EMU_UNIFIED_CONTINUOUS",env["EMU_UNIFIED_CONTINUOUS"])
    cmd=command(tmp_path)
    assert "--resume" not in cmd, "fresh cold start must not inherit a checkpoint"
    args=SimpleNamespace(
        unified_capability_focus=True,device="cuda",
        observation_schema=OBS_SCHEMA_TERRAIN,terrain_step_max_m=.08,
        open_frontier=True,envs=ENVIRONMENTS,steps=TARGET_STEPS,
        foundation_steps=FOUNDATION,foundation_mix="main",seed=SEED,
        resume="",run_dir=str(tmp_path/"pinnacle_cold260"))
    assert unified_launch_mode(args)=="continuous"


def test_plan_is_source_pinned_and_failed_root_is_not_reused():
    assert DEFAULT_ROOT.name=="v5032_pinnacle_s761_v2"
    plan=manifest("a"*40)
    assert plan["source_parent_checkpoint"] is None
    assert plan["release_authorisation"] is False
    assert plan["target_steps"]==TARGET_STEPS
    assert plan["obs_schema"]=="terrain_360"


def test_wrong_env_still_fails_closed(monkeypatch, tmp_path):
    """A CPU PPO smoke must not accidentally approve a long CUDA run."""
    monkeypatch.delenv("EMU_UNIFIED_CONTINUOUS",raising=False)
    monkeypatch.setenv("EMU_UNIFIED_CONTINUATION","pinnacle-coldstart-approved")
    args=SimpleNamespace(
        unified_capability_focus=True,device="cuda",
        observation_schema=OBS_SCHEMA_TERRAIN,terrain_step_max_m=.08,
        open_frontier=True,envs=ENVIRONMENTS,steps=TARGET_STEPS,
        foundation_steps=FOUNDATION,foundation_mix="main",seed=SEED,
        resume="",run_dir=str(tmp_path/"wrong_gate"))
    with pytest.raises(RuntimeError,match="EMU_UNIFIED_CONTINUOUS"):
        unified_launch_mode(args)
