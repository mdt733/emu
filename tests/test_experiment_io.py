"""Current experiment bookkeeping, not an archived tournament's JSON helpers."""
import json

import pytest

from emu.experiment_io import read_json, save_json, sha256_file


def test_atomic_experiment_roundtrip_and_identity(tmp_path):
    record = tmp_path / "campaign" / "scoreboard.json"
    payload = {"version": 2, "winner": "general", "policy_shas": ["abc"], "score": 0.3}
    save_json(record, payload)
    assert read_json(record) == payload
    original_hash = sha256_file(record)
    assert len(original_hash) == 64
    save_json(record, {"winner": "steps", "version": 2, "policy_shas": ["def"], "score": 0.4})
    assert read_json(record)["winner"] == "steps"
    assert sha256_file(record) != original_hash
    assert not (record.parent / (".scoreboard.json.tmp")).exists()


def test_invalid_records_do_not_replace_verified_scoreboards(tmp_path):
    record = tmp_path / "selection.json"
    save_json(record, {"winner": "general"})
    before = record.read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        save_json(record, {"winner": "steps", "score": float("nan")})
    assert record.read_bytes() == before
    assert not (tmp_path / ".selection.json.tmp").exists()
    assert json.loads(before)["winner"] == "general"


def test_missing_json_is_not_misread_as_empty_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "never_written.json")
