"""Source foundation choice: guard long GPU continuations against bad parents."""
import csv
from pathlib import Path

import pytest

from emu.foundation_readiness import (
    choose_foundation, reject_obviously_collapsed, recent_fixed_exam,
)


def _record(path, score=.5, *, step=20_054_016, envs=4096):
    return {
        "path":str(path.resolve()),"envs":envs,"step":step,
        "sha256":"a"*64,
        "source_evaluation":dict(eval_broad_score=score,
                                 eval_core_survival=.8,eval_stress30_survival=.7),
    }


def test_prefer_mature_validated_best_policy_over_arbitrary_final(tmp_path):
    directory=tmp_path/"foundation20"
    ck=directory/"checkpoints"
    ck.mkdir(parents=True)
    (ck/"best.pkl").write_bytes(b"best")
    (ck/"final.pkl").write_bytes(b"final")
    examined=[]
    def inspect(path):
        examined.append(path.name)
        return _record(path,score=.6 if path.name=="best.pkl" else -.5)
    selected=choose_foundation([directory],inspect=inspect)
    assert selected["path"].endswith("/best.pkl")
    assert examined==["best.pkl"]


def test_invalid_early_best_falls_back_to_mature_final(tmp_path):
    directory=tmp_path/"foundation20"
    ck=directory/"checkpoints"
    ck.mkdir(parents=True)
    for name in ("best.pkl","final.pkl"):
        (ck/name).write_bytes(name.encode())
    def inspect(path):
        if path.name=="best.pkl":
            raise ValueError("checkpoint at 9M steps is too early")
        return _record(path)
    selected=choose_foundation([directory],inspect=inspect)
    assert selected["path"].endswith("/final.pkl")


def test_prefer_known_root_not_mixed_experiments(tmp_path):
    a,b=(tmp_path/p for p in ("primary","other"))
    for dir in (a,b):
        (dir/"checkpoints").mkdir(parents=True)
        (dir/"checkpoints"/"best.pkl").write_bytes(b"saved")
    def inspect(path):
        if path.parent.parent==a:
            raise ValueError("incompatible 312-input sensor fingerprint")
        return _record(path)
    with pytest.raises(RuntimeError,match="sensor fingerprint"):
        choose_foundation([a,b],inspect=inspect)


def test_no_root_tries_only_the_explicit_ordered_fallback(tmp_path):
    missing=tmp_path/"no_fixed_foundation"
    original=tmp_path/"original"/"foundation20"
    (original/"checkpoints").mkdir(parents=True)
    (original/"checkpoints"/"final.pkl").write_bytes(b"final")
    selected=choose_foundation([missing,original],inspect=lambda p:_record(p))
    assert selected["path"].endswith("/original/foundation20/checkpoints/final.pkl")


def test_explicit_source_is_respected_and_catastrophic_evaluation_vetoes():
    assert reject_obviously_collapsed(None) is None
    assert reject_obviously_collapsed(dict(eval_core_survival=.8,eval_stress30_survival=.05)) is None
    assert "collapse" in reject_obviously_collapsed(dict(
        eval_core_survival=.05,eval_stress30_survival=.07))
    with pytest.raises(ValueError,match="collapse"):
        choose_foundation([],explicit=Path("/example/user-selection.pkl"),
            inspect=lambda p:_record(p) | {
                "source_evaluation":{"eval_core_survival":.03,
                                     "eval_stress30_survival":.04}})
    selection=choose_foundation([],explicit=Path("/example/user-selection.pkl"),
                                inspect=lambda p:_record(p))
    assert selection["selection_reason"]=="explicit checkpoint"


def test_only_exact_checkpoint_step_provides_foundation_health(tmp_path,monkeypatch):
    from emu import foundation_readiness as f
    directory=tmp_path/"foundation20"
    ck=directory/"checkpoints"
    ck.mkdir(parents=True)
    path=ck/"best.pkl"
    path.write_bytes(b"fake-pickle")
    monkeypatch.setattr(f,"__contract",lambda p:{})
    import emu.checkpoint_io as check
    monkeypatch.setattr(check,"checkpoint_step",lambda *args,**kwargs:20_054_016)
    columns=["step","eval_policy_step","eval_broad_score",
             "eval_core_survival","eval_stress30_survival"]
    with (directory/"fixed_eval.csv").open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=columns)
        writer.writeheader()
        writer.writerow(dict(step="19",eval_policy_step="19",eval_broad_score="9",
                             eval_core_survival="1.0",eval_stress30_survival="1.0"))
        writer.writerow(dict(step="20054016",eval_policy_step="20054016",
                             eval_broad_score=".51",
                             eval_core_survival=".8",eval_stress30_survival=".6"))
    result=recent_fixed_exam(path)
    assert result["eval_broad_score"]==pytest.approx(.51)
    assert result["eval_core_survival"]==pytest.approx(.8)
    monkeypatch.setattr(check,"checkpoint_step",lambda *args,**kwargs:19_000_000)
    assert recent_fixed_exam(path) is None
