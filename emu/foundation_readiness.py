"""Select a compatible, promising 312-input foundation for the first refork.

A *best* training checkpoint has a frozen-policy evaluation attached;
a *final* checkpoint is merely the most recent learner state. Never treat
the last checkpoint as the best walker, or a valid pickle as competent
locomotion. This is a launch preflight, not another curriculum controller.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Any

from .campaign_refork import _foundation
from .training_config import TrainingConfig


def recent_fixed_exam(checkpoint: Path) -> dict[str, float] | None:
    """Read ONLY the fixed exam that evaluated the exact checkpoint step."""
    from .checkpoint_io import checkpoint_step
    path = Path(checkpoint)
    history = path.parent.parent / "fixed_eval.csv"
    if not history.is_file():
        return None
    target = checkpoint_step(path,expected_observation_schema="rolling_tof_312",
                             expected_sensor_contract=__contract(path))
    with history.open(newline="",encoding="utf-8-sig") as fd:
        matches=[row for row in csv.DictReader(fd)
                 if row.get("eval_policy_step",row.get("step")) not in ("",None)
                 and float(row.get("eval_policy_step") or row.get("step")) == target]
    if not matches:
        return None
    result = {}
    for key in ("eval_broad_score","eval_core_survival",
                "eval_stress30_survival","eval_dynamic50_survival",
                "eval_resilience50_survival"):
        raw=matches[-1].get(key)
        if raw in (None,""):
            continue
        val=float(raw)
        if not math.isfinite(val):
            raise ValueError(f"nonfinite source fixed exam {key}")
        result[key]=val
    return result


def __contract(checkpoint: Path) -> dict:
    from .training_config import training_sensor_contract
    return training_sensor_contract(TrainingConfig.load(
        Path(checkpoint).parent.parent/"training_config.json"))


def source_foundation(checkpoint: Path, *, validate: Callable[...,dict]=_foundation
                      ) -> dict[str,Any]:
    checkpoint=Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config=TrainingConfig.load(checkpoint.parent.parent/"training_config.json")
    evidence=validate(checkpoint,envs=config.envs)
    # _foundation enforces 312-input fingerprint, world count and 12–22M
    # source maturity, then returns exact step+SHA of frozen policy.
    return {"path":str(checkpoint), "envs":config.envs,
            "step":evidence["step"], "sha256":evidence["sha256"],
            "source_evaluation":recent_fixed_exam(checkpoint)}


def reject_obviously_collapsed(evaluation: dict[str,float] | None) -> str | None:
    """Veto proven catastrophic core collapse; never assert a passing walker.

    Low growth/slope/obstacle scores are *not* a veto for an early foundation.
    Core locomotion failures on TWO easy independently evaluated panels are.
    An unavailable exact-step exam is explicitly unknown, not zero.
    """
    if evaluation is None:
        return None
    core=evaluation.get("eval_core_survival")
    stress=evaluation.get("eval_stress30_survival")
    if core is not None and stress is not None and core < .15 and stress < .15:
        return (f"severe low-difficulty collapse: core={core:.3f}, "
                f"stress30={stress:.3f}; do not spend 100M steps on this parent")

    # An actor can stand upright almost indefinitely on the easiest panel
    # while falling nearly immediately whenever given real commands.
    # The old 4096-world v5020 foundation is an observed counterexample:
    # core=.99988, stress30=.00171, dynamic50=.00183,
    # resilience50=.00012. This is NOT a transferable walking foundation.
    dynamic = evaluation.get("eval_dynamic50_survival")
    resilience = evaluation.get("eval_resilience50_survival")
    if (core is not None and core >= .70
            and stress is not None and stress < .03
            and dynamic is not None and dynamic < .03
            and resilience is not None and resilience < .03):
        return (f"narrow, non-transferring foundation: core={core:.3f} but "
                f"stress30={stress:.4f}, dynamic50={dynamic:.4f}, "
                f"resilience50={resilience:.4f}; review earlier snapshots "
                f"or train a new foundation before a long five-way refork")
    return None


def choose_foundation(directories: list[Path], *,
                      explicit: Path | None=None,
                      inspect: Callable[[Path],dict[str,Any]]=source_foundation
                      ) -> dict[str,Any]:
    """Prefer mature, individually verified best.pkl, then final.pkl.

    Never rewrite checkpoints or attempt to synthesise a foundation from
    a 232-input public demo. Explicit user choices are not substituted.
    If the first run directory is available, don't quietly mix checkpoints
    from another previous experiment.
    """
    if explicit is not None:
        result=inspect(Path(explicit))
        issue=reject_obviously_collapsed(result.get("source_evaluation"))
        if issue:
            raise ValueError(issue)
        return {**result,"selection_reason":"explicit checkpoint"}

    findings=[]
    for directory in directories:
        directory=Path(directory).expanduser()
        ckdir=directory/"checkpoints"
        if not ckdir.is_dir():
            continue
        for filename in ("best.pkl","final.pkl"):
            path=ckdir/filename
            if not path.is_file():
                continue
            try:
                record=inspect(path)
                # Automatic overnight selection needs evidence for the
                # selected EXACT policy step. A mature saved network with
                # no matching frozen evaluation has UNKNOWN competence,
                # and may only be chosen via an explicit user path.
                if record.get("source_evaluation") is None:
                    findings.append(f"{path}: no exact-checkpoint fixed exam; cannot auto-select")
                    continue
                issue=reject_obviously_collapsed(record.get("source_evaluation"))
                if issue:
                    findings.append(f"{path}: {issue}")
                    continue
                return {**record,"selection_reason":f"validated {filename} from preferred foundation root"}
            except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
                findings.append(f"{path}: {exc}")
        # Never cross over into another experiment when the first preferred
        # root contains candidates but they fail validation.
        if list(ckdir.glob("*.pkl")):
            break
    why="; ".join(findings) if findings else "no checkpoint files found in the expected foundation roots"
    raise RuntimeError(f"No suitable 312-input 12–22M source foundation: {why}")
