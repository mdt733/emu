"""Advisory mastery/retention ledger for fixed reference cohorts.

Use nonoverlapping evidence blocks, not frame counts or moving difficulty.
Conservative descriptive screening only: repeated worlds under one
changing actor are not independent training replications.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

MIN_ATTEMPTS = 40
MASTERY_LOWER = .70
REGRESSION_PP = .10


@dataclass(frozen=True)
class Block:
    attempts: int
    passes: int

    def __post_init__(self):
        if (type(self.attempts) is not int or type(self.passes) is not int
                or not 0 <= self.passes <= self.attempts):
            raise ValueError("Invalid integer skill block")

    @property
    def rate(self):
        return self.passes / self.attempts if self.attempts else None

    def wilson(self, z=1.96):
        if not self.attempts:
            return (0.,1.)
        n,p=self.attempts,self.rate
        d=1.+z*z/n
        c=(p+z*z/(2*n))/d
        r=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        return max(0.,c-r),min(1.,c+r)


def assess_reference(blocks):
    """Chronological, disjoint blocks of the SAME fixed task and policy epoch.

    Two consecutive supported blocks must establish mastery. Once acquired,
    compare later blocks with that fixed mastery pair, not a lucky maximum.
    Repeated regression is required for repair advice; no control action.
    """
    anchor=None
    previous=None
    regressions=0
    status="unobserved"
    for block in blocks:
        if block.attempts < MIN_ATTEMPTS:
            previous=None
            regressions=0
            status="insufficient_evidence"
            continue
        if anchor is None:
            status="developing"
            if (previous is not None and previous.wilson()[0]>=MASTERY_LOWER
                    and block.wilson()[0]>=MASTERY_LOWER):
                anchor=Block(previous.attempts+block.attempts,
                             previous.passes+block.passes)
                status="mastered_reference"
            previous=block
            continue
        loss=anchor.rate-block.rate
        separated=block.wilson()[1]<anchor.wilson()[0]
        if loss>=REGRESSION_PP and separated:
            regressions+=1
            status="repair_advised" if regressions>=2 else "regression_watch"
        else:
            regressions=0
            status="retained" if block.wilson()[0]>=MASTERY_LOWER else "maintenance_watch"
        previous=block
    return dict(status=status,
                anchor=None if anchor is None else
                dict(attempts=anchor.attempts,passes=anchor.passes),
                repeated_regression_blocks=regressions)


def summarize_segment(rows):
    """Offline read-only report. Never pool across process/rollback epochs.

    Aggregate each reference task until >=40 attempts, then close a block.
    Incomplete tails remain explicitly pending and cannot cause a decision.
    """
    from .live_skill_reference import REFERENCE_KICKS_MPS
    if not rows:
        return []
    result=[]
    epochs={}
    last=None
    for row in rows:
        if row["schema"]!="emu_live_skill_window_v1":
            raise ValueError("Wrong live evidence schema")
        if (row["reference_kicks_mps"]!=list(REFERENCE_KICKS_MPS) or
                row["reference_vx_mps"]!=.16 or
                row.get("reference_sensor_scale")!=1. or
                row.get("reference_step_command_m")!=.012 or
                row["reference_settle_deadline_s"]!=3. or
                row["reference_settle_frames"]!=25):
            raise ValueError("Reference contract differs")
        if last is not None and row["start_step"]!=last:
            raise ValueError("Noncontiguous or overlapping segment")
        if row["end_step"]<=row["start_step"]:
            raise ValueError("Invalid evidence window")
        last=row["end_step"]
        epochs.setdefault(row["policy_epoch"],[]).append(row)
    for epoch,windows in epochs.items():
        for case,kick in enumerate(REFERENCE_KICKS_MPS):
            blocks=[]
            n=k=0
            for row in windows:
                c=row["cells"].get(f"reference_recovery/settled/{case}",
                                   dict(attempts=0,passes=0))
                block=Block(c["attempts"],c["passes"])
                n+=block.attempts; k+=block.passes
                if n>=MIN_ATTEMPTS:
                    blocks.append(Block(n,k)); n=k=0
            result.append(dict(policy_epoch=epoch,case=case,kick_mps=kick,
                               blocks=[dict(attempts=b.attempts,passes=b.passes) for b in blocks],
                               pending=dict(attempts=n,passes=k),
                               **assess_reference(blocks)))
    return result
