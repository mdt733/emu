"""Bounded birth-practice allocation driven only by fixed reference evidence.

The reference distribution never changes. The separate practice allocation
has a uniform floor and is biased toward repeatedly regressing tasks, or
developing tasks at achievable difficulties. No examination, new physics
worlds, actor inputs or reward edits.
"""
from __future__ import annotations

from .live_skill_ledger import summarize_segment

UNIFORM = (1./6,)*6
TARGETED_FRACTION = .50
MAX_WEIGHT_CHANGE = .05


def practice_plan(rows, previous=UNIFORM):
    if len(previous)!=6 or any(not 0<=x<=1 for x in previous) or abs(sum(previous)-1)>1e-6:
        raise ValueError("Invalid previous practice allocation")
    if not rows:
        return dict(weights=list(UNIFORM), reason="no_reference_evidence")
    epoch=rows[-1]["policy_epoch"]
    current=[r for r in rows if r["policy_epoch"]==epoch]
    info=summarize_segment(current)
    # Repair takes precedence over acquisition, but only after repeated
    # regression on the same fixed physical task.
    repair=[x for x in info if x["status"]=="repair_advised"]
    eligible=repair or [x for x in info if x["status"]=="developing"
              and len(x["blocks"])>=2
              and all(.30<=b["passes"]/b["attempts"]<=.80
                      for b in x["blocks"][-2:])]
    desired=list(UNIFORM)
    reason="uniform_maintenance"
    if eligible:
        desired=[(1.-TARGETED_FRACTION)/6]*6
        for item in eligible:
            desired[item["case"]]+=TARGETED_FRACTION/len(eligible)
        reason="repair" if repair else "acquisition"
    # Convex interpolation keeps a normalized distribution and a uniform
    # maintenance floor; no abrupt redistribution at a noisy window.
    delta=max(abs(x-y) for x,y in zip(desired,previous))
    alpha=min(1.,MAX_WEIGHT_CHANGE/max(delta,1e-12))
    weights=[a+alpha*(b-a) for a,b in zip(previous,desired)]
    return dict(weights=weights, reason=reason,
                selected=[x["case"] for x in eligible],
                statuses=[x["status"] for x in info])
