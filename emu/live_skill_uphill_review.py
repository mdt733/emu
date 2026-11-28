"""Read-only diagnosis from recent fixed-height uphill reference blocks.

Use identical birth denominators, never shaped practice outcomes. This
advisory picks a testable prerequisite treatment, not an automatic
curriculum promotion. Each supplied block must be a disjoint chronological
aggregation of the same task under the same reference contract.
"""
from .live_skill_uphill import UP_HEIGHTS_M

GATES=("first_support","each_foot_five","traversed","mission")
MIN_ATTEMPTS=40
MIN_GAP=.15


def diagnose_uphill(blocks):
    """Last two nonoverlapping blocks, each {case: gate: {attempts,passes}}.

    A reference first-support deficit may be an approach problem; inspect
    approach and contact separately before authorizing extra credit.
    """
    if len(blocks)<2:
        return dict(status="insufficient_evidence",candidates=[])
    candidates=[]
    for case,height in enumerate(UP_HEIGHTS_M):
        counts=[]
        for block in blocks[-2:]:
            row=block.get(case,{})
            if not all(g in row for g in GATES):
                break
            n=row[GATES[0]]["attempts"]
            k=[row[g]["passes"] for g in GATES]
            if type(n) is not int or any(type(x) is not int for x in k) or any(
                    row[g]["attempts"]!=n for g in GATES) or not n>=k[0]>=k[1]>=k[2]>=k[3]>=0:
                raise ValueError("Inconsistent fixed-height reference gates")
            if n<MIN_ATTEMPTS:
                break
            counts.append([1.]+[x/n for x in k])
        if len(counts)!=2:
            continue
        for i,stage in enumerate(("first_support","bilateral","traversal","post_traversal_survival")):
            gaps=[r[i]-r[i+1] for r in counts]
            if min(gaps)>=MIN_GAP:
                candidates.append(dict(case=case,height_mm=height*1000,stage=stage,
                    repeated_gap=min(gaps),rates=counts))
    candidates.sort(key=lambda c:(-c["repeated_gap"],c["case"],c["stage"]))
    return dict(status="review" if candidates else "no_repeated_gap",candidates=candidates)
