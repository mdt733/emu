"""Opt-in cold-school research: mild stairs as shared specialist context.

The 13-arm late-school screen found apparent stair gains paired with
regression of clean recovery/slope missions. It did NOT establish that
mixed training cures forgetting or that the 32mm uphill barrier is
physically feasible. Keep original foundation, one actor, PPO, servo,
observation and independent up/down difficulty teacher.
"""
from __future__ import annotations

# Exactly capability_focus.FOCUS_NAMES index order. These are ADDITIONAL
# chances within each role's ordinary birth cohort, NOT whole-batch shares.
# STEP retains its separate Academy.
CROSS_STAIR_ROLE_PROBABILITIES=(.08,.12,.15,0.,.10,.06)
# Shared lips are capped at these metres per role AND by already-earned
# direction-specific frontiers; downhill 55mm cannot invade recovery.
CROSS_STAIR_ROLE_CAP_M=(.012,.016,.018,0.,.014,.008)


def cross_stair_exposure_blend(*,step:int,enabled:bool,
                               start_steps:int=90_000_000,
                               ramp_steps:int=20_000_000)->float:
    """Age ramps exposure 0→1, never stair difficulty or reward value."""
    if (type(step) is not int or type(start_steps) is not int
            or type(ramp_steps) is not int or step<0 or
            start_steps<0 or ramp_steps<=0 or type(enabled) is not bool):
        raise ValueError("Invalid developmental school exposure schedule")
    if not enabled:
        return 0.
    return min(1.,max(0.,(step-start_steps)/ramp_steps))


def cross_stair_plan(blend:float)->dict[str,float]:
    """Maximum extra birth chance in each EXISTING specialist cohort."""
    from .capability_focus import FOCUS_NAMES
    if not 0.<=blend<=1.:
        raise ValueError("Invalid cross-specialty stair blend")
    return {name:blend*prob for name,prob in zip(
        FOCUS_NAMES,CROSS_STAIR_ROLE_PROBABILITIES)}


def mission_refresher_plan(*, base_weights, reference, current,
                           enabled=False, max_extra_share=.05,
                           donor_fraction=.10, min_drop=.05,
                           min_attempted=10):
    """Offline host-only prototype; NOT connected to the live teacher.

    reference/current: role -> task ID -> exam seed -> (mission passes,
    attempts). Use source-authenticated, frozen examinations with matched
    task definitions, assigned worlds and two distinct independent seeds.
    Reference is a fixed earlier checkpoint of the SAME student, not its
    training reward, survival fraction, or merely a same-age control.
    Authentication of exam receipts/world identities belongs upstream.

    A repeated per-task drop requests clean refresher births. A one-seed
    warning protects that role from donating practice but does not boost it.
    STEP never donates or receives this budget; no change to Academy height.
    Ordinary and extra-clean shares are absolute within the same eligible
    cohort and together sum to one. Existing mixed lessons remain ordinary.
    Thresholds and budgets are research hypotheses, not statistical proof.
    """
    import math
    from .capability_focus import FOCUS_NAMES
    roles = tuple(r for r in FOCUS_NAMES if r != "step")
    weights = tuple(float(w) for w in base_weights)
    if (len(weights) != len(FOCUS_NAMES)
            or not all(math.isfinite(w) and 0. <= w <= 1. for w in weights)
            or not math.isclose(sum(weights), 1., abs_tol=1e-9)
            or type(enabled) is not bool
            or not 0. <= max_extra_share <= .05
            or not 0. <= donor_fraction <= .10
            or not 0. < min_drop < 1.
            or type(min_attempted) is not int or min_attempted < 2):
        raise ValueError("Invalid mission refresher budget")
    ordinary = dict(zip(FOCUS_NAMES, weights))
    clean = dict.fromkeys(FOCUS_NAMES, 0.)
    warnings = {}
    confirmed = {}
    if not enabled:
        return dict(ordinary=ordinary, extra_clean=clean,
                    warnings=warnings, confirmed=confirmed)
    if set(reference) != set(roles) or set(current) != set(roles):
        raise ValueError("Missing or unexpected mission roles")
    cohort = None
    priority = {}
    for role in roles:
        before, after = reference[role], current[role]
        if not before or set(before) != set(after):
            raise ValueError("Mission tasks must match and cannot be empty")
        for task in sorted(before):
            a, b = before[task], after[task]
            seeds = set(a)
            if (len(seeds) != 2 or seeds != set(b)
                    or not all(type(s) is int and s >= 0 for s in seeds)
                    or (cohort is not None and seeds != cohort)):
                raise ValueError("Need the same two distinct exam seeds")
            cohort = seeds
            losses = []
            for seed in sorted(seeds):
                p, n = a[seed]
                q, m = b[seed]
                if (not all(type(v) is int for v in (p,n,q,m))
                        or n != m or n < min_attempted
                        or not 0 <= p <= n or not 0 <= q <= m):
                    raise ValueError("Invalid or unmatched mission counts")
                losses.append((p-q)/n)
            if max(losses) > min_drop:
                warnings.setdefault(role, {})[task] = tuple(losses)
            if min(losses) > min_drop:
                confirmed.setdefault(role, {})[task] = tuple(losses)
                priority[role] = max(priority.get(role, 0.), min(losses))
    # A small redistribution, not an additional training stream. Never fund
    # refreshers by cutting STEP or any role with a caution on either seed.
    donors = {r: ordinary[r]*donor_fraction for r in roles
              if r not in warnings}
    priority = {r: v for r,v in priority.items() if ordinary[r] > 0.}
    available = sum(donors.values())
    budget = min(max_extra_share, available) if priority else 0.
    if budget > 0.:
        for role, capacity in donors.items():
            ordinary[role] -= budget*capacity/available
        total = sum(priority.values())
        for role, severity in priority.items():
            clean[role] = budget*severity/total
    return dict(ordinary=ordinary, extra_clean=clean,
                warnings=warnings, confirmed=confirmed)
