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


def refresher_birth_parameters(*, base_weights, academy_probability,
                                plan=None):
    """Translate an offline plan to one role draw + conditional clean draw.

    Input weights are the *post-Academy* non-athletic role allocation from
    birth_lesson_weights. STEP probability and its conditional Academy
    draw remain unchanged. No new environment or checkpoint fields yet.
    """
    import math
    from .capability_focus import FOCUS_NAMES
    weights = tuple(float(x) for x in base_weights)
    if (len(weights) != len(FOCUS_NAMES)
            or not all(math.isfinite(x) and 0. <= x <= 1. for x in weights)
            or not math.isclose(sum(weights),1.,abs_tol=1e-9)
            or not 0. <= academy_probability <= 1.):
        raise ValueError("Invalid baseline birth parameters")
    zeros = (0.,)*len(weights)
    if plan is None:
        return weights, academy_probability, zeros
    ordinary, clean = plan["ordinary"], plan["extra_clean"]
    if set(ordinary) != set(FOCUS_NAMES) or set(clean) != set(FOCUS_NAMES):
        raise ValueError("Unexpected refresher role")
    o = tuple(float(ordinary[r]) for r in FOCUS_NAMES)
    c = tuple(float(clean[r]) for r in FOCUS_NAMES)
    if (not all(math.isfinite(x) and 0. <= x <= 1. for x in o+c)
            or not math.isclose(sum(o)+sum(c),1.,abs_tol=1e-9)
            or sum(c) > .05+1e-9
            or any(x < .90*w-1e-9 or x > w+1e-9
                   for x,w in zip(o,weights))
            or any(x > 0. and (w <= 0. or q < w-1e-9)
                   for x,w,q in zip(c,weights,o))):
        raise ValueError("Refresher budget violates baseline protections")
    i = FOCUS_NAMES.index("step")
    if c[i] != 0. or o[i] != weights[i]:
        raise ValueError("Cannot alter STEP/Academy budget")
    combined = tuple(x+y for x,y in zip(o,c))
    conditional = tuple(y/t if t > 0. else 0.
                        for y,t in zip(c,combined))
    return combined, academy_probability, conditional


def clean_refresher_assignment(xp, focus, specialist, athletic, academy,
                               uniform, conditional):
    """Birth-only mask from an independently salted [0,1) draw.

    The caller must persist this birth assignment for the whole episode.
    Do not resample on each frame or infer it from a changing host plan.
    """
    from .capability_focus import STEP
    probs = xp.asarray(conditional, dtype=xp.float32)
    return (specialist & (athletic < .5) & ~academy & (focus != STEP)
            & (uniform < probs[focus]))


def clean_refresher_terrain(xp, focus, clean, slope, up, down, cross):
    """Clean means no competing lips; SLOPE retains its earned grade.

    This isolates terrain context only, not an exact frozen-exam mission.
    Original specialist commands, perturbations and rewards are untouched.
    """
    from .capability_focus import SLOPE
    return (xp.where(clean & (focus != SLOPE),0.,slope),
            xp.where(clean,0.,up), xp.where(clean,0.,down), cross & ~clean)


def replay_refresher_schedule(*, enabled, receipt_bytes, receipt_sha256,
                              lineage, step, base_weights,
                              academy_probability):
    """Pure host replay for a future explicit, checkpointed consumer.

    The caller supplies a TRUSTED receipt hash and lineage identities from
    authenticated checkpoint state, never from the receipt being checked.
    At activation the checkpoint hash identifies the exact frozen student;
    later it denotes the original activation ancestor, not the newest file.
    No receipt is read or applied when disabled. Expiry returns baseline.
    Invalid active receipts fail closed; caller must not silently replace
    them or infer identity from filenames.
    """
    import hashlib
    import json
    if type(enabled) is not bool:
        raise ValueError("Explicit Boolean refresher enable required")
    baseline = refresher_birth_parameters(base_weights=base_weights,
                                          academy_probability=academy_probability)
    if not enabled:
        return baseline
    if (not isinstance(receipt_bytes,bytes)
            or hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256):
        raise ValueError("Approved refresher receipt digest mismatch")
    r = json.loads(receipt_bytes)
    keys = ("run_name","training_seed","source_commit",
            "training_config_sha256","activation_checkpoint_sha256")
    if (r.get("schema") != "emu_mission_refresher_schedule_v1"
            or r.get("applied") is not False
            or any(k not in lineage or r.get(k) != lineage[k] for k in keys)):
        raise ValueError("Approved refresher lineage mismatch")
    start, end = r["start_step"], r["until_step"]
    if (type(start) is not int or type(end) is not int or type(step) is not int
            or not 0 < start < end <= start+10_000_000 or step < start):
        raise ValueError("Invalid refresher activation or expiry")
    if step >= end:
        return baseline
    if (tuple(r["base_weights"]) != baseline[0]
            or r["academy_probability"] != academy_probability):
        raise ValueError("Baseline teaching allocation changed; require new review")
    advisory = r["advisory"]
    if (advisory.get("schema") != "emu_mission_refresher_advisory_v1"
            or advisory.get("current_step") != start
            or advisory.get("applied") is not False):
        raise ValueError("Invalid underlying mission advisory")
    return refresher_birth_parameters(base_weights=base_weights,
               academy_probability=academy_probability,plan=advisory["plan"])


def refresher_recipe_sha256(config):
    """Stable substantive recipe identity across same-run target extensions."""
    from dataclasses import asdict
    import hashlib
    import json
    d = asdict(config)
    for name in ("steps","resume","device"):
        d.pop(name,None)
    return hashlib.sha256(json.dumps(
        d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def bind_refresher_receipt(*, enabled, run_dir, receipt_sha256, config_sha256,
                           source_commit, training_seed, resumed_meta,
                           resumed_checkpoint, mature_step,
                           expected_base=None):
    """Bind at one exact checkpoint; permanently pin one receipt per run.

    Called at process startup only. The independent operator supplies
    receipt_sha256; the receipt itself is never the source of trust.
    The immutable local lock survives a crash before the first new
    checkpoint. Subsequent checkpoints also carry the exact pin.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import tempfile
    root = Path(run_dir).resolve()
    pin_path = root/"mission_refresher_pin.json"
    receipt_path = root/"mission_refresher_approval.json"
    old = (resumed_meta or {}).get("mission_refresher_pin")
    if not enabled:
        if old is not None or pin_path.exists() or receipt_sha256:
            raise ValueError("Refresher approval present in disabled run")
        return None, None
    if not (receipt_sha256 or pin_path.exists() or old is not None):
        if receipt_path.exists():
            raise ValueError("Unapproved refresher receipt; supply trusted digest")
        return None, None
    if not re.fullmatch("[0-9a-f]{64}",receipt_sha256 or ""):
        raise ValueError("Refresher resume requires explicit pinned SHA256")
    data = receipt_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=receipt_sha256:
        raise ValueError("Refresher approval digest mismatch")
    receipt = json.loads(data)
    start = receipt["start_step"]
    end = receipt["until_step"]
    if (type(start) is not int or type(end) is not int
            or not mature_step<=start<end<=start+10_000_000):
        raise ValueError("Refresher activation requires stable mature allocation")
    identity = dict(run_name=root.name,training_seed=training_seed,
        source_commit=source_commit,training_config_sha256=config_sha256,
        activation_checkpoint_sha256=receipt["activation_checkpoint_sha256"])
    if (not re.fullmatch("[0-9a-f]{40}",source_commit)
            or not re.fullmatch("[0-9a-f]{64}",identity["activation_checkpoint_sha256"])
            or any(receipt.get(k)!=v for k,v in identity.items())):
        raise ValueError("Refresher approval identity mismatch")
    pin = dict(receipt_sha256=receipt_sha256,**identity,
               start_step=start,until_step=end)
    if (not resumed_meta or resumed_checkpoint is None
            or resumed_meta.get("run_name")!=root.name
            or (resumed_meta.get("source_provenance") or {}).get("git_commit")!=source_commit
            or Path(resumed_checkpoint).resolve().parent != root/"checkpoints"):
        raise ValueError("Refresher requires same-student same-source resume")
    step = resumed_meta["step"]
    if step<start or (old is not None and old!=pin):
        raise ValueError("Refresher checkpoint predates activation or has another pin")
    if pin_path.exists():
        if json.loads(pin_path.read_text(encoding="utf-8"))!=pin:
            raise ValueError("Refresher pin cannot be replaced")
    elif old is not None:
        raise ValueError("Missing permanent refresher pin")
    if old is None:
        # First activation (or crash before the first new checkpoint).
        if step!=start:
            raise ValueError("Unpinned checkpoint is not activation checkpoint")
        h=hashlib.sha256()
        with Path(resumed_checkpoint).open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                h.update(block)
        if h.hexdigest()!=identity["activation_checkpoint_sha256"]:
            raise ValueError("Refresher activation checkpoint digest mismatch")
    # Validate the actual host allocation before creating the permanent pin.
    if expected_base is not None and (
            tuple(receipt['base_weights'])!=tuple(expected_base[0]) or
            receipt['academy_probability']!=expected_base[1]):
        raise ValueError('Approved refresher baseline does not match live teacher')
    # Validate the replay contract before creating the permanent pin.
    replay_refresher_schedule(enabled=True,receipt_bytes=data,
        receipt_sha256=receipt_sha256,lineage=identity,step=start,
        base_weights=receipt["base_weights"],
        academy_probability=receipt["academy_probability"])
    if not pin_path.exists():
        # Exclusive hard link of complete fsynced bytes: a racing process
        # cannot replace an existing pin and partial JSON is never visible.
        tmp=None
        try:
            with tempfile.NamedTemporaryFile("w",dir=root,delete=False,
                                             encoding="utf-8") as stream:
                tmp=Path(stream.name)
                json.dump(pin,stream,sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(tmp,pin_path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return pin,data
