"""Offline mission-maintenance review of four pinned specialist pulse ZIPs.

No training, checkpoint loading, actor execution or teacher mutation.
The operator supplies trusted SHA256s recorded at exam completion; hashes
establish byte identity, not a signature or proof of honest GPU execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from emu.developmental_school import mission_refresher_plan
from scripts.examine_h40_specialties import SCHEMA, allocate, score

ROOT = Path(__file__).resolve().parents[1]
OWNED = ("scripts/examine_h40_specialties.py", "emu/ppo.py",
         "emu/environment.py", "emu/contracts.py", "emu/robot.xml",
         "emu/carpet_prism_mjx.py", "emu/terrain_carpet_mjx.py",
         "emu/physical_variants.py", "emu/policy_export.py",
         "emu/domain_randomization.py", "emu/tof_geometry.py")
META = {"world", "specialty", "kind", "level", "severity", "unit",
        "replicate", "command_vx", "command_vy", "command_yaw", "step_m",
        "grade_deg", "initial_kick_mps", "program",
        "physical_mission_pass", "failure_reasons"}
TEXT = {"specialty", "kind", "unit", "failure_reasons"}
BOOL = {"survived", "physical_mission_pass"}


def _hex(value, size=64):
    return isinstance(value, str) and re.fullmatch(
        "[0-9a-f]{%d}" % size, value) is not None


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {name: _digest((ROOT/name).read_bytes()) for name in OWNED}


def _boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Invalid world Boolean")
    return value == "True"


def read_pulse(path, *, archive_sha256, actor_source, artifact_sha256,
               seed, model_sha256, sensor_sha256, expected_sources):
    """Validate frozen archive against separately trusted metadata and
    re-score every world using the registered physical mission criteria.
    """
    path = Path(path)
    if not all(_hex(x) for x in (archive_sha256, artifact_sha256,
                                model_sha256, sensor_sha256,
                                actor_source.get("actor_sha256"))):
        raise ValueError("Missing trusted SHA256 identity")
    data = path.read_bytes()
    if _digest(data) != archive_sha256:
        raise ValueError("Pulse archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        m = json.loads(z.read("manifest.json"))
        rows = list(csv.DictReader(io.StringIO(
            z.read("worlds.csv").decode("utf-8"))))
    expected = dict(schema=SCHEMA, source_sha256=expected_sources,
                    actor_source=actor_source, artifact_sha256=artifact_sha256,
                    actor_sha256=actor_source["actor_sha256"],
                    actor_step=actor_source["step"],
                    robot_model_sha256=model_sha256, sensor_sha256=sensor_sha256,
                    physical_profile="h40-spring-only", seed=seed,
                    worlds=2048, assigned_worlds=2048, horizon_frames=600,
                    backend="gpu", training_updates=0, first_life_only=True)
    if any(m.get(k) != v for k,v in expected.items()):
        raise ValueError("Pulse provenance or first-life contract mismatch")
    if (set(expected_sources) != set(OWNED)
            or expected_sources != source_hashes()
            or actor_source.get("robot_model_sha256") != model_sha256):
        raise ValueError("Examiner or physical model identity mismatch")
    trials = allocate(2048)
    if len(rows) != len(trials):
        raise ValueError("Incomplete assigned worlds")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed world record")
    fields = set(rows[0]) - META
    physical = {k: [_boolean(r[k]) if k in BOOL else float(r[k])
                    for r in rows] for k in fields}
    rebuilt = score(trials, physical, 600)
    counts = {}
    # Comparing every canonical field also rejects swapped or duplicated
    # world IDs, task assignments, commands, signed geometry and mission flags.
    for old, new in zip(rows, rebuilt):
        if set(old) != set(new):
            raise ValueError("Unexpected world record fields")
        for k,v in new.items():
            given = old[k] if k in TEXT else (
                _boolean(old[k]) if k in BOOL else float(old[k]))
            if given != v:
                raise ValueError("World identity or mission score mismatch: "+k)
        if new["survived"] and new["first_life_frames"] != 600:
            raise ValueError("Survival requires a full first life")
        role = new["specialty"]
        task = f"{new['kind']}/{new['level']}"
        count = counts.setdefault(role, {}).setdefault(task, [0,0])
        count[0] += int(new["physical_mission_pass"])
        count[1] += 1
    actual = {f"{r}/{t}": n for r, tasks in counts.items()
              for t, (_,n) in tasks.items()}
    if m.get("trials_per_case") != actual:
        raise ValueError("Registered task allocation mismatch")
    return counts, m


def review(registration):
    """Explicit immutable comparison registration; exactly two exams per age.

    Registration contains operator-pinned identities, not values inferred
    from archives under review. The same run/seed/source/config lineage
    must be established by the upstream checkpoint custodian; run_name
    alone is not a globally unique lineage identifier.
    """
    r = registration
    if r.get("schema") != "emu_mission_refresher_review_v1":
        raise ValueError("Unknown review registration")
    seeds = r["seeds"]
    if (len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("Two distinct registered exam seeds required")
    before, after = (r["reference"], r["current"])
    a, b = before["actor_source"], after["actor_source"]
    if (not _hex(a.get("git_commit"), 40)
            or not a.get("run_name")
            or type(a.get("seed")) is not int
            or any(a.get(k) != b.get(k) for k in
                   ("git_commit", "run_name", "seed", "robot_model_sha256"))
            or type(a.get("step")) is not int or type(b.get("step")) is not int
            or not 0 < a["step"] < b["step"]
            or a.get("actor_sha256") == b.get("actor_sha256")):
        raise ValueError("Need ordered frozen checkpoints of one registered student")
    cohorts = []
    runtime = None
    for stage in (before, after):
        entries = stage["exams"]
        if len(entries) != 2 or {e["seed"] for e in entries} != set(seeds):
            raise ValueError("Missing or duplicate stage exam seed")
        grouped = {}
        for e in entries:
            counts, manifest = read_pulse(
                e["path"], archive_sha256=e["sha256"],
                actor_source=stage["actor_source"],
                artifact_sha256=stage["artifact_sha256"], seed=e["seed"],
                model_sha256=r["model_sha256"], sensor_sha256=r["sensor_sha256"],
                expected_sources=r["source_sha256"])
            if not isinstance(manifest.get("runtime"), dict):
                raise ValueError("Missing physical runtime provenance")
            if runtime is None:
                runtime = manifest["runtime"]
            elif runtime != manifest["runtime"]:
                raise ValueError("Exam runtime mismatch")
            for role,tasks in counts.items():
                if role == "step":
                    continue
                for task,pair in tasks.items():
                    grouped.setdefault(role, {}).setdefault(task, {})[
                        e["seed"]] = tuple(pair)
        cohorts.append(grouped)
    plan = mission_refresher_plan(base_weights=r["base_weights"],
                                   reference=cohorts[0],current=cohorts[1],
                                   enabled=True)
    return dict(schema="emu_mission_refresher_advisory_v1",
                registration_sha256=_digest(json.dumps(
                    r,sort_keys=True,separators=(",",":")).encode()),
                reference_step=a["step"], current_step=b["step"], seeds=seeds,
                reference_counts=cohorts[0], current_counts=cohorts[1],
                plan=plan, applied=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("registration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = review(json.loads(args.registration.read_text(encoding="utf-8")))
    # Exclusive creation protects earlier decisions from being rewritten.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
