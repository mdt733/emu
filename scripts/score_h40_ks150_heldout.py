#!/usr/bin/env python3
"""Post-hoc H40-KS150 seed2701 physical non-regression report.

Read-only: no PPO, no champion selection or writes to source checkpoints.
First-life survival and completed aligned traversal remain separate axes.
All three actors are examined on the *same* 2048-world signed prism fixture.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

SEED = 2701
MODEL = "21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
ACTORS = {
    120: "cc1f0ab126126815f6cea907dbc3a8a2b3faa6caff8ccbe2f05133d7c090771d",
    150: "9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240",
    160: "cbe47e8225079ba67b899732f920998469320a22e26aa9d67f784c916a4cb5ac",
}
BANDS = (
    ("safe_4_40", 4, 40),
    ("frontier_44_56", 44, 56),
    ("stretch_44_80", 44, 80),
)
EXPECTED_HEIGHTS = tuple(float(4*i) for i in range(1, 21))


def read_exam_zip(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json")), json.loads(
            archive.read("summary.json"))


def read_stage(root: Path, age: int, seed: int) -> dict:
    where = root / f"{age}m"
    stairs = json.loads((where / "stairs" / "matrix_summary.json").read_text())
    matrix = stairs["manifest"]
    assert matrix["schema"] == "h40_true_prism_heterogeneous_matrix_v5_signed80"
    assert matrix["seed"] == seed and matrix["mode"] == "stairs"
    assert matrix["worlds"] == 2048 and matrix["assigned_worlds"] == 2000
    assert matrix["levels"] == 20 and matrix["trials_per_level"] == 50
    assert matrix["first_life_only"] and matrix["zero_optimizer_updates"]
    assert matrix["robot_model_sha256"] == MODEL
    assert matrix["actor_sha256"] == ACTORS[age]

    curves = stairs["curves"]
    assert len(curves) == 40, f"{age}M: expected all signed 4..80mm levels"
    for kind, sign in (("step_up", 1), ("step_down", -1)):
        matching = [row for row in curves if row["kind"] == kind]
        assert len(matching) == 20
        assert tuple(sorted(sign*float(r["requested_step_mm"]) for r in matching)) == EXPECTED_HEIGHTS
        assert all(r["worlds"] == 50 for r in matching)
        assert all(abs(float(r["requested_step_mm"])-float(r["measured_step_mm_mean"])) <= .5
                   for r in matching), "Requested prism height differs from physical mesh"

    pulse_m, pulse_s = read_exam_zip(where / f"pulse-s{seed}.zip")
    assert pulse_m["schema"] == "emu_h40_compact_specialist_pulse_v1"
    assert pulse_m["seed"] == seed and pulse_m["worlds"] == 2048
    assert pulse_m["actor_sha256"] == ACTORS[age]
    assert pulse_m["robot_model_sha256"] == MODEL
    assert pulse_m["first_life_only"] and pulse_m["training_updates"] == 0
    groups = {x["specialty"]: x for x in pulse_s["headlines"]}
    assert len(groups) == 6

    deep_m, deep_s = read_exam_zip(where / f"deep-s{seed}.zip")
    assert deep_m["schema"] == "emu_physical_capability_v1"
    assert deep_m["seed"] == seed and deep_m["worlds"] == 8
    assert deep_m["actor_weights_sha256"] == ACTORS[age]
    assert deep_m["robot_model_sha256"] == MODEL
    assert deep_m["first_life_only"] and deep_m["no_optimizer_updates"]
    deep = {x["scenario"]: x for x in deep_s["scenarios"]}
    assert len(deep) == 9

    scores = {}
    for kind in ("step_up", "step_down"):
        for name, lo, hi in BANDS:
            subset = [x for x in curves
                      if x["kind"] == kind and lo <= abs(float(x["requested_step_mm"])) <= hi]
            n = sum(int(x["worlds"]) for x in subset)
            assert n == (hi-lo)//4*50 + 50, (age, kind, name, n)
            traversals = sum(int(x["traversed_5frame"]) for x in subset)
            survived = sum(round(float(x["survival_rate"])*x["worlds"]) for x in subset)
            assert 0 <= traversals <= n and 0 <= survived <= n
            scores[f"{kind}_{name}"] = {
                "traversals": traversals, "survived": survived, "n": n,
                "traversal_frac": traversals/n, "survival_frac": survived/n,
            }
    return {"age": age, "scores": scores, "pulse": groups, "deep": deep,
            "matrix_manifest": matrix, "pulse_manifest": pulse_m,
            "deep_manifest": deep_m}


def safe_ascent_blockers(reference_scores: dict, candidate_scores: dict) -> list[str]:
    """Predeclared SAFETY screen, never infer safety from traverse-only gain.

    These are engineering thresholds for future confirmation, not a
    probabilistic confidence or a retrospectively revised V1 selection.
    """
    reference_safe = reference_scores["step_up_safe_4_40"]["survival_frac"]
    candidate_safe = candidate_scores["step_up_safe_4_40"]["survival_frac"]
    frontier = candidate_scores["step_up_frontier_44_56"]["survival_frac"]
    problems = []
    if candidate_safe < reference_safe - .02 - 1e-10:
        problems.append("uphill 4–40mm survival regression")
    if frontier < .90:
        problems.append("uphill 44–56mm survival below 90%")
    return problems


def review(root: Path, seed: int = SEED) -> str:
    assert seed >= 0
    stages = {}
    for age in ACTORS:
        where = root / f"{age}m"
        needed = [where/"stairs"/"matrix_summary.json",
                  where/f"pulse-s{seed}.zip", where/f"deep-s{seed}.zip"]
        if all(x.is_file() for x in needed):
            stages[age] = read_stage(root, age, seed)
        elif any(x.exists() for x in needed):
            # Partial attempts are normal in a long unattended GPU exam,
            # but do not claim missing tests were completed.
            print(f"{age}M: some exams pending; no result for this actor yet")
    if stages:
        # No script, actor, model, or comparator drift within the seed cohort.
        for field, key in (("matrix_manifest", "exam_source_sha256"),
                           ("pulse_manifest", "source_sha256"),
                           ("deep_manifest", "source_files_sha256")):
            refs = [x[field][key] for x in stages.values()]
            assert all(x == refs[0] for x in refs), f"{field} source mismatch"
    lines = [
        "# H40-KS150 held-out physical audit",
        "",
        f"Evaluation seed **{seed}**, distinct from training seed 735 and prior exam seed 1701.",
        "These are frozen same-source actor comparisons, not new PPO training seeds.",
        "2048 true-prism worlds per actor; 50 per step height and direction; first life only.",
        "Absolute world XY station keeping and parked-goal arrivals are DIAGNOSTIC ONLY.",
        "",
        "| Age | Up 4–40mm traverse | Up 4–40mm survive | Up 44–56mm traverse | Up 44–56mm survive | Down 4–40mm survive | Down 44–80mm survive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for age in ACTORS:
        if age not in stages:
            lines.append(f"| {age}M | pending | pending | pending | pending | pending | pending |")
            continue
        s = stages[age]["scores"]
        def f(key: str, metric: str) -> str:
            row=s[key]
            return f"{row[metric]}/{row['n']} ({row[metric]/row['n']:.1%})"
        lines.append(
            f"| {age}M | {f('step_up_safe_4_40','traversals')} | "
            f"{f('step_up_safe_4_40','survived')} | "
            f"{f('step_up_frontier_44_56','traversals')} | "
            f"{f('step_up_frontier_44_56','survived')} | "
            f"{f('step_down_safe_4_40','survived')} | "
            f"{f('step_down_stretch_44_80','survived')} |")

    lines += ["", "## Core long-duration and compact checks", "",
              "| Age | 60s balance | 60s unwanted speed | Compact steps passed | Compact recovery passed |",
              "|---|---:|---:|---:|---:|"]
    for age in ACTORS:
        if age not in stages:
            lines.append(f"| {age}M | pending | pending | pending | pending |")
            continue
        s=stages[age]
        stand=s["deep"]["stand_zero_60s"]
        steps=s["pulse"]["step"]
        resilience=s["pulse"]["resilience"]
        lines.append(
            f"| {age}M | {stand['zero_command_balance_passes']}/8 | "
            f"{stand['mean_horizontal_speed_mps']:.4f} m/s | "
            f"{steps['passed']}/{steps['attempted']} | "
            f"{resilience['passed']}/{resilience['attempted']} |")

    lines += ["", "## Safety-first qualification", "",
              "The V1 screen rewarded additional traversals but missed uphill fall-rate regression.",
              "For a candidate to be **provisionally safe in its intended stair band**:",
              "- Survival uphill 4–40mm must be at least the frozen 120M reference minus 2 percentage points.",
              "- Survival uphill 44–56mm must be at least 90% before calling 5.5cm climbing dependable.",
              "- Traversals, descending and 30s velocity tracking must be reported alongside survival.",
              "- No single held-out evaluation seed certifies hardware reliability or causal efficacy.",
              "",
    ]
    if len(stages) == 3:
        for age in (150,160):
            v=stages[age]["scores"]
            reasons = safe_ascent_blockers(stages[120]["scores"], v)
            lines.append(
                f"- **{age}M**: " +
                ("meets both survival floors (still requires all-skill review)"
                 if not reasons else "not certified — " + "; ".join(reasons)) + ".")
        lines += ["", "No automatic champion promotion. Preserve all three frozen actors."]
    else:
        lines += ["**Incomplete study:** wait for the remaining frozen exams before deciding.",
                  "No candidate is certified from partial data."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    report=review(root,args.seed)
    root.mkdir(parents=True,exist_ok=True)
    (root/"heldout-report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
