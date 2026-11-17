"""Promotion-oriented evidence for ONE truly random-initialized H40 run.

No imported historical actor, no Academy reward as success, no changing
the examiner between stages. Audits real 120/140/160M frozen physics and
explicitly refuses to call a successful source-only warm fork reproducible.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile


AGES=(120,140,160)
TASKS=("flat","up_4_40","up_44_60","up_64_80","down_4_40","down_44_80")
MODEL_SHA="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_worlds(path:Path):
    with path.open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def groups(rows):
    return {
        "flat":[r for r in rows if r["kind"]=="flat_control"],
        "up_4_40":[r for r in rows if r["kind"]=="step_up"
                   and float(r["exact_step_m"])<=.040001],
        "up_44_60":[r for r in rows if r["kind"]=="step_up"
                    and .04399<=float(r["exact_step_m"])<=.060001],
        "up_64_80":[r for r in rows if r["kind"]=="step_up"
                    and float(r["exact_step_m"])>=.06399],
        "down_4_40":[r for r in rows if r["kind"]=="step_down"
                     and float(r["exact_step_m"])>=-.040001],
        "down_44_80":[r for r in rows if r["kind"]=="step_down"
                      and float(r["exact_step_m"])<=-.04399],
    }


def count(rows,name):
    return sum(str(r[name]).lower()=="true" for r in rows)


def aggregate(rows):
    n=len(rows)
    if not n: raise RuntimeError("Empty physical exam category")
    falls=sum(int(float(r["fall_frame"]))>=0 for r in rows)
    result=dict(
        attempts=n,falls=falls,survival=1.-falls/n,
        lead_foot=count(rows,"lead_foot_top")/n,
        both_feet=count(rows,"both_feet_top_five")/n,
        upright_cross=count(rows,"traversed_upright_5frame")/n,
        controlled_cross=count(rows,"traversed_5frame")/n,
        transfer_falls=count(rows,"fell_after_lead_before_both"),
        mean_forward_m=sum(float(r["best_forward_progress_m"]) for r in rows)/n,
        mean_lateral_m=sum(float(r["max_abs_lateral_offset_m"]) for r in rows)/n)
    for k in ("lead_foot","both_feet","upright_cross","controlled_cross"):
        result[k+"_count"]=round(result[k]*n)
    return result


def evaluate(root:Path):
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    from emu.contracts import ROBOT_XML
    if sha(ROBOT_XML)!=MODEL_SHA:
        raise RuntimeError("Different compiled robot mechanics from the physical H40 reference")
    expected_exam=exam_source_fingerprint()
    run_meta=json.loads((root/"run_meta.json").read_text(encoding="utf-8"))
    source=(run_meta.get("source_provenance") or {}).get("git_commit")
    if not source or len(source)!=40:
        raise RuntimeError("Not one fully recorded source-pinned training run")
    configs=json.loads((root/"training_config.json").read_text(encoding="utf-8"))
    if (not configs.get("holistic_cold_curriculum") or
            int(configs["steps"])!=160_000_000 or
            configs["seed"]!=735 or configs["envs"]!=2048 or
            configs["physical_profile"]!="h40-spring-v1" or
            configs.get("resume") not in ("","auto")):
        # The trainer stores auto or a normalized same-run internal parent
        # when resuming. Checkpoint/manifests below bind the actual lineage.
        if not (configs.get("holistic_cold_curriculum") and
                int(configs["steps"])==160_000_000 and
                configs["seed"]==735 and configs["envs"]==2048 and
                configs["physical_profile"]=="h40-spring-v1"):
            raise RuntimeError("Not the expected fully cold H40 curriculum")
    exam={}
    categories={}
    for age in AGES:
        stage=root/"stages"/f"{age}m"
        manifest=json.loads((stage/"stage_manifest.json").read_text(encoding="utf-8"))
        if (manifest["source_git_commit"]!=source or
                manifest["target_steps"]!=age*1_000_000 or
                manifest["physical_model_sha256"]!=MODEL_SHA or
                sha(stage/"final.pkl")!=manifest["checkpoint_sha256"] or
                sha(stage/"actor.emupolicy")!=manifest["actor_sha256"] or
                not manifest["no_historical_model_loaded"]):
            raise RuntimeError(f"{age}M actor is not from the authenticated cold run")
        cats={}
        for mode in ("stairs","flat"):
            folder=root/"heldout"/f"{age}m"/mode
            m=json.loads((folder/"matrix_summary.json").read_text(encoding="utf-8"))["manifest"]
            expected=dict(
                seed=1701,worlds=2048,assigned_worlds=2000,horizon=350,
                mode=mode,policy_sha256=manifest["actor_sha256"],
                robot_model_sha256=MODEL_SHA,
                exam_source_sha256=expected_exam,
                first_life_only=True,zero_optimizer_updates=True)
            if mode=="stairs":
                expected.update(levels=20,max_step_m=.08)
            for key,val in expected.items():
                if m.get(key)!=val:
                    raise RuntimeError(f"{age}M {mode}: mismatched {key}")
            rows=load_worlds(folder/"matrix_worlds.csv")
            if len(rows)!=2000:
                raise RuntimeError(f"{age}M {mode} has {len(rows)} not 2,000 trials")
            cats[mode]=rows
        paired=[("world","vx","vy","yaw")]
        if [tuple(r[k] for k in paired[0]) for r in cats["stairs"]] != [
                tuple(r[k] for k in paired[0]) for r in cats["flat"]]:
            raise RuntimeError(f"{age}M stairs/flat world commands differ")
        exam[age]=cats
        categories[age]={name:aggregate(rows) for name,rows in
                         groups(cats["stairs"]+cats["flat"]).items()}
    for kind in ("stairs","flat"):
        reference=[(r["world"],r["kind"],r["exact_step_m"],r["vx"],
                    r["vy"],r["yaw"]) for r in exam[120][kind]]
        for age in (140,160):
            candidate=[(r["world"],r["kind"],r["exact_step_m"],r["vx"],
                        r["vy"],r["yaw"]) for r in exam[age][kind]]
            if reference!=candidate:
                raise RuntimeError(f"Different exact held-out fixtures across 120M and {age}M")
    final=categories[160]
    # Required all-around evidence, not a reward or larger unverified step frontier.
    gates={
        "easy_up_4_40_controlled_at_least_38pct":
            final["up_4_40"]["controlled_cross"]>=.38,
        "easy_up_4_40_survival_at_least_90pct":
            final["up_4_40"]["survival"]>=.90,
        "up_44_60_both_feet_at_least_30pct":
            final["up_44_60"]["both_feet"]>=.30,
        "up_44_60_controlled_at_least_8pct":
            final["up_44_60"]["controlled_cross"]>=.08,
        "up_44_60_survival_at_least_60pct":
            final["up_44_60"]["survival"]>=.60,
        "down_44_80_controlled_at_least_45pct":
            final["down_44_80"]["controlled_cross"]>=.45,
        "down_44_80_survival_at_least_80pct":
            final["down_44_80"]["survival"]>=.80,
        "flat_survival_at_least_95pct":
            final["flat"]["survival"]>=.95,
        "flat_controlled_at_least_50pct":
            final["flat"]["controlled_cross"]>=.50,
        "up_64_80_survival_at_least_55pct":
            final["up_64_80"]["survival"]>=.55,
        # Downhill ought not silently disappear during later uphill school:
        "downhill_retention_vs_140M":
            final["down_44_80"]["controlled_cross"] >=
            categories[140]["down_44_80"]["controlled_cross"]-.10,
    }
    return dict(manifest=dict(
        training_source_sha=source,random_initialization=True,
        single_ppo_optimizer_lineage=True,
        physical_model_sha256=MODEL_SHA,
        independent_exam_source_sha256=expected_exam,
        training_seed=735,exam_seed=1701,
        no_historical_actor_loaded=True,
        caveat="A single cold 0-160M seed reproduces an executable process, not statistically reliable seed-to-seed ability; run a second seed next"),
        stages=categories,
        gates=gates,
        meets_provisional_promotion_bar=all(gates.values()))


def format_report(data):
    lines=["# H40 Spring V1 holistic curriculum — ONE cold 0→160M learner",
           "",
           "The 120M, 140M and 160M actors were all trained from the SAME randomly initialised PPO model, normaliser and curriculum; checkpoint ages are continuous, source SHA unchanged.",
           "Each actor was frozen and independently evaluated on 2,000 exact real-prism signed stairs and 2,000 matched flat controls.",
           "One training seed and one examiner seed. Academy reward is never a held-out success metric.",
           "",
           "| Task | Age | Attempts | Leading foot | Both feet | Upright | Controlled crossing | Survival | Mean forward |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in TASKS:
        for age in AGES:
            x=data["stages"][age][name]
            lines.append(
                f"| {name} | {age}M | {x['attempts']} | "
                f"{x['lead_foot']:.1%} | {x['both_feet']:.1%} | "
                f"{x['upright_cross']:.1%} | {x['controlled_cross']:.1%} | "
                f"{x['survival']:.1%} | {x['mean_forward_m']:.2f} m |")
    lines+=["","## Prespecified full-capability gates",""]
    for name,ok in data["gates"].items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name.replace('_',' ')}")
    lines += ["", "**Provisional candidate:** "+(
        "ALL current gates pass; repeat with another cold seed before proposing main."
        if data["meets_provisional_promotion_bar"] else
        "NOT YET. Preserve learner and inspect failed capabilities without automatic source tweaks."),
        "",
        "Do not infer real-hardware reliability from simulated 350-frame exams.",
        "Main and all previously trained 120M/140M/160M actors remain unchanged.",
        ""]
    return "\n".join(lines)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    root=args.run.expanduser().resolve()
    data=evaluate(root)
    (root/"holistic_cold_results.json").write_text(
        json.dumps(data,indent=2)+"\n",encoding="utf-8")
    (root/"holistic_cold_report.md").write_text(
        format_report(data),encoding="utf-8")
    output=root/"holistic_cold_results_bundle.zip"
    files=[root/x for x in (
        "holistic_cold_report.md","holistic_cold_results.json",
        "training_config.json","run_meta.json","metrics.csv",
        "fixed_eval.csv","results.md","results_data.json")]
    for age in (20,40,120,140,160):
        stage=root/"stages"/f"{age}m"
        files.extend([stage/"stage_manifest.json",stage/"actor.emupolicy"])
    for age in AGES:
        for mode in ("stairs","flat"):
            exam=root/"heldout"/f"{age}m"/mode
            files.extend([exam/name for name in (
                "matrix_summary.json","matrix_curve.csv","matrix_worlds.csv")])
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        for file in files:
            if not file.is_file():
                if file.name in ("fixed_eval.csv","results.md","results_data.json"):
                    continue
                raise FileNotFoundError(f"Complete evidence missing: {file}")
            z.write(file,arcname=file.relative_to(root).as_posix())
    print(format_report(data),flush=True)
    print("UPLOAD ZIP:",output,flush=True)


if __name__=="__main__":
    main()
