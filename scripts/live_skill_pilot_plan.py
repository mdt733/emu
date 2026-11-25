"""Registration for the first matched live-education pilot (no GPU launch).

The two A/120M full checkpoints were selected from the completed A/B/C
archive because their source and hashes are known, there is no refresher
pin, and their age is exactly on the 4096-world PPO update grid. The
operator's actual checkpoint bytes must still be authenticated locally.
"""
from __future__ import annotations

from pathlib import Path
import json
from scripts.ks150_abc_school import RECIPE, MODEL, sha

PARENT_SOURCE="7b6c3f9e10b9b6ad2f7302dad00d9d25bbd5d574"
PARENT_STEP=120_061_952
WORLDS=4096
ROLLOUT=32
UPDATES=40
TARGET=PARENT_STEP+UPDATES*WORLDS*ROLLOUT
PARENTS={
    9501:dict(checkpoint="8ceea10302a99b6e5877ce6832c58c3399559c828c94ce9d16b0976adf72adbb",
              config="75b5e9ad6f728e6f7939c4fa537456389b327a0c0300829462f59785d85a02d7",
              actor="c4a293e014ec5ab2109ecb46f2183fc13f578d72f9c242bd7e4a5f05e7f888b8"),
    9502:dict(checkpoint="4d1a4de07e5833d1280064aa6a7d5b66f5fa31907b3530628625b4da49e92420",
              config="7fdd288af94159273ad5e0adf19962841397a5eabb937dd26f280a88c9af5a86",
              actor="dd492699b2b5597122aaed7a146e76526df6303a9f120ad79bde6da8b87cce11"),
}
# Common classroom: all 4096 worlds learn; identical reference and
# practice birth shares. The one shared uniform control saves two runs.
ARMS={
    "U":(),
    "R":("--skill-adaptive-recovery",),
    "H":("--skill-adaptive-uphill",),
}
RECIPE_4K=list(RECIPE)
RECIPE_4K[RECIPE_4K.index("--envs")+1]=str(WORLDS)
RECIPE_4K[RECIPE_4K.index("--fixed-eval-every-steps")+1]="20000000"
COMMON=tuple(RECIPE_4K)+(
    "--live-skill-observer","--skill-reference-recovery-share",".15",
    "--skill-practice-recovery-share",".15","--skill-uphill-reference-share",".15",
    "--skill-uphill-practice-share",".15",
)


def registration(parent_root):
    root=Path(parent_root).expanduser().resolve()
    if PARENT_STEP%(WORLDS*ROLLOUT):
        raise RuntimeError("Parent age is not aligned to new PPO update grid")
    return dict(schema="ks150_live_skill_pilot_v1",parent_root=str(root),
        parent_source=PARENT_SOURCE,parent_step=PARENT_STEP,
        model_sha256=MODEL,worlds=WORLDS,rollout=ROLLOUT,
        updates=UPDATES,target=TARGET,parents=PARENTS,
        arms={k:list(v) for k,v in ARMS.items()},common=list(COMMON),
        total_additional_steps=len(PARENTS)*len(ARMS)*UPDATES*WORLDS*ROLLOUT,
        status="registration_only_no_launch")


def verify_parent(parent_root,seed):
    """Hash authentic raw parent artifacts before constructing a fork.

    The archive's immutable checkpoint.pkl lives in evidence/, not the
    trainer's required checkpoints/ directory. A future runner must stage
    an exact copy/link plus the matching saved config into a new parent
    directory, without modifying the A/B/C evidence.
    """
    if seed not in PARENTS:
        raise ValueError("Unregistered parent")
    p=Path(parent_root).expanduser().resolve()/"evidence"/f"A-s{seed}"/"120m"
    expected=PARENTS[seed]
    for name,key in (("checkpoint.pkl","checkpoint"),("training_config.json","config"),
                     ("actor.emupolicy","actor")):
        if not (p/name).is_file() or sha(p/name)!=expected[key]:
            raise RuntimeError(f"Missing or altered registered parent: {p/name}")
    receipt=json.loads((p/"stage.json").read_text())
    actor=receipt["actor_source"]
    if (actor["step"]!=PARENT_STEP or actor["seed"]!=seed or
            actor["git_commit"]!=PARENT_SOURCE or actor["robot_model_sha256"]!=MODEL or
            receipt["checkpoint_sha256"]!=expected["checkpoint"] or
            receipt["artifact_sha256"]!=expected["actor"]):
        raise RuntimeError("Parent stage receipt does not match registration")
    return p


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-root",type=Path,default=Path.home()/"emu-data/studies/ks150-abc-unattended-v2")
    p.add_argument("--verify-parents",action="store_true")
    a=p.parse_args()
    print(json.dumps(registration(a.parent_root),indent=2,sort_keys=True))
    if a.verify_parents:
        for seed in PARENTS:
            print("Authenticated parent:",verify_parent(a.parent_root,seed))


if __name__=="__main__":
    main()
