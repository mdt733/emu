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


class RecoveryTeacher:
    """Checkpoint-owned evidence clock, independent of report file windows.

    The clock counts observed world-steps in this learner's retained history,
    rather than wall/absolute training age. Rollback restores this clock and
    discards evidence learned after the selected snapshot. Live episode
    buffers restart exactly as the main learner's physics buffers do.
    """
    SCHEMA = "emu_recovery_teacher_v1"
    REVIEW_STEPS = 1_000_000

    def __init__(self):
        self.clock=0
        self.window_start=0
        self.pending=[[0,0] for _ in range(6)]
        self.rows=[]
        self.weights=list(UNIFORM)

    @staticmethod
    def _row(start,end,counts):
        from .live_skill_reference import (REFERENCE_KICKS_MPS, REFERENCE_VX_MPS,
            REFERENCE_SENSOR_SCALE, REFERENCE_STEP_COMMAND_M, SETTLE_DEADLINE_S,
            SETTLE_FRAMES)
        return dict(schema="emu_live_skill_window_v1",policy_epoch=0,
            start_step=start,end_step=end,
            reference_kicks_mps=list(REFERENCE_KICKS_MPS),
            reference_vx_mps=REFERENCE_VX_MPS,
            reference_sensor_scale=REFERENCE_SENSOR_SCALE,
            reference_step_command_m=REFERENCE_STEP_COMMAND_M,
            reference_settle_deadline_s=SETTLE_DEADLINE_S,
            reference_settle_frames=SETTLE_FRAMES,
            reference_horizon_s=SETTLE_DEADLINE_S,
            reference_mastery_gate="settled_and_survived",
            cells={f"reference_recovery/mission/{i}":dict(attempts=n,passes=k)
                   for i,(n,k) in enumerate(counts) if n})

    def observe(self,counts,world_steps):
        from .live_skill_ledger import Block
        if type(world_steps) is not int or world_steps<=0 or len(counts)!=6:
            raise ValueError("Invalid teacher rollout")
        for i,(n,k) in enumerate(counts):
            Block(n,k)
            self.pending[i][0]+=n
            self.pending[i][1]+=k
        self.clock+=world_steps
        if self.clock-self.window_start<self.REVIEW_STEPS:
            return None
        row=self._row(self.window_start,self.clock,self.pending)
        self.rows.append(row)
        # Empty reference windows are genuine absence of evidence, not a
        # fresh reason to increase the dose from an old regression.
        plan=(practice_plan(self.rows,self.weights) if any(n for n,k in self.pending)
              else dict(weights=list(self.weights),reason="no_new_reference_trials",
                        selected=[],statuses=[]))
        self.weights=list(plan["weights"])
        self.window_start=self.clock
        self.pending=[[0,0] for _ in range(6)]
        return dict(evidence_clock=self.clock,**plan)

    def state_dict(self,checkpoint_step):
        import copy
        return dict(schema=self.SCHEMA,checkpoint_step=int(checkpoint_step),
                    clock=self.clock,window_start=self.window_start,
                    pending=copy.deepcopy(self.pending),rows=copy.deepcopy(self.rows),
                    weights=list(self.weights))

    @classmethod
    def from_state_dict(cls,state,checkpoint_step):
        import copy
        from .live_skill_ledger import Block
        if (not isinstance(state,dict) or set(state)!={"schema","checkpoint_step",
                "clock","window_start","pending","rows","weights"} or
                state["schema"]!=cls.SCHEMA or state["checkpoint_step"]!=int(checkpoint_step)):
            raise ValueError("Teacher checkpoint identity mismatch")
        clock,start=state["clock"],state["window_start"]
        if (type(clock) is not int or type(start) is not int or
                not 0<=start<=clock<=int(checkpoint_step) or clock-start>=cls.REVIEW_STEPS):
            raise ValueError("Invalid teacher evidence clock")
        if len(state["pending"])!=6 or len(state["weights"])!=6:
            raise ValueError("Invalid teacher dimensions")
        for n,k in state["pending"]:
            Block(n,k)
        weights=state["weights"]
        if (any(not 1./12-1e-8<=x<=7./12+1e-8 for x in weights)
                or abs(sum(weights)-1.)>1e-6):
            raise ValueError("Invalid checkpoint practice weights")
        rows=state["rows"]
        if rows:
            if rows[0]["start_step"]!=0 or rows[-1]["end_step"]!=start or any(
                    r["policy_epoch"]!=0 for r in rows):
                raise ValueError("Invalid teacher history boundary")
            summarize_segment(rows)
        elif start!=0:
            raise ValueError("Missing teacher history")
        obj=cls()
        obj.clock=clock
        obj.window_start=start
        obj.pending=copy.deepcopy(state["pending"])
        obj.rows=copy.deepcopy(rows)
        obj.weights=list(weights)
        return obj
