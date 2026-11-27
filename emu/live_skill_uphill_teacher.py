"""Checkpoint-owned uphill teacher. Fixed references are the only feedback.

Each case closes a disjoint block when enough completed trials have arrived.
Two blocks must support a prerequisite diagnosis or a mastery decision.
Practice keeps half its allocation uniform, and reward is faded one unit
per newly supported block pair. No promotion of the ordinary Academy.
"""
from __future__ import annotations
from math import isfinite
from copy import deepcopy
from .live_skill_ledger import Block
from .live_skill_uphill import uphill_heights
from .live_skill_teacher import UNIFORM, MAX_WEIGHT_CHANGE

# Counts: attempts, contact, first sustained support, both feet, traversal,
# traversal + survival. The five successes must be nested.
MIN_ATTEMPTS=40
REVIEW_STEPS=1_000_000
GAP=.15
MAX_STALLED_PAIRS=3
MIN_PROGRESS=.05


def check_counts(row):
    if len(row)!=6 or any(type(x) is not int for x in row) or not all(
            a>=b for a,b in zip(row,row[1:])) or row[-1]<0:
        raise ValueError("Invalid nested uphill evidence")


def diagnose_pair(pair):
    if len(pair)!=2 or min(r[0] for r in pair)<MIN_ATTEMPTS:
        return 0
    # Only teach achievable prerequisites; a no-contact problem should
    # not be misdiagnosed as poor sustained foot support.
    candidates=[]
    for stage,idx in ((1,2),(2,3),(3,4)):
        if all(r[idx-1]/r[0]>=.30 and r[idx]/r[0]<=.80 and
               (r[idx-1]-r[idx])/r[0]>=GAP for r in pair):
            candidates.append((min((r[idx-1]-r[idx])/r[0] for r in pair),stage))
    return max(candidates,key=lambda x:(x[0],-x[1]))[1] if candidates else 0


class UphillTeacher:
    SCHEMA="emu_uphill_teacher_v3"
    def __init__(self,profile="ladder"):
        self.profile=profile
        self.heights=uphill_heights(profile)
        self.clock=0
        self.review_clock=0
        self.pending=[[0]*6 for _ in self.heights]
        self.blocks=[[] for _ in self.heights]
        self.stage=[0]*6
        self.bonus=[0.]*6
        self.weights=list(UNIFORM)
        self.unassisted=[False]*6
        self.stalled=[0]*6
        self.baseline=[0.]*6
        self.cooldown=[False]*6

    def observe(self,counts,world_steps):
        if type(world_steps) is not int or world_steps<=0 or len(counts)!=6:
            raise ValueError("Invalid uphill rollout")
        for row in counts:
            check_counts(row)
        self.clock+=world_steps
        for i,row in enumerate(counts):
            self.pending[i]=[a+b for a,b in zip(self.pending[i],row)]
        if self.clock-self.review_clock<REVIEW_STEPS:
            return None
        self.review_clock=self.clock
        closed=[]
        events=[]
        for i,row in enumerate(self.pending):
            if row[0]<MIN_ATTEMPTS:
                continue
            closed.append(i)
            self.blocks[i]=(self.blocks[i]+[row])[-2:]
            self.pending[i]=[0]*6
            pair=self.blocks[i]
            if len(pair)<2:
                continue
            # Reference trials always have the original reward. Mission
            # mastery is descriptive, not independent held-out evidence.
            self.unassisted[i]=all(Block(r[0],r[5]).wilson()[0]>=.70 for r in pair)
            stage=self.stage[i]
            if self.cooldown[i]:
                # One full fresh unshaped evidence pair before retry.
                self.cooldown[i]=False
                self.blocks[i]=[]
                events.append(dict(case=i,action='cooldown_complete'))
                continue
            if stage and self.bonus[i]>0:
                gate=stage+1
                if all(Block(r[0],r[gate]).wilson()[0]>=.70 for r in pair):
                    self.bonus[i]=max(0.,self.bonus[i]-1.)
                    if self.bonus[i]==0.:
                        self.stage[i]=0
                    # Fresh disjoint pair required for every fade or switch.
                    self.blocks[i]=[]
                    self.stalled[i]=0
                    events.append(dict(case=i,action='fade',bonus=self.bonus[i]))
                else:
                    rate=min(r[gate]/r[0] for r in pair)
                    if rate>=self.baseline[i]+MIN_PROGRESS:
                        self.baseline[i]=rate
                        self.stalled[i]=0
                    else:
                        self.stalled[i]+=1
                    self.blocks[i]=[]
                    if self.stalled[i]>=MAX_STALLED_PAIRS:
                        self.stage[i]=0; self.bonus[i]=0.
                        self.stalled[i]=0; self.cooldown[i]=True
                        events.append(dict(case=i,action='stalled_pause'))
                continue
            candidate=0 if self.unassisted[i] else diagnose_pair(pair)
            if candidate:
                self.stage[i]=candidate
                self.bonus[i]=4.
                self.baseline[i]=min(r[candidate+1]/r[0] for r in pair)
                self.stalled[i]=0
                self.blocks[i]=[]
                events.append(dict(case=i,action='start',stage=candidate))
        if not closed:
            return None
        selected=[i for i,s in enumerate(self.stage) if s and self.bonus[i]>0]
        desired=[1./12]*6 if selected else list(UNIFORM)
        for i in selected:
            desired[i]+=.5/len(selected)
        delta=max(abs(a-b) for a,b in zip(self.weights,desired))
        alpha=min(1.,MAX_WEIGHT_CHANGE/max(delta,1e-12))
        self.weights=[a+alpha*(b-a) for a,b in zip(self.weights,desired)]
        return dict(evidence_clock=self.clock,closed_cases=closed,stage=list(self.stage),
                    bonus=list(self.bonus),weights=list(self.weights),
                    reference_mastery=list(self.unassisted),events=events,heights_mm=[round(x*1000) for x in self.heights],
                    stalled_pairs=list(self.stalled),cooldown=list(self.cooldown))

    def state_dict(self,checkpoint_step):
        return dict(schema=self.SCHEMA,profile=self.profile,checkpoint_step=int(checkpoint_step),
            clock=self.clock,review_clock=self.review_clock,pending=deepcopy(self.pending),
            blocks=deepcopy(self.blocks),stage=list(self.stage),bonus=list(self.bonus),
            weights=list(self.weights),unassisted=list(self.unassisted),
            stalled=list(self.stalled),baseline=list(self.baseline),cooldown=list(self.cooldown))

    @classmethod
    def from_state_dict(cls,state,checkpoint_step,profile=None):
        if not isinstance(state,dict): raise ValueError('Uphill teacher checkpoint identity mismatch')
        obj=cls(profile if profile is not None else state.get('profile','ladder'))
        if state.get('profile')!=obj.profile:
            raise ValueError('Uphill teacher physical profile mismatch')
        if (not isinstance(state,dict) or set(state)!=set(obj.state_dict(checkpoint_step))
                or state["schema"]!=cls.SCHEMA or state["checkpoint_step"]!=int(checkpoint_step)):
            raise ValueError("Uphill teacher checkpoint identity mismatch")
        clock,review=state["clock"],state["review_clock"]
        if (type(clock) is not int or type(review) is not int or
                not 0<=review<=clock<=checkpoint_step or clock-review>=REVIEW_STEPS):
            raise ValueError("Invalid uphill teacher clock")
        for key in ("pending","blocks","stage","bonus","weights","unassisted","stalled","baseline","cooldown"):
            if len(state[key])!=6:
                raise ValueError("Invalid uphill teacher dimensions")
        for row,blocks,s,b,w,m,stall,base,cool in zip(*(state[k] for k in
                ("pending","blocks","stage","bonus","weights","unassisted","stalled","baseline","cooldown"))):
            check_counts(row)
            if len(blocks)>2:
                raise ValueError("Invalid uphill block history")
            for block in blocks:
                check_counts(block)
                if block[0]<MIN_ATTEMPTS:
                    raise ValueError("Insufficient closed uphill block")
            if (type(s) is not int or s not in range(4) or b not in (0.,1.,2.,3.,4.)
                    or (s==0)!=(b==0.) or not isfinite(w) or not 1/12-1e-8<=w<=7/12+1e-8
                    or type(m) is not bool or type(stall) is not int or not 0<=stall<MAX_STALLED_PAIRS
                    or not isfinite(base) or not 0.<=base<=1. or type(cool) is not bool
                    or (cool and s!=0) or (s==0 and stall!=0)):
                raise ValueError("Invalid uphill practice settings")
        if abs(sum(state["weights"])-1.)>1e-6:
            raise ValueError("Invalid uphill weights")
        for key in ("clock","review_clock","pending","blocks","stage","bonus","weights","unassisted","stalled","baseline","cooldown"):
            setattr(obj,key,deepcopy(state[key]))
        return obj
