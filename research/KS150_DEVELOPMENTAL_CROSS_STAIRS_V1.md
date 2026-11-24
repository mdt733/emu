# EMU next-school research: shared stair practice across specialists

**Status:** first implementation, CI/CPU contracts only; **not** a trained policy,
not proven to prevent regression, not merged into released `main`.

Parent research source: `research/h40-unified-school-cold-v1`.
New branch: `research/h40-developmental-cross-stairs-v1`.
Physical system remains the same Spring-only KS150, six actuators, 360
sensor inputs and eight real MJX triangular-prism colliders.

## Why change the school?

The finished 13-arm experiment (25 reviewed stages) exposed a tradeoff:
several late-stair treatments improved small-height transfers while
independent recovery/slope mission performance declined. The unchanged
control also suffered a severe late recovery-*mission* regression. This
supports investigating **shared skill maintenance**; it does NOT prove
that specialist mixing prevents forgetting. The uphill Academy remained
around 3–4mm while independent 32mm+ tests almost universally failed.

Older source already placed weak terrain in all roles: the old
`role_envelope` scaled step difficulty by focus, sometimes below
the physical lip-recognition threshold. A new-age-independent
`cross_stair_blend` now creates explicit additional **episode births**
with a physical signed stair in each role, except STEP (which keeps its
separate Academy). There is **no new reward or privileged actor channel**.

## New developmental lesson plan

`--cross-specialty-stairs` is an opt-in **fresh cold-school** treatment.
The original foundation and old rewards are the control. Exposure begins
at 90M world steps, rises smoothly to the full role budgets at 110M, and
never increases **difficulty** by age. The signed UP and DOWN heights
must still be earned by the existing physical curriculum. Shared lips
are capped by role, even when the downhill Academy has reached 55mm.

| Specialty | Extra role-birth chance at 110M | Max injected signed lip | What is *still* tested |
|---|---:|---:|---|
| COMMAND | 8% | 12mm | Command tracking, turns and reversals on mild steps |
| RESILIENCE | 12% | 16mm | Recovery from actual perturbations on uneven terrain |
| SLOPE | 15% | 18mm | Incline handling plus small authentic lips |
| STEP | No extra; existing Academy | Earned UP/DOWN Academy frontier | Foothold, two-foot transfer and course continuation |
| TRANSITION | 10% | 14mm | Start/stop and speed switching on uneven terrain |
| CONTROL | 6% | 8mm | Clean efficient gait with occasional mild steps |

Budgets are **conditional on the role**, not whole-batch shares; athletic
rehearsal and inactive foundation worlds are explicitly excluded. The
actual generated difficulty may be considerably lower than those caps,
and zero earned physical height cannot be fabricated by the schedule.
The original ordinary terrain stays available. The teacher samples
new role assignments only at episode birth; old assignments are not
reclassified when the dose ramp changes mid-episode.

## Isolation and evaluation obligations

The baseline differs from the treatment ONLY by enabling the new flag.
Train two independent **from-zero** students per selected seed, with
identical source, policy, PPO, sensor/physics, generalist and Academy
controls. Use matched two-seed independent physical examinations at
90M, 100M, 110M, 120M, and later; a first validation run is a screen,
not certification. A true retention success requires *mission* success
in COMMAND, CONTROL, TRANSITION, both signed slopes and RESILIENCE,
not merely survival.

Only increase lesson complexity if the student demonstrates each step:
first physical upper foothold → sustained bilateral support → controlled
post-lip traversal → multiple genuine lips → uninterrupted course.
Keep upward and downward promotion independent. **Investigate the
32mm uphill mechanical/teaching cliff separately**; no increased height
is authorised by this intervention.

The stage-one implementation logs `cross_stair_blend`,
`cross_stair_extra_birth_prob_<role>`, and actual completed
`cross_stair_<role>_episodes/survived`. These counts must be audited
before interpreting physical treatment effects. A high survival
fraction is NOT sufficient retention proof.

## Newborn/toddler roadmap (not yet implemented)

- Independent physical mission sentinels trigger targeted role refresher
  allocations, while a protected flat/generalist floor prevents starvation
- First-lip uphill feasibility probe at 28→32mm; test motor/geometry vs
  progression and reward timing without claiming that hardware can climb
- Demonstrated-ability reward fading: small bilateral support bonus
  only during transfer acquisition, then gradual reliance on
  follow-through and robust command tracking
- Short near-lip encounters interleaved with full clean recovery and
  multi-lip course tasks, evaluated in independent first-life worlds

**Do not launch an automatic full-cost new school until the contracts,
physical sampling and baseline-vs-treatment comparison pass review.**
