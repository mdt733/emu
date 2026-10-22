# v5020 cleanup checklist — conservative, incremental, testable

**Working ref only:** `feature/v5020-coldstart-terrain`.
**Protected reference:** `main` and its public demo. **Do not merge to main
or touch the published champion** without user approval and proven results.
**Status:** experimental research; source CI alone does not imply competence.

## Rules for every pass

1. Record branch head, write a small diff, run relevant unit + smoke tests,
   inspect GitHub Actions, then continue. No mass delete or source rewrite.
2. Preserve current sensor/PPO/checkpoint contracts, output roots and any
   campaign that might still be used to reproduce existing research.
3. Remove old code only after dependency search, importing tests and CLI
   compatibility implications have been resolved. Git history retains old code.
4. Move historical context to [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md)
   and keep current source comments focused on current invariants and *why*.
5. **No new Git feature branches. No touching `main`. No full GPU campaigns
   until we explicitly enter the acceptance stage.**

## Pass 0 — inventory, reference, research history

- [x] Audit `main` against v5020 and document current responsibilities.
- [x] Create [EMU_ARCHITECTURE.md](EMU_ARCHITECTURE.md).
- [x] Summarise old learning results/failures in
  [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md).
- [ ] Freeze a manifest of the local trained checkpoints, original run
      directories, source revisions and accessible result bundles.
      **Requires local run metadata**; not recoverable from Git alone.
- [ ] Confirm which historical commands the user still actively uses.

Exit: know exactly what can and cannot be removed.

## Pass 1 — explicit shared physical measurements

- [x] Begin audit of `environment.py`, `results.py`, `rewards.py`, `terrain.py`
      and `evaluation.py` for duplicate definitions of support, contact,
      crossing, falls, tracking and motor effort.
- [ ] Complete one canonical definition per physically measurable event,
      with precise episode counts, physical units, and missing-data semantics.
- [x] Add adversarial completed-trial and exact-height tests; additional
      foot-support adversaries remain pending.
- [ ] Add adversarial tests for real box contact without crossing,
      crossing without foot support, one-time completion, and empty cohorts.
- [ ] Confirm 220/232/312 actor contracts and baseline reward arithmetic
      remain unchanged.
- [ ] Never share training samples with the frozen held-out examiner.

Implemented first: `emu/scoring.py` owns completed training edge-trial
accounting and frozen exact-height grading, with 8 minimum eligible births
per tested height; the examiner creates true fixed 5/10/20/40/80/150mm
obstacles rather than mixing shorter sampled heights. **The other physical
facts and full GPU validation are still pending.**

Exit: physical facts are identical across consumers, without reweighting PPO.

## Pass 2 — physical and numerical safety

- [x] Detect extreme finite **spring energy and shaft power** across every
      32-frame rollout/world, retained across sparse CSV log intervals.
      **Contact impulse physical-scale guard still pending**.
- [x] Make 2M probes audit their complete recorded trajectory, evaluate
      recent collapse with completed-episode weighting, and retain earlier
      recovered-collapse warnings. Invalid/nonfinite evidence fails closed.
- [x] Use explicit per-run `log_every_updates=1` ONLY in short probes
      (about 16 logged updates at 4096 worlds), instead of drawing a
      "whole trajectory" conclusion from just two sparse CSV rows.
      Ordinary mature runs retain the existing 10-update cadence.
- [ ] Confirm early auto-abort never destroys the last valid checkpoint.
- [ ] Confirm true-contact slopes and collidable box geometry in MJX
      small-world smokes.
- [ ] Audit reward-only terrain payments for exploit paths and correct
      one-time supported passage.

Exit: no absurd physics silently leads to a 'successful' teaching signal.

## Pass 3 — coherent competence governor

- [ ] Represent for every skill: physical safety ceiling, training
      frontier, valid trial counts, success confidence and retreat state.
- [x] Add `emu/competence_shadow.py`: a GPU-free, read-only report of
      the current command limits, safety ceilings, edge tracking exposure,
      episode-weighted fall evidence and UNKNOWN held-out certification.
      It never writes to a checkpoint or changes the study recipe.
- [ ] Implement **true matched decision replay** for the inherited frontier
      algorithm against source-pinned training telemetry, without
      influencing the current 40/60/80M study. The visibility report is
      deliberately not yet a full alternative governor.
- [ ] Bring real slopes, acceleration and physical steps into the same
      *governance interface*, while retaining their distinct evidence rules.
- [ ] Preserve bootstrapping, core-walk rehearsal, per-regime evidence,
      hysteresis, shared expansion budget and recovery retreat.
- [ ] Do not enable `open_frontier=True` in all five active arms just
      because the software supports it. Require measured stable acceptance.

Exit: every frontier change is explained by valid physical evidence.

## Pass 4 — frozen capability certification

- [ ] One versioned evaluation doctrine, including exact sensor/physics
      contract, independent seeds, sufficient exposure and confidence.
- [ ] Common examiner panels for all five arms and the incumbent.
- [ ] Capability report: forward, lateral, yaw, command transitions,
      recovery, contact-slope, verified obstacle heights and effort.
- [ ] Unknown outcomes remain unknown. A flat trained policy still earns
      full credit for unseen skills demonstrated in independent tests.

Exit: report distinguishes training progress from certified skill.

## Pass 5 — minimal Champion Ladder coordinator

- [ ] Keep five study *recipes*: general, turning, recovery, efficiency,
      growing physical steps. Exactly one common ancestor per generation.
- [ ] Common held-out score + hard eligibility gates + incumbent comparison.
- [ ] Pin score revision, task definitions, source, candidate hashes,
      exam seeds and samples to each selection. No unearned promotion.
- [ ] Prove start/restart/reject/score/stop lineage using pure CPU tests.
- [ ] Retire legacy tournament code from the current branch only once
      the useful `emu train` functionality is separated and verified;
      keep historical Git commits and this research-history document.

Exit: one current tournament command and one authoritative scorer.

## Pass 6 — eliminate obsolete implementation and comments

- [ ] Remove old experiment commands no longer called from the current
      CLI, CI, source or a user-pinned run workflow.
- [x] Retire historical v5015/v5016 tournament modules and dead ladder
      configuration helpers. Keep protected single-run planning and resume
      validation in `emu/train_cli.py`; keep independent exams.
- [ ] Simplify doc navigation; avoid keeping multi-thousand-line old
      handovers as the front door. Avoid cosmetic line-count targets.
- [ ] Replace version-by-version inline narration with current contracts,
      links to this architecture and the history.
- [ ] Keep direct dependencies clear so the new coordinator **does not
      import historical campaigns just to read JSON or hash a checkpoint**.

Exit: a reader can locate physics, measurements, rewards, PPO, governor,
examiner, and experiment selection without spelunking through old layers.

## Overnight-readiness milestone

- [x] Add `scripts/overnight_v5020.sh`: one guarded, **opt-in** WSL
      preflight and first 20→40M five-arm run, with source identity,
      checkpoint/schema/world-count validation, tests, GPU doctor, an
      external timestamped log and an immutable root per source commit.
- [x] Display the source foundation's saved fixed-evaluation history;
      distinguish valid checkpoint structure from learned locomotion.
- [x] Veto the observed narrow v5020 20.05M foundation whose easy
      survival was ~99.99% but stress30/dynamic50 were <0.2% and
      resilience50 ~0.01%. Add opt-in **flat-first new 312-input 20M**
      fallback before the 40M tournament. Re-examine and stop if unfit.
- [x] Keep initial launch limited to `walk40`, with all five 2M
      survival probes and an independent frozen exam.
- [ ] Independently check the user's local foundation quality and confirm
      source/GPU before the real overnight launch.
- [ ] Demonstrate the full GPU campaign works and passes eligibility gates;
      `main` remains untouched regardless of this research outcome.

## Pass 7 — proof on actual hardware-backed MJX training

- [ ] Python/browser CI green; small GPU physics and PPO smoke.
- [ ] Choose a healthy, validated 312-input foundation from original
      local evidence instead of assuming the last 20M snapshot is best.
- [ ] Run five short probes and compare to frozen incumbent.
- [ ] Only then run the bounded 20→40M five-way tournament.
- [ ] Continue 60M and 80M only with demonstrably qualified winners.
- [ ] Preserve unmodified `main` reference until explicit user approval.

Exit: measured improvement with an independently reproducible champion,
not just a tidy code repository.

## Current constraints and migration notes

The active five-arm prototype uses `open_frontier=False` in its
40/60/80M lesson recipe (velocity/yaw envelopes are stage-bounded), but
uses an independent physical-height frontier for steps. This is known
architectural debt, **not** a reason to enable every axis simultaneously.

The historical `emu ladder` was retired from this branch after the
supported `emu train` entry point and validations were extracted into
`emu/train_cli.py`. The duplicate v5016 Pinnacle runner was also retired
from this experimental branch; its working reference is untouched on
`main`. Old pinned run directories are not migrated to new recipes.

An unverified clean-source requirement is present in the current refork
coordinator. **Don't run training with altered code against an existing
campaign directory**, even if it contains valid checkpoints from the earlier
implementation.
