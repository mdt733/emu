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
4. Move historical context to [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)
   and keep current source comments focused on current invariants and *why*.
5. **No new Git feature branches. No touching `main`. No full GPU campaigns
   until we explicitly enter the acceptance stage.**

## Pass 0 — inventory, reference, research history

- [x] Audit `main` against v5020 and document current responsibilities.
- [x] Create [EMU_ARCHITECTURE.md](EMU_ARCHITECTURE.md).
- [x] Summarise old learning results/failures in
  [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md).
- [ ] Freeze a manifest of the local trained checkpoints, original run
      directories, source revisions and accessible result bundles.
      **Requires local run metadata**; not recoverable from Git alone.
- [ ] Confirm which historical commands the user still actively uses.

Exit: know exactly what can and cannot be removed.

## Pass 1 — explicit shared physical measurements

- [ ] Audit `environment.py`, `results.py`, `rewards.py`, `terrain.py`
      and `evaluation.py` for duplicate definitions of support, contact,
      crossing, falls, tracking and motor effort.
- [ ] Consolidate one canonical definition per physically measurable event,
      with precise episode counts, physical units, and missing-data semantics.
- [ ] Add adversarial tests for real box contact without crossing,
      crossing without foot support, one-time completion, and empty cohorts.
- [ ] Confirm 220/232/312 actor contracts and baseline reward arithmetic
      remain unchanged.
- [ ] Never share training samples with the frozen held-out examiner.

Exit: physical facts are identical across consumers, without reweighting PPO.

## Pass 2 — physical and numerical safety

- [ ] Detect extreme finite spring/contact/servo values as well as NaN/Inf.
- [ ] Make 2M probes sensitive to progression throughout the probe,
      not merely an average of the last three logging rows.
- [ ] Confirm early auto-abort never destroys the last valid checkpoint.
- [ ] Confirm true-contact slopes and collidable box geometry in MJX
      small-world smokes.
- [ ] Audit reward-only terrain payments for exploit paths and correct
      one-time supported passage.

Exit: no absurd physics silently leads to a 'successful' teaching signal.

## Pass 3 — coherent competence governor

- [ ] Represent for every skill: physical safety ceiling, training
      frontier, valid trial counts, success confidence and retreat state.
- [ ] Shadow-run unified physical command frontier proposals beside the
      current fixed 40/60/80M study recipe, without modifying training.
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
- [ ] Retire historical tournament-only modules and dead helpers, retaining
      useful single-run protection and independent exam utilities.
- [ ] Simplify doc navigation; avoid keeping multi-thousand-line old
      handovers as the front door. Avoid cosmetic line-count targets.
- [ ] Replace version-by-version inline narration with current contracts,
      links to this architecture and the history.
- [ ] Keep direct dependencies clear so the new coordinator **does not
      import historical campaigns just to read JSON or hash a checkpoint**.

Exit: a reader can locate physics, measurements, rewards, PPO, governor,
examiner, and experiment selection without spelunking through old layers.

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

Historical `emu ladder` includes one-run code dependencies. It cannot
be deleted until `emu train` has an independent implementation. Historical
`campaign5016.py` is duplicated in the active research branch but still
represented permanently on `main`; active old run checkpoints cannot be
migrated silently to a new campaign recipe.

An unverified clean-source requirement is present in the current refork
coordinator. **Don't run training with altered code against an existing
campaign directory**, even if it contains valid checkpoints from the earlier
implementation.
