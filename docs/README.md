# Documentation map — choose the right source of truth

The documentation is split by **job**, not by version number. Read the current operational guides first; consult the archived rationale when changing experiments or interpreting results.

| Document | Owns / use it for | Status |
|---|---|---|
| [../README.md](../README.md) | Project introduction and command quick start | Current |
| [../HANDOVER.md](../HANDOVER.md) | Succinct state of the project, recent findings, safe continuation after a chat limit | Current — maintain this after meaningful milestones |
| [INSTALL.md](INSTALL.md) | Fresh clone, Python + Node setup, troubleshooting, workspace cleanup | Current |
| [tournament.md](tournament.md) | Current ladder recipe, fixed exams and promotion workflow | Current experiment guide |
| [v5014-production.md](v5014-production.md) | Reproducible historical v5014 400M recipe | Historical |
| [PLAYGROUND.md](PLAYGROUND.md) | Public static demo, local Vite preview, champion promotion, GitHub Pages | Current |
| [../viewer/README.md](../viewer/README.md) | Local live viewer, controls, render/performance options | Current |
| [../viewer/POLICY_FORMAT.md](../viewer/POLICY_FORMAT.md) | Exported policy schema and compatibility | Contract |
| [architecture.md](architecture.md) | Ownership boundaries between Python systems | Design reference |
| [hardware-model.md](hardware-model.md) | Robot-model/observation constraints | Design reference |
| [capability-map.md](capability-map.md) | Nodes, edges, recovery evidence and calibration | Design reference |
| [domain-randomization.md](domain-randomization.md) | Implemented domain families vs deliberately unintegrated extensions | Current with future-work section |
| [evaluation.md](evaluation.md) | Why held-out scoring is separate from live demos and curriculum | Design reference |
| [validation.md](validation.md) | How to test a new code release on a real machine | Checklist |
| [../RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md) | Observed test/build results and pending acceptance | Evidence ledger, not promises |
| [history.md](history.md) | Durable mistakes, experimental lessons and principles | Historical rationale |
| [training.md](training.md) | Trainer lifecycle and v5013 experimental mechanism | Mixed: common lifecycle plus historical context |
| [../results/v5013_tournament/README.md](../results/v5013_tournament/README.md) | Original tournament data and provenance | Frozen experiment evidence |
| [AUDIT_REPORT_5014.md](AUDIT_REPORT_5014.md) | Findings of the original RC1 code audit | Dated audit; not today's acceptance status |

**Maintainer rule:** new user-facing instructions go in the relevant current guide; architecture claims need corresponding code evidence; finished experiments live in the historical pages/results. Don't add a second general-purpose README or a second live handover. Retain Git history rather than carrying stale operational steps forward.

If you change the policy interface, update `viewer/POLICY_FORMAT.md` and its mirrored packaged copy together. If you change viewer source, run `emu viewer-build` to refresh the packaging mirror. For a new experiment edit the current recipe in `emu/ladder_recipe.py` and update `tournament.md` and `HANDOVER.md`. The v5014 production guide is history unless using an archived checkout to reproduce historical experiments.
