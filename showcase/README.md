# Public policy slot and upcoming coordinated release

**Current staged policy:** the genuine 120,061,952-step Pitch15 export
in `showcase/champion.emupolicy`. Git blob SHA (used by the manual Pages
workflow): `067d6156f0f9de629eabb1a69cf1b0ceb86db24c`. Its exported file SHA-256 is
`64b01101628137739215f5826cef3bcb301d7681c364000ca14e1ac1a7120220`.
The previously published public site may still contain the 60M champion
until the new `main` merge and manual Pages dispatch.

**Selected next candidate:** `round3/pitch15/checkpoints/final.pkl`,
step 120,061,952, from Git source
`a5996b8cda2c394c7c65e3e938282a3808ded121`.
Required raw PKL SHA-256:
`ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`.
See `docs/STATUS.md` and `HANDOVER.md` for what the experiment did
and did not establish.

**Do not put raw `.pkl` or training state in Git.** Use
`emu export-policy <final.pkl> <candidate.emupolicy>` from the original
source-pinned checkout. Before publication, verify the candidate's
source commit, step, 220×6 actor, trained robot XML, and all browser
controls. Both the exact exported policy and pinned Git blob hash are now staged
**together** with viewer and refactored code. Run CI, merge the approved
combined PR and manually deploy Pages afterward.

The campaign report alone did not contain network weights, but the genuine
`.emupolicy` was subsequently uploaded and staged. This is a browser-only
simulation demonstration, not a hardware-ready servo controller.

See [docs/PLAYGROUND.md](../docs/PLAYGROUND.md) for checks and release steps.
