# Public policy slot and upcoming coordinated release

**Current public champion:** a trained 60M policy in
`showcase/champion.emupolicy`, deliberately protected by its Git blob SHA
in `.github/workflows/emu-playground.yml`. It is not a zero-action test
fixture, and replacing only the viewer must not silently replace its brain.

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
controls. Once the actual candidate file is available, replace this
single slot and the manually pinned Git blob hash **in the same reviewed
code+viewer+policy merge**. Manually deploy Pages only afterward.

The uploaded campaign report does **not** contain the actor weights;
no 120M policy has been staged here yet.

See [docs/PLAYGROUND.md](../docs/PLAYGROUND.md) for checks and release steps.
