# EMU Studio

Local browser control centre for EMU research, inspired by Micro Moth.
The existing Python/MJX trainer and source-locked checkpoint lifecycle
remain authoritative. The visual playground is the existing CPU/browser
MuJoCo viewer; it is not the frozen physical GPU examiner.

## Open on WSL

Use the existing EMU venv and a clean pinned worktree. Run
`python -m scripts.studio --repo "$HOME/emu"` from that worktree, then
open http://127.0.0.1:8770/ in Windows. The Python executable must be
the existing EMU venv executable. No new server packages are required.
The optional embedded physics viewer uses port 8771 and the existing
viewer build/npm dependencies. The viewer watches the current study's
saved checkpoints; before any are available it uses the protected
showcase champion. It is not a stream of live training world physics.
The Studio itself remains in the foreground; its child output is
logged rather than mixed into the terminal. Ctrl-C requests interruption.

## Versioned launch contract

`launch.json` at the root of the trusted research branch has schema
`emu.launch.v1`. It supplies a stable study identifier, title/description,
Python module under scripts.*, argument vector, parent study slug,
expected ZIP filename, world count, student list, parent/target ages
and physical profile. Two exact placeholders are permitted:
`{study_root}` and `{parent_root}`. No shell expansion or eval.
The module must exist in the selected commit.

Check for update fetches only
`origin/research/ks150-live-skill-school-v1` (using FETCH_HEAD from
the explicit fetch). It reads and validates the manifest directly
from that commit. Run selected creates/reuses a detached worktree at
`~/.local/share/emu-studio/checkouts/<full SHA>`, verifies clean
source and the same manifest, then invokes the existing venv Python
with `-m module` and the registered argument array. Get update & run
performs both actions after one explicit click. It does not pull,
merge or edit main, the current developer checkout, or an active
training worktree. The study directory is automatically named
`~/emu-data/studies/<manifest-id>-<SHA12>` to prevent a different
source resuming a previous registration. Resume last run reads the
original pinned manifest and uses its original study directory.
This does not change the strict parent/model/checkpoint verification
inside the runner. Future experiments update the manifest and module
in Git; v1 bootstrap stays compatible with the same contract. A future
manifest schema change requires a Studio bootstrap update.

The launcher trusts executable code in the owner's named GitHub repo
and branch, not arbitrary third-party manifests. It is NOT a sandbox
for untrusted Python. Local HTTP only (127.0.0.1), Host/Origin and
per-process token checks for control requests, no arbitrary commands
or paths from the browser. Do not expose it to the network. A local
file lock guards Studio-managed GPU jobs, not manually launched
training in other terminals. Stop sends SIGINT to its own process
group; it does not promise a new checkpoint at interruption.
If the server was killed while a child remains, it will not start
another Studio-managed job while the inherited GPU lock is held.
External jobs are not controllable by Studio. Logs and selection/job
receipts are under `~/.local/share/emu-studio/`.

## Dashboard semantics

The selected source can differ from the running/last source. Student
progress and charts refer to that running/last job. Training age is
relative to the registered parent. An endpoint receipt marks a student
sealed; final checkpoint without endpoint means exams pending. Latest
reference/practice mission counts are from a single latest complete
JSONL window and epoch, not pooled across restarts or rollbacks. Zero
attempts display a dash. Actual measured training metrics are read
from metrics.csv; the browser never fabricates scores. Teacher
decision panels show the recorded latest decision. The ZIP is served
only from the registered path; checkpoints stay local.

## 27 Nov 2026 failure and fix

Operator ran the old pinned df4d624f pilot-v2. All six authenticated
fork checks passed. Combined GPU preflight demonstrated correct
recovery births (285 reference / 298 practice) and exact signed uphill
births (594 / 641), physical convex prism surface, recycle and 2-frame
4096-world physics. PPO initialization then called env.reset({}) and
failed on zero default weights in params_array. Fix d922dd1a: omitted
whole enabled teacher vectors default to uniform practice and zero
stage/bonus; explicit partial vectors and invalid full vectors still
fail. Host regression test added for the exact production params_array.
The new manifest is pilot-v3 with a source-derived root. Preserve the
failed v2 directory: no training students started in it. Actual
combined PPO preflight remains unexecuted after the fix.

Browser JS syntax checked on committed HTML; local Studio tests cover
manifest validation, dummy child Git worktree launch/resume and loopback
request guards. Run them with
`python -m unittest discover -s tests -p test_studio.py`.
The tests and full browser/WSL integration have not yet been executed
in the assistant container, which cannot access GitHub raw files or
CUDA. Do not claim the corrected PPO run or live viewer works until
operator evidence. Next: use Studio to launch the registered pilot,
inspect any real blocker, then analyze actual learning evidence.

## Studio progress and live diagnostic export (27 November)

Operator's v3 log confirmed all fork checks, combined birth/geometry
checks and the first real 4096 x 32 PPO update (131072 steps; 86.84s
including compile). No later acceptance update or research student
training is established by that log. Do not infer completion or failure
from silence. Studio now separates total student training from current
phase (fork checks, GPU acceptance, parent exams, training, student exams,
packaging). Old acceptance logs only establish a lower bound; new smoke
prints each completed update, new trainer prints each observed rollout.
Output age and process lock are shown without claiming GPU health.

Final ZIP remains the sealed runner output. Partial ZIP is always available
once a job exists, containing a bounded console log and available metrics,
lesson JSONL, manifests and exam evidence; omitted oversized files are
listed. No checkpoint or actor binaries. Partial snapshot is explicitly
marked incomplete and not a sealed result. A hash-matched copy of
studio_progress.py passed local tests covering the operator's old log,
retry boundaries, per-update progress, phase transitions, archive CRC,
partial JSONL and exclusion of checkpoints. Browser/HTTP integration
and actual corrected full GPU run remain unverified.

A read-only monitor can run alongside the original Studio on port 8772:
`python -m scripts.studio --repo "$HOME/emu" --monitor --port 8772`
from the updated Studio checkout using the existing EMU venv. It reads
the same state receipts and does not launch, stop or modify the pinned
job. Keep the old controlling terminal open while it is active. Once
idle, the new checkout can replace the old controller on port 8770.
Do not click Get update & run to merely refresh a running job: it selects
a new source/study. Resume last run preserves the original pin.
