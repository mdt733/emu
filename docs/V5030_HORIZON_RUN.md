# v5030b — bounded A/B learning-horizon continuation over SSH

**Source safety:** training is always performed from the unchanged
\`feature/v5030-reward-lab\` at SHA
\`a7be11b587f0912e642e100b22ec6d617f548f5b\`.
Do not merge, cherry-pick or pull the reporting branch into the training
checkout. \`main\` and the original 60M checkpoints are untouched.

The tooling is stored on \`feature/v5030-physical-report\` so that a remote
SSH/Tailscale terminal can fetch it with Git. **No ZIP, browser, or file
transfer is required.**

From WSL (preferably inside GNU Screen):

\`\`\`bash
mkdir -p ~/robot_lab
if [ ! -d ~/emu-v5030-horizon/.git ]; then
  git clone --single-branch --branch feature/v5030-reward-lab \
    https://github.com/mdt733/emu.git ~/emu-v5030-horizon
fi
cd ~/emu-v5030-horizon
test "$(git rev-parse HEAD)" = "a7be11b587f0912e642e100b22ec6d617f548f5b"
git fetch --no-tags origin feature/v5030-physical-report
git show FETCH_HEAD:scripts/v5030_horizon_run.py > ~/robot_lab/v5030_horizon_run.py
git show FETCH_HEAD:scripts/v5030_physical_report.py > ~/robot_lab/v5030_physical_report_tool.py
source ~/emu-graduation/.venv/bin/activate
python -m py_compile ~/robot_lab/v5030_horizon_run.py ~/robot_lab/v5030_physical_report_tool.py
python ~/robot_lab/v5030_horizon_run.py --target 80
\`\`\`

The plan command verifies the original run's manifest, 2048-world CUDA
acceptance stamp, config and both completed 60M checkpoint paths.

When the plan has succeeded, from the **original SHA checkout**:

\`\`\`bash
python -u ~/robot_lab/v5030_horizon_run.py --target 80 --run
\`\`\`

Both 60M policies are forked into **new distinct output roots** and trained
sequentially, with the identical curriculum/optimizer continuation contract.
The reward treatment stays A=0 or B=0.5 half-second forward tracking; no
other reward is modified. Both are evaluated using the same held-out physical
speed ladder and slope/step matrix, with an intermediate readable report:

\`~/robot_lab/emu_runs/v5030_ab_learning_horizon_s747/at80M/physical_comparison_report.md\`

Optional \`--target 100 --run\` is available **only after the 80M physical
comparison has completed** and the owner has reviewed the results.

\`--run\` has NOT been tested on the owner's GPU from GitHub; inspect console
errors before assuming successful continuation. GitHub stores code only,
not experimental policy checkpoints. Keep the original training checkout clean.
