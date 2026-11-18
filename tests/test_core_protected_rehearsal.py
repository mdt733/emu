"""Source-scoped protected-rehearsal curriculum, no GPU required."""
import math
import ast
from pathlib import Path
import pytest
from emu.capability_focus import (
    FOCUS_NAMES, COMMAND, CONTROL, STEP, attention_weights,
    protected_rehearsal_weights, holistic_stair_stage
)


def test_protected_cohorts_conserve_all_specialist_mass():
    base = attention_weights(schedule="terrain-early", step=145_000_000,
                             foundation_steps=20_000_000)
    # The existing holistic 15% retention floor is applied first by training.
    control = .15
    weights = list(base)
    rest = sum(weights[:-1])
    weights = [w * (1.-control) / rest for w in weights[:-1]] + [control]
    protected = protected_rehearsal_weights(weights)
    assert len(protected) == len(FOCUS_NAMES) == 6
    assert sum(protected) == pytest.approx(1.)
    assert protected[COMMAND] == pytest.approx(.24)
    assert protected[CONTROL] == pytest.approx(.26)
    assert protected[STEP] > 0
    assert all(protected[i] > 0 for i in range(6))
    other = [i for i in range(6) if i not in (COMMAND, CONTROL)]
    ratios = [protected[i] / weights[i] for i in other]
    assert max(ratios) - min(ratios) < 1e-12
    assert math.isclose(holistic_stair_stage(159_999_999).goal_weight, .22)
    assert math.isclose(holistic_stair_stage(160_000_000).bilateral_bonus, 6.)


@pytest.mark.parametrize("bad", [
    (0, 0, 0, 0, 0, 0),
    (0.5, 0.5),
    (-0.1, 0.2, 0.2, 0.2, 0.2, 0.3),
    (float('nan'), .2, .2, .2, .2, .2),
])
def test_protected_invalid_distribution_rejected(bad):
    with pytest.raises(ValueError):
        protected_rehearsal_weights(bad)


def test_training_scopes_protection_to_post_140m_and_launcher_parses():
    root = Path(__file__).resolve().parents[1]
    source = (root/"emu/training.py").read_text()
    ast.parse(source)
    assert "if args.holistic_cold_curriculum and step >= 140_000_000:" in source
    assert "weights=list(protected_rehearsal_weights(weights))" in source
    import re, subprocess, shutil
    runner = root/"scripts/run_h40_protected_rehearsal_wsl.sh"
    code = runner.read_text()
    assert "--steps 180000000" in code
    assert "work/emu-protected-rehearsal" in code
    assert "protected_rehearsal_180m_results.zip" in code
    assert "--research-parent-sha256" in code
    for block in re.findall(r"<<'PY'\n(.*?)\nPY", code, re.S):
        compile(block, str(runner), "exec")
    assert len(re.findall(r"<<'PY'", code)) == 3
    if shutil.which("bash"):
        subprocess.run(["bash","-n",str(runner)],check=True)
