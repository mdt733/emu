"""Explicit audited migration for the specific v5016 Round-1 examiner crash.

Never modifies a trainer, checkpoint, or run; compare all training-critical
files to the exact campaign-pinned original Git commit before allowing a
new read-only score examiner to resume an already-completed four-arm study.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .contracts import source_repo_root

CRITICAL = (
    'emu/robot.xml', 'emu/environment.py', 'emu/rewards.py',
    'emu/contracts.py', 'emu/curriculum.py', 'emu/training.py',
    'emu/training_config.py', 'emu/checkpoint_io.py',
    'emu/directional.py', 'emu/branch_lab.py', 'emu/evaluation.py',
    'emu/ppo.py',
)


def _file(commit: str, path: str) -> str:
    return subprocess.check_output(
        ['git', 'show', f'{commit}:{path}'],
        cwd=source_repo_root(), stderr=subprocess.DEVNULL, text=True)


def _without_exam(code: str) -> str:
    # ppo.py's frozen-policy efficiency exam has no influence on learning:
    # strip ONLY this whole nested function and its JIT wrapper for the
    # comparison, never any PPO gradient, actor, or training loop.
    first = code.index('    def evaluate_efficiency(')
    last = code.index('    def evaluate_trace(', first)
    return code[:first] + code[last:]


def verify_exam_fix_upgrade(old_commit: str, new_commit: str) -> dict:
    """Fail closed unless the only training-critical diff is read-only exam."""
    if not all(len(x)==40 and all(ch in '0123456789abcdef' for ch in x)
               for x in (old_commit, new_commit)):
        raise RuntimeError('source migration requires full 40-digit Git SHA values')
    if old_commit == new_commit:
        raise RuntimeError('source unchanged; no migration required')
    if subprocess.run(
        ['git','merge-base','--is-ancestor',old_commit,new_commit],
        cwd=source_repo_root(), capture_output=True).returncode:
        raise RuntimeError('original Pinnacle source is not an ancestor of this feature branch')
    for path in CRITICAL:
        try:
            before, after = _file(old_commit, path), _file(new_commit, path)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('cannot compare '+path) from exc
        if path == 'emu/ppo.py':
            if "jnp.sum(m['action_delta_norm'])" not in before or 'jnp.sum(action_delta_norm)' not in after:
                raise RuntimeError('original or new source lacks the exact known examination bug/fix')
            before, after = _without_exam(before), _without_exam(after)
        if before != after:
            raise RuntimeError('cannot migrate: training-critical source changed: '+path)
    return {'reason':'round1_reward_only_action_delta_examination_fix',
            'previous_commit':old_commit,'new_commit':new_commit,
            'training_files_checked':list(CRITICAL)}


def authorize_completed_round1(ledger: dict, root: Path,
                                new_commit: str, arms) -> dict:
    if (set(ledger.get('round1',{})) != set(arms) or ledger.get('selected')
            or ledger.get('round2') or ledger.get('polish')
            or (root / 'scoreboards' / 'round1.json').exists()):
        raise RuntimeError('this repair is only allowed immediately after all four 80M branches complete, before judging')
    return verify_exam_fix_upgrade(ledger['source_commit'], new_commit)
